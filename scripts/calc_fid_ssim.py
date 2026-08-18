import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.nn import functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_obj_from_file(module_name, file_path, object_name):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, object_name)


InceptionV3 = load_obj_from_file(
    "tigdldm_inception",
    PROJECT_ROOT / "saicinpainting" / "evaluation" / "losses" / "fid" / "inception.py",
    "InceptionV3",
)
SSIM = load_obj_from_file(
    "tigdldm_ssim",
    PROJECT_ROOT / "saicinpainting" / "evaluation" / "losses" / "ssim.py",
    "SSIM",
)

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_lpips():
    return load_obj_from_file(
        "tigdldm_lpips",
        PROJECT_ROOT / "saicinpainting" / "evaluation" / "losses" / "lpips.py",
        "PerceptualLoss",
    )


def list_images(root):
    root = Path(root)
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p.relative_to(root)).replace("\\", "/"))


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            raise ValueError("FID has a large imaginary component.")
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def fid_activations(files, model, device, batch_size, dims, image_size):
    transform = Compose([Resize(image_size), CenterCrop(image_size), ToTensor()])
    activations = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch_files = files[start:start + batch_size]
            batch = torch.stack([transform(load_rgb(path)) for path in batch_files]).to(device)
            pred = model(batch)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(-1).squeeze(-1).detach().cpu().numpy()
            activations.append(pred.reshape(pred.shape[0], dims))

    return np.concatenate(activations, axis=0)


def calculate_fid(real_files, fake_files, device, batch_size, dims, image_size):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    real_act = fid_activations(real_files, model, device, batch_size, dims, image_size)
    fake_act = fid_activations(fake_files, model, device, batch_size, dims, image_size)

    mu_real, sigma_real = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
    return float(calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake))


def make_inception_score_model(device):
    try:
        weights = models.Inception_V3_Weights.DEFAULT
        model = models.inception_v3(weights=weights, transform_input=False)
    except AttributeError:
        model = models.inception_v3(pretrained=True, transform_input=False)
    except TypeError:
        model = models.inception_v3(pretrained=True)
    model.to(device)
    model.eval()
    return model


def calculate_inception_score(fake_files, device, batch_size, image_size, splits):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = make_inception_score_model(device)
    probs = []

    with torch.no_grad():
        for start in range(0, len(fake_files), batch_size):
            batch_files = fake_files[start:start + batch_size]
            batch = torch.stack([transform(load_rgb(path)) for path in batch_files]).to(device)
            logits = model(batch)
            if hasattr(logits, "logits"):
                logits = logits.logits
            probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    splits = max(1, min(int(splits), len(probs)))
    scores = []
    for part in np.array_split(probs, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def paired_files(real_root, fake_root, pair_mode):
    real_files = list_images(real_root)
    fake_files = list_images(fake_root)

    if pair_mode == "sorted":
        count = min(len(real_files), len(fake_files))
        return list(zip(real_files[:count], fake_files[:count]))

    real_map = {
        str(path.relative_to(real_root)).replace("\\", "/"): path
        for path in real_files
    }
    fake_map = {
        str(path.relative_to(fake_root)).replace("\\", "/"): path
        for path in fake_files
    }
    common = sorted(set(real_map) & set(fake_map))
    return [(real_map[name], fake_map[name]) for name in common]


def calculate_pair_metrics(real_root, fake_root, device, batch_size, image_size, pair_mode, lpips_net):
    pairs = paired_files(Path(real_root), Path(fake_root), pair_mode)
    if not pairs:
        raise RuntimeError("No paired images found for PSNR/SSIM/LPIPS.")

    transform = Compose([Resize(image_size), CenterCrop(image_size), ToTensor()])
    ssim_metric = SSIM(window_size=11, size_average=False).to(device).eval()
    LPIPS = load_lpips()
    lpips_metric = LPIPS(model="net-lin", net=lpips_net, use_gpu=device.type == "cuda").to(device).eval()
    psnr_scores = []
    ssim_scores = []
    mae_scores = []
    lpips_scores = []

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start:start + batch_size]
            real_batch = torch.stack([transform(load_rgb(real)) for real, _ in batch_pairs]).to(device)
            fake_batch = torch.stack([transform(load_rgb(fake)) for _, fake in batch_pairs]).to(device)
            mse = torch.mean((fake_batch - real_batch) ** 2, dim=(1, 2, 3))
            mae = torch.mean(torch.abs(fake_batch - real_batch), dim=(1, 2, 3))
            psnr = torch.where(mse > 0, 10.0 * torch.log10(1.0 / mse), torch.full_like(mse, float("inf")))
            psnr_scores.extend(psnr.detach().cpu().numpy().tolist())
            mae_scores.extend(mae.detach().cpu().numpy().tolist())
            ssim_scores.extend(ssim_metric(fake_batch, real_batch).detach().cpu().numpy().tolist())
            lpips_scores.extend(lpips_metric(fake_batch, real_batch, normalize=True).flatten().detach().cpu().numpy().tolist())

    psnr_scores = np.asarray(psnr_scores, dtype=np.float64)
    ssim_scores = np.asarray(ssim_scores, dtype=np.float64)
    mae_scores = np.asarray(mae_scores, dtype=np.float64)
    lpips_scores = np.asarray(lpips_scores, dtype=np.float64)
    return {
        "pairs": len(pairs),
        "psnr_mean": float(psnr_scores.mean()),
        "psnr_std": float(psnr_scores.std()),
        "ssim_mean": float(ssim_scores.mean()),
        "ssim_std": float(ssim_scores.std()),
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std()),
        "lpips_mean": float(lpips_scores.mean()),
        "lpips_std": float(lpips_scores.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, LPIPS, FID and IS between image folders.")
    parser.add_argument("--real", required=True, help="Folder with ground-truth images.")
    parser.add_argument("--fake", required=True, help="Folder with generated images.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dims", type=int, default=2048, choices=sorted(InceptionV3.BLOCK_INDEX_BY_DIM))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pair-mode", default="name", choices=["name", "sorted"],
                        help="Pairing mode for PSNR/SSIM/LPIPS: same relative filename or sorted order.")
    parser.add_argument("--lpips-net", default="vgg", choices=["vgg", "alex", "squeeze"])
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    parser.add_argument("--skip-pair", action="store_true", help="Skip paired PSNR/SSIM/LPIPS metrics.")
    parser.add_argument("--out-json", default=None, help="Optional path to save all metrics as JSON.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    real_files = list_images(args.real)
    fake_files = list_images(args.fake)

    if not real_files:
        raise RuntimeError("No real images found under {}".format(args.real))
    if not fake_files:
        raise RuntimeError("No generated images found under {}".format(args.fake))

    print("real_images: {}".format(len(real_files)))
    print("fake_images: {}".format(len(fake_files)))
    print("device: {}".format(device))
    results = {
        "real": args.real,
        "fake": args.fake,
        "real_images": len(real_files),
        "fake_images": len(fake_files),
        "device": str(device),
        "image_size": args.image_size,
        "pair_mode": args.pair_mode,
    }

    if not args.skip_fid:
        if len(real_files) < 2 or len(fake_files) < 2:
            raise RuntimeError("FID needs at least two images in each folder.")
        fid = calculate_fid(real_files, fake_files, device, args.batch_size, args.dims, args.image_size)
        results["fid"] = fid
        print("FID: {:.6f}".format(fid))

    if not args.skip_is:
        is_score = calculate_inception_score(fake_files, device, args.batch_size, 299, args.is_splits)
        results["is_mean"] = is_score["mean"]
        results["is_std"] = is_score["std"]
        print("IS_mean: {:.6f}".format(is_score["mean"]))
        print("IS_std: {:.6f}".format(is_score["std"]))

    if not args.skip_pair:
        pair_metrics = calculate_pair_metrics(
            args.real, args.fake, device, args.batch_size, args.image_size, args.pair_mode, args.lpips_net
        )
        print("PSNR_mean: {:.6f}".format(pair_metrics["psnr_mean"]))
        print("PSNR_std: {:.6f}".format(pair_metrics["psnr_std"]))
        print("SSIM_mean: {:.6f}".format(pair_metrics["ssim_mean"]))
        print("SSIM_std: {:.6f}".format(pair_metrics["ssim_std"]))
        print("MAE_mean: {:.6f}".format(pair_metrics["mae_mean"]))
        print("MAE_std: {:.6f}".format(pair_metrics["mae_std"]))
        print("LPIPS_mean: {:.6f}".format(pair_metrics["lpips_mean"]))
        print("LPIPS_std: {:.6f}".format(pair_metrics["lpips_std"]))
        print("paired_images: {}".format(pair_metrics["pairs"]))
        results.update(pair_metrics)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Saved metrics JSON: {}".format(args.out_json))


if __name__ == "__main__":
    main()
