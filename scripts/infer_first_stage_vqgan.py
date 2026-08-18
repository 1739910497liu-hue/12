import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldm.util import instantiate_from_config  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_images(input_path, count=-1):
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if count is not None and count > 0:
        files = files[:count]
    return files


def load_image(path, size, device):
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous().to(device=device, dtype=torch.float32)


def tensor_to_pil(tensor):
    tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    array = ((tensor.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255)
    return Image.fromarray(array.astype(np.uint8))


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor[0]).save(path)


def save_comparison(source, reconstruction, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = torch.cat([source, reconstruction], dim=3)
    save_tensor_image(merged, path)


def mask_to_pil(mask):
    mask = mask.detach().float().cpu().squeeze().numpy()
    mask = (mask.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def first_stage_needs_mask(model):
    if model.__class__.__name__ == "VQModelWithMask":
        return True
    decode = getattr(model, "decode", None)
    if decode is None:
        return False
    try:
        import inspect

        params = list(inspect.signature(decode).parameters)
        return "image" in params and "mask" in params
    except (TypeError, ValueError):
        return False


def make_ellipse_mask(height, width, params):
    params = params or {}
    min_radius_x = float(params.get("min_radius_x", 42.0))
    max_radius_x = float(params.get("max_radius_x", 110.0))
    min_radius_y = float(params.get("min_radius_y", 28.0))
    max_radius_y = float(params.get("max_radius_y", 88.0))
    random_angle = bool(params.get("random_angle", True))
    angle_min = float(params.get("angle_min_deg", 0.0))
    angle_max = float(params.get("angle_max_deg", 360.0))

    if params.get("random_center", True):
        margin_x = max_radius_x / max(width - 1, 1)
        margin_y = max_radius_y / max(height - 1, 1)
        center_x = np.random.uniform(margin_x, 1.0 - margin_x) * (width - 1)
        center_y = np.random.uniform(margin_y, 1.0 - margin_y) * (height - 1)
    else:
        center_x = float(params.get("center_x", 0.5)) * (width - 1)
        center_y = float(params.get("center_y", 0.5)) * (height - 1)

    radius_x = np.random.uniform(min_radius_x, max_radius_x)
    radius_y = np.random.uniform(min_radius_y, max_radius_y)
    angle_deg = np.random.uniform(angle_min, angle_max) if random_angle else float(params.get("angle_deg", 0.0))
    angle = np.deg2rad(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float32) - center_x
    y = yy.astype(np.float32) - center_y
    x_rot = x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    return (((x_rot / radius_x) ** 2 + (y_rot / radius_y) ** 2) <= 1.0).astype(np.float32)


def make_mask(batch_size, image_size, device, mode, ellipse_params):
    if mode == "full":
        mask = np.ones((batch_size, image_size, image_size), dtype=np.float32)
    elif mode == "ellipse":
        mask = np.stack([make_ellipse_mask(image_size, image_size, ellipse_params) for _ in range(batch_size)], axis=0)
    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    return torch.from_numpy(mask).unsqueeze(1).to(device=device, dtype=torch.float32)


def get_ellipse_params(config):
    for split in ("validation", "train"):
        try:
            params = config.data.params[split].params.get("ellipse_params", {})
            if params:
                return dict(params)
        except Exception:
            pass
    return {}


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    if ckpt_path:
        config.model.params.ckpt_path = ckpt_path
    try:
        model = instantiate_from_config(config.model)
    except TypeError as err:
        message = str(err)
        retry_keys = []
        if "unexpected keyword argument" in message:
            retry_keys = [
                key for key in ("recompute_quant_loss", "quant_loss_beta")
                if key in config.model.params
            ]
        if not retry_keys:
            raise
        for key in retry_keys:
            del config.model.params[key]
            print(f"Removed unsupported inference config key: model.params.{key}")
        model = instantiate_from_config(config.model)
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Restored first-stage checkpoint from {ckpt_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    return model, config


def reconstruct(model, images, masks):
    if first_stage_needs_mask(model):
        reconstruction, _ = model((images, masks))
    else:
        output = model(images)
        reconstruction = output[0] if isinstance(output, (tuple, list)) else output
    return reconstruction


def main():
    parser = argparse.ArgumentParser(description="Run first-stage VQGAN reconstruction inference.")
    parser.add_argument("--config", default="configs/autoencoder/autoencoder_kl_32x32x4_large2_train.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", default="datasets/ss304/valid", help="Input image file or directory.")
    parser.add_argument("--outdir", default="outputs/first_stage_vqgan")
    parser.add_argument("--size", type=int, default=256, help="Resize to square size. Use 0 to keep original size.")
    parser.add_argument("--count", type=int, default=64, help="Number of images. Use -1 for all.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mask-mode", choices=["ellipse", "full"], default="ellipse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    model, config = load_model(args.config, str(ckpt_path), device)
    size = None if args.size == 0 else args.size
    image_size = int(size or config.model.params.ddconfig.get("resolution", 256))
    ellipse_params = get_ellipse_params(config)
    image_paths = iter_images(input_path, args.count)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_path}")

    outdir = Path(args.outdir)
    generated = 0
    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            images = torch.cat([load_image(path, size, device) for path in batch_paths], dim=0)
            masks = make_mask(len(batch_paths), image_size, device, args.mask_mode, ellipse_params)
            reconstructions = reconstruct(model, images, masks)

            for idx, image_path in enumerate(batch_paths):
                relpath = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
                stem = relpath.with_suffix("")
                save_tensor_image(images[idx:idx + 1], outdir / "inputs" / stem.with_suffix(".png"))
                save_tensor_image(reconstructions[idx:idx + 1], outdir / "reconstructions" / stem.with_suffix(".png"))
                save_comparison(images[idx:idx + 1], reconstructions[idx:idx + 1], outdir / "comparisons" / stem.with_suffix(".png"))
                mask_path = outdir / "masks" / stem.with_suffix(".png")
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask_to_pil(masks[idx]).save(mask_path)
                generated += 1
                print(f"Saved reconstruction for {image_path}")

    print(f"Done. Saved {generated} reconstructions in {outdir.resolve()}")


if __name__ == "__main__":
    main()
