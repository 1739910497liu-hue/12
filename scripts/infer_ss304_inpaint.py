import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(config_path, ckpt_path, first_stage_ckpt=None, device="cuda"):
    config = OmegaConf.load(config_path)
    if first_stage_ckpt is not None:
        config.model.params.first_stage_config.params.ckpt_path = first_stage_ckpt

    model = instantiate_from_config(config.model)
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if isinstance(checkpoint, dict) and "global_step" in checkpoint:
        print("Loaded stage-2 checkpoint global_step: {}".format(checkpoint["global_step"]))
    print("Missing keys: {}, unexpected keys: {}".format(len(missing), len(unexpected)))
    if missing:
        print("First missing keys: {}".format(missing[:20]))
    if unexpected:
        print("First unexpected keys: {}".format(unexpected[:20]))
    model.to(device)
    model.eval()
    return model, config


def pil_resize(image, size, nearest=False):
    resampling = getattr(Image, "Resampling", Image)
    mode = resampling.NEAREST if nearest else resampling.BICUBIC
    return image.resize((size, size), mode)


def load_image_tensor(path, size, device):
    image = Image.open(path).convert("RGB")
    image = pil_resize(image, size, nearest=False)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    return tensor * 2.0 - 1.0


def load_mask_tensor(path, size, device, invert=False):
    mask = Image.open(path).convert("L")
    mask = pil_resize(mask, size, nearest=True)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    if invert:
        arr = 1.0 - arr
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    x = (x * 255.0).round().byte().squeeze(0)
    if x.shape[0] == 1:
        return Image.fromarray(x.squeeze(0).numpy(), mode="L")
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def find_image_mask_pairs(indir):
    indir = Path(indir)
    masks = sorted(
        p for p in indir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS and p.stem.endswith("_mask")
    )
    pairs = []
    for mask in masks:
        stem = mask.stem[:-5]
        image = None
        for ext in IMG_EXTENSIONS:
            candidate = mask.with_name(stem + ext)
            if candidate.exists():
                image = candidate
                break
        if image is not None:
            pairs.append((image, mask))
    return pairs


@torch.no_grad()
def build_condition(model, image, hole_mask):
    masked_image_01 = (1.0 - hole_mask) * ((image + 1.0) / 2.0)
    masked_image = masked_image_01 * 2.0 - 1.0

    z = model.get_first_stage_encoding(model.encode_first_stage(image)).detach()
    masked_z = model.get_first_stage_encoding(model.encode_first_stage(masked_image)).detach()
    latent_h, latent_w = z.shape[-2:]
    latent_hole_mask = torch.nn.functional.interpolate(hole_mask, size=(latent_h, latent_w), mode="nearest")

    cond = {
        "c_concat": [torch.cat([latent_hole_mask, masked_z], dim=1)]
    }
    latent_keep_mask = 1.0 - latent_hole_mask
    return cond, z, masked_image, latent_keep_mask


@torch.no_grad()
def inpaint_one(model, sampler, image_path, mask_path, outdir, size, steps, eta, seed, invert_mask,
                save_debug=False, overwrite=False, save_original=True):
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    outdir = Path(outdir)
    outpath = outdir / image_path.name
    if outpath.exists() and not overwrite:
        return outpath

    if seed is not None:
        seed_everything(seed)

    device = model.device
    image = load_image_tensor(image_path, size, device)
    hole_mask = load_mask_tensor(mask_path, size, device, invert=invert_mask)
    cond, z, masked_image, latent_keep_mask = build_condition(model, image, hole_mask)

    samples, _ = sampler.sample(
        S=steps,
        batch_size=image.shape[0],
        shape=(model.channels, model.image_size, model.image_size),
        conditioning=cond,
        eta=eta,
        mask=latent_keep_mask,
        x0=z,
        verbose=False,
    )

    pred = model.decode_first_stage(samples, image=image, mask=hole_mask)
    pred01 = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
    image01 = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    inpainted = (1.0 - hole_mask) * image01 + hole_mask * pred01

    outdir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(inpainted).save(outpath)

    if save_original:
        original_dir = outdir / "originals"
        original_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(image01).save(original_dir / image_path.name)

    if save_debug:
        debug_dir = outdir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(image01).save(debug_dir / "{}_input.png".format(image_path.stem))
        tensor_to_pil(hole_mask).save(debug_dir / "{}_mask.png".format(image_path.stem))
        tensor_to_pil(((masked_image + 1.0) / 2.0).clamp(0.0, 1.0)).save(debug_dir / "{}_masked.png".format(image_path.stem))
        tensor_to_pil(pred01).save(debug_dir / "{}_pred.png".format(image_path.stem))

    return outpath


def main():
    parser = argparse.ArgumentParser(description="Run SS304 latent inpainting inference.")
    parser.add_argument("--config", default="configs/latent-diffusion/ss304-ldm-inpaint-vq.yaml")
    parser.add_argument("--ckpt", required=True, help="Stage-2 inpainting LDM checkpoint.")
    parser.add_argument("--first-stage-ckpt", default=None, help="Override first-stage VQGAN checkpoint.")
    parser.add_argument("--image", default=None, help="Single input image.")
    parser.add_argument("--mask", default=None, help="Single binary mask. White means hole/fill area.")
    parser.add_argument("--indir", default=None, help="Folder containing image.png and image_mask.png pairs.")
    parser.add_argument("--outdir", default="outputs/ss304_inpaint")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--invert-mask", action="store_true", help="Use if black means hole and white means known area.")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--no-save-original", action="store_true",
                        help="Do not save resized original images beside generated outputs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    seed_everything(args.seed)

    if args.indir is None and (args.image is None or args.mask is None):
        raise ValueError("Pass either --indir or both --image and --mask.")

    model, _ = load_model(args.config, args.ckpt, args.first_stage_ckpt, device=device)
    sampler = DDIMSampler(model)

    if args.indir is not None:
        pairs = find_image_mask_pairs(args.indir)
        if not pairs:
            raise RuntimeError("No image/mask pairs found under {}. Expected image.png and image_mask.png.".format(args.indir))
    else:
        pairs = [(Path(args.image), Path(args.mask))]

    print("Found {} image/mask pair(s).".format(len(pairs)))
    with model.ema_scope("SS304 inpainting inference"):
        for idx, (image_path, mask_path) in enumerate(tqdm(pairs, desc="Inpainting")):
            outpath = inpaint_one(
                model=model,
                sampler=sampler,
                image_path=image_path,
                mask_path=mask_path,
                outdir=args.outdir,
                size=args.size,
                steps=args.steps,
                eta=args.eta,
                seed=args.seed + idx if args.seed is not None else None,
                invert_mask=args.invert_mask,
                save_debug=args.save_debug,
                save_original=not args.no_save_original,
                overwrite=args.overwrite,
            )
            print("saved: {}".format(outpath))


if __name__ == "__main__":
    main()
