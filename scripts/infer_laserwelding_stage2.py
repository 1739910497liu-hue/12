import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import trange


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def load_model(config_path, ckpt_path, device, first_stage_ckpt=None, verbose=False):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    if first_stage_ckpt is not None:
        config.model.params.first_stage_config.params.ckpt_path = first_stage_ckpt

    print(f"Loading config: {config_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    model = instantiate_from_config(config.model)
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "global_step" in checkpoint:
        print(f"Stage-2 checkpoint global_step: {checkpoint['global_step']}")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Restored stage-2 checkpoint from {ckpt_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if verbose and missing:
        print("Missing keys:", missing)
    if verbose and unexpected:
        print("Unexpected keys:", unexpected)
    model.to(device)
    model.eval()
    return model, config


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def mask_to_pil(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().squeeze().numpy()
    mask = (np.asarray(mask, dtype=np.float32).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def first_stage_needs_mask(model):
    first_stage = getattr(model, "first_stage_model", None)
    if first_stage is None:
        return False
    if first_stage.__class__.__name__ == "VQModelWithMask":
        return True
    decode = getattr(first_stage, "decode", None)
    if decode is None:
        return False
    try:
        import inspect

        params = list(inspect.signature(decode).parameters)
        return "image" in params and "mask" in params
    except (TypeError, ValueError):
        return False


def get_config_resolution(config):
    return int(config.model.params.first_stage_config.params.ddconfig.get("resolution", 256))


def get_validation_ellipse_params(config):
    try:
        params = config.data.params.validation.params.get("ellipse_params", {})
    except Exception:
        params = {}
    return dict(params)


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


def load_decode_image(path, image_size, device):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def resolve_json_image(root, key):
    key_path = Path(str(key))
    if key_path.is_absolute() and key_path.exists():
        return key_path

    direct = root / key_path
    if direct.exists():
        return direct

    normalized = str(key).replace("\\", "/")
    for marker in ("/validation/", "/valid/", "/test/", "/train/"):
        if marker in normalized:
            candidate = root / normalized.split(marker, 1)[1]
            if candidate.exists():
                return candidate

    parts = Path(normalized).parts
    for idx, part in enumerate(parts):
        if part.lower() in {"validation", "valid", "test", "train"}:
            candidate = root / Path(*parts[idx + 1:])
            if candidate.exists():
                return candidate

    return direct


def list_input_items(input_root, input_json=None, count=None):
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    if input_json:
        json_candidates = [Path(input_json)]
    else:
        json_candidates = sorted(root.glob("*.json"))

    paths = []
    for json_path in json_candidates:
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = data.keys()
        elif isinstance(data, list):
            keys = []
            for item in data:
                if isinstance(item, dict):
                    keys.append(item.get("file") or item.get("path") or item.get("image") or item.get("image_path"))
                else:
                    keys.append(item)
        else:
            keys = []
        for key in keys:
            if key is None:
                continue
            p = resolve_json_image(root, key)
            if p.exists() and p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
        if paths:
            print(f"Loaded {len(paths)} input images from JSON: {json_path}")
            break

    if not paths:
        paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        print(f"Loaded {len(paths)} input images by scanning: {root}")

    if not paths:
        raise RuntimeError(f"No input images found under {input_root}")
    if count is not None and count > 0:
        paths = paths[:count]
    items = []
    for p in paths:
        try:
            relpath = str(p.relative_to(root)).replace("\\", "/")
        except ValueError:
            relpath = p.name
        items.append({"path": p, "relpath": relpath})
    return items


def build_decode_inputs(model, batch_items, image_size, device, mask_mode, ellipse_params):
    if not first_stage_needs_mask(model):
        return None, None, []

    if mask_mode == "full":
        image = torch.zeros(len(batch_items), 3, image_size, image_size, device=device)
        mask = torch.ones(len(batch_items), 1, image_size, image_size, device=device)
        masks = [np.ones((image_size, image_size), dtype=np.float32) for _ in batch_items]
        return image, mask, masks

    images = []
    masks = []
    for item in batch_items:
        mask_np = make_ellipse_mask(image_size, image_size, ellipse_params)
        masks.append(mask_np)
        images.append(load_decode_image(item["path"], image_size, device))
    image = torch.cat(images, dim=0)
    mask = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).to(device=device, dtype=torch.float32)
    return image, mask, masks


def decode_samples(model, samples_z, quantize_z=False, decode_image=None, decode_mask=None):
    if first_stage_needs_mask(model) and (decode_image is None or decode_mask is None):
        raise RuntimeError("First-stage VQModelWithMask requires decode_image and decode_mask.")

    if quantize_z and hasattr(model.first_stage_model, "quantize"):
        device = getattr(model, "device", samples_z.device)
        z = samples_z.to(device)
        z = 1.0 / model.scale_factor * z
        quant, _, _ = model.first_stage_model.quantize(z)
        if first_stage_needs_mask(model):
            return model.first_stage_model.decode(quant, decode_image, decode_mask)
        return model.first_stage_model.decode(quant)

    if first_stage_needs_mask(model):
        return model.decode_first_stage(samples_z, image=decode_image, mask=decode_mask)
    return model.decode_first_stage(samples_z)


def save_resized_input(input_path, save_path, output_image):
    image = Image.open(input_path)
    if image.mode != output_image.mode:
        image = image.convert(output_image.mode)
    image = image.resize(output_image.size, Image.BICUBIC)
    image.save(save_path)


def save_manifest_record(manifest, record):
    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest.flush()


def rel_to_outdir(path, outdir):
    return str(Path(path).relative_to(outdir)).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="Unconditional laser welding stage-2 LDM inference.")
    parser.add_argument("--config", default="configs/latent-diffusion/laserwelding-ldm-vq-lite.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--first-stage-ckpt", default=None)
    parser.add_argument("--input-root", "--indir", dest="input_root", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--outdir", default="outputs/laserwelding_stage2")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", "--steps", dest="ddim_steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--decode-mask-mode", choices=["ellipse", "full"], default="ellipse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ext", default="png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--quantize-z", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Report generation latency excluding image I/O.")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Number of initial batches excluded from benchmark timing.")
    parser.add_argument("--verbose-load", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    seed_everything(args.seed)

    model, config = load_model(
        args.config,
        args.ckpt,
        device,
        first_stage_ckpt=args.first_stage_ckpt,
        verbose=args.verbose_load,
    )

    image_size = get_config_resolution(config)
    ellipse_params = get_validation_ellipse_params(config)
    if args.input_root is None:
        try:
            args.input_root = str(config.data.params.validation.params.data_root)
        except Exception:
            raise ValueError("--input-root is required when config has no validation data_root.")
    items = list_input_items(args.input_root, args.input_json, args.count)
    total_count = len(items) if args.count < 0 else min(args.count, len(items))
    if total_count <= 0:
        raise ValueError("--count must be positive, or use -1 to process all input images.")

    outdir = Path(args.outdir)
    sample_dir = outdir / "samples"
    input_dir = outdir / "inputs"
    mask_dir = outdir / "masks"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.jsonl"
    start_index = 0 if args.overwrite else len(list(sample_dir.glob(f"*.{args.ext}")))

    cond = None
    context = model.ema_scope("Laser welding sampling") if getattr(model, "use_ema", False) and not args.no_ema else nullcontext()
    generated = 0
    timed_images = 0
    timed_seconds = 0.0
    batch_index = 0
    with torch.no_grad(), context, open(manifest_path, "a", encoding="utf-8") as manifest:
        for _ in trange((total_count + args.batch_size - 1) // args.batch_size, desc="Laser welding sampling"):
            current_bs = min(args.batch_size, total_count - generated)
            if current_bs <= 0:
                break
            batch_items = items[generated:generated + current_bs]
            decode_image, decode_mask, masks = build_decode_inputs(
                model,
                batch_items,
                image_size,
                device,
                args.decode_mask_mode,
                ellipse_params,
            )
            if args.benchmark and device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            samples_z, _ = model.sample_log(
                cond=cond,
                batch_size=current_bs,
                ddim=True,
                ddim_steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
            samples = decode_samples(
                model,
                samples_z,
                quantize_z=args.quantize_z,
                decode_image=decode_image,
                decode_mask=decode_mask,
            )
            if args.benchmark and device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_start
            if args.benchmark and batch_index >= args.warmup_batches:
                timed_seconds += batch_elapsed
                timed_images += current_bs

            for batch_idx, image in enumerate(samples):
                item = batch_items[batch_idx]
                index = start_index + generated
                stem = f"{index:06}"
                sample_path = sample_dir / f"{stem}.{args.ext}"
                input_path = input_dir / f"{stem}.{args.ext}"
                mask_path = mask_dir / f"{stem}.{args.ext}"
                if sample_path.exists() and not args.overwrite:
                    generated += 1
                    continue
                output_image = tensor_to_pil(image)
                output_image.save(sample_path)
                save_resized_input(item["path"], input_path, output_image)
                if masks:
                    mask_to_pil(masks[batch_idx]).save(mask_path)
                save_manifest_record(
                    manifest,
                    {
                        "index": index,
                        "output": rel_to_outdir(sample_path, outdir),
                        "input": rel_to_outdir(input_path, outdir),
                        "mask": rel_to_outdir(mask_path, outdir) if masks else None,
                        "source": str(item["path"]),
                        "relpath": item["relpath"],
                        "config": args.config,
                        "ckpt": args.ckpt,
                        "first_stage_ckpt": args.first_stage_ckpt,
                        "ddim_steps": args.ddim_steps,
                        "ddim_eta": args.ddim_eta,
                        "seed": args.seed,
                    },
                )
                generated += 1
            batch_index += 1

    print(f"Saved {generated} samples under: {sample_dir}")
    print(f"Saved resized inputs under: {input_dir}")
    if first_stage_needs_mask(model):
        print(f"Saved decode masks under: {mask_dir}")
    if args.benchmark:
        if timed_images > 0:
            print(
                "Benchmark: "
                f"gpu={device}, batch_size={args.batch_size}, resolution={image_size}x{image_size}, "
                f"ddim_steps={args.ddim_steps}, warmup_batches={args.warmup_batches}, "
                f"timed_images={timed_images}, total_generation_seconds={timed_seconds:.4f}, "
                f"avg_seconds_per_image={timed_seconds / timed_images:.4f}"
            )
        else:
            print("Benchmark: no timed images; reduce --warmup-batches or increase --count.")


if __name__ == "__main__":
    main()
