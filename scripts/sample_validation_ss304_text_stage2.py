import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate
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


def load_model(config_path, ckpt_path, device, first_stage_ckpt=None, clip_version=None, verbose=False):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    if first_stage_ckpt is not None:
        config.model.params.first_stage_config.params.ckpt_path = first_stage_ckpt
    if clip_version is not None:
        config.model.params.cond_stage_config.params.version = clip_version

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
    ema_missing = [key for key in missing if key.startswith("model_ema.")]
    if ema_missing and getattr(model, "use_ema", False):
        print(
            "Warning: checkpoint is missing EMA weights "
            f"({len(ema_missing)} model_ema keys). Disabling EMA for inference."
        )
        model.use_ema = False
    model.to(device)
    model.eval()
    return model, config


def assert_text_model(model):
    conditioning_key = getattr(getattr(model, "model", None), "conditioning_key", None)
    if conditioning_key != "crossattn":
        raise RuntimeError(
            "This script requires a text-conditioned stage-2 model "
            f"(conditioning_key='crossattn'); got conditioning_key={conditioning_key!r}."
        )


def first_stage_needs_mask(model):
    decode = getattr(getattr(model, "first_stage_model", None), "decode", None)
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


def mask_to_pil(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().squeeze().numpy()
    mask = (np.asarray(mask, dtype=np.float32).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def load_decode_image(path, image_size, device):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32) * 2.0 - 1.0


def get_config_resolution(config):
    return int(config.model.params.first_stage_config.params.ddconfig.get("resolution", 256))


def get_validation_ellipse_params(config):
    params = config.data.params.validation.params.get("ellipse_params", {})
    return dict(params)


def build_decode_inputs(model, batch_items, image_size, device, mask_mode, ellipse_params):
    if not first_stage_needs_mask(model):
        return None, None, []

    masks = []
    if mask_mode == "full":
        image = torch.zeros(len(batch_items), 3, image_size, image_size, device=device)
        mask = torch.ones(len(batch_items), 1, image_size, image_size, device=device)
        masks = [np.ones((image_size, image_size), dtype=np.float32) for _ in batch_items]
        return image, mask, masks

    images = []
    for item in batch_items:
        mask_np = make_ellipse_mask(image_size, image_size, ellipse_params)
        masks.append(mask_np)
        images.append(load_decode_image(item["path"], image_size, device))
    image = torch.cat(images, dim=0)
    mask = torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(1).to(device=device, dtype=torch.float32)
    return image, mask, masks


def decode_samples(model, samples_z, quantize_z=False, decode_image=None, decode_mask=None):
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


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_resized_input_like_output(input_path, save_path, output_image):
    image = Image.open(input_path)
    if image.mode != output_image.mode:
        image = image.convert(output_image.mode)
    image = image.resize(output_image.size, Image.BICUBIC)
    image.save(save_path)


def save_tensor_image_like_output(tensor, save_path, output_image):
    image = tensor_to_pil(tensor)
    if image.mode != output_image.mode:
        image = image.convert(output_image.mode)
    if image.size != output_image.size:
        image = image.resize(output_image.size, Image.BICUBIC)
    image.save(save_path)


def find_json(root, explicit_json=None):
    if explicit_json:
        path = Path(explicit_json)
        return path if path.exists() else None
    jsons = sorted(Path(root).glob("*.json"))
    return jsons[0] if jsons else None


def resolve_json_image(root, key):
    root = Path(root)
    key_path = Path(key)
    if key_path.is_absolute() and key_path.exists():
        return key_path
    direct = root / key_path
    if direct.exists():
        return direct
    normalized = key.replace("\\", "/")
    for marker in ("/valid/", "/validation/", "/test/", "/train/"):
        if marker in normalized:
            candidate = root / normalized.split(marker, 1)[1]
            if candidate.exists():
                return candidate
    return direct


def caption_from_path(path):
    stem = Path(path).stem.replace("_", " ").replace("-", " ")
    parent = Path(path).parent.name.replace("_", " ").replace("-", " ")
    return f"{parent} {stem}".strip()


def extract_numeric_params_from_caption(caption):
    text = str(caption)
    patterns = [
        r"voltage\s*[:=]?\s*([-+]?\d*\.?\d+)\s*v",
        r"current\s*[:=]?\s*([-+]?\d*\.?\d+)\s*a",
        r"speed\s*[:=]?\s*([-+]?\d*\.?\d+)\s*cm\s*/?\s*min",
        r"gas\s*flow\s*[:=]?\s*([-+]?\d*\.?\d+)\s*l\s*/?\s*min",
    ]
    defaults = [17.2, 200.0, 19.0, 30.0]
    values = []
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        values.append(float(match.group(1)) if match else defaults[idx])
    return values


def build_crossattn_condition(model, captions, numeric_params, device):
    cond_stage_key = getattr(model, "cond_stage_key", None)
    cond_stage_model = getattr(model, "cond_stage_model", None)
    if cond_stage_key == "text_params" or hasattr(cond_stage_model, "physical_encoder"):
        numeric = torch.tensor(numeric_params, dtype=torch.float32, device=device)
        c = model.get_learned_conditioning({
            "caption": captions,
            "numeric_params": numeric,
        })
    else:
        c = model.get_learned_conditioning(captions)
    return {"c_crossattn": [c]}


def normalize_key(value):
    return str(value).replace("\\", "/").strip().lstrip("./")


def load_prompt_map(prompt_json):
    if prompt_json is None:
        return {}, None

    prompt_json = Path(prompt_json)
    if prompt_json.is_dir():
        valid_json = prompt_json / "valid.json"
        if valid_json.exists():
            prompt_json = valid_json
        else:
            jsons = sorted(prompt_json.glob("*.json"))
            if not jsons:
                raise FileNotFoundError(f"No JSON files found under prompt JSON directory: {prompt_json}")
            prompt_json = jsons[0]
    if not prompt_json.exists():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_json}")

    with open(prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt_map = {}
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = []
        for item in data:
            if isinstance(item, dict):
                key = item.get("file") or item.get("path") or item.get("image") or item.get("image_path")
                prompt = item.get("caption") or item.get("prompt") or item.get("text")
                if key is not None and prompt is not None:
                    iterable.append((key, prompt))
    else:
        iterable = []

    for key, prompt in iterable:
        if isinstance(prompt, str) and prompt.strip():
            prompt_map[normalize_key(key)] = prompt.strip()

    print(f"Loaded {len(prompt_map)} prompts from: {prompt_json}")
    return prompt_map, prompt_json


def prompt_candidates(root, path, json_key=None):
    root = Path(root)
    candidates = []
    if json_key is not None:
        candidates.append(normalize_key(json_key))
    try:
        candidates.append(normalize_key(Path(path).relative_to(root)))
    except ValueError:
        pass
    candidates.append(normalize_key(path))
    name = Path(path).name
    candidates.extend([key for key in (name, Path(path).stem) if key])
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def find_prompt(prompt_map, root, path, json_key=None):
    for candidate in prompt_candidates(root, path, json_key=json_key):
        if candidate in prompt_map:
            return prompt_map[candidate]
    path_key = normalize_key(path)
    for key, prompt in prompt_map.items():
        if path_key.endswith(key) or key.endswith(normalize_key(Path(path).name)):
            return prompt
    return None


def load_validation_items(root, validation_json, prompt_json, default_caption,
                          caption_from_filename, caption_from_json_value):
    root = Path(root)
    json_path = find_json(root, validation_json)
    prompt_map, prompt_json_path = load_prompt_map(prompt_json)
    items = []

    if json_path is not None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = data.items() if isinstance(data, dict) else [(item, None) for item in data]
        for key, value in pairs:
            path = resolve_json_image(root, str(key))
            if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            txt_path = path.with_suffix(".txt")
            caption = default_caption
            prompt = find_prompt(prompt_map, root, path, json_key=key)
            if prompt is not None:
                caption = prompt
            elif txt_path.exists():
                caption = txt_path.read_text(encoding="utf-8").strip() or caption
            elif caption_from_json_value and isinstance(value, str) and value.strip():
                caption = value.strip()
            elif caption_from_filename:
                caption = caption_from_path(path)
            items.append({
                "path": path,
                "caption": caption,
                "json_value": value,
                "json_key": key,
                "prompt_json": str(prompt_json_path) if prompt_json_path is not None else None,
            })
        if items:
            print(f"Loaded {len(items)} validation images from JSON: {json_path}")
            return items

    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        txt_path = path.with_suffix(".txt")
        caption = default_caption
        prompt = find_prompt(prompt_map, root, path)
        if prompt is not None:
            caption = prompt
        elif txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip() or caption
        elif caption_from_filename:
            caption = caption_from_path(path)
        items.append({
            "path": path,
            "caption": caption,
            "json_value": None,
            "json_key": None,
            "prompt_json": str(prompt_json_path) if prompt_json_path is not None else None,
        })
    if not items:
        raise FileNotFoundError(f"No validation images found under: {root}")
    print(f"Loaded {len(items)} validation images by scanning: {root}")
    return items


def rel_to_outdir(path, outdir):
    return str(Path(path).relative_to(outdir)).replace("\\", "/")


def save_manifest_record(manifest, record):
    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest.flush()


def load_validation_dataset_from_config(config):
    from ldm.util import instantiate_from_config

    return instantiate_from_config(config.data.params.validation)


def batch_from_dataset(dataset, start, batch_size, prompt_map=None):
    samples = []
    count = min(batch_size, len(dataset) - start)
    for idx in range(start, start + count):
        sample = dict(dataset[idx])
        path = sample.get("file_path_", "")
        relpath = sample.get("relpath", None)
        prompt = find_prompt(prompt_map or {}, getattr(dataset, "data_root", ""), path, json_key=relpath)
        if prompt is not None:
            sample["caption"] = prompt
        samples.append(sample)
    return default_collate(samples)


def sample_training_validation_batch(model, batch, ddim_steps, ddim_eta, quantize_z=False, use_ema=True):
    batch_size = len(batch[model.first_stage_key])
    old_use_ema = getattr(model, "use_ema", False)
    if not use_ema:
        model.use_ema = False
    try:
        logs = model.log_images(
            batch,
            N=batch_size,
            n_row=min(batch_size, 4),
            sample=True,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            quantize_denoised=quantize_z,
            inpaint=False,
            plot_denoise_rows=False,
            plot_progressive_rows=False,
            plot_diffusion_rows=False,
        )
    finally:
        model.use_ema = old_use_ema
    sample_key = "samples_x0_quantized" if quantize_z and "samples_x0_quantized" in logs else "samples"
    masks = (logs["mask"] + 1.0) / 2.0 if "mask" in logs else None
    return logs[sample_key], logs["inputs"], masks


def main():
    parser = argparse.ArgumentParser(description="Sample text-conditioned SS304 stage-2 model on validation images.")
    parser.add_argument("--config", default="configs/latent-diffusion/ss304-ldm-text-vq.yaml")
    parser.add_argument("--ckpt", required=True, help="Text-conditioned stage-2 checkpoint.")
    parser.add_argument("--first-stage-ckpt", default=None, help="Override first-stage VQGAN checkpoint in config.")
    parser.add_argument("--clip-version", default=None, help="Override local CLIP folder in config.")
    parser.add_argument("--input-root", default="datasets/ss304/valid")
    parser.add_argument("--validation-json", default=None, help="Optional validation JSON. Defaults to first JSON under input-root.")
    parser.add_argument("--prompt-json", default=None,
                        help="Optional prompt JSON file or directory. If omitted, captions follow the validation dataset training logic.")
    parser.add_argument("--default-caption", default="SS304 TIG welding heat source image")
    parser.add_argument("--caption-from-filename", action="store_true")
    parser.add_argument("--caption-from-json-value", action="store_true", help="Use JSON string values as prompts when available.")
    parser.add_argument("--outdir", default="outputs/ss304_text_stage2_validation")
    parser.add_argument("--count", type=int, default=-1, help="Number of validation images to sample. Use -1 for all.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ext", default="png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--quantize-z", action="store_true")
    parser.add_argument("--decode-mask-mode", default="ellipse", choices=["ellipse", "full"],
                        help="Only used with --manual-sampling. Default validation mode uses the config validation dataset mask.")
    parser.add_argument("--manual-sampling", action="store_true",
                        help="Use the older manual path instead of model.get_input/log_images-style validation.")
    parser.add_argument("--no-save-masks", action="store_true")
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
        clip_version=args.clip_version,
        verbose=args.verbose_load,
    )
    assert_text_model(model)
    image_size = get_config_resolution(config)
    ellipse_params = get_validation_ellipse_params(config)
    prompt_map, prompt_json_path = load_prompt_map(args.prompt_json)

    if args.manual_sampling:
        items = load_validation_items(
            args.input_root,
            args.validation_json,
            args.prompt_json,
            args.default_caption,
            args.caption_from_filename,
            args.caption_from_json_value,
        )
        validation_dataset = None
    else:
        config.data.params.validation.params.data_root = args.input_root
        validation_dataset = load_validation_dataset_from_config(config)
        items = None
        print("Using training validation rule: dataset batch -> model.get_input -> sample_log(cond=c) -> decode_first_stage(image=x, mask=batch['mask']).")
        print(f"Loaded validation dataset from config: {len(validation_dataset)} samples")

    data_len = len(items) if items is not None else len(validation_dataset)
    total_count = data_len if args.count < 0 else min(args.count, data_len)
    if total_count <= 0:
        raise ValueError("No validation samples selected.")

    outdir = Path(args.outdir)
    sample_dir = outdir / "samples"
    input_dir = outdir / "inputs"
    mask_dir = outdir / "masks"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.jsonl"
    start_index = 0 if args.overwrite else len(list(sample_dir.glob(f"*.{args.ext}")))

    generated = 0
    with torch.no_grad(), open(manifest_path, "a", encoding="utf-8") as manifest:
        for _ in trange((total_count + args.batch_size - 1) // args.batch_size, desc="Validation sampling"):
            current_bs = min(args.batch_size, total_count - generated)
            if current_bs <= 0:
                break

            if args.manual_sampling:
                batch_items = items[generated : generated + current_bs]
                captions = [item["caption"] for item in batch_items]
                numeric_params = [extract_numeric_params_from_caption(caption) for caption in captions]
                cond = build_crossattn_condition(model, captions, numeric_params, device)
                samples_z, _ = model.sample_log(
                    cond=cond,
                    batch_size=current_bs,
                    ddim=True,
                    ddim_steps=args.ddim_steps,
                    eta=args.ddim_eta,
                )
                decode_image, decode_mask, masks = build_decode_inputs(
                    model,
                    batch_items,
                    image_size,
                    device,
                    args.decode_mask_mode,
                    ellipse_params,
                )
                samples = decode_samples(
                    model,
                    samples_z,
                    quantize_z=args.quantize_z,
                    decode_image=decode_image,
                    decode_mask=decode_mask,
                )
                input_batch = None
            else:
                batch = batch_from_dataset(validation_dataset, generated, current_bs, prompt_map=prompt_map)
                batch_items = [
                    {
                        "path": Path(path),
                        "caption": caption,
                        "json_value": None,
                        "json_key": relpath,
                        "prompt_json": str(prompt_json_path) if prompt_json_path is not None else None,
                    }
                    for path, relpath, caption in zip(batch["file_path_"], batch["relpath"], batch["caption"])
                ]
                samples, input_batch, first_stage_mask = sample_training_validation_batch(
                    model,
                    batch,
                    args.ddim_steps,
                    args.ddim_eta,
                    quantize_z=args.quantize_z,
                    use_ema=not args.no_ema,
                )
                masks = first_stage_mask.detach().float().cpu().numpy()[:, 0] if first_stage_mask is not None else []

            for batch_idx, image in enumerate(samples):
                item = batch_items[batch_idx]
                index = start_index + generated
                stem = f"{index:06}"
                sample_path = sample_dir / f"{stem}.{args.ext}"
                input_path = input_dir / f"{stem}.{args.ext}"
                mask_path = mask_dir / f"{stem}.{args.ext}"
                caption_path = sample_dir / f"{stem}.txt"
                if sample_path.exists() and not args.overwrite:
                    generated += 1
                    continue

                output_image = tensor_to_pil(image)
                output_image.save(sample_path)
                if input_batch is None:
                    save_resized_input_like_output(item["path"], input_path, output_image)
                else:
                    save_tensor_image_like_output(input_batch[batch_idx], input_path, output_image)
                has_masks = len(masks) > batch_idx
                if not args.no_save_masks and has_masks:
                    mask_to_pil(masks[batch_idx]).save(mask_path)
                caption_path.write_text(item["caption"], encoding="utf-8")

                save_manifest_record(
                    manifest,
                    {
                        "index": index,
                        "output": rel_to_outdir(sample_path, outdir),
                        "input": rel_to_outdir(input_path, outdir),
                        "mask": rel_to_outdir(mask_path, outdir) if not args.no_save_masks and has_masks else None,
                        "decode_mask_mode": args.decode_mask_mode if args.manual_sampling else "training_validation_dataset",
                        "input_source": str(item["path"]),
                        "prompt": item["caption"],
                        "prompt_json": item["prompt_json"],
                        "json_key": item["json_key"],
                        "json_value": item["json_value"],
                        "config": args.config,
                        "ckpt": args.ckpt,
                        "first_stage_ckpt": args.first_stage_ckpt,
                        "clip_version": args.clip_version,
                        "ddim_steps": args.ddim_steps,
                        "ddim_eta": args.ddim_eta,
                        "seed": args.seed,
                    },
                )
                generated += 1

    print(f"Saved {generated} generated images under: {sample_dir}")
    print(f"Saved resized validation inputs under: {input_dir}")
    if not args.no_save_masks:
        print(f"Saved decoder masks under: {mask_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
