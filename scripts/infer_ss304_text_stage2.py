import argparse
import json
import os
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
    model.to(device)
    model.eval()
    return model


def assert_text_model(model):
    conditioning_key = getattr(getattr(model, "model", None), "conditioning_key", None)
    cond_stage_key = getattr(model, "cond_stage_key", None)
    if conditioning_key != "crossattn":
        raise RuntimeError(
            "This script requires a text-conditioned stage-2 model "
            f"(conditioning_key='crossattn'); got conditioning_key={conditioning_key!r}."
        )
    if cond_stage_key != "caption":
        print(f"Warning: cond_stage_key is {cond_stage_key!r}, expected 'caption'.")


def tensor_to_pil(x):
    x = x.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def tensor_batch_to_pil(x):
    return [tensor_to_pil(item) for item in x]


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


def decode_samples(model, samples_z, quantize_z=False):
    if quantize_z and hasattr(model.first_stage_model, "quantize"):
        device = getattr(model, "device", samples_z.device)
        z = samples_z.to(device)
        z = 1.0 / model.scale_factor * z
        quant, _, _ = model.first_stage_model.quantize(z)
        if first_stage_needs_mask(model):
            batch_size = quant.shape[0]
            image_size = int(getattr(getattr(model.first_stage_model, "ddconfig", {}), "resolution", 256))
            dummy_image = torch.zeros(batch_size, 3, image_size, image_size, device=quant.device, dtype=quant.dtype)
            full_mask = torch.ones(batch_size, 1, image_size, image_size, device=quant.device, dtype=quant.dtype)
            return model.first_stage_model.decode(quant, dummy_image, full_mask)
        return model.first_stage_model.decode(quant)

    if first_stage_needs_mask(model):
        batch_size = samples_z.shape[0]
        image_size = int(getattr(getattr(model.first_stage_model, "ddconfig", {}), "resolution", 256))
        dummy_image = torch.zeros(batch_size, 3, image_size, image_size, device=samples_z.device, dtype=samples_z.dtype)
        full_mask = torch.ones(batch_size, 1, image_size, image_size, device=samples_z.device, dtype=samples_z.dtype)
        return model.decode_first_stage(samples_z, image=dummy_image, mask=full_mask)

    return model.decode_first_stage(samples_z)


def read_prompts(args):
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    if not prompts:
        prompts.append("SS304 TIG welding heat source image")
    return prompts


def batched_prompts(prompts, start, batch_size):
    return [prompts[(start + i) % len(prompts)] for i in range(batch_size)]


def parse_params(text):
    values = [float(item.strip()) for item in str(text).split(",")]
    if len(values) != 4:
        raise ValueError("--params must be: voltage,current,speed_cm_min,gas_flow_l_min")
    return values


def batched_params(params_list, start, batch_size):
    return [params_list[(start + i) % len(params_list)] for i in range(batch_size)]


def load_validation_items(config_path, limit=None):
    from omegaconf import OmegaConf
    from ldm.data.imagenet import FolderImageDataset

    config = OmegaConf.load(config_path)
    val_params = config.data.params.validation.params
    dataset = FolderImageDataset(
        data_root=val_params.data_root,
        default_caption=val_params.get("default_caption", "SS304 TIG welding heat source image"),
        caption_from_filename=val_params.get("caption_from_filename", False),
        caption_file=val_params.get("caption_file", None),
        parameter_normalizer=val_params.get("parameter_normalizer", (22.0, 350.0, 75.0, 40.0)),
        numeric_file=val_params.get("numeric_file", None),
    )
    count = len(dataset) if limit is None else min(int(limit), len(dataset))
    return [dataset[i] for i in range(count)]


def load_validation_dataset(config_path, input_root=None):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    if input_root is not None:
        config.data.params.validation.params.data_root = input_root
    return instantiate_from_config(config.data.params.validation)


def save_input_image(path, save_path, size=256, ext="png"):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        if ext.lower() in ("jpg", "jpeg"):
            image.save(save_path, quality=95)
        else:
            image.save(save_path)


def save_pil(image, save_path, ext="png"):
    if ext.lower() in ("jpg", "jpeg"):
        image.save(save_path, quality=95)
    else:
        image.save(save_path)


def build_enhanced_validation_prompt(caption, params):
    voltage, current, speed, gas_flow = [float(v) for v in params]
    return (
        "SS304 TIG welding heat source thermal image, grayscale infrared molten pool morphology, "
        "arc heat distribution, weld seam heat affected zone, stainless steel plate welding. "
        f"{caption} "
        f"Process parameters: voltage {voltage:g} V, current {current:g} A, "
        f"travel speed {speed:g} cm/min, shielding gas flow {gas_flow:g} L/min."
    )


def parse_param_ranges(items):
    if items is None:
        return [(17.0, 22.0), (180.0, 320.0), (15.0, 45.0), (10.0, 40.0)]
    if len(items) != 4:
        raise ValueError("--param-ranges requires four ranges: voltage current speed gas_flow")
    ranges = []
    for item in items:
        parts = [float(value.strip()) for value in str(item).split(",")]
        if len(parts) != 2:
            raise ValueError("Each --param-ranges item must be min,max")
        low, high = parts
        if high < low:
            low, high = high, low
        ranges.append((low, high))
    return ranges


def random_params(ranges):
    return [float(np.random.uniform(low, high)) for low, high in ranges]


def power_density_from_params(params):
    voltage, current, speed, _gas_flow = [float(v) for v in params]
    if abs(speed) < 1e-12:
        return float("inf")
    return voltage * current / speed


def normalize_defect_label(value):
    return str(value).strip().lower().replace("_", " ")


def defect_from_text(text):
    import re

    match = re.search(r"defect\s+type\s+(.+?)(?:\.|,|;|$)", str(text), flags=re.IGNORECASE)
    if match:
        return normalize_defect_label(match.group(1))
    known = [
        "burn through",
        "high travel speed",
        "good weld",
        "contamination",
        "lack of shielding gas",
        "lack of fusion",
    ]
    lowered = normalize_defect_label(text)
    for label in known:
        if label in lowered:
            return label
    if "no defect" in lowered or "no visible defect" in lowered:
        return "good weld"
    return None


def select_indices_by_power_density(dataset, target_power_density, count, target_defect=None):
    scored = []
    for idx in range(len(dataset)):
        caption = None
        if hasattr(dataset, "paths") and hasattr(dataset, "_numeric_from_map") and hasattr(dataset, "_extract_numeric_params"):
            path = dataset.paths[idx]
            numeric_source = dataset._numeric_from_map(path)
            if hasattr(dataset, "_caption_from_map"):
                caption = dataset._caption_from_map(path)
            if numeric_source is None:
                numeric_source = caption
            if numeric_source is None:
                numeric_source = getattr(dataset, "default_caption", "")
            params, _params_norm = dataset._extract_numeric_params(numeric_source)
            params = np.asarray(params, dtype=np.float32).tolist()
        else:
            item = dataset[idx]
            caption = item.get("caption")
            params = np.asarray(item.get("numeric_params"), dtype=np.float32).tolist()
        if target_defect is not None:
            item_defect = defect_from_text(caption)
            if item_defect != target_defect:
                continue
        power_density = power_density_from_params(params)
        scored.append((abs(power_density - target_power_density), idx, power_density, params))
    if not scored and target_defect is not None:
        print(f"Warning: no validation inputs found for defect={target_defect!r}; falling back to power-density only.")
        return select_indices_by_power_density(dataset, target_power_density, count, target_defect=None)
    scored.sort(key=lambda row: row[0])
    selected = scored[: min(count, len(scored))]
    if selected:
        best = selected[0]
        worst = selected[-1]
        print(
            "Selected input images by nearest power density "
            f"target={target_power_density:g}; "
            f"best={best[2]:.6g} params={best[3]}; "
            f"worst_selected={worst[2]:.6g} params={worst[3]}."
        )
    return [idx for _distance, idx, _power_density, _params in selected]


def build_condition(model, captions, params_batch, device):
    cond_stage_key = getattr(model, "cond_stage_key", None)
    if cond_stage_key == "text_params" or isinstance(getattr(model, "cond_stage_model", None), torch.nn.Module) and hasattr(model.cond_stage_model, "physical_encoder"):
        numeric = torch.tensor(params_batch, dtype=torch.float32, device=device)
        conditioning = model.get_learned_conditioning({
            "caption": captions,
            "numeric_params": numeric,
        })
    else:
        conditioning = model.get_learned_conditioning(captions)
    return {"c_crossattn": [conditioning]}


def save_manifest_record(manifest, record):
    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest.flush()


def rel_to_outdir(path, outdir):
    return str(Path(path).relative_to(outdir)).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="Text-conditioned SS304 stage-2 LDM inference.")
    parser.add_argument("--config", default="configs/latent-diffusion/ss304-ldm-text-vq.yaml")
    parser.add_argument("--ckpt", required=True, help="Text-conditioned stage-2 checkpoint.")
    parser.add_argument("--first-stage-ckpt", default=None, help="Override first-stage VQGAN checkpoint in config.")
    parser.add_argument("--clip-version", default=None, help="Override local CLIP folder in config.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None, help="Text file with one prompt per line.")
    parser.add_argument("--input-root", default=None, help="Input image folder used by validation-log-images-mode.")
    parser.add_argument("--validation-mode", action="store_true", help="Use validation data from the config, matching training validation mode.")
    parser.add_argument("--validation-log-images-mode", action="store_true", help="Run model.log_images on validation batches, matching the training ImageLogger path.")
    parser.add_argument("--enhance-validation-prompt", action="store_true", help="Rewrite validation captions into generation-oriented thermal-source prompts.")
    parser.add_argument("--save-inputs", action="store_true", help="Save validation input images resized to generated image size/type.")
    parser.add_argument("--random-params", action="store_true", help="Use random welding parameters for each input image.")
    parser.add_argument("--param-ranges", nargs=4, default=None, help="Four ranges: voltage current speed gas_flow, each as min,max.")
    parser.add_argument("--target-power-density", type=float, default=None, help="Target power density: voltage * current / speed.")
    parser.add_argument("--select-input-by-power-density", action="store_true", help="Select validation input images closest to --target-power-density.")
    parser.add_argument("--select-input-defect", default=None, help="When selecting inputs, prefer the same defect type before nearest power density.")
    parser.add_argument(
        "--params",
        action="append",
        default=None,
        help="Raw welding parameters as voltage,current,speed_cm_min,gas_flow_l_min. Can be repeated.",
    )
    parser.add_argument("--outdir", default="outputs/ss304_text_stage2")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--ext", default="png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--quantize-z", action="store_true", help="Snap final latents to the VQ codebook before decoding.")
    parser.add_argument("--verbose-load", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    seed_everything(args.seed)

    model = load_model(
        args.config,
        args.ckpt,
        device,
        first_stage_ckpt=args.first_stage_ckpt,
        clip_version=args.clip_version,
        verbose=args.verbose_load,
    )
    assert_text_model(model)

    outdir = Path(args.outdir)
    sample_dir = outdir / "samples"
    input_dir = outdir / "inputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if args.save_inputs or args.validation_mode or args.validation_log_images_mode:
        input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.jsonl"
    start_index = 0 if args.overwrite else len(list(sample_dir.glob(f"*.{args.ext}")))

    if args.validation_log_images_mode:
        validation_dataset = load_validation_dataset(args.config, input_root=args.input_root)
        count = min(args.count, len(validation_dataset))
        selected_indices = None
        if args.select_input_by_power_density:
            if args.target_power_density is None:
                raise ValueError("--select-input-by-power-density requires --target-power-density")
            target_defect = normalize_defect_label(args.select_input_defect) if args.select_input_defect else None
            selected_indices = select_indices_by_power_density(validation_dataset, args.target_power_density, count, target_defect=target_defect)
            count = len(selected_indices)
        prompts = read_prompts(args)
        fixed_params_list = [parse_params(item) for item in args.params] if args.params else None
        param_ranges = parse_param_ranges(args.param_ranges)
        generated = 0
        with torch.no_grad(), open(manifest_path, "a", encoding="utf-8") as manifest:
            for start in trange(0, count, args.batch_size, desc="Validation log_images"):
                batch_indices = range(start, min(start + args.batch_size, count))
                if selected_indices is None:
                    items = [validation_dataset[i] for i in batch_indices]
                else:
                    items = [validation_dataset[selected_indices[i]] for i in batch_indices]
                for local_idx, item in enumerate(items):
                    global_idx = start + local_idx
                    if args.prompt or args.prompts_file:
                        item["caption"] = prompts[global_idx % len(prompts)]
                    if args.random_params:
                        item["numeric_params"] = np.asarray(random_params(param_ranges), dtype=np.float32)
                    elif fixed_params_list is not None:
                        item["numeric_params"] = np.asarray(fixed_params_list[global_idx % len(fixed_params_list)], dtype=np.float32)
                batch = default_collate(items)
                logs = model.log_images(
                    batch,
                    N=len(items),
                    n_row=min(2, len(items)),
                    sample=True,
                    ddim_steps=args.ddim_steps,
                    ddim_eta=args.ddim_eta,
                    quantize_denoised=False,
                    inpaint=False,
                    plot_denoise_rows=False,
                    plot_progressive_rows=False,
                    plot_diffusion_rows=False,
                    return_keys=["inputs", "reconstruction", "samples"],
                )
                sample_images = tensor_batch_to_pil(logs["samples"])
                input_images = tensor_batch_to_pil(logs["inputs"])
                recon_images = tensor_batch_to_pil(logs["reconstruction"])
                recon_dir = outdir / "reconstructions"
                direct_recon_dir = outdir / "stage1_direct_reconstructions"
                recon_dir.mkdir(parents=True, exist_ok=True)
                direct_recon_dir.mkdir(parents=True, exist_ok=True)
                x_for_direct = batch["image"].to(device)
                if len(x_for_direct.shape) == 4 and x_for_direct.shape[-1] in (1, 3):
                    x_for_direct = x_for_direct.permute(0, 3, 1, 2).contiguous().float()
                mask_for_direct = batch.get("mask", None)
                direct_recon_images = None
                if mask_for_direct is not None and hasattr(model.first_stage_model, "decode"):
                    mask_for_direct = mask_for_direct.to(device).float()
                    if len(mask_for_direct.shape) == 3:
                        mask_for_direct = mask_for_direct[:, None, ...]
                    try:
                        direct_recon, _ = model.first_stage_model([x_for_direct, mask_for_direct])
                        direct_recon_images = tensor_batch_to_pil(direct_recon)
                    except Exception as exc:
                        print(f"Warning: failed to save direct first-stage reconstructions: {exc}")
                for batch_idx, sample_image in enumerate(sample_images):
                    index = start_index + generated
                    sample_path = sample_dir / f"{index:06}.{args.ext}"
                    input_path = input_dir / f"{index:06}.{args.ext}"
                    recon_path = recon_dir / f"{index:06}.{args.ext}"
                    direct_recon_path = direct_recon_dir / f"{index:06}.{args.ext}"
                    save_pil(sample_image, sample_path, args.ext)
                    save_pil(input_images[batch_idx], input_path, args.ext)
                    save_pil(recon_images[batch_idx], recon_path, args.ext)
                    direct_recon_relpath = None
                    if direct_recon_images is not None:
                        save_pil(direct_recon_images[batch_idx], direct_recon_path, args.ext)
                        direct_recon_relpath = rel_to_outdir(direct_recon_path, outdir)
                    item = items[batch_idx]
                    save_manifest_record(
                        manifest,
                        {
                            "index": index,
                            "output": rel_to_outdir(sample_path, outdir),
                            "input": rel_to_outdir(input_path, outdir),
                            "reconstruction": rel_to_outdir(recon_path, outdir),
                            "stage1_direct_reconstruction": direct_recon_relpath,
                            "prompt": item.get("caption"),
                            "numeric_params": np.asarray(item.get("numeric_params"), dtype=np.float32).tolist(),
                            "power_density": power_density_from_params(np.asarray(item.get("numeric_params"), dtype=np.float32).tolist()),
                            "target_power_density": args.target_power_density,
                            "source_image": item.get("file_path_"),
                            "source_relpath": item.get("relpath"),
                            "mode": "validation_log_images",
                            "ddim_steps": args.ddim_steps,
                            "ddim_eta": args.ddim_eta,
                            "seed": args.seed,
                        },
                    )
                    generated += 1
        print(f"Saved {generated} validation-log_images samples under: {sample_dir}")
        return

    validation_items = load_validation_items(args.config, limit=args.count) if args.validation_mode else None
    if validation_items is not None:
        args.count = len(validation_items)
    else:
        prompts = read_prompts(args)
        params_list = [parse_params(item) for item in args.params] if args.params else [[17.2, 200.0, 19.0, 30.0]]

    context = model.ema_scope("Text sampling") if getattr(model, "use_ema", False) and not args.no_ema else nullcontext()
    generated = 0
    with torch.no_grad(), context, open(manifest_path, "a", encoding="utf-8") as manifest:
        for _ in trange((args.count + args.batch_size - 1) // args.batch_size, desc="Text sampling"):
            current_bs = min(args.batch_size, args.count - generated)
            if current_bs <= 0:
                break

            if validation_items is not None:
                batch_items = validation_items[generated:generated + current_bs]
                batch_params = [np.asarray(item["numeric_params"], dtype=np.float32).tolist() for item in batch_items]
                if args.enhance_validation_prompt:
                    batch_prompt = [
                        build_enhanced_validation_prompt(item["caption"], params)
                        for item, params in zip(batch_items, batch_params)
                    ]
                else:
                    batch_prompt = [item["caption"] for item in batch_items]
            else:
                batch_items = [None] * current_bs
                batch_prompt = batched_prompts(prompts, generated, current_bs)
                batch_params = batched_params(params_list, generated, current_bs)
            cond = build_condition(model, batch_prompt, batch_params, device)
            samples_z, _ = model.sample_log(
                cond=cond,
                batch_size=current_bs,
                ddim=True,
                ddim_steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
            samples = decode_samples(model, samples_z, quantize_z=args.quantize_z)

            for batch_idx, image in enumerate(samples):
                index = start_index + generated
                save_path = sample_dir / f"{index:06}.{args.ext}"
                if save_path.exists() and not args.overwrite:
                    generated += 1
                    continue
                tensor_to_pil(image).save(save_path)
                input_relpath = None
                if validation_items is not None or args.save_inputs:
                    item = batch_items[batch_idx]
                    if item is not None:
                        input_path = input_dir / f"{index:06}.{args.ext}"
                        save_input_image(item["file_path_"], input_path, size=256, ext=args.ext)
                        input_relpath = rel_to_outdir(input_path, outdir)
                save_manifest_record(
                    manifest,
                    {
                        "index": index,
                        "output": rel_to_outdir(save_path, outdir),
                        "input": input_relpath,
                        "prompt": batch_prompt[batch_idx],
                        "numeric_params": batch_params[batch_idx],
                        "source_image": None if batch_items[batch_idx] is None else batch_items[batch_idx]["file_path_"],
                        "source_relpath": None if batch_items[batch_idx] is None else batch_items[batch_idx]["relpath"],
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

    print(f"Saved {generated} samples under: {sample_dir}")


if __name__ == "__main__":
    main()
