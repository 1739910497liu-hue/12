import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def set_if_exists(config, dotted_path, value):
    node = config
    keys = dotted_path.split(".")
    for key in keys[:-1]:
        if key not in node:
            return
        node = node[key]
    node[keys[-1]] = value


def as_bool_text(value):
    return "True" if value else "False"


def build_config(args):
    from omegaconf import OmegaConf

    base_config = ROOT / args.base
    if not base_config.exists():
        base_config = Path(args.base)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base}")

    train_root = Path(args.train_root)
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {args.train_root}")

    val_root = Path(args.val_root) if args.val_root else train_root
    if not val_root.exists():
        raise FileNotFoundError(f"Validation root not found: {val_root}")

    config = OmegaConf.load(str(base_config))
    config.model.base_learning_rate = float(args.lr)
    config.model.params.cond_stage_key = args.cond_stage_key

    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {args.ckpt}")
        config.model.params.ckpt_path = str(ckpt)

    if args.first_stage_ckpt:
        first_stage_ckpt = Path(args.first_stage_ckpt)
        if not first_stage_ckpt.exists():
            raise FileNotFoundError(f"First-stage checkpoint not found: {args.first_stage_ckpt}")
        config.model.params.first_stage_config.params.ckpt_path = str(first_stage_ckpt)

    ignore_keys = list(args.ignore_key)
    if args.ignore_unet_attn:
        ignore_keys.extend([
            "model.diffusion_model.input_blocks",
            "model.diffusion_model.middle_block",
            "model.diffusion_model.output_blocks",
            "model.diffusion_model.out",
        ])
    if ignore_keys:
        config.model.params.ignore_keys = ignore_keys

    if args.batch_size is not None:
        config.data.params.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        config.data.params.num_workers = int(args.num_workers)
    if args.accumulate_grad_batches is not None:
        set_if_exists(config, "lightning.trainer.accumulate_grad_batches", int(args.accumulate_grad_batches))

    train_params = {
        "data_root": str(train_root),
        "size": int(args.size),
        "image_read_mode": args.image_read_mode,
        "degradation": args.degradation,
        "Cf_r": float(args.cf_r),
        "mask_mode": args.mask_mode,
        "default_caption": args.default_caption,
        "caption_from_filename": bool(args.caption_from_filename),
        "caption_file": args.train_caption_file,
    }
    val_params = {
        "data_root": str(val_root),
        "size": int(args.size),
        "image_read_mode": args.image_read_mode,
        "degradation": args.degradation,
        "Cf_r": float(args.cf_r),
        "mask_mode": args.mask_mode,
        "default_caption": args.default_caption,
        "caption_from_filename": bool(args.caption_from_filename),
        "caption_file": args.val_caption_file,
    }
    if args.train_numeric_file:
        train_params["numeric_file"] = args.train_numeric_file
    if args.val_numeric_file:
        val_params["numeric_file"] = args.val_numeric_file
    if args.parameter_normalizer:
        normalizer = [float(item) for item in args.parameter_normalizer.split(",")]
        train_params["parameter_normalizer"] = normalizer
        val_params["parameter_normalizer"] = normalizer

    if args.caption_key != "caption":
        train_params["caption_key"] = args.caption_key
        val_params["caption_key"] = args.caption_key

    if "cond_stage_config" in config.model.params and "params" in config.model.params.cond_stage_config:
        cond_params = config.model.params.cond_stage_config.params
        if args.clip_version:
            cond_params.version = args.clip_version
        if args.short_encoder_type:
            cond_params.short_encoder_type = args.short_encoder_type
        if args.short_encoder_ckpt:
            cond_params.short_encoder_ckpt = args.short_encoder_ckpt
        if args.freeze_short_encoder is not None:
            cond_params.freeze_short_encoder = bool(args.freeze_short_encoder)
        if args.normalize_parameters is not None:
            cond_params.normalize_parameters = bool(args.normalize_parameters)

    config.data.params.train.target = "ldm.data.imagenet.SS304SRTrain"
    config.data.params.train.params = train_params
    config.data.params.validation.target = "ldm.data.imagenet.SS304SRValidation"
    config.data.params.validation.params = val_params

    out_config = Path(args.out_config) if args.out_config else (
        ROOT / args.logdir / args.tag / "finetune_configs" / "ss304-ldm-text-numeric-vq-finetune.yaml"
    )
    out_config.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, str(out_config))
    return out_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune/train SS304 text+numeric conditioned stage-2 LDM.")
    parser.add_argument("--base", default="configs/latent-diffusion/ss304-ldm-text-numeric-vq.yaml")
    parser.add_argument("--ckpt", default=None, help="Optional pretrained text stage-2 checkpoint.")
    parser.add_argument("--first-stage-ckpt", default=None, help="Optional VQGAN/VQModelWithMask checkpoint override.")
    parser.add_argument("--train-root", default="H:\\\u6570\u636e\u96c6\\ss304\\train")
    parser.add_argument("--val-root", default="H:\\\u6570\u636e\u96c6\\ss304\\valid")
    parser.add_argument("--train-caption-file", default="datasets/text/train_text_categories_label_corrected_original_format.json")
    parser.add_argument("--val-caption-file", default="datasets/text/valid_categories_label_corrected_original_format.json")
    parser.add_argument("--train-numeric-file", default="datasets/text/train_text_actual_parameters_original_format.json")
    parser.add_argument("--val-numeric-file", default="datasets/text/valid_actual_parameters_original_format.json")
    parser.add_argument("--out-config", default=None, help="Where to save the generated text fine-tune config.")
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--image-read-mode", choices=["resize", "crop"], default="resize")
    parser.add_argument("--degradation", default="pil_nearest")
    parser.add_argument("--cf-r", type=float, default=0.0)
    parser.add_argument("--mask-mode", choices=["random", "heat_source", "ellipse"], default="ellipse")
    parser.add_argument("--default-caption", default="SS304 TIG welding heat source image")
    parser.add_argument("--caption-from-filename", action="store_true")
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--cond-stage-key", default="text_params")
    parser.add_argument("--parameter-normalizer", default="22.0,350.0,75.0,40.0")
    parser.add_argument("--clip-version", default=None)
    parser.add_argument("--short-encoder-type", choices=["none", "ccgan_y2h"], default="none")
    parser.add_argument("--short-encoder-ckpt", default=None)
    parser.add_argument("--freeze-short-encoder", action="store_true")
    parser.add_argument("--normalize-parameters", action="store_true")
    parser.add_argument("--ignore-key", action="append", default=[])
    parser.add_argument(
        "--ignore-unet-attn",
        action="store_true",
        help="Skip UNet weights when loading a checkpoint with a different attention/text architecture.",
    )
    parser.add_argument("--tag", default="ss304_ldm_text_numeric_finetune")
    parser.add_argument("--gpus", default="0,")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-lr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_config = build_config(args)
    command = [
        sys.executable,
        "main.py",
        "--base",
        str(out_config),
        "-t",
        "--no-test",
        "True",
        "--epoch",
        str(args.epoch),
        "--tag",
        args.tag,
        "--gpus",
        args.gpus,
        "--logdir",
        args.logdir,
        "--seed",
        str(args.seed),
        "--scale_lr",
        as_bool_text(args.scale_lr),
    ]

    print("Generated text training config:", out_config)
    print("Run command:")
    print(" ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
