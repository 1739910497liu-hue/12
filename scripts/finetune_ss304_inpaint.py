import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def set_if_exists(config, path, value):
    node = config
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in node:
            return
        node = node[part]
    node[parts[-1]] = value


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SS304 latent inpainting diffusion from an existing model checkpoint."
    )
    parser.add_argument(
        "--base",
        default="configs/latent-diffusion/ss304-ldm-inpaint-vq.yaml",
        help="Base training config.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Existing stage-2 LDM checkpoint. This loads model weights only and starts a new fine-tuning run.",
    )
    parser.add_argument(
        "--first-stage-ckpt",
        default=None,
        help="Optional first-stage VQGAN checkpoint. If omitted, the ckpt_path in the base config is kept.",
    )
    parser.add_argument("--train-root", default=None, help="Optional replacement training dataset root.")
    parser.add_argument("--val-root", default=None, help="Optional replacement validation dataset root.")
    parser.add_argument("--lr", type=float, default=1.0e-5, help="Fine-tuning base learning rate.")
    parser.add_argument("--epoch", type=int, default=80, help="Max epochs for the new fine-tuning run.")
    parser.add_argument("--tag", default="ss304_ldm_inpaint_finetune", help="Log subfolder and wandb tag.")
    parser.add_argument("--gpus", default="0,", help="Lightning GPU argument, for example '0,'.")
    parser.add_argument("--logdir", default="logs", help="Root log directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument(
        "--ignore-key",
        action="append",
        default=[],
        help="Checkpoint key prefix to skip when loading weights. Can be used multiple times.",
    )
    parser.add_argument(
        "--ignore-input-conv",
        action="store_true",
        help="Skip the UNet first input convolution, useful when fine-tuning from a non-inpainting 4-channel LDM.",
    )
    parser.add_argument("--scale-lr", action="store_true", help="Use main.py learning-rate scaling.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the generated config and print command.")
    args = parser.parse_args()

    base_path = Path(args.base)
    ckpt_path = Path(args.ckpt)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = OmegaConf.load(base_path)
    config.model.base_learning_rate = args.lr
    config.model.params.ckpt_path = str(ckpt_path)
    ignore_keys = list(args.ignore_key)
    if args.ignore_input_conv:
        ignore_keys.append("model.diffusion_model.input_blocks.0.0")
    if ignore_keys:
        config.model.params.ignore_keys = ignore_keys

    if args.first_stage_ckpt is not None:
        first_stage_ckpt = Path(args.first_stage_ckpt)
        if not first_stage_ckpt.is_file():
            raise FileNotFoundError(f"First-stage checkpoint not found: {first_stage_ckpt}")
        set_if_exists(config, "model.params.first_stage_config.params.ckpt_path", str(first_stage_ckpt))

    if args.train_root is not None:
        set_if_exists(config, "data.params.train.params.data_root", args.train_root)
    if args.val_root is not None:
        set_if_exists(config, "data.params.validation.params.data_root", args.val_root)
    if args.batch_size is not None:
        config.data.params.batch_size = args.batch_size
    if args.accumulate_grad_batches is not None:
        set_if_exists(config, "lightning.trainer.accumulate_grad_batches", args.accumulate_grad_batches)

    generated_dir = Path(args.logdir) / args.tag / "finetune_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_config = generated_dir / "ss304-ldm-inpaint-vq-finetune.yaml"
    OmegaConf.save(config, generated_config)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("WANDB_MODE", "offline")

    command = [
        sys.executable,
        "main.py",
        "--base",
        str(generated_config),
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
        "True" if args.scale_lr else "False",
    ]

    print("Generated config:", generated_config)
    print("Command:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
