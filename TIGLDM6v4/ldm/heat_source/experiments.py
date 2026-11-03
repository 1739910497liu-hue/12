"""One-factor-at-a-time study with shared baselines and fixed compute budgets."""
import copy
import json
from pathlib import Path

BASE = {
    "experiment": "baseline", "stage1_group": "baseline",
    "image_size": 256, "factor": 8, "embed_dim": 4, "codebook_size": 16384,
    "channels": 128, "res_blocks": 2, "fusion": "gated", "perturbation": "baseline",
    "condition": "none", "unet_channels": 192, "unet_res_blocks": 2,
    "unet_mult": [1, 2, 2, 4, 4], "attention_resolutions": [4, 8],
    "heads": 8, "context_dim": 512, "timesteps": 1000,
    "beta_start": 0.0015, "beta_end": 0.0155,
    "batch_size": 2, "accumulate": 4, "workers": 4,
    "stage1_steps": 50000, "stage2_steps": 100000,
    "stage1_lr": 4.5e-6, "stage2_lr": 5e-5,
    "warmup_steps": 1000, "disc_start": 10000, "disc_weight": 0.1,
    "perceptual_weight": 1.0, "codebook_weight": 1.0,
    "validate_every": 1000, "save_every": 1000, "log_every": 50,
    "precision": "fp32", "clip_grad": 1.0, "ddim_steps": 50,
    "latent_scale_images": 512,
}
VARIANTS = {
    "baseline": {},
    "fusion_plain": {"fusion": "plain"},
    "perturb_none": {"perturbation": "none"},
    "perturb_small": {"perturbation": "small"},
    "perturb_large": {"perturbation": "large"},
    "compression_f4": {"factor": 4},
    "compression_f16": {"factor": 16},
    "codebook_8192": {"codebook_size": 8192},
    "codebook_32768": {"codebook_size": 32768},
    "condition_text": {"condition": "text", "perturbation": "none", "stage1_group": "perturb_none"},
    "condition_numeric": {"condition": "numeric", "perturbation": "none", "stage1_group": "perturb_none"},
    "condition_both": {"condition": "both", "perturbation": "none", "stage1_group": "perturb_none"},
}
GROUPS = {
    "fusion": ["fusion_plain", "baseline"],
    "perturbation": ["perturb_none", "perturb_small", "baseline", "perturb_large"],
    "compression": ["compression_f4", "baseline", "compression_f16"],
    "codebook": ["codebook_8192", "baseline", "codebook_32768"],
    "condition": ["perturb_none", "condition_text", "condition_numeric", "condition_both"],
}


def experiment_config(name):
    config = copy.deepcopy(BASE)
    config.update(experiment=name, stage1_group=name)
    config.update(VARIANTS[name])
    return config


def write_configs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        (directory / (name + ".json")).write_text(json.dumps(experiment_config(name), indent=2) + "\n", encoding="utf-8")
    (directory / "study.json").write_text(json.dumps({"seeds": [42, 123, 2026], "groups": GROUPS,
        "unique_stage1_per_seed": 9, "unique_stage2_per_seed": 12}, indent=2) + "\n", encoding="utf-8")
