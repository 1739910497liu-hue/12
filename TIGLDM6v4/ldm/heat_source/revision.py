"""Reviewer protocol v2. Legacy configurations remain in configs/heat_source/."""
import copy
import json
from pathlib import Path
from .experiments import BASE

VARIANTS = {
    "full": {},
    "no_prior": {"fusion": "off"},
    "plain": {"fusion": "plain"},
    "perturb_none": {"perturbation": "none"},
    "perturb_small": {"perturbation": "small"},
    "perturb_large": {"perturbation": "large"},
    "f4": {"factor": 4}, "f16": {"factor": 16},
    "k8192": {"codebook_size": 8192}, "k32768": {"codebook_size": 32768},
    "condition_none": {"condition": "none", "stage1_group": "full"},
    "condition_text": {"condition": "text", "stage1_group": "full"},
    "condition_numeric": {"condition": "numeric", "stage1_group": "full"},
    "concat_condition": {"diffusion_conditioning": "concat", "stage1_group": "full"},
    "report_raw": {"caption_field": "caption_raw", "stage1_group": "full"},
    "simulation_hybrid": {"prior_source": "simulation", "perturbation": "none"},
}
GROUPS = {
    "fusion": ["no_prior", "plain", "full"],
    "perturbation": ["perturb_none", "perturb_small", "full", "perturb_large"],
    "compression": ["f4", "full", "f16"],
    "codebook": ["k8192", "full", "k32768"],
    "condition": ["condition_none", "condition_text", "condition_numeric", "full"],
    "report": ["condition_numeric", "full", "report_raw"],
    "comparison": ["no_prior", "plain", "concat_condition", "full"],
    "hybrid": ["perturb_none", "simulation_hybrid"],
}
CORE = list(dict.fromkeys(name for group in ("fusion", "perturbation", "compression", "codebook", "condition") for name in GROUPS[group]))


def experiment_config(name):
    config = copy.deepcopy(BASE)
    config.update(protocol="reviewer_v2", experiment=name, stage1_group=name,
                  condition="both", caption_field="caption", prior_source="analytic",
                  diffusion_conditioning="residual")
    config.update(VARIANTS[name])
    return config


def write_configs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        (directory / (name + ".json")).write_text(json.dumps(experiment_config(name), indent=2) + "\n", encoding="utf-8")
    study = {"protocol": "reviewer_v2", "seeds": [42, 123, 2026], "groups": GROUPS,
             "experiments": list(VARIANTS), "core": CORE,
             "core_stage1_per_seed": len({experiment_config(n)["stage1_group"] for n in CORE}),
             "core_stage2_per_seed": len(CORE),
             "all_stage1_per_seed": len({experiment_config(n)["stage1_group"] for n in VARIANTS}),
             "all_stage2_per_seed": len(VARIANTS),
             "runtime_steps": [10, 20, 50, 100], "runtime_warmup": 5, "runtime_repeats": 30,
             "runtime_batch_size": 1, "deployment_deadline_ms": None,
             "external_comparisons": "Require published implementation or real simulation fields; no fabricated results."}
    (directory / "study.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
