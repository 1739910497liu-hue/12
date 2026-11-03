"""Batch TIGLDM training time/parameters/FLOPs with isolated CPU/GPU workers.

Only stage1/last.pt and stage2/last.pt are training records. Evaluation folders
are not separate trained models. --metadata-only needs no CLIP or image data.
Full profiling uses synthetic, fixed-shape conditions and requires local CLIP
only when the trained model has a text branch; it does not evaluate image quality.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


VQ_REQUIRED = ("image_size", "factor", "embed_dim", "codebook_size", "channels", "res_blocks", "fusion", "perturbation")
VQ_DEFAULTS = {"prior_source": "analytic", "decoder_gate_mode": "mask", "decoder_gate_init": .5,
               "include_raw_prior": False, "decoder_gate_alpha_init": 1.,
               "decoder_gate_alpha_scale": 10000., "decoder_gate_bias_init": -1.}


def load_checkpoint(path):
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(str(path), map_location="cpu", weights_only=False)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metadata(state):
    keys = ("config", "seed", "stage", "step", "manifest_hashes", "first_stage_hash",
            "parameters", "trainable_parameters", "latent_scale", "training_wall_seconds")
    result = {key: state.get(key) for key in keys}
    stage, step, config = result["stage"], result["step"], result["config"]
    if type(stage) is not int or stage not in (1, 2) or not isinstance(config, dict):
        raise ValueError("Not a heat_source.py training checkpoint.")
    budget = config.get("stage{}_steps".format(stage))
    if type(step) is not int or step < 0 or type(budget) is not int or budget < 1:
        raise ValueError("Missing/invalid training step or budget.")
    if type(result["seed"]) is not int or not isinstance(result["manifest_hashes"], dict):
        raise ValueError("Missing training seed/data identity.")
    total, trainable = result["parameters"], result["trainable_parameters"]
    if type(total) is not int or type(trainable) is not int or not 0 <= trainable <= total or total < 1:
        raise ValueError("Missing saved model parameter counts; no tensor/buffer heuristic is used.")
    seconds = result["training_wall_seconds"]
    if not (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
            and math.isfinite(seconds) and seconds > 0):
        result["training_wall_seconds"] = None
    return result


def timing(meta):
    seconds, stage, config = meta.get("training_wall_seconds"), meta["stage"], meta["config"]
    valid = (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
             and math.isfinite(seconds) and seconds > 0)
    complete = meta["step"] == config["stage{}_steps".format(stage)] and not config.get("smoke_test", False)
    return {
        "stage": stage, "step": meta["step"], "budget": config["stage{}_steps".format(stage)],
        "budget_completed": complete, "hours_at_checkpoint": seconds / 3600. if valid else None,
        "completed_training_h": seconds / 3600. if valid and complete else None,
        "status": "missing_training_wall_seconds" if not valid else "complete" if complete else "incomplete_or_smoke",
        "scope": "Accumulated training loop including validation and earlier saves; excludes setup, downtime and final checkpoint save. metrics.jsonl seconds_since_start is per-session and is not substituted.",
    }


def verify_pair(first, second):
    if first["stage"] != 1 or second["stage"] != 2:
        raise ValueError("Expected stage1 and stage2 checkpoints.")
    if first["seed"] != second["seed"] or first["manifest_hashes"] != second["manifest_hashes"]:
        raise ValueError("Stage-1 seed/manifests differ from stage 2.")
    a, b = first["config"], second["config"]
    differences = [key for key in VQ_REQUIRED if a.get(key) != b.get(key)]
    differences += [key for key, default in VQ_DEFAULTS.items() if a.get(key, default) != b.get(key, default)]
    if a.get("decoder_fusion", a["fusion"]) != b.get("decoder_fusion", b["fusion"]):
        differences.append("decoder_fusion")
    if first["step"] != b["stage1_steps"] or first["step"] != a["stage1_steps"]:
        differences.append("stage1_training_budget")
    if differences:
        raise ValueError("Stage-1 dependency mismatch: " + ", ".join(differences))


def discover(roots):
    files, skipped = set(), []
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            skipped.append({"path": str(root), "reason": "directory_missing"})
            continue
        found = [path.resolve() for path in root.rglob("last.pt") if path.parent.name in ("stage1", "stage2")]
        if not found:
            skipped.append({"path": str(root), "reason": "no_training_checkpoints; evaluation images are not training runs"})
        files.update(found)
    # A two-stage model gets one row; stage-1-only runs stay visible with no
    # DDIM generation metrics. Nested/overlapping --runs are deduplicated.
    second = sorted(path for path in files if path.parent.name == "stage2")
    stage2_runs = {path.parent.parent for path in second}
    first_only = sorted(path for path in files if path.parent.name == "stage1" and path.parent.parent not in stage2_runs)
    return second + first_only, skipped


def first_stage_candidates(checkpoint, meta, roots):
    candidates = [checkpoint.parent.parent / "stage1/last.pt"]
    config = meta["config"]
    group = config.get("stage1_group")
    if group and checkpoint.parent.parent.name.startswith("seed_"):
        candidates.append(checkpoint.parents[3] / group / ("seed_{}".format(meta["seed"])) / "stage1/last.pt")
    for root in roots:
        if root.is_dir():
            candidates.extend(sorted(path for path in root.rglob("*.pt") if path.parent.name == "stage1"))
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def locate_first_stage(checkpoint, meta, roots):
    expected = meta.get("first_stage_hash")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Stage-2 checkpoint has no valid first_stage_hash.")
    for candidate in first_stage_candidates(checkpoint, meta, roots):
        if file_hash(candidate) == expected:
            return candidate
    raise FileNotFoundError("No exact stage-1 checkpoint matches first_stage_hash. Add its training root with --first-stage-search. Do not substitute another decoder.")


def base_summary(path, meta, first_meta, dataset):
    stage = meta["stage"]
    first = first_meta if stage == 2 else meta
    t1 = timing(first)
    t2 = timing(meta) if stage == 2 else None
    combined = (t1["completed_training_h"] + t2["completed_training_h"]
                if t2 and t1["completed_training_h"] is not None and t2["completed_training_h"] is not None else None)
    config = meta["config"]
    row = {
        "Experiment": config.get("experiment", path.parent.parent.name), "Seed": meta["seed"],
        "Stage": stage, "Status": "two_stage" if stage == 2 else "stage1_only_no_generation",
        "Checkpoint": str(path), "Factor": config["factor"], "Latent_side": config["image_size"] // config["factor"],
        "Codebook_K": config["codebook_size"], "Condition": config.get("condition"),
        "Decoder_fusion": config.get("decoder_fusion", config["fusion"]),
        "Decoder_gate": config.get("decoder_gate_mode", "mask"), "Perturbation": config.get("perturbation"),
        "Stage1_training_h": t1["completed_training_h"], "Stage2_training_h": t2["completed_training_h"] if t2 else None,
        "TIG_training_time_h": combined if dataset == "TIG" else None,
        "Laser_training_time_h": combined if dataset == "Laser" else None,
        "Stage1_hours_at_checkpoint": t1["hours_at_checkpoint"], "Stage2_hours_at_checkpoint": t2["hours_at_checkpoint"] if t2 else None,
        "Stage1_time_status": t1["status"], "Stage2_time_status": t2["status"] if t2 else "not_trained",
        "Params_total_M": (first["parameters"] + (meta["parameters"] if stage == 2 else 0)) / 1e6,
        "Params_VQ_M": first["parameters"] / 1e6,
        "Params_diffusion_M": meta["parameters"] / 1e6 if stage == 2 else None,
        "Stage1_trainable_params_M": first["trainable_parameters"] / 1e6,
        "Stage2_trainable_params_M": meta["trainable_parameters"] / 1e6 if stage == 2 else None,
        "FLOPs_denoiser_single_step_G": None, "FLOPs_decode_with_requantization_G": None,
        "FLOPs_full_generation_estimate_G": None,
    }
    return row


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_one(args):
    root, checkpoint = args.project_root, args.worker
    state = load_checkpoint(checkpoint)
    meta = metadata(state)
    if meta["config"].get("smoke_test") and not args.allow_smoke:
        raise ValueError("Smoke checkpoint cannot enter the formal table.")
    if checkpoint.parent.name != "stage{}".format(meta["stage"]):
        raise ValueError("Checkpoint stage disagrees with its folder.")
    for key in ("optimizer", "disc_optimizer", "discriminator", "scheduler"):
        state.pop(key, None)
    first_path, first_meta, first_state = checkpoint, meta, None
    if meta["stage"] == 2:
        print("Locating exact stage-1 checkpoint by SHA256...", flush=True)
        first_path = locate_first_stage(checkpoint, meta, args.first_stage_search)
        first_state = load_checkpoint(first_path)
        first_meta = metadata(first_state)
        verify_pair(first_meta, meta)
        if first_meta["config"].get("smoke_test") and not args.allow_smoke:
            raise ValueError("Stage-1 dependency is a smoke checkpoint.")
        for key in ("optimizer", "disc_optimizer", "discriminator", "scheduler"):
            first_state.pop(key, None)
    row = base_summary(checkpoint, meta, first_meta, args.dataset)
    report = {
        "format": "tig_heat_complexity_v1", "complete": False, "metadata_only": args.metadata_only,
        "checkpoint": str(checkpoint), "stage1_checkpoint": str(first_path),
        "checkpoint_sha256": file_hash(checkpoint),
        "stage1_sha256": meta["first_stage_hash"] if meta["stage"] == 2 else file_hash(checkpoint),
        "stage1_metadata": first_meta, "stage2_metadata": meta if meta["stage"] == 2 else None,
        "summary": row, "paper_table_scope_verified": False,
        "parameter_scope": "Full VQ + diffusion, including frozen CLIP and unused frozen modules; excludes training-only discriminator/LPIPS. Stage1-only rows contain VQ only. Metadata mode reads counts recorded during training.",
        "time_scope": timing(first_meta)["scope"],
        "cost_attribution": "Each two-stage model includes its dependent stage1 cost. Shared stage1 hashes must be deduplicated when totaling the entire study.",
        "FLOPs_scope": "Forward generation major convolution/matrix operations; 1 MAC=2 FLOPs. Includes codebook-search matrix products; excludes normalization/activation/elementwise/DDIM arithmetic and argmin comparisons. Not training FLOPs.",
    }
    if not args.metadata_only:
        import torch
        sys.path.insert(0, str(root))
        # Dense attention for arithmetic profiling only, before model construction.
        import ldm.modules.attention as attention
        import ldm.modules.diffusionmodules.model as vae_module
        if hasattr(attention, "XFORMERS_IS_AVAILBLE"):
            attention.XFORMERS_IS_AVAILBLE = False
        if hasattr(vae_module, "XFORMERS_IS_AVAILBLE"):
            vae_module.XFORMERS_IS_AVAILBLE = False
        from ldm.heat_source.models import HeatVQ, HeatDiffusion
        from ldm.heat_source.data import heat_template
        from tig_heat_profile_core import count_heat_models
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Selected CPU/CUDA device is unavailable.")
        if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("Use --precision fp32 for CPU or GPUs without BF16.")
        torch.manual_seed(2026)
        print("Loading VQ and diffusion modules on {}...".format(device), flush=True)
        vq = HeatVQ(first_meta["config"]).to(device)
        vq.load_state_dict((first_state if first_state is not None else state)["model"], strict=True)
        if sum(p.numel() for p in vq.parameters()) != first_meta["parameters"]:
            raise ValueError("VQ parameter count differs from training metadata.")
        if sum(p.numel() for p in vq.parameters() if p.requires_grad) != first_meta["trainable_parameters"]:
            raise ValueError("VQ trainable parameter count differs from training metadata.")
        if first_state is not None:
            del first_state
        diffusion = None
        config = meta["config"]
        if meta["stage"] == 2:
            if config["condition"] in ("text", "both") and args.clip_path is None:
                raise ValueError("Text branch requires --clip-path to the local ViT-B/32 used in training.")
            diffusion = HeatDiffusion(config, str(args.clip_path) if args.clip_path else None).to(device)
            diffusion.load_state_dict(state["model"], strict=True)
            if sum(p.numel() for p in diffusion.parameters()) != meta["parameters"]:
                raise ValueError("Diffusion parameter count differs from training metadata.")
            if sum(p.numel() for p in diffusion.parameters() if p.requires_grad) != meta["trainable_parameters"]:
                raise ValueError("Diffusion trainable parameter count differs from training metadata.")
            vq.requires_grad_(False)
        del state
        gc.collect()
        template = torch.from_numpy(heat_template(config["image_size"], "none", 2026,
                                   include_raw_prior=config.get("include_raw_prior", False)))[None].to(device)
        numeric = torch.zeros(1, 3, device=device)
        print("Counting batch=1 forward arithmetic (no image dataset or 50-step rendering)...", flush=True)
        counted = count_heat_models(vq, diffusion, config, template, numeric, ["TIG welding"],
                                    args.steps, meta.get("latent_scale", 1.), args.precision)
        if abs(counted["parameters"]["total_parameters_M"] - row["Params_total_M"]) > 1e-8:
            raise ValueError("Complete parameter count differs from recorded two-stage total.")
        report["profile"] = counted
        row.update(counted["summary"])
        report["profiling_input"] = "Synthetic fixed-shape template, zero standardized numeric inputs and a placeholder caption padded to 77 tokens; architecture arithmetic only, no image-quality evaluation."
        report.update(device=str(device), precision=args.precision, torch_version=torch.__version__)
    row["Metadata_only"] = args.metadata_only
    report["complete"] = True
    write_json(args.result, report)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--runs", nargs="+", type=Path)
    p.add_argument("--first-stage-search", nargs="+", type=Path, help="Additional roots for hash-matched shared stage1 files")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--clip-path", type=Path)
    p.add_argument("--dataset", choices=("TIG", "Laser"), default="TIG")
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--allow-smoke", action="store_true", help="Software tests only; not formal paper results")
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()

    def resolve(path):
        return (path.expanduser() if path.expanduser().is_absolute() else args.project_root / path.expanduser()).resolve()

    if args.steps < 2:
        p.error("--steps must be >= 2.")
    args.runs = [resolve(path) for path in args.runs or []]
    args.first_stage_search = list(dict.fromkeys(args.runs + [resolve(path) for path in args.first_stage_search or []]))
    if args.clip_path:
        args.clip_path = resolve(args.clip_path)
    if args.worker:
        args.worker, args.result = resolve(args.worker), resolve(args.result)
        run_one(args)
        return 0
    if not args.runs:
        p.error("Provide --runs with the training roots.")
    checkpoints, skipped = discover(args.runs)
    for item in skipped:
        print("Skipping {}: {}".format(item["path"], item["reason"]), flush=True)
    if not checkpoints:
        p.error("No stage1/last.pt or stage2/last.pt found under the supplied roots.")
    if args.out:
        out = resolve(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError("Use a new/empty report directory: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
    else:
        (args.project_root / "outputs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="heat_complexity_", dir=str(args.project_root / "outputs")))
    rows, records = [], []
    for index, checkpoint in enumerate(checkpoints):
        result = out / ("run_{:03d}.json".format(index + 1))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--project-root", str(args.project_root),
                   "--worker", str(checkpoint), "--result", str(result), "--dataset", args.dataset,
                   "--device", args.device, "--precision", args.precision, "--steps", str(args.steps),
                   "--first-stage-search", *map(str, args.first_stage_search)]
        if args.clip_path:
            command += ["--clip-path", str(args.clip_path)]
        if args.metadata_only:
            command.append("--metadata-only")
        if args.allow_smoke:
            command.append("--allow-smoke")
        print("[{}/{}] {}".format(index + 1, len(checkpoints), checkpoint), flush=True)
        process = subprocess.run(command, cwd=str(args.project_root))
        if process.returncode == 0 and result.is_file():
            report = json.loads(result.read_text(encoding="utf-8"))
            rows.append(report["summary"])
            records.append({"checkpoint": str(checkpoint), "status": "ok", "report": str(result)})
        else:
            rows.append({"Checkpoint": str(checkpoint), "Status": "failed", "Exit_code": process.returncode})
            records.append({"checkpoint": str(checkpoint), "status": "failed", "exit_code": process.returncode})
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "summary.json", {"runs": records, "skipped": skipped, "rows": rows,
                                      "time_note": "Shared stage1 training is included per model; do not sum rows without deduplicating stage1 hashes."})
    print("Results: " + str(out / "summary.csv"), flush=True)
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    sys.exit(main())
