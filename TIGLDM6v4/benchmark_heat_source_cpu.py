"""CPU FP32 timing of real TIGLDM DDIM generation, one image per repetition.

Each experiment runs in an isolated worker. Requires completed stage2 weights
and their exact stage1 dependency. Stage1-only runs are explicitly skipped.
No checkpoint/data changes, training, image saving or GPU measurements.
"""
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from profile_heat_source_runs import (
    discover, file_hash, load_checkpoint, locate_first_stage, metadata, verify_pair,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure_cpu(generate, validate, warmup, repeats, clock=time.perf_counter, before=None, after=None):
    if warmup < 1 or repeats < 2:
        raise ValueError("Require warmup >= 1 and repeats >= 2.")
    for index in range(warmup):
        if before:
            before(index, True)
        output = generate(index)
        validate(output)
        del output
    elapsed = []
    for index in range(repeats):
        if before:
            before(index, False)
        start = clock()
        output = generate(index)
        seconds = clock() - start  # CPU tensor operations complete synchronously.
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid CPU measurement interval.")
        validate(output)
        del output
        elapsed.append(seconds)
        if after:
            after(index, seconds)
    return {
        "seconds_per_image": statistics.mean(elapsed),
        "median_seconds_per_image": statistics.median(elapsed),
        "p95_seconds_per_image": percentile(elapsed, .95),
        "std_seconds_per_image": statistics.stdev(elapsed),
        "measured_seconds": elapsed,
    }


def cpu_description():
    result = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return result


def eligible_checkpoints(roots):
    found, skipped = discover(roots)
    eligible = []
    for path in found:
        if path.parent.name == "stage1":
            skipped.append({"path": str(path), "reason": "stage1_only: cannot benchmark DDIM generation without stage2"})
        else:
            eligible.append(path)
    return eligible, skipped


def run_worker(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    report = {
        "format": "tig_heat_cpu_runtime_v1", "complete": False,
        "checkpoint": str(args.worker), "progress": "reading_checkpoint", "device": "cpu",
        "CPU": cpu_description(), "platform": platform.platform(),
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "precision": "fp32",
        "sampler": "DDIM", "steps": args.steps, "batch_size": 1,
        "warmup": args.warmup, "repeats": args.repeats, "sample_seed": args.sample_seed,
        "inference_peak_allocated_GiB": None, "gpu_memory_status": "not_applicable_cpu",
        "timing_scope": "CPU initial noise, tokenization/CLIP/numeric encoding, full native DDIM and requantization+VQ decoding. Excludes model loading, dataset I/O, CPU heat-template preprocessing, output finite checks and image saving.",
    }

    def progress(value):
        report["progress"] = value
        write_json(args.result, report)
        print(value, flush=True)

    progress("Reading stage2 checkpoint on CPU...")
    state = load_checkpoint(args.worker)
    meta = metadata(state)
    config = meta["config"]
    if meta["stage"] != 2:
        raise ValueError("DDIM timing needs a stage2 checkpoint.")
    if not args.allow_smoke and (config.get("smoke_test") or meta["step"] != config["stage2_steps"]):
        raise ValueError("Use a completed formal stage2 checkpoint. --allow-smoke is only for software tests.")
    if not 2 <= args.steps <= config["timesteps"]:
        raise ValueError("DDIM steps outside the saved model schedule.")
    scale = meta["latent_scale"]
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid latent scale in checkpoint.")
    # Only model weights and small provenance survive the checkpoint container.
    diffusion_weights = state["model"]
    del state
    gc.collect()
    progress("Verifying exact stage1 dependency and test manifests...")
    first_path = locate_first_stage(args.worker, meta, args.first_stage_search)
    first_state = load_checkpoint(first_path)
    first_meta = metadata(first_state)
    verify_pair(first_meta, meta)
    if first_meta["config"].get("smoke_test") and not args.allow_smoke:
        raise ValueError("Stage1 dependency is a smoke checkpoint.")
    first_weights = first_state["model"]
    del first_state
    gc.collect()
    actual_manifest = {name: file_hash(args.manifests / name)
                       for name in ("train.jsonl", "val.jsonl", "test.jsonl", "numeric_stats.json")}
    if actual_manifest != meta["manifest_hashes"]:
        raise ValueError("Manifests differ from the training protocol.")
    sys.path.insert(0, str(args.project_root))
    from ldm.heat_source.data import HeatDataset, stable_seed
    from tig_heat_cpu_loading import load_heat_model
    if config["condition"] in ("text", "both") and args.clip_path is None:
        raise ValueError("Text conditions require --clip-path to the local CLIP ViT-B/32 directory.")
    stats = json.loads((args.manifests / "numeric_stats.json").read_text(encoding="utf-8"))
    data = HeatDataset(args.manifests / "test.jsonl", args.data_root, config["image_size"],
                       config["perturbation"], meta["seed"], False, config["condition"], stats,
                       config.get("caption_field", "caption"), config.get("prior_source", "analytic"),
                       include_raw_prior=config.get("include_raw_prior", False))
    needed = max(args.warmup, args.repeats)
    if len(data) < needed:
        raise ValueError("The test split has fewer than warmup/repeats distinct conditions.")
    conditions = []
    for index in range(needed):
        item = data[index]
        conditions.append({"template": item["template"][None], "numeric": item["numeric"][None],
                           "captions": [item["caption"]], "id": item["id"]})
    del item, data
    report.update(experiment=config["experiment"], training_seed=meta["seed"],
                  checkpoint_step=meta["step"], checkpoint_sha256=file_hash(args.worker),
                  first_stage=str(first_path), first_stage_hash=meta["first_stage_hash"],
                  manifest_hashes=actual_manifest, training_config=config,
                  training_precision=config["precision"], condition_ids=[row["id"] for row in conditions])
    progress("Loading VQ with strict checkpoint weights...")
    vq, vq_loading = load_heat_model("stage1", first_meta["config"], first_weights)
    del first_weights
    vq.eval().requires_grad_(False)
    if sum(p.numel() for p in vq.parameters()) != first_meta["parameters"]:
        raise ValueError("VQ parameter count differs from training metadata.")
    progress("Loading diffusion and condition encoders with strict checkpoint weights...")
    diffusion, diffusion_loading = load_heat_model("stage2", config, diffusion_weights,
                                                  str(args.clip_path) if args.clip_path else None)
    del diffusion_weights
    gc.collect()
    diffusion.eval()
    if sum(p.numel() for p in diffusion.parameters()) != meta["parameters"]:
        raise ValueError("Diffusion parameter count differs from training metadata.")
    report["loading"] = {"vq": vq_loading, "diffusion": diffusion_loading}
    clip = getattr(diffusion.conditioner, "clip", None)
    report["clip_attention_classes"] = sorted({type(module).__name__ for module in clip.modules()
                                               if "Attention" in type(module).__name__}) if clip is not None else []
    report["resident_pipeline_parameters"] = first_meta["parameters"] + meta["parameters"]

    def generate(index):
        item = conditions[index]
        rng = torch.Generator(device="cpu").manual_seed(stable_seed(args.sample_seed, 0, item["id"]))
        # Explicit FP32 CPU inference; saved training precision is unchanged.
        latent = diffusion.sample(item["template"], item["numeric"], item["captions"], args.steps, rng)
        return vq.decode(latent / scale, item["template"], requantize=True)

    def validate(output):
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Non-finite image; timing is invalid.")

    def before(index, warm):
        progress("{} image {}/{} (DDIM {})".format("Warmup" if warm else "Timing", index + 1,
                                                   args.warmup if warm else args.repeats, args.steps))

    def after(index, seconds):
        print(json.dumps({"repeat": index + 1, "seconds_per_image": seconds}), flush=True)

    with torch.no_grad():
        measured = measure_cpu(generate, validate, args.warmup, args.repeats, before=before, after=after)
    report.update(measured, complete=True, progress="complete")
    row = {
        "Experiment": config["experiment"], "Seed": meta["seed"], "Device": "cpu", "CPU": report["CPU"],
        "Sampler": "DDIM", "Steps": args.steps, "Precision": "fp32", "Batch_size": 1,
        "Threads": args.threads, "Warmup": args.warmup, "Repeats": args.repeats,
        "Seconds_per_image": measured["seconds_per_image"],
        "Median_seconds_per_image": measured["median_seconds_per_image"],
        "P95_seconds_per_image": measured["p95_seconds_per_image"],
        "Inference_peak_VRAM_GiB": None, "VRAM_status": "not_applicable_cpu",
        "Factor": config["factor"], "Codebook_K": config["codebook_size"], "Image_size": config["image_size"],
        "Condition": config["condition"], "Report": str(args.result), "Status": "complete",
    }
    report["summary"] = row
    write_json(args.result, report)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def exit_detail(code):
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = "signal_{}".format(-code)
        return "{} ({}); externally terminated; check system/cgroup logs before diagnosing OOM".format(code, name)
    return "exit_code={}".format(code)


def save_batch(out, rows, skipped, failures, pending):
    write_json(out / "summary.json", {"complete": not failures and not pending,
                                      "results": rows, "skipped": skipped, "failures": failures,
                                      "not_attempted": list(map(str, pending))})
    keys = list(dict.fromkeys(key for row in rows for key in row)) or ["Experiment", "Seconds_per_image", "Status"]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--runs", nargs="+", type=Path)
    p.add_argument("--first-stage-search", nargs="+", type=Path)
    p.add_argument("--data-root", type=Path, default=Path("datasets/ss304"))
    p.add_argument("--manifests", type=Path, default=Path("datasets/ss304_manifests"))
    p.add_argument("--clip-path", type=Path)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--precision", choices=("fp32",), default="fp32")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--allow-smoke", action="store_true", help="Software tests only; excluded from formal measurements")
    p.add_argument("--out", type=Path)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.steps < 2 or args.warmup < 1 or args.repeats < 2 or args.threads < 1:
        p.error("Require steps>=2, warmup>=1, repeats>=2, threads>=1.")
    args.project_root = args.project_root.expanduser().resolve()

    def resolve(path):
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (args.project_root / path).resolve()

    args.runs = [resolve(path) for path in args.runs or []]
    args.first_stage_search = list(dict.fromkeys(args.runs + [resolve(path) for path in args.first_stage_search or []]))
    args.data_root, args.manifests = resolve(args.data_root), resolve(args.manifests)
    if args.clip_path:
        args.clip_path = resolve(args.clip_path)
    if args.worker:
        args.worker, args.result = resolve(args.worker), resolve(args.result)
        run_worker(args)
        return 0
    if not args.runs:
        p.error("Provide --runs training roots.")
    checkpoints, skipped = eligible_checkpoints(args.runs)
    for item in skipped:
        print("Skipping {}: {}".format(item["path"], item["reason"]), flush=True)
    if args.check_only:
        print(json.dumps({"eligible_stage2": list(map(str, checkpoints)), "skipped": skipped}, indent=2), flush=True)
        return 0
    if not checkpoints:
        p.error("No stage2 checkpoints available; stage1 cannot generate through DDIM.")
    if args.out:
        out = resolve(args.out)
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError("Use a new/empty output directory: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
    else:
        (args.project_root / "outputs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="cpu_ddim_runtime_", dir=str(args.project_root / "outputs")))
    rows, failures = [], []
    pending = list(checkpoints)
    save_batch(out, rows, skipped, failures, pending)
    for index, checkpoint in enumerate(checkpoints):
        pending.pop(0)
        target = out / ("run_{:03d}".format(index + 1))
        target.mkdir()
        result = target / "runtime.json"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--project-root", str(args.project_root),
                   "--worker", str(checkpoint), "--result", str(result), "--data-root", str(args.data_root),
                   "--manifests", str(args.manifests), "--steps", str(args.steps), "--warmup", str(args.warmup),
                   "--repeats", str(args.repeats), "--threads", str(args.threads), "--sample-seed", str(args.sample_seed),
                   "--first-stage-search", *map(str, args.first_stage_search)]
        if args.clip_path:
            command += ["--clip-path", str(args.clip_path)]
        if args.allow_smoke:
            command.append("--allow-smoke")
        print("[{}/{}] CPU benchmark: {}".format(index + 1, len(checkpoints), checkpoint), flush=True)
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        finished = subprocess.run(command, cwd=str(args.project_root), env=environment)
        report = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        if finished.returncode == 0 and report.get("complete") is True:
            rows.append(report["summary"])
        else:
            reason = exit_detail(finished.returncode)
            failures.append({"checkpoint": str(checkpoint), "reason": reason, "exit_code": finished.returncode,
                             "last_progress": report.get("progress"), "report": str(result)})
            print("FAILED: {} | last progress: {}".format(reason, report.get("progress")), flush=True)
        save_batch(out, rows, skipped, failures, pending)
        if finished.returncode < 0:
            print("Stopping remaining jobs after worker signal termination.", flush=True)
            break
    print("Completed models: {}; failures: {}; not attempted: {}".format(len(rows), len(failures), len(pending)), flush=True)
    print("Results: " + str(out / "summary.csv"), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
