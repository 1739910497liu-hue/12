"""Generate fixed/scalar gate samples and evaluate FID, IS and KID at three image counts.

Run from the TIGLDM project root on the Linux GPU server.
Only the largest requested set is generated; smaller sets reuse its images.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


METRICS_CODE = r'''
import json
import sys
from pathlib import Path
import torch_fidelity

p, device, batch, seed = sys.argv[1:]
p = Path(p)
info = json.loads((p / "sampling.json").read_text(encoding="utf-8"))
n = info["images"]
ids = json.loads((p / "evaluation_ids.json").read_text(encoding="utf-8"))
expected = {i + ".png" for i in ids}
if info["stage"] != 2 or len(ids) != n or len(expected) != n:
    raise ValueError("Invalid stage-2 sample metadata/count.")
for folder in ("real", "samples"):
    if {q.name for q in (p / folder).glob("*.png")} != expected:
        raise ValueError("Metric image files differ from selected evaluation IDs.")
s = torch_fidelity.calculate_metrics(
    input1=str(p / "samples"), input2=str(p / "real"),
    cuda=device.startswith("cuda"), batch_size=int(batch),
    fid=True, isc=True, kid=True, isc_splits=10,
    kid_subset_size=min(1000, n), kid_subsets=100,
    rng_seed=int(seed), verbose=True,
)
r = {
    "experiment": info["experiment"],
    "decoder_gate_mode": info["config"]["decoder_gate_mode"],
    "decoder_gate_init": info["config"]["decoder_gate_init"],
    "decoder_fusion": info["config"].get("decoder_fusion", info["config"]["fusion"]),
    "factor": info["config"]["factor"],
    "latent_height": info["config"]["image_size"] // info["config"]["factor"],
    "latent_width": info["config"]["image_size"] // info["config"]["factor"],
    "latent_channels": info["config"]["embed_dim"],
    "image_size": info["config"]["image_size"],
    "unet_mult": info["config"]["unet_mult"],
    "condition": info["config"]["condition"],
    "training_seed": info["training_seed"],
    "stage": info["stage"],
    "evaluation_type": "stage2_generation",
    "codebook_size": info["config"]["codebook_size"],
    "images": n,
    "FID": float(s["frechet_inception_distance"]),
    "IS_mean": float(s["inception_score_mean"]),
    "IS_split_std": float(s["inception_score_std"]),
    "KID_mean": float(s["kernel_inception_distance_mean"]),
    "KID_subset_std": float(s["kernel_inception_distance_std"]),
    "sample_seed": info["sample_seed"],
    "sampling_steps": info["sampling_steps"],
    "checkpoint_sha256": info["checkpoint_sha256"],
    "first_stage_hash": info["first_stage_hash"],
    "evaluation_ids_sha256": info["evaluation_ids_sha256"],
    "metric_seed": int(seed), "isc_splits": 10,
    "kid_subset_size": min(1000, n), "kid_subsets": 100,
    "output_directory": str(p),
}
(p / "metrics_is_fid_kid.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
print(json.dumps(r, indent=2), flush=True)
'''


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_ids(manifests, count):
    rows = [json.loads(line) for line in
            (manifests / "test.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    if len(rows) < count:
        raise ValueError("Test set has {} rows; {} requested.".format(len(rows), count))
    ids = [row["id"] for row in rows[:count]]
    if any(not isinstance(i, str) or not i or i in (".", "..") or
           any(c in i for c in ("/", "\\", "\x00")) for i in ids):
        raise ValueError("Test IDs must be safe, nonempty filenames.")
    if len(set(ids)) != count:
        raise ValueError("Duplicate IDs in selected test rows.")
    return ids


def check_images(directory, ids):
    expected = {i + ".png" for i in ids}
    for name in ("real", "samples"):
        actual = {p.name for p in (directory / name).glob("*.png")}
        if actual != expected:
            raise ValueError("Image IDs/count mismatch in " + str(directory / name))


def annotate(directory, ids, source=None):
    info = json.loads((directory / "sampling.json").read_text(encoding="utf-8"))
    info["images"] = len(ids)
    info["evaluation_ids_sha256"] = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    info["subset_selection"] = "first N rows of original test.jsonl, in manifest order"
    if source is not None:
        info["source_sampling_directory"] = str(source)
        info["mean_seconds_per_image"] = None
        info["steady_seconds_per_image"] = None
    write_json(directory / "sampling.json", info)
    write_json(directory / "evaluation_ids.json", ids)
    return info


def make_subset(source, target, ids):
    target.mkdir(parents=True, exist_ok=False)
    for name in ("real", "samples"):
        (target / name).mkdir()
        for image_id in ids:
            src, dst = source / name / (image_id + ".png"), target / name / (image_id + ".png")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    shutil.copy2(source / "sampling.json", target / "sampling.json")
    check_images(target, ids)
    annotate(target, ids, source=source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs", type=Path, default=Path("outputs/ss304_gate_ablation/full"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/ss304"))
    parser.add_argument("--manifests", type=Path, default=Path("datasets/ss304_manifests"))
    parser.add_argument("--clip-path", type=Path, default=Path("model/clip/clip-vit-base-patch32"))
    parser.add_argument("--experiments", nargs="+", choices=["g1_fixed", "g2_scalar"], default=["g1_fixed", "g2_scalar"])
    parser.add_argument("--counts", type=int, nargs="+", default=[3500, 4000, 4172])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--checkpoint", choices=["last", "best"], default="best")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    counts = sorted(set(args.counts))
    if counts[0] < 10 or args.batch_size < 1:
        parser.error("Counts must be >= 10 and batch size positive.")
    for name in ("runs", "data_root", "manifests", "clip_path"):
        value = getattr(args, name)
        setattr(args, name, (value if value.is_absolute() else root / value).resolve())
    entry = root / "scripts/heat_source.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    for path in (args.data_root, args.clip_path):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "numeric_stats.json"):
        if not (args.manifests / name).is_file():
            raise FileNotFoundError(args.manifests / name)
    ids = test_ids(args.manifests, max(counts))
    jobs = []
    for experiment in dict.fromkeys(args.experiments):
        run = args.runs / experiment / "seed_{}".format(args.seed)
        checkpoint = run / "stage2" / (args.checkpoint + ".pt")
        first_stage = run / "stage1/last.pt"
        for path in (checkpoint, first_stage):
            if not path.is_file():
                raise FileNotFoundError(path)
        jobs.append((experiment, checkpoint, first_stage))
        print("{}: stage2={}, stage1={}".format(experiment, checkpoint, first_stage), flush=True)
    if args.check_only:
        print("Path/manifest checks passed. Checkpoint contents were not loaded.", flush=True)
        return
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    subprocess.run([sys.executable, "-c", "import torch_fidelity"],
                   cwd=str(root), env=env, check=True)
    if args.out is None:
        (root / "outputs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="ss304_gates_3500_4000_4172_", dir=str(root / "outputs")))
    else:
        out = (args.out if args.out.is_absolute() else root / args.out).resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a new/empty output directory: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
    print("Evaluation output: " + str(out), flush=True)
    results = []
    reference_protocol = None
    for experiment, checkpoint, first_stage in jobs:
        parent = out / experiment / "seed_{}".format(args.seed)
        generated = parent / "n{}".format(max(counts))
        weight_stats = [(p.stat().st_size, p.stat().st_mtime_ns)
                        for p in (checkpoint, first_stage)]
        subprocess.run([
            sys.executable, str(entry), "sample", "--checkpoint", str(checkpoint),
            "--first-stage", str(first_stage), "--data-root", str(args.data_root),
            "--manifests", str(args.manifests), "--clip-path", str(args.clip_path),
            "--device", args.device, "--sample-seed", str(args.sample_seed),
            "--limit", str(max(counts)), "--out", str(generated),
        ], cwd=str(root), env=env, check=True)
        if weight_stats != [(p.stat().st_size, p.stat().st_mtime_ns)
                            for p in (checkpoint, first_stage)]:
            raise RuntimeError("Checkpoint changed during sampling; use finished training runs.")
        info = json.loads((generated / "sampling.json").read_text(encoding="utf-8"))
        if (info["images"] != max(counts) or info["stage"] != 2 or
                info["training_seed"] != args.seed or info["sample_seed"] != args.sample_seed or
                info["experiment"] != experiment):
            raise ValueError("Sampling metadata does not match requested experiment/count.")
        config = info["config"]
        expected_gate = {"g1_fixed": "fixed", "g2_scalar": "scalar"}[experiment]
        if (config.get("condition") != "both" or config.get("image_size") != 256
                or config.get("factor") != 8 or config.get("embed_dim") != 4
                or config.get("codebook_size") != 16384 or config.get("fusion") != "gated"
                or config.get("stage1_group") != experiment or config.get("experiment") != experiment
                or config.get("decoder_gate_mode") != expected_gate
                or config.get("decoder_gate_init") != 0.5
                or config.get("decoder_fusion", config["fusion"]) != "gated"):
            raise ValueError("Checkpoint does not match the expected fixed/scalar residual decoder gate experiment.")
        shared_config = {key: value for key, value in config.items()
                         if key not in ("experiment", "stage1_group", "decoder_gate_mode", "workers")}
        shared_config.setdefault("decoder_fusion", config["fusion"])
        protocol = dict(config=shared_config, sampling_steps=info["sampling_steps"],
                        manifest_hashes=info["manifest_hashes"])
        if reference_protocol is not None and reference_protocol != protocol:
            raise ValueError("Gate checkpoints differ in settings beyond the intended decoder gate choice.")
        reference_protocol = protocol
        check_images(generated, ids)
        annotate(generated, ids)
        for n in counts:
            directory = parent / "n{}".format(n)
            if n != max(counts):
                make_subset(generated, directory, ids[:n])
            print("Evaluating gate={}, N={}".format(experiment, n), flush=True)
            subprocess.run([sys.executable, "-c", METRICS_CODE, str(directory), args.device,
                            str(args.batch_size), str(args.sample_seed)],
                           cwd=str(root), env=env, check=True)
            results.append(json.loads((directory / "metrics_is_fid_kid.json").read_text(encoding="utf-8")))
            write_json(out / "summary.json", results)
            with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    print("Completed: " + str(out / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
