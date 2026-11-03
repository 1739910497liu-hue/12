"""Evaluate existing perturbation samples at matched image counts.

This script never runs inference or loads training checkpoints.
It creates paired image subsets, then calculates FID, IS and KID.
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
    raise ValueError("Metric metadata must identify stage-2 samples and exact image count.")
for folder in ("real", "samples"):
    if {q.name for q in (p / folder).glob("*.png")} != expected:
        raise ValueError("Image set differs from recorded evaluation IDs.")
s = torch_fidelity.calculate_metrics(
    input1=str(p / "samples"), input2=str(p / "real"),
    cuda=device.startswith("cuda"), batch_size=int(batch),
    fid=True, isc=True, kid=True, isc_splits=10,
    kid_subset_size=min(1000, n), kid_subsets=100,
    rng_seed=int(seed), verbose=True,
)
r = {
    "experiment": info["experiment"],
    "condition": info["config"]["condition"],
    "training_perturbation": info["config"]["perturbation"],
    "source_sampling_directory": info["source_sampling_directory"],
    "test_manifest_sha256": info["manifest_hashes"]["test.jsonl"],
    "stage": info["stage"],
    "training_seed": info["training_seed"],
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


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(directory, experiment, ids, test_hash):
    metadata = directory / "sampling.json"
    if not metadata.is_file():
        raise FileNotFoundError("Completed sampling.json missing: " + str(metadata))
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info["stage"] != 2 or info["experiment"] != experiment:
        raise ValueError("Expected stage-2 " + experiment + " samples in " + str(directory))
    config = info["config"]
    if (config.get("condition") != "both" or config.get("experiment") != experiment
            or config.get("stage1_group") != experiment):
        raise ValueError("Expected the both-condition perturbation experiment with its own stage1 group: "
                         + str(metadata))
    if config["perturbation"] != experiment[len("perturb_"):]:
        raise ValueError("Wrong perturbation configuration in " + str(metadata))
    if info["manifest_hashes"].get("test.jsonl") != test_hash:
        raise ValueError("Original test.jsonl differs from sampling metadata: " + str(directory))
    if info["images"] < len(ids):
        raise ValueError("{} has {} images; {} requested.".format(directory, info["images"], len(ids)))
    real = {p.name for p in (directory / "real").glob("*.png")}
    fake = {p.name for p in (directory / "samples").glob("*.png")}
    if real != fake or len(real) != info["images"]:
        raise ValueError("Source image count/paired IDs disagree with sampling.json: " + str(directory))
    if not {i + ".png" for i in ids}.issubset(real):
        raise ValueError("Source does not contain the requested first test-manifest IDs: " + str(directory))
    for key in ("checkpoint_sha256", "first_stage_hash", "sample_seed", "sampling_steps", "training_seed"):
        if key not in info:
            raise ValueError("Missing sampling metadata: " + key)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--experiments", choices=["perturb_small", "perturb_large"], nargs="+",
                        default=["perturb_small", "perturb_large"])
    parser.add_argument("--manifests", type=Path, default=Path("datasets/ss304_manifests"))
    parser.add_argument("--counts", type=int, nargs="+", default=[3500, 4000])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--metric-seed", type=int, default=2026)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    counts = sorted(set(args.counts))
    if counts[0] < 10 or args.batch_size < 1:
        parser.error("Counts must be >= 10 and batch size positive.")
    source_root, manifests = args.samples_root.resolve(), args.manifests.resolve()
    ids = test_ids(manifests, max(counts))
    test_hash = file_hash(manifests / "test.jsonl")
    jobs = []
    reference_protocol = None
    for experiment in dict.fromkeys(args.experiments):
        directory = source_root / experiment
        if source_root.name == experiment and (source_root / "sampling.json").is_file():
            directory = source_root
        info = validate_source(directory, experiment, ids, test_hash)
        # These are the intended treatment differences, including separately trained VQ weights.
        config = {key: value for key, value in info["config"].items()
                  if key not in ("experiment", "stage1_group", "perturbation")}
        protocol = dict(config=config, manifest_hashes=info["manifest_hashes"],
                        training_seed=info["training_seed"], sample_seed=info["sample_seed"],
                        sampling_steps=info["sampling_steps"])
        if reference_protocol is not None and reference_protocol != protocol:
            raise ValueError("Experiments differ in settings other than the intended training perturbation.")
        reference_protocol = protocol
        jobs.append((experiment, directory, file_hash(directory / "sampling.json")))
        print("Existing {}: {} paired images; selected counts {}".format(
            experiment, info["images"], counts), flush=True)
    if args.check_only:
        print("Metadata, test-manifest hash and image-ID checks passed; no inference/metrics run.", flush=True)
        return
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    subprocess.run([sys.executable, "-c", "import torch_fidelity"], env=env, check=True)
    if args.out is None:
        outputs = Path.cwd() / "outputs"
        outputs.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="ss304_perturbation_metrics_3500_4000_", dir=str(outputs)))
    else:
        out = args.out.resolve()
        for _, directory, _ in jobs:
            if out == directory or directory in out.parents:
                raise ValueError("Output must be separate from the original sample directory.")
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Use a new/empty output directory: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
    print("Metrics output: " + str(out), flush=True)
    results = []
    for experiment, directory, metadata_hash in jobs:
        for n in counts:
            if file_hash(directory / "sampling.json") != metadata_hash:
                raise RuntimeError("Source metadata changed during evaluation: " + str(directory))
            target = out / experiment / "n{}".format(n)
            make_subset(directory, target, ids[:n])
            print("Calculating FID/IS/KID: {}, N={}".format(experiment, n), flush=True)
            subprocess.run([sys.executable, "-c", METRICS_CODE, str(target), args.device,
                            str(args.batch_size), str(args.metric_seed)], env=env, check=True)
            results.append(json.loads((target / "metrics_is_fid_kid.json").read_text(encoding="utf-8")))
            write_json(out / "summary.json", results)
            with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    print("Completed: " + str(out / "summary.csv"), flush=True)


if __name__ == "__main__":
    main()
