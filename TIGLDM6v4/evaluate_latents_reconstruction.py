"""Paired stage-1 reconstruction metrics for matched latent-resolution runs.

CPU FP32 by default. Uses the same first N test IDs for every factor; original
manifest hashes are verified and source checkpoints/manifests are never edited.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from profile_heat_source_runs import file_hash, load_checkpoint, metadata


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def selected_rows(manifests, limit):
    rows = [json.loads(line) for line in (manifests / "test.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Test manifest must have nonempty, unique string IDs.")
    if limit is not None and (limit < 1 or limit > len(rows)):
        raise ValueError("Requested {} images, but the test manifest has {}.".format(limit, len(rows)))
    selected = rows if limit is None else rows[:limit]
    if any(value in (".", "..") or any(char in value for char in ("/", "\\", "\x00", ":")) for value in [row["id"] for row in selected]):
        raise ValueError("Selected IDs must be safe image filenames.")
    return selected


def check_reference_paths(rows, data_root):
    root = data_root.resolve()
    missing = []
    for row in rows:
        path = (root / row["image"]).resolve()
        if root not in path.parents or not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("{} selected reference images missing or outside --data-root. "
                                "Reconstruction requires real target pixels; correct --data-root or restore these files.\n{}".format(
                                    len(missing), "\n".join(missing[:5])))


def find_jobs(root, factors, seed):
    if not root.is_dir():
        raise FileNotFoundError("Training root does not exist: " + str(root))
    candidates = list(root.rglob("stage1/last.pt"))
    jobs = []
    for factor in sorted(set(factors)):
        name = "paper_gate_f{}".format(factor)
        matches = [path for path in candidates if path.parents[1].name == "seed_{}".format(seed)
                   and path.parents[2].name == name]
        if len(matches) != 1:
            raise ValueError("Expected exactly one {}/seed_{}/stage1/last.pt under {}; found {}.".format(name, seed, root, len(matches)))
        jobs.append((factor, matches[0].resolve()))
    return jobs


def worker(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu.")
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; use --precision fp32.")
    rows = selected_rows(args.manifests, args.limit)
    check_reference_paths(rows, args.data_root)
    hashes = {name: file_hash(args.manifests / name) for name in
              ("train.jsonl", "val.jsonl", "test.jsonl", "numeric_stats.json")}
    print("Loading stage1 checkpoint: " + str(args.worker), flush=True)
    state = load_checkpoint(args.worker)
    meta = metadata(state)
    config = meta["config"]
    if meta["stage"] != 1 or meta["seed"] != args.seed or config["factor"] != args.expected_factor:
        raise ValueError("Checkpoint stage/seed/factor differs from the requested experiment.")
    if not args.allow_smoke and (config.get("smoke_test") or meta["step"] != config["stage1_steps"]):
        raise ValueError("Use a completed stage1 checkpoint; --allow-smoke is for software tests only.")
    if meta["manifest_hashes"] != hashes:
        raise ValueError("Evaluation manifests differ from this training checkpoint.")
    weights = state["model"]
    del state
    gc.collect()
    target = args.result.parent
    ids = [row["id"] for row in rows]
    subset_path = target / "test_subset.jsonl"
    subset_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    sys.path.insert(0, str(args.project_root))
    from ldm.heat_source.data import HeatDataset
    from tig_heat_cpu_loading import load_heat_model
    from tig_heat_reconstruction_metrics import SSIM, load_lpips, paired_scores, quantize_image
    stats = json.loads((args.manifests / "numeric_stats.json").read_text(encoding="utf-8"))
    data = HeatDataset(subset_path, args.data_root, config["image_size"], config["perturbation"],
                       meta["seed"], False, "none", stats, config.get("caption_field", "caption"),
                       config.get("prior_source", "analytic"), include_raw_prior=config.get("include_raw_prior", False))
    print("Preparing SSIM and {}...".format("MAE/PSNR" if args.skip_lpips else "pretrained VGG LPIPS"), flush=True)
    ssim = SSIM(window_size=11, size_average=False).to(device).eval()
    perceptual = None if args.skip_lpips else load_lpips(device, args.lpips_backbone_weights)
    print("Loading strict VQ weights; reconstruction precision=" + args.precision, flush=True)
    model, loading = load_heat_model("stage1", config, weights)
    del weights
    gc.collect()
    model.eval().requires_grad_(False).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != meta["parameters"]:
        raise ValueError("Parameter count differs from saved training metadata.")
    report = {"complete": False, "evaluation_type": "stage1_paired_reconstruction",
              "checkpoint": str(args.worker), "checkpoint_sha256": file_hash(args.worker),
              "training_config": config, "training_precision": config["precision"],
              "device": str(device), "precision": args.precision, "threads": args.threads,
              "manifest_hashes": hashes, "evaluation_ids": ids,
              "evaluation_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
              "subset_selection": "first N original test.jsonl rows, same order for every model",
              "image_domain": "native 8-bit RGB PNG quantization; MAE/PSNR/SSIM inputs [0,1]",
              "ssim_definition": "project SSIM: Gaussian11, sigma1.5, zero padding5, RGB",
              "psnr_definition": "mean per-image PSNR; data_range=1; MSE floor 1e-12 (120 dB cap)",
              "lpips_backbone": None if args.skip_lpips else "vgg", "lpips_status": "skipped_explicitly" if args.skip_lpips else "enabled",
              "lpips_backbone_weights_sha256": file_hash(args.lpips_backbone_weights) if args.lpips_backbone_weights else None,
              "loading": loading, "images_saved": args.save_images}
    write_json(args.result, report)
    metric_names = ["mae", "psnr", "ssim"] + ([] if args.skip_lpips else ["lpips"])
    totals = {name: 0. for name in metric_names}
    if args.save_images:
        from PIL import Image
        for folder in ("real", "samples"):
            (target / folder).mkdir()
    with (target / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id"] + metric_names)
        writer.writeheader()
        with torch.no_grad():
            for index in range(len(data)):
                item = data[index]
                original = item["image"][None].to(device)
                template = item["template"][None].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and args.precision == "bf16"):
                    reconstruction = model(original, template)[0]
                real_u8, fake_u8 = quantize_image(original), quantize_image(reconstruction)
                scores = paired_scores(real_u8.to(device), fake_u8.to(device), ssim, perceptual)[0]
                writer.writerow(dict(scores, id=item["id"]))
                for name in metric_names:
                    totals[name] += scores[name]
                if args.save_images:
                    for folder, tensor in (("real", real_u8), ("samples", fake_u8)):
                        Image.fromarray(tensor[0].permute(1, 2, 0).cpu().numpy()).save(target / folder / (item["id"] + ".png"))
                del original, template, reconstruction, real_u8, fake_u8
                if (index + 1) % 10 == 0 or index + 1 == len(data):
                    stream.flush()
                    print("Reconstruction metrics: {}/{}".format(index + 1, len(data)), flush=True)
    summary = {"Experiment": config["experiment"], "Seed": meta["seed"], "Stage": 1,
               "Factor": config["factor"], "Latent_side": config["image_size"] // config["factor"],
               "Latent_channels": config["embed_dim"], "Codebook_K": config["codebook_size"],
               "Image_size": config["image_size"], "Images": len(data), "Device": str(device), "Precision": args.precision,
               "MAE": totals["mae"] / len(data), "PSNR": totals["psnr"] / len(data), "SSIM": totals["ssim"] / len(data),
               "LPIPS": totals["lpips"] / len(data) if "lpips" in totals else None,
               "LPIPS_status": report["lpips_status"], "Report": str(args.result), "Status": "complete"}
    report.update(complete=True, summary=summary)
    write_json(args.result, report)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs", type=Path, default=Path("outputs/ss304_paper_gate_latents"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets/ss304"))
    parser.add_argument("--manifests", type=Path, default=Path("datasets/ss304_manifests"))
    parser.add_argument("--factors", type=int, nargs="+", choices=(4, 8, 16, 32), default=[4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Evaluate the same first N test images for every model.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lpips-backbone-weights", type=Path, help="Optional local torchvision VGG16 .pth file.")
    parser.add_argument("--skip-lpips", action="store_true", help="Explicitly report only MAE/PSNR/SSIM; LPIPS remains empty.")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check selected original image paths and checkpoint paths only.")
    parser.add_argument("--allow-smoke", action="store_true", help="Software tests only.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-factor", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and limit must be positive.")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("CPU reconstruction requires --precision fp32.")
    args.project_root = args.project_root.expanduser().resolve()
    for name in ("runs", "data_root", "manifests", "out", "worker", "result", "lpips_backbone_weights"):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser()
            setattr(args, name, (value if value.is_absolute() else args.project_root / value).resolve())
    if args.worker:
        worker(args)
        return 0
    if not (args.project_root / "ldm/heat_source/models.py").is_file():
        raise FileNotFoundError("Use --project-root for the TIGLDM project.")
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "numeric_stats.json"):
        if not (args.manifests / name).is_file():
            raise FileNotFoundError(args.manifests / name)
    rows = selected_rows(args.manifests, args.limit)
    check_reference_paths(rows, args.data_root)
    jobs = find_jobs(args.runs, args.factors, args.seed)
    for factor, checkpoint in jobs:
        print("f={}: {}".format(factor, checkpoint), flush=True)
    print("Matched test subset: {} images per model.".format(len(rows)), flush=True)
    if args.check_only:
        print("Paths verified. Checkpoint contents and LPIPS weights have not been loaded.", flush=True)
        return 0
    if args.out:
        out = args.out
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError("Use a new/empty output directory: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
    else:
        (args.project_root / "outputs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="latents_reconstruction_{}img_".format(len(rows)), dir=str(args.project_root / "outputs")))
    summaries, failures = [], []
    for factor, checkpoint in jobs:
        folder = out / "paper_gate_f{}".format(factor)
        folder.mkdir()
        result = folder / "metrics_reconstruction.json"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--project-root", str(args.project_root),
                   "--worker", str(checkpoint), "--expected-factor", str(factor), "--result", str(result),
                   "--data-root", str(args.data_root), "--manifests", str(args.manifests), "--seed", str(args.seed),
                   "--limit", str(len(rows)), "--device", args.device, "--precision", args.precision,
                   "--threads", str(args.threads)]
        for flag in ("skip_lpips", "save_images", "allow_smoke"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        if args.lpips_backbone_weights:
            command += ["--lpips-backbone-weights", str(args.lpips_backbone_weights)]
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        completed = subprocess.run(command, cwd=str(args.project_root), env=environment)
        report = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        if completed.returncode == 0 and report.get("complete"):
            summaries.append(report["summary"])
        else:
            failures.append({"factor": factor, "checkpoint": str(checkpoint), "exit_code": completed.returncode})
        with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]) if summaries else ["Experiment", "Status"])
            writer.writeheader()
            writer.writerows(summaries)
        write_json(out / "summary.json", {"complete": not failures and len(summaries) == len(jobs),
                                         "rows": summaries, "failures": failures,
                                         "requested_factors": args.factors, "images_per_model": len(rows)})
        if completed.returncode != 0:
            print("Stopped after failure; successful rows are preserved.", flush=True)
            break
    print("Results: " + str(out / "summary.csv"), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
