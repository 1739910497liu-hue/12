"""Matched-noise robustness probes and synchronized generation benchmarks."""
import copy
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import heat_template, stable_seed
from .models import HeatDiffusion
from .training import (load_checkpoint, file_hash, manifest_identity, make_vq, datasets,
                       to_device, autocast, save_image)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clean_output_directory(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    return out


class InferenceSession:
    def __init__(self, checkpoint, first_stage, data_root, manifests, clip_path=None, device="cuda", allow_smoke=False):
        self.device = torch.device(device)
        self.state = load_checkpoint(checkpoint)
        self.config = self.state["config"]
        if self.state["stage"] != 2:
            raise ValueError("Runtime and failure studies require a stage-2 checkpoint.")
        if not allow_smoke and (self.config.get("smoke_test") or self.state["step"] != self.config["stage2_steps"]):
            raise ValueError("Use a completed formal checkpoint; --allow-smoke is only for software tests.")
        if self.state["manifest_hashes"] != manifest_identity(manifests):
            raise ValueError("Manifest hashes differ from the training data protocol.")
        if file_hash(first_stage) != self.state["first_stage_hash"]:
            raise ValueError("The exact first-stage weights from training are required.")
        self.vq, _ = make_vq(first_stage, self.config, self.device)
        self.model = HeatDiffusion(self.config, clip_path).to(self.device).eval()
        self.model.load_state_dict(self.state["model"], strict=True)
        self.dataset = datasets(self.config, data_root, manifests, self.state["seed"], 2)[2]
        self.identity = {"experiment": self.config["experiment"], "training_seed": self.state["seed"],
                         "checkpoint_sha256": file_hash(checkpoint), "first_stage_hash": self.state["first_stage_hash"],
                         "manifest_hashes": self.state["manifest_hashes"], "config": self.config}

    def batch(self, index):
        item = self.dataset[index]
        return {key: value[None] if torch.is_tensor(value) else [value]
                for key, value in item.items() if key != "image"}

    @torch.no_grad()
    def generate(self, batch, steps, sample_seed=2026, decoder_template=None):
        batch = to_device(batch, self.device)
        generator = torch.Generator(device=self.device).manual_seed(stable_seed(sample_seed, 0, batch["id"][0]))
        with autocast(self.config, self.device):
            latent = self.model.sample(batch["template"], batch["numeric"], batch["caption"], steps, generator)
            prior = batch["template"] if decoder_template is None else decoder_template.to(self.device)
            generated = self.vq.decode(latent / self.state["latent_scale"], prior, requantize=True)
        return generated


def latency_statistics(seconds):
    values = np.array(seconds, dtype=np.float64)
    return {"latency_mean_ms": float(values.mean() * 1000), "latency_median_ms": float(np.median(values) * 1000),
            "latency_p95_ms": float(np.percentile(values, 95) * 1000),
            "latency_std_ms": float(values.std(ddof=1) * 1000) if len(values) > 1 else 0.,
            "samples_per_second": float(1 / values.mean())}


def benchmark(session, out, steps=(10, 20, 50, 100), warmup=5, repeats=30, deadline_ms=None, profile_flops=False):
    if warmup < 1 or repeats < 2 or (deadline_ms is not None and deadline_ms <= 0):
        raise ValueError("Require warmup>=1, repeats>=2 and a positive optional deployment deadline.")
    if any(not 2 <= n <= session.config["timesteps"] for n in steps):
        raise ValueError("All sampling step counts must lie in [2, timesteps].")
    out = clean_output_directory(out)
    device = session.device
    batches = [session.batch(i) for i in range(min(repeats, len(session.dataset)))]
    records = []
    for count in steps:
        for index in range(warmup):
            session.generate(batches[index % len(batches)], count)
        sync(device)
        resident = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for index in range(repeats):
            sync(device)
            start = time.perf_counter()
            generated = session.generate(batches[index % len(batches)], count)
            sync(device)
            elapsed.append(time.perf_counter() - start)
            if not torch.isfinite(generated).all():
                raise FloatingPointError("Non-finite benchmark output; do not report latency as usable generation.")
            del generated
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        row = dict(sampling_steps=count, batch_size=1, warmup=warmup, repeats=repeats,
                   resident_allocated_bytes=resident, inference_peak_allocated_bytes=peak,
                   inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   inference_extra_allocated_bytes=peak - resident if peak is not None else None,
                   **latency_statistics(elapsed))
        row["deadline_ms"] = deadline_ms
        row["p95_meets_generator_budget"] = None if deadline_ms is None else row["latency_p95_ms"] <= deadline_ms
        if profile_flops:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                session.generate(batches[0], count)
                sync(device)
            row["profiled_flops_partial"] = int(sum(event.flops for event in profiler.key_averages()))
        records.append(row)
    report = dict(session.identity, torch_version=torch.__version__, python_version=platform.python_version(),
                  device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                  precision=session.config["precision"],
                  resident_pipeline_parameters=sum(p.numel() for p in session.model.parameters()) + sum(p.numel() for p in session.vq.parameters()),
                  timing_scope="host-to-device inputs + report tokenization/CLIP + DDIM + requantization + decoder; excludes model loading, disk I/O and upstream simulation",
                  flops_scope="Partial profiler count, mainly supported matrix multiplications/convolutions; not a complete FLOP inventory.",
                  runtime=records)
    (out / "runtime.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "runtime.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def shift_template(template, fraction=.25):
    shifted = torch.zeros_like(template)
    offset = max(1, int(template.shape[-1] * fraction))
    shifted[..., offset:] = template[..., :-offset]
    return shifted


def corrupt_batch(batch, case, dataset, index, seed=2026):
    output = {key: value.clone() if torch.is_tensor(value) else list(value) for key, value in batch.items()}
    details, decoder_template = {"case": case}, None
    if case == "clean":
        pass
    elif case in ("prior_rot90", "decoder_prior_conflict"):
        rotated = torch.rot90(batch["template"], 1, (-2, -1))
        if case == "decoder_prior_conflict":
            decoder_template = rotated
            details["intervention"] = "clean-prior diffusion latent, rotated-prior decoder; identical initial noise"
        else:
            output["template"] = rotated
    elif case == "prior_shift25":
        output["template"] = shift_template(batch["template"])
    elif case == "prior_large":
        if dataset.prior_source != "analytic":
            raise ValueError("prior_large is an analytical-template stress test.")
        output["template"] = torch.from_numpy(heat_template(
            dataset.size, "large", stable_seed(seed, 0, batch["id"][0]),
            include_raw_prior=getattr(dataset, "include_raw_prior", False)))[None]
    elif case in ("empty_report", "wrong_report"):
        if dataset.condition not in ("text", "both"):
            raise ValueError("This checkpoint does not use report text.")
        if case == "empty_report":
            output["caption"] = [""]
        else:
            row = dataset.rows[index]
            candidates = [r for r in dataset.rows if r["group"] != row["group"]
                          and r.get(dataset.caption_field) != row.get(dataset.caption_field)]
            if not candidates:
                raise ValueError("wrong_report needs a different report from another test welding run.")
            chosen = candidates[stable_seed(seed, 0, row["id"]) % len(candidates)]
            output["caption"] = [chosen[dataset.caption_field]]
            details["report_donor_id"] = chosen["id"]
    elif case in ("numeric_error10", "numeric_error30"):
        if dataset.condition not in ("numeric", "both"):
            raise ValueError("This checkpoint does not use numeric inputs.")
        magnitude = .1 if case == "numeric_error10" else .3
        rng = np.random.RandomState(stable_seed(seed, 0, batch["id"][0]))
        signs = torch.tensor(rng.choice([-1., 1.], size=3), dtype=torch.float32)
        mean = torch.tensor(dataset.statistics["mean"], dtype=torch.float32)
        std = torch.tensor(dataset.statistics["std"], dtype=torch.float32)
        original = batch["numeric"] * std + mean
        changed = original * (1 + signs * magnitude)
        output["numeric"] = (changed - mean) / std
        details.update(original_numeric=original[0].tolist(), perturbed_numeric=changed[0].tolist(),
                       numeric_fields=dataset.statistics["fields"])
    else:
        raise ValueError("Unknown failure-study intervention: " + case)
    details["input_caption"] = output["caption"][0]
    return output, decoder_template, details


def morphology(tensor, threshold=.8):
    rgb = ((tensor[0].detach().float().cpu().clamp(-1, 1) + 1) / 2).numpy()
    gray = .299 * rgb[0] + .587 * rgb[1] + .114 * rgb[2]
    yy, xx = np.nonzero(gray >= threshold)
    result = {"bright_area_fraction": float(len(xx) / gray.size), "mean_brightness": float(gray.mean()),
              "output_clipped_fraction": float((tensor.detach().abs() > 1).float().mean())}
    if len(xx) > 2:
        covariance = np.cov(np.stack((xx, yy)))
        values = np.linalg.eigvalsh(covariance)
        result.update(bright_aspect_ratio=float(np.sqrt(values[-1] / max(values[0], 1e-8))),
                      center_x=float(xx.mean() / gray.shape[1]), center_y=float(yy.mean() / gray.shape[0]))
    return result


FAILURE_CASES = ["clean", "prior_rot90", "prior_shift25", "prior_large", "decoder_prior_conflict",
                 "empty_report", "wrong_report", "numeric_error10", "numeric_error30"]


def failures(session, out, limit=32, steps=50, sample_seed=2026, cases=None):
    if limit < 1:
        raise ValueError("limit must be positive")
    cases = list(dict.fromkeys(["clean"] + list(cases or FAILURE_CASES)))
    if any(case not in FAILURE_CASES for case in cases):
        raise ValueError("Unknown failure-study case")
    out = clean_output_directory(out)
    records, skipped = [], []
    session.vq.decoder.capture_fusion_stats = True
    try:
        for index in range(min(limit, len(session.dataset))):
            batch = session.batch(index)
            clean = session.generate(batch, steps, sample_seed)
            if not torch.isfinite(clean).all():
                raise FloatingPointError("Clean generation is non-finite for " + batch["id"][0])
            for case in cases:
                try:
                    changed, decoder_template, details = corrupt_batch(batch, case, session.dataset, index, sample_seed)
                except ValueError as error:
                    skipped.append({"id": batch["id"][0], "case": case, "reason": str(error)})
                    continue
                generated = session.generate(changed, steps, sample_seed, decoder_template)
                if not torch.isfinite(generated).all():
                    skipped.append({"id": batch["id"][0], "case": case, "reason": "non-finite generated output",
                                    "invalid_output": True})
                    continue
                name = batch["id"][0]
                folder = out / case
                folder.mkdir(exist_ok=True)
                save_image(generated[0], folder / (name + ".png"))
                records.append(dict(id=name, group=session.dataset.rows[index]["group"],
                                    delta_to_clean_mae=float((generated.clamp(-1, 1) - clean.clamp(-1, 1)).abs().mean() / 2),
                                    fusion_diagnostics=copy.deepcopy(session.vq.decoder.fusion_stats),
                                    **morphology(generated), **details))
    finally:
        session.vq.decoder.capture_fusion_stats = False
    report = dict(session.identity, sample_seed=sample_seed, sampling_steps=steps,
                  cases=cases, samples=min(limit, len(session.dataset)), records=records, skipped=skipped,
                  interpretation="Controlled input sensitivity probes; human annotation is required to label a case as a failure.")
    (out / "failures.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "blind_review.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "image", "reference_report", "reference_numeric_A_V_mm_s",
                                                    "morphology_degraded", "verified_report_consistent", "notes"])
        writer.writeheader()
        order = np.random.RandomState(sample_seed).permutation(len(records))
        blind = out / "blind"
        blind.mkdir()
        mapping = []
        reference_rows = {row["id"]: row for row in session.dataset.rows}
        import shutil
        for number, position in enumerate(order):
            record = records[position]
            review_id = "{:05d}".format(number)
            source = out / record["case"] / (record["id"] + ".png")
            shutil.copyfile(source, blind / (review_id + ".png"))
            reference = reference_rows[record["id"]]
            writer.writerow({"review_id": review_id, "image": "blind/" + review_id + ".png",
                             "reference_report": reference.get("caption_raw", reference.get("caption", "")),
                             "reference_numeric_A_V_mm_s": json.dumps([reference.get(k) for k in ("current_A", "voltage_V", "speed_mm_s")])})
            mapping.append({"review_id": review_id, "id": record["id"], "case": record["case"]})
        (out / "review_key.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return report
