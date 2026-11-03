"""Explicit manifests, train-only numeric scaling, deterministic physical priors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUMERIC_FIELDS = ("current_A", "voltage_V", "speed_mm_s")
PERTURBATIONS = {
    "none": (0.0, 0.0, 0.0),
    "small": (15.0, 0.05, 0.025),
    "baseline": (45.0, 0.10, 0.05),
    "large": (75.0, 0.20, 0.10),
}


def stable_seed(seed, epoch, sample_id):
    key = "{}:{}:{}".format(seed, epoch, sample_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def heat_template(size, profile="none", seed=0, threshold=0.1, include_raw_prior=False):
    """Same two full elliptical Gaussians as imagenet.py, NOT a Goldak PDE solver.

    Parameters are measured at 256 px. Rigid rotation + common isotropic scale
    preserve the ratio of both lobes. No image pixels or numeric labels enter.
    Returns normalized intensity and binary region as two separate channels.
    Optionally appends the pre-normalization heat field for decoder-only gates.
    The first two channels stay identical to the legacy training protocol.
    """
    angle_range, scale_range, shift_range = PERTURBATIONS[profile]
    rng = np.random.RandomState(seed)
    angle = np.deg2rad(rng.uniform(-angle_range, angle_range))
    scale = rng.uniform(1 - scale_range, 1 + scale_range)
    cx, cy = 0.5 + rng.uniform(-shift_range, shift_range, size=2)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    x, y = (xx - cx * (size - 1)) * 256 / size, (yy - cy * (size - 1)) * 256 / size
    xr, yr = np.cos(angle) * x + np.sin(angle) * y, -np.sin(angle) * x + np.cos(angle) * y
    heat = np.zeros((size, size), dtype=np.float32)
    for alpha, beta in ((48., 24.), (72., 32.)):
        alpha, beta = alpha * scale, beta * scale
        heat += 0.5 / (np.pi * alpha * beta) * np.exp(-3 * ((xr / alpha) ** 2 + (yr / beta) ** 2))
    raw_heat = heat.copy() if include_raw_prior else None
    heat /= max(float(heat.max()), 1e-8)
    mask = (heat >= threshold).astype(np.float32)
    channels = (heat, mask, raw_heat) if include_raw_prior else (heat, mask)
    return np.stack(channels).astype(np.float32)


def read_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest: {}".format(path))
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs: {}".format(path))
    return rows


def numeric_statistics(rows):
    missing = [(row.get("id", row.get("image", "unknown")), [key for key in NUMERIC_FIELDS if key not in row])
               for row in rows if any(key not in row for key in NUMERIC_FIELDS)]
    if missing:
        raise ValueError("Missing numeric labels (A/V/mm/s), examples: {}".format(missing[:5]))
    values = np.array([[row[k] for k in NUMERIC_FIELDS] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Current, voltage, speed must be finite and > 0, in A/V/mm/s.")
    return {"fields": list(NUMERIC_FIELDS), "mean": values.mean(0).tolist(),
            "std": np.maximum(values.std(0), 1e-6).tolist(), "count": len(rows)}


class HeatDataset(Dataset):
    def __init__(self, manifest, data_root, size=256, profile="none", seed=42,
                 training=False, condition="none", statistics=None, caption_field="caption", prior_source="analytic",
                 include_raw_prior=False):
        self.rows = read_manifest(manifest)
        self.root = Path(data_root).resolve()
        self.size, self.profile, self.seed = size, profile, seed
        self.training, self.condition, self.statistics = training, condition, statistics
        self.caption_field, self.prior_source = caption_field, prior_source
        self.include_raw_prior = include_raw_prior
        if prior_source not in ("analytic", "simulation"):
            raise ValueError("Unknown physical prior source: " + prior_source)
        if include_raw_prior and prior_source != "analytic":
            raise ValueError("Raw decoder-gate prior currently requires prior_source='analytic'.")
        if prior_source == "simulation" and profile != "none":
            raise ValueError("Simulation comparison uses fixed fields; compare against analytic perturb_none.")
        self.epoch = 0
        if condition in ("numeric", "both"):
            if statistics is None or statistics["fields"] != list(NUMERIC_FIELDS):
                raise ValueError("Numeric conditions require training-set statistics in A/V/mm/s.")
            numeric_statistics(self.rows)  # validate only; never fit validation/test normalization
        for row in self.rows:
            if condition in ("text", "both") and not str(row.get(caption_field, "")).strip():
                raise ValueError("Missing real report caption for {}".format(row["id"]))
            path = (self.root / row["image"]).resolve()
            if self.root not in path.parents or not path.is_file():
                raise ValueError("Image missing or outside data root: {}".format(path))
            if prior_source == "simulation":
                simulation = (self.root / row.get("simulation_field", "")).resolve()
                if (self.root not in simulation.parents or not simulation.is_file()
                        or not row.get("simulation_sha256") or not row.get("simulation_provenance")):
                    raise ValueError("Real simulation_field (.npy), hash and provenance required: " + row["id"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["image"]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
            x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.
        template_seed = stable_seed(self.seed, self.epoch if self.training else 0, row["id"])
        template = heat_template(self.size, self.profile if self.training else "none", template_seed,
                                 include_raw_prior=self.include_raw_prior)
        if self.prior_source == "simulation":
            field = np.load(self.root / row["simulation_field"], allow_pickle=False)
            if field.ndim != 2 or not np.isfinite(field).all() or field.min() < 0 or field.max() <= 0:
                raise ValueError("Simulation field must be a finite, nonnegative 2D array with positive maximum.")
            heat = torch.tensor(field / field.max(), dtype=torch.float32)[None, None]
            heat = torch.nn.functional.interpolate(heat, size=(self.size, self.size), mode="bilinear", align_corners=False)[0, 0].numpy()
            template = np.stack((heat, (heat >= .1).astype(np.float32)))
        numeric = np.zeros(3, dtype=np.float32)
        if self.condition in ("numeric", "both"):
            numeric = (np.array([row[k] for k in NUMERIC_FIELDS]) - self.statistics["mean"]) / self.statistics["std"]
        return {"image": torch.from_numpy(x), "template": torch.from_numpy(template),
                "numeric": torch.tensor(numeric, dtype=torch.float32),
                "caption": str(row.get(self.caption_field, "")), "id": row["id"]}
