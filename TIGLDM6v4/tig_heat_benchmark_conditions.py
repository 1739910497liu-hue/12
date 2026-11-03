"""Native analytic evaluation conditions without reading target image pixels.

Only for generation timing. This class does not evaluate image quality, validate
target images, or change the manifest/statistics provenance checks of the caller.
The caller must explicitly opt in and verify the original training manifests.
"""
import numbers

import numpy as np
import torch


class HeatBenchmarkConditions:
    def __init__(self, manifest, size, seed, condition, statistics,
                 caption_field="caption", prior_source="analytic", include_raw_prior=False):
        if prior_source != "analytic":
            raise ValueError("Conditions-only timing supports prior_source='analytic' only; "
                             "simulation fields require the native dataset loader.")
        if condition not in ("none", "text", "numeric", "both"):
            raise ValueError("Unknown condition mode: {!r}".format(condition))
        if not isinstance(size, numbers.Integral) or isinstance(size, bool) or size <= 0:
            raise ValueError("Template size must be a positive integer.")
        if not isinstance(caption_field, str) or not caption_field:
            raise ValueError("caption_field must be a nonempty string.")
        # Resolve the project native implementation after the worker has selected
        # its project root, rather than shipping a second template implementation.
        from ldm.heat_source.data import (
            NUMERIC_FIELDS, heat_template, numeric_statistics, read_manifest, stable_seed,
        )
        self.rows = read_manifest(manifest)
        self.size, self.seed, self.condition = size, seed, condition
        self.statistics, self.caption_field = statistics, caption_field
        self.include_raw_prior = include_raw_prior
        self._fields = NUMERIC_FIELDS
        self._heat_template, self._stable_seed = heat_template, stable_seed
        if condition in ("numeric", "both"):
            if not isinstance(statistics, dict) or statistics.get("fields") != list(NUMERIC_FIELDS):
                raise ValueError("Numeric conditions require training-set statistics in A/V/mm/s.")
            for key in ("mean", "std"):
                values = statistics.get(key)
                if (not isinstance(values, (list, tuple)) or len(values) != len(NUMERIC_FIELDS)
                        or any(not isinstance(value, numbers.Real) or isinstance(value, bool)
                               for value in values)):
                    raise ValueError("Numeric statistics {} must contain three real numbers.".format(key))
                array = np.asarray(values, dtype=np.float64)
                if not np.isfinite(array).all() or (key == "std" and (array <= 0).any()):
                    raise ValueError("Numeric statistics must be finite with strictly positive std.")
            for row in self.rows:
                if any(not isinstance(row.get(key), numbers.Real) or isinstance(row.get(key), bool)
                       for key in NUMERIC_FIELDS):
                    raise ValueError("Missing or malformed numeric labels for {!r}.".format(row["id"]))
            numeric_statistics(self.rows)  # Validate labels only; never refit test scaling.
        if condition in ("text", "both"):
            for row in self.rows:
                caption = row.get(caption_field)
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError("Missing real report caption for {}".format(row["id"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        template = self._heat_template(
            self.size, "none", self._stable_seed(self.seed, 0, row["id"]),
            include_raw_prior=self.include_raw_prior,
        )
        numeric = np.zeros(3, dtype=np.float32)
        if self.condition in ("numeric", "both"):
            # Keep the expression and final dtype identical to HeatDataset.
            numeric = (np.array([row[key] for key in self._fields]) - self.statistics["mean"]) / self.statistics["std"]
        return {"template": torch.from_numpy(template),
                "numeric": torch.tensor(numeric, dtype=torch.float32),
                "caption": str(row.get(self.caption_field, "")), "id": row["id"]}
