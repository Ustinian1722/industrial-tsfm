from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def compute_distribution_shift(
    frame: pd.DataFrame,
    source_entities: Iterable[str],
    target_entities: Iterable[str],
    feature_names: Iterable[str],
    source_scale: np.ndarray,
) -> dict[str, Any]:
    """Quantify source-to-target covariate shift using source-only reference statistics."""
    feature_names = tuple(feature_names)
    source_entities = tuple(sorted(set(source_entities)))
    target_entities = tuple(sorted(set(target_entities)))
    source = frame[frame["entity_id"].isin(source_entities)]
    target = frame[frame["entity_id"].isin(target_entities)]
    if source.empty or target.empty:
        raise ValueError("Both source and target entities must have rows for shift analysis")
    source_scale = np.asarray(source_scale, dtype=np.float64)
    if source_scale.shape != (len(feature_names),) or np.any(source_scale <= 0):
        raise ValueError("source_scale must contain one positive value per feature")
    source_values = source.loc[:, feature_names].to_numpy(dtype=np.float64)
    target_values = target.loc[:, feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(source_values).all() or not np.isfinite(target_values).all():
        raise ValueError("Shift analysis cannot use non-finite source or target values")
    source_mean = source_values.mean(axis=0)
    target_mean = target_values.mean(axis=0)
    source_std = source_values.std(axis=0)
    target_std = target_values.std(axis=0)
    mean_shift = target_mean - source_mean
    std_ratio = target_std / np.maximum(source_std, 1.0e-12)
    log_std_ratio = np.log(np.maximum(std_ratio, 1.0e-12))
    rows = [
        {
            "feature": feature,
            "source_mean": float(source_mean[index]),
            "target_mean": float(target_mean[index]),
            "mean_shift_raw": float(mean_shift[index]),
            "standardized_mean_shift": float(abs(mean_shift[index]) / source_scale[index]),
            "source_std": float(source_std[index]),
            "target_std": float(target_std[index]),
            "std_ratio": float(std_ratio[index]),
            "abs_log_std_ratio": float(abs(log_std_ratio[index])),
        }
        for index, feature in enumerate(feature_names)
    ]
    standardized = np.abs(mean_shift) / source_scale
    return {
        "reference": "source_entity_raw_values",
        "source_entities": list(source_entities),
        "target_entities": list(target_entities),
        "source_rows": len(source),
        "target_rows": len(target),
        "features": rows,
        "aggregate": {
            "mean_abs_standardized_mean_shift": float(standardized.mean()),
            "rms_standardized_mean_shift": float(np.sqrt(np.square(standardized).mean())),
            "max_abs_standardized_mean_shift": float(standardized.max()),
            "mean_abs_log_std_ratio": float(np.abs(log_std_ratio).mean()),
            "max_abs_log_std_ratio": float(np.abs(log_std_ratio).max()),
        },
    }
