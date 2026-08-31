from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_METRICS = (
    "mae",
    "rmse",
    "mase",
    "normalized_mae",
    "normalized_rmse",
    "macro_entity_mae",
    "macro_entity_rmse",
    "macro_entity_normalized_mae",
    "macro_entity_normalized_rmse",
    "fit_seconds",
    "inference_seconds",
    "peak_gpu_memory_mb",
)


def aggregate_result_table(
    result_table: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate complete per-seed rows without hiding single-seed uncertainty."""
    if result_table.empty:
        return pd.DataFrame()
    missing_groups = [column for column in group_columns if column not in result_table.columns]
    if missing_groups:
        raise ValueError(f"Missing result grouping columns: {missing_groups}")
    if "seed" not in result_table.columns:
        raise ValueError("Result table must contain a seed column")

    complete = result_table[result_table["status"].astype(str) == "complete"].copy()
    if complete.empty:
        raise ValueError("Cannot aggregate a result table without complete rows")

    rows: list[dict[str, object]] = []
    for group_key, group in complete.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key, strict=True))
        n_seeds = int(group["seed"].nunique())
        row["n_seeds"] = n_seeds
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                margin = 1.96 * std / np.sqrt(len(values))
                ci_low = mean - margin
                ci_high = mean + margin
            else:
                # A single seed has no empirical uncertainty estimate.
                ci_low = np.nan
                ci_high = np.nan
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def write_result_summary(
    result_table: pd.DataFrame,
    output_path: str | Path,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    summary = aggregate_result_table(result_table, group_columns)
    summary.to_csv(output_path, index=False)
    return summary
