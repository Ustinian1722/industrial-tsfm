from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_TRADEOFF_OBJECTIVES = ("normalized_rmse", "fit_seconds", "total_parameters")


def summarize_tradeoffs(
    result_table: pd.DataFrame,
    group_columns: Sequence[str] = ("model",),
    objectives: Sequence[str] = DEFAULT_TRADEOFF_OBJECTIVES,
) -> pd.DataFrame:
    """Aggregate complete results and mark models on the lower-is-better Pareto front."""
    missing_groups = [column for column in group_columns if column not in result_table.columns]
    if missing_groups:
        raise ValueError(f"Trade-off table is missing grouping columns: {missing_groups}")
    missing_objectives = [column for column in objectives if column not in result_table.columns]
    if missing_objectives:
        raise ValueError(f"Trade-off table is missing objectives: {missing_objectives}")
    complete = result_table[result_table["status"].astype(str) == "complete"].copy()
    if complete.empty:
        raise ValueError("Cannot compute trade-offs without complete rows")
    for objective in objectives:
        complete[objective] = pd.to_numeric(complete[objective], errors="coerce")
    complete = complete[np.isfinite(complete[list(objectives)].to_numpy(dtype=float)).all(axis=1)]
    if complete.empty:
        raise ValueError("Trade-off objectives contain no finite complete rows")
    rows: list[dict[str, object]] = []
    for group_key, group in complete.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key, strict=True))
        row["n_rows"] = len(group)
        if "seed" in group:
            row["n_seeds"] = int(group["seed"].nunique())
        for objective in objectives:
            row[objective] = float(group[objective].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    values = summary[list(objectives)].to_numpy(dtype=float)
    pareto = np.ones(len(summary), dtype=bool)
    for index in range(len(summary)):
        no_worse = np.all(values <= values[index], axis=1)
        strictly_better = np.any(values < values[index], axis=1)
        pareto[index] = not np.any(no_worse & strictly_better)
    summary["pareto_optimal"] = pareto
    return summary.sort_values(
        ["pareto_optimal", *objectives],
        ascending=[False, *([True] * len(objectives))],
        kind="stable",
    ).reset_index(drop=True)
