from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _constraint_dict(constraints: Mapping[str, Any] | None) -> dict[str, Any]:
    constraints = dict(constraints or {})
    allowed = constraints.get("allowed_training_modes")
    if allowed is not None:
        constraints["allowed_training_modes"] = [str(value) for value in allowed]
    return constraints


def rank_model_candidates(
    result_table: pd.DataFrame,
    metric: str = "normalized_rmse",
    constraints: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Rank models from validation-only per-seed metrics and explicit resource constraints."""
    required = {"model", "status", metric}
    missing = sorted(required - set(result_table.columns))
    if missing:
        raise ValueError(f"Selection table is missing columns: {missing}")
    constraints = _constraint_dict(constraints)
    candidates = result_table[result_table["status"].astype(str) == "complete"].copy()
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates[np.isfinite(candidates[metric].to_numpy(dtype=float))]
    if "max_parameters" in constraints and constraints["max_parameters"] is not None:
        if "total_parameters" not in candidates:
            raise ValueError("max_parameters requires a total_parameters column")
        limit = float(constraints["max_parameters"])
        values = pd.to_numeric(candidates["total_parameters"], errors="coerce")
        candidates = candidates[values.notna() & (values <= limit)]
    if "max_inference_seconds" in constraints and constraints["max_inference_seconds"] is not None:
        if "inference_seconds" not in candidates:
            raise ValueError("max_inference_seconds requires an inference_seconds column")
        limit = float(constraints["max_inference_seconds"])
        values = pd.to_numeric(candidates["inference_seconds"], errors="coerce")
        candidates = candidates[values.notna() & (values <= limit)]
    allowed_modes = constraints.get("allowed_training_modes")
    if allowed_modes is not None:
        if "training_mode" not in candidates:
            raise ValueError("allowed_training_modes requires a training_mode column")
        candidates = candidates[candidates["training_mode"].astype(str).isin(allowed_modes)]
    if candidates.empty:
        raise ValueError("No complete validation model remains after selection constraints")

    rows: list[dict[str, object]] = []
    for model, group in candidates.groupby("model", sort=True):
        values = group[metric].to_numpy(dtype=float)
        row: dict[str, object] = {
            "model": str(model),
            "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(values),
            f"{metric}_mean": float(values.mean()),
            f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
        for column in ("fit_seconds", "inference_seconds", "total_parameters"):
            if column in group:
                numeric = pd.to_numeric(group[column], errors="coerce").dropna()
                if not numeric.empty:
                    row[f"{column}_mean"] = float(numeric.mean())
        modes = sorted(group["training_mode"].astype(str).unique().tolist()) if "training_mode" in group else []
        row["training_modes"] = ",".join(modes)
        rows.append(row)
    ranking = pd.DataFrame(rows)
    sort_columns = [f"{metric}_mean"]
    ascending = [True]
    if "inference_seconds_mean" in ranking:
        sort_columns.append("inference_seconds_mean")
        ascending.append(True)
    sort_columns.append("model")
    ascending.append(True)
    ranking = ranking.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def select_best_model(
    result_table: pd.DataFrame,
    metric: str = "normalized_rmse",
    constraints: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    ranking = rank_model_candidates(result_table, metric=metric, constraints=constraints)
    return {
        "selection_metric": metric,
        "selection_split": "validation",
        "selected_model": str(ranking.iloc[0]["model"]),
        "constraints": _constraint_dict(constraints),
        "ranking": ranking.to_dict(orient="records"),
    }


def recommend_adaptation_strategy(
    model: str,
    training_mode: str,
    support_fraction: float,
    n_support_windows: int,
    residual_calibration_enabled: bool = True,
    min_calibration_windows: int = 2,
) -> dict[str, object]:
    """Return a transparent support-aware policy recommendation without target labels."""
    if support_fraction <= 0.0:
        strategy = "source_only"
        reason = "no target support is available"
    elif training_mode == "supervised":
        strategy = "few_shot_finetune"
        reason = "the model exposes a supervised trainable adapter"
    elif residual_calibration_enabled and n_support_windows >= min_calibration_windows:
        strategy = "target_residual_calibration"
        reason = "observed support is large enough for a held-in residual calibrator"
    else:
        strategy = "target_scaling"
        reason = "use frozen zero-shot inference with target-support scaling"
    return {
        "model": model,
        "training_mode": training_mode,
        "support_fraction": float(support_fraction),
        "n_support_windows": int(n_support_windows),
        "recommended_strategy": strategy,
        "selection_basis": "model_contract_and_observed_support_only",
        "reason": reason,
    }


def compute_generalization_gap(
    validation_table: pd.DataFrame,
    target_table: pd.DataFrame,
    metric: str = "normalized_rmse",
) -> pd.DataFrame:
    """Join validation and target metrics by model/seed without changing either metric."""
    required = {"model", "seed", "status", metric}
    for name, table in (("validation", validation_table), ("target", target_table)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} table is missing columns: {missing}")
    validation = validation_table[validation_table["status"].astype(str) == "complete"].copy()
    target = target_table[target_table["status"].astype(str) == "complete"].copy()
    validation = validation[["model", "seed", metric]]
    target = target[["model", "seed", metric]]
    validation[metric] = pd.to_numeric(validation[metric], errors="coerce")
    target[metric] = pd.to_numeric(target[metric], errors="coerce")
    validation = (
        validation.groupby(["model", "seed"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "validation_metric"})
    )
    target = (
        target.groupby(["model", "seed"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "target_metric"})
    )
    merged = validation.merge(target, on=["model", "seed"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No model/seed pairs overlap between validation and target tables")
    merged["absolute_gap"] = merged["target_metric"] - merged["validation_metric"]
    denominator = merged["validation_metric"].abs().clip(lower=1.0e-12)
    merged["relative_gap"] = merged["absolute_gap"] / denominator
    return merged.sort_values(["model", "seed"], kind="stable").reset_index(drop=True)


def summarize_generalization_gap(gap_table: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "absolute_gap", "relative_gap"}
    missing = sorted(required - set(gap_table.columns))
    if missing:
        raise ValueError(f"Generalization-gap table is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for model, group in gap_table.groupby("model", sort=True):
        absolute = group["absolute_gap"].to_numpy(dtype=float)
        relative = group["relative_gap"].to_numpy(dtype=float)
        rows.append(
            {
                "model": str(model),
                "n_seeds": len(group),
                "absolute_gap_mean": float(absolute.mean()),
                "absolute_gap_std": float(absolute.std(ddof=1)) if len(absolute) > 1 else 0.0,
                "relative_gap_mean": float(relative.mean()),
                "relative_gap_std": float(relative.std(ddof=1)) if len(relative) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("model", kind="stable").reset_index(drop=True)
