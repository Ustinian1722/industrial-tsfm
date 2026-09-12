from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LagAnalysisRequest:
    target_column: str
    feature_columns: tuple[str, ...] = ()
    max_lag: int = 12
    min_pairs: int = 12
    top_k: int = 12

    def validate(self) -> None:
        if not self.target_column.strip():
            raise ValueError("target_column must be non-empty")
        if self.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if self.min_pairs < 3:
            raise ValueError("min_pairs must be at least 3")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is missing")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def analyze_lagged_relationships(
    frame: pd.DataFrame,
    data_audit: dict[str, Any],
    request: LagAnalysisRequest,
) -> dict[str, Any]:
    """Rank lagged feature/target associations without making a causal claim.

    A lag of k means feature[t-k] is compared with target[t]. Only historical or
    simultaneous feature values are considered; future feature values are never
    shifted backward into the target time step.
    """

    request.validate()
    if frame.empty:
        raise ValueError("cannot analyze an empty dataframe")
    target = _numeric_series(frame, request.target_column)
    if int(target.notna().sum()) < request.min_pairs:
        raise ValueError("target has too few finite observations for lag analysis")

    if request.feature_columns:
        features = list(request.feature_columns)
    else:
        features = [
            column
            for column in data_audit.get("usable_numeric_features", [])
            if column != request.target_column
        ]
    if not features:
        raise ValueError("no feature columns are available for lag analysis")

    rankings: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        values = _numeric_series(frame, feature)
        curve: list[dict[str, Any]] = []
        for lag in range(request.max_lag + 1):
            shifted = values.shift(lag)
            paired = pd.concat([shifted.rename("x"), target.rename("y")], axis=1).dropna()
            if len(paired) < request.min_pairs:
                continue
            pearson = float(paired["x"].corr(paired["y"], method="pearson"))
            spearman = float(paired["x"].corr(paired["y"], method="spearman"))
            if not np.isfinite(pearson):
                pearson = 0.0
            if not np.isfinite(spearman):
                spearman = 0.0
            curve.append(
                {
                    "lag": lag,
                    "pairs": len(paired),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
        curves[feature] = curve
        if not curve:
            continue
        best = max(curve, key=lambda row: abs(float(row["pearson"])))
        rankings.append(
            {
                "feature": feature,
                "best_lag": int(best["lag"]),
                "best_pearson": float(best["pearson"]),
                "best_abs_pearson": abs(float(best["pearson"])),
                "spearman_at_best_lag": float(best["spearman"]),
                "pairs_at_best_lag": int(best["pairs"]),
            }
        )

    rankings.sort(key=lambda row: float(row["best_abs_pearson"]), reverse=True)
    return {
        "schema_version": "industrial_tsfm.lag_analysis.v1",
        "analysis_type": "lagged_association",
        "interpretation_boundary": "association_not_causality",
        "future_information_used": False,
        "request": asdict(request),
        "target_column": request.target_column,
        "rankings": rankings[: request.top_k],
        "curves": {row["feature"]: curves[row["feature"]] for row in rankings[: request.top_k]},
    }
