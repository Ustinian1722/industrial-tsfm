from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionShiftRequest:
    """Chronological source-to-target shift analysis over one audited table."""

    feature_columns: tuple[str, ...] = ()
    reference_fraction: float = 0.50
    target_fraction: float = 0.50
    min_rows_per_partition: int = 16
    top_k: int = 12

    def validate(self) -> None:
        if not 0.0 < self.reference_fraction < 1.0:
            raise ValueError("reference_fraction must be in (0, 1)")
        if not 0.0 < self.target_fraction < 1.0:
            raise ValueError("target_fraction must be in (0, 1)")
        if self.reference_fraction + self.target_fraction > 1.0 + 1e-12:
            raise ValueError("reference_fraction + target_fraction must not exceed 1")
        if self.min_rows_per_partition < 4:
            raise ValueError("min_rows_per_partition must be at least 4")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class CrossSourceShiftRequest:
    """Distribution-shift request for two explicitly different data sources."""

    feature_columns: tuple[str, ...] = ()
    min_rows_per_source: int = 16
    top_k: int = 12

    def validate(self) -> None:
        if self.min_rows_per_source < 4:
            raise ValueError("min_rows_per_source must be at least 4")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


def _default_features(data_audit: dict[str, Any]) -> list[str]:
    targets = set(data_audit.get("targets", {}).get("configured", []))
    return [
        str(column)
        for column in data_audit.get("usable_numeric_features", [])
        if str(column) not in targets
    ]


def _common_default_features(
    reference_audit: dict[str, Any],
    target_audit: dict[str, Any],
) -> list[str]:
    reference = _default_features(reference_audit)
    target = set(_default_features(target_audit))
    return [feature for feature in reference if feature in target]


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is missing")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _summarize_feature_shift(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    min_rows: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        source_values = _finite_values(reference, feature)
        target_values = _finite_values(target, feature)
        if len(source_values) < min_rows or len(target_values) < min_rows:
            continue

        source_mean = float(source_values.mean())
        target_mean = float(target_values.mean())
        source_std = float(source_values.std(ddof=0))
        target_std = float(target_values.std(ddof=0))
        source_scale = max(source_std, 1.0e-12)
        standardized_mean_shift = abs(target_mean - source_mean) / source_scale
        std_ratio = target_std / source_scale
        abs_log_std_ratio = abs(float(np.log(max(std_ratio, 1.0e-12))))

        source_median = float(np.median(source_values))
        target_median = float(np.median(target_values))
        source_mad = float(np.median(np.abs(source_values - source_median)))
        robust_scale = max(1.4826 * source_mad, 1.0e-12)
        robust_median_shift = abs(target_median - source_median) / robust_scale

        rows.append(
            {
                "feature": feature,
                "reference_rows_finite": len(source_values),
                "target_rows_finite": len(target_values),
                "reference_mean": source_mean,
                "target_mean": target_mean,
                "mean_shift_raw": target_mean - source_mean,
                "reference_std": source_std,
                "target_std": target_std,
                "standardized_mean_shift": float(standardized_mean_shift),
                "std_ratio": float(std_ratio),
                "abs_log_std_ratio": float(abs_log_std_ratio),
                "reference_median": source_median,
                "target_median": target_median,
                "robust_median_shift": float(robust_median_shift),
            }
        )

    if not rows:
        raise ValueError("no features have enough finite observations in both partitions")

    rows.sort(key=lambda item: float(item["standardized_mean_shift"]), reverse=True)
    standardized = np.asarray(
        [float(item["standardized_mean_shift"]) for item in rows], dtype=np.float64
    )
    log_std = np.asarray([float(item["abs_log_std_ratio"]) for item in rows], dtype=np.float64)
    robust = np.asarray([float(item["robust_median_shift"]) for item in rows], dtype=np.float64)
    rms_shift = float(np.sqrt(np.mean(np.square(standardized))))
    aggregate = {
        "mean_abs_standardized_mean_shift": float(standardized.mean()),
        "rms_standardized_mean_shift": rms_shift,
        "max_abs_standardized_mean_shift": float(standardized.max()),
        "mean_abs_log_std_ratio": float(log_std.mean()),
        "max_abs_log_std_ratio": float(log_std.max()),
        "mean_robust_median_shift": float(robust.mean()),
        "max_robust_median_shift": float(robust.max()),
        "shift_score": rms_shift,
    }
    return rows[:top_k], aggregate


def analyze_distribution_shift(
    frame: pd.DataFrame,
    data_audit: dict[str, Any],
    request: DistributionShiftRequest,
) -> dict[str, Any]:
    """Compare early-reference and late-target feature distributions.

    The primary score is the RMS source-standardized mean shift, matching the
    convention already consumed by the adaptation engine. Dispersion change is
    reported separately so routing can remain transparent.
    """

    request.validate()
    if frame.empty:
        raise ValueError("cannot analyze an empty dataframe")

    row_count = len(frame)
    reference_rows = int(np.floor(row_count * request.reference_fraction))
    target_rows = int(np.floor(row_count * request.target_fraction))
    if reference_rows < request.min_rows_per_partition:
        raise ValueError("reference partition has too few rows")
    if target_rows < request.min_rows_per_partition:
        raise ValueError("target partition has too few rows")
    if reference_rows + target_rows > row_count:
        raise ValueError("reference and target partitions overlap")

    reference = frame.iloc[:reference_rows]
    target = frame.iloc[row_count - target_rows :]
    features = list(request.feature_columns) if request.feature_columns else _default_features(data_audit)
    if not features:
        raise ValueError("no feature columns are available for distribution-shift analysis")
    feature_rows, aggregate = _summarize_feature_shift(
        reference,
        target,
        features,
        request.min_rows_per_partition,
        request.top_k,
    )

    return {
        "schema_version": "industrial_tsfm.distribution_shift.v1",
        "analysis_type": "chronological_distribution_shift",
        "reference_scope": "earliest_prefix",
        "target_scope": "latest_suffix",
        "target_labels_used": False,
        "future_information_used_for_reference_statistics": False,
        "request": asdict(request),
        "partition": {
            "rows_total": row_count,
            "reference_row_start": 0,
            "reference_row_end": reference_rows,
            "target_row_start": row_count - target_rows,
            "target_row_end": row_count,
            "gap_rows": row_count - reference_rows - target_rows,
        },
        "aggregate": aggregate,
        "features": feature_rows,
    }


def compare_source_distributions(
    reference_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    reference_audit: dict[str, Any],
    target_audit: dict[str, Any],
    request: CrossSourceShiftRequest,
) -> dict[str, Any]:
    """Compare feature drift between an explicit reference and target data source."""

    request.validate()
    if reference_frame.empty or target_frame.empty:
        raise ValueError("reference and target data sources must both be non-empty")
    features = (
        list(request.feature_columns)
        if request.feature_columns
        else _common_default_features(reference_audit, target_audit)
    )
    if not features:
        raise ValueError("reference and target sources have no common usable feature columns")
    feature_rows, aggregate = _summarize_feature_shift(
        reference_frame,
        target_frame,
        features,
        request.min_rows_per_source,
        request.top_k,
    )
    return {
        "schema_version": "industrial_tsfm.cross_source_shift.v1",
        "analysis_type": "cross_source_distribution_shift",
        "reference_scope": "explicit_reference_source",
        "target_scope": "explicit_target_source",
        "target_labels_used": False,
        "reference_statistics_use_target_data": False,
        "request": asdict(request),
        "partition": {
            "reference_rows": len(reference_frame),
            "target_rows": len(target_frame),
        },
        "aggregate": aggregate,
        "features": feature_rows,
    }
