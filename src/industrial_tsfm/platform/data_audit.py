from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _infer_time_summary(frame: pd.DataFrame, timestamp_column: str | None) -> dict[str, Any]:
    if not timestamp_column:
        return {"timestamp_column": None, "valid": False, "reason": "not_configured"}
    if timestamp_column not in frame.columns:
        return {"timestamp_column": timestamp_column, "valid": False, "reason": "missing_column"}

    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return {"timestamp_column": timestamp_column, "valid": False, "reason": "unparseable"}

    diffs = valid.diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    median_seconds = None
    if not positive.empty:
        median_seconds = _safe_float(positive.median().total_seconds())

    return {
        "timestamp_column": timestamp_column,
        "valid": True,
        "parse_failure_ratio": float(parsed.isna().mean()),
        "duplicate_timestamps": int(valid.duplicated().sum()),
        "out_of_order_transitions": int((diffs < pd.Timedelta(0)).sum()),
        "start": valid.min().isoformat(),
        "end": valid.max().isoformat(),
        "median_step_seconds": median_seconds,
    }


def _high_correlation_pairs(
    numeric: pd.DataFrame,
    columns: Iterable[str],
    threshold: float,
) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in numeric.columns]
    if len(selected) < 2:
        return []
    corr = numeric[selected].corr().abs()
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                pairs.append({"left": left, "right": right, "abs_correlation": float(value)})
    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs


def audit_dataframe(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    target_columns: Iterable[str] = (),
    correlation_threshold: float = 0.98,
    missing_feature_limit: float = 0.50,
) -> dict[str, Any]:
    """Create a deterministic, JSON-serialisable industrial data-health report.

    The audit is intentionally model agnostic. It does not impute, resample, or
    silently drop variables; it reports risks and proposes a conservative set of
    numerically usable features for the next pipeline stage.
    """

    if frame.empty:
        raise ValueError("cannot audit an empty dataframe")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if not 0.0 <= missing_feature_limit <= 1.0:
        raise ValueError("missing_feature_limit must be in [0, 1]")

    targets = tuple(target_columns)
    missing_targets = [column for column in targets if column not in frame.columns]
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

    variable_rows: list[dict[str, Any]] = []
    constant_columns: list[str] = []
    high_missing_columns: list[str] = []
    usable_features: list[str] = []

    for column in frame.columns:
        series = frame[column]
        numeric_series = numeric[column] if column in numeric.columns else None
        missing_ratio = float(series.isna().mean())
        unique_non_null = int(series.nunique(dropna=True))
        is_constant = unique_non_null <= 1
        if is_constant:
            constant_columns.append(column)
        if missing_ratio > missing_feature_limit:
            high_missing_columns.append(column)
        if (
            column in numeric.columns
            and column != timestamp_column
            and not is_constant
            and missing_ratio <= missing_feature_limit
        ):
            usable_features.append(column)

        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "is_numeric": column in numeric.columns,
            "missing_ratio": missing_ratio,
            "unique_non_null": unique_non_null,
            "is_constant": is_constant,
            "is_target": column in targets,
        }
        if numeric_series is not None:
            row.update(
                {
                    "mean": _safe_float(numeric_series.mean()),
                    "std": _safe_float(numeric_series.std(ddof=0)),
                    "min": _safe_float(numeric_series.min()),
                    "max": _safe_float(numeric_series.max()),
                }
            )
        variable_rows.append(row)

    time_summary = _infer_time_summary(frame, timestamp_column)
    duplicate_rows = int(frame.duplicated().sum())
    correlations = _high_correlation_pairs(
        numeric,
        [column for column in numeric_columns if column != timestamp_column],
        correlation_threshold,
    )

    penalties: dict[str, float] = {
        "duplicate_rows": min(15.0, duplicate_rows / max(len(frame), 1) * 100.0),
        "constant_columns": min(20.0, len(constant_columns) * 4.0),
        "high_missing_columns": min(25.0, len(high_missing_columns) * 5.0),
        "missing_targets": min(30.0, len(missing_targets) * 15.0),
    }
    if time_summary.get("valid"):
        penalties["timestamp_parse_failures"] = min(
            10.0, float(time_summary.get("parse_failure_ratio", 0.0)) * 100.0
        )
        penalties["timestamp_order"] = min(
            10.0,
            float(time_summary.get("out_of_order_transitions", 0)) / max(len(frame) - 1, 1) * 100.0,
        )
    elif timestamp_column:
        penalties["timestamp_invalid"] = 15.0

    readiness_score = max(0.0, 100.0 - sum(penalties.values()))
    risks: list[str] = []
    if duplicate_rows:
        risks.append(f"{duplicate_rows} duplicate rows detected")
    if constant_columns:
        risks.append(f"constant/degenerate columns: {', '.join(constant_columns)}")
    if high_missing_columns:
        risks.append(f"high-missing columns: {', '.join(high_missing_columns)}")
    if missing_targets:
        risks.append(f"missing target columns: {', '.join(missing_targets)}")
    if time_summary.get("out_of_order_transitions", 0):
        risks.append("timestamp sequence contains out-of-order transitions")
    if correlations:
        risks.append(f"{len(correlations)} highly correlated numeric feature pairs detected")

    return {
        "schema_version": "industrial_tsfm.data_audit.v1",
        "summary": {
            "rows": len(frame),
            "columns": len(frame.columns),
            "numeric_columns": len(numeric_columns),
            "duplicate_rows": duplicate_rows,
            "readiness_score": round(readiness_score, 2),
        },
        "time": time_summary,
        "targets": {
            "configured": list(targets),
            "missing": missing_targets,
        },
        "variables": variable_rows,
        "usable_numeric_features": usable_features,
        "constant_columns": constant_columns,
        "high_missing_columns": high_missing_columns,
        "high_correlation_pairs": correlations,
        "penalties": penalties,
        "risks": risks,
    }
