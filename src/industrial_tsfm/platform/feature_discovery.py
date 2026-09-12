from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .analytics import LagAnalysisRequest, analyze_lagged_relationships
from .shift_analysis import DistributionShiftRequest, analyze_distribution_shift


@dataclass(frozen=True)
class VariableDiscoveryRequest:
    """Transparent candidate ranking for an industrial forecasting target.

    Candidates are rewarded for historical association and observed coverage,
    then penalized for chronological distribution drift. This is a screening
    heuristic, not causal discovery or model-based feature importance.
    """

    target_column: str
    feature_columns: tuple[str, ...] = ()
    max_lag: int = 12
    min_pairs: int = 24
    top_k: int = 12
    shift_penalty_weight: float = 0.35
    reference_fraction: float = 0.50
    target_fraction: float = 0.50
    min_rows_per_partition: int = 16

    def validate(self) -> None:
        if not self.target_column.strip():
            raise ValueError("target_column must be non-empty")
        if self.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if self.min_pairs < 3:
            raise ValueError("min_pairs must be at least 3")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.shift_penalty_weight < 0.0:
            raise ValueError("shift_penalty_weight must be non-negative")
        DistributionShiftRequest(
            reference_fraction=self.reference_fraction,
            target_fraction=self.target_fraction,
            min_rows_per_partition=self.min_rows_per_partition,
        ).validate()


def _candidate_features(data_audit: dict[str, Any], request: VariableDiscoveryRequest) -> list[str]:
    if request.feature_columns:
        return list(request.feature_columns)
    return [
        str(column)
        for column in data_audit.get("usable_numeric_features", [])
        if str(column) != request.target_column
    ]


def discover_variables(
    frame,
    data_audit: dict[str, Any],
    request: VariableDiscoveryRequest,
) -> dict[str, Any]:
    """Rank candidate inputs using lag association, coverage, and shift stability."""

    request.validate()
    features = _candidate_features(data_audit, request)
    if not features:
        raise ValueError("no candidate feature columns are available")

    lag = analyze_lagged_relationships(
        frame,
        data_audit,
        LagAnalysisRequest(
            target_column=request.target_column,
            feature_columns=tuple(features),
            max_lag=request.max_lag,
            min_pairs=request.min_pairs,
            top_k=len(features),
        ),
    )
    shift = analyze_distribution_shift(
        frame,
        data_audit,
        DistributionShiftRequest(
            feature_columns=tuple(features),
            reference_fraction=request.reference_fraction,
            target_fraction=request.target_fraction,
            min_rows_per_partition=request.min_rows_per_partition,
            top_k=len(features),
        ),
    )

    lag_by_feature = {row["feature"]: row for row in lag["rankings"]}
    shift_by_feature = {row["feature"]: row for row in shift["features"]}
    audit_by_feature = {row["column"]: row for row in data_audit.get("variables", [])}

    rankings: list[dict[str, Any]] = []
    for feature in features:
        lag_row = lag_by_feature.get(feature)
        shift_row = shift_by_feature.get(feature)
        if lag_row is None or shift_row is None:
            continue
        missing_ratio = float(audit_by_feature.get(feature, {}).get("missing_ratio", 0.0))
        coverage = max(0.0, min(1.0, 1.0 - missing_ratio))
        association = float(lag_row["best_abs_pearson"])
        standardized_shift = float(shift_row["standardized_mean_shift"])
        penalty = 1.0 + request.shift_penalty_weight * standardized_shift
        discovery_score = association * coverage / penalty
        rankings.append(
            {
                "feature": feature,
                "discovery_score": float(discovery_score),
                "best_lag": int(lag_row["best_lag"]),
                "best_abs_pearson": association,
                "spearman_at_best_lag": float(lag_row["spearman_at_best_lag"]),
                "coverage": coverage,
                "standardized_mean_shift": standardized_shift,
                "std_ratio": float(shift_row["std_ratio"]),
                "shift_penalty": float(penalty),
            }
        )

    if not rankings:
        raise ValueError("candidate variables do not have enough evidence for discovery")
    rankings.sort(key=lambda row: float(row["discovery_score"]), reverse=True)

    return {
        "schema_version": "industrial_tsfm.variable_discovery.v1",
        "analysis_type": "variable_candidate_discovery",
        "interpretation_boundary": "screening_heuristic_not_causality_or_feature_importance",
        "target_labels_used_for_shift": False,
        "future_feature_values_used_for_lag_association": False,
        "request": asdict(request),
        "target_column": request.target_column,
        "rankings": rankings[: request.top_k],
        "evidence": {
            "lag_analysis_schema": lag["schema_version"],
            "shift_analysis_schema": shift["schema_version"],
            "shift_score": shift["aggregate"]["shift_score"],
        },
    }
