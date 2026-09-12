from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform.data_audit import audit_dataframe
from industrial_tsfm.platform.feature_discovery import (
    VariableDiscoveryRequest,
    discover_variables,
)


def test_variable_discovery_ranks_stable_historical_driver_first() -> None:
    rng = np.random.default_rng(31)
    rows = 180
    driver = rng.normal(size=rows)
    nuisance = rng.normal(size=rows)
    target = np.zeros(rows)
    target[3:] = 1.8 * driver[:-3] + rng.normal(scale=0.05, size=rows - 3)
    frame = pd.DataFrame({"driver": driver, "nuisance": nuisance, "target": target})
    audit = audit_dataframe(frame, target_columns=("target",))

    result = discover_variables(
        frame,
        audit,
        VariableDiscoveryRequest(
            target_column="target",
            max_lag=6,
            min_pairs=40,
            top_k=2,
            min_rows_per_partition=40,
        ),
    )

    assert result["interpretation_boundary"] == (
        "screening_heuristic_not_causality_or_feature_importance"
    )
    assert result["target_labels_used_for_shift"] is False
    assert result["future_feature_values_used_for_lag_association"] is False
    assert result["rankings"][0]["feature"] == "driver"
    assert result["rankings"][0]["best_lag"] == 3
    assert result["rankings"][0]["best_abs_pearson"] > 0.95
    assert result["rankings"][0]["discovery_score"] > result["rankings"][1][
        "discovery_score"
    ]


def test_variable_discovery_penalizes_shift_transparently() -> None:
    rng = np.random.default_rng(37)
    rows = 200
    stable = rng.normal(size=rows)
    shifted = stable.copy()
    shifted[100:] += 4.0
    target = 0.8 * stable + rng.normal(scale=0.1, size=rows)
    frame = pd.DataFrame({"stable": stable, "shifted": shifted, "target": target})
    audit = audit_dataframe(frame, target_columns=("target",))

    result = discover_variables(
        frame,
        audit,
        VariableDiscoveryRequest(
            target_column="target",
            feature_columns=("stable", "shifted"),
            max_lag=0,
            min_pairs=50,
            top_k=2,
            shift_penalty_weight=0.5,
            min_rows_per_partition=50,
        ),
    )

    rows_by_feature = {row["feature"]: row for row in result["rankings"]}
    assert rows_by_feature["shifted"]["standardized_mean_shift"] > 3.0
    assert rows_by_feature["shifted"]["shift_penalty"] > rows_by_feature["stable"][
        "shift_penalty"
    ]
    assert result["rankings"][0]["feature"] == "stable"
