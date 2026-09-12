from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform.analytics import LagAnalysisRequest, analyze_lagged_relationships
from industrial_tsfm.platform.data_audit import audit_dataframe


def test_lag_analysis_recovers_historical_driver_without_future_shift() -> None:
    rng = np.random.default_rng(11)
    rows = 140
    driver = rng.normal(size=rows)
    target = np.zeros(rows)
    target[3:] = 2.5 * driver[:-3] + rng.normal(scale=0.05, size=rows - 3)
    nuisance = rng.normal(size=rows)
    frame = pd.DataFrame({"driver": driver, "nuisance": nuisance, "target": target})
    audit = audit_dataframe(frame, target_columns=("target",))

    result = analyze_lagged_relationships(
        frame,
        audit,
        LagAnalysisRequest(target_column="target", max_lag=6, min_pairs=40),
    )

    assert result["future_information_used"] is False
    assert result["interpretation_boundary"] == "association_not_causality"
    assert result["rankings"][0]["feature"] == "driver"
    assert result["rankings"][0]["best_lag"] == 3
    assert result["rankings"][0]["best_abs_pearson"] > 0.95


def test_lag_analysis_rejects_missing_target() -> None:
    frame = pd.DataFrame({"x": np.arange(20, dtype=float)})
    audit = audit_dataframe(frame)
    try:
        analyze_lagged_relationships(
            frame,
            audit,
            LagAnalysisRequest(target_column="missing", max_lag=2),
        )
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing target must fail")
