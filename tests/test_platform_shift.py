from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform.data_audit import audit_dataframe
from industrial_tsfm.platform.shift_analysis import (
    DistributionShiftRequest,
    analyze_distribution_shift,
)


def test_distribution_shift_detects_late_covariate_drift_and_excludes_targets() -> None:
    rng = np.random.default_rng(17)
    rows = 240
    feature = rng.normal(0.0, 1.0, rows)
    feature[120:] += 2.5
    stable = rng.normal(0.0, 1.0, rows)
    label = 0.7 * feature + rng.normal(0.0, 0.1, rows)
    frame = pd.DataFrame({"feature": feature, "stable": stable, "label": label})
    audit = audit_dataframe(frame, target_columns=("label",))

    result = analyze_distribution_shift(
        frame,
        audit,
        DistributionShiftRequest(
            reference_fraction=0.5,
            target_fraction=0.5,
            min_rows_per_partition=40,
        ),
    )

    assert result["target_labels_used"] is False
    assert result["future_information_used_for_reference_statistics"] is False
    assert result["partition"]["gap_rows"] == 0
    assert result["features"][0]["feature"] == "feature"
    assert {row["feature"] for row in result["features"]} == {"feature", "stable"}
    assert result["aggregate"]["rms_standardized_mean_shift"] > 1.5
    assert result["aggregate"]["shift_score"] == result["aggregate"][
        "rms_standardized_mean_shift"
    ]


def test_distribution_shift_rejects_overlapping_partitions() -> None:
    frame = pd.DataFrame({"x": np.arange(40, dtype=float)})
    audit = audit_dataframe(frame)

    try:
        analyze_distribution_shift(
            frame,
            audit,
            DistributionShiftRequest(reference_fraction=0.7, target_fraction=0.5),
        )
    except ValueError as exc:
        assert "must not exceed 1" in str(exc)
    else:
        raise AssertionError("overlapping chronological partitions must fail")
