from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    CrossSourceShiftRequest,
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    compare_source_distributions,
)
from industrial_tsfm.platform.data_audit import audit_dataframe
from industrial_tsfm.platform.services import run_project_cross_source_shift


def test_cross_source_shift_uses_reference_statistics_without_target_labels() -> None:
    rng = np.random.default_rng(41)
    reference = pd.DataFrame(
        {
            "x": rng.normal(0.0, 1.0, 120),
            "label": rng.normal(size=120),
        }
    )
    target = pd.DataFrame({"x": rng.normal(2.5, 1.0, 100)})
    reference_audit = audit_dataframe(reference, target_columns=("label",))
    target_audit = audit_dataframe(target, target_columns=("label",))

    result = compare_source_distributions(
        reference,
        target,
        reference_audit,
        target_audit,
        CrossSourceShiftRequest(min_rows_per_source=40),
    )

    assert result["target_labels_used"] is False
    assert result["reference_statistics_use_target_data"] is False
    assert result["features"][0]["feature"] == "x"
    assert result["aggregate"]["rms_standardized_mean_shift"] > 2.0


def test_project_cross_source_shift_allows_unlabelled_target_source(tmp_path) -> None:
    rng = np.random.default_rng(43)
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=80, freq="10min"),
            "x": rng.normal(0.0, 1.0, 80),
            "label": rng.normal(size=80),
        }
    ).to_csv(reference_path, index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-01", periods=80, freq="10min"),
            "x": rng.normal(1.8, 1.0, 80),
        }
    ).to_csv(target_path, index=False)

    project = ProjectSpec(
        name="Cross-source demo",
        data_sources=(
            DataSourceSpec(
                name="reference",
                kind=DataSourceKind.CSV,
                location=str(reference_path),
                timestamp_column="timestamp",
            ),
            DataSourceSpec(
                name="target",
                kind=DataSourceKind.CSV,
                location=str(target_path),
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-label",
                task_type=TaskType.FORECASTING,
                target_columns=("label",),
                context_length=8,
                horizon=2,
            ),
        ),
    )

    result = run_project_cross_source_shift(
        project,
        "reference",
        "target",
        {"analysis": {"min_rows_per_source": 30}},
    )

    assert result["reference_source_name"] == "reference"
    assert result["target_source_name"] == "target"
    assert result["target_labels_used"] is False
    assert result["features"][0]["feature"] == "x"
