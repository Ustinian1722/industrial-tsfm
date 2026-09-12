from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.services import route_project_forecast


def _project_and_payload(tmp_path):
    rng = np.random.default_rng(23)
    rows = 200
    x = rng.normal(size=rows)
    x[100:] += 3.0
    y = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "x": x,
            "y": y,
        }
    )
    path = tmp_path / "shifted.csv"
    frame.to_csv(path, index=False)
    source = DataSourceSpec(
        name="shifted",
        kind=DataSourceKind.CSV,
        location=str(path),
        timestamp_column="timestamp",
    )
    project = ProjectSpec(
        name="Auto shift demo",
        data_sources=(source,),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
                context_length=16,
                horizon=4,
            ),
        ),
    )
    payload = {
        "validation_evidence": [
            {
                "model": "chronos2",
                "status": "complete",
                "normalized_rmse": 0.42,
                "training_mode": "frozen",
                "inference_seconds": 0.01,
                "total_parameters": 120_000_000,
            }
        ],
        "routing": {
            "auto_shift": True,
            "target_data_fraction": 0.10,
            "support_windows": 20,
        },
        "shift_analysis": {
            "reference_fraction": 0.5,
            "target_fraction": 0.5,
            "min_rows_per_partition": 40,
        },
    }
    return project, payload


def test_auto_shift_feeds_unlabelled_shift_score_into_router(tmp_path) -> None:
    project, payload = _project_and_payload(tmp_path)

    result = route_project_forecast(project, "shifted", "forecast-y", payload)

    assert result["target_labels_used"] is False
    assert result["request"]["shift_score"] > 2.0
    assert result["shift_evidence"]["target_labels_used"] is False
    assert result["shift_evidence"]["top_features"][0]["feature"] == "x"
    assert result["selected_strategy"] == "target_residual_calibration"
    assert "large source-standardized shift" in result["strategy_reason"]


def test_explicit_shift_score_has_priority_over_auto_shift(tmp_path) -> None:
    project, payload = _project_and_payload(tmp_path)
    payload["routing"]["shift_score"] = 0.25

    result = route_project_forecast(project, "shifted", "forecast-y", payload)

    assert result["request"]["shift_score"] == 0.25
    assert result["shift_evidence"] is None
