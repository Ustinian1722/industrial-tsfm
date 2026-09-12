from __future__ import annotations

import json

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    audit_dataframe,
    build_platform_report,
)


def test_data_audit_detects_basic_industrial_data_risks() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="1h"),
            "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "sensor_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "dead_tag": [1.0] * 6,
            "target": [3.0, 3.1, 3.2, 3.3, np.nan, 3.5],
        }
    )
    audit = audit_dataframe(
        frame,
        timestamp_column="timestamp",
        target_columns=("target",),
        correlation_threshold=0.95,
    )

    assert audit["summary"]["rows"] == 6
    assert "dead_tag" in audit["constant_columns"]
    assert "dead_tag" not in audit["usable_numeric_features"]
    assert audit["targets"]["missing"] == []
    assert audit["time"]["valid"] is True
    assert audit["high_correlation_pairs"]
    assert audit["summary"]["readiness_score"] < 100.0
    json.dumps(audit)


def test_platform_report_renders_project_and_data_health(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="10min"),
            "x": np.arange(8, dtype=float),
            "y": np.arange(8, dtype=float) * 0.5,
        }
    )
    project = ProjectSpec(
        name="Process demo",
        data_sources=(
            DataSourceSpec(
                name="fixture",
                kind=DataSourceKind.MEMORY,
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
                context_length=4,
                horizon=2,
            ),
        ),
    )
    audit = audit_dataframe(frame, timestamp_column="timestamp", target_columns=("y",))
    output = build_platform_report(project, audit, tmp_path / "index.html")
    text = output.read_text(encoding="utf-8")

    assert "Industrial Intelligence Platform" in text
    assert "Process demo" in text
    assert "forecast-y" in text
    assert "Data source &amp; health" in text


def test_contracts_reject_invalid_forecasting_task() -> None:
    task = TaskDefinition(name="bad", task_type=TaskType.FORECASTING)
    try:
        task.validate()
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("invalid forecasting task should fail validation")
