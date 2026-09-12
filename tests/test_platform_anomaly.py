from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    AnomalyDetectionRequest,
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    audit_dataframe,
    run_pca_spe_anomaly_detection,
)


def test_pca_spe_anomaly_engine_uses_train_only_threshold_and_ranks_sensors() -> None:
    rng = np.random.default_rng(4)
    rows = 120
    x = rng.normal(0.0, 1.0, rows)
    y = 0.5 * x + rng.normal(0.0, 0.1, rows)
    z = rng.normal(0.0, 0.2, rows)
    x[95:105] += 8.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="1min"),
            "sensor_x": x,
            "sensor_y": y,
            "sensor_z": z,
        }
    )
    project = ProjectSpec(
        name="Anomaly demo",
        data_sources=(
            DataSourceSpec(
                name="memory",
                kind=DataSourceKind.MEMORY,
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="process-anomaly",
                task_type=TaskType.ANOMALY,
                constraints={"feature_columns": ["sensor_x", "sensor_y", "sensor_z"]},
            ),
        ),
    )
    audit = audit_dataframe(frame, timestamp_column="timestamp")

    result = run_pca_spe_anomaly_detection(
        project,
        frame,
        audit,
        AnomalyDetectionRequest(
            task_name="process-anomaly",
            train_fraction=0.60,
            threshold_quantile=0.99,
            min_train_rows=24,
        ),
    )

    assert result["status"] == "complete"
    assert result["model"] == "pca_spe_product"
    assert result["target_labels_used"] is False
    assert result["fit_scope"] == "normal_training_prefix_only"
    assert result["threshold"]["fit_rows"] == result["train_rows"]
    assert result["summary"]["anomaly_points_test"] > 0
    top_two = {row["sensor"] for row in result["top_sensors"][:2]}
    assert "sensor_x" in top_two
    assert len(result["point_scores"]) == rows
    assert len(result["anomaly_flags"]) == rows


def test_anomaly_engine_rejects_wrong_task_type() -> None:
    frame = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)})
    project = ProjectSpec(
        name="wrong-task",
        data_sources=(DataSourceSpec(name="memory", kind=DataSourceKind.MEMORY),),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
            ),
        ),
    )
    audit = audit_dataframe(frame, target_columns=("y",))

    try:
        run_pca_spe_anomaly_detection(
            project,
            frame,
            audit,
            AnomalyDetectionRequest(task_name="forecast-y", min_train_rows=8),
        )
    except ValueError as exc:
        assert "anomaly_detection" in str(exc)
    else:
        raise AssertionError("forecasting task must not be accepted by anomaly engine")
