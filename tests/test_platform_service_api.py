from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.api import create_app


def _project_and_validation(tmp_path):
    rng = np.random.default_rng(2)
    rows = 80
    x = rng.normal(size=rows)
    y = 2.0 * x + rng.normal(scale=0.1, size=rows)
    x[68:74] += 7.0
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "x": x,
            "y": y,
        }
    )
    data_path = tmp_path / "process.csv"
    frame.to_csv(data_path, index=False)
    project = ProjectSpec(
        name="End to End API Demo",
        data_sources=(
            DataSourceSpec(
                name="process",
                kind=DataSourceKind.CSV,
                location=str(data_path),
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
                context_length=8,
                horizon=2,
            ),
            TaskDefinition(
                name="process-anomaly",
                task_type=TaskType.ANOMALY,
                constraints={"feature_columns": ["x", "y"]},
            ),
        ),
    )
    validation = [
        {
            "model": "chronos2",
            "seed": 0,
            "status": "complete",
            "normalized_rmse": 0.4,
            "fit_seconds": 0.0,
            "inference_seconds": 0.02,
            "total_parameters": 120_000_000,
            "training_mode": "frozen",
        },
        {
            "model": "patchtst",
            "seed": 0,
            "status": "complete",
            "normalized_rmse": 0.6,
            "fit_seconds": 1.0,
            "inference_seconds": 0.005,
            "total_parameters": 70_000,
            "training_mode": "supervised",
        },
    ]
    return project, validation


def test_api_routes_materializes_and_runs_anomaly_task(tmp_path) -> None:
    project, validation = _project_and_validation(tmp_path)
    client = TestClient(
        create_app(tmp_path / "results", tmp_path / "workspace")
    )
    created = client.post("/v1/projects", json=project.to_dict())
    project_id = created.json()["project_id"]
    forecasting_payload = {
        "source_name": "process",
        "validation_evidence": validation,
        "routing": {
            "target_data_fraction": 0.1,
            "support_windows": 8,
            "max_parameters": 200_000_000,
        },
        "replay_batch_size": 16,
    }

    route = client.post(
        f"/v1/projects/{project_id}/route/forecast-y",
        json=forecasting_payload,
    )
    application = client.post(
        f"/v1/projects/{project_id}/applications/forecast-y",
        json=forecasting_payload,
    )
    anomaly = client.post(
        f"/v1/projects/{project_id}/anomaly/process-anomaly",
        json={
            "source_name": "process",
            "anomaly": {
                "train_fraction": 0.6,
                "threshold_quantile": 0.99,
                "min_train_rows": 24,
            },
        },
    )
    applications = client.get("/v1/applications")

    assert route.status_code == 200
    assert route.json()["selected_model"] == "chronos2"
    assert route.json()["selection_evidence"] == "validation_only"
    assert route.json()["target_labels_used"] is False
    assert application.status_code == 201
    assert application.json()["status"] == "ready"
    assert applications.json()["count"] == 1
    assert anomaly.status_code == 200
    assert anomaly.json()["model"] == "pca_spe_product"
    assert anomaly.json()["target_labels_used"] is False
    assert anomaly.json()["summary"]["anomaly_points_test"] > 0
