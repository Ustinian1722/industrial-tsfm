from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProductRoutingRequest,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    build_platform_application,
)
from industrial_tsfm.platform.api import create_app


def _materialize_application(root) -> str:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=16, freq="5min"),
            "x": [float(value) for value in range(16)],
            "y": [float(value * 2) for value in range(16)],
        }
    )
    source = DataSourceSpec(
        name="memory-source",
        kind=DataSourceKind.MEMORY,
        timestamp_column="timestamp",
    )
    project = ProjectSpec(
        name="API Demo",
        data_sources=(source,),
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
    validation = pd.DataFrame(
        [
            {
                "model": "chronos2",
                "seed": 0,
                "status": "complete",
                "normalized_rmse": 0.4,
                "fit_seconds": 0.0,
                "inference_seconds": 0.02,
                "total_parameters": 120_000_000,
                "training_mode": "frozen",
            }
        ]
    )
    application, _ = build_platform_application(
        project,
        source_name=source.name,
        task_name="forecast-y",
        validation_table=validation,
        routing_request=ProductRoutingRequest(task_name="forecast-y"),
        memory_frames={source.name: frame},
        replay_batch_size=4,
    )
    application.write(root / "api-demo")
    return application.application_id


def test_platform_api_exposes_capabilities_and_application_artifacts(tmp_path) -> None:
    application_id = _materialize_application(tmp_path)
    client = TestClient(create_app(tmp_path))

    health = client.get("/health")
    capabilities = client.get("/v1/capabilities")
    applications = client.get("/v1/applications")
    detail = client.get(f"/v1/applications/{application_id}")
    route = client.get(f"/v1/applications/{application_id}/route")
    audit = client.get(f"/v1/applications/{application_id}/audit")
    replay = client.get(f"/v1/applications/{application_id}/replay")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert capabilities.status_code == 200
    assert "chronos2" in {model["name"] for model in capabilities.json()["models"]}
    assert applications.json()["count"] == 1
    assert detail.json()["application_id"] == application_id
    assert route.json()["selection_evidence"] == "validation_only"
    assert route.json()["target_labels_used"] is False
    assert audit.json()["summary"]["rows"] == 16
    assert replay.json()["total_batches"] == 4


def test_platform_api_returns_404_for_unknown_application(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.get("/v1/applications/does-not-exist")
    assert response.status_code == 404
