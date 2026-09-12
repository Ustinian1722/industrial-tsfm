from __future__ import annotations

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


def test_workspace_api_creates_project_and_audits_csv_source(tmp_path) -> None:
    data_path = tmp_path / "process.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="10min"),
            "x": [float(value) for value in range(12)],
            "y": [float(value * 2) for value in range(12)],
        }
    ).to_csv(data_path, index=False)
    project = ProjectSpec(
        name="Workspace API Demo",
        data_sources=(
            DataSourceSpec(
                name="process-csv",
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
                context_length=4,
                horizon=2,
            ),
        ),
    )
    client = TestClient(
        create_app(tmp_path / "results", tmp_path / "workspace")
    )

    created = client.post("/v1/projects", json=project.to_dict())
    listing = client.get("/v1/projects")
    project_id = created.json()["project_id"]
    audit = client.post(f"/v1/projects/{project_id}/audit/process-csv")

    assert created.status_code == 201
    assert listing.status_code == 200
    assert listing.json()["count"] == 1
    assert audit.status_code == 200
    assert audit.json()["provenance"]["rows"] == 12
    assert audit.json()["audit"]["targets"]["missing"] == []
    assert (
        tmp_path
        / "workspace"
        / "projects"
        / project_id
        / "derived"
        / "data-audit-process-csv.json"
    ).exists()


def test_workspace_api_rejects_duplicate_project(tmp_path) -> None:
    project = ProjectSpec(
        name="Duplicate Demo",
        data_sources=(DataSourceSpec(name="memory", kind=DataSourceKind.MEMORY),),
        tasks=(
            TaskDefinition(
                name="anomaly",
                task_type=TaskType.ANOMALY,
            ),
        ),
    )
    client = TestClient(create_app(tmp_path / "results", tmp_path / "workspace"))

    first = client.post("/v1/projects", json=project.to_dict())
    second = client.post("/v1/projects", json=project.to_dict())

    assert first.status_code == 201
    assert second.status_code == 409
