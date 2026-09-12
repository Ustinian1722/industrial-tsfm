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


def test_lag_analysis_api_persists_association_artifact(tmp_path) -> None:
    rng = np.random.default_rng(14)
    rows = 96
    driver = rng.normal(size=rows)
    target = np.zeros(rows)
    target[4:] = 1.8 * driver[:-4] + rng.normal(scale=0.04, size=rows - 4)
    path = tmp_path / "lag.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min"),
            "driver": driver,
            "target": target,
        }
    ).to_csv(path, index=False)
    project = ProjectSpec(
        name="Lag API Demo",
        data_sources=(
            DataSourceSpec(
                name="process",
                kind=DataSourceKind.CSV,
                location=str(path),
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-target",
                task_type=TaskType.FORECASTING,
                target_columns=("target",),
                context_length=16,
                horizon=4,
            ),
        ),
    )
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(tmp_path / "results", workspace))
    created = client.post("/v1/projects", json=project.to_dict())
    project_id = created.json()["project_id"]

    response = client.post(
        f"/v1/projects/{project_id}/analysis/lag",
        json={
            "source_name": "process",
            "analysis": {
                "target_column": "target",
                "feature_columns": ["driver"],
                "max_lag": 8,
                "min_pairs": 40,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["future_information_used"] is False
    assert payload["interpretation_boundary"] == "association_not_causality"
    assert payload["rankings"][0]["feature"] == "driver"
    assert payload["rankings"][0]["best_lag"] == 4
    artifact = (
        workspace
        / "projects"
        / project_id
        / "derived"
        / "lag-analysis.json"
    )
    assert artifact.exists()


def test_capabilities_expose_lagged_association_tool(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "results", tmp_path / "workspace"))
    response = client.get("/v1/capabilities")
    assert response.status_code == 200
    assert "lagged_association" in response.json()["analytics"]
