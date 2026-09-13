from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from industrial_tsfm.platform.contracts import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.diagnosis import DiagnosisRequest, LiveDiagnosisApplication
from industrial_tsfm.platform.diagnosis_registry import LiveDiagnosisRegistry
from industrial_tsfm.platform.online_anomaly import fit_pca_spe_artifact
from industrial_tsfm.platform.opcua_api import create_live_app
from industrial_tsfm.platform.runtime import DataReplayRuntime, ReplayConfig
from industrial_tsfm.platform.workspace import WorkspaceStore


def _registry() -> LiveDiagnosisRegistry:
    x = np.linspace(-1.0, 1.0, 40)
    y = 0.5 * x + 0.03 * np.sin(np.arange(40, dtype=float))
    artifact = fit_pca_spe_artifact(
        pd.DataFrame({"x": x, "y": y}),
        ("x", "y"),
        threshold_quantile=0.95,
        n_components=1,
        min_train_rows=20,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="s", tz="UTC").astype(str),
            "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7.0],
            "y": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        }
    )
    replay = DataReplayRuntime(frame, ReplayConfig(timestamp_column="timestamp"))
    application = LiveDiagnosisApplication(
        artifact,
        DiagnosisRequest(context_observations=8, top_k_sensors=2),
    )
    registry = LiveDiagnosisRegistry()
    registry.register(
        application_id="demo-diagnosis",
        runtime_id="replay-demo",
        project_id="demo",
        source_name="demo-source",
        task_name="detect-process",
        application=application,
        provider=replay,
    )
    return registry


def _workspace_with_anomaly_task(root) -> WorkspaceStore:
    workspace = WorkspaceStore(root)
    project = ProjectSpec(
        name="Demo",
        data_sources=(
            DataSourceSpec(
                name="plc",
                kind=DataSourceKind.OPCUA,
                location="opc.tcp://127.0.0.1:4840/demo",
                metadata={
                    "node_ids": {
                        "temperature": "ns=2;s=temperature",
                        "pressure": "ns=2;s=pressure",
                    }
                },
            ),
        ),
        tasks=(
            TaskDefinition(
                name="detect-process",
                task_type=TaskType.ANOMALY,
                target_columns=(),
                constraints={"feature_columns": ["temperature", "pressure"]},
            ),
        ),
    )
    workspace.create_project(project, project_id="demo")
    return workspace


def test_live_app_exposes_inference_only_diagnosis_registry(tmp_path) -> None:
    registry = _registry()
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=tmp_path / "workspace",
        diagnosis_registry=registry,
    )
    client = TestClient(app)

    capabilities = client.get("/v1/diagnosis/capabilities")
    listing = client.get("/v1/diagnosis/applications")
    status = client.get("/v1/diagnosis/applications/demo-diagnosis")
    inferred = client.post("/v1/diagnosis/applications/demo-diagnosis/infer")

    assert capabilities.status_code == 200
    assert capabilities.json()["http_artifact_registration"] is False
    assert capabilities.json()["causal_root_cause_claim"] is False
    assert capabilities.json()["control_actions_enabled"] is False
    assert "build_deployment_spec" in capabilities.json()["operations"]
    assert listing.json()["count"] == 1
    assert status.json()["application_id"] == "demo-diagnosis"
    assert inferred.status_code == 200
    assert inferred.json()["status"] == "alert"
    assert inferred.json()["claim_scope"].endswith("not_causal_root_cause")
    assert registry.status("demo-diagnosis")["inference_count"] == 1


def test_diagnosis_deployment_endpoint_persists_offline_contract(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace = _workspace_with_anomaly_task(workspace_root)
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=workspace_root,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/projects/demo/diagnosis/detect-process/deployment",
        json={
            "source_name": "plc",
            "context_observations": 32,
            "top_k_sensors": 2,
            "artifact_ref": "workspace://demo/pca-spe.json",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["project_id"] == "demo"
    assert payload["feature_columns"] == ["temperature", "pressure"]
    assert payload["fit_scope"] == "offline_training_or_calibration_only"
    assert payload["target_labels_used"] is False
    assert payload["online_training"] is False
    assert payload["causal_root_cause_claim"] is False
    assert payload["control_actions_enabled"] is False
    persisted = workspace.read_derived_artifact("demo", "diagnosis-deployment-detect-process")
    assert persisted["deployment_id"] == payload["deployment_id"]
    assert persisted["artifact_ref"] == "workspace://demo/pca-spe.json"


def test_live_diagnosis_api_returns_404_for_unknown_application(tmp_path) -> None:
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.post("/v1/diagnosis/applications/missing/infer")

    assert response.status_code == 404
