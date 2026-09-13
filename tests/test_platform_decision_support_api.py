from __future__ import annotations

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
from industrial_tsfm.platform.opcua_api import create_live_app
from industrial_tsfm.platform.workspace import WorkspaceStore


class _ForecastRegistry:
    def status(self, application_id: str) -> dict:
        if application_id != "forecast-live":
            raise KeyError(application_id)
        return {
            "application_id": application_id,
            "project_id": "demo",
            "source_name": "plc",
            "runtime_id": "runtime-1",
            "online_training": False,
        }

    def infer_payload(self, application_id: str) -> dict:
        self.status(application_id)
        return {
            "schema_version": "industrial_tsfm.live_forecasting.v1",
            "source_kind": "live",
            "observed_through": "2026-01-01T00:00:10+00:00",
            "forecasts": [
                {"target": "temperature", "step": 1, "value": 92.0},
                {"target": "temperature", "step": 2, "value": 94.0},
            ],
        }

    def list(self) -> list[dict]:
        return [self.status("forecast-live")]

    def remove(self, application_id: str) -> None:
        self.status(application_id)


class _DiagnosisRegistry:
    def status(self, application_id: str) -> dict:
        if application_id != "diagnosis-live":
            raise KeyError(application_id)
        return {
            "application_id": application_id,
            "project_id": "demo",
            "source_name": "plc",
            "runtime_id": "runtime-1",
            "online_training": False,
            "control_actions_enabled": False,
        }

    def infer(self, application_id: str) -> dict:
        self.status(application_id)
        return {
            "schema_version": "industrial_tsfm.live_diagnosis.v1",
            "source_kind": "live",
            "observed_through": "2026-01-01T00:00:10+00:00",
            "severity": "high",
            "score": {"ratio_to_threshold": 2.5},
            "event": {"trailing_anomaly_points": 4, "latest_anomaly": True},
        }

    def list(self) -> list[dict]:
        return [self.status("diagnosis-live")]

    def remove(self, application_id: str) -> None:
        self.status(application_id)


def _workspace(root) -> WorkspaceStore:
    workspace = WorkspaceStore(root)
    workspace.create_project(
        ProjectSpec(
            name="Demo",
            data_sources=(
                DataSourceSpec(
                    name="plc",
                    kind=DataSourceKind.OPCUA,
                    location="opc.tcp://127.0.0.1:4840/demo",
                ),
            ),
            tasks=(
                TaskDefinition(
                    name="detect-process",
                    task_type=TaskType.ANOMALY,
                    constraints={"feature_columns": ["temperature"]},
                ),
            ),
        ),
        project_id="demo",
    )
    return workspace


def _registration_payload() -> dict:
    return {
        "application_id": "decision-live",
        "source_name": "plc",
        "runtime_id": "runtime-1",
        "forecasting_application_id": "forecast-live",
        "diagnosis_application_id": "diagnosis-live",
        "policy": {
            "policy_id": "operator-triage-v1",
            "minimum_recommendation_score": 1.0,
            "max_observation_skew_seconds": 30.0,
            "scenarios": [
                {
                    "scenario_id": "review-derate",
                    "title": "Review a 10% load reduction",
                    "advisory_parameters": {"load_reduction_pct": 10},
                }
            ],
            "rules": [
                {
                    "rule_id": "high-severity",
                    "scenario_id": "review-derate",
                    "condition": {
                        "source": "diagnosis",
                        "metric": "severity",
                        "operator": "gte",
                        "value": "high",
                    },
                    "points": 3.0,
                    "rationale": "High diagnostic severity warrants operator review.",
                },
                {
                    "rule_id": "temperature-high",
                    "scenario_id": "review-derate",
                    "condition": {
                        "source": "forecast",
                        "metric": "max",
                        "target": "temperature",
                        "operator": "gt",
                        "value": 90.0,
                    },
                    "points": 2.0,
                    "rationale": "Forecast temperature exceeds the review threshold.",
                },
            ],
        },
    }


def test_live_app_exposes_advisory_decision_support_and_persists_policy(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace = _workspace(workspace_root)
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=workspace_root,
        forecasting_registry=_ForecastRegistry(),  # type: ignore[arg-type]
        diagnosis_registry=_DiagnosisRegistry(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    capabilities = client.get("/v1/decision-support/capabilities")
    registered = client.post(
        "/v1/projects/demo/decision-support/applications",
        json=_registration_payload(),
    )
    evaluated = client.post("/v1/decision-support/applications/decision-live/evaluate")

    assert capabilities.status_code == 200
    caps = capabilities.json()
    assert caps["advisory_only"] is True
    assert caps["operator_confirmation_required"] is True
    assert caps["execution_enabled"] is False
    assert caps["control_actions_enabled"] is False
    assert caps["opcua_writes_enabled"] is False
    assert caps["llm_actions_enabled"] is False
    assert registered.status_code == 201
    assert registered.json()["policy"]["policy_id"] == "operator-triage-v1"
    assert evaluated.status_code == 200
    result = evaluated.json()
    assert result["top_recommendation"]["scenario_id"] == "review-derate"
    assert result["execution_enabled"] is False
    persisted = workspace.read_derived_artifact("demo", "decision-support-policy-decision-live")
    assert persisted["advisory_only"] is True
    assert persisted["control_actions_enabled"] is False
    assert persisted["policy"]["policy_id"] == "operator-triage-v1"


def test_decision_support_api_rejects_unknown_evidence_binding(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    _workspace(workspace_root)
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=workspace_root,
        forecasting_registry=_ForecastRegistry(),  # type: ignore[arg-type]
        diagnosis_registry=_DiagnosisRegistry(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    payload = _registration_payload()
    payload["forecasting_application_id"] = "missing"

    response = client.post("/v1/projects/demo/decision-support/applications", json=payload)

    assert response.status_code == 404


def test_decision_support_api_returns_404_for_unknown_application(tmp_path) -> None:
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.post("/v1/decision-support/applications/missing/evaluate")

    assert response.status_code == 404
