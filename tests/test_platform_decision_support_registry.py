from __future__ import annotations

import pytest

from industrial_tsfm.platform.decision_support import (
    DecisionPolicy,
    DecisionRule,
    DecisionSupportApplication,
    EvidenceCondition,
    RecommendationScenario,
)
from industrial_tsfm.platform.decision_support_registry import DecisionSupportRegistry


class _ForecastRegistry:
    def __init__(self, *, project_id: str = "demo", source_name: str = "plc") -> None:
        self.project_id = project_id
        self.source_name = source_name
        self.calls = 0

    def status(self, application_id: str) -> dict:
        if application_id != "forecast":
            raise KeyError(application_id)
        return {
            "application_id": application_id,
            "project_id": self.project_id,
            "source_name": self.source_name,
            "runtime_id": "runtime-1",
            "online_training": False,
        }

    def infer_payload(self, application_id: str) -> dict:
        self.status(application_id)
        self.calls += 1
        return {
            "schema_version": "industrial_tsfm.live_forecasting.v1",
            "source_kind": "live",
            "observed_through": "2026-01-01T00:00:10+00:00",
            "forecasts": [
                {"target": "temperature", "step": 1, "value": 95.0},
                {"target": "temperature", "step": 2, "value": 96.0},
            ],
        }


class _DiagnosisRegistry:
    def __init__(self, *, project_id: str = "demo", source_name: str = "plc") -> None:
        self.project_id = project_id
        self.source_name = source_name
        self.calls = 0

    def status(self, application_id: str) -> dict:
        if application_id != "diagnosis":
            raise KeyError(application_id)
        return {
            "application_id": application_id,
            "project_id": self.project_id,
            "source_name": self.source_name,
            "runtime_id": "runtime-1",
            "online_training": False,
            "control_actions_enabled": False,
        }

    def infer(self, application_id: str) -> dict:
        self.status(application_id)
        self.calls += 1
        return {
            "schema_version": "industrial_tsfm.live_diagnosis.v1",
            "source_kind": "live",
            "observed_through": "2026-01-01T00:00:10+00:00",
            "severity": "critical",
            "score": {"ratio_to_threshold": 5.0},
            "event": {"trailing_anomaly_points": 6, "latest_anomaly": True},
        }


def _application() -> DecisionSupportApplication:
    return DecisionSupportApplication(
        DecisionPolicy(
            policy_id="demo-policy",
            scenarios=(
                RecommendationScenario(
                    scenario_id="review-derate",
                    title="Review a derating option",
                    advisory_parameters={"load_reduction_pct": 10},
                ),
            ),
            rules=(
                DecisionRule(
                    rule_id="critical",
                    scenario_id="review-derate",
                    condition=EvidenceCondition(
                        source="diagnosis",
                        metric="severity",
                        operator="gte",
                        value="high",
                    ),
                    points=5.0,
                    rationale="High diagnostic severity requires operator review.",
                ),
            ),
        )
    )


def test_registry_rechecks_binding_and_collects_fresh_evidence() -> None:
    forecast = _ForecastRegistry()
    diagnosis = _DiagnosisRegistry()
    registry = DecisionSupportRegistry()

    status = registry.register(
        application_id="decision",
        project_id="demo",
        source_name="plc",
        runtime_id="runtime-1",
        forecasting_application_id="forecast",
        diagnosis_application_id="diagnosis",
        application=_application(),
        forecasting_registry=forecast,  # type: ignore[arg-type]
        diagnosis_registry=diagnosis,  # type: ignore[arg-type]
    )
    result = registry.evaluate("decision")

    assert status["advisory_only"] is True
    assert status["execution_enabled"] is False
    assert result["top_recommendation"]["scenario_id"] == "review-derate"
    assert result["binding"]["runtime_id"] == "runtime-1"
    assert forecast.calls == 1
    assert diagnosis.calls == 1
    updated = registry.status("decision")
    assert updated["evaluation_count"] == 1
    assert updated["last_top_scenario_id"] == "review-derate"


def test_registry_rejects_cross_project_evidence() -> None:
    registry = DecisionSupportRegistry()

    with pytest.raises(ValueError, match="project"):
        registry.register(
            application_id="decision",
            project_id="demo",
            source_name="plc",
            runtime_id="runtime-1",
            forecasting_application_id="forecast",
            diagnosis_application_id="diagnosis",
            application=_application(),
            forecasting_registry=_ForecastRegistry(project_id="other"),  # type: ignore[arg-type]
            diagnosis_registry=_DiagnosisRegistry(),  # type: ignore[arg-type]
        )


def test_registry_rejects_cross_source_evidence() -> None:
    registry = DecisionSupportRegistry()

    with pytest.raises(ValueError, match="source"):
        registry.register(
            application_id="decision",
            project_id="demo",
            source_name="plc",
            runtime_id="runtime-1",
            forecasting_application_id="forecast",
            diagnosis_application_id="diagnosis",
            application=_application(),
            forecasting_registry=_ForecastRegistry(source_name="other"),  # type: ignore[arg-type]
            diagnosis_registry=_DiagnosisRegistry(),  # type: ignore[arg-type]
        )
