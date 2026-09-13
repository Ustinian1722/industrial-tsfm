from __future__ import annotations

import pytest

from industrial_tsfm.platform.decision_support import (
    DecisionPolicy,
    DecisionRule,
    DecisionSupportApplication,
    EvidenceCondition,
    RecommendationScenario,
    decision_policy_from_dict,
)


def _forecast(*, observed_through: str = "2026-01-01T00:00:10+00:00") -> dict:
    return {
        "schema_version": "industrial_tsfm.live_forecasting.v1",
        "source_kind": "live",
        "observed_through": observed_through,
        "forecasts": [
            {
                "target": "temperature",
                "step": 1,
                "value": 82.0,
                "lower": 79.0,
                "upper": 85.0,
            },
            {
                "target": "temperature",
                "step": 2,
                "value": 87.0,
                "lower": 83.0,
                "upper": 91.0,
            },
        ],
    }


def _diagnosis(*, observed_through: str = "2026-01-01T00:00:09+00:00") -> dict:
    return {
        "schema_version": "industrial_tsfm.live_diagnosis.v1",
        "source_kind": "live",
        "observed_through": observed_through,
        "severity": "high",
        "score": {"ratio_to_threshold": 2.7},
        "event": {"trailing_anomaly_points": 4, "latest_anomaly": True},
    }


def _policy() -> DecisionPolicy:
    return DecisionPolicy(
        policy_id="temperature-triage-v1",
        scenarios=(
            RecommendationScenario(
                scenario_id="maintain",
                title="Maintain current operating plan",
                advisory_parameters={"load_reduction_pct": 0},
            ),
            RecommendationScenario(
                scenario_id="derate-10",
                title="Consider a 10% load reduction",
                advisory_parameters={"load_reduction_pct": 10},
                guardrails=(
                    EvidenceCondition(
                        source="diagnosis",
                        metric="severity",
                        operator="gte",
                        value="warning",
                    ),
                ),
            ),
        ),
        rules=(
            DecisionRule(
                rule_id="high-diagnosis",
                scenario_id="derate-10",
                condition=EvidenceCondition(
                    source="diagnosis",
                    metric="severity",
                    operator="gte",
                    value="high",
                ),
                points=3.0,
                rationale="Diagnostic severity is high or critical.",
            ),
            DecisionRule(
                rule_id="temperature-upper",
                scenario_id="derate-10",
                condition=EvidenceCondition(
                    source="forecast",
                    metric="upper_max",
                    target="temperature",
                    operator="gt",
                    value=90.0,
                ),
                points=2.0,
                rationale="Forecast upper interval exceeds the review threshold.",
            ),
            DecisionRule(
                rule_id="normal-maintain",
                scenario_id="maintain",
                condition=EvidenceCondition(
                    source="diagnosis",
                    metric="severity",
                    operator="eq",
                    value="normal",
                ),
                points=2.0,
                rationale="No diagnostic escalation is present.",
            ),
        ),
    )


def test_decision_support_ranks_advisory_scenario_without_execution() -> None:
    result = DecisionSupportApplication(_policy()).evaluate(_forecast(), _diagnosis())

    assert result["status"] == "recommendation_available"
    assert result["top_recommendation"]["scenario_id"] == "derate-10"
    assert result["top_recommendation"]["score"] == 5.0
    assert result["advisory_only"] is True
    assert result["operator_confirmation_required"] is True
    assert result["execution_enabled"] is False
    assert result["control_actions_enabled"] is False
    assert result["opcua_writes_enabled"] is False
    assert result["llm_actions_enabled"] is False
    assert result["scenario_evaluation_scope"].endswith("not_process_simulation")
    assert result["claim_scope"] == "advisory_evidence_ranking_not_causal_or_control"
    assert result["evidence"]["observation_skew_seconds"] == 1.0


def test_decision_support_refuses_temporally_misaligned_evidence() -> None:
    app = DecisionSupportApplication(_policy())

    with pytest.raises(RuntimeError, match="temporally misaligned"):
        app.evaluate(
            _forecast(observed_through="2026-01-01T00:05:00+00:00"),
            _diagnosis(observed_through="2026-01-01T00:00:00+00:00"),
        )


def test_missing_forecast_interval_evidence_does_not_silently_match() -> None:
    policy = DecisionPolicy(
        policy_id="interval-required",
        scenarios=(
            RecommendationScenario(
                scenario_id="review",
                title="Review operation",
                guardrails=(
                    EvidenceCondition(
                        source="forecast",
                        metric="upper_max",
                        target="temperature",
                        operator="gt",
                        value=90.0,
                    ),
                ),
            ),
        ),
        rules=(),
        minimum_recommendation_score=0.0,
    )
    forecast = _forecast()
    for point in forecast["forecasts"]:
        point.pop("upper")
    result = DecisionSupportApplication(policy).evaluate(forecast, _diagnosis())

    assert result["status"] == "operator_review_required"
    evaluation = result["scenario_evaluations"][0]
    assert evaluation["eligible"] is False
    assert evaluation["guardrails"][0]["available"] is False


def test_policy_parser_rejects_negative_rule_points() -> None:
    payload = _policy().to_dict()
    payload["rules"][0]["points"] = -1

    with pytest.raises(ValueError, match="non-negative"):
        decision_policy_from_dict(payload)
