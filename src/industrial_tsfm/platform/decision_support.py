from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


_SEVERITY_RANK = {"normal": 0, "warning": 1, "high": 2, "critical": 3}
_ALLOWED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte"}
_ALLOWED_FORECAST_METRICS = {"min", "max", "mean", "last", "lower_min", "upper_max"}
_ALLOWED_DIAGNOSIS_METRICS = {
    "severity",
    "ratio_to_threshold",
    "trailing_anomaly_points",
    "latest_anomaly",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "lte":
        return left <= right
    raise ValueError(f"unsupported operator {operator!r}")


@dataclass(frozen=True)
class EvidenceCondition:
    """One transparent condition over forecast or diagnosis evidence."""

    source: str
    metric: str
    operator: str
    value: Any
    target: str | None = None

    def validate(self) -> None:
        if self.source not in {"forecast", "diagnosis"}:
            raise ValueError("condition source must be forecast or diagnosis")
        if self.operator not in _ALLOWED_OPERATORS:
            raise ValueError(f"unsupported condition operator {self.operator!r}")
        if self.source == "forecast":
            if self.metric not in _ALLOWED_FORECAST_METRICS:
                raise ValueError(f"unsupported forecast metric {self.metric!r}")
            if not str(self.target or "").strip():
                raise ValueError("forecast conditions require target")
            _finite_float(self.value, label="forecast condition value")
        else:
            if self.metric not in _ALLOWED_DIAGNOSIS_METRICS:
                raise ValueError(f"unsupported diagnosis metric {self.metric!r}")
            if self.target is not None:
                raise ValueError("diagnosis conditions do not accept target")
            if self.metric == "severity":
                if str(self.value) not in _SEVERITY_RANK:
                    raise ValueError("severity condition value must be normal/warning/high/critical")
            elif self.metric == "latest_anomaly":
                if not isinstance(self.value, bool):
                    raise ValueError("latest_anomaly condition value must be boolean")
                if self.operator not in {"eq", "ne"}:
                    raise ValueError("latest_anomaly supports eq/ne only")
            else:
                _finite_float(self.value, label="diagnosis condition value")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RecommendationScenario:
    """Operator-visible advisory scenario; it never maps to a control write."""

    scenario_id: str
    title: str
    description: str = ""
    advisory_parameters: dict[str, Any] = field(default_factory=dict)
    base_priority: float = 0.0
    guardrails: tuple[EvidenceCondition, ...] = ()

    def validate(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not self.title.strip():
            raise ValueError("scenario title must not be empty")
        _finite_float(self.base_priority, label="scenario base_priority")
        for condition in self.guardrails:
            condition.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["guardrails"] = [condition.to_dict() for condition in self.guardrails]
        payload["operator_confirmation_required"] = True
        payload["execution_enabled"] = False
        return payload


@dataclass(frozen=True)
class DecisionRule:
    rule_id: str
    scenario_id: str
    condition: EvidenceCondition
    points: float
    rationale: str

    def validate(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id must not be empty")
        if not self.scenario_id.strip():
            raise ValueError("rule scenario_id must not be empty")
        self.condition.validate()
        points = _finite_float(self.points, label="rule points")
        if points < 0.0:
            raise ValueError("rule points must be non-negative")
        if not self.rationale.strip():
            raise ValueError("rule rationale must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["condition"] = self.condition.to_dict()
        return payload


@dataclass(frozen=True)
class DecisionPolicy:
    policy_id: str
    scenarios: tuple[RecommendationScenario, ...]
    rules: tuple[DecisionRule, ...]
    minimum_recommendation_score: float = 1.0
    max_observation_skew_seconds: float = 120.0

    def validate(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty")
        if not self.scenarios:
            raise ValueError("decision policy requires at least one scenario")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario ids must be unique")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("rule ids must be unique")
        for scenario in self.scenarios:
            scenario.validate()
        scenario_set = set(scenario_ids)
        for rule in self.rules:
            rule.validate()
            if rule.scenario_id not in scenario_set:
                raise ValueError(f"rule {rule.rule_id!r} references unknown scenario")
        minimum = _finite_float(
            self.minimum_recommendation_score,
            label="minimum_recommendation_score",
        )
        if minimum < 0.0:
            raise ValueError("minimum_recommendation_score must be non-negative")
        skew = _finite_float(
            self.max_observation_skew_seconds,
            label="max_observation_skew_seconds",
        )
        if skew < 0.0:
            raise ValueError("max_observation_skew_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "industrial_tsfm.decision_policy.v1",
            "policy_id": self.policy_id,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "rules": [rule.to_dict() for rule in self.rules],
            "minimum_recommendation_score": self.minimum_recommendation_score,
            "max_observation_skew_seconds": self.max_observation_skew_seconds,
            "advisory_only": True,
            "operator_confirmation_required": True,
            "execution_enabled": False,
            "control_actions_enabled": False,
            "opcua_writes_enabled": False,
            "online_learning": False,
            "llm_actions_enabled": False,
        }


def decision_policy_from_dict(payload: dict[str, Any]) -> DecisionPolicy:
    raw_scenarios = payload.get("scenarios", ())
    raw_rules = payload.get("rules", ())
    if not isinstance(raw_scenarios, (list, tuple)):
        raise TypeError("scenarios must be an array")
    if not isinstance(raw_rules, (list, tuple)):
        raise TypeError("rules must be an array")

    def parse_condition(raw: Any) -> EvidenceCondition:
        if not isinstance(raw, dict):
            raise TypeError("condition must be an object")
        return EvidenceCondition(
            source=str(raw.get("source", "")),
            metric=str(raw.get("metric", "")),
            operator=str(raw.get("operator", "")),
            value=raw.get("value"),
            target=None if raw.get("target") is None else str(raw.get("target")),
        )

    scenarios: list[RecommendationScenario] = []
    for raw in raw_scenarios:
        if not isinstance(raw, dict):
            raise TypeError("scenario must be an object")
        raw_guardrails = raw.get("guardrails", ())
        if not isinstance(raw_guardrails, (list, tuple)):
            raise TypeError("scenario guardrails must be an array")
        raw_parameters = raw.get("advisory_parameters", {})
        if not isinstance(raw_parameters, dict):
            raise TypeError("advisory_parameters must be an object")
        scenarios.append(
            RecommendationScenario(
                scenario_id=str(raw.get("scenario_id", "")),
                title=str(raw.get("title", "")),
                description=str(raw.get("description", "")),
                advisory_parameters=dict(raw_parameters),
                base_priority=float(raw.get("base_priority", 0.0)),
                guardrails=tuple(parse_condition(item) for item in raw_guardrails),
            )
        )
    rules: list[DecisionRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            raise TypeError("rule must be an object")
        rules.append(
            DecisionRule(
                rule_id=str(raw.get("rule_id", "")),
                scenario_id=str(raw.get("scenario_id", "")),
                condition=parse_condition(raw.get("condition", {})),
                points=float(raw.get("points", 0.0)),
                rationale=str(raw.get("rationale", "")),
            )
        )
    policy = DecisionPolicy(
        policy_id=str(payload.get("policy_id", "")),
        scenarios=tuple(scenarios),
        rules=tuple(rules),
        minimum_recommendation_score=float(payload.get("minimum_recommendation_score", 1.0)),
        max_observation_skew_seconds=float(payload.get("max_observation_skew_seconds", 120.0)),
    )
    policy.validate()
    return policy


def _forecast_summary(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = payload.get("forecasts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("forecast evidence requires a non-empty forecasts list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError("forecast point must be an object")
        target = str(raw.get("target", "")).strip()
        if not target:
            raise ValueError("forecast point target must not be empty")
        grouped.setdefault(target, []).append(raw)
    result: dict[str, dict[str, float]] = {}
    for target, points in grouped.items():
        ordered = sorted(points, key=lambda item: int(item.get("step", 0)))
        values = [_finite_float(item.get("value"), label="forecast value") for item in ordered]
        summary = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1],
        }
        lowers = [item.get("lower") for item in ordered if item.get("lower") is not None]
        uppers = [item.get("upper") for item in ordered if item.get("upper") is not None]
        if lowers:
            summary["lower_min"] = min(
                _finite_float(value, label="forecast lower interval") for value in lowers
            )
        if uppers:
            summary["upper_max"] = max(
                _finite_float(value, label="forecast upper interval") for value in uppers
            )
        result[target] = summary
    return result


def _diagnosis_summary(payload: dict[str, Any]) -> dict[str, Any]:
    severity = str(payload.get("severity", ""))
    if severity not in _SEVERITY_RANK:
        raise ValueError("diagnosis evidence severity is invalid")
    score = payload.get("score")
    event = payload.get("event")
    if not isinstance(score, dict) or not isinstance(event, dict):
        raise ValueError("diagnosis evidence requires score and event objects")
    return {
        "severity": severity,
        "severity_rank": _SEVERITY_RANK[severity],
        "ratio_to_threshold": _finite_float(
            score.get("ratio_to_threshold"), label="diagnosis ratio_to_threshold"
        ),
        "trailing_anomaly_points": int(event.get("trailing_anomaly_points", 0)),
        "latest_anomaly": bool(event.get("latest_anomaly", False)),
    }


def _condition_result(
    condition: EvidenceCondition,
    forecast: dict[str, dict[str, float]],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    condition.validate()
    available = True
    if condition.source == "forecast":
        target = str(condition.target)
        if target not in forecast or condition.metric not in forecast[target]:
            available = False
            observed: Any = None
            expected: Any = float(condition.value)
        else:
            observed = forecast[target][condition.metric]
            expected = float(condition.value)
    elif condition.metric == "severity":
        observed = str(diagnosis["severity"])
        expected = str(condition.value)
    elif condition.metric == "latest_anomaly":
        observed = bool(diagnosis["latest_anomaly"])
        expected = bool(condition.value)
    else:
        observed = diagnosis[condition.metric]
        expected = float(condition.value)

    if not available:
        matched = False
    elif condition.metric == "severity":
        matched = _compare(
            _SEVERITY_RANK[str(observed)],
            condition.operator,
            _SEVERITY_RANK[str(expected)],
        )
    else:
        matched = _compare(observed, condition.operator, expected)
    return {
        "condition": condition.to_dict(),
        "available": available,
        "matched": matched,
        "observed_value": observed if available else None,
    }


class DecisionSupportApplication:
    """Deterministic, advisory-only rule engine over frozen runtime evidence."""

    def __init__(self, policy: DecisionPolicy) -> None:
        policy.validate()
        self.policy = policy

    def evaluate(
        self,
        forecast_payload: dict[str, Any],
        diagnosis_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(forecast_payload, dict) or not isinstance(diagnosis_payload, dict):
            raise TypeError("forecast and diagnosis evidence must be objects")
        forecast_kind = str(forecast_payload.get("source_kind", "")).strip()
        diagnosis_kind = str(diagnosis_payload.get("source_kind", "")).strip()
        if not forecast_kind or not diagnosis_kind or forecast_kind != diagnosis_kind:
            raise RuntimeError(
                "decision support refused mixed evidence source kinds: "
                f"forecast={forecast_kind or 'missing'}, diagnosis={diagnosis_kind or 'missing'}"
            )
        forecast = _forecast_summary(forecast_payload)
        diagnosis = _diagnosis_summary(diagnosis_payload)
        forecast_cutoff = forecast_payload.get("observed_through")
        diagnosis_cutoff = diagnosis_payload.get("observed_through")
        forecast_time = _parse_time(forecast_cutoff)
        diagnosis_time = _parse_time(diagnosis_cutoff)
        skew_seconds: float | None = None
        if forecast_time is not None and diagnosis_time is not None:
            skew_seconds = abs((forecast_time - diagnosis_time).total_seconds())
            if skew_seconds > self.policy.max_observation_skew_seconds:
                raise RuntimeError(
                    "decision support refused temporally misaligned evidence: "
                    f"skew={skew_seconds:.3f}s exceeds "
                    f"{self.policy.max_observation_skew_seconds:.3f}s"
                )

        evaluations: list[dict[str, Any]] = []
        for scenario in self.policy.scenarios:
            guardrail_results = [
                _condition_result(condition, forecast, diagnosis)
                for condition in scenario.guardrails
            ]
            eligible = all(item["matched"] for item in guardrail_results)
            matched_rules: list[dict[str, Any]] = []
            score = float(scenario.base_priority)
            for rule in self.policy.rules:
                if rule.scenario_id != scenario.scenario_id:
                    continue
                result = _condition_result(rule.condition, forecast, diagnosis)
                if result["matched"]:
                    score += float(rule.points)
                    matched_rules.append(
                        {
                            "rule_id": rule.rule_id,
                            "points": float(rule.points),
                            "rationale": rule.rationale,
                            **result,
                        }
                    )
            qualifies = eligible and score >= self.policy.minimum_recommendation_score
            evaluations.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "advisory_parameters": dict(scenario.advisory_parameters),
                    "eligible": eligible,
                    "qualifies": qualifies,
                    "score": score,
                    "base_priority": float(scenario.base_priority),
                    "guardrails": guardrail_results,
                    "matched_rules": matched_rules,
                    "operator_confirmation_required": True,
                    "execution_enabled": False,
                }
            )
        evaluations.sort(key=lambda item: (-float(item["score"]), str(item["scenario_id"])))
        qualifying = [item for item in evaluations if item["qualifies"]]
        top = dict(qualifying[0]) if qualifying else None
        return {
            "schema_version": "industrial_tsfm.decision_support.v1",
            "generated_at": _utc_now(),
            "status": "recommendation_available" if top is not None else "operator_review_required",
            "policy_id": self.policy.policy_id,
            "evidence": {
                "source_kind": forecast_kind,
                "forecast": {
                    "schema_version": forecast_payload.get("schema_version"),
                    "source_kind": forecast_kind,
                    "observed_through": forecast_cutoff,
                    "summary": forecast,
                },
                "diagnosis": {
                    "schema_version": diagnosis_payload.get("schema_version"),
                    "source_kind": diagnosis_kind,
                    "observed_through": diagnosis_cutoff,
                    "summary": diagnosis,
                },
                "observation_skew_seconds": skew_seconds,
            },
            "scenario_evaluations": evaluations,
            "top_recommendation": top,
            "minimum_recommendation_score": self.policy.minimum_recommendation_score,
            "scenario_evaluation_scope": "deterministic_policy_rule_evaluation_not_process_simulation",
            "claim_scope": "advisory_evidence_ranking_not_causal_or_control",
            "advisory_only": True,
            "operator_confirmation_required": True,
            "execution_enabled": False,
            "control_actions_enabled": False,
            "opcua_writes_enabled": False,
            "online_learning": False,
            "llm_actions_enabled": False,
        }
