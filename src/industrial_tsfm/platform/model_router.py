from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ..adaptation_engine import AdaptationRequest, build_adaptation_plan
from .contracts import ProjectSpec, TaskDefinition, TaskType


@dataclass(frozen=True)
class ProductRoutingRequest:
    """Product-facing routing constraints for one task."""

    task_name: str
    target_data_fraction: float = 0.0
    support_windows: int | None = None
    shift_score: float = 0.0
    max_adaptation_seconds: float | None = None
    max_parameters: int | None = None
    max_inference_seconds: float | None = None
    min_readiness_score: float = 40.0
    metric: str = "normalized_rmse"

    def validate(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must be non-empty")
        if not 0.0 <= self.target_data_fraction <= 1.0:
            raise ValueError("target_data_fraction must be in [0, 1]")
        if self.support_windows is not None and self.support_windows < 0:
            raise ValueError("support_windows must be non-negative")
        if self.shift_score < 0.0:
            raise ValueError("shift_score must be non-negative")
        if not 0.0 <= self.min_readiness_score <= 100.0:
            raise ValueError("min_readiness_score must be in [0, 100]")


def _find_task(project: ProjectSpec, name: str) -> TaskDefinition:
    matches = [task for task in project.tasks if task.name == name]
    if not matches:
        raise ValueError(f"task {name!r} does not exist in project {project.name!r}")
    if len(matches) > 1:
        raise ValueError(f"task name {name!r} is not unique")
    return matches[0]


def _blocked_decision(
    project: ProjectSpec,
    task: TaskDefinition,
    request: ProductRoutingRequest,
    reasons: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "industrial_tsfm.product_route.v1",
        "project": project.name,
        "task": task.name,
        "task_type": task.task_type.value,
        "status": "blocked",
        "route_ready": False,
        "blocking_reasons": reasons,
        "warnings": warnings,
        "request": asdict(request),
        "target_labels_used": False,
        "selected_model": None,
        "selected_strategy": None,
    }


def route_task(
    project: ProjectSpec,
    validation_table: pd.DataFrame,
    data_audit: dict[str, Any],
    request: ProductRoutingRequest,
) -> dict[str, Any]:
    """Route an audited task to the existing validation-only adaptation engine.

    This is intentionally a thin product wrapper rather than a new selector. It
    preserves the research engine's evidence boundary: final target labels are
    never read for model selection or adaptation-strategy choice.
    """

    project.validate()
    request.validate()
    task = _find_task(project, request.task_name)

    warnings: list[str] = []
    reasons: list[str] = []
    readiness = float(data_audit.get("summary", {}).get("readiness_score", 0.0))
    missing_targets = set(data_audit.get("targets", {}).get("missing", []))

    if task.task_type != TaskType.FORECASTING:
        reasons.append(
            f"task type {task.task_type.value!r} has no product router adapter in V1"
        )
    task_missing_targets = [target for target in task.target_columns if target in missing_targets]
    if task_missing_targets:
        reasons.append(
            "configured targets are missing from the audited data source: "
            + ", ".join(task_missing_targets)
        )
    if readiness < request.min_readiness_score:
        reasons.append(
            f"data readiness {readiness:.1f} is below the routing threshold "
            f"{request.min_readiness_score:.1f}"
        )
    elif readiness < 70.0:
        warnings.append(
            f"data readiness is only {readiness:.1f}; route is allowed but remediation is recommended"
        )

    if reasons:
        return _blocked_decision(project, task, request, reasons, warnings)

    adaptation_request = AdaptationRequest(
        target_data_fraction=request.target_data_fraction,
        shift_score=request.shift_score,
        support_windows=request.support_windows,
        max_adaptation_seconds=request.max_adaptation_seconds,
        max_parameters=request.max_parameters,
        max_inference_seconds=request.max_inference_seconds,
        metric=request.metric,
    )
    plan = build_adaptation_plan(validation_table, adaptation_request)

    return {
        "schema_version": "industrial_tsfm.product_route.v1",
        "project": project.name,
        "task": task.name,
        "task_type": task.task_type.value,
        "status": "ready",
        "route_ready": True,
        "blocking_reasons": [],
        "warnings": warnings,
        "readiness_score": readiness,
        "request": asdict(request),
        "target_labels_used": bool(plan.get("target_labels_used", False)),
        "selection_evidence": plan.get("selection_evidence"),
        "selected_model": plan.get("selected_model"),
        "selected_training_mode": plan.get("selected_training_mode"),
        "selected_strategy": plan.get("selected_strategy"),
        "strategy_reason": plan.get("strategy_reason"),
        "selected_validation_metric": plan.get("selected_validation_metric"),
        "estimated_support_windows": plan.get("estimated_support_windows"),
        "adaptation_budget_status": plan.get("adaptation_budget_status"),
        "strategy_cost": plan.get("strategy_cost"),
        "model_ranking": plan.get("model_ranking", []),
    }
