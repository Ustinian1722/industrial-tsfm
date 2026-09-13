from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import ProjectSpec, TaskType
from .model_catalog import get_model_descriptor


@dataclass(frozen=True)
class ForecastingDeploymentSpec:
    """Auditable bridge from validation-only routing to live inference.

    This contract intentionally contains references/metadata only. It does not
    load checkpoints, fit models, or inspect target-stream labels.
    """

    deployment_id: str
    project_name: str
    source_name: str
    task_name: str
    target_columns: tuple[str, ...]
    context_length: int
    horizon: int
    selected_model: str
    selected_strategy: str
    selection_evidence: str
    target_labels_used: bool
    window_mode: str
    requires_aligned_rows: bool
    project_id: str | None = None
    checkpoint_ref: str | None = None
    uq_artifact_ref: str | None = None

    def validate(self) -> None:
        if not self.deployment_id.strip():
            raise ValueError("deployment_id must not be empty")
        if self.project_id is not None and not str(self.project_id).strip():
            raise ValueError("project_id must not be empty when configured")
        if not self.target_columns:
            raise ValueError("target_columns must not be empty")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.selection_evidence != "validation_only":
            raise ValueError("live deployment requires validation_only model selection evidence")
        if self.target_labels_used:
            raise ValueError("live deployment cannot be built from target-label-guided routing")
        if self.window_mode not in {"per_tag_univariate", "strict_multivariate"}:
            raise ValueError("unsupported window_mode")
        if self.requires_aligned_rows and self.window_mode != "strict_multivariate":
            raise ValueError("native multivariate deployment requires strict_multivariate windows")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["schema_version"] = "industrial_tsfm.forecasting_deployment.v1"
        payload["online_training"] = False
        payload["online_model_selection"] = False
        payload["online_uq_calibration"] = False
        return payload


def build_forecasting_deployment_spec(
    project: ProjectSpec,
    *,
    source_name: str,
    task_name: str,
    model_route: dict[str, Any],
    deployment_id: str | None = None,
    project_id: str | None = None,
    window_mode: str | None = None,
    checkpoint_ref: str | None = None,
    uq_artifact_ref: str | None = None,
) -> ForecastingDeploymentSpec:
    project.validate()
    sources = [source for source in project.data_sources if source.name == source_name]
    if len(sources) != 1:
        raise KeyError(source_name)
    tasks = [task for task in project.tasks if task.name == task_name]
    if len(tasks) != 1:
        raise KeyError(task_name)
    task = tasks[0]
    if task.task_type != TaskType.FORECASTING:
        raise ValueError("live forecasting deployment requires a forecasting task")
    if task.context_length is None or task.horizon is None:
        raise ValueError("forecasting task requires context_length and horizon")
    if not bool(model_route.get("route_ready")):
        raise ValueError("model route is not ready for deployment")
    route_task = str(model_route.get("task", ""))
    if route_task and route_task != task_name:
        raise ValueError("model route task does not match deployment task")
    selected_model = str(model_route.get("selected_model", "")).strip()
    if not selected_model:
        raise ValueError("model route has no selected_model")
    descriptor = get_model_descriptor(selected_model)
    strategy = str(model_route.get("selected_strategy", "")).strip()
    if not strategy:
        raise ValueError("model route has no selected_strategy")
    evidence = str(model_route.get("selection_evidence", "")).strip()
    labels_used = bool(model_route.get("target_labels_used", False))
    resolved_mode = window_mode or (
        "strict_multivariate" if descriptor.native_multivariate else "per_tag_univariate"
    )
    spec = ForecastingDeploymentSpec(
        deployment_id=(deployment_id or f"{project.name}--{task_name}--live").strip(),
        project_name=project.name,
        source_name=source_name,
        task_name=task_name,
        target_columns=tuple(task.target_columns),
        context_length=int(task.context_length),
        horizon=int(task.horizon),
        selected_model=selected_model,
        selected_strategy=strategy,
        selection_evidence=evidence,
        target_labels_used=labels_used,
        window_mode=resolved_mode,
        requires_aligned_rows=bool(descriptor.native_multivariate),
        project_id=str(project_id).strip() if project_id is not None else None,
        checkpoint_ref=checkpoint_ref,
        uq_artifact_ref=uq_artifact_ref,
    )
    spec.validate()
    return spec
