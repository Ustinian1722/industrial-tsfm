from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import ProjectSpec, TaskType
from .diagnosis import DiagnosisRequest


@dataclass(frozen=True)
class DiagnosisDeploymentSpec:
    deployment_id: str
    project_name: str
    project_id: str | None
    source_name: str
    task_name: str
    feature_columns: tuple[str, ...]
    context_observations: int
    top_k_sensors: int
    artifact_ref: str | None = None
    fit_scope: str = "offline_training_or_calibration_only"
    target_labels_used: bool = False

    def validate(self) -> None:
        if not self.deployment_id.strip():
            raise ValueError("deployment_id must not be empty")
        if self.project_id is not None and not self.project_id.strip():
            raise ValueError("project_id must not be empty when configured")
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty")
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique")
        DiagnosisRequest(
            context_observations=self.context_observations,
            top_k_sensors=self.top_k_sensors,
        ).validate()
        if self.fit_scope != "offline_training_or_calibration_only":
            raise ValueError("diagnosis deployment must use offline-fitted artifact scope")
        if self.target_labels_used:
            raise ValueError("diagnosis deployment cannot depend on target labels")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["schema_version"] = "industrial_tsfm.diagnosis_deployment.v1"
        payload["feature_columns"] = list(self.feature_columns)
        payload["online_training"] = False
        payload["threshold_updates"] = False
        payload["online_artifact_updates"] = False
        payload["causal_root_cause_claim"] = False
        payload["control_actions_enabled"] = False
        return payload


def build_diagnosis_deployment_spec(
    project: ProjectSpec,
    *,
    project_id: str | None,
    source_name: str,
    task_name: str,
    feature_columns: tuple[str, ...] = (),
    context_observations: int = 64,
    top_k_sensors: int = 5,
    artifact_ref: str | None = None,
    deployment_id: str | None = None,
) -> DiagnosisDeploymentSpec:
    project.validate()
    sources = [source for source in project.data_sources if source.name == source_name]
    if len(sources) != 1:
        raise ValueError(f"source {source_name!r} must exist exactly once")
    tasks = [task for task in project.tasks if task.name == task_name]
    if len(tasks) != 1:
        raise ValueError(f"task {task_name!r} must exist exactly once")
    task = tasks[0]
    if task.task_type != TaskType.ANOMALY:
        raise ValueError(f"task {task_name!r} is not an anomaly_detection task")

    configured = tuple(str(value) for value in task.constraints.get("feature_columns", ()))
    resolved_features = configured or tuple(str(value) for value in feature_columns)
    if not resolved_features:
        raise ValueError(
            "diagnosis feature columns must be configured on the anomaly task or supplied explicitly"
        )
    resolved_id = deployment_id or f"{project_id or project.name}--{task_name}--live-diagnosis"
    spec = DiagnosisDeploymentSpec(
        deployment_id=str(resolved_id),
        project_name=project.name,
        project_id=project_id,
        source_name=source_name,
        task_name=task_name,
        feature_columns=resolved_features,
        context_observations=int(context_observations),
        top_k_sensors=int(top_k_sensors),
        artifact_ref=artifact_ref,
    )
    spec.validate()
    return spec
