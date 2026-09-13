from __future__ import annotations

from .diagnosis import DiagnosisRequest, LiveDiagnosisApplication
from .diagnosis_deployment import DiagnosisDeploymentSpec
from .diagnosis_registry import LiveDiagnosisRegistry
from .online_anomaly import PCASPEArtifact
from .opcua_registry import OPCUALiveRuntimeRegistry
from .runtime import TimeSeriesWindow


class _ActiveOPCUADiagnosisWindowProvider:
    """Fail closed if the bound OPC-UA runtime is not actively streaming."""

    def __init__(self, registry: OPCUALiveRuntimeRegistry, runtime_id: str) -> None:
        self.registry = registry
        self.runtime_id = runtime_id

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow:
        status = self.registry.status(self.runtime_id)
        runtime_state = str(status.get("state", {}).get("status", "unknown"))
        if bool(status.get("task_done")) or runtime_state != "running":
            raise RuntimeError(
                f"OPC-UA runtime {self.runtime_id!r} is not actively running "
                f"(state={runtime_state!r}); refusing stale live diagnosis"
            )
        provider = self.registry.window_provider(self.runtime_id)
        return provider.materialize_window(max_observations)


def register_prepared_live_diagnosis(
    *,
    deployment: DiagnosisDeploymentSpec,
    runtime_id: str,
    artifact: PCASPEArtifact,
    opcua_registry: OPCUALiveRuntimeRegistry,
    diagnosis_registry: LiveDiagnosisRegistry,
    request: DiagnosisRequest | None = None,
) -> dict[str, object]:
    """Bind an offline-fitted PCA-SPE artifact to a running read-only OPC-UA source.

    Artifact fitting and threshold selection must happen before registration.
    This live path performs no fitting, threshold updates, adaptation, or control.
    """

    deployment.validate()
    if tuple(artifact.feature_columns) != tuple(deployment.feature_columns):
        raise ValueError("diagnosis artifact features do not match deployment feature columns")
    resolved_request = request or DiagnosisRequest(
        context_observations=deployment.context_observations,
        top_k_sensors=deployment.top_k_sensors,
    )
    resolved_request.validate()
    if resolved_request.context_observations != deployment.context_observations:
        raise ValueError("diagnosis request context does not match deployment")
    if resolved_request.top_k_sensors != deployment.top_k_sensors:
        raise ValueError("diagnosis request top_k_sensors does not match deployment")

    scope = opcua_registry.runtime_scope(runtime_id)
    if scope["source_name"] != deployment.source_name:
        raise ValueError("runtime source does not match diagnosis deployment")
    if deployment.project_id is not None and scope["project_id"] != deployment.project_id:
        raise ValueError("runtime project does not match diagnosis deployment")

    runtime_status = opcua_registry.status(runtime_id)
    runtime_state = str(runtime_status.get("state", {}).get("status", "unknown"))
    if bool(runtime_status.get("task_done")) or runtime_state != "running":
        raise RuntimeError(
            f"OPC-UA runtime {runtime_id!r} must be running before diagnosis registration"
        )
    opcua_registry.window_provider(runtime_id)
    provider = _ActiveOPCUADiagnosisWindowProvider(opcua_registry, runtime_id)
    application = LiveDiagnosisApplication(artifact, resolved_request)
    return diagnosis_registry.register(
        application_id=deployment.deployment_id,
        runtime_id=runtime_id,
        project_id=scope["project_id"],
        source_name=deployment.source_name,
        task_name=deployment.task_name,
        application=application,
        provider=provider,
    )
