from __future__ import annotations

from .diagnosis import DiagnosisRequest, LiveDiagnosisApplication
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
    application_id: str,
    project_id: str | None,
    source_name: str,
    task_name: str,
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

    resolved_source = str(source_name).strip()
    resolved_task = str(task_name).strip()
    if not resolved_source:
        raise ValueError("source_name must not be empty")
    if not resolved_task:
        raise ValueError("task_name must not be empty")

    scope = opcua_registry.runtime_scope(runtime_id)
    if scope["source_name"] != resolved_source:
        raise ValueError("runtime source does not match diagnosis source")
    if project_id is not None and scope["project_id"] != project_id:
        raise ValueError("runtime project does not match diagnosis project")

    runtime_status = opcua_registry.status(runtime_id)
    runtime_state = str(runtime_status.get("state", {}).get("status", "unknown"))
    if bool(runtime_status.get("task_done")) or runtime_state != "running":
        raise RuntimeError(
            f"OPC-UA runtime {runtime_id!r} must be running before diagnosis registration"
        )
    opcua_registry.window_provider(runtime_id)
    provider = _ActiveOPCUADiagnosisWindowProvider(opcua_registry, runtime_id)
    application = LiveDiagnosisApplication(artifact, request or DiagnosisRequest())
    return diagnosis_registry.register(
        application_id=application_id,
        runtime_id=runtime_id,
        project_id=scope["project_id"],
        source_name=resolved_source,
        task_name=resolved_task,
        application=application,
        provider=provider,
    )
