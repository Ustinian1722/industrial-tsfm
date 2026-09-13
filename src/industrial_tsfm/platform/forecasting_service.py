from __future__ import annotations

from typing import Any

from .forecasting import (
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    LiveForecastingApplication,
    RuntimeForecastModel,
)
from .forecasting_deployment import ForecastingDeploymentSpec
from .forecasting_registry import LiveForecastingRegistry
from .forecasting_uq import ConformalForecastArtifact
from .opcua_registry import OPCUALiveRuntimeRegistry
from .runtime import TimeSeriesWindow


class _ActiveOPCUAWindowProvider:
    """Fail closed when a bound OPC-UA runtime is no longer actively streaming."""

    def __init__(self, registry: OPCUALiveRuntimeRegistry, runtime_id: str) -> None:
        self.registry = registry
        self.runtime_id = runtime_id

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow:
        status = self.registry.status(self.runtime_id)
        runtime_state = str(status.get("state", {}).get("status", "unknown"))
        if bool(status.get("task_done")) or runtime_state != "running":
            raise RuntimeError(
                f"OPC-UA runtime {self.runtime_id!r} is not actively running "
                f"(state={runtime_state!r}); refusing stale live forecast"
            )
        provider = self.registry.window_provider(self.runtime_id)
        return provider.materialize_window(max_observations)


def register_prepared_live_forecast(
    *,
    deployment: ForecastingDeploymentSpec,
    runtime_id: str,
    prepared_model: RuntimeForecastModel,
    opcua_registry: OPCUALiveRuntimeRegistry,
    forecasting_registry: LiveForecastingRegistry,
    uq_artifact: ConformalForecastArtifact | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind an already prepared model to an active OPC-UA runtime.

    This is intentionally not a model loader. The caller must prepare/freeze the
    model through the existing research/product selection path before calling this
    function. No fitting, adaptation, model selection, or calibration occurs here.
    Runtime availability is re-checked on every inference so a stopped or
    reconnecting source cannot silently emit a forecast from a stale buffer.
    """

    deployment.validate()
    scope = opcua_registry.runtime_scope(runtime_id)
    if scope["source_name"] != deployment.source_name:
        raise ValueError("runtime source does not match forecasting deployment")
    if deployment.project_id is not None and scope["project_id"] != deployment.project_id:
        raise ValueError("runtime project does not match forecasting deployment")

    runtime_status = opcua_registry.status(runtime_id)
    runtime_state = str(runtime_status.get("state", {}).get("status", "unknown"))
    if bool(runtime_status.get("task_done")) or runtime_state != "running":
        raise RuntimeError(
            f"OPC-UA runtime {runtime_id!r} must be running before forecast registration"
        )
    opcua_registry.window_provider(runtime_id)
    provider = _ActiveOPCUAWindowProvider(opcua_registry, runtime_id)
    metadata = {
        **dict(model_metadata or {}),
        "selected_model": deployment.selected_model,
        "selected_strategy": deployment.selected_strategy,
        "selection_evidence": deployment.selection_evidence,
        "target_labels_used": deployment.target_labels_used,
        "checkpoint_ref": deployment.checkpoint_ref,
        "deployment_id": deployment.deployment_id,
        "project_id": deployment.project_id,
        "source_name": deployment.source_name,
        "task_name": deployment.task_name,
    }
    application = LiveForecastingApplication(
        ForecastModelRuntimeAdapter(prepared_model, model_metadata=metadata),
        ForecastingRequest(
            target_columns=deployment.target_columns,
            context_length=deployment.context_length,
            horizon=deployment.horizon,
            mode=deployment.window_mode,
            require_good_quality=True,
            require_chronological=True,
        ),
    )
    return forecasting_registry.register(
        application_id=deployment.deployment_id,
        runtime_id=runtime_id,
        project_id=scope["project_id"],
        source_name=deployment.source_name,
        task_name=deployment.task_name,
        application=application,
        provider=provider,
        uq_artifact=uq_artifact,
    )
