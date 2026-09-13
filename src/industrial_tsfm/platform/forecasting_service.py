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
    """

    deployment.validate()
    scope = opcua_registry.runtime_scope(runtime_id)
    if scope["source_name"] != deployment.source_name:
        raise ValueError("runtime source does not match forecasting deployment")
    runtime_project = scope["project_id"]
    if runtime_project is not None and runtime_project != deployment.project_name:
        # Project IDs and display names may differ. Callers that use workspace IDs
        # should pass the ID through model_metadata and use the explicit check below.
        expected_project_id = None
        if model_metadata is not None:
            expected_project_id = model_metadata.get("project_id")
        if expected_project_id is not None and runtime_project != str(expected_project_id):
            raise ValueError("runtime project does not match forecasting deployment")

    provider = opcua_registry.window_provider(runtime_id)
    metadata = {
        "selected_model": deployment.selected_model,
        "selected_strategy": deployment.selected_strategy,
        "selection_evidence": deployment.selection_evidence,
        "target_labels_used": deployment.target_labels_used,
        "checkpoint_ref": deployment.checkpoint_ref,
        "deployment_id": deployment.deployment_id,
        **dict(model_metadata or {}),
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
