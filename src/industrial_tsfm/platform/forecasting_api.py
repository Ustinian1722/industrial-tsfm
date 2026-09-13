from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .forecasting_deployment import build_forecasting_deployment_spec
from .forecasting_registry import LiveForecastingRegistry
from .workspace import WorkspaceStore


def attach_forecasting_routes(
    app: FastAPI,
    workspace: WorkspaceStore,
    registry: LiveForecastingRegistry | None = None,
) -> LiveForecastingRegistry:
    """Attach validation-bound, inference-only live forecasting routes."""

    active_registry = registry or LiveForecastingRegistry()
    router = APIRouter(prefix="/v1", tags=["forecasting-live"])

    @router.get("/forecasting/capabilities")
    def forecasting_capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.live_forecasting_capabilities.v1",
            "runtime_registry": "process_local_v1",
            "operations": [
                "build_deployment_spec",
                "list",
                "status",
                "infer",
                "remove",
            ],
            "http_model_registration": False,
            "online_training": False,
            "online_model_selection": False,
            "online_calibration": False,
            "window_contract": "WindowProvider",
            "modes": ["per_tag_univariate", "strict_multivariate"],
            "hidden_sorting": False,
            "hidden_interpolation": False,
            "irregular_cadence_timestamp_policy": "do_not_invent_future_timestamps",
        }

    @router.post("/projects/{project_id}/forecasting/{task_name}/deployment", status_code=201)
    def build_deployment(
        project_id: str,
        task_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        source_name = str(payload.get("source_name", "")).strip()
        if not source_name:
            raise HTTPException(status_code=422, detail="source_name is required")
        try:
            project = workspace.get_project(project_id)
            model_route = workspace.read_derived_artifact(
                project_id,
                f"model-route-{task_name}",
            )
            spec = build_forecasting_deployment_spec(
                project,
                source_name=source_name,
                task_name=task_name,
                model_route=model_route,
                deployment_id=payload.get("deployment_id"),
                window_mode=payload.get("window_mode"),
                checkpoint_ref=payload.get("checkpoint_ref"),
                uq_artifact_ref=payload.get("uq_artifact_ref"),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="project, source, task, or persisted model route not found",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = spec.to_dict()
        result["project_id"] = project_id
        workspace.write_derived_artifact(
            project_id,
            f"forecasting-deployment-{task_name}",
            result,
        )
        return result

    @router.get("/forecasting/applications")
    def list_forecasting_applications() -> dict[str, Any]:
        rows = active_registry.list()
        return {"count": len(rows), "applications": rows}

    @router.get("/forecasting/applications/{application_id}")
    def forecasting_application_status(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.status(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="forecasting application not found") from exc

    @router.post("/forecasting/applications/{application_id}/infer")
    def infer_forecasting_application(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.infer(application_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="forecasting application not found") from exc
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/forecasting/applications/{application_id}", status_code=204)
    def remove_forecasting_application(application_id: str) -> None:
        try:
            active_registry.remove(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="forecasting application not found") from exc

    app.include_router(router)
    app.state.forecasting_registry = active_registry
    return active_registry
