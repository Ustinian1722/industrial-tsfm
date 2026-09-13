from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .forecasting_registry import LiveForecastingRegistry


def attach_forecasting_routes(
    app: FastAPI,
    registry: LiveForecastingRegistry | None = None,
) -> LiveForecastingRegistry:
    """Attach inference-only live forecasting routes to the platform API.

    Model loading/registration is intentionally kept outside the HTTP surface in
    this slice. Only already prepared, frozen forecasting applications may be
    executed through these routes.
    """

    active_registry = registry or LiveForecastingRegistry()
    router = APIRouter(prefix="/v1", tags=["forecasting-live"])

    @router.get("/forecasting/capabilities")
    def forecasting_capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.live_forecasting_capabilities.v1",
            "runtime_registry": "process_local_v1",
            "operations": ["list", "status", "infer", "remove"],
            "http_model_registration": False,
            "online_training": False,
            "online_calibration": False,
            "window_contract": "WindowProvider",
            "modes": ["per_tag_univariate", "strict_multivariate"],
            "hidden_sorting": False,
            "hidden_interpolation": False,
            "irregular_cadence_timestamp_policy": "do_not_invent_future_timestamps",
        }

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
