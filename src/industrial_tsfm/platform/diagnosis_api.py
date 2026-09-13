from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .diagnosis_registry import LiveDiagnosisRegistry


def attach_diagnosis_routes(
    app: FastAPI,
    registry: LiveDiagnosisRegistry | None = None,
) -> LiveDiagnosisRegistry:
    """Attach inference-only live diagnosis routes."""

    active_registry = registry or LiveDiagnosisRegistry()
    router = APIRouter(prefix="/v1/diagnosis", tags=["diagnosis"])

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.live_diagnosis_capabilities.v1",
            "operations": ["list", "status", "infer", "remove"],
            "detector": "frozen_pca_spe",
            "triage": "score_ratio_and_persistence",
            "http_artifact_registration": False,
            "online_training": False,
            "threshold_updates": False,
            "online_artifact_updates": False,
            "causal_root_cause_claim": False,
            "control_actions_enabled": False,
            "claim_scope": "statistical_reconstruction_contribution_not_causal_root_cause",
        }

    @router.get("/applications")
    def applications() -> dict[str, Any]:
        rows = active_registry.list()
        return {"count": len(rows), "applications": rows}

    @router.get("/applications/{application_id}")
    def application_status(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.status(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc

    @router.post("/applications/{application_id}/infer")
    def infer(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.infer(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/applications/{application_id}", status_code=204)
    def remove(application_id: str) -> None:
        try:
            active_registry.remove(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc

    app.include_router(router)
    app.state.diagnosis_registry = active_registry
    return active_registry
