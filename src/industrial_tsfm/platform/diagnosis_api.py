from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .diagnosis_deployment import build_diagnosis_deployment_spec
from .diagnosis_registry import LiveDiagnosisRegistry
from .workspace import WorkspaceStore


def attach_diagnosis_routes(
    app: FastAPI,
    workspace: WorkspaceStore,
    registry: LiveDiagnosisRegistry | None = None,
) -> LiveDiagnosisRegistry:
    """Attach offline-artifact-bound, inference-only live diagnosis routes."""

    active_registry = registry or LiveDiagnosisRegistry()
    router = APIRouter(prefix="/v1", tags=["diagnosis-live"])

    @router.get("/diagnosis/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.live_diagnosis_capabilities.v1",
            "operations": ["build_deployment_spec", "list", "status", "infer", "remove"],
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

    @router.post("/projects/{project_id}/diagnosis/{task_name}/deployment", status_code=201)
    def build_deployment(
        project_id: str,
        task_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        source_name = str(payload.get("source_name", "")).strip()
        if not source_name:
            raise HTTPException(status_code=422, detail="source_name is required")
        raw_features = payload.get("feature_columns", ())
        if raw_features is None:
            raw_features = ()
        if not isinstance(raw_features, (list, tuple)):
            raise HTTPException(status_code=422, detail="feature_columns must be an array")
        try:
            project = workspace.get_project(project_id)
            spec = build_diagnosis_deployment_spec(
                project,
                project_id=project_id,
                source_name=source_name,
                task_name=task_name,
                feature_columns=tuple(str(value) for value in raw_features),
                context_observations=int(payload.get("context_observations", 64)),
                top_k_sensors=int(payload.get("top_k_sensors", 5)),
                artifact_ref=payload.get("artifact_ref"),
                deployment_id=payload.get("deployment_id"),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="project, source, or task not found",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = spec.to_dict()
        workspace.write_derived_artifact(
            project_id,
            f"diagnosis-deployment-{task_name}",
            result,
        )
        return result

    @router.get("/diagnosis/applications")
    def applications() -> dict[str, Any]:
        rows = active_registry.list()
        return {"count": len(rows), "applications": rows}

    @router.get("/diagnosis/applications/{application_id}")
    def application_status(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.status(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc

    @router.post("/diagnosis/applications/{application_id}/infer")
    def infer(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.infer(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/diagnosis/applications/{application_id}", status_code=204)
    def remove(application_id: str) -> None:
        try:
            active_registry.remove(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="diagnosis application not found") from exc

    app.include_router(router)
    app.state.diagnosis_registry = active_registry
    return active_registry
