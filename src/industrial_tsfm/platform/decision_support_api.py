from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .decision_support import DecisionSupportApplication, decision_policy_from_dict
from .decision_support_registry import DecisionSupportRegistry
from .diagnosis_registry import LiveDiagnosisRegistry
from .forecasting_registry import LiveForecastingRegistry
from .workspace import WorkspaceStore


def attach_decision_support_routes(
    app: FastAPI,
    workspace: WorkspaceStore,
    forecasting_registry: LiveForecastingRegistry,
    diagnosis_registry: LiveDiagnosisRegistry,
    registry: DecisionSupportRegistry | None = None,
) -> DecisionSupportRegistry:
    """Attach deterministic, advisory-only decision support routes."""

    active_registry = registry or DecisionSupportRegistry()
    router = APIRouter(prefix="/v1", tags=["decision-support"])

    @router.get("/decision-support/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.decision_support_capabilities.v1",
            "operations": ["register", "list", "status", "evaluate", "remove"],
            "inputs": ["live_forecasting", "live_diagnosis"],
            "ranking": "deterministic_transparent_rule_points",
            "scenario_evaluation": "policy_rule_evaluation",
            "process_simulation": False,
            "optimization_solver": False,
            "advisory_only": True,
            "operator_confirmation_required": True,
            "execution_enabled": False,
            "control_actions_enabled": False,
            "opcua_writes_enabled": False,
            "online_learning": False,
            "llm_actions_enabled": False,
            "claim_scope": "advisory_evidence_ranking_not_causal_or_control",
        }

    @router.post("/projects/{project_id}/decision-support/applications", status_code=201)
    def register_application(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        application_id = str(payload.get("application_id", "")).strip()
        source_name = str(payload.get("source_name", "")).strip()
        runtime_id = str(payload.get("runtime_id", "")).strip()
        forecasting_application_id = str(payload.get("forecasting_application_id", "")).strip()
        diagnosis_application_id = str(payload.get("diagnosis_application_id", "")).strip()
        policy_payload = payload.get("policy")
        if not application_id:
            raise HTTPException(status_code=422, detail="application_id is required")
        if not source_name:
            raise HTTPException(status_code=422, detail="source_name is required")
        if not runtime_id:
            raise HTTPException(status_code=422, detail="runtime_id is required")
        if not forecasting_application_id or not diagnosis_application_id:
            raise HTTPException(
                status_code=422,
                detail="forecasting_application_id and diagnosis_application_id are required",
            )
        if not isinstance(policy_payload, dict):
            raise HTTPException(status_code=422, detail="policy must be an object")
        try:
            project = workspace.get_project(project_id)
            matches = [source for source in project.data_sources if source.name == source_name]
            if len(matches) != 1:
                raise KeyError(source_name)
            policy = decision_policy_from_dict(policy_payload)
            application = DecisionSupportApplication(policy)
            result = active_registry.register(
                application_id=application_id,
                project_id=project_id,
                source_name=source_name,
                runtime_id=runtime_id,
                forecasting_application_id=forecasting_application_id,
                diagnosis_application_id=diagnosis_application_id,
                application=application,
                forecasting_registry=forecasting_registry,
                diagnosis_registry=diagnosis_registry,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="project, source, forecasting application, or diagnosis application not found",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(
            project_id,
            f"decision-support-policy-{application_id}",
            {
                "schema_version": "industrial_tsfm.decision_support_binding.v1",
                "application_id": application_id,
                "project_id": project_id,
                "source_name": source_name,
                "runtime_id": runtime_id,
                "forecasting_application_id": forecasting_application_id,
                "diagnosis_application_id": diagnosis_application_id,
                "policy": policy.to_dict(),
                "advisory_only": True,
                "operator_confirmation_required": True,
                "execution_enabled": False,
                "control_actions_enabled": False,
                "opcua_writes_enabled": False,
            },
        )
        return result

    @router.get("/decision-support/applications")
    def applications() -> dict[str, Any]:
        rows = active_registry.list()
        return {"count": len(rows), "applications": rows}

    @router.get("/decision-support/applications/{application_id}")
    def application_status(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.status(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="decision support application not found") from exc

    @router.post("/decision-support/applications/{application_id}/evaluate")
    def evaluate(application_id: str) -> dict[str, Any]:
        try:
            return active_registry.evaluate(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="decision support application not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/decision-support/applications/{application_id}", status_code=204)
    def remove(application_id: str) -> None:
        try:
            active_registry.remove(application_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="decision support application not found") from exc

    app.include_router(router)
    app.state.decision_support_registry = active_registry
    return active_registry
