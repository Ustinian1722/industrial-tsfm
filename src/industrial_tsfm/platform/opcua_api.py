from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .api import create_app as create_base_app
from .contracts import DataSourceKind, DataSourceSpec
from .diagnosis_api import attach_diagnosis_routes
from .diagnosis_registry import LiveDiagnosisRegistry
from .forecasting_api import attach_forecasting_routes
from .forecasting_registry import LiveForecastingRegistry
from .opcua import OPCUAConnectorError, browse_opcua_source, read_opcua_snapshot
from .opcua_registry import OPCUALiveRuntimeRegistry
from .workspace import WorkspaceStore


def _opcua_source(workspace: WorkspaceStore, project_id: str, source_name: str) -> DataSourceSpec:
    try:
        project = workspace.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    matches = [source for source in project.data_sources if source.name == source_name]
    if len(matches) != 1:
        raise HTTPException(status_code=404, detail="data source not found")
    source = matches[0]
    if source.kind != DataSourceKind.OPCUA:
        raise HTTPException(status_code=422, detail="data source is not an OPC-UA source")
    return source


def _connection_error(exc: Exception) -> HTTPException:
    if isinstance(exc, OPCUAConnectorError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=502, detail=f"OPC-UA operation failed: {type(exc).__name__}")


def attach_opcua_routes(
    app: FastAPI,
    workspace: WorkspaceStore,
    registry: OPCUALiveRuntimeRegistry | None = None,
) -> OPCUALiveRuntimeRegistry:
    """Attach read-only OPC-UA routes to an existing platform FastAPI app."""

    active_registry = registry or OPCUALiveRuntimeRegistry()
    router = APIRouter(prefix="/v1", tags=["opcua"])

    @router.get("/opcua/capabilities")
    async def opcua_capabilities() -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.opcua_capabilities.v1",
            "dependency_available": importlib.util.find_spec("asyncua") is not None,
            "operations": ["browse", "snapshot_read", "subscribe", "runtime_status", "runtime_stop"],
            "read_only": True,
            "writes_enabled": False,
            "credential_values_persisted": False,
            "runtime_registry": "process_local_v1",
            "runtime_scope": "project_and_source",
            "quality_and_timestamps_preserved": True,
            "hidden_resampling": False,
        }

    @router.post("/projects/{project_id}/opcua/{source_name}/browse")
    async def browse(project_id: str, source_name: str) -> dict[str, Any]:
        source = _opcua_source(workspace, project_id, source_name)
        try:
            result = await browse_opcua_source(source)
        except Exception as exc:
            raise _connection_error(exc) from exc
        workspace.write_derived_artifact(project_id, f"opcua-browse-{source_name}", result)
        return result

    @router.post("/projects/{project_id}/opcua/{source_name}/snapshot")
    async def snapshot(
        project_id: str,
        source_name: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = _opcua_source(workspace, project_id, source_name)
        node_ids = None
        if payload and payload.get("node_ids") is not None:
            raw_nodes = payload["node_ids"]
            if not isinstance(raw_nodes, dict):
                raise HTTPException(status_code=422, detail="node_ids must be an object")
            node_ids = {str(key): str(value) for key, value in raw_nodes.items()}
        try:
            result = await read_opcua_snapshot(source, node_ids=node_ids)
        except Exception as exc:
            raise _connection_error(exc) from exc
        workspace.write_derived_artifact(project_id, f"opcua-snapshot-{source_name}", result)
        return result

    @router.post("/projects/{project_id}/opcua/{source_name}/live/start", status_code=201)
    async def start_live(project_id: str, source_name: str) -> dict[str, Any]:
        if importlib.util.find_spec("asyncua") is None:
            raise HTTPException(
                status_code=503,
                detail='OPC-UA dependency is not installed; install industrial-tsfm[opcua]',
            )
        source = _opcua_source(workspace, project_id, source_name)
        try:
            result = await active_registry.start(source, project_id=project_id)
        except (OPCUAConnectorError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result

    @router.get("/opcua/runtimes")
    async def list_runtimes() -> dict[str, Any]:
        rows = active_registry.list()
        return {"count": len(rows), "runtimes": rows}

    @router.get("/opcua/runtimes/{runtime_id}")
    async def runtime_status(runtime_id: str) -> dict[str, Any]:
        try:
            return active_registry.status(runtime_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="OPC-UA runtime not found") from exc

    @router.post("/opcua/runtimes/{runtime_id}/stop")
    async def stop_runtime(runtime_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        timeout = float((payload or {}).get("timeout_seconds", 5.0))
        try:
            return await active_registry.stop(runtime_id, timeout_seconds=timeout)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="OPC-UA runtime not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.delete("/opcua/runtimes/{runtime_id}", status_code=204)
    async def remove_runtime(runtime_id: str) -> None:
        try:
            active_registry.remove(runtime_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="OPC-UA runtime not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    app.include_router(router)
    app.state.opcua_registry = active_registry

    @app.on_event("shutdown")
    async def _shutdown_opcua_runtimes() -> None:
        await active_registry.stop_all()

    return active_registry


def create_live_app(
    artifact_root: str | Path = "results",
    workspace_root: str | Path = ".industsfm/workspace",
    *,
    opcua_registry: OPCUALiveRuntimeRegistry | None = None,
    forecasting_registry: LiveForecastingRegistry | None = None,
    diagnosis_registry: LiveDiagnosisRegistry | None = None,
) -> FastAPI:
    """Compose Local V1 with read-only OPC-UA, forecasting, and diagnosis extensions."""

    app = create_base_app(artifact_root=artifact_root, workspace_root=workspace_root)
    workspace = WorkspaceStore(workspace_root)
    attach_opcua_routes(app, workspace, opcua_registry)
    attach_forecasting_routes(app, workspace, forecasting_registry)
    attach_diagnosis_routes(app, workspace, diagnosis_registry)
    return app
