from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .connectors import ConnectorError, load_data_source
from .contracts import DataSourceKind, TaskType
from .data_audit import audit_dataframe
from .model_catalog import model_catalog
from .services import (
    materialize_project_application,
    route_project_forecast,
    run_project_anomaly,
    run_project_cross_source_shift,
    run_project_lag_analysis,
    run_project_shift_analysis,
    run_project_variable_discovery,
)
from .ui import render_dashboard
from .workspace import WorkspaceStore, project_from_dict


class ApplicationArtifactStore:
    """Read-only view over materialized platform application artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _application_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(
            {path.parent for path in self.root.rglob("application.json")},
            key=lambda path: str(path),
        )

    def list_applications(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for directory in self._application_dirs():
            payload = self._read_json(directory / "application.json")
            payload["artifact_dir"] = str(directory)
            payload["available_artifacts"] = sorted(
                path.name for path in directory.iterdir() if path.is_file()
            )
            rows.append(payload)
        return rows

    def resolve(self, application_id: str) -> Path:
        matches: list[Path] = []
        for directory in self._application_dirs():
            payload = self._read_json(directory / "application.json")
            if str(payload.get("application_id")) == application_id:
                matches.append(directory)
        if not matches:
            raise KeyError(application_id)
        if len(matches) > 1:
            raise RuntimeError(f"application_id {application_id!r} is not unique")
        return matches[0]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object in {path}")
        return value

    def read_artifact(self, application_id: str, filename: str) -> dict[str, Any]:
        allowed = {
            "application.json",
            "project.json",
            "data_provenance.json",
            "data_audit.json",
            "model_route.json",
            "replay_snapshot.json",
            "anomaly_result.json",
        }
        if filename not in allowed:
            raise ValueError(f"unsupported application artifact: {filename}")
        return self._read_json(self.resolve(application_id) / filename)


def platform_capabilities() -> dict[str, Any]:
    return {
        "schema_version": "industrial_tsfm.capabilities.v1",
        "task_types": [task.value for task in TaskType],
        "implemented_product_routes": [
            TaskType.FORECASTING.value,
            TaskType.ANOMALY.value,
        ],
        "analytics": [
            "lagged_association",
            "chronological_distribution_shift",
            "cross_source_distribution_shift",
            "variable_candidate_discovery",
        ],
        "data_source_contracts": [kind.value for kind in DataSourceKind],
        "implemented_local_connectors": [
            DataSourceKind.CSV.value,
            DataSourceKind.PARQUET.value,
            DataSourceKind.MEMORY.value,
        ],
        "streaming_connectors": {
            "mqtt": "contract_only",
            "opcua": "contract_only",
            "sql": "contract_only",
        },
        "models": model_catalog(),
        "application_runtime": "offline_replay",
        "workspace_store": "local_json_v1",
        "dashboard": "server_rendered_v1",
        "agent_orchestration": "not_enabled_in_v1",
    }


def _artifact_or_404(
    store: ApplicationArtifactStore,
    application_id: str,
    filename: str,
) -> dict[str, Any]:
    try:
        return store.read_artifact(application_id, filename)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="application artifact not found") from exc


def _project_or_404(workspace: WorkspaceStore, project_id: str):
    try:
        return workspace.get_project(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc


def _required_name(payload: dict[str, Any], key: str) -> str:
    name = str(payload.get(key, "")).strip()
    if not name:
        raise HTTPException(status_code=422, detail=f"{key} is required")
    return name


def _source_name(payload: dict[str, Any]) -> str:
    return _required_name(payload, "source_name")


def create_app(
    artifact_root: str | Path = "results",
    workspace_root: str | Path = ".industsfm/workspace",
) -> FastAPI:
    """Create the V1 local platform API.

    Materialized applications are generated from explicit project contracts and
    validation evidence. Remote authentication, arbitrary uploads, and live OT
    control are deliberately out of scope for this V1 service.
    """

    artifacts = ApplicationArtifactStore(artifact_root)
    workspace = WorkspaceStore(workspace_root)
    app = FastAPI(
        title="IndusTSFM Industrial Intelligence Platform",
        version="0.1.0",
        description=(
            "Product API for industrial projects, audited data, validation-only model routing, "
            "anomaly triage, lag/shift/variable-discovery analytics, and deterministic application "
            "replay."
        ),
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return platform_capabilities()

    @app.get("/v1/projects")
    def projects() -> dict[str, Any]:
        rows = workspace.list_projects()
        return {"count": len(rows), "projects": rows}

    @app.post("/v1/projects", status_code=201)
    def create_project(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            project = project_from_dict(payload)
            return workspace.create_project(project)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail="project already exists") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/projects/{project_id}")
    def project(project_id: str) -> dict[str, Any]:
        try:
            return workspace.get_project_record(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post("/v1/projects/{project_id}/audit/{source_name}")
    def audit_project_source(project_id: str, source_name: str) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        matches = [source for source in spec.data_sources if source.name == source_name]
        if len(matches) != 1:
            raise HTTPException(status_code=404, detail="data source not found")
        source = matches[0]
        if source.kind == DataSourceKind.MEMORY:
            raise HTTPException(
                status_code=400,
                detail="memory sources cannot be loaded through the persisted workspace API",
            )
        try:
            loaded = load_data_source(source)
        except (ConnectorError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target_columns = tuple(
            dict.fromkeys(target for task in spec.tasks for target in task.target_columns)
        )
        audit = audit_dataframe(
            loaded.frame,
            timestamp_column=source.timestamp_column,
            target_columns=target_columns,
        )
        result = {
            "schema_version": "industrial_tsfm.workspace_audit.v1",
            "project_id": project_id,
            "source_name": source.name,
            "provenance": loaded.provenance,
            "audit": audit,
        }
        workspace.write_derived_artifact(project_id, f"data-audit-{source.name}", result)
        return result

    @app.post("/v1/projects/{project_id}/analysis/lag")
    def lag_analysis(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            result = run_project_lag_analysis(spec, _source_name(payload), payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(project_id, "lag-analysis", result)
        return result

    @app.post("/v1/projects/{project_id}/analysis/shift")
    def shift_analysis(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            result = run_project_shift_analysis(spec, _source_name(payload), payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(project_id, "distribution-shift", result)
        return result

    @app.post("/v1/projects/{project_id}/analysis/shift-sources")
    def cross_source_shift(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        reference_name = _required_name(payload, "reference_source_name")
        target_name = _required_name(payload, "target_source_name")
        try:
            result = run_project_cross_source_shift(
                spec,
                reference_name,
                target_name,
                payload,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(
            project_id,
            f"cross-source-shift-{reference_name}-{target_name}",
            result,
        )
        return result

    @app.post("/v1/projects/{project_id}/analysis/discover")
    def variable_discovery(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            result = run_project_variable_discovery(spec, _source_name(payload), payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(project_id, "variable-discovery", result)
        return result

    @app.post("/v1/projects/{project_id}/route/{task_name}")
    def route_project(project_id: str, task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            result = route_project_forecast(spec, _source_name(payload), task_name, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source or task not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(project_id, f"model-route-{task_name}", result)
        return result

    @app.post("/v1/projects/{project_id}/applications/{task_name}", status_code=201)
    def create_project_application(
        project_id: str,
        task_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            application = materialize_project_application(
                spec,
                _source_name(payload),
                task_name,
                payload,
                artifacts.root,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source or task not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return application.manifest()

    @app.post("/v1/projects/{project_id}/anomaly/{task_name}")
    def run_anomaly(project_id: str, task_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = _project_or_404(workspace, project_id)
        try:
            result = run_project_anomaly(spec, _source_name(payload), task_name, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project source or task not found") from exc
        except (ConnectorError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        workspace.write_derived_artifact(project_id, f"anomaly-{task_name}", result)
        return result

    @app.get("/v1/applications")
    def applications() -> dict[str, Any]:
        rows = artifacts.list_applications()
        return {"count": len(rows), "applications": rows}

    @app.get("/v1/applications/{application_id}")
    def application(application_id: str) -> dict[str, Any]:
        return _artifact_or_404(artifacts, application_id, "application.json")

    @app.get("/v1/applications/{application_id}/audit")
    def application_audit(application_id: str) -> dict[str, Any]:
        return _artifact_or_404(artifacts, application_id, "data_audit.json")

    @app.get("/v1/applications/{application_id}/route")
    def application_route(application_id: str) -> dict[str, Any]:
        return _artifact_or_404(artifacts, application_id, "model_route.json")

    @app.get("/v1/applications/{application_id}/replay")
    def application_replay(application_id: str) -> dict[str, Any]:
        return _artifact_or_404(artifacts, application_id, "replay_snapshot.json")

    @app.get("/v1/applications/{application_id}/anomaly")
    def application_anomaly(application_id: str) -> dict[str, Any]:
        return _artifact_or_404(artifacts, application_id, "anomaly_result.json")

    return app
