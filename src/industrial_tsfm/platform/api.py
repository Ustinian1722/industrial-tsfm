from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .contracts import DataSourceKind, TaskType
from .model_catalog import model_catalog


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
            raise ValueError(f"expected JSON object in {path}")
        return value

    def read_artifact(self, application_id: str, filename: str) -> dict[str, Any]:
        allowed = {
            "application.json",
            "project.json",
            "data_provenance.json",
            "data_audit.json",
            "model_route.json",
            "replay_snapshot.json",
        }
        if filename not in allowed:
            raise ValueError(f"unsupported application artifact: {filename}")
        return self._read_json(self.resolve(application_id) / filename)


def platform_capabilities() -> dict[str, Any]:
    return {
        "schema_version": "industrial_tsfm.capabilities.v1",
        "task_types": [task.value for task in TaskType],
        "implemented_product_routes": [TaskType.FORECASTING.value],
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
        "agent_orchestration": "not_enabled_in_v1",
    }


def create_app(artifact_root: str | Path = "results") -> FastAPI:
    """Create the read-only V1 platform API over materialized applications."""

    store = ApplicationArtifactStore(artifact_root)
    app = FastAPI(
        title="IndusTSFM Industrial Intelligence Platform",
        version="0.1.0",
        description=(
            "Product API for audited industrial data, validation-only model routing, "
            "and deterministic application replay."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return platform_capabilities()

    @app.get("/v1/applications")
    def applications() -> dict[str, Any]:
        rows = store.list_applications()
        return {"count": len(rows), "applications": rows}

    @app.get("/v1/applications/{application_id}")
    def application(application_id: str) -> dict[str, Any]:
        try:
            return store.read_artifact(application_id, "application.json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="application not found") from exc

    @app.get("/v1/applications/{application_id}/audit")
    def application_audit(application_id: str) -> dict[str, Any]:
        try:
            return store.read_artifact(application_id, "data_audit.json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="application not found") from exc

    @app.get("/v1/applications/{application_id}/route")
    def application_route(application_id: str) -> dict[str, Any]:
        try:
            return store.read_artifact(application_id, "model_route.json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="application not found") from exc

    @app.get("/v1/applications/{application_id}/replay")
    def application_replay(application_id: str) -> dict[str, Any]:
        try:
            return store.read_artifact(application_id, "replay_snapshot.json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="application not found") from exc

    return app
