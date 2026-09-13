from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import DataSourceKind, DataSourceSpec, ProjectSpec, TaskDefinition, TaskType


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return text or "project"


def project_from_dict(payload: dict[str, Any]) -> ProjectSpec:
    """Parse a JSON-compatible product project without accepting unknown magic."""

    sources = tuple(
        DataSourceSpec(
            name=str(item["name"]),
            kind=DataSourceKind(str(item["kind"])),
            location=item.get("location"),
            timestamp_column=item.get("timestamp_column"),
            entity_column=item.get("entity_column"),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("data_sources", [])
    )
    tasks = tuple(
        TaskDefinition(
            name=str(item["name"]),
            task_type=TaskType(str(item["task_type"])),
            target_columns=tuple(str(value) for value in item.get("target_columns", [])),
            context_length=item.get("context_length"),
            horizon=item.get("horizon"),
            constraints=dict(item.get("constraints", {})),
            business_kpi=item.get("business_kpi"),
        )
        for item in payload.get("tasks", [])
    )
    project = ProjectSpec(
        name=str(payload.get("name", "")),
        description=str(payload.get("description", "")),
        data_sources=sources,
        tasks=tasks,
        tags=tuple(str(value) for value in payload.get("tags", [])),
    )
    project.validate()
    return project


class WorkspaceStore:
    """Small local workspace store for project specifications and derived artifacts.

    V1 intentionally stores transparent JSON on disk rather than introducing a
    database. The API can later swap this implementation for PostgreSQL without
    changing the project contract.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.projects_root = self.root / "projects"

    def create_project(
        self,
        project: ProjectSpec,
        *,
        project_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        project.validate()
        resolved_id = _slug(project_id or project.name)
        directory = self.projects_root / resolved_id
        record_path = directory / "project.json"
        if record_path.exists() and not overwrite:
            raise FileExistsError(resolved_id)
        directory.mkdir(parents=True, exist_ok=True)
        previous = self._read_json(record_path) if record_path.exists() else None
        created_at = (
            str(previous.get("created_at"))
            if previous and previous.get("created_at")
            else _utc_now()
        )
        record = {
            "schema_version": "industrial_tsfm.workspace_project.v1",
            "project_id": resolved_id,
            "created_at": created_at,
            "updated_at": _utc_now(),
            "spec": project.to_dict(),
        }
        self._write_json(record_path, record)
        return record

    def list_projects(self) -> list[dict[str, Any]]:
        if not self.projects_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.projects_root.glob("*/project.json")):
            rows.append(self._read_json(path))
        return rows

    def get_project_record(self, project_id: str) -> dict[str, Any]:
        path = self.projects_root / _slug(project_id) / "project.json"
        if not path.exists():
            raise KeyError(project_id)
        return self._read_json(path)

    def get_project(self, project_id: str) -> ProjectSpec:
        record = self.get_project_record(project_id)
        return project_from_dict(dict(record["spec"]))

    def _derived_path(self, project_id: str, relative_name: str) -> Path:
        self.get_project_record(project_id)
        safe_name = _slug(relative_name)
        return self.projects_root / _slug(project_id) / "derived" / f"{safe_name}.json"

    def write_derived_artifact(
        self,
        project_id: str,
        relative_name: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self._derived_path(project_id, relative_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(path, payload)
        return path

    def read_derived_artifact(self, project_id: str, relative_name: str) -> dict[str, Any]:
        path = self._derived_path(project_id, relative_name)
        if not path.exists():
            raise KeyError(relative_name)
        return self._read_json(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object in {path}")
        return value

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
