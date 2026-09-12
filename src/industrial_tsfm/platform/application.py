from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .connectors import LoadedDataSource, load_data_source
from .contracts import DataSourceSpec, ProjectSpec, TaskDefinition
from .data_audit import audit_dataframe
from .model_router import ProductRoutingRequest, route_task
from .runtime import DataReplayRuntime, ReplayConfig


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return compact or "application"


def _find_source(project: ProjectSpec, name: str) -> DataSourceSpec:
    matches = [source for source in project.data_sources if source.name == name]
    if len(matches) != 1:
        raise ValueError(f"data source {name!r} must exist exactly once")
    return matches[0]


def _find_task(project: ProjectSpec, name: str) -> TaskDefinition:
    matches = [task for task in project.tasks if task.name == name]
    if len(matches) != 1:
        raise ValueError(f"task {name!r} must exist exactly once")
    return matches[0]


@dataclass(frozen=True)
class PlatformApplication:
    application_id: str
    project: dict[str, Any]
    source_provenance: dict[str, Any]
    data_audit: dict[str, Any]
    model_route: dict[str, Any]
    replay: dict[str, Any]
    validation_evidence: tuple[dict[str, Any], ...]

    @property
    def status(self) -> str:
        return "ready" if self.model_route.get("route_ready") else "blocked"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.application.v1",
            "application_id": self.application_id,
            "project": self.project.get("name"),
            "status": self.status,
            "source": self.source_provenance.get("source_name"),
            "source_kind": self.source_provenance.get("kind"),
            "task": self.model_route.get("task"),
            "task_type": self.model_route.get("task_type"),
            "selected_model": self.model_route.get("selected_model"),
            "selected_strategy": self.model_route.get("selected_strategy"),
            "selection_evidence": self.model_route.get("selection_evidence"),
            "target_labels_used": self.model_route.get("target_labels_used", False),
            "runtime": {
                "mode": "offline_replay",
                "batch_size": self.replay.get("batch_size"),
                "preserves_input_order": self.replay.get("preserves_input_order"),
                "wall_clock_sleep": self.replay.get("wall_clock_sleep"),
            },
        }

    def write(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {
            "project.json": self.project,
            "data_provenance.json": self.source_provenance,
            "data_audit.json": self.data_audit,
            "model_route.json": self.model_route,
            "replay_snapshot.json": self.replay,
            "application.json": self.manifest(),
        }
        for filename, payload in artifacts.items():
            (root / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        pd.DataFrame(self.validation_evidence).to_csv(
            root / "validation_evidence.csv", index=False
        )
        return root


def build_platform_application(
    project: ProjectSpec,
    *,
    source_name: str,
    task_name: str,
    validation_table: pd.DataFrame,
    routing_request: ProductRoutingRequest,
    memory_frames: Mapping[str, pd.DataFrame] | None = None,
    replay_batch_size: int = 32,
) -> tuple[PlatformApplication, LoadedDataSource]:
    """Build a deterministic product application from existing IndusTSFM evidence.

    This function is the first non-LLM orchestration layer: data loading, audit,
    validation-only model routing, and application replay are composed into one
    auditable application bundle. It does not execute target-test evaluation.
    """

    project.validate()
    if routing_request.task_name != task_name:
        raise ValueError("routing_request.task_name must match task_name")
    source = _find_source(project, source_name)
    task = _find_task(project, task_name)
    loaded = load_data_source(source, memory_frames=memory_frames)
    audit = audit_dataframe(
        loaded.frame,
        timestamp_column=source.timestamp_column,
        target_columns=task.target_columns,
    )
    route = route_task(project, validation_table, audit, routing_request)
    replay = DataReplayRuntime(
        loaded.frame,
        ReplayConfig(timestamp_column=source.timestamp_column, batch_size=replay_batch_size),
    ).snapshot(preview_batches=3)
    app = PlatformApplication(
        application_id=f"{_slug(project.name)}--{_slug(task.name)}",
        project=project.to_dict(),
        source_provenance=dict(loaded.provenance),
        data_audit=audit,
        model_route=route,
        replay=replay,
        validation_evidence=tuple(
            validation_table.where(pd.notna(validation_table), None).to_dict(orient="records")
        ),
    )
    return app, loaded
