from __future__ import annotations

import json

import pandas as pd

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.workspace import WorkspaceStore, project_from_dict


def _project(location: str | None = None) -> ProjectSpec:
    return ProjectSpec(
        name="Process Workspace Demo",
        description="workspace persistence smoke",
        data_sources=(
            DataSourceSpec(
                name="process-csv" if location else "memory",
                kind=DataSourceKind.CSV if location else DataSourceKind.MEMORY,
                location=location,
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
                context_length=8,
                horizon=2,
            ),
        ),
        tags=("workspace",),
    )


def test_workspace_roundtrip_and_duplicate_protection(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    project = _project()

    record = store.create_project(project)
    loaded = store.get_project(record["project_id"])

    assert record["project_id"] == "process-workspace-demo"
    assert loaded.to_dict() == project.to_dict()
    assert len(store.list_projects()) == 1
    try:
        store.create_project(project)
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate project creation must fail closed")


def test_project_from_dict_parses_public_contract() -> None:
    payload = _project().to_dict()
    parsed = project_from_dict(payload)

    assert parsed.name == "Process Workspace Demo"
    assert parsed.data_sources[0].kind == DataSourceKind.MEMORY
    assert parsed.tasks[0].task_type == TaskType.FORECASTING


def test_workspace_derived_artifacts_remain_under_project_directory(tmp_path) -> None:
    store = WorkspaceStore(tmp_path / "workspace")
    project = _project()
    record = store.create_project(project)
    path = store.write_derived_artifact(
        record["project_id"],
        "data audit / source",
        {"status": "ok", "rows": 10},
    )

    assert path.parent.name == "derived"
    assert path.name == "data-audit-source.json"
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 10


def test_csv_project_spec_is_json_roundtrippable(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "y": [1.0]}).to_csv(path, index=False)
    project = _project(str(path))

    payload = project.to_dict()
    assert project_from_dict(payload).to_dict() == payload
