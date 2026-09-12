from __future__ import annotations

import json

import pandas as pd

from industrial_tsfm.platform import (
    DataReplayRuntime,
    DataSourceKind,
    DataSourceSpec,
    ProductRoutingRequest,
    ProjectSpec,
    ReplayConfig,
    TaskDefinition,
    TaskType,
    audit_dataframe,
    load_data_source,
    route_task,
)


def _project(source: DataSourceSpec) -> ProjectSpec:
    return ProjectSpec(
        name="router-demo",
        data_sources=(source,),
        tasks=(
            TaskDefinition(
                name="forecast-y",
                task_type=TaskType.FORECASTING,
                target_columns=("y",),
                context_length=8,
                horizon=2,
            ),
        ),
    )


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "chronos2",
                "seed": 0,
                "status": "complete",
                "normalized_rmse": 0.40,
                "fit_seconds": 0.0,
                "inference_seconds": 0.02,
                "total_parameters": 120_000_000,
                "training_mode": "frozen",
            },
            {
                "model": "patchtst",
                "seed": 0,
                "status": "complete",
                "normalized_rmse": 0.55,
                "fit_seconds": 1.0,
                "inference_seconds": 0.005,
                "total_parameters": 70_000,
                "training_mode": "supervised",
            },
        ]
    )


def test_csv_connector_records_provenance_and_preserves_order(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:10:00", "2026-01-01 00:00:00"],
            "x": [2.0, 1.0],
            "y": [3.0, 2.0],
        }
    )
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    source = DataSourceSpec(
        name="csv-source",
        kind=DataSourceKind.CSV,
        location=str(path),
        timestamp_column="timestamp",
    )

    loaded = load_data_source(source)

    assert loaded.frame["x"].tolist() == [2.0, 1.0]
    assert loaded.provenance["rows"] == 2
    assert len(str(loaded.provenance["sha256"])) == 64
    assert loaded.provenance["kind"] == "csv"


def test_product_router_wraps_validation_only_engine() -> None:
    source = DataSourceSpec(name="memory", kind=DataSourceKind.MEMORY, timestamp_column="timestamp")
    project = _project(source)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="10min"),
            "x": range(32),
            "y": [float(value) for value in range(32)],
        }
    )
    audit = audit_dataframe(frame, timestamp_column="timestamp", target_columns=("y",))

    decision = route_task(
        project,
        _validation(),
        audit,
        ProductRoutingRequest(
            task_name="forecast-y",
            target_data_fraction=0.10,
            support_windows=10,
            max_parameters=200_000_000,
        ),
    )

    assert decision["status"] == "ready"
    assert decision["selected_model"] == "chronos2"
    assert decision["selected_strategy"] == "target_residual_calibration"
    assert decision["selection_evidence"] == "validation_only"
    assert decision["target_labels_used"] is False


def test_product_router_blocks_missing_target() -> None:
    source = DataSourceSpec(name="memory", kind=DataSourceKind.MEMORY)
    project = _project(source)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    audit = audit_dataframe(frame, target_columns=("y",))

    decision = route_task(
        project,
        _validation(),
        audit,
        ProductRoutingRequest(task_name="forecast-y", min_readiness_score=0.0),
    )

    assert decision["status"] == "blocked"
    assert decision["selected_model"] is None
    assert any("missing" in reason for reason in decision["blocking_reasons"])


def test_replay_runtime_is_deterministic_and_batched() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
            "x": [4, 3, 2, 1, 0],
        }
    )
    runtime = DataReplayRuntime(frame, ReplayConfig(timestamp_column="timestamp", batch_size=2))
    batches = list(runtime.iter_batches())
    snapshot = runtime.snapshot(preview_batches=2)

    assert [batch.row_start for batch in batches] == [0, 2, 4]
    assert batches[0].records[0]["x"] == 4
    assert snapshot["total_batches"] == 3
    assert snapshot["preserves_input_order"] is True
    assert snapshot["wall_clock_sleep"] is False


def test_project_serialization_uses_product_enum_values() -> None:
    source = DataSourceSpec(name="memory", kind=DataSourceKind.MEMORY)
    payload = _project(source).to_dict()

    assert payload["data_sources"][0]["kind"] == "memory"
    assert payload["tasks"][0]["task_type"] == "forecasting"
    json.dumps(payload)
