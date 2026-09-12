from __future__ import annotations

import json

import pandas as pd

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProductRoutingRequest,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    build_platform_application,
    get_model_descriptor,
    model_catalog,
)


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "chronos2",
                "seed": 0,
                "status": "complete",
                "normalized_rmse": 0.41,
                "fit_seconds": 0.0,
                "inference_seconds": 0.02,
                "total_parameters": 120_000_000,
                "training_mode": "frozen",
            },
            {
                "model": "patchtst",
                "seed": 0,
                "status": "complete",
                "normalized_rmse": 0.52,
                "fit_seconds": 1.0,
                "inference_seconds": 0.005,
                "total_parameters": 70_000,
                "training_mode": "supervised",
            },
        ]
    )


def test_application_orchestrator_materializes_auditable_bundle(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="5min"),
            "x": [float(value) for value in range(24)],
            "y": [float(value * 2) for value in range(24)],
        }
    )
    source = DataSourceSpec(
        name="memory-source",
        kind=DataSourceKind.MEMORY,
        timestamp_column="timestamp",
    )
    project = ProjectSpec(
        name="Application Demo",
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

    app, loaded = build_platform_application(
        project,
        source_name="memory-source",
        task_name="forecast-y",
        validation_table=_validation(),
        routing_request=ProductRoutingRequest(
            task_name="forecast-y",
            target_data_fraction=0.10,
            support_windows=8,
        ),
        memory_frames={"memory-source": frame},
        replay_batch_size=6,
    )
    root = app.write(tmp_path / "app")

    assert loaded.provenance["kind"] == "memory"
    assert app.status == "ready"
    assert app.manifest()["selected_model"] == "chronos2"
    assert app.manifest()["target_labels_used"] is False
    assert app.replay["total_batches"] == 4
    for filename in (
        "application.json",
        "project.json",
        "data_provenance.json",
        "data_audit.json",
        "model_route.json",
        "replay_snapshot.json",
        "validation_evidence.csv",
    ):
        assert (root / filename).exists()
    json.loads((root / "application.json").read_text(encoding="utf-8"))


def test_model_catalog_only_surfaces_implemented_forecasting_adapters() -> None:
    catalog = model_catalog()
    names = {row["name"] for row in catalog}

    assert {"chronos2", "timesfm", "moirai2", "patchtst"}.issubset(names)
    assert get_model_descriptor("chronos2").native_multivariate is True
    assert get_model_descriptor("timesfm_transformers").training_modes == ("peft",)
    assert all(row["task_types"] == ["forecasting"] for row in catalog)
