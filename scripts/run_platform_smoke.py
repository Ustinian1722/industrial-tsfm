from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    build_platform_report,
    load_data_source,
    route_task,
)


def build_fixture(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    time = pd.date_range("2026-01-01", periods=rows, freq="10min")
    load = 60 + 8 * np.sin(np.linspace(0, 10, rows)) + rng.normal(0, 1.0, rows)
    temperature = 40 + 0.25 * load + rng.normal(0, 0.8, rows)
    pressure = 2.1 + 0.015 * load + rng.normal(0, 0.03, rows)
    target = np.roll(temperature, -1)
    target[-1] = target[-2]
    frame = pd.DataFrame(
        {
            "timestamp": time,
            "load": load,
            "temperature": temperature,
            "pressure": pressure,
            "temperature_next": target,
            "dead_tag": 1.0,
        }
    )
    frame.loc[25:34, "pressure"] = np.nan
    return frame


def build_validation_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    profiles = {
        "chronos2": (0.46, 0.015, 120_000_000, "frozen"),
        "timesfm": (0.49, 0.021, 231_000_000, "frozen"),
        "patchtst": (0.53, 0.006, 70_000, "supervised"),
    }
    for model, (metric, latency, parameters, mode) in profiles.items():
        for seed, delta in enumerate((0.0, 0.01, -0.006)):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "complete",
                    "normalized_rmse": metric + delta,
                    "fit_seconds": 0.0 if mode == "frozen" else 1.2,
                    "inference_seconds": latency,
                    "total_parameters": parameters,
                    "training_mode": mode,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = Path("results/platform-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = output_dir / "synthetic_process.csv"
    build_fixture().to_csv(fixture_path, index=False)

    source = DataSourceSpec(
        name="synthetic-process",
        kind=DataSourceKind.CSV,
        location=str(fixture_path),
        timestamp_column="timestamp",
    )
    project = ProjectSpec(
        name="Tennessee-style Process Forecast Demo",
        description=(
            "Synthetic process data used only to validate the product-facing data, routing, "
            "and replay path."
        ),
        data_sources=(source,),
        tasks=(
            TaskDefinition(
                name="temperature-forecast",
                task_type=TaskType.FORECASTING,
                target_columns=("temperature_next",),
                context_length=32,
                horizon=8,
                business_kpi="keep process temperature inside the operating envelope",
            ),
        ),
        tags=("platform-smoke", "industrial-ai"),
    )

    loaded = load_data_source(source)
    audit = audit_dataframe(
        loaded.frame,
        timestamp_column=source.timestamp_column,
        target_columns=("temperature_next",),
    )
    validation = build_validation_table()
    route = route_task(
        project,
        validation,
        audit,
        ProductRoutingRequest(
            task_name="temperature-forecast",
            target_data_fraction=0.10,
            support_windows=20,
            max_parameters=250_000_000,
            max_inference_seconds=0.05,
        ),
    )
    replay = DataReplayRuntime(
        loaded.frame,
        ReplayConfig(timestamp_column="timestamp", batch_size=32),
    ).snapshot(preview_batches=3)

    artifacts = {
        "project.json": project.to_dict(),
        "data_provenance.json": loaded.provenance,
        "data_audit.json": audit,
        "model_route.json": route,
        "replay_snapshot.json": replay,
    }
    for filename, payload in artifacts.items():
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    validation.to_csv(output_dir / "validation_evidence.csv", index=False)

    report = build_platform_report(
        project,
        audit,
        output_dir / "index.html",
        route_decision=route,
        replay_snapshot=replay,
        provenance=loaded.provenance,
    )
    print(f"readiness={audit['summary']['readiness_score']}")
    print(f"selected_model={route.get('selected_model')}")
    print(f"selected_strategy={route.get('selected_strategy')}")
    print(f"replay_batches={replay['total_batches']}")
    print(report)


if __name__ == "__main__":
    main()
