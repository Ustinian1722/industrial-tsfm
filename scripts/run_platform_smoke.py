from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    AnomalyDetectionRequest,
    DataSourceKind,
    DataSourceSpec,
    LagAnalysisRequest,
    ProductRoutingRequest,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    analyze_lagged_relationships,
    build_platform_application,
    build_platform_report,
    model_catalog,
    run_pca_spe_anomaly_detection,
)


def build_fixture(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    time = pd.date_range("2026-01-01", periods=rows, freq="10min")
    load = 60 + 8 * np.sin(np.linspace(0, 10, rows)) + rng.normal(0, 1.0, rows)
    temperature = 40 + 0.25 * load + rng.normal(0, 0.8, rows)
    pressure = 2.1 + 0.015 * load + rng.normal(0, 0.03, rows)
    temperature[200:208] += 12.0
    pressure[200:208] += 0.8
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
        name="Tennessee-style Process Intelligence Demo",
        description=(
            "Synthetic process data used only to validate product-facing data audit, model routing, "
            "anomaly triage, lag analysis, API artifacts, and replay."
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
            TaskDefinition(
                name="process-anomaly",
                task_type=TaskType.ANOMALY,
                constraints={"feature_columns": ["load", "temperature", "pressure"]},
                business_kpi="surface abnormal process excursions before operator intervention",
            ),
        ),
        tags=("platform-smoke", "industrial-ai"),
    )

    application, loaded = build_platform_application(
        project,
        source_name=source.name,
        task_name="temperature-forecast",
        validation_table=build_validation_table(),
        routing_request=ProductRoutingRequest(
            task_name="temperature-forecast",
            target_data_fraction=0.10,
            support_windows=20,
            max_parameters=250_000_000,
            max_inference_seconds=0.05,
        ),
        replay_batch_size=32,
    )
    application.write(output_dir)

    anomaly = run_pca_spe_anomaly_detection(
        project,
        loaded.frame,
        application.data_audit,
        AnomalyDetectionRequest(
            task_name="process-anomaly",
            train_fraction=0.60,
            threshold_quantile=0.99,
            min_train_rows=32,
        ),
    )
    lag_analysis = analyze_lagged_relationships(
        loaded.frame,
        application.data_audit,
        LagAnalysisRequest(
            target_column="temperature",
            feature_columns=("load", "pressure"),
            max_lag=6,
            min_pairs=32,
        ),
    )
    (output_dir / "anomaly_result.json").write_text(
        json.dumps(anomaly, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "lag_analysis.json").write_text(
        json.dumps(lag_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "model_catalog.json").write_text(
        json.dumps(model_catalog(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_platform_report(
        project,
        application.data_audit,
        output_dir / "index.html",
        route_decision=application.model_route,
        replay_snapshot=application.replay,
        provenance=application.source_provenance,
    )

    print(f"application={application.application_id}")
    print(f"status={application.status}")
    print(f"selected_model={application.model_route.get('selected_model')}")
    print(f"selected_strategy={application.model_route.get('selected_strategy')}")
    print(f"anomaly_points={anomaly['summary']['anomaly_points_test']}")
    print(f"lag_top={lag_analysis['rankings'][0]['feature']}")
    print(f"replay_batches={application.replay['total_batches']}")
    print(report)


if __name__ == "__main__":
    main()
