from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .analytics import LagAnalysisRequest, analyze_lagged_relationships
from .anomaly_engine import AnomalyDetectionRequest, run_pca_spe_anomaly_detection
from .application import PlatformApplication, build_platform_application
from .connectors import LoadedDataSource, load_data_source
from .contracts import DataSourceSpec, ProjectSpec
from .data_audit import audit_dataframe
from .model_router import ProductRoutingRequest, route_task
from .shift_analysis import DistributionShiftRequest, analyze_distribution_shift


def find_source(project: ProjectSpec, source_name: str) -> DataSourceSpec:
    matches = [source for source in project.data_sources if source.name == source_name]
    if len(matches) != 1:
        raise KeyError(source_name)
    return matches[0]


def audit_source(
    project: ProjectSpec,
    source_name: str,
) -> tuple[LoadedDataSource, dict[str, Any]]:
    source = find_source(project, source_name)
    loaded = load_data_source(source)
    targets = tuple(
        dict.fromkeys(target for task in project.tasks for target in task.target_columns)
    )
    audit = audit_dataframe(
        loaded.frame,
        timestamp_column=source.timestamp_column,
        target_columns=targets,
    )
    return loaded, audit


def validation_table_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    records = payload.get("validation_evidence", [])
    if not isinstance(records, list) or not records:
        raise ValueError("validation_evidence must be a non-empty list of records")
    table = pd.DataFrame(records)
    required = {"model", "status", "normalized_rmse", "training_mode"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("validation_evidence is missing columns: " + ", ".join(missing))
    defaults: dict[str, Any] = {
        "seed": 0,
        "fit_seconds": 0.0,
        "inference_seconds": 0.0,
        "total_parameters": 0,
    }
    for column, default in defaults.items():
        if column not in table.columns:
            table[column] = default
    return table


def routing_request_from_payload(task_name: str, payload: dict[str, Any]) -> ProductRoutingRequest:
    route = dict(payload.get("routing", {}))
    return ProductRoutingRequest(
        task_name=task_name,
        target_data_fraction=float(route.get("target_data_fraction", 0.0)),
        support_windows=route.get("support_windows"),
        shift_score=float(route.get("shift_score", 0.0)),
        max_adaptation_seconds=route.get("max_adaptation_seconds"),
        max_parameters=route.get("max_parameters"),
        max_inference_seconds=route.get("max_inference_seconds"),
        min_readiness_score=float(route.get("min_readiness_score", 40.0)),
        metric=str(route.get("metric", "normalized_rmse")),
    )


def route_project_forecast(
    project: ProjectSpec,
    source_name: str,
    task_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _, audit = audit_source(project, source_name)
    validation = validation_table_from_payload(payload)
    request = routing_request_from_payload(task_name, payload)
    return route_task(project, validation, audit, request)


def materialize_project_application(
    project: ProjectSpec,
    source_name: str,
    task_name: str,
    payload: dict[str, Any],
    output_root: str | Path,
) -> PlatformApplication:
    validation = validation_table_from_payload(payload)
    request = routing_request_from_payload(task_name, payload)
    replay_batch_size = int(payload.get("replay_batch_size", 32))
    if replay_batch_size <= 0:
        raise ValueError("replay_batch_size must be positive")
    application, _ = build_platform_application(
        project,
        source_name=source_name,
        task_name=task_name,
        validation_table=validation,
        routing_request=request,
        replay_batch_size=replay_batch_size,
    )
    output_dir = Path(output_root) / application.application_id
    application.write(output_dir)
    return application


def anomaly_request_from_payload(task_name: str, payload: dict[str, Any]) -> AnomalyDetectionRequest:
    config = dict(payload.get("anomaly", {}))
    return AnomalyDetectionRequest(
        task_name=task_name,
        train_fraction=float(config.get("train_fraction", 0.50)),
        threshold_quantile=float(config.get("threshold_quantile", 0.995)),
        n_components=config.get("n_components", 0.95),
        min_train_rows=int(config.get("min_train_rows", 16)),
        top_k_sensors=int(config.get("top_k_sensors", 5)),
    )


def run_project_anomaly(
    project: ProjectSpec,
    source_name: str,
    task_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    loaded, audit = audit_source(project, source_name)
    request = anomaly_request_from_payload(task_name, payload)
    return run_pca_spe_anomaly_detection(project, loaded.frame, audit, request)


def lag_request_from_payload(payload: dict[str, Any]) -> LagAnalysisRequest:
    config = dict(payload.get("analysis", {}))
    target_column = str(config.get("target_column", "")).strip()
    features = tuple(str(value) for value in config.get("feature_columns", []))
    return LagAnalysisRequest(
        target_column=target_column,
        feature_columns=features,
        max_lag=int(config.get("max_lag", 12)),
        min_pairs=int(config.get("min_pairs", 12)),
        top_k=int(config.get("top_k", 12)),
    )


def run_project_lag_analysis(
    project: ProjectSpec,
    source_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    loaded, audit = audit_source(project, source_name)
    request = lag_request_from_payload(payload)
    return analyze_lagged_relationships(loaded.frame, audit, request)


def shift_request_from_payload(payload: dict[str, Any]) -> DistributionShiftRequest:
    config = dict(payload.get("analysis", {}))
    features = tuple(str(value) for value in config.get("feature_columns", []))
    return DistributionShiftRequest(
        feature_columns=features,
        reference_fraction=float(config.get("reference_fraction", 0.50)),
        target_fraction=float(config.get("target_fraction", 0.50)),
        min_rows_per_partition=int(config.get("min_rows_per_partition", 16)),
        top_k=int(config.get("top_k", 12)),
    )


def run_project_shift_analysis(
    project: ProjectSpec,
    source_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    loaded, audit = audit_source(project, source_name)
    request = shift_request_from_payload(payload)
    return analyze_distribution_shift(loaded.frame, audit, request)
