from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .contracts import ProjectSpec, TaskType


@dataclass(frozen=True)
class AnomalyDetectionRequest:
    task_name: str
    train_fraction: float = 0.50
    threshold_quantile: float = 0.995
    n_components: float | int = 0.95
    min_train_rows: int = 16
    top_k_sensors: int = 5

    def validate(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must be non-empty")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if self.min_train_rows < 4:
            raise ValueError("min_train_rows must be at least 4")
        if self.top_k_sensors < 1:
            raise ValueError("top_k_sensors must be positive")
        if isinstance(self.n_components, float) and not 0.0 < self.n_components <= 1.0:
            raise ValueError("float n_components must be in (0, 1]")
        if isinstance(self.n_components, int) and self.n_components < 1:
            raise ValueError("integer n_components must be positive")


def _task(project: ProjectSpec, name: str):
    matches = [task for task in project.tasks if task.name == name]
    if len(matches) != 1:
        raise ValueError(f"task {name!r} must exist exactly once")
    task = matches[0]
    if task.task_type != TaskType.ANOMALY:
        raise ValueError(f"task {name!r} is not an anomaly_detection task")
    return task


def _event_count(flags: np.ndarray) -> int:
    binary = np.asarray(flags, dtype=bool).reshape(-1)
    if not len(binary):
        return 0
    starts = binary & np.r_[True, ~binary[:-1]]
    return int(starts.sum())


def _pattern_label(scores: np.ndarray, threshold: float, train_end: int) -> str:
    values = np.asarray(scores, dtype=float)
    test = values[train_end:]
    if len(test) < 4:
        return "insufficient_context"
    baseline = max(float(np.median(values[:train_end])), 1.0e-12)
    if float(test.max()) >= max(threshold * 4.0, baseline * 12.0):
        return "spike_or_excursion"
    x = np.arange(len(test), dtype=float)
    slope = float(np.polyfit(x, test, 1)[0])
    scale = max(float(np.median(np.abs(test - np.median(test)))), baseline, 1.0e-12)
    if abs(slope) * len(test) > 3.0 * scale:
        return "drift_or_degradation"
    diffs = np.diff(test)
    if len(diffs) >= 3:
        sign_changes = np.mean(np.sign(diffs[1:]) != np.sign(diffs[:-1]))
        if float(sign_changes) > 0.75 and float(np.std(diffs)) > scale:
            return "oscillation_or_noise"
    if float(np.median(test)) > max(threshold, baseline * 3.0):
        return "level_shift"
    return "mixed_or_low_severity"


def run_pca_spe_anomaly_detection(
    project: ProjectSpec,
    frame: pd.DataFrame,
    data_audit: dict[str, Any],
    request: AnomalyDetectionRequest,
) -> dict[str, Any]:
    """Fit a deployment-safe PCA-SPE baseline using only a normal training prefix.

    The training prefix supplies imputation values, scaler statistics, PCA
    parameters, and the score threshold. No test labels are read. Sensor-level
    squared reconstruction residuals provide lightweight attribution, not a
    causal root-cause claim.
    """

    project.validate()
    request.validate()
    task = _task(project, request.task_name)
    if frame.empty:
        raise ValueError("cannot run anomaly detection on an empty dataframe")

    configured_features = tuple(task.constraints.get("feature_columns", ()))
    usable = list(data_audit.get("usable_numeric_features", []))
    features = list(configured_features) if configured_features else usable
    excluded = set(task.target_columns)
    label_column = task.constraints.get("label_column")
    if label_column:
        excluded.add(str(label_column))
    features = [column for column in features if column not in excluded]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"anomaly feature columns are missing: {missing}")
    if not features:
        raise ValueError("no usable numeric features are available for anomaly detection")

    numeric = frame[features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    requested_train_end = int(np.floor(len(numeric) * request.train_fraction))
    train_end = max(request.min_train_rows, requested_train_end)
    train_end = min(train_end, len(numeric) - 1)
    if train_end < request.min_train_rows or train_end <= 1:
        raise ValueError(
            f"only {len(numeric)} rows are available; at least {request.min_train_rows + 1} are required"
        )

    train = numeric.iloc[:train_end].copy()
    medians = train.median(axis=0, skipna=True)
    valid_features = [column for column in features if pd.notna(medians[column])]
    dropped = [column for column in features if column not in valid_features]
    if not valid_features:
        raise ValueError("all anomaly features are missing in the training prefix")

    train = numeric[valid_features].iloc[:train_end].fillna(medians[valid_features])
    all_values = numeric[valid_features].fillna(medians[valid_features])
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.to_numpy(dtype=float))
    all_scaled = scaler.transform(all_values.to_numpy(dtype=float))

    n_components: float | int = request.n_components
    if isinstance(n_components, int):
        n_components = min(n_components, train_scaled.shape[1], train_scaled.shape[0])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    pca.fit(train_scaled)
    reconstruction = pca.inverse_transform(pca.transform(all_scaled))
    sensor_scores = (all_scaled - reconstruction) ** 2
    point_scores = sensor_scores.sum(axis=1)
    train_scores = point_scores[:train_end]
    threshold = float(np.quantile(train_scores, request.threshold_quantile))
    anomaly_flags = point_scores >= threshold
    test_flags = anomaly_flags[train_end:]

    attribution_scope = sensor_scores[train_end:][test_flags]
    if not len(attribution_scope):
        attribution_scope = sensor_scores[train_end:]
    mean_sensor = attribution_scope.mean(axis=0) if len(attribution_scope) else sensor_scores.mean(axis=0)
    order = np.argsort(mean_sensor)[::-1][: min(request.top_k_sensors, len(valid_features))]
    top_sensors = [
        {
            "rank": rank + 1,
            "sensor": valid_features[int(index)],
            "mean_spe_contribution": float(mean_sensor[index]),
        }
        for rank, index in enumerate(order)
    ]

    score_quantiles = {
        str(level): float(np.quantile(point_scores, level))
        for level in (0.5, 0.9, 0.95, 0.99)
    }
    return {
        "schema_version": "industrial_tsfm.anomaly_result.v1",
        "task": task.name,
        "task_type": task.task_type.value,
        "model": "pca_spe_product",
        "status": "complete",
        "selection_evidence": "task_contract",
        "fit_scope": "normal_training_prefix_only",
        "target_labels_used": False,
        "request": asdict(request),
        "feature_columns": valid_features,
        "dropped_training_all_missing_features": dropped,
        "train_rows": int(train_end),
        "test_rows": int(len(frame) - train_end),
        "threshold": {
            "strategy": "train_score_quantile",
            "quantile": float(request.threshold_quantile),
            "value": threshold,
            "fit_rows": int(train_end),
        },
        "pca": {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "summary": {
            "anomaly_points_total": int(anomaly_flags.sum()),
            "anomaly_points_test": int(test_flags.sum()),
            "test_anomaly_fraction": float(test_flags.mean()) if len(test_flags) else 0.0,
            "test_events": _event_count(test_flags),
            "pattern": _pattern_label(point_scores, threshold, train_end),
            "score_quantiles": score_quantiles,
        },
        "top_sensors": top_sensors,
        "point_scores": [float(value) for value in point_scores],
        "anomaly_flags": [bool(value) for value in anomaly_flags],
    }
