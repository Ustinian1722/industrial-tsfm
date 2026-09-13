from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .runtime import TimeSeriesWindow, WindowProvider


def _chronology_valid(window: TimeSeriesWindow) -> bool:
    if window.timestamp_column is None:
        return True
    return (
        int(window.metadata.get("parse_failures", 0)) == 0
        and int(window.metadata.get("out_of_order_transitions", 0)) == 0
    )


@dataclass(frozen=True)
class PCASPEArtifact:
    """Frozen PCA-SPE calibration state for deployment-time anomaly scoring."""

    feature_columns: tuple[str, ...]
    medians: tuple[float, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    pca_mean: tuple[float, ...]
    pca_components: tuple[tuple[float, ...], ...]
    threshold: float
    threshold_quantile: float
    fit_rows: int
    explained_variance_ratio_sum: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "industrial_tsfm.pca_spe_artifact.v1"
        payload["fit_scope"] = "offline_training_or_calibration_only"
        payload["online_updates_allowed"] = False
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PCASPEArtifact:
        return cls(
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            medians=tuple(float(value) for value in payload["medians"]),
            scaler_mean=tuple(float(value) for value in payload["scaler_mean"]),
            scaler_scale=tuple(float(value) for value in payload["scaler_scale"]),
            pca_mean=tuple(float(value) for value in payload["pca_mean"]),
            pca_components=tuple(
                tuple(float(value) for value in row) for row in payload["pca_components"]
            ),
            threshold=float(payload["threshold"]),
            threshold_quantile=float(payload["threshold_quantile"]),
            fit_rows=int(payload["fit_rows"]),
            explained_variance_ratio_sum=float(payload["explained_variance_ratio_sum"]),
        )


def _numeric_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"PCA-SPE feature columns are missing: {missing}")
    return frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def fit_pca_spe_artifact(
    training_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    threshold_quantile: float = 0.995,
    n_components: float = 0.95,
    min_train_rows: int = 16,
) -> PCASPEArtifact:
    """Fit all anomaly parameters from an explicit offline training/calibration frame."""

    if training_frame.empty:
        raise ValueError("cannot fit PCA-SPE artifact on an empty dataframe")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be in (0, 1)")
    if min_train_rows < 4:
        raise ValueError("min_train_rows must be at least 4")
    if len(training_frame) < min_train_rows:
        raise ValueError(f"at least {min_train_rows} training rows are required")
    if isinstance(n_components, float) and not 0.0 < n_components <= 1.0:
        raise ValueError("float n_components must be in (0, 1]")
    if isinstance(n_components, int) and n_components < 1:
        raise ValueError("integer n_components must be positive")

    numeric = _numeric_features(training_frame, feature_columns)
    medians = numeric.median(axis=0, skipna=True)
    valid_features = tuple(column for column in feature_columns if pd.notna(medians[column]))
    if not valid_features:
        raise ValueError("all PCA-SPE features are missing in the training frame")
    numeric = numeric[list(valid_features)]
    medians = medians[list(valid_features)]
    train = numeric.fillna(medians)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(train.to_numpy(dtype=float))
    resolved_components = n_components
    if isinstance(resolved_components, int):
        resolved_components = min(resolved_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=resolved_components, svd_solver="full", random_state=0)
    transformed = pca.fit_transform(scaled)
    reconstruction = pca.inverse_transform(transformed)
    point_scores = ((scaled - reconstruction) ** 2).sum(axis=1)
    threshold = float(np.quantile(point_scores, threshold_quantile))

    return PCASPEArtifact(
        feature_columns=valid_features,
        medians=tuple(float(medians[column]) for column in valid_features),
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        pca_mean=tuple(float(value) for value in pca.mean_),
        pca_components=tuple(tuple(float(value) for value in row) for row in pca.components_),
        threshold=threshold,
        threshold_quantile=float(threshold_quantile),
        fit_rows=len(training_frame),
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
    )


def score_pca_spe_artifact(
    artifact: PCASPEArtifact,
    frame: pd.DataFrame,
    *,
    top_k_sensors: int = 5,
) -> dict[str, Any]:
    """Score a frame using only frozen artifact state; no fitting or threshold updates occur."""

    if frame.empty:
        raise ValueError("cannot score an empty dataframe")
    if top_k_sensors < 1:
        raise ValueError("top_k_sensors must be positive")
    numeric = _numeric_features(frame, artifact.feature_columns)
    missing_by_feature = {
        column: int(numeric[column].isna().sum()) for column in artifact.feature_columns
    }
    median_imputed_values = int(sum(missing_by_feature.values()))
    median = pd.Series(artifact.medians, index=artifact.feature_columns, dtype=float)
    values = numeric.fillna(median).to_numpy(dtype=float)
    mean = np.asarray(artifact.scaler_mean, dtype=float)
    scale = np.asarray(artifact.scaler_scale, dtype=float)
    scaled = (values - mean) / scale
    pca_mean = np.asarray(artifact.pca_mean, dtype=float)
    components = np.asarray(artifact.pca_components, dtype=float)
    transformed = (scaled - pca_mean) @ components.T
    reconstruction = transformed @ components + pca_mean
    sensor_scores = (scaled - reconstruction) ** 2
    point_scores = sensor_scores.sum(axis=1)
    flags = point_scores >= artifact.threshold

    scope = sensor_scores[flags]
    if not len(scope):
        scope = sensor_scores
    means = scope.mean(axis=0)
    order = np.argsort(means)[::-1][: min(top_k_sensors, len(artifact.feature_columns))]
    top_sensors = [
        {
            "rank": rank + 1,
            "sensor": artifact.feature_columns[int(index)],
            "mean_spe_contribution": float(means[index]),
        }
        for rank, index in enumerate(order)
    ]
    return {
        "schema_version": "industrial_tsfm.online_pca_spe.v1",
        "model": "pca_spe_product",
        "online_training": False,
        "threshold_updated_online": False,
        "threshold": artifact.threshold,
        "threshold_quantile": artifact.threshold_quantile,
        "artifact_fit_rows": artifact.fit_rows,
        "feature_columns": list(artifact.feature_columns),
        "missing_value_policy": "frozen_training_median",
        "median_imputed_values": median_imputed_values,
        "median_imputed_by_feature": missing_by_feature,
        "point_scores": [float(value) for value in point_scores],
        "anomaly_flags": [bool(value) for value in flags],
        "latest_score": float(point_scores[-1]),
        "latest_anomaly": bool(flags[-1]),
        "anomaly_points": int(flags.sum()),
        "top_sensors": top_sensors,
    }


class OnlinePCASPEAnomalyDetector:
    """Window-provider adapter for fixed-artifact online anomaly inference.

    The default runtime path rejects nonchronological timestamp evidence rather
    than reordering it. Thresholds and preprocessing remain frozen regardless of
    runtime data.
    """

    def __init__(
        self,
        artifact: PCASPEArtifact,
        *,
        context_observations: int | None = None,
        top_k_sensors: int = 5,
        require_chronological: bool = True,
    ) -> None:
        if context_observations is not None and context_observations < 1:
            raise ValueError("context_observations must be positive when configured")
        if top_k_sensors < 1:
            raise ValueError("top_k_sensors must be positive")
        self.artifact = artifact
        self.context_observations = context_observations
        self.top_k_sensors = int(top_k_sensors)
        self.require_chronological = bool(require_chronological)

    @staticmethod
    def _observed_through(window: TimeSeriesWindow) -> str | None:
        column = window.timestamp_column
        if not column or column not in window.frame.columns or window.frame.empty:
            return None
        value = window.frame[column].iloc[-1]
        return None if pd.isna(value) else str(value)

    def infer(self, provider: WindowProvider) -> dict[str, Any]:
        window = provider.materialize_window(self.context_observations)
        if self.require_chronological and not _chronology_valid(window):
            raise ValueError(
                "online anomaly inference requires chronological timestamps; "
                "the runtime preserves arrival order and will not sort or repair the window"
            )
        result = score_pca_spe_artifact(
            self.artifact,
            window.frame,
            top_k_sensors=self.top_k_sensors,
        )
        result.update(
            {
                "source_kind": window.source_kind,
                "observed_through": self._observed_through(window),
                "chronology_valid": _chronology_valid(window),
                "window": window.snapshot(),
            }
        )
        return result
