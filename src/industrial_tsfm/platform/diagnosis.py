from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .online_anomaly import OnlinePCASPEAnomalyDetector, PCASPEArtifact
from .runtime import WindowProvider


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_artifact(artifact: PCASPEArtifact) -> None:
    feature_count = len(artifact.feature_columns)
    if feature_count < 1:
        raise ValueError("diagnosis artifact must contain feature columns")
    if len(set(artifact.feature_columns)) != feature_count:
        raise ValueError("diagnosis artifact feature columns must be unique")
    if not np.isfinite(float(artifact.threshold)) or float(artifact.threshold) < 0.0:
        raise ValueError("diagnosis artifact threshold must be finite and non-negative")
    if not 0.0 < float(artifact.threshold_quantile) < 1.0:
        raise ValueError("diagnosis artifact threshold_quantile must be in (0, 1)")
    if int(artifact.fit_rows) < 1:
        raise ValueError("diagnosis artifact fit_rows must be positive")
    vectors = (
        artifact.medians,
        artifact.scaler_mean,
        artifact.scaler_scale,
        artifact.pca_mean,
    )
    for values in vectors:
        if len(values) != feature_count:
            raise ValueError("diagnosis artifact vector dimensions do not match feature columns")
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("diagnosis artifact vectors must be finite")
    scale = np.asarray(artifact.scaler_scale, dtype=float)
    if bool((scale <= 0.0).any()):
        raise ValueError("diagnosis artifact scaler_scale must be positive")
    components = np.asarray(artifact.pca_components, dtype=float)
    if components.ndim != 2 or components.shape[1] != feature_count:
        raise ValueError("diagnosis artifact PCA component dimensions are invalid")
    if components.shape[0] < 1 or not np.isfinite(components).all():
        raise ValueError("diagnosis artifact PCA components must be finite and non-empty")


@dataclass(frozen=True)
class DiagnosisRequest:
    """Operational triage policy over a frozen anomaly detector.

    Severity bands are deterministic operational thresholds over the anomaly
    score ratio and persistence. They are not failure probabilities and do not
    constitute causal root-cause identification.
    """

    context_observations: int = 64
    top_k_sensors: int = 5
    high_score_ratio: float = 2.0
    critical_score_ratio: float = 4.0
    persistent_points: int = 3

    def validate(self) -> None:
        if self.context_observations < 2:
            raise ValueError("context_observations must be at least 2")
        if self.top_k_sensors < 1:
            raise ValueError("top_k_sensors must be positive")
        if not 1.0 <= self.high_score_ratio < self.critical_score_ratio:
            raise ValueError("diagnosis score-ratio bands are invalid")
        if self.persistent_points < 1:
            raise ValueError("persistent_points must be positive")


class LiveDiagnosisApplication:
    """Inference-only anomaly triage over replay or live WindowProvider data."""

    def __init__(self, artifact: PCASPEArtifact, request: DiagnosisRequest) -> None:
        _validate_artifact(artifact)
        request.validate()
        self.artifact = artifact
        self.request = request
        self.detector = OnlinePCASPEAnomalyDetector(
            artifact,
            context_observations=request.context_observations,
            top_k_sensors=request.top_k_sensors,
            require_chronological=True,
        )

    @staticmethod
    def _trailing_anomaly_points(flags: list[bool]) -> int:
        count = 0
        for flag in reversed(flags):
            if not flag:
                break
            count += 1
        return count

    def _severity(self, *, latest_anomaly: bool, score_ratio: float, trailing_points: int) -> str:
        if not latest_anomaly:
            return "normal"
        if (
            score_ratio >= self.request.critical_score_ratio
            or trailing_points >= self.request.persistent_points * 2
        ):
            return "critical"
        if (
            score_ratio >= self.request.high_score_ratio
            or trailing_points >= self.request.persistent_points
        ):
            return "high"
        return "warning"

    def infer(self, provider: WindowProvider) -> dict[str, Any]:
        result = self.detector.infer(provider)
        flags = [bool(value) for value in result["anomaly_flags"]]
        trailing = self._trailing_anomaly_points(flags)
        threshold = float(result["threshold"])
        latest_score = float(result["latest_score"])
        denominator = max(abs(threshold), 1.0e-12)
        score_ratio = latest_score / denominator
        latest_anomaly = bool(result["latest_anomaly"])
        severity = self._severity(
            latest_anomaly=latest_anomaly,
            score_ratio=score_ratio,
            trailing_points=trailing,
        )
        return {
            "schema_version": "industrial_tsfm.live_diagnosis.v1",
            "generated_at": _utc_now(),
            "source_kind": result["source_kind"],
            "observed_through": result["observed_through"],
            "status": "alert" if latest_anomaly else "normal",
            "severity": severity,
            "score": {
                "latest": latest_score,
                "threshold": threshold,
                "ratio_to_threshold": float(score_ratio),
            },
            "event": {
                "latest_anomaly": latest_anomaly,
                "trailing_anomaly_points": trailing,
                "anomaly_points_in_window": int(result["anomaly_points"]),
                "window_points": len(flags),
            },
            "contributors": result["top_sensors"],
            "detector": {
                "model": result["model"],
                "feature_columns": result["feature_columns"],
                "artifact_fit_rows": result["artifact_fit_rows"],
                "threshold_quantile": result["threshold_quantile"],
                "chronology_valid": result["chronology_valid"],
            },
            "request": asdict(self.request),
            "claim_scope": "statistical_reconstruction_contribution_not_causal_root_cause",
            "online_training": False,
            "threshold_updated_online": False,
            "online_artifact_updates": False,
            "control_actions_enabled": False,
        }
