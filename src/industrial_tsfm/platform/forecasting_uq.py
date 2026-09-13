from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .forecasting import ForecastingRecord


@dataclass(frozen=True)
class ConformalForecastArtifact:
    """Offline-calibrated absolute-residual conformal interval artifact.

    The artifact stores one symmetric radius per target and horizon step. It is
    immutable at runtime: live observations are never used to update radii.
    """

    target_columns: tuple[str, ...]
    horizon: int
    alpha: float
    radii: tuple[tuple[float, ...], ...]
    calibration_rows: int
    calibration_scope: str = "offline_calibration_only"

    def validate(self) -> None:
        if not self.target_columns:
            raise ValueError("conformal artifact target_columns must not be empty")
        if len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("conformal artifact target_columns must be unique")
        if self.horizon < 1:
            raise ValueError("conformal artifact horizon must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("conformal artifact alpha must be in (0, 1)")
        if self.calibration_rows < 1:
            raise ValueError("conformal artifact calibration_rows must be positive")
        if self.calibration_scope != "offline_calibration_only":
            raise ValueError("conformal artifact must use offline_calibration_only scope")
        if len(self.radii) != len(self.target_columns):
            raise ValueError("conformal artifact radii target dimension mismatch")
        for row in self.radii:
            if len(row) != self.horizon:
                raise ValueError("conformal artifact radii horizon dimension mismatch")
            values = np.asarray(row, dtype=float)
            if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
                raise ValueError("conformal artifact radii must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["schema_version"] = "industrial_tsfm.forecast_conformal.v1"
        payload["nominal_coverage"] = 1.0 - self.alpha
        payload["online_updates_allowed"] = False
        payload["radii"] = [list(row) for row in self.radii]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConformalForecastArtifact:
        artifact = cls(
            target_columns=tuple(str(value) for value in payload["target_columns"]),
            horizon=int(payload["horizon"]),
            alpha=float(payload["alpha"]),
            radii=tuple(tuple(float(value) for value in row) for row in payload["radii"]),
            calibration_rows=int(payload["calibration_rows"]),
            calibration_scope=str(payload.get("calibration_scope", "offline_calibration_only")),
        )
        artifact.validate()
        return artifact


def _finite_sample_radius(residuals: np.ndarray, alpha: float) -> float:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("conformal calibration residuals are empty")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(values))
    ordered = np.sort(values)
    return float(ordered[rank - 1])


def calibrate_conformal_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_columns: tuple[str, ...],
    *,
    alpha: float = 0.1,
    min_calibration_rows: int = 20,
) -> ConformalForecastArtifact:
    """Calibrate split-conformal radii from an explicit offline residual set.

    Inputs must have shape ``[calibration_examples, horizon, targets]``. No
    runtime/live data is accepted by this function implicitly.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min_calibration_rows < 1:
        raise ValueError("min_calibration_rows must be positive")
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError("actual and predicted must have identical shapes")
    if y.ndim != 3:
        raise ValueError("actual and predicted must have shape [rows, horizon, targets]")
    if y.shape[0] < min_calibration_rows:
        raise ValueError(f"at least {min_calibration_rows} calibration rows are required")
    if y.shape[2] != len(target_columns):
        raise ValueError("target_columns does not match calibration target dimension")
    if not target_columns:
        raise ValueError("target_columns must not be empty")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("target_columns must be unique")
    residuals = np.abs(y - yhat)
    if not np.isfinite(residuals).all():
        raise ValueError("calibration residuals must be finite")
    radii: list[tuple[float, ...]] = []
    for target_index in range(y.shape[2]):
        radii.append(
            tuple(
                _finite_sample_radius(residuals[:, step, target_index], alpha)
                for step in range(y.shape[1])
            )
        )
    artifact = ConformalForecastArtifact(
        target_columns=tuple(target_columns),
        horizon=int(y.shape[1]),
        alpha=float(alpha),
        radii=tuple(radii),
        calibration_rows=int(y.shape[0]),
    )
    artifact.validate()
    return artifact


def apply_conformal_forecast(
    artifact: ConformalForecastArtifact,
    record: ForecastingRecord | dict[str, Any],
) -> dict[str, Any]:
    """Attach frozen symmetric intervals to a forecasting record."""

    artifact.validate()
    payload = record.to_dict() if isinstance(record, ForecastingRecord) else dict(record)
    forecasts = payload.get("forecasts")
    if not isinstance(forecasts, list):
        raise TypeError("forecast record must contain a forecasts list")
    radius_by_key = {
        (target, step + 1): artifact.radii[target_index][step]
        for target_index, target in enumerate(artifact.target_columns)
        for step in range(artifact.horizon)
    }
    enriched: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in forecasts:
        row = dict(raw)
        key = (str(row.get("target")), int(row.get("step", 0)))
        if key not in radius_by_key:
            raise ValueError(f"forecast point {key!r} is not covered by conformal artifact")
        value = float(row["value"])
        radius = float(radius_by_key[key])
        row.update(
            {
                "lower": value - radius,
                "upper": value + radius,
                "interval_radius": radius,
                "nominal_coverage": 1.0 - artifact.alpha,
            }
        )
        enriched.append(row)
        seen.add(key)
    expected = set(radius_by_key)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"forecast record is missing conformal points: {missing}")
    payload["forecasts"] = enriched
    payload["uq"] = {
        "schema_version": "industrial_tsfm.forecast_conformal_runtime.v1",
        "method": "split_conformal_absolute_residual",
        "alpha": artifact.alpha,
        "nominal_coverage": 1.0 - artifact.alpha,
        "calibration_rows": artifact.calibration_rows,
        "calibration_scope": artifact.calibration_scope,
        "online_updates": False,
    }
    return payload
