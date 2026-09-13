from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsfm.models.base import ForecastModel

from .runtime import TimeSeriesWindow, WindowProvider


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_python(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


@dataclass(frozen=True)
class ForecastingRequest:
    """Deterministic runtime contract for one forecasting application.

    ``per_tag_univariate`` is designed for asynchronous OPC-UA notifications: each
    target is forecast from its own arrival-order sample stream, so no cross-tag
    interpolation or timestamp alignment is invented. ``strict_multivariate`` is
    reserved for sources that already provide complete, aligned rows.
    """

    target_columns: tuple[str, ...]
    context_length: int
    horizon: int
    mode: str = "per_tag_univariate"
    require_good_quality: bool = True
    require_chronological: bool = True

    def validate(self) -> None:
        if not self.target_columns:
            raise ValueError("target_columns must not be empty")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.mode not in {"per_tag_univariate", "strict_multivariate"}:
            raise ValueError("mode must be per_tag_univariate or strict_multivariate")


@dataclass(frozen=True)
class ForecastPoint:
    target: str
    step: int
    value: float
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastingRecord:
    generated_at: str
    source_kind: str
    observed_through: str | None
    model: dict[str, Any]
    request: dict[str, Any]
    forecasts: tuple[ForecastPoint, ...]
    context: dict[str, Any]
    quality: dict[str, Any]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "industrial_tsfm.live_forecasting.v1"
        payload["online_training"] = False
        payload["forecasts"] = [point.to_dict() for point in self.forecasts]
        return payload


class ForecastModelRuntimeAdapter:
    """Inference-only adapter over the repository's existing ``ForecastModel`` API.

    The adapter never calls ``fit``. Loading/fitting/adaptation must happen before
    the model enters the live runtime, preserving the research/evaluation boundary
    and preventing target-stream labels from changing the model online.
    """

    def __init__(self, model: ForecastModel, *, model_metadata: dict[str, Any] | None = None) -> None:
        self.model = model
        self.model_metadata = dict(model_metadata or {})

    def metadata(self) -> dict[str, Any]:
        return {
            "name": getattr(self.model, "name", type(self.model).__name__),
            "training_mode": getattr(self.model, "training_mode", "unknown"),
            "parameter_count": self.model.parameter_count(),
            "trainable_parameter_count": self.model.trainable_parameter_count(),
            "online_training": False,
            **self.model_metadata,
        }

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, context_length, features]")
        configured_horizon = getattr(self.model, "horizon", None)
        if configured_horizon is not None and int(configured_horizon) != int(horizon):
            raise ValueError(
                f"runtime horizon {horizon} does not match model horizon {configured_horizon}"
            )
        output = np.asarray(self.model.predict(context), dtype=np.float32)
        expected = (context.shape[0], horizon, context.shape[2])
        if output.shape != expected:
            raise RuntimeError(f"forecast model returned {output.shape}, expected {expected}")
        return output


def _parse_timestamps(values: list[Any]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if bool(parsed.isna().any()):
        raise ValueError("forecast context contains unparseable timestamps")
    return pd.DatetimeIndex(parsed)


def _chronology_evidence(values: list[Any]) -> dict[str, Any]:
    parsed = _parse_timestamps(values)
    if len(parsed) < 2:
        return {
            "chronology_valid": True,
            "duplicate_timestamps": 0,
            "out_of_order_transitions": 0,
        }
    series = pd.Series(parsed)
    diffs = series.diff().dropna()
    return {
        "chronology_valid": bool((diffs >= pd.Timedelta(0)).all()),
        "duplicate_timestamps": int((diffs == pd.Timedelta(0)).sum()),
        "out_of_order_transitions": int((diffs < pd.Timedelta(0)).sum()),
    }


def _future_timestamps(values: list[Any], horizon: int) -> tuple[list[str | None], dict[str, Any]]:
    parsed = _parse_timestamps(values)
    if len(parsed) < 2:
        return [None] * horizon, {"cadence": "unknown", "period_seconds": None}
    diffs = pd.Series(parsed).diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if len(positive) != len(diffs) or positive.nunique() != 1:
        return [None] * horizon, {"cadence": "irregular", "period_seconds": None}
    step = positive.iloc[0]
    last = parsed[-1]
    stamps = [(last + step * index).isoformat() for index in range(1, horizon + 1)]
    return stamps, {
        "cadence": "regular_observed",
        "period_seconds": float(step.total_seconds()),
    }


class LiveForecastingApplication:
    """Source-neutral, inference-only forecasting application for replay or live windows."""

    def __init__(self, model: ForecastModelRuntimeAdapter, request: ForecastingRequest) -> None:
        request.validate()
        self.model = model
        self.request = request

    def _predict_per_tag(self, window: TimeSeriesWindow) -> tuple[list[ForecastPoint], dict[str, Any]]:
        arrival = window.arrival_frame
        required = {"tag", "value", "timestamp"}
        if not required.issubset(arrival.columns):
            raise ValueError(
                "per_tag_univariate mode requires arrival-frame tag/value/timestamp columns"
            )
        all_points: list[ForecastPoint] = []
        target_context: dict[str, Any] = {}
        for target in self.request.target_columns:
            rows = arrival[arrival["tag"] == target].tail(self.request.context_length)
            if len(rows) < self.request.context_length:
                raise ValueError(
                    f"target {target!r} has {len(rows)} samples, "
                    f"needs {self.request.context_length}"
                )
            values = pd.to_numeric(rows["value"], errors="coerce")
            if bool(values.isna().any()) or not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
                raise ValueError(f"target {target!r} context contains non-numeric or non-finite values")
            if self.request.require_good_quality and "status_good" in rows.columns:
                if not bool(rows["status_good"].fillna(False).astype(bool).all()):
                    raise ValueError(f"target {target!r} context contains bad-quality samples")
            timestamps = rows["timestamp"].tolist()
            chronology = _chronology_evidence(timestamps)
            if self.request.require_chronological and not chronology["chronology_valid"]:
                raise ValueError(
                    f"target {target!r} context is nonchronological; runtime will not sort it"
                )
            context = values.to_numpy(dtype=np.float32).reshape(1, -1, 1)
            prediction = self.model.predict(context, self.request.horizon)[0, :, 0]
            future, cadence = _future_timestamps(timestamps, self.request.horizon)
            for step, (value, timestamp) in enumerate(zip(prediction, future, strict=True), start=1):
                all_points.append(
                    ForecastPoint(
                        target=target,
                        step=step,
                        value=float(value),
                        timestamp=timestamp,
                    )
                )
            target_context[target] = {
                "rows": len(rows),
                "first_timestamp": str(timestamps[0]),
                "last_timestamp": str(timestamps[-1]),
                **chronology,
                **cadence,
            }
        return all_points, target_context

    def _predict_strict_multivariate(
        self, window: TimeSeriesWindow
    ) -> tuple[list[ForecastPoint], dict[str, Any]]:
        frame = window.frame.tail(self.request.context_length)
        missing = [column for column in self.request.target_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"forecast target columns are missing: {missing}")
        if len(frame) < self.request.context_length:
            raise ValueError(
                f"window has {len(frame)} rows, needs {self.request.context_length}"
            )
        values = frame[list(self.request.target_columns)].apply(pd.to_numeric, errors="coerce")
        if bool(values.isna().any().any()) or not bool(
            np.isfinite(values.to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                "strict_multivariate mode requires complete finite rows; no filling is performed"
            )
        timestamp_column = window.timestamp_column
        timestamps: list[Any] = []
        chronology: dict[str, Any] = {"chronology_valid": True}
        future: list[str | None] = [None] * self.request.horizon
        cadence: dict[str, Any] = {"cadence": "unknown", "period_seconds": None}
        if timestamp_column and timestamp_column in frame.columns:
            timestamps = frame[timestamp_column].tolist()
            chronology = _chronology_evidence(timestamps)
            if self.request.require_chronological and not chronology["chronology_valid"]:
                raise ValueError("multivariate context is nonchronological; runtime will not sort it")
            future, cadence = _future_timestamps(timestamps, self.request.horizon)
        context = values.to_numpy(dtype=np.float32).reshape(1, len(values), len(values.columns))
        prediction = self.model.predict(context, self.request.horizon)[0]
        points: list[ForecastPoint] = []
        for feature_index, target in enumerate(self.request.target_columns):
            for step in range(self.request.horizon):
                points.append(
                    ForecastPoint(
                        target=target,
                        step=step + 1,
                        value=float(prediction[step, feature_index]),
                        timestamp=future[step],
                    )
                )
        context_meta = {
            "rows": len(frame),
            "columns": list(self.request.target_columns),
            **chronology,
            **cadence,
        }
        return points, context_meta

    def infer(self, provider: WindowProvider) -> ForecastingRecord:
        window = provider.materialize_window(None)
        started = time.perf_counter()
        if self.request.mode == "per_tag_univariate":
            forecasts, context = self._predict_per_tag(window)
        else:
            forecasts, context = self._predict_strict_multivariate(window)
        latency_ms = (time.perf_counter() - started) * 1000.0
        observed = [point.timestamp for point in forecasts if point.timestamp is not None]
        observed_through = None
        if window.timestamp_column and window.timestamp_column in window.frame.columns and not window.frame.empty:
            observed_through = str(window.frame[window.timestamp_column].iloc[-1])
        quality = {
            "bad_quality_observations": int(window.metadata.get("bad_quality_observations", 0)),
            "dropped_observations": int(window.metadata.get("buffer_dropped_rows", 0)),
            "out_of_order_transitions": int(window.metadata.get("out_of_order_transitions", 0)),
            "forecast_timestamps_available": bool(observed),
        }
        return ForecastingRecord(
            generated_at=_utc_now(),
            source_kind=window.source_kind,
            observed_through=observed_through,
            model=self.model.metadata(),
            request=asdict(self.request),
            forecasts=tuple(forecasts),
            context=context,
            quality=quality,
            latency_ms=float(latency_ms),
        )
