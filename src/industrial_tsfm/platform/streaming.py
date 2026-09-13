from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import pandas as pd

from .runtime import TimeSeriesWindow, WindowProvider


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Predictor(Protocol):
    """Minimal deterministic inference boundary for online applications."""

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PredictionRecord:
    """One auditable streaming inference result."""

    generated_at: str
    observed_through: str | None
    source_kind: str
    predictions: dict[str, Any]
    latency_ms: float
    model: dict[str, Any]
    quality: dict[str, Any]
    window: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicLastValuePredictor:
    """Lightweight CI predictor that repeats the latest observed target values.

    This is intentionally a runtime-contract test double, not a product model.
    Real TSFM/checkpoint adapters can implement the same ``Predictor`` protocol
    without changing the live runtime or prediction-record semantics.
    """

    def __init__(self, target_columns: tuple[str, ...], model_id: str = "last-value-ci") -> None:
        if not target_columns:
            raise ValueError("target_columns must not be empty")
        self.target_columns = tuple(target_columns)
        self.model_id = model_id

    def predict(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            raise ValueError("predictor requires a non-empty context frame")
        output: dict[str, Any] = {}
        for column in self.target_columns:
            if column not in frame.columns:
                raise ValueError(f"target column {column!r} is missing from context frame")
            valid = frame[column].dropna()
            if valid.empty:
                raise ValueError(f"target column {column!r} has no observed values in context frame")
            value = valid.iloc[-1]
            output[column] = value.item() if hasattr(value, "item") else value
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": "deterministic_test_double",
            "strategy": "last_observed_value",
            "target_columns": list(self.target_columns),
            "online_training": False,
        }


class StreamingInferenceEngine:
    """Source-agnostic online inference over replay or live runtime windows."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        context_observations: int | None = None,
        minimum_rows: int = 1,
    ) -> None:
        if context_observations is not None and context_observations < 1:
            raise ValueError("context_observations must be positive when configured")
        if minimum_rows < 1:
            raise ValueError("minimum_rows must be positive")
        self.predictor = predictor
        self.context_observations = context_observations
        self.minimum_rows = int(minimum_rows)

    @staticmethod
    def _observed_through(window: TimeSeriesWindow) -> str | None:
        column = window.timestamp_column
        if not column or column not in window.frame.columns or window.frame.empty:
            return None
        value = window.frame[column].iloc[-1]
        return None if pd.isna(value) else str(value)

    @staticmethod
    def _quality(window: TimeSeriesWindow) -> dict[str, Any]:
        missing_cells = int(window.frame.isna().sum().sum())
        bad = int(window.metadata.get("bad_quality_observations", 0))
        dropped = int(window.metadata.get("buffer_dropped_rows", 0))
        out_of_order = int(window.metadata.get("out_of_order_transitions", 0))
        if bad or dropped or out_of_order:
            state = "degraded"
        elif missing_cells:
            state = "partial"
        else:
            state = "good"
        return {
            "state": state,
            "bad_quality_observations": bad,
            "dropped_observations": dropped,
            "out_of_order_transitions": out_of_order,
            "missing_cells": missing_cells,
            "quality_mask_available": not window.quality_mask.empty,
        }

    def infer(self, provider: WindowProvider) -> PredictionRecord:
        window = provider.materialize_window(self.context_observations)
        if len(window.frame) < self.minimum_rows:
            raise ValueError(
                f"insufficient context rows: have {len(window.frame)}, need {self.minimum_rows}"
            )
        started = time.perf_counter()
        predictions = self.predictor.predict(window.frame.copy())
        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionRecord(
            generated_at=_utc_now(),
            observed_through=self._observed_through(window),
            source_kind=window.source_kind,
            predictions=predictions,
            latency_ms=float(latency_ms),
            model=dict(self.predictor.metadata()),
            quality=self._quality(window),
            window=window.snapshot(),
        )
