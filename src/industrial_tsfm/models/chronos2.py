from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import ForecastModel


def _count_parameters(value: Any) -> int | None:
    try:
        return int(sum(parameter.numel() for parameter in value.parameters()))
    except (AttributeError, TypeError):
        return None


class Chronos2Model(ForecastModel):
    """Adapter for the native multivariate Chronos-2 forecasting pipeline."""

    name = "chronos2"
    training_mode = "zero_shot"

    def __init__(
        self,
        config: dict[str, Any],
        context_length: int,
        horizon: int,
        device: str,
    ) -> None:
        self.config = config
        self.context_length = context_length
        self.horizon = horizon
        self.device = device
        self.pipeline: Any | None = None
        self.quantile_levels: list[float] | None = None

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        del train_context, train_target, validation_context, validation_target
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise RuntimeError(
                'Chronos-2 is not installed. Run `python -m pip install -e ".[tsfm]"`.'
            ) from exc

        torch.manual_seed(seed)
        model_id = self.config.get("model_id", "amazon/chronos-2")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self.quantile_levels = [float(value) for value in self.pipeline.quantiles]
        return {
            "fit_status": "not_applicable_zero_shot",
            "checkpoint": model_id,
            "input_contract": "multivariate [batch, features, history]",
            "prediction_contract": "median quantile",
            "quantile_levels": self.quantile_levels,
            "inference_batch_size": int(self.config.get("inference_batch_size", 64)),
            "cross_learning": bool(self.config.get("cross_learning", False)),
            "max_context": int(getattr(self.pipeline, "model_context_length", 8192)),
            "torch_version": torch.__version__,
        }

    @staticmethod
    def _point_forecast(
        predictions: Sequence[Any],
        batch_size: int,
        n_features: int,
        horizon: int,
        quantile_levels: Sequence[float],
    ) -> np.ndarray:
        """Convert Chronos-2 [features, quantiles, horizon] outputs to runner shape."""
        if len(predictions) != batch_size:
            raise RuntimeError(
                f"Unexpected Chronos-2 batch length: {len(predictions)} != {batch_size}"
            )
        try:
            median_index = list(quantile_levels).index(0.5)
        except ValueError as exc:
            raise RuntimeError("Chronos-2 output does not contain the 0.5 quantile") from exc

        point_batches: list[np.ndarray] = []
        for prediction in predictions:
            values = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
            if values.ndim != 3:
                raise RuntimeError(f"Unexpected Chronos-2 output rank: {values.shape}")
            if values.shape[0] != n_features or values.shape[1] <= median_index:
                raise RuntimeError(
                    "Unexpected Chronos-2 feature/quantile shape: "
                    f"{values.shape}; expected [{n_features}, quantiles, {horizon}]"
                )
            if values.shape[2] != horizon:
                raise RuntimeError(
                    f"Unexpected Chronos-2 horizon: {values.shape[2]} != {horizon}"
                )
            point_batches.append(values[:, median_index, :].T)
        return np.stack(point_batches, axis=0).astype(np.float32)

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.pipeline is None or self.quantile_levels is None:
            raise RuntimeError("Chronos2Model must be initialized before predict")
        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3:
            raise ValueError("Expected context with shape [batch, context_length, features]")
        batch_size, observed_length, n_features = context.shape
        if observed_length != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}, got {observed_length}"
            )
        inference_batch_size = int(self.config.get("inference_batch_size", 64))
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        max_context = int(
            self.config.get("max_context", getattr(self.pipeline, "model_context_length", 8192))
        )
        effective_context_length = min(self.context_length, max_context)
        if effective_context_length <= 0:
            raise ValueError("max_context must be positive")

        # Chronos-2 treats each item in the batch as one multivariate task. Keeping
        # cross_learning disabled by default prevents unrelated entities/windows from
        # sharing information during evaluation.
        inputs = context.transpose(0, 2, 1)
        predictions = self.pipeline.predict(
            inputs,
            prediction_length=self.horizon,
            batch_size=inference_batch_size,
            context_length=effective_context_length,
            cross_learning=bool(self.config.get("cross_learning", False)),
        )
        return self._point_forecast(
            predictions,
            batch_size=batch_size,
            n_features=n_features,
            horizon=self.horizon,
            quantile_levels=self.quantile_levels,
        )

    def parameter_count(self) -> int | None:
        if self.pipeline is None:
            return None
        for attribute in ("model", "inner_model"):
            inner = getattr(self.pipeline, attribute, None)
            if inner is not None:
                count = _count_parameters(inner)
                if count is not None:
                    return count
        return None

    def trainable_parameter_count(self) -> int | None:
        return 0 if self.pipeline is not None else None
