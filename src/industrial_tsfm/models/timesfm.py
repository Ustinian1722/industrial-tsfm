from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForecastModel


def _count_parameters(value: Any) -> int | None:
    try:
        parameters = value.parameters()
    except AttributeError:
        nested = getattr(value, "model", None)
        if nested is None or nested is value:
            return None
        return _count_parameters(nested)
    return int(sum(parameter.numel() for parameter in parameters))


class TimesFMModel(ForecastModel):
    name = "timesfm"
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
        self.model: Any | None = None

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        del train_context, train_target, validation_context, validation_target, seed
        try:
            import timesfm
            import torch
        except ImportError as exc:
            raise RuntimeError(
                'TimesFM is not installed. Run `python -m pip install -e ".[tsfm]"`.'
            ) from exc
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise RuntimeError("Installed timesfm package does not expose TimesFM 2.5 PyTorch API")
        model_id = self.config.get("model_id", "google/timesfm-2.5-200m-pytorch")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        if self.device != "cpu" and hasattr(self.model, "to"):
            self.model.to(self.device)
        forecast_config = timesfm.ForecastConfig(
            max_context=int(self.config.get("max_context", max(self.context_length, 1024))),
            max_horizon=max(self.horizon, int(self.config.get("max_horizon", self.horizon))),
            normalize_inputs=bool(self.config.get("normalize_inputs", False)),
            per_core_batch_size=int(self.config.get("per_core_batch_size", 32)),
            force_flip_invariance=bool(self.config.get("force_flip_invariance", True)),
            infer_is_positive=bool(self.config.get("infer_is_positive", False)),
        )
        self.model.compile(forecast_config)
        return {
            "fit_status": "not_applicable_zero_shot",
            "checkpoint": model_id,
            "normalization_in_model": forecast_config.normalize_inputs,
            "torch_version": torch.__version__,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TimesFMModel must be initialized before predict")
        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3:
            raise ValueError("Expected context with shape [batch, context_length, features]")
        batch_size, _, n_features = context.shape
        flat_inputs = [
            context[i, :, channel].astype(np.float32)
            for i in range(batch_size)
            for channel in range(n_features)
        ]
        inference_batch_size = int(self.config.get("inference_batch_size", 256))
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        point_batches: list[np.ndarray] = []
        for start in range(0, len(flat_inputs), inference_batch_size):
            point_forecast, _ = self.model.forecast(
                horizon=self.horizon,
                inputs=flat_inputs[start : start + inference_batch_size],
            )
            point_batches.append(np.asarray(point_forecast, dtype=np.float32))
        forecast = np.concatenate(point_batches, axis=0)
        if forecast.shape != (batch_size * n_features, self.horizon):
            raise RuntimeError(f"Unexpected TimesFM output shape: {forecast.shape}")
        return forecast.reshape(batch_size, n_features, self.horizon).transpose(0, 2, 1)

    def parameter_count(self) -> int | None:
        return None if self.model is None else _count_parameters(self.model)

    def trainable_parameter_count(self) -> int | None:
        return 0 if self.model is not None else None
