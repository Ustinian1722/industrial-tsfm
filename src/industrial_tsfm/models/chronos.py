from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForecastModel


def _count_parameters(value: Any) -> int | None:
    try:
        parameters = value.parameters()
    except AttributeError:
        return None
    return int(sum(parameter.numel() for parameter in parameters))


class ChronosModel(ForecastModel):
    name = "chronos"
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
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise RuntimeError(
                'Chronos is not installed. Run `python -m pip install -e ".[tsfm]"`.'
            ) from exc
        torch.manual_seed(seed)
        model_id = self.config.get("model_id", "amazon/chronos-t5-tiny")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        return {
            "fit_status": "not_applicable_zero_shot",
            "checkpoint": model_id,
            "num_samples": int(self.config.get("num_samples", 20)),
            "inference_batch_size": int(self.config.get("inference_batch_size", 16)),
            "internal_context_scaling": True,
            "torch_version": torch.__version__,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("ChronosModel must be initialized before predict")
        import torch

        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3:
            raise ValueError("Expected context with shape [batch, context_length, features]")
        batch_size, _, n_features = context.shape
        flat = context.transpose(0, 2, 1).reshape(batch_size * n_features, -1)
        inference_batch_size = int(self.config.get("inference_batch_size", 16))
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        point_batches: list[np.ndarray] = []
        for start in range(0, len(flat), inference_batch_size):
            context_tensor = torch.from_numpy(flat[start : start + inference_batch_size])
            samples = self.pipeline.predict(
                context_tensor,
                prediction_length=self.horizon,
                num_samples=int(self.config.get("num_samples", 20)),
                temperature=float(self.config.get("temperature", 1.0)),
                top_k=int(self.config.get("top_k", 50)),
                top_p=float(self.config.get("top_p", 1.0)),
            )
            # [batch, samples, H] -> median point forecast -> [batch, H].
            point_batches.append(samples.median(dim=1).values.detach().cpu().numpy())
        point = np.concatenate(point_batches, axis=0).astype(np.float32)
        if point.shape != (batch_size * n_features, self.horizon):
            raise RuntimeError(f"Unexpected Chronos output shape: {point.shape}")
        return point.reshape(batch_size, n_features, self.horizon).transpose(0, 2, 1)

    def parameter_count(self) -> int | None:
        if self.pipeline is None:
            return None
        inner = getattr(self.pipeline, "inner_model", None)
        return _count_parameters(inner) if inner is not None else None

    def trainable_parameter_count(self) -> int | None:
        return 0 if self.pipeline is not None else None
