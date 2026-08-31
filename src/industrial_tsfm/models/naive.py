from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForecastModel


class NaiveModel(ForecastModel):
    name = "naive"
    training_mode = "none"

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        del train_context, train_target, validation_context, validation_target, seed
        return {"fit_status": "not_applicable"}

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3:
            raise ValueError("Expected context with shape [batch, context_length, features]")
        horizon = getattr(self, "horizon", None)
        if horizon is None:
            raise RuntimeError("NaiveModel.horizon must be set before predict")
        return np.repeat(context[:, -1:, :], horizon, axis=1)

    def configure(self, horizon: int) -> NaiveModel:
        self.horizon = int(horizon)
        return self

    def parameter_count(self) -> int:
        return 0

    def trainable_parameter_count(self) -> int:
        return 0
