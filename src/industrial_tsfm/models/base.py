from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ForecastModel(ABC):
    name: str
    training_mode: str

    @abstractmethod
    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameter_count(self) -> int | None:
        return None

    def trainable_parameter_count(self) -> int | None:
        return None
