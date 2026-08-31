from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrainOnlyStandardScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    fit_entities: tuple[str, ...]
    fit_rows: int
    epsilon: float = 1.0e-8

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        entities: Iterable[str],
        feature_names: Iterable[str],
        epsilon: float = 1.0e-8,
    ) -> TrainOnlyStandardScaler:
        feature_names = tuple(feature_names)
        fit_entities = tuple(sorted(set(entities)))
        subset = frame[frame["entity_id"].isin(fit_entities)]
        if subset.empty:
            raise ValueError("Cannot fit scaler: no model-train rows were selected")
        values = subset.loc[:, feature_names].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Cannot fit scaler on non-finite values")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > epsilon, scale, 1.0)
        return cls(feature_names, mean, scale, fit_entities, len(subset), epsilon)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean.astype(np.float32)) / self.scale.astype(np.float32)).astype(
            np.float32
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values * self.scale.astype(np.float32) + self.mean.astype(np.float32)).astype(
            np.float32
        )

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame.loc[:, self.feature_names].to_numpy(dtype=np.float32))

    def mase_scale(self, frame: pd.DataFrame) -> np.ndarray:
        """Return train-only one-step naive denominators, one value per feature."""
        subset = frame[frame["entity_id"].isin(self.fit_entities)]
        differences: list[np.ndarray] = []
        for _, entity_frame in subset.groupby("entity_id", sort=False):
            values = entity_frame.loc[:, self.feature_names].to_numpy(dtype=np.float64)
            if len(values) > 1:
                differences.append(np.abs(np.diff(values, axis=0)))
        if not differences:
            return np.ones(len(self.feature_names), dtype=np.float64)
        denominator = np.concatenate(differences, axis=0).mean(axis=0)
        return np.where(denominator > self.epsilon, denominator, 1.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fit_entities": list(self.fit_entities),
            "fit_rows": self.fit_rows,
            "epsilon": self.epsilon,
        }
