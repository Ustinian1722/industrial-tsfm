from __future__ import annotations

from typing import Any

import numpy as np

from .base import ForecastModel


def _count_parameters(value: Any) -> int | None:
    try:
        return int(sum(parameter.numel() for parameter in value.parameters()))
    except (AttributeError, TypeError):
        return None


class Moirai2Model(ForecastModel):
    """Adapter for Salesforce's Moirai 2.0-R-small native multivariate forecaster.

    Uni2TS exposes Moirai through a GluonTS predictor.  The project runner still
    owns windowing and source-only preprocessing; this adapter only converts a
    batch of fixed windows to the official wide multivariate input contract and
    returns the official p50 quantile as the point forecast.
    """

    name = "moirai2"
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
        self.forecast_model: Any | None = None
        self.predictor: Any | None = None
        self.n_features: int | None = None
        self.quantile_levels: list[float] | None = None

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        del train_target, validation_context, validation_target, seed
        train_context = np.asarray(train_context)
        if train_context.ndim != 3:
            raise ValueError("Expected training context with shape [batch, context_length, features]")
        if train_context.shape[1] != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}, got {train_context.shape[1]}"
            )
        self.n_features = int(train_context.shape[-1])
        try:
            import torch
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise RuntimeError(
                "Moirai-2 requires uni2ts 2.0.0, GluonTS, and jaxtyping. "
                "Install them in the optional Moirai environment; see docs/extension_roadmap.md."
            ) from exc

        model_id = str(self.config.get("model_id", "Salesforce/moirai-2.0-R-small"))
        module = Moirai2Module.from_pretrained(model_id)
        self.forecast_model = Moirai2Forecast(
            module=module,
            prediction_length=self.horizon,
            context_length=self.context_length,
            target_dim=self.n_features,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.forecast_model.eval()
        predictor_device = str(self.config.get("predictor_device", self.device))
        batch_size = int(self.config.get("inference_batch_size", 32))
        if batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.predictor = self.forecast_model.create_predictor(
            batch_size=batch_size,
            device=predictor_device,
        )
        self.quantile_levels = [float(value) for value in module.quantile_levels]
        point_quantile = float(self.config.get("point_quantile", 0.5))
        if not 0.0 < point_quantile < 1.0:
            raise ValueError("point_quantile must be between 0 and 1")
        if point_quantile not in self.quantile_levels:
            raise ValueError(
                f"point_quantile {point_quantile} is not exposed by Moirai-2: "
                f"{self.quantile_levels}"
            )
        try:
            import importlib.metadata

            uni2ts_version = importlib.metadata.version("uni2ts")
        except importlib.metadata.PackageNotFoundError:
            uni2ts_version = "unknown"
        return {
            "fit_status": "not_applicable_zero_shot",
            "checkpoint": model_id,
            "input_contract": "native multivariate [batch, features, history] via GluonTS wide target",
            "prediction_contract": f"Moirai-2 quantile {point_quantile}",
            "point_quantile": point_quantile,
            "quantile_levels": self.quantile_levels,
            "inference_batch_size": batch_size,
            "internal_scaler": "official PackedStdScaler",
            "external_scaler": "source_only_runner_scaler",
            "uni2ts_version": uni2ts_version,
            "torch_version": torch.__version__,
        }

    @staticmethod
    def _point_forecast(forecast: Any, horizon: int, n_features: int, quantile: float) -> np.ndarray:
        if not hasattr(forecast, "quantile"):
            raise RuntimeError("Moirai-2 predictor returned an object without quantile()")
        values = np.asarray(forecast.quantile(quantile), dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape != (horizon, n_features):
            raise RuntimeError(
                "Unexpected Moirai-2 forecast shape: "
                f"{values.shape}; expected {(horizon, n_features)}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("Moirai-2 returned non-finite forecasts")
        return values

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.predictor is None or self.n_features is None:
            raise RuntimeError("Moirai2Model must be initialized before predict")
        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3:
            raise ValueError("Expected context with shape [batch, context_length, features]")
        batch_size, observed_length, n_features = context.shape
        if observed_length != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}, got {observed_length}"
            )
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
        if batch_size == 0:
            return np.empty((0, self.horizon, n_features), dtype=np.float32)
        if not np.isfinite(context).all():
            raise ValueError("Moirai-2 context contains non-finite values")

        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset

        columns = [f"feature_{index:04d}" for index in range(n_features)]
        frames = {
            f"window_{index:06d}": pd.DataFrame(
                values,
                index=pd.date_range("2000-01-01", periods=self.context_length, freq="h"),
                columns=columns,
            )
            for index, values in enumerate(context)
        }
        dataset = PandasDataset(
            frames,
            target=columns,
            freq="h",
            assume_sorted=True,
        )
        quantile = float(self.config.get("point_quantile", 0.5))
        forecasts = [
            self._point_forecast(forecast, self.horizon, n_features, quantile)
            for forecast in self.predictor.predict(dataset)
        ]
        if len(forecasts) != batch_size:
            raise RuntimeError(
                f"Unexpected number of Moirai-2 forecasts: {len(forecasts)} != {batch_size}"
            )
        return np.stack(forecasts, axis=0).astype(np.float32)

    def parameter_count(self) -> int | None:
        if self.forecast_model is None:
            return None
        return _count_parameters(getattr(self.forecast_model, "module", self.forecast_model))

    def trainable_parameter_count(self) -> int | None:
        return 0 if self.forecast_model is not None else None
