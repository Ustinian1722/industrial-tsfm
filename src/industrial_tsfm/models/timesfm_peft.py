from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import ForecastModel


class TimesFMTransformersModel(ForecastModel):
    """TimesFM 2.5 Transformers API with an optional PEFT LoRA adapter.

    This adapter intentionally uses raw values. The official Transformers
    checkpoint performs instance normalization internally and accepts a
    sequence of one-dimensional series rather than a batched 3-D tensor.
    """

    name = "timesfm_transformers"
    training_mode = "peft"

    def __init__(
        self,
        config: dict[str, Any],
        context_length: int,
        horizon: int,
        device: str,
        use_lora: bool = True,
    ) -> None:
        self.config = config
        self.context_length = context_length
        self.horizon = horizon
        self.device = torch.device(device)
        self.use_lora = use_lora
        self.model: Any | None = None

    def _load_model(self) -> dict[str, Any]:
        try:
            from transformers import TimesFm2_5ModelForPrediction
        except ImportError as exc:
            raise RuntimeError("Transformers does not expose TimesFm2_5ModelForPrediction") from exc
        model_id = self.config.get("model_id", "google/timesfm-2.5-200m-transformers")
        model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
        lora_info: dict[str, Any] = {"enabled": False}
        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError(
                    "PEFT is not installed. Install the project's [peft] extra."
                ) from exc
            lora_config = LoraConfig(
                r=int(self.config.get("lora_r", 4)),
                lora_alpha=int(self.config.get("lora_alpha", 8)),
                lora_dropout=float(self.config.get("lora_dropout", 0.05)),
                target_modules=self.config.get("target_modules", "all-linear"),
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            lora_info = {
                "enabled": True,
                "r": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "dropout": lora_config.lora_dropout,
                "target_modules": self.config.get("target_modules", "all-linear"),
            }
        self.model = model.to(self.device)
        return {
            "checkpoint": model_id,
            "input_contract": "raw_values_with_internal_instance_normalization",
            "forecast_context_len": self.context_length,
            "force_flip_invariance": bool(self.config.get("force_flip_invariance", False)),
            "lora": lora_info,
        }

    @staticmethod
    def _flatten_series(context: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        context = np.asarray(context, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if context.ndim != 3 or target.ndim != 3:
            raise ValueError("Expected context/target with shape [batch, time, features]")
        if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
            raise ValueError("Context and target batch/features must match")
        flat_context = context.transpose(0, 2, 1).reshape(-1, context.shape[1])
        flat_target = target.transpose(0, 2, 1).reshape(-1, target.shape[1])
        return flat_context, flat_target

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        del validation_context, validation_target
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        fit_info = self._load_model()
        if not self.use_lora:
            return {"fit_status": "not_applicable_base_transformers", **fit_info}
        if self.model is None:
            raise RuntimeError("TimesFM Transformers model failed to load")
        flat_context, flat_target = self._flatten_series(train_context, train_target)
        max_series = int(self.config.get("max_train_series", 5000))
        if max_series > 0 and len(flat_context) > max_series:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(flat_context), size=max_series, replace=False)
            flat_context = flat_context[indices]
            flat_target = flat_target[indices]
        if len(flat_context) == 0:
            raise ValueError("At least one series is required for PEFT training")
        dataset = TensorDataset(torch.from_numpy(flat_context), torch.from_numpy(flat_target))
        batch_size = int(self.config.get("batch_size", 32))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=float(self.config.get("learning_rate", 1.0e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        epochs = int(self.config.get("epochs", 1))
        if epochs <= 0:
            raise ValueError("PEFT epochs must be positive")
        loss_history: list[float] = []
        for _ in range(epochs):
            self.model.train()
            losses: list[float] = []
            for batch_context, batch_target in loader:
                past_values = [row.to(self.device, non_blocking=True) for row in batch_context]
                future_values = batch_target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(
                    past_values=past_values,
                    future_values=future_values,
                    forecast_context_len=self.context_length,
                    force_flip_invariance=bool(self.config.get("force_flip_invariance", False)),
                )
                if outputs.loss is None:
                    raise RuntimeError("TimesFM Transformers did not return a training loss")
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(self.config.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                losses.append(float(outputs.loss.detach().cpu()))
            loss_history.append(float(np.mean(losses)))
        return {
            "fit_status": "complete",
            **fit_info,
            "n_train_series": len(flat_context),
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_history": loss_history,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TimesFMTransformersModel must be fitted before predict")
        context = np.asarray(context, dtype=np.float32)
        if context.ndim != 3 or context.shape[1] != self.context_length:
            raise ValueError(
                "Expected context with shape [batch, configured_context_length, features]"
            )
        batch_size, _, n_features = context.shape
        flat_context = context.transpose(0, 2, 1).reshape(-1, context.shape[1])
        inference_batch_size = int(self.config.get("inference_batch_size", 32))
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        predictions: list[np.ndarray] = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(flat_context), inference_batch_size):
                batch = torch.from_numpy(flat_context[start : start + inference_batch_size])
                outputs = self.model(
                    past_values=[row.to(self.device, non_blocking=True) for row in batch],
                    forecast_context_len=self.context_length,
                    force_flip_invariance=bool(self.config.get("force_flip_invariance", False)),
                )
                predictions.append(
                    outputs.mean_predictions[:, : self.horizon].detach().cpu().numpy()
                )
        flat_prediction = np.concatenate(predictions, axis=0).astype(np.float32)
        if flat_prediction.shape != (batch_size * n_features, self.horizon):
            raise RuntimeError(
                f"Unexpected TimesFM Transformers output shape: {flat_prediction.shape}"
            )
        return flat_prediction.reshape(batch_size, n_features, self.horizon).transpose(0, 2, 1)

    def save_adapter(self, path: str | Path) -> None:
        if self.model is None or not self.use_lora:
            raise RuntimeError("Only a fitted PEFT model can save an adapter")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    def parameter_count(self) -> int | None:
        return (
            None
            if self.model is None
            else sum(parameter.numel() for parameter in self.model.parameters())
        )

    def trainable_parameter_count(self) -> int | None:
        return (
            None
            if self.model is None
            else sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        )
