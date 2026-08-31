from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ForecastModel


class PatchTSTNetwork(nn.Module):
    """Compact PatchTST-style channel-independent forecasting network."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        n_features: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_len > context_length:
            raise ValueError("patch_len cannot exceed context_length")
        n_patches = (context_length - patch_len) // stride + 1
        if n_patches < 1:
            raise ValueError("Patch configuration produces no patches")
        self.n_features = n_features
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = n_patches
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B*C, N, d_model], which shares weights across channels.
        patches = inputs.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_embedding(patches) + self.position
        batch_size, channels, n_patches, d_model = tokens.shape
        tokens = tokens.reshape(batch_size * channels, n_patches, d_model)
        encoded = self.norm(self.encoder(tokens))
        output = self.head(encoded.reshape(batch_size * channels, -1))
        return output.reshape(batch_size, channels, self.horizon).transpose(1, 2)


class PatchTSTModel(ForecastModel):
    name = "patchtst"
    training_mode = "supervised"

    def __init__(
        self, config: dict[str, Any], context_length: int, horizon: int, device: str
    ) -> None:
        self.config = config
        self.context_length = context_length
        self.horizon = horizon
        self.device = torch.device(device)
        self.network: PatchTSTNetwork | None = None

    def fit(
        self,
        train_context: np.ndarray,
        train_target: np.ndarray,
        validation_context: np.ndarray | None,
        validation_target: np.ndarray | None,
        seed: int,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        n_features = int(train_context.shape[-1])
        self.network = PatchTSTNetwork(
            context_length=self.context_length,
            horizon=self.horizon,
            n_features=n_features,
            patch_len=int(self.config.get("patch_len", 8)),
            stride=int(self.config.get("stride", 4)),
            d_model=int(self.config.get("d_model", 64)),
            n_heads=int(self.config.get("n_heads", 4)),
            e_layers=int(self.config.get("e_layers", 2)),
            d_ff=int(self.config.get("d_ff", 128)),
            dropout=float(self.config.get("dropout", 0.1)),
        ).to(self.device)
        train_dataset = TensorDataset(
            torch.from_numpy(train_context.astype(np.float32)),
            torch.from_numpy(train_target.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.config.get("batch_size", 128)),
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        if validation_context is not None and validation_target is not None:
            validation_dataset = TensorDataset(
                torch.from_numpy(validation_context.astype(np.float32)),
                torch.from_numpy(validation_target.astype(np.float32)),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(self.config.get("batch_size", 128)),
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
        else:
            validation_loader = None
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.config.get("learning_rate", 1.0e-3)),
            weight_decay=float(self.config.get("weight_decay", 1.0e-4)),
        )
        loss_function = nn.MSELoss()
        max_epochs = int(self.config.get("epochs", 10))
        patience = int(self.config.get("patience", 3))
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []

        for epoch in range(1, max_epochs + 1):
            self.network.train()
            epoch_losses: list[float] = []
            for batch_context, batch_target in train_loader:
                batch_context = batch_context.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(batch_context)
                loss = loss_function(prediction, batch_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(epoch_losses))
            train_losses.append(train_loss)
            if validation_loader is None:
                validation_loss = train_loss
            else:
                self.network.eval()
                val_losses: list[float] = []
                with torch.inference_mode():
                    for batch_context, batch_target in validation_loader:
                        prediction = self.network(batch_context.to(self.device, non_blocking=True))
                        val_losses.append(
                            float(loss_function(prediction, batch_target.to(self.device)).cpu())
                        )
                validation_loss = float(np.mean(val_losses))
            validation_losses.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if validation_loader is not None and epochs_without_improvement >= patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return {
            "fit_status": "complete",
            "best_epoch": best_epoch,
            "best_validation_mse_scaled": best_loss,
            "train_loss_history": train_losses,
            "validation_loss_history": validation_losses,
        }

    def fine_tune(
        self,
        context: np.ndarray,
        target: np.ndarray,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        seed: int,
    ) -> dict[str, Any]:
        """Continue training the source model on target support windows.

        The caller is responsible for restoring a source checkpoint before
        each independent support-budget run.  No target validation windows
        are used here; the adaptation budget and optimizer settings are fixed
        in the experiment configuration.
        """
        if self.network is None:
            raise RuntimeError("PatchTSTModel must be fitted before fine_tune")
        if len(context) == 0:
            raise ValueError("At least one target support window is required")
        if epochs <= 0 or learning_rate <= 0 or batch_size <= 0:
            raise ValueError("epochs, learning_rate, and batch_size must be positive")
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        dataset = TensorDataset(
            torch.from_numpy(context.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_function = nn.MSELoss()
        loss_history: list[float] = []
        for _ in range(epochs):
            self.network.train()
            losses: list[float] = []
            for batch_context, batch_target in loader:
                batch_context = batch_context.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.network(batch_context)
                loss = loss_function(prediction, batch_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            loss_history.append(float(np.mean(losses)))
        return {
            "fit_status": "complete",
            "adaptation": "few_shot_finetune",
            "epochs": epochs,
            "n_support_windows": len(context),
            "final_support_mse_scaled": loss_history[-1],
            "loss_history": loss_history,
        }

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("PatchTSTModel must be fitted before predict")
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(context.astype(np.float32)))
        loader = DataLoader(
            dataset, batch_size=int(self.config.get("batch_size", 128)), shuffle=False
        )
        predictions: list[np.ndarray] = []
        self.network.eval()
        with torch.inference_mode():
            for (batch_context,) in loader:
                prediction = self.network(batch_context.to(self.device, non_blocking=True))
                predictions.append(prediction.cpu().numpy())
        return np.concatenate(predictions, axis=0).astype(np.float32)

    def parameter_count(self) -> int | None:
        return None if self.network is None else sum(p.numel() for p in self.network.parameters())

    def trainable_parameter_count(self) -> int | None:
        return (
            None
            if self.network is None
            else sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        )
