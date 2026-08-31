from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError("Expected y_true/y_pred with identical shape [batch, horizon, features]")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics cannot be computed on non-finite predictions/targets")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: np.ndarray,
    entity_ids: Iterable[str],
    normalized_scale: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    mase_scale = np.asarray(mase_scale, dtype=np.float64)
    if mase_scale.shape != (y_true.shape[-1],) or np.any(mase_scale <= 0):
        raise ValueError("mase_scale must contain one positive value per feature")
    if normalized_scale is not None:
        normalized_scale = np.asarray(normalized_scale, dtype=np.float64)
        if normalized_scale.shape != (y_true.shape[-1],) or np.any(normalized_scale <= 0):
            raise ValueError("normalized_scale must contain one positive value per feature")
    entities = np.asarray(tuple(entity_ids), dtype=object)
    if len(entities) != y_true.shape[0]:
        raise ValueError("entity_ids must contain one entry per forecast window")
    absolute_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    per_window_mae = absolute_error.mean(axis=(1, 2))
    per_window_rmse = np.sqrt(squared_error.mean(axis=(1, 2)))
    unique_entities = sorted(set(entities.tolist()))
    entity_mae = []
    entity_rmse = []
    for entity in unique_entities:
        mask = entities == entity
        entity_mae.append(float(per_window_mae[mask].mean()))
        entity_rmse.append(float(per_window_rmse[mask].mean()))
    scaled_absolute_error = absolute_error / mase_scale[None, None, :]
    metrics = {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "mase": float(scaled_absolute_error.mean()),
        "macro_entity_mae": float(np.mean(entity_mae)),
        "macro_entity_rmse": float(np.mean(entity_rmse)),
        "n_windows": float(y_true.shape[0]),
        "n_entities": float(len(unique_entities)),
    }
    if normalized_scale is not None:
        normalized_error = absolute_error / normalized_scale[None, None, :]
        normalized_squared_error = squared_error / np.square(normalized_scale)[None, None, :]
        per_window_normalized_mae = normalized_error.mean(axis=(1, 2))
        per_window_normalized_rmse = np.sqrt(normalized_squared_error.mean(axis=(1, 2)))
        entity_normalized_mae = []
        entity_normalized_rmse = []
        for entity in unique_entities:
            mask = entities == entity
            entity_normalized_mae.append(float(per_window_normalized_mae[mask].mean()))
            entity_normalized_rmse.append(float(per_window_normalized_rmse[mask].mean()))
        metrics.update(
            {
                "normalized_mae": float(normalized_error.mean()),
                "normalized_rmse": float(np.sqrt(normalized_squared_error.mean())),
                "macro_entity_normalized_mae": float(np.mean(entity_normalized_mae)),
                "macro_entity_normalized_rmse": float(np.mean(entity_normalized_rmse)),
            }
        )
    return metrics


def metrics_detail_rows(
    model: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: np.ndarray,
    feature_names: Iterable[str],
    entity_ids: Iterable[str],
    normalized_scale: np.ndarray | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    if normalized_scale is not None:
        normalized_scale = np.asarray(normalized_scale, dtype=np.float64)
        if normalized_scale.shape != (y_true.shape[-1],) or np.any(normalized_scale <= 0):
            raise ValueError("normalized_scale must contain one positive value per feature")
    feature_names = tuple(feature_names)
    entity_ids = tuple(entity_ids)
    error = y_true - y_pred
    abs_error = np.abs(error)
    sq_error = np.square(error)
    normalized_abs_error = (
        abs_error / normalized_scale[None, None, :] if normalized_scale is not None else None
    )
    normalized_sq_error = (
        sq_error / np.square(normalized_scale)[None, None, :]
        if normalized_scale is not None
        else None
    )
    rows: list[dict[str, object]] = []
    for horizon_index in range(y_true.shape[1]):
        for metric_name, value in (
            ("mae", abs_error[:, horizon_index, :].mean()),
            ("rmse", np.sqrt(sq_error[:, horizon_index, :].mean())),
            ("mase", (abs_error[:, horizon_index, :] / mase_scale).mean()),
        ):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "scope": "horizon",
                    "index": horizon_index + 1,
                    "name": f"horizon_{horizon_index + 1}",
                    "metric": metric_name,
                    "value": float(value),
                }
            )
        if normalized_abs_error is not None and normalized_sq_error is not None:
            for metric_name, value in (
                ("normalized_mae", normalized_abs_error[:, horizon_index, :].mean()),
                ("normalized_rmse", np.sqrt(normalized_sq_error[:, horizon_index, :].mean())),
            ):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "scope": "horizon",
                        "index": horizon_index + 1,
                        "name": f"horizon_{horizon_index + 1}",
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    for feature_index, feature_name in enumerate(feature_names):
        for metric_name, value in (
            ("mae", abs_error[:, :, feature_index].mean()),
            ("rmse", np.sqrt(sq_error[:, :, feature_index].mean())),
            ("mase", (abs_error[:, :, feature_index] / mase_scale[feature_index]).mean()),
            ):
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "scope": "feature",
                    "index": feature_index,
                    "name": feature_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
        if normalized_abs_error is not None and normalized_sq_error is not None:
            for metric_name, value in (
                ("normalized_mae", normalized_abs_error[:, :, feature_index].mean()),
                ("normalized_rmse", np.sqrt(normalized_sq_error[:, :, feature_index].mean())),
            ):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "scope": "feature",
                        "index": feature_index,
                        "name": feature_name,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    for window_index, entity_id in enumerate(entity_ids):
        rows.append(
            {
                "model": model,
                "seed": seed,
                "scope": "entity_window",
                "index": window_index,
                "name": entity_id,
                "metric": "mae",
                "value": float(abs_error[window_index].mean()),
            }
        )
        if normalized_abs_error is not None:
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "scope": "entity_window",
                    "index": window_index,
                    "name": entity_id,
                    "metric": "normalized_mae",
                    "value": float(normalized_abs_error[window_index].mean()),
                }
            )
    return pd.DataFrame(rows)
