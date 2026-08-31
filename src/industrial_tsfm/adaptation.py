from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .config import load_config, resolve_device, write_json
from .data import (
    TrainOnlyStandardScaler,
    load_cmapss,
    make_windows,
    observed_prefix_before_origins,
)
from .evaluation import compute_metrics, metrics_detail_rows, write_result_summary
from .models import (
    Chronos2Model,
    ChronosModel,
    Moirai2Model,
    NaiveModel,
    PatchTSTModel,
    TimesFMModel,
)
from .selection import recommend_adaptation_strategy

SUPPORTED_MODELS = {"naive", "patchtst", "timesfm", "chronos", "chronos2", "moirai2"}


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_gpu_memory(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_gpu_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))


def _package_versions() -> dict[str, str]:
    names = [
        "industrial-tsfm",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "torch",
        "timesfm",
        "chronos-forecasting",
        "uni2ts",
        "gluonts",
        "transformers",
        "peft",
        "accelerate",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _make_model(
    name: str, config: dict[str, Any], context_length: int, horizon: int, device: str
) -> Any:
    model_config = config.get("models", {}).get(name, {})
    if name == "naive":
        return NaiveModel().configure(horizon)
    if name == "patchtst":
        return PatchTSTModel(model_config, context_length, horizon, device)
    if name == "timesfm":
        return TimesFMModel(model_config, context_length, horizon, device)
    if name == "chronos":
        return ChronosModel(model_config, context_length, horizon, device)
    if name == "chronos2":
        return Chronos2Model(model_config, context_length, horizon, device)
    if name == "moirai2":
        return Moirai2Model(model_config, context_length, horizon, device)
    raise ValueError(f"Unknown model: {name}")


def _support_label(fraction: float) -> str:
    return f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def _select_support_budgets(
    candidate_entities: tuple[str, ...], fractions: list[float], seed: int
) -> list[dict[str, Any]]:
    if not candidate_entities:
        raise ValueError("At least one target support candidate is required")
    if not fractions:
        raise ValueError("At least one support fraction is required")
    if any(fraction < 0 or fraction > 1 for fraction in fractions):
        raise ValueError("support_fractions must be between 0 and 1")
    rng = np.random.default_rng(seed)
    permutation = np.asarray(candidate_entities, dtype=object)[
        rng.permutation(len(candidate_entities))
    ]
    budgets: list[dict[str, Any]] = []
    for fraction in fractions:
        if fraction == 0:
            n_support = 0
        else:
            n_support = min(
                len(candidate_entities), max(1, round(fraction * len(candidate_entities)))
            )
        support_entities = tuple(sorted(permutation[:n_support].tolist()))
        budgets.append(
            {
                "fraction": float(fraction),
                "label": _support_label(float(fraction)),
                "n_support_entities": n_support,
                "support_entities": support_entities,
            }
        )
    return budgets


def _scaled_windows(windows: Any, scaler: TrainOnlyStandardScaler) -> tuple[np.ndarray, np.ndarray]:
    return scaler.transform(windows.context), scaler.transform(windows.target)


def _write_variant_outputs(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    detail_frames: list[pd.DataFrame],
    model_name: str,
    adaptation: str,
    scaling: str,
    support_fraction: float,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: np.ndarray,
    normalized_scale: np.ndarray,
    feature_names: tuple[str, ...],
    entity_ids: tuple[str, ...],
    support_entities: tuple[str, ...],
    n_support_windows: int,
    base_fit_seconds: float,
    adaptation_fit_seconds: float,
    inference_seconds: float,
    fit_info: dict[str, Any],
    total_parameters: int | None,
    trainable_parameters: int | None,
    peak_gpu_memory_mb: float | None,
) -> None:
    variant = f"{model_name}_{adaptation}_{scaling}_support_{_support_label(support_fraction)}"
    prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
    np.savez_compressed(
        prediction_path,
        y_true=y_true,
        y_pred=y_pred,
        entity_ids=np.asarray(entity_ids, dtype=str),
        origins=np.asarray(fit_info["evaluation_origins"], dtype=np.int64),
        feature_names=np.asarray(feature_names, dtype=str),
        support_entity_ids=np.asarray(support_entities, dtype=str),
    )
    entity_array = np.asarray(entity_ids, dtype=object)
    masks: list[tuple[str, np.ndarray]] = [("all_test", np.ones(len(entity_ids), dtype=bool))]
    if support_entities:
        support_set = set(support_entities)
        heldout_mask = np.asarray([entity not in support_set for entity in entity_ids], dtype=bool)
        if heldout_mask.any():
            masks.append(("heldout_target", heldout_mask))
    for evaluation_scope, mask in masks:
        metrics = compute_metrics(
            y_true[mask],
            y_pred[mask],
            mase_scale,
            entity_array[mask].tolist(),
            normalized_scale=normalized_scale,
        )
        result_rows.append(
            {
                "model": model_name,
                "adaptation": adaptation,
                "scaling": scaling,
                "support_fraction": support_fraction,
                "evaluation_scope": evaluation_scope,
                "seed": seed,
                "status": "complete",
                "training_mode": {
                    "patchtst": "supervised",
                    "timesfm": "zero_shot",
                    "chronos": "zero_shot",
                    "chronos2": "zero_shot",
                    "moirai2": "zero_shot",
                    "naive": "none",
                }.get(
                    model_name,
                    "unknown",
                ),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mase": metrics["mase"],
                "normalized_mae": metrics["normalized_mae"],
                "normalized_rmse": metrics["normalized_rmse"],
                "macro_entity_mae": metrics["macro_entity_mae"],
                "macro_entity_rmse": metrics["macro_entity_rmse"],
                "macro_entity_normalized_mae": metrics["macro_entity_normalized_mae"],
                "macro_entity_normalized_rmse": metrics["macro_entity_normalized_rmse"],
                "n_windows": int(metrics["n_windows"]),
                "n_entities": int(metrics["n_entities"]),
                "n_support_entities": len(support_entities),
                "n_support_windows": n_support_windows,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "base_fit_seconds": base_fit_seconds,
                "adaptation_fit_seconds": adaptation_fit_seconds,
                "inference_seconds": inference_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "fit_info": json.dumps(
                    {**fit_info, "variant": variant, "evaluation_scope": evaluation_scope},
                    sort_keys=True,
                    default=str,
                ),
            }
        )
        detail = metrics_detail_rows(
            variant,
            seed,
            y_true[mask],
            y_pred[mask],
            mase_scale,
            feature_names,
            entity_array[mask].tolist(),
            normalized_scale=normalized_scale,
        )
        detail["adaptation"] = adaptation
        detail["scaling"] = scaling
        detail["support_fraction"] = support_fraction
        detail["evaluation_scope"] = evaluation_scope
        detail_frames.append(detail)


def run_adaptation(config_path: str | Path, requested_models: list[str] | None = None) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    runtime = config.get("runtime", {})
    dataset_config = config.get("dataset", {})
    adaptation_config = config.get("adaptation", {})
    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    device = resolve_device(str(runtime.get("device", "auto")))
    torch.set_num_threads(int(runtime.get("torch_num_threads", 1)))
    raw_dir = project_root / dataset_config.get("raw_dir", "data/raw")
    dataset = load_cmapss(
        raw_dir=raw_dir,
        subset=str(dataset_config.get("subset", "FD001")),
        include_settings=bool(dataset_config.get("include_settings", True)),
        include_sensors=bool(dataset_config.get("include_sensors", True)),
        train_fraction=float(dataset_config.get("train_fraction", 0.8)),
        validation_fraction=float(dataset_config.get("validation_fraction", 0.2)),
        entity_split_seed=int(dataset_config.get("entity_split_seed", 2026)),
    )
    source_scaler = TrainOnlyStandardScaler.fit(
        dataset.train_frame,
        dataset.splits.train,
        dataset.feature_columns,
        epsilon=float(config.get("preprocessing", {}).get("epsilon", 1.0e-8)),
    )
    train_windows = make_windows(
        dataset.train_frame,
        dataset.splits.train,
        dataset.feature_columns,
        context_length,
        horizon,
        stride=int(dataset_config.get("train_window_stride", 1)),
        last_only=False,
    )
    validation_windows = make_windows(
        dataset.train_frame,
        dataset.splits.validation,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    test_windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    train_context, train_target = _scaled_windows(train_windows, source_scaler)
    validation_context, validation_target = _scaled_windows(validation_windows, source_scaler)
    test_context_source = source_scaler.transform(test_windows.context)
    target_raw = test_windows.target.astype(np.float32)
    mase_scale = source_scaler.mase_scale(dataset.train_frame)

    observed_prefix = observed_prefix_before_origins(
        dataset.test_frame, test_windows.entity_ids, test_windows.origins
    )
    support_window_stride = int(adaptation_config.get("support_window_stride", 16))
    all_support_windows = make_windows(
        observed_prefix,
        test_windows.entity_ids,
        dataset.feature_columns,
        context_length,
        horizon,
        stride=support_window_stride,
        last_only=False,
    )
    support_candidate_entities = tuple(
        entity
        for entity in test_windows.entity_ids
        if entity not in all_support_windows.skipped_entities
    )
    support_ineligible_entities = tuple(
        entity
        for entity in test_windows.entity_ids
        if entity in all_support_windows.skipped_entities
    )
    support_fractions = [
        float(value)
        for value in adaptation_config.get("support_fractions", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    ]
    support_budgets = _select_support_budgets(
        support_candidate_entities,
        support_fractions,
        int(adaptation_config.get("support_seed", 2026)),
    )
    support_window_by_label: dict[str, Any] = {}
    target_scaler_by_label: dict[str, TrainOnlyStandardScaler] = {}
    budget_records: list[dict[str, Any]] = []
    for budget in support_budgets:
        support_entities = tuple(budget["support_entities"])
        if support_entities:
            support_windows = make_windows(
                observed_prefix,
                support_entities,
                dataset.feature_columns,
                context_length,
                horizon,
                stride=support_window_stride,
                last_only=False,
            )
            target_scaler = TrainOnlyStandardScaler.fit(
                observed_prefix,
                support_entities,
                dataset.feature_columns,
                epsilon=float(config.get("preprocessing", {}).get("epsilon", 1.0e-8)),
            )
        else:
            support_windows = None
            target_scaler = source_scaler
        support_window_by_label[budget["label"]] = support_windows
        target_scaler_by_label[budget["label"]] = target_scaler
        budget_records.append(
            {
                "fraction": budget["fraction"],
                "label": budget["label"],
                "n_support_entities": budget["n_support_entities"],
                "support_entities": list(support_entities),
                "n_support_windows": 0 if support_windows is None else len(support_windows),
                "support_window_skipped_entities": []
                if support_windows is None
                else list(support_windows.skipped_entities),
                "target_scaler_fit_rows": target_scaler.fit_rows,
                "target_scaler_fit_entities": list(target_scaler.fit_entities),
            }
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{_config_hash(config)}"
    output_dir = project_root / runtime.get("output_dir", "results") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("predictions", "calibrators", "scalers"):
        (output_dir / directory).mkdir()
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "dataset_summary.json", dataset.summary())
    write_json(output_dir / "split_manifest.json", dataset.splits.as_dict())
    source_scaler_manifest = source_scaler.as_dict()
    source_scaler_manifest["mase_scale"] = mase_scale.tolist()
    write_json(output_dir / "source_scaler.json", source_scaler_manifest)
    write_json(
        output_dir / "support_manifest.json",
        {
            "protocol": "target support is restricted to observed rows with cycle < final evaluation origin",
            "context_length": context_length,
            "horizon": horizon,
            "support_window_stride": support_window_stride,
            "support_seed": int(adaptation_config.get("support_seed", 2026)),
            "evaluation_origins_by_entity": dict(
                zip(test_windows.entity_ids, test_windows.origins.tolist(), strict=True)
            ),
            "target_observed_prefix_rows_by_entity": {
                entity: len(frame)
                for entity, frame in observed_prefix.groupby("entity_id", sort=True)
            },
            "support_candidate_entities": list(support_candidate_entities),
            "support_ineligible_entities": list(support_ineligible_entities),
            "budgets": budget_records,
        },
    )
    for budget in budget_records:
        target_scaler = target_scaler_by_label[budget["label"]]
        scaler_manifest = target_scaler.as_dict()
        scaler_manifest.update(
            {
                "scaling_scope": "source_train"
                if budget["n_support_entities"] == 0
                else "target_support_observed_prefix",
                "support_fraction": budget["fraction"],
                "support_entities": budget["support_entities"],
                "mase_scale": mase_scale.tolist(),
            }
        )
        write_json(output_dir / "scalers" / f"support_{budget['label']}.json", scaler_manifest)

    model_names = requested_models
    if model_names is None:
        model_names = [
            name
            for name in ("naive", "patchtst", "timesfm", "chronos", "chronos2", "moirai2")
            if bool(config.get("models", {}).get(name, {}).get("enabled", False))
        ]
    unknown = set(model_names) - SUPPORTED_MODELS
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if not model_names:
        raise ValueError("No enabled models were selected")

    training_modes = {
        "naive": "none",
        "patchtst": "supervised",
        "timesfm": "zero_shot",
        "chronos": "zero_shot",
        "chronos2": "zero_shot",
        "moirai2": "zero_shot",
    }
    strategy_config = adaptation_config.get("strategy_selection", {})
    strategy_recommendations = [
        {
            **recommend_adaptation_strategy(
                model_name,
                training_modes[model_name],
                float(budget["fraction"]),
                int(budget["n_support_windows"]),
                residual_calibration_enabled=bool(
                    adaptation_config.get("residual_calibration", {}).get("enabled", True)
                ),
                min_calibration_windows=int(strategy_config.get("min_calibration_windows", 2)),
            ),
            "support_entities": list(budget["support_entities"]),
        }
        for model_name in model_names
        for budget in budget_records
    ]
    write_json(
        output_dir / "strategy_recommendations.json",
        {
            "policy": "support_aware_contract_policy_v1",
            "uses_target_labels": False,
            "recommendations": strategy_recommendations,
        },
    )

    result_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    seeds = [int(seed) for seed in runtime.get("seeds", [42])]
    for seed in seeds:
        _set_seed(seed)
        for model_name in model_names:
            _reset_gpu_memory(device)
            model = _make_model(model_name, config, context_length, horizon, device)
            base_fit_start = time.perf_counter()
            base_fit_info: dict[str, Any]
            try:
                base_fit_info = model.fit(
                    train_context,
                    train_target,
                    validation_context,
                    validation_target,
                    seed,
                )
            except Exception as exc:
                write_json(
                    output_dir / "failure.json",
                    {"model": model_name, "seed": seed, "error": f"{type(exc).__name__}: {exc}"},
                )
                raise
            base_fit_seconds = time.perf_counter() - base_fit_start
            base_peak_memory = _peak_gpu_memory_mb(device)

            if model_name == "naive":
                inference_start = time.perf_counter()
                prediction_scaled = model.predict(test_context_source)
                inference_seconds = time.perf_counter() - inference_start
                prediction_raw = source_scaler.inverse_transform(prediction_scaled)
                _write_variant_outputs(
                    output_dir,
                    result_rows,
                    detail_frames,
                    model_name,
                    "none",
                    "source_train",
                    0.0,
                    seed,
                    target_raw,
                    prediction_raw,
                    mase_scale,
                    source_scaler.scale,
                    dataset.feature_columns,
                    test_windows.entity_ids,
                    (),
                    0,
                    base_fit_seconds,
                    0.0,
                    inference_seconds,
                    {
                        **base_fit_info,
                        "evaluation_origins": test_windows.origins.tolist(),
                    },
                    model.parameter_count(),
                    model.trainable_parameter_count(),
                    base_peak_memory,
                )
                del model
                continue

            if model_name == "patchtst":
                if not isinstance(model, PatchTSTModel) or model.network is None:
                    raise RuntimeError("PatchTST source model was not initialized")
                source_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.network.state_dict().items()
                }
                patch_config = adaptation_config.get("patchtst", {})
                for budget in support_budgets:
                    label = budget["label"]
                    support_windows = support_window_by_label[label]
                    support_entities = tuple(budget["support_entities"])
                    model.network.load_state_dict(source_state)
                    adaptation_fit_seconds = 0.0
                    adaptation_fit_info: dict[str, Any] = {}
                    if support_windows is None or len(support_windows) == 0:
                        if budget["n_support_entities"] > 0:
                            continue
                        adaptation = "source_supervised"
                        scaling = "source_train"
                    else:
                        support_context, support_target = _scaled_windows(
                            support_windows, source_scaler
                        )
                        adaptation = "few_shot_finetune"
                        scaling = "source_train"
                        adaptation_fit_start = time.perf_counter()
                        adaptation_fit_info = model.fine_tune(
                            support_context,
                            support_target,
                            epochs=int(patch_config.get("epochs", 3)),
                            learning_rate=float(patch_config.get("learning_rate", 1.0e-4)),
                            weight_decay=float(patch_config.get("weight_decay", 1.0e-4)),
                            batch_size=int(patch_config.get("batch_size", 64)),
                            seed=seed,
                        )
                        adaptation_fit_seconds = time.perf_counter() - adaptation_fit_start
                    inference_start = time.perf_counter()
                    prediction_scaled = model.predict(test_context_source)
                    inference_seconds = time.perf_counter() - inference_start
                    prediction_raw = source_scaler.inverse_transform(prediction_scaled)
                    _write_variant_outputs(
                        output_dir,
                        result_rows,
                        detail_frames,
                        model_name,
                        adaptation,
                        scaling,
                        budget["fraction"],
                        seed,
                        target_raw,
                        prediction_raw,
                        mase_scale,
                        source_scaler.scale,
                        dataset.feature_columns,
                        test_windows.entity_ids,
                        support_entities,
                        0 if support_windows is None else len(support_windows),
                        base_fit_seconds,
                        adaptation_fit_seconds,
                        inference_seconds,
                        {
                            **base_fit_info,
                            **adaptation_fit_info,
                            "evaluation_origins": test_windows.origins.tolist(),
                            "support_entities": list(support_entities),
                        },
                        model.parameter_count(),
                        model.trainable_parameter_count(),
                        _peak_gpu_memory_mb(device),
                    )
                del model
                continue

            if model_name not in {"timesfm", "chronos", "chronos2", "moirai2"}:
                raise AssertionError(f"Unhandled adaptation model: {model_name}")
            calibration_enabled = bool(
                adaptation_config.get("residual_calibration", {}).get("enabled", True)
            )
            for budget in support_budgets:
                label = budget["label"]
                support_entities = tuple(budget["support_entities"])
                support_windows = support_window_by_label[label]
                target_scaler = target_scaler_by_label[label]
                scaling = "source_train" if not support_entities else "target_support"
                final_context_scaled = target_scaler.transform(test_windows.context)
                inference_start = time.perf_counter()
                final_prediction_scaled = model.predict(final_context_scaled)
                final_inference_seconds = time.perf_counter() - inference_start
                final_prediction_raw = target_scaler.inverse_transform(final_prediction_scaled)
                fit_info = {
                    **base_fit_info,
                    "evaluation_origins": test_windows.origins.tolist(),
                    "support_entities": list(support_entities),
                    "support_scaling": scaling,
                }
                _write_variant_outputs(
                    output_dir,
                    result_rows,
                    detail_frames,
                    model_name,
                    "zero_shot",
                    scaling,
                    budget["fraction"],
                    seed,
                    target_raw,
                    final_prediction_raw,
                    mase_scale,
                    source_scaler.scale,
                    dataset.feature_columns,
                    test_windows.entity_ids,
                    support_entities,
                    0 if support_windows is None else len(support_windows),
                    base_fit_seconds,
                    0.0,
                    final_inference_seconds,
                    fit_info,
                    model.parameter_count(),
                    model.trainable_parameter_count(),
                    _peak_gpu_memory_mb(device),
                )
                if not calibration_enabled or not support_entities or support_windows is None:
                    continue
                support_context_scaled = target_scaler.transform(support_windows.context)
                support_inference_start = time.perf_counter()
                support_prediction_scaled = model.predict(support_context_scaled)
                support_inference_seconds = time.perf_counter() - support_inference_start
                support_prediction_raw = target_scaler.inverse_transform(support_prediction_scaled)
                bias = (support_windows.target - support_prediction_raw).mean(axis=(0, 1))
                calibrated_prediction_raw = final_prediction_raw + bias[None, None, :]
                calibrator = {
                    "model": model_name,
                    "method": "featurewise_additive_residual_bias",
                    "support_fraction": budget["fraction"],
                    "support_entities": list(support_entities),
                    "n_support_windows": len(support_windows),
                    "scaling": scaling,
                    "feature_names": list(dataset.feature_columns),
                    "bias_raw": bias.tolist(),
                }
                write_json(
                    output_dir / "calibrators" / f"{model_name}_support_{label}_seed{seed}.json",
                    calibrator,
                )
                _write_variant_outputs(
                    output_dir,
                    result_rows,
                    detail_frames,
                    model_name,
                    "target_residual_calibration",
                    scaling,
                    budget["fraction"],
                    seed,
                    target_raw,
                    calibrated_prediction_raw,
                    mase_scale,
                    source_scaler.scale,
                    dataset.feature_columns,
                    test_windows.entity_ids,
                    support_entities,
                    len(support_windows),
                    base_fit_seconds,
                    0.0,
                    final_inference_seconds + support_inference_seconds,
                    {
                        **fit_info,
                        "calibrator": calibrator,
                    },
                    model.parameter_count(),
                    model.trainable_parameter_count(),
                    _peak_gpu_memory_mb(device),
                )
            del model

    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(output_dir / "result_table.csv", index=False)
    write_result_summary(
        result_table,
        output_dir / "result_summary.csv",
        ["model", "adaptation", "scaling", "support_fraction", "evaluation_scope"],
    )
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            output_dir / "metrics_detail.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "metrics_detail.csv", index=False)
    write_json(
        output_dir / "manifest.json",
        {
            "run_id": run_id,
            "config_path": str(config_path),
            "config_hash": _config_hash(config),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(project_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "package_versions": _package_versions(),
            "data_raw_dir": str(raw_dir),
            "raw_file_sha256": {
                name: _file_sha256(raw_dir / name)
                for name in (
                    f"train_{dataset.subset}.txt",
                    f"test_{dataset.subset}.txt",
                    f"RUL_{dataset.subset}.txt",
                )
            },
            "window_shapes": {
                "train_context": list(train_context.shape),
                "validation_context": list(validation_context.shape),
                "test_context": list(test_context_source.shape),
                "all_support_windows": list(all_support_windows.context.shape),
            },
            "support_candidate_entities": list(support_candidate_entities),
            "support_ineligible_entities": list(support_ineligible_entities),
            "no_final_target_used_for_adaptation": True,
        },
    )
    return output_dir
