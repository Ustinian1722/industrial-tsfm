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

from .adaptation import _select_support_budgets, _support_label
from .config import load_config, resolve_device, write_json
from .data import (
    TrainOnlyStandardScaler,
    load_cmapss,
    make_windows,
    observed_prefix_before_origins,
)
from .evaluation import compute_metrics, metrics_detail_rows
from .models import TimesFMTransformersModel


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _package_versions() -> dict[str, str]:
    names = [
        "industrial-tsfm",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
        "torch",
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


def _write_variant_outputs(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    detail_frames: list[pd.DataFrame],
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
    origins: np.ndarray,
    support_entities: tuple[str, ...],
    n_support_windows: int,
    fit_seconds: float,
    inference_seconds: float,
    fit_info: dict[str, Any],
    total_parameters: int | None,
    trainable_parameters: int | None,
    peak_gpu_memory_mb: float | None,
) -> None:
    variant = f"timesfm_transformers_{adaptation}_support_{_support_label(support_fraction)}"
    prediction_path = output_dir / "predictions" / f"{variant}_seed{seed}.npz"
    np.savez_compressed(
        prediction_path,
        y_true=y_true,
        y_pred=y_pred,
        entity_ids=np.asarray(entity_ids, dtype=str),
        origins=np.asarray(origins, dtype=np.int64),
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
                "model": "timesfm_transformers",
                "adaptation": adaptation,
                "scaling": scaling,
                "support_fraction": support_fraction,
                "evaluation_scope": evaluation_scope,
                "seed": seed,
                "status": "complete",
                "training_mode": "zero_shot" if adaptation == "base_zero_shot" else "peft",
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
                "fit_seconds": fit_seconds,
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


def run_peft(config_path: str | Path, requested_models: list[str] | None = None) -> Path:
    del requested_models
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    runtime = config.get("runtime", {})
    dataset_config = config["dataset"]
    peft_config = config.get("peft", {})
    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    if context_length % 32 != 0:
        raise ValueError(
            "The official TimesFM PEFT protocol requires a context length divisible by 32"
        )
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
    test_windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    target_raw = test_windows.target.astype(np.float32)
    mase_scale = source_scaler.mase_scale(dataset.train_frame)
    observed_prefix = observed_prefix_before_origins(
        dataset.test_frame, test_windows.entity_ids, test_windows.origins
    )
    support_window_stride = int(peft_config.get("support_window_stride", 32))
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
    budgets = _select_support_budgets(
        support_candidate_entities,
        [float(value) for value in peft_config.get("support_fractions", [0.0, 0.05, 0.1])],
        int(peft_config.get("support_seed", 2026)),
    )
    support_windows_by_label: dict[str, Any] = {}
    budget_records: list[dict[str, Any]] = []
    for budget in budgets:
        support_entities = tuple(budget["support_entities"])
        support_windows = (
            make_windows(
                observed_prefix,
                support_entities,
                dataset.feature_columns,
                context_length,
                horizon,
                stride=support_window_stride,
                last_only=False,
            )
            if support_entities
            else None
        )
        support_windows_by_label[budget["label"]] = support_windows
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
            }
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{_config_hash(config)}"
    output_dir = project_root / runtime.get("output_dir", "results") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("predictions", "adapters"):
        (output_dir / directory).mkdir()
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "dataset_summary.json", dataset.summary())
    write_json(output_dir / "split_manifest.json", dataset.splits.as_dict())
    scaler_manifest = source_scaler.as_dict()
    scaler_manifest["mase_scale"] = mase_scale.tolist()
    scaler_manifest["input_scaling"] = "none_raw_values"
    write_json(output_dir / "source_scaler.json", scaler_manifest)
    write_json(
        output_dir / "support_manifest.json",
        {
            "protocol": "official TimesFM Transformers API; raw values; target support cycle < final origin",
            "context_length": context_length,
            "horizon": horizon,
            "support_window_stride": support_window_stride,
            "support_seed": int(peft_config.get("support_seed", 2026)),
            "evaluation_entities": list(test_windows.entity_ids),
            "evaluation_skipped_entities": list(test_windows.skipped_entities),
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

    result_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    seeds = [int(seed) for seed in runtime.get("seeds", [42])]
    model_config = config.get("models", {}).get("timesfm_transformers", {})
    for seed in seeds:
        _set_seed(seed)
        _reset_gpu_memory(device)
        base_model = TimesFMTransformersModel(
            model_config, context_length, horizon, device, use_lora=False
        )
        base_fit_start = time.perf_counter()
        base_fit_info = base_model.fit(
            train_windows.context,
            train_windows.target,
            None,
            None,
            seed,
        )
        base_fit_seconds = time.perf_counter() - base_fit_start
        base_inference_start = time.perf_counter()
        base_prediction = base_model.predict(test_windows.context)
        base_inference_seconds = time.perf_counter() - base_inference_start
        _write_variant_outputs(
            output_dir,
            result_rows,
            detail_frames,
            "base_zero_shot",
            "none_raw",
            0.0,
            seed,
            target_raw,
            base_prediction,
            mase_scale,
            source_scaler.scale,
            dataset.feature_columns,
            test_windows.entity_ids,
            test_windows.origins,
            (),
            0,
            base_fit_seconds,
            base_inference_seconds,
            {
                **base_fit_info,
                "evaluation_origins": test_windows.origins.tolist(),
                "raw_input_contract": True,
            },
            base_model.parameter_count(),
            base_model.trainable_parameter_count(),
            _peak_gpu_memory_mb(device),
        )
        del base_model
        if bool(peft_config.get("source_adapter", {}).get("enabled", True)):
            _reset_gpu_memory(device)
            source_model = TimesFMTransformersModel(
                model_config, context_length, horizon, device, use_lora=True
            )
            fit_start = time.perf_counter()
            fit_info = source_model.fit(
                train_windows.context,
                train_windows.target,
                None,
                None,
                seed,
            )
            fit_seconds = time.perf_counter() - fit_start
            adapter_dir = output_dir / "adapters" / f"source_seed{seed}"
            source_model.save_adapter(adapter_dir)
            inference_start = time.perf_counter()
            prediction = source_model.predict(test_windows.context)
            inference_seconds = time.perf_counter() - inference_start
            _write_variant_outputs(
                output_dir,
                result_rows,
                detail_frames,
                "source_peft",
                "none_raw",
                0.0,
                seed,
                target_raw,
                prediction,
                mase_scale,
                source_scaler.scale,
                dataset.feature_columns,
                test_windows.entity_ids,
                test_windows.origins,
                (),
                0,
                fit_seconds,
                inference_seconds,
                {
                    **fit_info,
                    "adaptation_scope": "source_model_train_entities",
                    "evaluation_origins": test_windows.origins.tolist(),
                    "adapter_dir": str(adapter_dir.relative_to(output_dir)),
                },
                source_model.parameter_count(),
                source_model.trainable_parameter_count(),
                _peak_gpu_memory_mb(device),
            )
            del source_model
        if bool(peft_config.get("target_adapter", {}).get("enabled", True)):
            for budget in budgets:
                support_entities = tuple(budget["support_entities"])
                support_windows = support_windows_by_label[budget["label"]]
                if not support_entities or support_windows is None or len(support_windows) == 0:
                    continue
                _reset_gpu_memory(device)
                target_model = TimesFMTransformersModel(
                    model_config, context_length, horizon, device, use_lora=True
                )
                fit_start = time.perf_counter()
                fit_info = target_model.fit(
                    support_windows.context,
                    support_windows.target,
                    None,
                    None,
                    seed,
                )
                fit_seconds = time.perf_counter() - fit_start
                adapter_dir = (
                    output_dir / "adapters" / f"target_support_{budget['label']}_seed{seed}"
                )
                target_model.save_adapter(adapter_dir)
                inference_start = time.perf_counter()
                prediction = target_model.predict(test_windows.context)
                inference_seconds = time.perf_counter() - inference_start
                _write_variant_outputs(
                    output_dir,
                    result_rows,
                    detail_frames,
                    "target_peft",
                    "none_raw",
                    budget["fraction"],
                    seed,
                    target_raw,
                    prediction,
                    mase_scale,
                    source_scaler.scale,
                    dataset.feature_columns,
                    test_windows.entity_ids,
                    test_windows.origins,
                    support_entities,
                    len(support_windows),
                    fit_seconds,
                    inference_seconds,
                    {
                        **fit_info,
                        "adaptation_scope": "target_support_observed_prefix",
                        "support_entities": list(support_entities),
                        "evaluation_origins": test_windows.origins.tolist(),
                        "adapter_dir": str(adapter_dir.relative_to(output_dir)),
                    },
                    target_model.parameter_count(),
                    target_model.trainable_parameter_count(),
                    _peak_gpu_memory_mb(device),
                )
                del target_model

    pd.DataFrame(result_rows).to_csv(output_dir / "result_table.csv", index=False)
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
            "input_contract": "raw_values_with_internal_instance_normalization",
            "window_shapes": {
                "train_context": list(train_windows.context.shape),
                "test_context": list(test_windows.context.shape),
                "all_support_windows": list(all_support_windows.context.shape),
            },
            "no_final_target_used_for_adaptation": True,
        },
    )
    return output_dir
