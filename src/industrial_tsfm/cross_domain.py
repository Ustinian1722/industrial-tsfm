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
from .data import TrainOnlyStandardScaler, load_cross_domain_dataset, make_windows
from .evaluation import (
    compute_distribution_shift,
    compute_metrics,
    metrics_detail_rows,
    summarize_tradeoffs,
    write_result_summary,
)
from .models import (
    Chronos2Model,
    ChronosModel,
    Moirai2Model,
    NaiveModel,
    PatchTSTModel,
    TimesFMModel,
)
from .selection import compute_generalization_gap, select_best_model, summarize_generalization_gap

MODEL_NAMES = ("naive", "patchtst", "timesfm", "chronos", "chronos2", "moirai2")


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


def raw_file_hashes(raw_dir: Path, dataset_name: str) -> dict[str, str]:
    """Return the reproducibility hash scope for a registered dataset."""
    dataset_name = str(dataset_name).lower()
    if dataset_name == "uci_gas_turbine":
        paths = [path for path in sorted(raw_dir.glob("gt_*.csv")) if path.is_file()]
    elif dataset_name == "tennessee_eastman":
        paths = [path for path in sorted(raw_dir.rglob("*")) if path.is_file()]
    elif dataset_name == "ims_bearings":
        paths = [
            path
            for path in sorted(raw_dir.rglob("*"))
            if path.is_file()
            and path.suffix.lower() != ".zip"
            and path.name != "source_manifest.json"
        ]
    else:
        raise ValueError(f"Unsupported cross-domain dataset: {dataset_name}")
    if not paths:
        raise FileNotFoundError(f"No raw files found for {dataset_name} under {raw_dir}")
    return {
        (
            path.name
            if dataset_name == "uci_gas_turbine"
            else str(path.relative_to(raw_dir)).replace("\\", "/")
        ):
        _file_sha256(path)
        for path in paths
    }


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
        "timesfm",
        "chronos-forecasting",
        "uni2ts",
        "gluonts",
        "transformers",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


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


def _write_target_result(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    detail_frames: list[pd.DataFrame],
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: np.ndarray,
    normalized_scale: np.ndarray,
    feature_names: tuple[str, ...],
    entity_ids: tuple[str, ...],
    origins: np.ndarray,
    fit_seconds: float,
    inference_seconds: float,
    fit_info: dict[str, Any],
    total_parameters: int | None,
    trainable_parameters: int | None,
    peak_gpu_memory_mb: float | None,
) -> None:
    variant = f"{model_name}_source_{source_dataset}_to_{target_dataset}"
    np.savez_compressed(
        output_dir / "predictions" / f"{variant}_seed{seed}.npz",
        y_true=y_true,
        y_pred=y_pred,
        entity_ids=np.asarray(entity_ids, dtype=str),
        origins=np.asarray(origins, dtype=np.int64),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    metrics = compute_metrics(
        y_true, y_pred, mase_scale, entity_ids, normalized_scale=normalized_scale
    )
    result_rows.append(
        {
            "axis": "cross_domain",
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "model": model_name,
            "seed": seed,
            "status": "complete",
            "training_mode": {
                "naive": "none",
                "patchtst": "supervised",
                "timesfm": "zero_shot",
                "chronos": "zero_shot",
                "chronos2": "zero_shot",
                "moirai2": "zero_shot",
            }[model_name],
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
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "fit_info": json.dumps({**fit_info, "variant": variant}, sort_keys=True, default=str),
        }
    )
    detail = metrics_detail_rows(
        variant,
        seed,
        y_true,
        y_pred,
        mase_scale,
        feature_names,
        entity_ids,
        normalized_scale=normalized_scale,
    )
    detail["axis"] = "cross_domain"
    detail["source_dataset"] = source_dataset
    detail["target_dataset"] = target_dataset
    detail_frames.append(detail)


def run_cross_domain(config_path: str | Path, requested_models: list[str] | None = None) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    runtime = config.get("runtime", {})
    dataset_config = config["dataset"]
    source_dataset = str(dataset_config.get("source_dataset", "uci_gas_turbine")).lower()
    target_dataset = str(dataset_config.get("target_dataset", "uci_gas_turbine")).lower()
    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    device = resolve_device(str(runtime.get("device", "auto")))
    torch.set_num_threads(int(runtime.get("torch_num_threads", 1)))
    default_raw_dir = (
        "data/raw/tennessee_eastman"
        if source_dataset == "tennessee_eastman"
        else "data/raw/ims_bearings"
        if source_dataset == "ims_bearings"
        else "data/raw/gas_turbine"
    )
    raw_dir = project_root / dataset_config.get("raw_dir", default_raw_dir)
    dataset = load_cross_domain_dataset(source_dataset, raw_dir, dataset_config)
    if source_dataset != target_dataset:
        raise ValueError(
            "The cross-domain runner currently evaluates source-to-target shifts within "
            "one registered dataset; source_dataset and target_dataset must match"
        )
    scaler = TrainOnlyStandardScaler.fit(
        dataset.frame,
        dataset.splits.train,
        dataset.feature_columns,
        epsilon=float(config.get("preprocessing", {}).get("epsilon", 1.0e-8)),
    )
    train_windows = make_windows(
        dataset.frame,
        dataset.splits.train,
        dataset.feature_columns,
        context_length,
        horizon,
        stride=int(dataset_config.get("train_window_stride", 1)),
        last_only=False,
    )
    validation_windows = make_windows(
        dataset.frame,
        dataset.splits.validation,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    target_last_only = str(dataset_config.get("evaluation_origins", "stride")).lower() == "last"
    target_windows = make_windows(
        dataset.frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        stride=int(dataset_config.get("evaluation_window_stride", 168)),
        last_only=target_last_only,
        include_last=bool(dataset_config.get("include_last_evaluation_origin", True)),
    )
    train_context = scaler.transform(train_windows.context)
    train_target = scaler.transform(train_windows.target)
    validation_context = scaler.transform(validation_windows.context)
    validation_target = scaler.transform(validation_windows.target)
    target_context = scaler.transform(target_windows.context)
    mase_scale = scaler.mase_scale(dataset.frame)
    shift_summary = compute_distribution_shift(
        dataset.frame,
        dataset.splits.train,
        dataset.splits.test,
        dataset.feature_columns,
        scaler.scale,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{_config_hash(config)}"
    output_dir = project_root / runtime.get("output_dir", "results") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "dataset_summary.json", dataset.summary())
    write_json(output_dir / "split_manifest.json", dataset.splits.as_dict())
    write_json(
        output_dir / "source_scaler.json", {**scaler.as_dict(), "mase_scale": mase_scale.tolist()}
    )
    write_json(output_dir / "shift_summary.json", shift_summary)
    write_json(
        output_dir / "target_window_manifest.json",
        {
            "entity_ids": list(target_windows.entity_ids),
            "origins": target_windows.origins.tolist(),
            "skipped_entities": list(target_windows.skipped_entities),
            "context_shape": list(target_windows.context.shape),
            "target_shape": list(target_windows.target.shape),
        },
    )
    source_manifest_path = raw_dir / "source_manifest.json"
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        write_json(output_dir / "raw_source_manifest.json", source_manifest)
    split_info = {
        "source_entities": list(dataset.splits.train),
        "validation_entities": list(dataset.splits.validation),
        "target_entities": list(dataset.splits.test),
    }
    if source_dataset == "uci_gas_turbine":
        split_info.update(
            {
                "source_years": list(dataset_config.get("train_years", [2011, 2012])),
                "validation_years": list(dataset_config.get("validation_years", [2013])),
                "target_years": list(dataset_config.get("test_years", [2014, 2015])),
            }
        )
        protocol = (
            "UCI Gas Turbine source years 2011/2012, validation 2013, target years 2014/2015; "
            "source-only scaling and no target fitting"
        )
    elif source_dataset == "tennessee_eastman":
        split_info.update(
            {
                "source_disturbances": list(
                    dataset_config.get("train_disturbances", range(1, 9))
                ),
                "validation_disturbances": list(
                    dataset_config.get("validation_disturbances", [9, 10])
                ),
                "target_disturbances": list(
                    dataset_config.get("test_disturbances", range(11, 16))
                ),
            }
        )
        protocol = (
            "Tennessee Eastman IDV1-8 source, IDV9-10 validation, IDV11-15 target; "
            "10-minute outputs, source-only scaling and no target fitting"
        )
    elif source_dataset == "ims_bearings":
        split_info.update(
            {
                "subsets": list(dataset_config.get("subsets", ["1st_test", "2nd_test", "3rd_test"])),
                "file_interval_seconds": int(dataset_config.get("file_interval_seconds", 600)),
                "feature_contract": dataset.feature_contract,
                "channel_groups": dataset.summary().get("channel_groups", {}),
            }
        )
        protocol = (
            "NASA/IMS raw vibration files converted to per-file RMS/std/kurtosis/crest-factor "
            "features; bearing entities split by test condition, source-only scaling and no "
            "target fitting"
        )
    else:
        raise ValueError(f"Unsupported cross-domain dataset: {source_dataset}")
    model_names = requested_models
    if model_names is None:
        model_names = [
            name
            for name in MODEL_NAMES
            if bool(config.get("models", {}).get(name, {}).get("enabled", False))
        ]
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if not model_names:
        raise ValueError("No enabled models were selected")
    result_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    seeds = [int(seed) for seed in runtime.get("seeds", [42])]
    for seed in seeds:
        _set_seed(seed)
        for model_name in model_names:
            _reset_gpu_memory(device)
            model = _make_model(model_name, config, context_length, horizon, device)
            fit_start = time.perf_counter()
            fit_info = model.fit(
                train_context,
                train_target,
                validation_context,
                validation_target,
                seed,
            )
            fit_seconds = time.perf_counter() - fit_start
            base_peak_memory = _peak_gpu_memory_mb(device)
            validation_inference_start = time.perf_counter()
            validation_prediction_scaled = model.predict(validation_context)
            validation_inference_seconds = time.perf_counter() - validation_inference_start
            validation_prediction_raw = scaler.inverse_transform(validation_prediction_scaled)
            validation_metrics = compute_metrics(
                validation_windows.target.astype(np.float32),
                validation_prediction_raw,
                mase_scale,
                validation_windows.entity_ids,
                normalized_scale=scaler.scale,
            )
            validation_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "status": "complete",
                    "training_mode": model.training_mode,
                    "mae": validation_metrics["mae"],
                    "rmse": validation_metrics["rmse"],
                    "mase": validation_metrics["mase"],
                    "normalized_mae": validation_metrics["normalized_mae"],
                    "normalized_rmse": validation_metrics["normalized_rmse"],
                    "macro_entity_mae": validation_metrics["macro_entity_mae"],
                    "macro_entity_rmse": validation_metrics["macro_entity_rmse"],
                    "macro_entity_normalized_mae": validation_metrics[
                        "macro_entity_normalized_mae"
                    ],
                    "macro_entity_normalized_rmse": validation_metrics[
                        "macro_entity_normalized_rmse"
                    ],
                    "n_windows": int(validation_metrics["n_windows"]),
                    "n_entities": int(validation_metrics["n_entities"]),
                    "total_parameters": model.parameter_count(),
                    "fit_seconds": fit_seconds,
                    "inference_seconds": validation_inference_seconds,
                    "fit_info": json.dumps(
                        {
                            "selection_split": "source_validation_entities",
                            "validation_entities": list(validation_windows.entity_ids),
                        },
                        sort_keys=True,
                    ),
                }
            )
            inference_start = time.perf_counter()
            prediction_scaled = model.predict(target_context)
            inference_seconds = time.perf_counter() - inference_start
            prediction_raw = scaler.inverse_transform(prediction_scaled)
            _write_target_result(
                output_dir,
                result_rows,
                detail_frames,
                source_dataset,
                target_dataset,
                model_name,
                seed,
                target_windows.target.astype(np.float32),
                prediction_raw,
                mase_scale,
                scaler.scale,
                dataset.feature_columns,
                target_windows.entity_ids,
                target_windows.origins,
                fit_seconds,
                inference_seconds,
                {
                    **fit_info,
                    **split_info,
                    "target_entities": list(target_windows.entity_ids),
                    "target_skipped_entities": list(target_windows.skipped_entities),
                    "source_scale_only": True,
                },
                model.parameter_count(),
                model.trainable_parameter_count(),
                _peak_gpu_memory_mb(device) or base_peak_memory,
            )
            del model
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(output_dir / "result_table.csv", index=False)
    summarize_tradeoffs(result_table).to_csv(output_dir / "tradeoff_summary.csv", index=False)
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(output_dir / "validation_result_table.csv", index=False)
    selection_config = config.get("selection", {})
    selection_metric = str(selection_config.get("metric", "normalized_rmse"))
    selection = select_best_model(
        validation_table,
        metric=selection_metric,
        constraints=selection_config.get("constraints", {}),
    )
    write_json(output_dir / "selection.json", selection)
    gap_table = compute_generalization_gap(
        validation_table, result_table, metric=selection_metric
    )
    gap_table.to_csv(output_dir / "generalization_gap.csv", index=False)
    summarize_generalization_gap(gap_table).to_csv(
        output_dir / "generalization_gap_summary.csv", index=False
    )
    write_result_summary(
        result_table,
        output_dir / "result_summary.csv",
        ["source_dataset", "target_dataset", "model"],
    )
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            output_dir / "metrics_detail.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "metrics_detail.csv", index=False)
    hashes = raw_file_hashes(raw_dir, source_dataset)
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
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            **split_info,
            "raw_file_sha256": hashes,
            "protocol": protocol,
            "selection": {
                "metric": selection_metric,
                "selected_model": selection["selected_model"],
                "split": "validation_only",
            },
            "shift_summary": shift_summary["aggregate"],
            "source_window_shapes": {
                "train_context": list(train_context.shape),
                "validation_context": list(validation_context.shape),
            },
            "target_window_shape": list(target_context.shape),
        },
    )
    return output_dir
