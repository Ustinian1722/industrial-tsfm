from __future__ import annotations

import argparse
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
from .data import TrainOnlyStandardScaler, load_cmapss, make_windows
from .evaluation import (
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


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _peak_gpu_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))


def _reset_gpu_memory(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))


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


def _make_model(name: str, config: dict[str, Any], context_length: int, horizon: int, device: str):
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


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_experiment(config_path: str | Path, requested_models: list[str] | None = None) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    runtime = config.get("runtime", {})
    dataset_config = config.get("dataset", {})
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
    scaler = TrainOnlyStandardScaler.fit(
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
    train_context = scaler.transform(train_windows.context)
    train_target = scaler.transform(train_windows.target)
    validation_context = scaler.transform(validation_windows.context)
    validation_target = scaler.transform(validation_windows.target)
    test_context = scaler.transform(test_windows.context)
    mase_scale = scaler.mase_scale(dataset.train_frame)
    target_raw = test_windows.target.astype(np.float32)

    enabled = [
        name
        for name in ("naive", "patchtst", "timesfm", "chronos", "chronos2", "moirai2")
        if bool(config.get("models", {}).get(name, {}).get("enabled", False))
    ]
    model_names = requested_models if requested_models is not None else enabled
    unknown = set(model_names) - {
        "naive",
        "patchtst",
        "timesfm",
        "chronos",
        "chronos2",
        "moirai2",
    }
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if not model_names:
        raise ValueError("No enabled models were selected")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{_config_hash(config)}"
    output_dir = project_root / runtime.get("output_dir", "results") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    with (output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    write_json(output_dir / "dataset_summary.json", dataset.summary())
    write_json(output_dir / "split_manifest.json", dataset.splits.as_dict())
    scaler_manifest = scaler.as_dict()
    scaler_manifest["mase_scale"] = mase_scale.tolist()
    write_json(output_dir / "scaler.json", scaler_manifest)
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
                "test_context": list(test_context.shape),
            },
            "skipped_entities": {
                "train": list(train_windows.skipped_entities),
                "validation": list(validation_windows.skipped_entities),
                "test": list(test_windows.skipped_entities),
            },
        },
    )

    result_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    seeds = [int(seed) for seed in runtime.get("seeds", [42])]
    allow_missing = bool(runtime.get("allow_missing_models", False))
    for seed in seeds:
        _set_seed(seed)
        for model_name in model_names:
            model = _make_model(model_name, config, context_length, horizon, device)
            _reset_gpu_memory(device)
            fit_start = time.perf_counter()
            status = "complete"
            error_message = ""
            fit_info: dict[str, Any] = {}
            predictions: np.ndarray | None = None
            try:
                fit_info = model.fit(
                    train_context,
                    train_target,
                    validation_context,
                    validation_target,
                    seed,
                )
                fit_seconds = time.perf_counter() - fit_start
                inference_start = time.perf_counter()
                predictions = model.predict(test_context)
                inference_seconds = time.perf_counter() - inference_start
                predictions_raw = scaler.inverse_transform(predictions)
                validation_inference_start = time.perf_counter()
                validation_predictions = model.predict(validation_context)
                validation_inference_seconds = time.perf_counter() - validation_inference_start
                validation_predictions_raw = scaler.inverse_transform(validation_predictions)
                validation_metrics = compute_metrics(
                    validation_windows.target.astype(np.float32),
                    validation_predictions_raw,
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
                    }
                )
                metrics = compute_metrics(
                    target_raw,
                    predictions_raw,
                    mase_scale,
                    test_windows.entity_ids,
                    normalized_scale=scaler.scale,
                )
                detail_frames.append(
                    metrics_detail_rows(
                        model_name,
                        seed,
                        target_raw,
                        predictions_raw,
                        mase_scale,
                        dataset.feature_columns,
                        test_windows.entity_ids,
                        normalized_scale=scaler.scale,
                    )
                )
                np.savez_compressed(
                    output_dir / "predictions" / f"{model_name}_seed{seed}.npz",
                    y_true=target_raw,
                    y_pred=predictions_raw,
                    entity_ids=np.asarray(test_windows.entity_ids, dtype=str),
                    origins=test_windows.origins,
                    feature_names=np.asarray(dataset.feature_columns, dtype=str),
                )
                result_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "status": status,
                        "training_mode": model.training_mode,
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
                        "total_parameters": model.parameter_count(),
                        "trainable_parameters": model.trainable_parameter_count(),
                        "fit_seconds": fit_seconds,
                        "inference_seconds": inference_seconds,
                        "peak_gpu_memory_mb": _peak_gpu_memory_mb(device),
                        "fit_info": json.dumps(fit_info, sort_keys=True, default=str),
                    }
                )
            except Exception as exc:
                fit_seconds = time.perf_counter() - fit_start
                status = "unavailable_or_failed"
                error_message = f"{type(exc).__name__}: {exc}"
                result_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "status": status,
                        "training_mode": model.training_mode,
                        "error": error_message,
                        "fit_seconds": fit_seconds,
                        "inference_seconds": None,
                        "peak_gpu_memory_mb": _peak_gpu_memory_mb(device),
                    }
                )
                if not allow_missing:
                    write_json(
                        output_dir / "failure.json",
                        {"model": model_name, "seed": seed, "error": error_message},
                    )
                    raise
            finally:
                del model
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(output_dir / "result_table.csv", index=False)
    summarize_tradeoffs(result_table).to_csv(output_dir / "tradeoff_summary.csv", index=False)
    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(output_dir / "validation_result_table.csv", index=False)
    if not validation_table.empty:
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
    write_result_summary(result_table, output_dir / "result_summary.csv", ["model"])
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            output_dir / "metrics_detail.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "metrics_detail.csv", index=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-1 Industrial TSFM experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/cmapss_fd001_forecast.yaml"))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    parser.add_argument("--allow-missing-models", action="store_true")
    args = parser.parse_args()
    if args.allow_missing_models:
        config = load_config(args.config)
        config.setdefault("runtime", {})["allow_missing_models"] = True
        temporary_config = args.config.parent / ".config.runtime.yaml"
        with temporary_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        try:
            output = run_experiment(
                temporary_config, args.models.split(",") if args.models else None
            )
        finally:
            temporary_config.unlink(missing_ok=True)
    else:
        output = run_experiment(args.config, args.models.split(",") if args.models else None)
    print(output)


if __name__ == "__main__":
    main()
