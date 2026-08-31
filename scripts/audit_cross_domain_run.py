from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_tsfm.config import load_config
from industrial_tsfm.cross_domain import raw_file_hashes
from industrial_tsfm.data import TrainOnlyStandardScaler, load_cross_domain_dataset, make_windows
from industrial_tsfm.evaluation import compute_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_close(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if not np.allclose(left, right, rtol=1e-6, atol=1e-6):
        raise AssertionError(f"{name} differs from its recomputed value")


def audit_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.resolved.yaml")
    project_root = run_dir.parent.parent
    dataset_config = config["dataset"]
    source_dataset = str(dataset_config.get("source_dataset", "uci_gas_turbine")).lower()
    default_raw_dir = (
        "data/raw/tennessee_eastman"
        if source_dataset == "tennessee_eastman"
        else "data/raw/ims_bearings"
        if source_dataset == "ims_bearings"
        else "data/raw/gas_turbine"
    )
    raw_dir = project_root / dataset_config.get("raw_dir", default_raw_dir)
    dataset = load_cross_domain_dataset(source_dataset, raw_dir, dataset_config)
    expected_hashes = raw_file_hashes(raw_dir, source_dataset)
    if manifest["raw_file_sha256"] != expected_hashes:
        raise AssertionError("Cross-domain raw-file hashes do not match the run manifest")
    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    last_only = str(dataset_config.get("evaluation_origins", "stride")).lower() == "last"
    target_windows = make_windows(
        dataset.frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        stride=int(dataset_config.get("evaluation_window_stride", 168)),
        last_only=last_only,
        include_last=bool(dataset_config.get("include_last_evaluation_origin", True)),
    )
    target_manifest = json.loads(
        (run_dir / "target_window_manifest.json").read_text(encoding="utf-8")
    )
    if target_manifest["entity_ids"] != list(target_windows.entity_ids):
        raise AssertionError("Target entity IDs changed")
    if target_manifest["origins"] != target_windows.origins.tolist():
        raise AssertionError("Target origins changed")
    if target_manifest["skipped_entities"] != list(target_windows.skipped_entities):
        raise AssertionError("Target skipped entities changed")
    scaler = TrainOnlyStandardScaler.fit(
        dataset.frame,
        dataset.splits.train,
        dataset.feature_columns,
        epsilon=float(config["preprocessing"]["epsilon"]),
    )
    scaler_manifest = json.loads((run_dir / "source_scaler.json").read_text(encoding="utf-8"))
    _assert_close(np.asarray(scaler_manifest["mean"]), scaler.mean, "source mean")
    _assert_close(np.asarray(scaler_manifest["scale"]), scaler.scale, "source scale")
    mase_scale = np.asarray(scaler_manifest["mase_scale"], dtype=np.float64)
    normalized_scale = np.asarray(scaler_manifest["scale"], dtype=np.float64)
    _assert_close(mase_scale, scaler.mase_scale(dataset.frame), "MASE scale")

    result_table = pd.read_csv(run_dir / "result_table.csv")
    if (result_table["status"] != "complete").any():
        raise AssertionError("Cross-domain result table contains a non-complete row")
    expected_modes = {
        "naive": "none",
        "patchtst": "supervised",
        "timesfm": "zero_shot",
        "chronos": "zero_shot",
        "chronos2": "zero_shot",
        "moirai2": "zero_shot",
    }
    prediction_files = list((run_dir / "predictions").glob("*.npz"))
    if len(result_table) != len(prediction_files):
        raise AssertionError("Every cross-domain result row must have one prediction artifact")
    target_raw = target_windows.target.astype(np.float32)
    for _, row in result_table.iterrows():
        if row["axis"] != "cross_domain":
            raise AssertionError("Invalid cross-domain row metadata")
        if row["training_mode"] != expected_modes[row["model"]]:
            raise AssertionError(f"Unexpected training mode for {row['model']}")
        fit_info = json.loads(row["fit_info"])
        variant = fit_info["variant"]
        prediction_path = run_dir / "predictions" / f"{variant}_seed{int(row['seed'])}.npz"
        if not prediction_path.exists():
            raise AssertionError(f"Missing prediction artifact {prediction_path.name}")
        with np.load(prediction_path, allow_pickle=False) as artifact:
            y_true = artifact["y_true"]
            y_pred = artifact["y_pred"]
            entity_ids = tuple(artifact["entity_ids"].astype(str).tolist())
            origins = artifact["origins"].astype(int).tolist()
            feature_names = tuple(artifact["feature_names"].astype(str).tolist())
        if y_true.shape != target_raw.shape or y_pred.shape != y_true.shape:
            raise AssertionError(f"Prediction shape mismatch for {variant}")
        _assert_close(y_true, target_raw, f"targets {variant}")
        if entity_ids != target_windows.entity_ids or origins != target_windows.origins.tolist():
            raise AssertionError(f"Entity/origin mismatch for {variant}")
        if feature_names != dataset.feature_columns:
            raise AssertionError(f"Feature-name mismatch for {variant}")
        if not np.isfinite(y_pred).all():
            raise AssertionError(f"Non-finite predictions for {variant}")
        metrics = compute_metrics(
            y_true, y_pred, mase_scale, entity_ids, normalized_scale=normalized_scale
        )
        metric_names = [
            "mae",
            "rmse",
            "mase",
            "macro_entity_mae",
            "macro_entity_rmse",
        ]
        if "normalized_mae" in result_table.columns:
            metric_names.extend(
                [
                    "normalized_mae",
                    "normalized_rmse",
                    "macro_entity_normalized_mae",
                    "macro_entity_normalized_rmse",
                ]
            )
        for metric_name in metric_names:
            if not np.isclose(float(row[metric_name]), metrics[metric_name], rtol=1e-6, atol=1e-6):
                raise AssertionError(f"Metric mismatch for {variant}: {metric_name}")
        if not fit_info["source_scale_only"]:
            raise AssertionError(f"Target fitting flag changed for {variant}")
    validation_path = run_dir / "validation_result_table.csv"
    selection_path = run_dir / "selection.json"
    gap_path = run_dir / "generalization_gap.csv"
    shift_path = run_dir / "shift_summary.json"
    if validation_path.exists() or selection_path.exists() or gap_path.exists() or shift_path.exists():
        for path in (validation_path, selection_path, gap_path, shift_path):
            if not path.exists():
                raise AssertionError(f"Incomplete automatic-selection artifact set: {path.name}")
        validation_table = pd.read_csv(validation_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selection_split") != "validation":
            raise AssertionError("Automatic model selection must use validation metrics")
        if selection.get("selected_model") not in set(validation_table["model"].astype(str)):
            raise AssertionError("Selected model is absent from validation results")
        gap = pd.read_csv(gap_path)
        if gap.empty or not np.isfinite(gap.select_dtypes(include=[np.number]).to_numpy()).all():
            raise AssertionError("Generalization-gap artifact is empty or non-finite")
        shift = json.loads(shift_path.read_text(encoding="utf-8"))
        if not np.isfinite(
            np.asarray(list(shift["aggregate"].values()), dtype=np.float64)
        ).all():
            raise AssertionError("Shift summary contains non-finite aggregate values")
    engine_path = run_dir / "engine_decision.json"
    if engine_path.exists():
        engine = json.loads(engine_path.read_text(encoding="utf-8"))
        if engine.get("selection_evidence") != "validation_only":
            raise AssertionError("Adaptation engine selection evidence is not validation-only")
        if engine.get("target_labels_used") is not False:
            raise AssertionError("Adaptation engine must not use target labels")
        if engine.get("selected_model") not in set(pd.read_csv(validation_path)["model"].astype(str)):
            raise AssertionError("Adaptation engine selected a model absent from validation evidence")
        allowed_strategies = {
            "zero_shot",
            "target_scaling",
            "target_residual_calibration",
            "few_shot_finetune",
            "target_peft",
        }
        if engine.get("selected_strategy") not in allowed_strategies:
            raise AssertionError("Adaptation engine emitted an unknown strategy")
    print(f"CROSS_DOMAIN_AUDIT_OK {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a registered cross-domain run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    audit_run(args.run_dir)


if __name__ == "__main__":
    main()
