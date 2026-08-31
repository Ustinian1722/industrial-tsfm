from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_tsfm.config import load_config
from industrial_tsfm.data import (
    TrainOnlyStandardScaler,
    load_cmapss,
    make_windows,
    observed_prefix_before_origins,
)
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
    raw_dir = project_root / dataset_config.get("raw_dir", "data/raw")
    dataset = load_cmapss(
        raw_dir,
        subset=dataset_config["subset"],
        include_settings=dataset_config["include_settings"],
        include_sensors=dataset_config["include_sensors"],
        train_fraction=dataset_config["train_fraction"],
        validation_fraction=dataset_config["validation_fraction"],
        entity_split_seed=dataset_config["entity_split_seed"],
    )
    expected_hashes = {
        name: _sha256(raw_dir / name)
        for name in (
            f"train_{dataset.subset}.txt",
            f"test_{dataset.subset}.txt",
            f"RUL_{dataset.subset}.txt",
        )
    }
    if manifest["raw_file_sha256"] != expected_hashes:
        raise AssertionError("Raw-file hashes do not match the run manifest")

    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    test_windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    support_manifest = json.loads((run_dir / "support_manifest.json").read_text(encoding="utf-8"))
    expected_origins = dict(
        zip(test_windows.entity_ids, test_windows.origins.tolist(), strict=True)
    )
    if support_manifest["evaluation_origins_by_entity"] != expected_origins:
        raise AssertionError("Evaluation origins do not match the recomputed final windows")
    observed_prefix = observed_prefix_before_origins(
        dataset.test_frame, test_windows.entity_ids, test_windows.origins
    )
    for entity, frame in observed_prefix.groupby("entity_id", sort=True):
        if not (frame["cycle"].to_numpy() < expected_origins[entity]).all():
            raise AssertionError(f"Observed prefix crosses the final target origin for {entity}")
    if set(support_manifest["support_candidate_entities"]) & set(
        support_manifest["support_ineligible_entities"]
    ):
        raise AssertionError("Support candidate and ineligible pools overlap")
    if set(support_manifest["support_candidate_entities"]) | set(
        support_manifest["support_ineligible_entities"]
    ) != set(test_windows.entity_ids):
        raise AssertionError("Support pools do not partition the evaluation entities")

    source_scaler_manifest = json.loads(
        (run_dir / "source_scaler.json").read_text(encoding="utf-8")
    )
    source_scaler = TrainOnlyStandardScaler.fit(
        dataset.train_frame,
        dataset.splits.train,
        dataset.feature_columns,
        epsilon=float(config["preprocessing"]["epsilon"]),
    )
    _assert_close(np.asarray(source_scaler_manifest["mean"]), source_scaler.mean, "source mean")
    _assert_close(np.asarray(source_scaler_manifest["scale"]), source_scaler.scale, "source scale")
    mase_scale = np.asarray(source_scaler_manifest["mase_scale"], dtype=np.float64)
    normalized_scale = np.asarray(source_scaler_manifest["scale"], dtype=np.float64)
    _assert_close(mase_scale, source_scaler.mase_scale(dataset.train_frame), "MASE scale")

    budgets = support_manifest["budgets"]
    previous_support: set[str] = set()
    budget_by_label: dict[str, dict[str, object]] = {}
    for budget in budgets:
        label = str(budget["label"])
        support_entities = set(budget["support_entities"])
        if not support_entities.issubset(set(support_manifest["support_candidate_entities"])):
            raise AssertionError(f"Support budget {label} includes an ineligible entity")
        if not previous_support.issubset(support_entities):
            raise AssertionError("Support budgets are not nested")
        previous_support = support_entities
        scaler_manifest = json.loads(
            (run_dir / "scalers" / f"support_{label}.json").read_text(encoding="utf-8")
        )
        if not support_entities:
            expected_scaler = source_scaler
        else:
            expected_scaler = TrainOnlyStandardScaler.fit(
                observed_prefix,
                sorted(support_entities),
                dataset.feature_columns,
                epsilon=float(config["preprocessing"]["epsilon"]),
            )
        if set(scaler_manifest["fit_entities"]) != set(expected_scaler.fit_entities):
            raise AssertionError(f"Scaler entity scope mismatch for support budget {label}")
        if scaler_manifest["fit_rows"] != expected_scaler.fit_rows:
            raise AssertionError(f"Scaler row count mismatch for support budget {label}")
        _assert_close(np.asarray(scaler_manifest["mean"]), expected_scaler.mean, f"mean {label}")
        _assert_close(np.asarray(scaler_manifest["scale"]), expected_scaler.scale, f"scale {label}")
        budget_by_label[label] = budget

    result_table = pd.read_csv(run_dir / "result_table.csv")
    if (result_table["status"] != "complete").any():
        raise AssertionError("Phase-2 result table contains a non-complete row")
    expected_training_modes = {
        "naive": "none",
        "patchtst": "supervised",
        "timesfm": "zero_shot",
        "chronos": "zero_shot",
        "chronos2": "zero_shot",
        "moirai2": "zero_shot",
    }
    for _, row in result_table.iterrows():
        if row["training_mode"] != expected_training_modes[row["model"]]:
            raise AssertionError(f"Unexpected training mode for {row['model']}")

    all_rows = result_table[result_table["evaluation_scope"] == "all_test"]
    if len(all_rows) != len(result_table[result_table["evaluation_scope"] == "all_test"]):
        raise AssertionError("Invalid evaluation scope")
    if len(all_rows) != len(list((run_dir / "predictions").glob("*.npz"))):
        raise AssertionError("Every prediction artifact must have one all-test result row")
    strategy_path = run_dir / "strategy_recommendations.json"
    if strategy_path.exists():
        strategy_manifest = json.loads(strategy_path.read_text(encoding="utf-8"))
        if strategy_manifest.get("uses_target_labels"):
            raise AssertionError("Strategy recommendations must not use final target labels")
        if not strategy_manifest.get("recommendations"):
            raise AssertionError("Strategy recommendation manifest is empty")
    target_raw = test_windows.target.astype(np.float32)
    for _, row in all_rows.iterrows():
        fit_info = json.loads(row["fit_info"])
        variant = fit_info["variant"]
        prediction_path = run_dir / "predictions" / f"{variant}_seed{int(row['seed'])}.npz"
        if not prediction_path.exists():
            raise AssertionError(f"Missing prediction artifact: {prediction_path.name}")
        with np.load(prediction_path, allow_pickle=False) as artifact:
            y_true = artifact["y_true"]
            y_pred = artifact["y_pred"]
            entity_ids = tuple(artifact["entity_ids"].astype(str).tolist())
            origins = artifact["origins"].astype(int).tolist()
        if y_true.shape != target_raw.shape or y_pred.shape != target_raw.shape:
            raise AssertionError(f"Prediction shape mismatch for {variant}")
        if entity_ids != test_windows.entity_ids or origins != test_windows.origins.tolist():
            raise AssertionError(f"Entity/origin mismatch for {variant}")
        _assert_close(y_true, target_raw, f"targets {variant}")
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
        label = _support_label(float(row["support_fraction"]))
        budget = budget_by_label[label]
        if int(row["n_support_entities"]) != int(budget["n_support_entities"]):
            raise AssertionError(f"Support entity count mismatch for {variant}")
        if int(row["n_support_windows"]) != int(budget["n_support_windows"]):
            raise AssertionError(f"Support window count mismatch for {variant}")
        if row["adaptation"] == "target_residual_calibration":
            calibrator = json.loads(
                (
                    run_dir
                    / "calibrators"
                    / f"{row['model']}_support_{label}_seed{int(row['seed'])}.json"
                ).read_text(encoding="utf-8")
            )
            if not np.isfinite(np.asarray(calibrator["bias_raw"], dtype=np.float64)).all():
                raise AssertionError(f"Non-finite calibrator for {variant}")
            if set(calibrator["support_entities"]) != set(budget["support_entities"]):
                raise AssertionError(f"Calibrator support mismatch for {variant}")
    print(f"PHASE2_AUDIT_OK {run_dir}")


def _support_label(fraction: float) -> str:
    return f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a Phase-2 adaptation run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    audit_run(args.run_dir)


if __name__ == "__main__":
    main()
