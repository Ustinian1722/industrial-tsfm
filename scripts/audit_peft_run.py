from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_tsfm.config import load_config
from industrial_tsfm.data import load_cmapss, make_windows, observed_prefix_before_origins
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


def _support_label(fraction: float) -> str:
    return f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def audit_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.resolved.yaml")
    project_root = run_dir.parent.parent
    dataset_config = config["dataset"]
    context_length = int(dataset_config["context_length"])
    horizon = int(dataset_config["horizon"])
    if context_length % 32 != 0:
        raise AssertionError("PEFT run does not use a context length divisible by 32")
    dataset = load_cmapss(
        project_root / dataset_config.get("raw_dir", "data/raw"),
        subset=dataset_config["subset"],
        include_settings=dataset_config["include_settings"],
        include_sensors=dataset_config["include_sensors"],
        train_fraction=dataset_config["train_fraction"],
        validation_fraction=dataset_config["validation_fraction"],
        entity_split_seed=dataset_config["entity_split_seed"],
    )
    raw_dir = project_root / dataset_config.get("raw_dir", "data/raw")
    expected_hashes = {
        name: _sha256(raw_dir / name)
        for name in (
            f"train_{dataset.subset}.txt",
            f"test_{dataset.subset}.txt",
            f"RUL_{dataset.subset}.txt",
        )
    }
    if manifest["raw_file_sha256"] != expected_hashes:
        raise AssertionError("Raw-file hashes do not match the PEFT run manifest")
    test_windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length,
        horizon,
        last_only=True,
    )
    support_manifest = json.loads((run_dir / "support_manifest.json").read_text(encoding="utf-8"))
    if list(test_windows.skipped_entities) != support_manifest["evaluation_skipped_entities"]:
        raise AssertionError("Evaluation skipped-entity list changed")
    expected_origins = dict(
        zip(test_windows.entity_ids, test_windows.origins.tolist(), strict=True)
    )
    if support_manifest["evaluation_origins_by_entity"] != expected_origins:
        raise AssertionError("Evaluation origins do not match recomputed PEFT windows")
    observed_prefix = observed_prefix_before_origins(
        dataset.test_frame, test_windows.entity_ids, test_windows.origins
    )
    for entity, frame in observed_prefix.groupby("entity_id", sort=True):
        if not (frame["cycle"].to_numpy() < expected_origins[entity]).all():
            raise AssertionError(f"Target support crosses the final origin for {entity}")
    candidate = set(support_manifest["support_candidate_entities"])
    ineligible = set(support_manifest["support_ineligible_entities"])
    evaluation_entities = set(test_windows.entity_ids)
    if candidate & ineligible or candidate | ineligible != evaluation_entities:
        raise AssertionError("PEFT support pools do not partition evaluation entities")

    budgets = support_manifest["budgets"]
    previous_support: set[str] = set()
    budget_by_label: dict[str, dict[str, object]] = {}
    for budget in budgets:
        label = str(budget["label"])
        support_entities = set(budget["support_entities"])
        if not support_entities.issubset(candidate):
            raise AssertionError(f"PEFT budget {label} includes an ineligible entity")
        if not previous_support.issubset(support_entities):
            raise AssertionError("PEFT support budgets are not nested")
        previous_support = support_entities
        support_windows = (
            make_windows(
                observed_prefix,
                sorted(support_entities),
                dataset.feature_columns,
                context_length,
                horizon,
                stride=int(config["peft"]["support_window_stride"]),
                last_only=False,
            )
            if support_entities
            else None
        )
        expected_window_count = 0 if support_windows is None else len(support_windows)
        if int(budget["n_support_windows"]) != expected_window_count:
            raise AssertionError(f"Support window count mismatch for PEFT budget {label}")
        budget_by_label[label] = budget

    source_scaler = json.loads((run_dir / "source_scaler.json").read_text(encoding="utf-8"))
    mase_scale = np.asarray(source_scaler["mase_scale"], dtype=np.float64)
    normalized_scale = np.asarray(source_scaler["scale"], dtype=np.float64)
    target_raw = test_windows.target.astype(np.float32)
    result_table = pd.read_csv(run_dir / "result_table.csv")
    if (result_table["status"] != "complete").any():
        raise AssertionError("PEFT result table contains a non-complete row")
    prediction_files = list((run_dir / "predictions").glob("*.npz"))
    all_rows = result_table[result_table["evaluation_scope"] == "all_test"]
    if len(all_rows) != len(prediction_files):
        raise AssertionError("Every PEFT prediction artifact must have one all-test row")
    for _, row in all_rows.iterrows():
        fit_info = json.loads(row["fit_info"])
        variant = fit_info["variant"]
        prediction_path = run_dir / "predictions" / f"{variant}_seed{int(row['seed'])}.npz"
        if not prediction_path.exists():
            raise AssertionError(f"Missing PEFT prediction artifact {prediction_path.name}")
        with np.load(prediction_path, allow_pickle=False) as artifact:
            y_true = artifact["y_true"]
            y_pred = artifact["y_pred"]
            entity_ids = tuple(artifact["entity_ids"].astype(str).tolist())
            origins = artifact["origins"].astype(int).tolist()
        if y_true.shape != target_raw.shape or y_pred.shape != target_raw.shape:
            raise AssertionError(f"Prediction shape mismatch for {variant}")
        _assert_close(y_true, target_raw, f"targets {variant}")
        if entity_ids != test_windows.entity_ids or origins != test_windows.origins.tolist():
            raise AssertionError(f"Entity/origin mismatch for {variant}")
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
        if row["training_mode"] == "zero_shot":
            if row["adaptation"] != "base_zero_shot" or float(row["support_fraction"]) != 0:
                raise AssertionError("Invalid base PEFT row")
        else:
            if int(row["trainable_parameters"]) <= 0:
                raise AssertionError(f"PEFT row has no trainable parameters: {variant}")
            adapter_dir = run_dir / fit_info["adapter_dir"]
            if not (adapter_dir / "adapter_config.json").exists():
                raise AssertionError(f"Missing adapter config for {variant}")
            if not (adapter_dir / "adapter_model.safetensors").exists():
                raise AssertionError(f"Missing adapter weights for {variant}")
            if row["adaptation"] == "target_peft":
                label = _support_label(float(row["support_fraction"]))
                budget = budget_by_label[label]
                if set(fit_info["support_entities"]) != set(budget["support_entities"]):
                    raise AssertionError(f"Target PEFT support mismatch for {variant}")
                if int(row["n_support_windows"]) != int(budget["n_support_windows"]):
                    raise AssertionError(f"Target PEFT window mismatch for {variant}")
    print(f"PEFT_AUDIT_OK {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a TimesFM Transformers+PEFT run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    audit_run(args.run_dir)


if __name__ == "__main__":
    main()
