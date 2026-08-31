from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from industrial_tsfm.config import load_config, write_json
from industrial_tsfm.data import (
    TrainOnlyStandardScaler,
    load_cmapss,
    load_cross_domain_dataset,
    make_windows,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def _ood_bins(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 4:
        return np.full(len(scores), "all", dtype=object)
    ranks = pd.Series(scores).rank(method="first", pct=True).to_numpy()
    return np.select(
        [ranks <= 0.25, ranks <= 0.50, ranks <= 0.75],
        ["q1_low", "q2", "q3"],
        default="q4_high",
    ).astype(object)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rank(left), _rank(right))


def _load_evaluation_context(
    run_dir: Path, config: dict[str, object], result_table: pd.DataFrame
) -> dict[str, object]:
    axis = str(result_table["axis"].iloc[0])
    dataset_config = config["dataset"]
    project_root = run_dir.parent.parent
    if axis == "cross_regime":
        source_subset = str(dataset_config["source_subset"]).upper()
        source = load_cmapss(
            project_root / dataset_config.get("raw_dir", "data/raw"),
            subset=source_subset,
            include_settings=bool(dataset_config["include_settings"]),
            include_sensors=bool(dataset_config["include_sensors"]),
            train_fraction=float(dataset_config["train_fraction"]),
            validation_fraction=float(dataset_config["validation_fraction"]),
            entity_split_seed=int(dataset_config["entity_split_seed"]),
        )
        targets = {
            str(subset).upper(): load_cmapss(
                project_root / dataset_config.get("raw_dir", "data/raw"),
                subset=str(subset).upper(),
                include_settings=bool(dataset_config["include_settings"]),
                include_sensors=bool(dataset_config["include_sensors"]),
                train_fraction=float(dataset_config["train_fraction"]),
                validation_fraction=float(dataset_config["validation_fraction"]),
                entity_split_seed=int(dataset_config["entity_split_seed"]),
            )
            for subset in dataset_config["target_subsets"]
        }
        target_windows = {
            subset: make_windows(
                target.test_frame,
                target.splits.test,
                target.feature_columns,
                int(dataset_config["context_length"]),
                int(dataset_config["horizon"]),
                last_only=True,
            )
            for subset, target in targets.items()
        }
        frame = source.train_frame
        train_entities = source.splits.train
        validation_entities = source.splits.validation
        feature_names = source.feature_columns
    elif axis == "cross_domain":
        source_dataset = str(dataset_config.get("source_dataset", "uci_gas_turbine")).lower()
        default_raw_dir = (
            "data/raw/tennessee_eastman"
            if source_dataset == "tennessee_eastman"
            else "data/raw/gas_turbine"
        )
        raw_dir = project_root / dataset_config.get("raw_dir", default_raw_dir)
        source = load_cross_domain_dataset(source_dataset, raw_dir, dataset_config)
        target_last_only = str(dataset_config.get("evaluation_origins", "stride")).lower() == "last"
        target_windows = {
            str(dataset_config["target_dataset"]): make_windows(
                source.frame,
                source.splits.test,
                source.feature_columns,
                int(dataset_config["context_length"]),
                int(dataset_config["horizon"]),
                stride=int(dataset_config.get("evaluation_window_stride", 168)),
                last_only=target_last_only,
                include_last=bool(dataset_config.get("include_last_evaluation_origin", True)),
            )
        }
        frame = source.frame
        train_entities = source.splits.train
        validation_entities = source.splits.validation
        feature_names = source.feature_columns
    else:
        raise ValueError(f"Analysis currently supports cross_regime/cross_domain runs, not {axis}")
    scaler = TrainOnlyStandardScaler.fit(
        frame,
        train_entities,
        feature_names,
        epsilon=float(config["preprocessing"]["epsilon"]),
    )
    validation_windows = make_windows(
        frame,
        validation_entities,
        feature_names,
        int(dataset_config["context_length"]),
        int(dataset_config["horizon"]),
        stride=int(dataset_config.get("validation_window_stride", 1)),
        last_only=False,
    )
    train_windows = make_windows(
        frame,
        train_entities,
        feature_names,
        int(dataset_config["context_length"]),
        int(dataset_config["horizon"]),
        stride=int(dataset_config.get("train_window_stride", 1)),
        last_only=False,
    )
    return {
        "axis": axis,
        "feature_names": tuple(feature_names),
        "target_windows": target_windows,
        "validation_windows": validation_windows,
        "train_windows": train_windows,
        "scaler": scaler,
        "mase_scale": scaler.mase_scale(frame),
    }


def analyze_run(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "config.resolved.yaml")
    result_table = pd.read_csv(run_dir / "result_table.csv")
    if result_table.empty:
        raise ValueError("Cannot analyze an empty result table")
    context = _load_evaluation_context(run_dir, config, result_table)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    scaler = context["scaler"]
    feature_names = context["feature_names"]
    target_windows = context["target_windows"]
    validation_windows = context["validation_windows"]
    train_windows = context["train_windows"]
    mase_scale = np.asarray(context["mase_scale"], dtype=np.float64)
    normalized_scale = scaler.scale.astype(np.float64)

    source_context_z = scaler.transform(train_windows.context).astype(np.float64)
    source_context_flat = source_context_z.reshape(source_context_z.shape[0], -1)
    source_pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    source_pca.fit(source_context_flat)
    source_projection = source_pca.transform(source_context_flat)
    projection_scale = np.maximum(source_projection.std(axis=0), 1.0e-8)

    validation_naive = np.repeat(
        validation_windows.context[:, -1:, :], validation_windows.target.shape[1], axis=1
    )
    validation_residual = validation_windows.target.astype(np.float64) - validation_naive.astype(
        np.float64
    )
    coverage_nominal = float(config.get("analysis", {}).get("interval_coverage", 0.90))
    if not 0.0 < coverage_nominal < 1.0:
        raise ValueError("analysis.interval_coverage must be between zero and one")
    interval_q = np.quantile(np.abs(validation_residual), coverage_nominal, axis=0, method="higher")

    feature_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    ood_rows: list[dict[str, object]] = []
    ood_feature_rows: list[dict[str, object]] = []
    representation_rows: list[dict[str, object]] = []
    ood_error_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    reliability_feature_rows: list[dict[str, object]] = []
    seen_ood_keys: set[str] = set()

    for _, result in result_table.iterrows():
        fit_info = json.loads(result["fit_info"])
        variant = str(fit_info["variant"])
        target_key = (
            str(result["target_subset"])
            if context["axis"] == "cross_regime"
            else str(result["target_dataset"])
        )
        windows = target_windows[target_key]
        prediction_path = run_dir / "predictions" / f"{variant}_seed{int(result['seed'])}.npz"
        with np.load(prediction_path, allow_pickle=False) as artifact:
            y_true = artifact["y_true"].astype(np.float64)
            y_pred = artifact["y_pred"].astype(np.float64)
            entity_ids = tuple(artifact["entity_ids"].astype(str).tolist())
            origins = artifact["origins"].astype(np.int64)
        if entity_ids != windows.entity_ids or not np.array_equal(origins, windows.origins):
            raise AssertionError(
                f"Prediction metadata does not match reconstructed windows: {variant}"
            )
        if not np.allclose(y_true, windows.target, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                f"Prediction targets do not match reconstructed windows: {variant}"
            )
        window_index = {
            (entity, int(origin)): index
            for index, (entity, origin) in enumerate(
                zip(windows.entity_ids, windows.origins, strict=True)
            )
        }
        indices = [
            window_index[(entity, int(origin))]
            for entity, origin in zip(entity_ids, origins, strict=True)
        ]
        raw_context = windows.context[np.asarray(indices)]
        z_context = scaler.transform(raw_context).astype(np.float64)
        abs_z = np.abs(z_context)
        ood_score = np.sqrt(np.mean(np.square(z_context), axis=(1, 2)))
        ood_mean_abs = np.mean(abs_z, axis=(1, 2))
        ood_max_abs = np.max(abs_z, axis=(1, 2))
        error = y_true - y_pred
        abs_error = np.abs(error)
        squared_error = np.square(error)
        per_window_mae = abs_error.mean(axis=(1, 2))
        ood_bin = _ood_bins(ood_score)
        for feature_index, feature_name in enumerate(feature_names):
            feature_error = error[:, :, feature_index]
            feature_abs = np.abs(feature_error)
            feature_rows.append(
                {
                    "model": result["model"],
                    "target": target_key,
                    "seed": int(result["seed"]),
                    "feature": feature_name,
                    "mae": float(feature_abs.mean()),
                    "rmse": float(np.sqrt(np.square(feature_error).mean())),
                    "signed_bias": float(feature_error.mean()),
                    "source_mase": float((feature_abs / mase_scale[feature_index]).mean()),
                    "normalized_mae": float((feature_abs / normalized_scale[feature_index]).mean()),
                    "normalized_rmse": float(
                        np.sqrt(
                            np.square(feature_error / normalized_scale[feature_index]).mean()
                        )
                    ),
                }
            )
        for horizon_index in range(y_true.shape[1]):
            horizon_error = error[:, horizon_index, :]
            horizon_rows.append(
                {
                    "model": result["model"],
                    "target": target_key,
                    "seed": int(result["seed"]),
                    "horizon": horizon_index + 1,
                    "mae": float(np.abs(horizon_error).mean()),
                    "rmse": float(np.sqrt(np.square(horizon_error).mean())),
                    "signed_bias": float(horizon_error.mean()),
                    "normalized_mae": float(
                        (np.abs(horizon_error) / normalized_scale[None, :]).mean()
                    ),
                    "normalized_rmse": float(
                        np.sqrt(
                            np.square(horizon_error / normalized_scale[None, :]).mean()
                        )
                    ),
                }
            )
        for entity in sorted(set(entity_ids)):
            mask = np.asarray(entity_ids) == entity
            entity_error = error[mask]
            entity_rows.append(
                {
                    "model": result["model"],
                    "target": target_key,
                    "seed": int(result["seed"]),
                    "entity": entity,
                    "n_windows": int(mask.sum()),
                    "mae": float(np.abs(entity_error).mean()),
                    "rmse": float(np.sqrt(np.square(entity_error).mean())),
                    "signed_bias": float(entity_error.mean()),
                    "normalized_mae": float(
                        (np.abs(entity_error) / normalized_scale[None, None, :]).mean()
                    ),
                    "normalized_rmse": float(
                        np.sqrt(
                            np.square(entity_error / normalized_scale[None, None, :]).mean()
                        )
                    ),
                    "mean_ood_score": float(ood_score[mask].mean()),
                    "max_ood_score": float(ood_score[mask].max()),
                }
            )
        if target_key not in seen_ood_keys:
            target_projection = source_pca.transform(z_context.reshape(z_context.shape[0], -1))
            target_radius = np.sqrt(
                np.square(target_projection / projection_scale[None, :]).sum(axis=1)
            )
            for index, (entity, origin) in enumerate(zip(entity_ids, origins, strict=True)):
                representation_rows.append(
                    {
                        "target": target_key,
                        "entity": entity,
                        "origin": int(origin),
                        "pca_1": float(target_projection[index, 0]),
                        "pca_2": float(target_projection[index, 1]),
                        "source_pca_radius": float(target_radius[index]),
                    }
                )
            for feature_index, feature_name in enumerate(feature_names):
                feature_z = z_context[:, :, feature_index]
                ood_feature_rows.append(
                    {
                        "target": target_key,
                        "feature": feature_name,
                        "mean_abs_z": float(np.abs(feature_z).mean()),
                        "rms_z": float(np.sqrt(np.square(feature_z).mean())),
                        "max_abs_z": float(np.abs(feature_z).max()),
                    }
                )
            for index, (entity, origin) in enumerate(zip(entity_ids, origins, strict=True)):
                ood_rows.append(
                    {
                        "target": target_key,
                        "entity": entity,
                        "origin": int(origin),
                        "mean_abs_z": float(ood_mean_abs[index]),
                        "rms_z": float(ood_score[index]),
                        "max_abs_z": float(ood_max_abs[index]),
                    }
                )
            seen_ood_keys.add(target_key)
        for label in sorted(set(ood_bin.tolist())):
            mask = ood_bin == label
            ood_error_rows.append(
                {
                    "model": result["model"],
                    "target": target_key,
                    "seed": int(result["seed"]),
                    "ood_bin": label,
                    "n_windows": int(mask.sum()),
                    "mean_ood_score": float(ood_score[mask].mean()),
                    "mae": float(abs_error[mask].mean()),
                    "rmse": float(np.sqrt(squared_error[mask].mean())),
                }
            )
        lower = y_pred - interval_q[None, :, :]
        upper = y_pred + interval_q[None, :, :]
        inside = (y_true >= lower) & (y_true <= upper)
        reliability_rows.append(
            {
                "model": result["model"],
                "target": target_key,
                "seed": int(result["seed"]),
                "nominal_coverage": coverage_nominal,
                "calibration_method": "source_validation_naive_absolute_residual",
                "calibration_windows": int(validation_windows.context.shape[0]),
                "coverage": float(inside.mean()),
                "lower_miss_rate": float((y_true < lower).mean()),
                "upper_miss_rate": float((y_true > upper).mean()),
                "mean_interval_width": float((upper - lower).mean()),
                "mean_ood_score": float(ood_score.mean()),
                "ood_mae_pearson": _pearson(ood_score, per_window_mae),
                "ood_mae_spearman": _spearman(ood_score, per_window_mae),
            }
        )
        for feature_index, feature_name in enumerate(feature_names):
            feature_inside = inside[:, :, feature_index]
            reliability_feature_rows.append(
                {
                    "model": result["model"],
                    "target": target_key,
                    "seed": int(result["seed"]),
                    "feature": feature_name,
                    "nominal_coverage": coverage_nominal,
                    "coverage": float(feature_inside.mean()),
                    "mean_interval_width": float(
                        (upper[:, :, feature_index] - lower[:, :, feature_index]).mean()
                    ),
                }
            )

    _write_rows(analysis_dir / "feature_error_summary.csv", feature_rows)
    _write_rows(analysis_dir / "horizon_error_summary.csv", horizon_rows)
    _write_rows(analysis_dir / "entity_error_summary.csv", entity_rows)
    _write_rows(analysis_dir / "ood_window_summary.csv", ood_rows)
    _write_rows(analysis_dir / "ood_feature_summary.csv", ood_feature_rows)
    _write_rows(analysis_dir / "context_representation_summary.csv", representation_rows)
    _write_rows(analysis_dir / "error_by_ood_bin.csv", ood_error_rows)
    _write_rows(analysis_dir / "reliability_summary.csv", reliability_rows)
    _write_rows(analysis_dir / "reliability_feature_summary.csv", reliability_feature_rows)
    write_json(
        analysis_dir / "analysis_summary.json",
        {
            "run_dir": str(run_dir),
            "axis": context["axis"],
            "models": sorted(result_table["model"].astype(str).unique().tolist()),
            "targets": sorted(set(target_windows)),
            "features": list(feature_names),
            "validation_windows": int(validation_windows.context.shape[0]),
            "source_train_windows": int(train_windows.context.shape[0]),
            "representation": {
                "method": "source-only PCA on flattened standardized contexts",
                "components": 2,
                "explained_variance_ratio": source_pca.explained_variance_ratio_.tolist(),
            },
            "interval_nominal_coverage": coverage_nominal,
            "interval_calibration": "source-validation naive absolute residuals; no target fitting",
            "artifacts": [
                "feature_error_summary.csv",
                "horizon_error_summary.csv",
                "entity_error_summary.csv",
                "ood_window_summary.csv",
                "ood_feature_summary.csv",
                "context_representation_summary.csv",
                "error_by_ood_bin.csv",
                "reliability_summary.csv",
                "reliability_feature_summary.csv",
            ],
        },
    )
    return analysis_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a cross-regime or cross-domain run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(analyze_run(args.run_dir))


if __name__ == "__main__":
    main()
