from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from industrial_tsfm.cross_domain import run_cross_domain
from industrial_tsfm.data import load_ims_bearings


def _write_ims_fixture(root: Path) -> None:
    for subset, n_channels in (("1st_test", 8), ("2nd_test", 4), ("3rd_test", 4)):
        subset_dir = root / subset
        subset_dir.mkdir(parents=True)
        for cycle in range(6):
            time = np.arange(64, dtype=np.float64)
            columns = []
            for channel in range(n_channels):
                columns.append(
                    np.sin(time / 5.0 + channel * 0.1) + 0.02 * cycle + 0.01 * channel
                )
            np.savetxt(subset_dir / f"2026.01.01.00.00.{cycle:02d}", np.column_stack(columns))


def test_ims_loader_builds_entity_cycle_feature_contract(tmp_path: Path) -> None:
    _write_ims_fixture(tmp_path)
    dataset = load_ims_bearings(tmp_path, file_interval_seconds=600)

    assert dataset.feature_columns == ("rms", "std", "kurtosis", "crest_factor")
    assert dataset.frame.shape == (12 * 6, 10)
    assert dataset.frame.loc[:, dataset.feature_columns].notna().all().all()
    assert np.isfinite(dataset.frame.loc[:, dataset.feature_columns].to_numpy()).all()
    assert dataset.frame.groupby("entity_id").size().eq(6).all()
    assert dataset.splits.train == tuple(
        f"ims_1st_test_bearing_{index:02d}" for index in range(1, 5)
    )
    assert dataset.splits.validation == tuple(
        f"ims_2nd_test_bearing_{index:02d}" for index in range(1, 5)
    )
    assert dataset.splits.test == tuple(
        f"ims_3rd_test_bearing_{index:02d}" for index in range(1, 5)
    )
    assert dataset.summary()["file_interval_seconds"] == 600
    assert dataset.summary()["raw_files"] == 18


def test_ims_loader_accepts_explicit_channel_groups_and_splits(tmp_path: Path) -> None:
    _write_ims_fixture(tmp_path)
    entities = [
        f"ims_1st_test_bearing_{index:02d}" for index in range(1, 5)
    ] + [
        f"ims_2nd_test_bearing_{index:02d}" for index in range(1, 5)
    ] + [
        f"ims_3rd_test_bearing_{index:02d}" for index in range(1, 5)
    ]
    dataset = load_ims_bearings(
        tmp_path,
        config={
            "channel_groups": {
                "1st_test": [[0, 1], [2, 3], [4, 5], [6, 7]],
                "2nd_test": [[0], [1], [2], [3]],
                "3rd_test": [[0], [1], [2], [3]],
            },
            "train_entities": entities[:4],
            "validation_entities": entities[4:8],
            "test_entities": entities[8:],
        },
    )
    assert dataset.channel_groups["1st_test"] == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert dataset.splits.as_dict()["train"] == entities[:4]
    assert dataset.splits.as_dict()["validation"] == entities[4:8]
    assert dataset.splits.as_dict()["test"] == entities[8:]


def test_ims_cross_domain_runner_writes_selection_shift_and_gap_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "ims_bearings"
    _write_ims_fixture(raw_dir)
    config_path = tmp_path / "configs" / "ims.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "ims-test"},
                "dataset": {
                    "source_dataset": "ims_bearings",
                    "target_dataset": "ims_bearings",
                    "raw_dir": "data/raw/ims_bearings",
                    "subsets": ["1st_test", "2nd_test", "3rd_test"],
                    "file_interval_seconds": 600,
                    "context_length": 2,
                    "horizon": 1,
                    "train_window_stride": 1,
                    "evaluation_origins": "last",
                    "evaluation_window_stride": 1,
                    "include_last_evaluation_origin": True,
                },
                "preprocessing": {"epsilon": 1.0e-8},
                "selection": {"metric": "normalized_rmse", "constraints": {}},
                "runtime": {"seeds": [42], "device": "cpu", "output_dir": "results"},
                "models": {"naive": {"enabled": True}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_cross_domain(config_path, ["naive"])
    assert (run_dir / "result_table.csv").exists()
    assert (run_dir / "selection.json").exists()
    assert (run_dir / "shift_summary.json").exists()
    assert (run_dir / "generalization_gap.csv").exists()
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"source_dataset": "ims_bearings"' in manifest
    assert "NASA/IMS raw vibration files" in manifest
