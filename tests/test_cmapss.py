from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from industrial_tsfm.data import (
    TrainOnlyStandardScaler,
    load_cmapss,
    make_windows,
    observed_prefix_before_origins,
)


def _write_cmapss_fixture(root: Path) -> None:
    columns = 26
    train_rows = []
    for unit_id in (1, 2, 3, 4):
        for cycle in range(1, 13):
            train_rows.append([unit_id, cycle, 0.1, 0.2, 0.3, *([unit_id + cycle / 10] * 21)])
    test_rows = []
    for unit_id in (1, 2):
        for cycle in range(1, 13):
            test_rows.append([unit_id, cycle, 0.1, 0.2, 0.3, *([unit_id + cycle / 10] * 21)])
    assert all(len(row) == columns for row in train_rows + test_rows)
    np.savetxt(root / "train_FD001.txt", np.asarray(train_rows), fmt="%.6f")
    np.savetxt(root / "test_FD001.txt", np.asarray(test_rows), fmt="%.6f")
    np.savetxt(root / "RUL_FD001.txt", np.asarray([[3], [4]]), fmt="%.6f")


def test_cmapss_entity_split_and_train_only_scaler(tmp_path: Path) -> None:
    _write_cmapss_fixture(tmp_path)
    dataset = load_cmapss(
        tmp_path, train_fraction=0.75, validation_fraction=0.25, entity_split_seed=7
    )
    dataset.splits.assert_disjoint()
    assert len(dataset.feature_columns) == 24
    assert set(dataset.splits.test) == {"test_unit_001", "test_unit_002"}

    scaler = TrainOnlyStandardScaler.fit(
        dataset.train_frame, dataset.splits.train, dataset.feature_columns
    )
    assert set(scaler.fit_entities) == set(dataset.splits.train)
    assert scaler.fit_rows == len(dataset.train_entity_frame)
    windows = make_windows(
        dataset.train_frame,
        dataset.splits.train,
        dataset.feature_columns,
        context_length=4,
        horizon=2,
    )
    transformed = scaler.transform(windows.context)
    assert transformed.shape == windows.context.shape
    assert np.isfinite(transformed).all()


def test_short_entities_are_reported_not_padded(tmp_path: Path) -> None:
    _write_cmapss_fixture(tmp_path)
    dataset = load_cmapss(
        tmp_path, train_fraction=0.75, validation_fraction=0.25, entity_split_seed=7
    )
    with pytest.raises(ValueError, match="No valid forecast windows"):
        make_windows(
            dataset.test_frame,
            dataset.splits.test,
            dataset.feature_columns,
            context_length=10,
            horizon=4,
            last_only=True,
        )


def test_observed_prefix_excludes_final_target_origin(tmp_path: Path) -> None:
    _write_cmapss_fixture(tmp_path)
    dataset = load_cmapss(
        tmp_path, train_fraction=0.75, validation_fraction=0.25, entity_split_seed=7
    )
    windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length=4,
        horizon=2,
        last_only=True,
    )
    prefix = observed_prefix_before_origins(dataset.test_frame, windows.entity_ids, windows.origins)
    assert len(prefix) == 20
    assert (prefix.groupby("entity_id")["cycle"].max().to_numpy() < windows.origins).all()


def test_stride_windows_can_include_each_entity_final_origin(tmp_path: Path) -> None:
    _write_cmapss_fixture(tmp_path)
    dataset = load_cmapss(
        tmp_path, train_fraction=0.75, validation_fraction=0.25, entity_split_seed=7
    )
    windows = make_windows(
        dataset.test_frame,
        dataset.splits.test,
        dataset.feature_columns,
        context_length=4,
        horizon=2,
        stride=4,
        include_last=True,
    )
    assert windows.origins.tolist() == [5, 9, 11, 5, 9, 11]
