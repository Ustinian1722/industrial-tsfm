from __future__ import annotations

from pathlib import Path

import numpy as np

from industrial_tsfm.data import load_tennessee_eastman


def _write_te_fixture(root: Path) -> None:
    for disturbance in range(1, 16):
        entity_dir = root / f"idv{disturbance}"
        entity_dir.mkdir(parents=True)
        y = np.arange(12 * 51, dtype=np.float64).reshape(12, 51) + disturbance
        t = np.column_stack(
            [np.arange(12, dtype=np.float64) / 6.0, np.ones(12), -np.ones(12)]
        )
        np.savetxt(entity_dir / "y.dat", y)
        np.savetxt(entity_dir / "t.dat", t)


def test_tennessee_eastman_loader_preserves_entity_safe_splits(tmp_path: Path) -> None:
    _write_te_fixture(tmp_path)
    dataset = load_tennessee_eastman(
        tmp_path,
        train_disturbances=tuple(range(1, 9)),
        validation_disturbances=(9, 10),
        test_disturbances=tuple(range(11, 16)),
    )
    assert dataset.frame.shape == (15 * 12, 46)
    assert len(dataset.feature_columns) == 41
    assert dataset.splits.train == tuple(f"te_idv_{i:02d}" for i in range(1, 9))
    assert dataset.splits.validation == ("te_idv_09", "te_idv_10")
    assert dataset.splits.test == tuple(f"te_idv_{i:02d}" for i in range(11, 16))
    assert dataset.sampling_minutes == 10.0
    assert dataset.frame.groupby("entity_id").size().eq(12).all()
