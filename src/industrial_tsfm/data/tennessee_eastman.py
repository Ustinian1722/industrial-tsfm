from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cmapss import EntitySplits

DISTURBANCES = tuple(range(1, 16))
MEASURED_COLUMNS = tuple(f"xmeas_{index:03d}" for index in range(1, 42))
EXTRA_OUTPUT_COLUMNS = tuple(f"derived_{index:03d}" for index in range(42, 52))
ALL_OUTPUT_COLUMNS = MEASURED_COLUMNS + EXTRA_OUTPUT_COLUMNS


@dataclass
class TennesseeEastmanDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    splits: EntitySplits
    disturbances: tuple[int, ...]
    sampling_minutes: float
    feature_set: str

    def summary(self) -> dict[str, object]:
        return {
            "dataset": "tennessee_eastman_challenge_idv_archive",
            "feature_set": self.feature_set,
            "feature_columns": list(self.feature_columns),
            "n_features": len(self.feature_columns),
            "rows": len(self.frame),
            "entities": int(self.frame["entity_id"].nunique()),
            "disturbances": list(self.disturbances),
            "sampling_minutes": self.sampling_minutes,
            "duration_hours": 50.0,
            "splits": self.splits.as_dict(),
            "task_boundary": (
                "forecast measured process outputs; disturbance ID is the entity; "
                "no fault label or future target is used for fitting"
            ),
        }


def _read_dat(path: Path, expected_columns: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Tennessee Eastman member {path}. "
            "Run scripts/fetch_tennessee_eastman.py first."
        )
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise ValueError(
            f"Unexpected Tennessee Eastman shape in {path}: {values.shape}; "
            f"expected [time, {expected_columns}]"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Tennessee Eastman file contains non-finite values: {path}")
    return values


def load_tennessee_eastman(
    raw_dir: str | Path,
    train_disturbances: tuple[int, ...] = tuple(range(1, 9)),
    validation_disturbances: tuple[int, ...] = (9, 10),
    test_disturbances: tuple[int, ...] = tuple(range(11, 16)),
    feature_set: str = "measured",
) -> TennesseeEastmanDataset:
    raw_dir = Path(raw_dir)
    train_disturbances = tuple(int(value) for value in train_disturbances)
    validation_disturbances = tuple(int(value) for value in validation_disturbances)
    test_disturbances = tuple(int(value) for value in test_disturbances)
    all_split_ids = train_disturbances + validation_disturbances + test_disturbances
    if set(all_split_ids) != set(DISTURBANCES) or len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError(
            f"Tennessee Eastman splits must partition disturbances {DISTURBANCES} exactly"
        )
    feature_set = str(feature_set).lower()
    if feature_set == "measured":
        feature_columns = MEASURED_COLUMNS
        selected = slice(0, 41)
    elif feature_set in {"all_outputs", "all"}:
        feature_columns = ALL_OUTPUT_COLUMNS
        selected = slice(0, 51)
    else:
        raise ValueError("feature_set must be 'measured' or 'all_outputs'")

    frames: list[pd.DataFrame] = []
    sampling_minutes: float | None = None
    for disturbance in all_split_ids:
        entity_id = f"te_idv_{disturbance:02d}"
        entity_dir = raw_dir / f"idv{disturbance}"
        outputs = _read_dat(entity_dir / "y.dat", expected_columns=51)
        time_values = _read_dat(entity_dir / "t.dat", expected_columns=3)[:, 0]
        if len(outputs) != len(time_values) or len(outputs) < 2:
            raise ValueError(f"Tennessee Eastman y/t length mismatch for {entity_id}")
        time_delta = np.diff(time_values)
        if np.any(time_delta <= 0):
            raise ValueError(f"Tennessee Eastman time values are not increasing for {entity_id}")
        entity_sampling_minutes = float(np.round(np.median(time_delta) * 60.0, 3))
        if sampling_minutes is None:
            sampling_minutes = entity_sampling_minutes
        elif not np.isclose(sampling_minutes, entity_sampling_minutes, rtol=0.0, atol=1.0e-3):
            raise ValueError("Tennessee Eastman disturbances have inconsistent sampling intervals")
        selected_values = outputs[:, selected].astype(np.float64)
        frame = pd.DataFrame(selected_values, columns=feature_columns)
        frame["cycle"] = np.arange(1, len(frame) + 1, dtype=np.int64)
        frame["time_hours"] = time_values
        frame["disturbance_id"] = disturbance
        frame["source"] = "tennessee_eastman"
        frame["entity_id"] = entity_id
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["entity_id", "cycle"], kind="stable"
    )
    splits = EntitySplits(
        tuple(f"te_idv_{value:02d}" for value in train_disturbances),
        tuple(f"te_idv_{value:02d}" for value in validation_disturbances),
        tuple(f"te_idv_{value:02d}" for value in test_disturbances),
    )
    splits.assert_disjoint()
    return TennesseeEastmanDataset(
        combined.reset_index(drop=True),
        feature_columns,
        splits,
        DISTURBANCES,
        float(sampling_minutes),
        feature_set,
    )
