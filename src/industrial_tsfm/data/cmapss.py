from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass(frozen=True)
class EntitySplits:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def assert_disjoint(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                overlap = left & right
                if overlap:
                    raise AssertionError(f"Entity leakage detected: {sorted(overlap)}")


@dataclass
class CMAPSSDataset:
    subset: str
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    rul: np.ndarray
    feature_columns: tuple[str, ...]
    splits: EntitySplits

    @property
    def train_entity_frame(self) -> pd.DataFrame:
        return self.train_frame[self.train_frame["entity_id"].isin(self.splits.train)]

    @property
    def validation_entity_frame(self) -> pd.DataFrame:
        return self.train_frame[self.train_frame["entity_id"].isin(self.splits.validation)]

    @property
    def test_entity_frame(self) -> pd.DataFrame:
        return self.test_frame[self.test_frame["entity_id"].isin(self.splits.test)]

    def summary(self) -> dict[str, object]:
        return {
            "subset": self.subset,
            "feature_columns": list(self.feature_columns),
            "n_features": len(self.feature_columns),
            "train_rows": len(self.train_frame),
            "test_rows": len(self.test_frame),
            "train_entities": int(self.train_frame["entity_id"].nunique()),
            "validation_entities": int(self.validation_entity_frame["entity_id"].nunique()),
            "official_test_entities": int(self.test_frame["entity_id"].nunique()),
            "rul_values": int(self.rul.shape[0]),
            "split_entities": self.splits.as_dict(),
        }


def _read_sequence(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=ALL_COLUMNS,
        usecols=range(len(ALL_COLUMNS)),
        engine="python",
    )
    if frame.empty:
        raise ValueError(f"C-MAPSS file is empty: {path}")
    if frame.isna().any().any():
        raise ValueError(f"C-MAPSS file contains missing values: {path}")
    numeric = frame[ALL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"C-MAPSS file contains non-finite/non-numeric values: {path}")
    frame = numeric.astype({"unit_id": "int64", "cycle": "int64"})
    frame["source"] = source
    frame["entity_id"] = frame["unit_id"].map(lambda value: f"{source}_unit_{value:03d}")
    frame = frame.sort_values(["entity_id", "cycle"], kind="stable").reset_index(drop=True)
    for entity_id, entity_frame in frame.groupby("entity_id", sort=False):
        cycles = entity_frame["cycle"].to_numpy()
        if len(cycles) < 2 or np.any(np.diff(cycles) <= 0):
            raise ValueError(f"Cycles must be strictly increasing for {entity_id}")
    return frame


def _read_rul(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS RUL file: {path}")
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"RUL file contains no finite values: {path}")
    return values


def _feature_columns(include_settings: bool, include_sensors: bool) -> tuple[str, ...]:
    columns: list[str] = []
    if include_settings:
        columns.extend(SETTING_COLUMNS)
    if include_sensors:
        columns.extend(SENSOR_COLUMNS)
    if not columns:
        raise ValueError("At least one of include_settings/include_sensors must be true")
    return tuple(columns)


def _split_train_entities(
    train_entities: Iterable[str],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train_fraction and validation_fraction must be positive")
    if not np.isclose(train_fraction + validation_fraction, 1.0):
        raise ValueError("train_fraction + validation_fraction must equal 1")
    entities = np.array(sorted(set(train_entities)), dtype=object)
    if len(entities) < 2:
        raise ValueError("At least two training entities are required for train/validation split")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(entities))
    n_train = round(len(entities) * train_fraction)
    n_train = min(max(n_train, 1), len(entities) - 1)
    train = tuple(sorted(entities[permutation[:n_train]].tolist()))
    validation = tuple(sorted(entities[permutation[n_train:]].tolist()))
    return train, validation


def load_cmapss(
    raw_dir: str | Path,
    subset: str = "FD001",
    include_settings: bool = True,
    include_sensors: bool = True,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    entity_split_seed: int = 2026,
) -> CMAPSSDataset:
    subset = subset.upper()
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
    raw_dir = Path(raw_dir)
    train_frame = _read_sequence(raw_dir / f"train_{subset}.txt", "train")
    test_frame = _read_sequence(raw_dir / f"test_{subset}.txt", "test")
    rul = _read_rul(raw_dir / f"RUL_{subset}.txt")
    n_test_entities = test_frame["entity_id"].nunique()
    if len(rul) != n_test_entities:
        raise ValueError(
            f"RUL count ({len(rul)}) does not match official test entities ({n_test_entities})"
        )
    train_entities, validation_entities = _split_train_entities(
        train_frame["entity_id"].unique(), train_fraction, validation_fraction, entity_split_seed
    )
    test_entities = tuple(sorted(test_frame["entity_id"].unique().tolist()))
    splits = EntitySplits(train_entities, validation_entities, test_entities)
    splits.assert_disjoint()
    return CMAPSSDataset(
        subset=subset,
        train_frame=train_frame,
        test_frame=test_frame,
        rul=rul,
        feature_columns=_feature_columns(include_settings, include_sensors),
        splits=splits,
    )
