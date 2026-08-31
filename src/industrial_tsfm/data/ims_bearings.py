from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cmapss import EntitySplits

IMS_SUBSETS = ("1st_test", "2nd_test", "3rd_test")
FEATURE_NAMES = ("rms", "std", "kurtosis", "crest_factor")


@dataclass
class IMSBearingsDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    splits: EntitySplits
    subsets: tuple[str, ...]
    file_interval_seconds: int
    feature_contract: str
    channel_groups: dict[str, tuple[tuple[int, ...], ...]]

    def summary(self) -> dict[str, object]:
        return {
            "dataset": "nasa_ims_bearings",
            "feature_columns": list(self.feature_columns),
            "n_features": len(self.feature_columns),
            "rows": len(self.frame),
            "entities": int(self.frame["entity_id"].nunique()),
            "subsets": list(self.subsets),
            "file_interval_seconds": self.file_interval_seconds,
            "feature_contract": self.feature_contract,
            "channel_groups": {
                subset: [list(group) for group in groups]
                for subset, groups in self.channel_groups.items()
            },
            "raw_files": int(self.frame["raw_file"].nunique()),
            "splits": self.splits.as_dict(),
        }


def _normalise_subset(value: str) -> str:
    value = str(value).strip().lower().replace(" ", "_")
    aliases = {"1st": "1st_test", "2nd": "2nd_test", "3rd": "3rd_test"}
    value = aliases.get(value, value)
    if value not in IMS_SUBSETS:
        raise ValueError(f"Unsupported IMS subset: {value}")
    return value


def _find_subset_dir(raw_dir: Path, subset: str) -> Path:
    candidates = [
        raw_dir / subset,
        raw_dir / "IMS" / subset,
        *(path for path in raw_dir.rglob("*") if path.is_dir() and path.name.lower() == subset.lower()),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(f"Missing IMS subset directory for {subset} under {raw_dir}")


def _signal_files(path: Path) -> list[Path]:
    files = []
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if not candidate.is_file() or candidate.name.lower().startswith(("readme", ".")):
            continue
        if candidate.suffix.lower() in {".zip", ".md", ".json", ".yaml", ".yml"}:
            continue
        if candidate.name.lower().startswith(("readme", "info", "license")):
            continue
        files.append(candidate)
    if not files:
        raise FileNotFoundError(f"No IMS signal files found under {path}")
    return files


def _read_signal(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not parse IMS signal file: {path}") from exc
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"IMS signal file must be a 2-D table with at least two rows: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"IMS signal file contains non-finite values: {path}")
    return values


def _default_channel_groups(n_channels: int, subset: str) -> tuple[tuple[int, ...], ...]:
    if subset == "1st_test" and n_channels >= 8:
        return tuple((index, index + 1) for index in range(0, 8, 2))
    if n_channels >= 4:
        return tuple((index,) for index in range(4))
    return tuple((index,) for index in range(n_channels))


def _feature_vector(signal: np.ndarray, channel_indices: tuple[int, ...]) -> list[float]:
    channels = signal[:, channel_indices]
    means = channels.mean(axis=0)
    centered = channels - means
    variance = np.mean(np.square(centered), axis=0)
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(np.square(channels), axis=0))
    kurtosis = np.mean(np.power(centered, 4), axis=0) / np.maximum(variance**2, 1.0e-24)
    crest_factor = np.max(np.abs(channels), axis=0) / np.maximum(rms, 1.0e-12)
    return [
        float(np.mean(values))
        for values in (rms, std, kurtosis, crest_factor)
    ]


def _entity_split(
    entities: tuple[str, ...],
    config: dict[str, Any],
) -> EntitySplits:
    explicit = {
        key: tuple(str(value) for value in config.get(key, []))
        for key in ("train_entities", "validation_entities", "test_entities")
    }
    if any(explicit.values()):
        if set().union(*explicit.values()) != set(entities):
            raise ValueError("Explicit IMS entity splits must cover every extracted bearing exactly once")
        if any(len(set(values)) != len(values) for values in explicit.values()):
            raise ValueError("IMS entity splits cannot contain duplicates")
        splits = EntitySplits(explicit["train_entities"], explicit["validation_entities"], explicit["test_entities"])
        splits.assert_disjoint()
        return splits
    by_subset = {subset: tuple(entity for entity in entities if f"_{subset}_" in entity) for subset in IMS_SUBSETS}
    if all(by_subset.get(subset) for subset in config.get("subsets", IMS_SUBSETS)):
        selected_subsets = tuple(_normalise_subset(value) for value in config.get("subsets", IMS_SUBSETS))
        if len(selected_subsets) >= 3:
            splits = EntitySplits(by_subset[selected_subsets[0]], by_subset[selected_subsets[1]], by_subset[selected_subsets[2]])
            splits.assert_disjoint()
            return splits
    if len(entities) < 3:
        raise ValueError("IMS default split needs at least three bearing entities")
    splits = EntitySplits(entities[:-2], (entities[-2],), (entities[-1],))
    splits.assert_disjoint()
    return splits


def load_ims_bearings(
    raw_dir: str | Path,
    subsets: tuple[str, ...] = IMS_SUBSETS,
    file_interval_seconds: int = 600,
    max_files_per_subset: int | None = None,
    config: dict[str, Any] | None = None,
) -> IMSBearingsDataset:
    """Extract a forecasting-ready IMS feature table without loading the full ZIP into memory.

    The contract uses the first channel of each bearing group and computes
    RMS, standard deviation, kurtosis, and crest factor per raw signal file.
    Channel grouping can be overridden through ``config['channel_groups']``.
    """
    config = dict(config or {})
    subsets = tuple(_normalise_subset(value) for value in subsets)
    if not subsets:
        raise ValueError("At least one IMS subset is required")
    if file_interval_seconds <= 0:
        raise ValueError("file_interval_seconds must be positive")
    rows: list[dict[str, Any]] = []
    feature_columns = list(FEATURE_NAMES)
    channel_groups_by_subset: dict[str, tuple[tuple[int, ...], ...]] = {}
    for subset in subsets:
        subset_dir = _find_subset_dir(Path(raw_dir), subset)
        files = _signal_files(subset_dir)
        if max_files_per_subset is not None:
            if max_files_per_subset <= 0:
                raise ValueError("max_files_per_subset must be positive")
            files = files[:max_files_per_subset]
        first_signal = _read_signal(files[0])
        configured_groups = config.get("channel_groups", {}).get(subset)
        if configured_groups is None:
            channel_groups = _default_channel_groups(first_signal.shape[1], subset)
        else:
            channel_groups = tuple(tuple(int(index) for index in group) for group in configured_groups)
        if not channel_groups or any(not group for group in channel_groups):
            raise ValueError(f"IMS channel_groups for {subset} cannot be empty")
        if any(max(group) >= first_signal.shape[1] or min(group) < 0 for group in channel_groups):
            raise ValueError(f"IMS channel_groups for {subset} exceed signal channel count")
        channel_groups_by_subset[subset] = channel_groups
        for cycle, path in enumerate(files, start=1):
            signal = first_signal if cycle == 1 else _read_signal(path)
            if signal.shape[1] != first_signal.shape[1]:
                raise ValueError(f"IMS channel count changed within {subset}: {path}")
            vectors = [_feature_vector(signal, group) for group in channel_groups]
            for bearing_index, vector in enumerate(vectors, start=1):
                entity_id = f"ims_{subset}_bearing_{bearing_index:02d}"
                rows.append(
                    {
                        **dict(zip(feature_columns, vector, strict=True)),
                        "entity_id": entity_id,
                        "source": "nasa_ims_bearings",
                        "subset": subset,
                        "cycle": cycle,
                        "time_seconds": (cycle - 1) * file_interval_seconds,
                        "raw_file": str(path.relative_to(Path(raw_dir))).replace("\\", "/"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.loc[:, feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("IMS feature extraction produced no finite rows")
    frame = frame.sort_values(["entity_id", "cycle"], kind="stable").reset_index(drop=True)
    entities = tuple(sorted(frame["entity_id"].unique().tolist()))
    split_config = {**config, "subsets": list(subsets)}
    splits = _entity_split(entities, split_config)
    return IMSBearingsDataset(
        frame=frame,
        feature_columns=tuple(feature_columns),
        splits=splits,
        subsets=subsets,
        file_interval_seconds=file_interval_seconds,
        feature_contract=(
            "per-channel RMS/std/kurtosis/crest_factor averaged within each configured "
            "bearing group; one entity per bearing"
        ),
        channel_groups=channel_groups_by_subset,
    )
