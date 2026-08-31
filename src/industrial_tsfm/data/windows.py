from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastWindows:
    context: np.ndarray
    target: np.ndarray
    entity_ids: tuple[str, ...]
    origins: np.ndarray
    skipped_entities: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.context.shape[0])


def observed_prefix_before_origins(
    frame: pd.DataFrame,
    entity_ids: Iterable[str],
    origins: Iterable[int],
) -> pd.DataFrame:
    """Return rows observed before each entity's held-out forecast origin.

    ``origins`` are the cycle values at which the final evaluation target
    starts.  This helper is used by adaptation experiments so that target
    support scalers and support windows cannot consume the final target
    horizon, even though the raw C-MAPSS test file contains it.
    """
    entity_ids = tuple(entity_ids)
    origins = tuple(int(value) for value in origins)
    if len(entity_ids) != len(origins):
        raise ValueError("entity_ids and origins must have the same length")
    origin_by_entity = dict(zip(entity_ids, origins, strict=True))
    prefixes: list[pd.DataFrame] = []
    for entity_id, origin in origin_by_entity.items():
        entity_frame = frame[frame["entity_id"] == entity_id]
        prefix = entity_frame[entity_frame["cycle"] < origin]
        if prefix.empty:
            raise ValueError(f"No observed prefix exists before origin for {entity_id}")
        prefixes.append(prefix)
    return (
        pd.concat(prefixes, ignore_index=True)
        .sort_values(["entity_id", "cycle"], kind="stable")
        .reset_index(drop=True)
    )


def make_windows(
    frame: pd.DataFrame,
    entities: Iterable[str],
    feature_columns: Iterable[str],
    context_length: int,
    horizon: int,
    stride: int = 1,
    last_only: bool = False,
    include_last: bool = False,
) -> ForecastWindows:
    if context_length <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("context_length, horizon, and stride must be positive")
    feature_columns = tuple(feature_columns)
    requested_entities = tuple(sorted(set(entities)))
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    entity_ids: list[str] = []
    origins: list[int] = []
    skipped: list[str] = []
    for entity_id in requested_entities:
        entity_frame = frame[frame["entity_id"] == entity_id].sort_values("cycle")
        values = entity_frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
        cycles = entity_frame["cycle"].to_numpy(dtype=np.int64)
        n_origins = len(values) - context_length - horizon + 1
        if n_origins <= 0:
            skipped.append(entity_id)
            continue
        if last_only:
            origin_indices = [n_origins - 1]
        else:
            origin_indices = list(range(0, n_origins, stride))
            if include_last and origin_indices[-1] != n_origins - 1:
                origin_indices.append(n_origins - 1)
        for start in origin_indices:
            origin = start + context_length
            contexts.append(values[start:origin])
            targets.append(values[origin : origin + horizon])
            entity_ids.append(entity_id)
            origins.append(int(cycles[origin]))
    if not contexts:
        raise ValueError(
            "No valid forecast windows. Reduce context_length/horizon or provide longer entities."
        )
    return ForecastWindows(
        context=np.stack(contexts).astype(np.float32),
        target=np.stack(targets).astype(np.float32),
        entity_ids=tuple(entity_ids),
        origins=np.asarray(origins, dtype=np.int64),
        skipped_entities=tuple(skipped),
    )
