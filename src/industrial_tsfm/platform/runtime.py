from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReplayConfig:
    timestamp_column: str | None = None
    batch_size: int = 32

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class ReplayBatch:
    batch_index: int
    row_start: int
    row_end: int
    records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataReplayRuntime:
    """Deterministic offline replay primitive for product/application development.

    It preserves input row order and never sleeps, interpolates, or sorts. That
    makes it suitable for CI and for validating application semantics before a
    wall-clock streaming connector is introduced.
    """

    def __init__(self, frame: pd.DataFrame, config: ReplayConfig | None = None) -> None:
        if frame.empty:
            raise ValueError("cannot replay an empty dataframe")
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config or ReplayConfig()
        self.config.validate()
        if self.config.timestamp_column and self.config.timestamp_column not in self.frame.columns:
            raise ValueError(
                f"timestamp column {self.config.timestamp_column!r} is missing from replay data"
            )

    def iter_batches(self) -> Iterator[ReplayBatch]:
        size = self.config.batch_size
        for batch_index, start in enumerate(range(0, len(self.frame), size)):
            end = min(start + size, len(self.frame))
            records = tuple(
                self.frame.iloc[start:end].where(pd.notna(self.frame.iloc[start:end]), None).to_dict(
                    orient="records"
                )
            )
            yield ReplayBatch(
                batch_index=batch_index,
                row_start=start,
                row_end=end,
                records=records,
            )

    def snapshot(self, preview_batches: int = 2) -> dict[str, Any]:
        if preview_batches < 0:
            raise ValueError("preview_batches must be non-negative")
        preview: list[dict[str, Any]] = []
        total_batches = 0
        for batch in self.iter_batches():
            total_batches += 1
            if len(preview) < preview_batches:
                preview.append(
                    {
                        "batch_index": batch.batch_index,
                        "row_start": batch.row_start,
                        "row_end": batch.row_end,
                        "record_count": len(batch.records),
                    }
                )

        time_summary: dict[str, Any] = {
            "timestamp_column": self.config.timestamp_column,
            "start": None,
            "end": None,
        }
        if self.config.timestamp_column:
            values = pd.to_datetime(
                self.frame[self.config.timestamp_column], errors="coerce"
            ).dropna()
            if not values.empty:
                time_summary["start"] = values.iloc[0].isoformat()
                time_summary["end"] = values.iloc[-1].isoformat()

        return {
            "schema_version": "industrial_tsfm.replay.v1",
            "rows": len(self.frame),
            "columns": [str(column) for column in self.frame.columns],
            "batch_size": self.config.batch_size,
            "total_batches": total_batches,
            "preserves_input_order": True,
            "wall_clock_sleep": False,
            "time": time_summary,
            "preview_batches": preview,
        }
