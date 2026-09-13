from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

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


@dataclass(frozen=True)
class TimeSeriesWindow:
    """Auditable dataframe handoff shared by replay and live runtimes.

    ``frame`` is the model/audit-facing representation. ``arrival_frame`` keeps
    the observations in the exact order accepted by the runtime so consumers
    can verify that no timestamp sort, interpolation, or hidden resampling was
    introduced. ``quality_mask`` is aligned with the feature columns in frame.
    """

    frame: pd.DataFrame
    quality_mask: pd.DataFrame
    arrival_frame: pd.DataFrame
    source_kind: str
    timestamp_column: str | None
    metadata: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.runtime_window.v1",
            "source_kind": self.source_kind,
            "rows": len(self.frame),
            "arrival_rows": len(self.arrival_frame),
            "columns": [str(column) for column in self.frame.columns],
            "quality_columns": [str(column) for column in self.quality_mask.columns],
            "timestamp_column": self.timestamp_column,
            "metadata": dict(self.metadata),
        }


class WindowProvider(Protocol):
    """Source-agnostic contract consumed by streaming inference applications."""

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow: ...


def _validate_window_limit(max_observations: int | None) -> None:
    if max_observations is not None and max_observations < 1:
        raise ValueError("max_observations must be positive when configured")


def _time_order_evidence(values: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    valid = parsed.dropna()
    diffs = valid.diff().dropna()
    return {
        "parse_failures": int(parsed.isna().sum()),
        "out_of_order_transitions": int((diffs < pd.Timedelta(0)).sum()),
        "duplicate_timestamps": int(valid.duplicated().sum()),
    }


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

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow:
        """Return the latest replay rows without changing their input order."""

        _validate_window_limit(max_observations)
        start = 0 if max_observations is None else max(0, len(self.frame) - max_observations)
        frame = self.frame.iloc[start:].reset_index(drop=True).copy()
        arrival_frame = frame.copy()
        arrival_frame.insert(0, "arrival_index", range(start, start + len(frame)))
        feature_columns = [
            column for column in frame.columns if column != self.config.timestamp_column
        ]
        quality_mask = frame[feature_columns].notna().copy()
        time_evidence: dict[str, Any] = {
            "parse_failures": 0,
            "out_of_order_transitions": 0,
            "duplicate_timestamps": 0,
        }
        if self.config.timestamp_column:
            time_evidence = _time_order_evidence(frame[self.config.timestamp_column])
        metadata = {
            "source_rows": len(frame),
            "source_row_start": start,
            "preserves_arrival_order": True,
            "timestamp_sort_applied": False,
            "resampling": False,
            "interpolation": False,
            "coalescing": "none",
            **time_evidence,
        }
        return TimeSeriesWindow(
            frame=frame,
            quality_mask=quality_mask,
            arrival_frame=arrival_frame,
            source_kind="replay",
            timestamp_column=self.config.timestamp_column,
            metadata=metadata,
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


@dataclass(frozen=True)
class LiveSample:
    tag: str
    value: Any
    timestamp: str
    status: str = "Good"
    status_good: bool = True
    node_id: str | None = None
    source_timestamp: str | None = None
    server_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedLiveBuffer:
    """Bounded, append-only live sample buffer shared by streaming connectors.

    The buffer never reorders, resamples or interpolates incoming samples. When
    capacity is exceeded it drops the oldest record and exposes that drop count
    in the runtime snapshot so data loss is visible to operators.
    """

    def __init__(self, max_rows: int = 4096) -> None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        self.max_rows = int(max_rows)
        self._rows: deque[LiveSample] = deque(maxlen=self.max_rows)
        self.received_rows = 0
        self.dropped_rows = 0

    def append(self, sample: LiveSample) -> None:
        if len(self._rows) == self.max_rows:
            self.dropped_rows += 1
        self._rows.append(sample)
        self.received_rows += 1

    def extend(self, samples: Iterator[LiveSample] | list[LiveSample] | tuple[LiveSample, ...]) -> None:
        for sample in samples:
            self.append(sample)

    def clear(self) -> None:
        self._rows.clear()
        self.received_rows = 0
        self.dropped_rows = 0

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(sample.to_dict() for sample in self._rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records())

    def latest_by_tag(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for sample in self._rows:
            latest[sample.tag] = sample.to_dict()
        return latest

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow:
        """Create a wide chronological *evidence* window without sorting timestamps.

        Samples are consumed strictly in callback arrival order. Adjacent samples
        with the same effective timestamp are coalesced only when they refer to
        different tags. A repeated tag at the same timestamp starts a new output
        row rather than overwriting an earlier sample. This preserves every
        accepted observation while making a dataframe suitable for Data Audit and
        downstream inference. Missing tags remain missing; no interpolation or
        forward fill is performed.
        """

        _validate_window_limit(max_observations)
        buffered = list(self._rows)
        absolute_start = self.received_rows - len(buffered)
        if max_observations is not None and len(buffered) > max_observations:
            trim = len(buffered) - max_observations
            buffered = buffered[trim:]
            absolute_start += trim

        arrival_records: list[dict[str, Any]] = []
        for offset, sample in enumerate(buffered):
            record = sample.to_dict()
            record["arrival_index"] = absolute_start + offset
            arrival_records.append(record)
        arrival_frame = pd.DataFrame(arrival_records)

        tags = list(dict.fromkeys(sample.tag for sample in buffered))
        wide_rows: list[dict[str, Any]] = []
        quality_rows: list[dict[str, bool]] = []
        current_timestamp: str | None = None
        current_values: dict[str, Any] = {}
        current_quality: dict[str, bool] = {}

        def flush() -> None:
            if current_timestamp is None:
                return
            row: dict[str, Any] = {"timestamp": current_timestamp}
            row.update(current_values)
            wide_rows.append(row)
            quality_rows.append({tag: bool(current_quality.get(tag, False)) for tag in tags})

        for sample in buffered:
            should_flush = current_timestamp is not None and (
                sample.timestamp != current_timestamp or sample.tag in current_values
            )
            if should_flush:
                flush()
                current_values = {}
                current_quality = {}
            if current_timestamp is None or should_flush:
                current_timestamp = sample.timestamp
            current_values[sample.tag] = sample.value
            current_quality[sample.tag] = sample.status_good
        flush()

        frame = pd.DataFrame(wide_rows, columns=["timestamp", *tags])
        quality_mask = pd.DataFrame(quality_rows, columns=tags, dtype=bool)
        time_evidence: dict[str, Any] = {
            "parse_failures": 0,
            "out_of_order_transitions": 0,
            "duplicate_timestamps": 0,
        }
        if not frame.empty:
            time_evidence = _time_order_evidence(frame["timestamp"])
        metadata = {
            "source_rows": len(buffered),
            "buffer_received_rows": self.received_rows,
            "buffer_dropped_rows": self.dropped_rows,
            "preserves_arrival_order": True,
            "timestamp_sort_applied": False,
            "resampling": False,
            "interpolation": False,
            "coalescing": "contiguous_equal_timestamp_distinct_tags_only",
            "bad_quality_observations": sum(int(not sample.status_good) for sample in buffered),
            **time_evidence,
        }
        return TimeSeriesWindow(
            frame=frame,
            quality_mask=quality_mask,
            arrival_frame=arrival_frame,
            source_kind="live",
            timestamp_column="timestamp",
            metadata=metadata,
        )

    def snapshot(self) -> dict[str, Any]:
        rows = list(self._rows)
        bad = sum(1 for sample in rows if not sample.status_good)
        return {
            "schema_version": "industrial_tsfm.live_buffer.v1",
            "buffered_rows": len(rows),
            "capacity": self.max_rows,
            "received_rows": self.received_rows,
            "dropped_rows": self.dropped_rows,
            "bad_quality_rows_buffered": bad,
            "preserves_arrival_order": True,
            "resampling": False,
            "interpolation": False,
            "first_timestamp": rows[0].timestamp if rows else None,
            "last_timestamp": rows[-1].timestamp if rows else None,
            "latest_by_tag": self.latest_by_tag(),
        }
