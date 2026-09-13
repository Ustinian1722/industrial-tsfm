from __future__ import annotations

import pandas as pd

from industrial_tsfm.platform.data_audit import audit_dataframe
from industrial_tsfm.platform.runtime import (
    BoundedLiveBuffer,
    DataReplayRuntime,
    LiveSample,
    ReplayConfig,
)


def test_live_window_preserves_arrival_order_quality_and_no_hidden_fill() -> None:
    buffer = BoundedLiveBuffer(max_rows=8)
    buffer.extend(
        [
            LiveSample(tag="pressure", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(
                tag="temperature",
                value=30.0,
                timestamp="2026-01-01T00:00:00Z",
                status="Bad",
                status_good=False,
            ),
            LiveSample(tag="pressure", value=1.2, timestamp="2026-01-01T00:00:01Z"),
        ]
    )

    window = buffer.materialize_window()

    assert window.frame["timestamp"].tolist() == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:01Z",
    ]
    assert window.frame["pressure"].tolist() == [1.0, 1.2]
    assert pd.isna(window.frame.loc[1, "temperature"])
    assert bool(window.quality_mask.loc[0, "pressure"]) is True
    assert bool(window.quality_mask.loc[0, "temperature"]) is False
    assert window.arrival_frame["arrival_index"].tolist() == [0, 1, 2]
    assert window.metadata["timestamp_sort_applied"] is False
    assert window.metadata["resampling"] is False
    assert window.metadata["interpolation"] is False
    assert window.metadata["bad_quality_observations"] == 1

    audit = audit_dataframe(window.frame, timestamp_column="timestamp")
    assert audit["time"]["valid"] is True
    assert audit["time"]["out_of_order_transitions"] == 0


def test_live_window_never_overwrites_repeated_tag_at_same_timestamp() -> None:
    buffer = BoundedLiveBuffer(max_rows=4)
    buffer.extend(
        [
            LiveSample(tag="speed", value=100.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="speed", value=101.0, timestamp="2026-01-01T00:00:00Z"),
        ]
    )

    window = buffer.materialize_window()

    assert len(window.frame) == 2
    assert window.frame["speed"].tolist() == [100.0, 101.0]
    assert window.metadata["duplicate_timestamps"] == 1


def test_live_window_exposes_out_of_order_evidence_without_sorting() -> None:
    buffer = BoundedLiveBuffer(max_rows=4)
    buffer.extend(
        [
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:02Z"),
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )

    window = buffer.materialize_window()

    assert window.frame["x"].tolist() == [2.0, 1.0]
    assert window.metadata["out_of_order_transitions"] == 1
    assert window.metadata["timestamp_sort_applied"] is False


def test_live_window_limit_uses_latest_arrivals_and_keeps_absolute_indices() -> None:
    buffer = BoundedLiveBuffer(max_rows=3)
    for index in range(5):
        buffer.append(
            LiveSample(
                tag="x",
                value=float(index),
                timestamp=f"2026-01-01T00:00:0{index}Z",
            )
        )

    window = buffer.materialize_window(max_observations=2)

    assert window.frame["x"].tolist() == [3.0, 4.0]
    assert window.arrival_frame["arrival_index"].tolist() == [3, 4]
    assert window.metadata["buffer_received_rows"] == 5
    assert window.metadata["buffer_dropped_rows"] == 2


def test_replay_and_live_share_materialize_window_contract() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"],
            "x": [2.0, 1.0],
        }
    )
    replay = DataReplayRuntime(frame, ReplayConfig(timestamp_column="timestamp", batch_size=1))

    window = replay.materialize_window(max_observations=2)

    assert window.source_kind == "replay"
    assert window.frame["x"].tolist() == [2.0, 1.0]
    assert window.arrival_frame["arrival_index"].tolist() == [0, 1]
    assert window.metadata["out_of_order_transitions"] == 1
    assert window.metadata["timestamp_sort_applied"] is False
    assert window.metadata["resampling"] is False
    assert window.metadata["interpolation"] is False
