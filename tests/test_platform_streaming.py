from __future__ import annotations

import pandas as pd

from industrial_tsfm.platform.runtime import (
    BoundedLiveBuffer,
    DataReplayRuntime,
    LiveSample,
    ReplayConfig,
)
from industrial_tsfm.platform.streaming import (
    DeterministicLastValuePredictor,
    StreamingInferenceEngine,
)


def test_streaming_engine_uses_same_contract_for_replay_and_live() -> None:
    predictor = DeterministicLastValuePredictor(("pressure",))
    engine = StreamingInferenceEngine(predictor, context_observations=4)

    replay = DataReplayRuntime(
        pd.DataFrame(
            {
                "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
                "pressure": [1.0, 1.2],
            }
        ),
        ReplayConfig(timestamp_column="timestamp"),
    )
    replay_record = engine.infer(replay)

    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="pressure", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="pressure", value=1.2, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    live_record = engine.infer(live)

    assert replay_record.predictions == {"pressure": 1.2}
    assert live_record.predictions == {"pressure": 1.2}
    assert replay_record.source_kind == "replay"
    assert live_record.source_kind == "live"
    assert replay_record.model["online_training"] is False
    assert live_record.model["family"] == "deterministic_test_double"
    assert replay_record.quality["state"] == "good"
    assert live_record.quality["state"] == "good"
    assert replay_record.quality["chronology_valid"] is True
    assert live_record.quality["chronology_valid"] is True
    assert replay_record.latency_ms >= 0.0
    assert live_record.latency_ms >= 0.0


def test_streaming_record_surfaces_live_quality_and_drop_risk() -> None:
    live = BoundedLiveBuffer(max_rows=2)
    live.extend(
        [
            LiveSample(tag="pressure", value=0.8, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(
                tag="pressure",
                value=1.0,
                timestamp="2026-01-01T00:00:01Z",
                status="Bad",
                status_good=False,
            ),
            LiveSample(tag="pressure", value=1.2, timestamp="2026-01-01T00:00:02Z"),
        ]
    )
    engine = StreamingInferenceEngine(DeterministicLastValuePredictor(("pressure",)))

    record = engine.infer(live)

    assert record.predictions == {"pressure": 1.2}
    assert record.quality["state"] == "degraded"
    assert record.quality["bad_quality_observations"] == 1
    assert record.quality["dropped_observations"] == 1
    assert record.quality["chronology_valid"] is True
    assert record.observed_through == "2026-01-01T00:00:02Z"
    assert record.window["metadata"]["timestamp_sort_applied"] is False


def test_streaming_engine_rejects_insufficient_context_rows() -> None:
    live = BoundedLiveBuffer(max_rows=4)
    live.append(LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"))
    engine = StreamingInferenceEngine(
        DeterministicLastValuePredictor(("x",)),
        minimum_rows=2,
    )

    try:
        engine.infer(live)
    except ValueError as exc:
        assert "insufficient context rows" in str(exc)
    else:
        raise AssertionError("insufficient context must block inference")


def test_streaming_engine_rejects_out_of_order_time_without_sorting() -> None:
    live = BoundedLiveBuffer(max_rows=4)
    live.extend(
        [
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:02Z"),
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    engine = StreamingInferenceEngine(DeterministicLastValuePredictor(("x",)))

    try:
        engine.infer(live)
    except ValueError as exc:
        assert "requires chronological timestamps" in str(exc)
    else:
        raise AssertionError("out-of-order context must block default streaming inference")

    window = live.materialize_window()
    assert window.frame["x"].tolist() == [2.0, 1.0]
    assert window.metadata["out_of_order_transitions"] == 1
    assert window.metadata["timestamp_sort_applied"] is False


def test_streaming_engine_can_explicitly_accept_nonchronological_diagnostic_context() -> None:
    live = BoundedLiveBuffer(max_rows=4)
    live.extend(
        [
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:02Z"),
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    engine = StreamingInferenceEngine(
        DeterministicLastValuePredictor(("x",)),
        require_chronological=False,
    )

    record = engine.infer(live)

    assert record.predictions == {"x": 1.0}
    assert record.quality["state"] == "degraded"
    assert record.quality["chronology_valid"] is False
    assert record.quality["out_of_order_transitions"] == 1
