from __future__ import annotations

import pandas as pd
import pytest

from industrial_tsfm.platform.forecasting import (
    DeterministicPersistenceForecastModel,
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    LiveForecastingApplication,
)
from industrial_tsfm.platform.runtime import (
    BoundedLiveBuffer,
    DataReplayRuntime,
    LiveSample,
    ReplayConfig,
)


def _persistence(horizon: int) -> ForecastModelRuntimeAdapter:
    return ForecastModelRuntimeAdapter(
        DeterministicPersistenceForecastModel(horizon),
        model_metadata={"checkpoint": "none", "selection_evidence": "ci_fixture"},
    )


def test_per_tag_live_forecasting_uses_arrival_stream_without_alignment_or_fill() -> None:
    live = BoundedLiveBuffer(max_rows=32)
    live.extend(
        [
            LiveSample(tag="temperature", value=10.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="pressure", value=2.0, timestamp="2026-01-01T00:00:00.500000Z"),
            LiveSample(tag="temperature", value=11.0, timestamp="2026-01-01T00:00:01Z"),
            LiveSample(tag="pressure", value=2.2, timestamp="2026-01-01T00:00:01.500000Z"),
            LiveSample(tag="temperature", value=12.0, timestamp="2026-01-01T00:00:02Z"),
            LiveSample(tag="pressure", value=2.4, timestamp="2026-01-01T00:00:02.500000Z"),
        ]
    )
    app = LiveForecastingApplication(
        _persistence(2),
        ForecastingRequest(
            target_columns=("temperature", "pressure"),
            context_length=3,
            horizon=2,
            mode="per_tag_univariate",
        ),
    )

    result = app.infer(live).to_dict()

    assert result["schema_version"] == "industrial_tsfm.live_forecasting.v1"
    assert result["source_kind"] == "live"
    assert result["online_training"] is False
    assert result["model"]["online_training"] is False
    assert result["model"]["name"] == "persistence-ci"
    forecasts = {(row["target"], row["step"]): row for row in result["forecasts"]}
    assert forecasts[("temperature", 1)]["value"] == pytest.approx(12.0)
    assert forecasts[("temperature", 2)]["value"] == pytest.approx(12.0)
    assert forecasts[("pressure", 1)]["value"] == pytest.approx(2.4)
    assert forecasts[("temperature", 1)]["timestamp"] == "2026-01-01T00:00:03+00:00"
    assert forecasts[("pressure", 1)]["timestamp"] == "2026-01-01T00:00:03.500000+00:00"
    assert result["context"]["temperature"]["cadence"] == "regular_observed"
    assert result["context"]["pressure"]["period_seconds"] == 1.0


def test_irregular_live_cadence_does_not_invent_future_timestamps() -> None:
    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:01Z"),
            LiveSample(tag="x", value=3.0, timestamp="2026-01-01T00:00:03Z"),
        ]
    )
    app = LiveForecastingApplication(
        _persistence(2),
        ForecastingRequest(target_columns=("x",), context_length=3, horizon=2),
    )

    result = app.infer(live).to_dict()

    assert result["context"]["x"]["cadence"] == "irregular"
    assert [row["timestamp"] for row in result["forecasts"]] == [None, None]
    assert result["quality"]["forecast_timestamps_available"] is False


def test_bad_quality_live_context_is_rejected_by_default() -> None:
    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(
                tag="x",
                value=2.0,
                timestamp="2026-01-01T00:00:01Z",
                status="Bad",
                status_good=False,
            ),
        ]
    )
    app = LiveForecastingApplication(
        _persistence(1),
        ForecastingRequest(target_columns=("x",), context_length=2, horizon=1),
    )

    try:
        app.infer(live)
    except ValueError as exc:
        assert "bad-quality" in str(exc)
    else:
        raise AssertionError("bad-quality context must be rejected")


def test_strict_multivariate_replay_uses_complete_rows_only_without_filling() -> None:
    replay = DataReplayRuntime(
        pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                ],
                "temperature": [10.0, 11.0, 12.0],
                "pressure": [2.0, 2.2, 2.4],
            }
        ),
        ReplayConfig(timestamp_column="timestamp"),
    )
    app = LiveForecastingApplication(
        _persistence(2),
        ForecastingRequest(
            target_columns=("temperature", "pressure"),
            context_length=3,
            horizon=2,
            mode="strict_multivariate",
        ),
    )

    result = app.infer(replay).to_dict()

    assert result["source_kind"] == "replay"
    forecasts = {(row["target"], row["step"]): row["value"] for row in result["forecasts"]}
    assert forecasts[("temperature", 1)] == pytest.approx(12.0)
    assert forecasts[("pressure", 2)] == pytest.approx(2.4)
    assert result["context"]["cadence"] == "regular_observed"
    assert result["context"]["period_seconds"] == 1.0
