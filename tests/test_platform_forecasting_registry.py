from __future__ import annotations

from industrial_tsfm.platform.forecasting import (
    DeterministicPersistenceForecastModel,
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    LiveForecastingApplication,
)
from industrial_tsfm.platform.forecasting_registry import LiveForecastingRegistry
from industrial_tsfm.platform.forecasting_uq import ConformalForecastArtifact
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


def _app() -> LiveForecastingApplication:
    model = ForecastModelRuntimeAdapter(DeterministicPersistenceForecastModel(1))
    request = ForecastingRequest(target_columns=("x",), context_length=2, horizon=1)
    return LiveForecastingApplication(model, request)


def _provider() -> BoundedLiveBuffer:
    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    return live


def test_forecasting_registry_tracks_inference_without_mutating_model() -> None:
    registry = LiveForecastingRegistry()
    status = registry.register(
        application_id="demo-live-forecast",
        runtime_id="opcua-0001-plc",
        project_id="demo",
        source_name="plc",
        task_name="forecast-x",
        application=_app(),
        provider=_provider(),
    )

    assert status["online_training"] is False
    assert status["online_calibration"] is False
    assert status["inference_count"] == 0
    assert status["model"]["name"] == "persistence-ci"

    record = registry.infer("demo-live-forecast").to_dict()
    assert record["forecasts"][0]["value"] == 2.0
    after = registry.status("demo-live-forecast")
    assert after["inference_count"] == 1
    assert after["last_generated_at"] == record["generated_at"]
    assert after["last_observed_through"] == record["observed_through"]
    assert after["model"]["online_training"] is False


def test_forecasting_registry_applies_frozen_uq_without_recalibration() -> None:
    registry = LiveForecastingRegistry()
    artifact = ConformalForecastArtifact(
        target_columns=("x",),
        horizon=1,
        alpha=0.1,
        radii=((0.25,),),
        calibration_rows=40,
    )
    status = registry.register(
        application_id="uq-live",
        runtime_id="opcua-0001-plc",
        source_name="plc",
        task_name="forecast-x",
        application=_app(),
        provider=_provider(),
        uq_artifact=artifact,
    )

    assert status["uq"]["nominal_coverage"] == 0.9
    assert status["uq"]["online_updates"] is False
    payload = registry.infer_payload("uq-live")
    assert payload["forecasts"][0]["value"] == 2.0
    assert payload["forecasts"][0]["lower"] == 1.75
    assert payload["forecasts"][0]["upper"] == 2.25
    assert payload["uq"]["online_updates"] is False
    assert artifact.radii == ((0.25,),)


def test_forecasting_registry_rejects_duplicate_ids_and_supports_remove() -> None:
    registry = LiveForecastingRegistry()
    kwargs = {
        "application_id": "same",
        "runtime_id": "runtime-1",
        "source_name": "plc",
        "task_name": "forecast-x",
        "application": _app(),
        "provider": _provider(),
    }
    registry.register(**kwargs)
    try:
        registry.register(**kwargs)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate application id must be rejected")

    registry.remove("same")
    assert registry.list() == []
