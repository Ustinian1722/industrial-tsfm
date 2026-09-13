from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from industrial_tsfm.models.naive import NaiveModel
from industrial_tsfm.platform.forecasting import (
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    LiveForecastingApplication,
)
from industrial_tsfm.platform.forecasting_registry import LiveForecastingRegistry
from industrial_tsfm.platform.opcua_api import create_live_app
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


def _registry() -> LiveForecastingRegistry:
    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    application = LiveForecastingApplication(
        ForecastModelRuntimeAdapter(NaiveModel().configure(1)),
        ForecastingRequest(target_columns=("x",), context_length=2, horizon=1),
    )
    registry = LiveForecastingRegistry()
    registry.register(
        application_id="demo-live",
        project_id="demo",
        source_name="plc",
        task_name="forecast-x",
        runtime_id="opcua-0001-plc",
        application=application,
        provider=live,
    )
    return registry


def test_live_app_exposes_inference_only_forecasting_registry(tmp_path) -> None:
    registry = _registry()
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=tmp_path / "workspace",
        forecasting_registry=registry,
    )
    client = TestClient(app)

    capabilities = client.get("/v1/forecasting/capabilities")
    listing = client.get("/v1/forecasting/applications")
    status = client.get("/v1/forecasting/applications/demo-live")
    inferred = client.post("/v1/forecasting/applications/demo-live/infer")

    assert capabilities.status_code == 200
    assert capabilities.json()["http_model_registration"] is False
    assert capabilities.json()["online_training"] is False
    assert listing.json()["count"] == 1
    assert status.json()["application_id"] == "demo-live"
    assert inferred.status_code == 200
    assert inferred.json()["forecasts"][0]["value"] == 2.0
    assert registry.status("demo-live")["inference_count"] == 1


def test_live_forecasting_api_returns_404_for_unknown_application(tmp_path) -> None:
    app = create_live_app(
        artifact_root=tmp_path / "results",
        workspace_root=tmp_path / "workspace",
    )
    client = TestClient(app)

    response = client.post("/v1/forecasting/applications/missing/infer")

    assert response.status_code == 404
