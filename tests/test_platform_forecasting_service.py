from __future__ import annotations

import asyncio

from industrial_tsfm.platform.contracts import DataSourceKind, DataSourceSpec
from industrial_tsfm.platform.forecasting import DeterministicPersistenceForecastModel
from industrial_tsfm.platform.forecasting_deployment import ForecastingDeploymentSpec
from industrial_tsfm.platform.forecasting_registry import LiveForecastingRegistry
from industrial_tsfm.platform.forecasting_service import register_prepared_live_forecast
from industrial_tsfm.platform.opcua_registry import OPCUALiveRuntimeRegistry
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


class FakeLiveRuntime:
    def __init__(self, source: DataSourceSpec) -> None:
        self.source = source
        self.buffer = BoundedLiveBuffer(max_rows=8)
        self.buffer.extend(
            [
                LiveSample(tag="x", value=1.0, timestamp="2026-01-01T00:00:00Z"),
                LiveSample(tag="x", value=2.0, timestamp="2026-01-01T00:00:01Z"),
            ]
        )
        self.status_name = "created"

    async def run(self, stop_event: asyncio.Event) -> None:
        self.status_name = "running"
        await stop_event.wait()
        self.status_name = "stopped"

    def materialize_window(self, max_observations=None):
        return self.buffer.materialize_window(max_observations)

    def snapshot(self):
        return {
            "schema_version": "fake-live-runtime",
            "source_name": self.source.name,
            "state": {"status": self.status_name},
            "read_only": True,
            "writes_enabled": False,
        }


def _source() -> DataSourceSpec:
    return DataSourceSpec(
        name="plc",
        kind=DataSourceKind.OPCUA,
        location="opc.tcp://127.0.0.1:4840/demo",
        metadata={"node_ids": {"x": "ns=2;s=x"}},
    )


def _deployment() -> ForecastingDeploymentSpec:
    return ForecastingDeploymentSpec(
        deployment_id="demo-live",
        project_name="Demo",
        source_name="plc",
        task_name="forecast-x",
        target_columns=("x",),
        context_length=2,
        horizon=1,
        selected_model="chronos",
        selected_strategy="zero_shot",
        selection_evidence="validation_only",
        target_labels_used=False,
        window_mode="per_tag_univariate",
        requires_aligned_rows=False,
        checkpoint_ref="amazon/chronos-t5-tiny",
    )


def test_prepared_model_binding_never_trains_or_selects_online() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeLiveRuntime)
        started = await opcua.start(_source(), project_id="demo")
        forecasts = LiveForecastingRegistry()

        status = register_prepared_live_forecast(
            deployment=_deployment(),
            runtime_id=started["runtime_id"],
            prepared_model=DeterministicPersistenceForecastModel(1),
            opcua_registry=opcua,
            forecasting_registry=forecasts,
            model_metadata={"project_id": "demo", "prepared_offline": True},
        )

        assert status["online_training"] is False
        assert status["model"]["selection_evidence"] == "validation_only"
        assert status["model"]["target_labels_used"] is False
        assert status["model"]["prepared_offline"] is True
        payload = forecasts.infer_payload("demo-live")
        assert payload["forecasts"][0]["value"] == 2.0
        await opcua.stop_all()

    asyncio.run(scenario())
