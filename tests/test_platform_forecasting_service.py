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


def _deployment(project_id: str = "demo") -> ForecastingDeploymentSpec:
    return ForecastingDeploymentSpec(
        deployment_id="demo-live",
        project_name="Demo",
        project_id=project_id,
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


def test_prepared_model_binding_never_trains_or_allows_metadata_override() -> None:
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
            model_metadata={
                "prepared_offline": True,
                "name": "attacker-model",
                "training_mode": "online-fit",
                "online_training": True,
                "selected_model": "attacker-model",
                "selected_strategy": "online-fit",
                "selection_evidence": "target_guided",
                "target_labels_used": True,
                "checkpoint_ref": "untrusted-ref",
                "deployment_id": "wrong-deployment",
                "project_id": "wrong-project",
            },
        )

        model = status["model"]
        assert status["online_training"] is False
        assert model["name"] == "persistence-ci"
        assert model["training_mode"] == "none"
        assert model["online_training"] is False
        assert model["selected_model"] == "chronos"
        assert model["selected_strategy"] == "zero_shot"
        assert model["selection_evidence"] == "validation_only"
        assert model["target_labels_used"] is False
        assert model["checkpoint_ref"] == "amazon/chronos-t5-tiny"
        assert model["deployment_id"] == "demo-live"
        assert model["project_id"] == "demo"
        assert model["prepared_offline"] is True
        payload = forecasts.infer_payload("demo-live")
        assert payload["forecasts"][0]["value"] == 2.0
        assert payload["model"]["online_training"] is False
        await opcua.stop_all()

    asyncio.run(scenario())


def test_prepared_model_binding_rejects_runtime_from_other_project() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeLiveRuntime)
        started = await opcua.start(_source(), project_id="other-project")
        forecasts = LiveForecastingRegistry()

        try:
            register_prepared_live_forecast(
                deployment=_deployment(project_id="demo"),
                runtime_id=started["runtime_id"],
                prepared_model=DeterministicPersistenceForecastModel(1),
                opcua_registry=opcua,
                forecasting_registry=forecasts,
            )
        except ValueError as exc:
            assert "runtime project" in str(exc)
        else:
            raise AssertionError("cross-project runtime binding must be rejected")
        await opcua.stop_all()

    asyncio.run(scenario())
