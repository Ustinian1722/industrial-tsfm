from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from asyncua import Server, ua

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    DeterministicPersistenceForecastModel,
    DiagnosisRequest,
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    LiveDiagnosisApplication,
    LiveForecastingApplication,
    OPCUALiveRuntime,
    browse_opcua_source,
    fit_pca_spe_artifact,
    read_opcua_snapshot,
)


async def _build_server(endpoint: str, *, temperature_value: float, pressure_value: float):
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    server.set_server_name("IndusTSFM CI OPC-UA Server")
    namespace = await server.register_namespace("urn:industsfm:ci")
    process = await server.nodes.objects.add_object(
        ua.NodeId("Process", namespace),
        "Process",
    )
    temperature = await process.add_variable(
        ua.NodeId("Temperature", namespace),
        "Temperature",
        temperature_value,
    )
    pressure = await process.add_variable(
        ua.NodeId("Pressure", namespace),
        "Pressure",
        pressure_value,
    )
    # Test-server writes simulate PLC value changes. Product/client code remains read-only.
    await temperature.set_writable()
    await pressure.set_writable()
    return server, process, temperature, pressure


def _offline_diagnosis_artifact():
    temperature = np.linspace(40.0, 46.0, 32)
    pressure = 1.95 + 0.08 * (temperature - 40.0) + 0.01 * np.sin(np.arange(32))
    training = pd.DataFrame(
        {
            "temperature": temperature,
            "pressure": pressure,
        }
    )
    return fit_pca_spe_artifact(
        training,
        ("temperature", "pressure"),
        threshold_quantile=0.95,
        n_components=1,
        min_train_rows=16,
    )


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float,
    description: str,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {description}")


async def main() -> None:
    output_dir = Path("results/opcua-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = "opc.tcp://127.0.0.1:4841/industsfm/server/"

    server, process, temperature, pressure = await _build_server(
        endpoint,
        temperature_value=42.5,
        pressure_value=2.15,
    )
    active_server: Server | None = server
    stop_event: asyncio.Event | None = None
    live_task: asyncio.Task[None] | None = None

    await server.start()
    try:
        source = DataSourceSpec(
            name="ci-opcua",
            kind=DataSourceKind.OPCUA,
            location=endpoint,
            metadata={
                "browse_root": process.nodeid.to_string(),
                "node_ids": {
                    "temperature": temperature.nodeid.to_string(),
                    "pressure": pressure.nodeid.to_string(),
                },
                "request_timeout_seconds": 1.0,
                "max_browse_depth": 2,
                "max_browse_nodes": 32,
                "sampling_interval_ms": 100,
                "max_buffer_rows": 64,
            },
        )
        browse = await browse_opcua_source(source)
        snapshot = await read_opcua_snapshot(source)
        (output_dir / "browse.json").write_text(
            json.dumps(browse, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        if browse["count"] < 3:
            raise RuntimeError("expected process root plus two child variables")
        if snapshot["sample_count"] != 2 or snapshot["good_count"] != 2:
            raise RuntimeError("OPC-UA snapshot did not return two good samples")
        values = {row["tag"]: row["value"] for row in snapshot["samples"]}
        if abs(float(values["temperature"]) - 42.5) > 1.0e-6:
            raise RuntimeError("temperature value mismatch")
        if abs(float(values["pressure"]) - 2.15) > 1.0e-6:
            raise RuntimeError("pressure value mismatch")

        live = OPCUALiveRuntime(source, reconnect_initial_seconds=0.1, reconnect_max_seconds=0.5)
        stop_event = asyncio.Event()
        live_task = asyncio.create_task(live.run(stop_event))
        await _wait_until(
            lambda: live.state.status == "running" and live.state.samples_received >= 2,
            timeout_seconds=5.0,
            description="initial OPC-UA subscription",
        )
        await temperature.write_value(43.25)
        await pressure.write_value(2.25)
        initial_samples = live.state.samples_received
        await _wait_until(
            lambda: live.state.samples_received > initial_samples,
            timeout_seconds=3.0,
            description="initial subscription data change",
        )

        await server.stop()
        active_server = None
        await _wait_until(
            lambda: live.state.status == "reconnecting" and live.state.disconnects >= 1,
            timeout_seconds=6.0,
            description="disconnect detection",
        )
        disconnected_snapshot = live.snapshot()
        (output_dir / "disconnected_runtime.json").write_text(
            json.dumps(disconnected_snapshot, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        server2, _process2, temperature2, pressure2 = await _build_server(
            endpoint,
            temperature_value=44.0,
            pressure_value=2.35,
        )
        await server2.start()
        active_server = server2
        await _wait_until(
            lambda: live.state.status == "running" and live.state.reconnects >= 1,
            timeout_seconds=8.0,
            description="OPC-UA reconnect and resubscribe",
        )
        samples_after_reconnect = live.state.samples_received
        await temperature2.write_value(45.0)
        await pressure2.write_value(2.45)
        await _wait_until(
            lambda: live.state.samples_received > samples_after_reconnect,
            timeout_seconds=3.0,
            description="post-reconnect subscription data",
        )

        forecast_app = LiveForecastingApplication(
            ForecastModelRuntimeAdapter(
                DeterministicPersistenceForecastModel(horizon=2),
                model_metadata={"purpose": "runtime_contract_smoke"},
            ),
            ForecastingRequest(
                target_columns=("temperature",),
                context_length=2,
                horizon=2,
                mode="per_tag_univariate",
            ),
        )
        live_forecast = forecast_app.infer(live).to_dict()
        (output_dir / "live_forecast.json").write_text(
            json.dumps(live_forecast, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if live_forecast["source_kind"] != "live":
            raise RuntimeError("forecast did not consume the live runtime window")
        if live_forecast["online_training"] is not False:
            raise RuntimeError("live forecast smoke must remain inference-only")
        if live_forecast["model"]["name"] != "persistence-ci":
            raise RuntimeError("unexpected live forecast CI model")
        if len(live_forecast["forecasts"]) != 2:
            raise RuntimeError("live forecast did not emit the configured horizon")

        diagnosis_app = LiveDiagnosisApplication(
            _offline_diagnosis_artifact(),
            DiagnosisRequest(context_observations=8, top_k_sensors=2),
        )
        live_diagnosis = diagnosis_app.infer(live)
        (output_dir / "live_diagnosis.json").write_text(
            json.dumps(live_diagnosis, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if live_diagnosis["source_kind"] != "live":
            raise RuntimeError("diagnosis did not consume the live runtime window")
        if live_diagnosis["online_training"] is not False:
            raise RuntimeError("live diagnosis smoke must remain inference-only")
        if live_diagnosis["threshold_updated_online"] is not False:
            raise RuntimeError("live diagnosis must not update its threshold")
        if live_diagnosis["control_actions_enabled"] is not False:
            raise RuntimeError("live diagnosis must not enable control actions")
        if not str(live_diagnosis["claim_scope"]).endswith("not_causal_root_cause"):
            raise RuntimeError("live diagnosis smoke lost the noncausal claim boundary")

        stop_event.set()
        await asyncio.wait_for(live_task, timeout=3.0)
        live_snapshot = live.snapshot()
        (output_dir / "live_runtime.json").write_text(
            json.dumps(live_snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        if live_snapshot["state"]["samples_received"] < 5:
            raise RuntimeError("OPC-UA subscription did not receive expected data changes")
        if live_snapshot["buffer"]["buffered_rows"] < 5:
            raise RuntimeError("live buffer did not preserve subscription samples")
        if live_snapshot["state"]["disconnects"] < 1 or live_snapshot["state"]["reconnects"] < 1:
            raise RuntimeError("disconnect/reconnect cycle was not recorded")
        if not float(live_snapshot["state"]["data_gap_seconds"]) > 0.0:
            raise RuntimeError("disconnect data gap was not recorded")
        transitions = [row["to"] for row in live_snapshot["state_transitions"]]
        if "reconnecting" not in transitions or transitions.count("running") < 2:
            raise RuntimeError("runtime state transitions do not show reconnect/resubscribe")
        if live_snapshot["read_only"] is not True or live_snapshot["writes_enabled"] is not False:
            raise RuntimeError("live product runtime must remain read-only")

        print(f"endpoint={endpoint}")
        print(f"browse_count={browse['count']}")
        print(f"good_samples={snapshot['good_count']}")
        print(f"subscription_samples={live_snapshot['state']['samples_received']}")
        print(f"disconnects={live_snapshot['state']['disconnects']}")
        print(f"reconnects={live_snapshot['state']['reconnects']}")
        print(f"data_gap_seconds={live_snapshot['state']['data_gap_seconds']:.3f}")
        print(f"forecast_points={len(live_forecast['forecasts'])}")
        print(f"diagnosis_severity={live_diagnosis['severity']}")
    finally:
        if stop_event is not None:
            stop_event.set()
        if live_task is not None and not live_task.done():
            live_task.cancel()
            try:
                await live_task
            except asyncio.CancelledError:
                pass
        if active_server is not None:
            await active_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
