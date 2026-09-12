from __future__ import annotations

import asyncio
import json
from pathlib import Path

from asyncua import Server

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    OPCUALiveRuntime,
    browse_opcua_source,
    read_opcua_snapshot,
)


async def main() -> None:
    output_dir = Path("results/opcua-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    server = Server()
    await server.init()
    endpoint = "opc.tcp://127.0.0.1:4841/industsfm/server/"
    server.set_endpoint(endpoint)
    server.set_server_name("IndusTSFM CI OPC-UA Server")
    namespace = await server.register_namespace("urn:industsfm:ci")
    process = await server.nodes.objects.add_object(namespace, "Process")
    temperature = await process.add_variable(namespace, "Temperature", 42.5)
    pressure = await process.add_variable(namespace, "Pressure", 2.15)
    await temperature.set_writable()
    await pressure.set_writable()

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
        task = asyncio.create_task(live.run(stop_event))
        await asyncio.sleep(0.4)
        await temperature.write_value(43.25)
        await pressure.write_value(2.25)
        await asyncio.sleep(0.4)
        await temperature.write_value(44.0)
        await asyncio.sleep(0.4)
        stop_event.set()
        await asyncio.wait_for(task, timeout=3.0)
        live_snapshot = live.snapshot()
        (output_dir / "live_runtime.json").write_text(
            json.dumps(live_snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        if live_snapshot["state"]["samples_received"] < 3:
            raise RuntimeError("OPC-UA subscription did not receive expected data changes")
        if live_snapshot["buffer"]["buffered_rows"] < 3:
            raise RuntimeError("live buffer did not preserve subscription samples")
        if live_snapshot["read_only"] is not True or live_snapshot["writes_enabled"] is not False:
            raise RuntimeError("live product runtime must remain read-only")

        print(f"endpoint={endpoint}")
        print(f"browse_count={browse['count']}")
        print(f"good_samples={snapshot['good_count']}")
        print(f"subscription_samples={live_snapshot['state']['samples_received']}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
