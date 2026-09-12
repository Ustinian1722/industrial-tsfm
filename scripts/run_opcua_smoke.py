from __future__ import annotations

import asyncio
import json
from pathlib import Path

from asyncua import Server

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
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
                "sampling_interval_ms": 250,
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
        print(f"endpoint={endpoint}")
        print(f"browse_count={browse['count']}")
        print(f"good_samples={snapshot['good_count']}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
