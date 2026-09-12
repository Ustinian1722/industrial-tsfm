from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from industrial_tsfm.platform.contracts import DataSourceKind, DataSourceSpec
from industrial_tsfm.platform.opcua import (
    OPCUAConnectionConfig,
    OPCUASample,
    OPCUATag,
    browse_opcua_source,
    config_from_source,
    public_connection_metadata,
    read_opcua_snapshot,
)
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


class FakeTransport:
    async def browse(self, config: OPCUAConnectionConfig) -> list[OPCUATag]:
        assert config.endpoint == "opc.tcp://127.0.0.1:4840/demo"
        return [
            OPCUATag(name="Temperature", node_id="ns=2;s=Temperature", node_class="Variable"),
            OPCUATag(name="Pressure", node_id="ns=2;s=Pressure", node_class="Variable"),
        ]

    async def read_snapshot(
        self,
        config: OPCUAConnectionConfig,
        nodes: dict[str, str],
    ) -> list[OPCUASample]:
        now = datetime.now(timezone.utc).isoformat()
        values = {"temperature": 42.5, "pressure": 2.15}
        return [
            OPCUASample(
                tag=tag,
                node_id=node_id,
                value=values[tag],
                source_timestamp=now,
                server_timestamp=now,
                received_timestamp=now,
                status="Good",
                status_good=True,
            )
            for tag, node_id in nodes.items()
        ]


def _source() -> DataSourceSpec:
    return DataSourceSpec(
        name="plc-demo",
        kind=DataSourceKind.OPCUA,
        location="opc.tcp://127.0.0.1:4840/demo",
        metadata={
            "node_ids": {
                "temperature": "ns=2;s=Temperature",
                "pressure": "ns=2;s=Pressure",
            },
            "sampling_interval_ms": 500,
            "max_buffer_rows": 16,
        },
    )


def test_opcua_config_and_public_metadata_do_not_expose_secret_values(monkeypatch) -> None:
    source = DataSourceSpec(
        name="secure-plc",
        kind=DataSourceKind.OPCUA,
        location="opc.tcp://plc.internal:4840/plant",
        metadata={
            "username_env": "INDUSTSFM_OPCUA_USER",
            "password_env": "INDUSTSFM_OPCUA_PASSWORD",
            "security_string_env": "INDUSTSFM_OPCUA_SECURITY",
        },
    )
    monkeypatch.setenv("INDUSTSFM_OPCUA_USER", "operator")
    monkeypatch.setenv("INDUSTSFM_OPCUA_PASSWORD", "super-secret")
    monkeypatch.setenv("INDUSTSFM_OPCUA_SECURITY", "Basic256Sha256,SignAndEncrypt,cert,key")

    config = config_from_source(source)
    public = public_connection_metadata(config)

    assert public["authentication"] == "environment_credentials"
    assert public["security"] == "environment_security_string"
    serialized = str(public)
    assert "super-secret" not in serialized
    assert "operator" not in serialized


def test_opcua_browse_and_snapshot_use_explicit_transport_contract() -> None:
    source = _source()
    browsed = asyncio.run(browse_opcua_source(source, transport=FakeTransport()))
    snapshot = asyncio.run(read_opcua_snapshot(source, transport=FakeTransport()))

    assert browsed["count"] == 2
    assert {row["name"] for row in browsed["tags"]} == {"Temperature", "Pressure"}
    assert snapshot["sample_count"] == 2
    assert snapshot["good_count"] == 2
    assert {row["tag"] for row in snapshot["samples"]} == {"temperature", "pressure"}


def test_live_buffer_is_bounded_and_surfaces_drop_and_quality_counts() -> None:
    buffer = BoundedLiveBuffer(max_rows=2)
    buffer.append(LiveSample(tag="a", value=1.0, timestamp="2026-01-01T00:00:00Z"))
    buffer.append(
        LiveSample(
            tag="b",
            value=None,
            timestamp="2026-01-01T00:00:01Z",
            status="BadSensorFailure",
            status_good=False,
        )
    )
    buffer.append(LiveSample(tag="a", value=2.0, timestamp="2026-01-01T00:00:02Z"))

    snapshot = buffer.snapshot()
    assert snapshot["buffered_rows"] == 2
    assert snapshot["received_rows"] == 3
    assert snapshot["dropped_rows"] == 1
    assert snapshot["bad_quality_rows_buffered"] == 1
    assert snapshot["preserves_arrival_order"] is True
    assert snapshot["resampling"] is False
    assert snapshot["latest_by_tag"]["a"]["value"] == 2.0
