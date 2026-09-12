from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .contracts import DataSourceSpec
from .opcua import AsyncuaTransport, OPCUAConnectorError, config_from_source, public_connection_metadata
from .runtime import BoundedLiveBuffer, LiveSample


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


@dataclass
class OPCUALiveState:
    status: str = "created"
    connected: bool = False
    connect_attempts: int = 0
    reconnects: int = 0
    samples_received: int = 0
    bad_quality_samples: int = 0
    started_at: str | None = None
    connected_at: str | None = None
    last_sample_at: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SubscriptionHandler:
    def __init__(self, runtime: "OPCUALiveRuntime") -> None:
        self.runtime = runtime

    def datachange_notification(self, node: Any, value: Any, data: Any) -> None:
        self.runtime._record_datachange(node, value, data)

    def event_notification(self, event: Any) -> None:
        del event

    def status_change_notification(self, status: Any) -> None:
        if status is not None:
            self.runtime.state.last_error = f"subscription_status:{status}"


class OPCUALiveRuntime:
    """Read-only OPC-UA subscription runtime with bounded buffering and reconnects.

    This runtime deliberately exposes no OPC-UA writes. It records source/server
    timestamps and status quality when the server provides them, preserves
    callback arrival order, and reconnects with bounded exponential backoff.
    """

    def __init__(
        self,
        source: DataSourceSpec,
        *,
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 10.0,
    ) -> None:
        self.source = source
        self.config = config_from_source(source)
        if not self.config.node_ids:
            raise OPCUAConnectorError(
                "live subscription requires metadata.node_ids mapping tag names to OPC-UA node ids"
            )
        if reconnect_initial_seconds <= 0 or reconnect_max_seconds <= 0:
            raise ValueError("reconnect delays must be positive")
        if reconnect_initial_seconds > reconnect_max_seconds:
            raise ValueError("reconnect_initial_seconds cannot exceed reconnect_max_seconds")
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.buffer = BoundedLiveBuffer(max_rows=self.config.max_buffer_rows)
        self.state = OPCUALiveState()
        self._node_to_tag = {node_id: tag for tag, node_id in self.config.node_ids.items()}

    def _record_datachange(self, node: Any, value: Any, data: Any) -> None:
        received = _utc_now()
        node_id = node.nodeid.to_string() if hasattr(node, "nodeid") else str(node)
        tag = self._node_to_tag.get(node_id, node_id)
        monitored = getattr(data, "monitored_item", None)
        data_value = getattr(monitored, "Value", None)
        status_obj = getattr(data_value, "StatusCode", None)
        status = str(status_obj) if status_obj is not None else "unknown"
        status_good = bool(status_obj.is_good()) if hasattr(status_obj, "is_good") else True
        source_timestamp = _iso(getattr(data_value, "SourceTimestamp", None))
        server_timestamp = _iso(getattr(data_value, "ServerTimestamp", None))
        sample = LiveSample(
            tag=tag,
            value=value,
            timestamp=source_timestamp or server_timestamp or received,
            status=status,
            status_good=status_good,
            node_id=node_id,
            source_timestamp=source_timestamp,
            server_timestamp=server_timestamp,
        )
        self.buffer.append(sample)
        self.state.samples_received += 1
        if not status_good:
            self.state.bad_quality_samples += 1
        self.state.last_sample_at = received

    async def _run_connected(self, stop_event: asyncio.Event) -> None:
        client = await AsyncuaTransport()._client(self.config)
        subscription = None
        handles: Any = None
        try:
            await client.connect()
            self.state.connected = True
            self.state.status = "running"
            self.state.connected_at = _utc_now()
            self.state.last_error = None
            handler = _SubscriptionHandler(self)
            subscription = await client.create_subscription(self.config.sampling_interval_ms, handler)
            nodes = [client.get_node(node_id) for node_id in self.config.node_ids.values()]
            handles = await subscription.subscribe_data_change(nodes)
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
        finally:
            self.state.connected = False
            if subscription is not None:
                try:
                    if handles is not None:
                        await subscription.unsubscribe(handles)
                except Exception:
                    pass
                try:
                    await subscription.delete()
                except Exception:
                    pass
            try:
                await client.disconnect()
            except Exception:
                pass

    async def run(self, stop_event: asyncio.Event) -> None:
        if self.state.status == "running":
            raise RuntimeError("OPC-UA runtime is already running")
        self.state.started_at = self.state.started_at or _utc_now()
        self.state.status = "starting"
        backoff = self.reconnect_initial_seconds
        ever_connected = False
        while not stop_event.is_set():
            self.state.connect_attempts += 1
            try:
                await self._run_connected(stop_event)
                if stop_event.is_set():
                    break
                if ever_connected:
                    self.state.reconnects += 1
                ever_connected = True
                backoff = self.reconnect_initial_seconds
            except asyncio.CancelledError:
                self.state.status = "cancelled"
                raise
            except Exception as exc:
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.status = "reconnecting"
                if ever_connected:
                    self.state.reconnects += 1
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)
        self.state.connected = False
        self.state.status = "stopped"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.opcua_live_runtime.v1",
            "source_name": self.source.name,
            "connection": public_connection_metadata(self.config),
            "read_only": True,
            "writes_enabled": False,
            "state": self.state.to_dict(),
            "buffer": self.buffer.snapshot(),
        }
