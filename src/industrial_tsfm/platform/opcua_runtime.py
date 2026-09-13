from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .contracts import DataSourceSpec
from .opcua import (
    AsyncuaTransport,
    OPCUAConnectorError,
    config_from_source,
    public_connection_metadata,
)
from .runtime import BoundedLiveBuffer, LiveSample, TimeSeriesWindow

_LOGGER = logging.getLogger(__name__)


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
    subscribed_tag_count: int = 0
    connect_attempts: int = 0
    reconnects: int = 0
    disconnects: int = 0
    samples_received: int = 0
    bad_quality_samples: int = 0
    data_gap_seconds: float = 0.0
    last_gap_seconds: float | None = None
    started_at: str | None = None
    connected_at: str | None = None
    last_sample_at: str | None = None
    last_disconnect_at: str | None = None
    last_reconnected_at: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SubscriptionHandler:
    def __init__(self, runtime: OPCUALiveRuntime) -> None:
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
        self._ever_connected = False
        self._disconnect_monotonic: float | None = None
        self._state_transitions: list[dict[str, str]] = [
            {"at": _utc_now(), "from": "none", "to": "created"}
        ]

    def _set_status(self, status: str) -> None:
        if self.state.status == status:
            return
        previous = self.state.status
        self.state.status = status
        self._state_transitions.append({"at": _utc_now(), "from": previous, "to": status})
        if len(self._state_transitions) > 64:
            self._state_transitions = self._state_transitions[-64:]

    def _mark_disconnect(self) -> None:
        if self._disconnect_monotonic is not None:
            return
        self._disconnect_monotonic = time.monotonic()
        self.state.disconnects += 1
        self.state.last_disconnect_at = _utc_now()

    def _mark_subscription_ready(self) -> None:
        now = _utc_now()
        self.state.connected = True
        self.state.subscribed_tag_count = len(self.config.node_ids or {})
        self.state.connected_at = now
        if self._ever_connected and self._disconnect_monotonic is not None:
            gap = max(0.0, time.monotonic() - self._disconnect_monotonic)
            self.state.reconnects += 1
            self.state.last_gap_seconds = float(gap)
            self.state.data_gap_seconds += float(gap)
            self.state.last_reconnected_at = now
        self._disconnect_monotonic = None
        self._ever_connected = True
        self.state.last_error = None
        self._set_status("running")

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

    async def _check_connection(self, client: Any) -> None:
        checker = getattr(client, "check_connection", None)
        if checker is not None:
            await checker()
            return
        await client.nodes.server_state.read_value()

    async def _run_connected(self, stop_event: asyncio.Event) -> None:
        client = await AsyncuaTransport()._client(self.config)
        subscription = None
        handles: Any = None
        try:
            await client.connect()
            handler = _SubscriptionHandler(self)
            subscription = await client.create_subscription(self.config.sampling_interval_ms, handler)
            nodes = [client.get_node(node_id) for node_id in self.config.node_ids.values()]
            handles = await subscription.subscribe_data_change(nodes)
            self._mark_subscription_ready()
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    await self._check_connection(client)
        finally:
            self.state.connected = False
            self.state.subscribed_tag_count = 0
            if subscription is not None:
                try:
                    if handles is not None:
                        await subscription.unsubscribe(handles)
                except Exception as exc:  # noqa: BLE001 - cleanup is best effort after link loss
                    _LOGGER.debug("OPC-UA unsubscribe cleanup failed: %r", exc)
                try:
                    await subscription.delete()
                except Exception as exc:  # noqa: BLE001 - cleanup is best effort after link loss
                    _LOGGER.debug("OPC-UA subscription delete cleanup failed: %r", exc)
            try:
                await client.disconnect()
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort after link loss
                _LOGGER.debug("OPC-UA disconnect cleanup failed: %r", exc)

    async def run(self, stop_event: asyncio.Event) -> None:
        if self.state.status == "running":
            raise RuntimeError("OPC-UA runtime is already running")
        self.state.started_at = self.state.started_at or _utc_now()
        self._set_status("starting")
        backoff = self.reconnect_initial_seconds
        while not stop_event.is_set():
            self.state.connect_attempts += 1
            try:
                await self._run_connected(stop_event)
                if stop_event.is_set():
                    break
                backoff = self.reconnect_initial_seconds
            except asyncio.CancelledError:
                self._set_status("cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect boundary must contain transport failures
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                if self._ever_connected:
                    self._mark_disconnect()
                self._set_status("reconnecting")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2.0, self.reconnect_max_seconds)
        self.state.connected = False
        self.state.subscribed_tag_count = 0
        self._set_status("stopped")

    def materialize_window(self, max_observations: int | None = None) -> TimeSeriesWindow:
        """Expose the standard source-agnostic runtime window contract."""

        return self.buffer.materialize_window(max_observations)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "industrial_tsfm.opcua_live_runtime.v1",
            "source_name": self.source.name,
            "connection": public_connection_metadata(self.config),
            "read_only": True,
            "writes_enabled": False,
            "state": self.state.to_dict(),
            "state_transitions": list(self._state_transitions),
            "buffer": self.buffer.snapshot(),
        }
