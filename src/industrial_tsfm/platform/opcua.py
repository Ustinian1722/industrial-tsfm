from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

from .contracts import DataSourceKind, DataSourceSpec


class OPCUAConnectorError(RuntimeError):
    """Raised when an OPC-UA source cannot be configured or queried safely."""


@dataclass(frozen=True)
class OPCUAConnectionConfig:
    endpoint: str
    browse_root: str = "i=85"  # OPC UA Objects folder
    node_ids: dict[str, str] | None = None
    username_env: str | None = None
    password_env: str | None = None
    security_string_env: str | None = None
    request_timeout_seconds: float = 8.0
    max_browse_depth: int = 4
    max_browse_nodes: int = 500
    sampling_interval_ms: int = 1000
    max_buffer_rows: int = 4096

    def validate(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "opc.tcp" or not parsed.hostname:
            raise OPCUAConnectorError("OPC-UA endpoint must use opc.tcp:// and include a host")
        if self.request_timeout_seconds <= 0:
            raise OPCUAConnectorError("request_timeout_seconds must be positive")
        if self.max_browse_depth < 0:
            raise OPCUAConnectorError("max_browse_depth must be non-negative")
        if self.max_browse_nodes < 1:
            raise OPCUAConnectorError("max_browse_nodes must be positive")
        if self.sampling_interval_ms < 50:
            raise OPCUAConnectorError("sampling_interval_ms must be at least 50 ms")
        if self.max_buffer_rows < 1:
            raise OPCUAConnectorError("max_buffer_rows must be positive")
        if self.node_ids is not None:
            for tag, node_id in self.node_ids.items():
                if not str(tag).strip() or not str(node_id).strip():
                    raise OPCUAConnectorError("configured OPC-UA tag aliases and node ids must be non-empty")
        if bool(self.username_env) != bool(self.password_env):
            raise OPCUAConnectorError(
                "username_env and password_env must either both be configured or both be omitted"
            )


@dataclass(frozen=True)
class OPCUATag:
    name: str
    node_id: str
    browse_name: str | None = None
    node_class: str | None = None
    parent_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OPCUASample:
    tag: str
    node_id: str
    value: Any
    source_timestamp: str | None
    server_timestamp: str | None
    received_timestamp: str
    status: str
    status_good: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OPCUATransport(Protocol):
    async def browse(self, config: OPCUAConnectionConfig) -> list[OPCUATag]: ...

    async def read_snapshot(
        self,
        config: OPCUAConnectionConfig,
        nodes: dict[str, str],
    ) -> list[OPCUASample]: ...


def config_from_source(spec: DataSourceSpec) -> OPCUAConnectionConfig:
    spec.validate()
    if spec.kind != DataSourceKind.OPCUA:
        raise OPCUAConnectorError("source kind must be opcua")
    assert spec.location is not None
    metadata = dict(spec.metadata)
    raw_nodes = metadata.get("node_ids")
    node_ids: dict[str, str] | None = None
    if raw_nodes is not None:
        if not isinstance(raw_nodes, dict):
            raise OPCUAConnectorError("metadata.node_ids must be an object mapping tag names to node ids")
        node_ids = {str(key): str(value) for key, value in raw_nodes.items()}

    config = OPCUAConnectionConfig(
        endpoint=spec.location,
        browse_root=str(metadata.get("browse_root", "i=85")),
        node_ids=node_ids,
        username_env=(str(metadata["username_env"]) if metadata.get("username_env") else None),
        password_env=(str(metadata["password_env"]) if metadata.get("password_env") else None),
        security_string_env=(
            str(metadata["security_string_env"]) if metadata.get("security_string_env") else None
        ),
        request_timeout_seconds=float(metadata.get("request_timeout_seconds", 8.0)),
        max_browse_depth=int(metadata.get("max_browse_depth", 4)),
        max_browse_nodes=int(metadata.get("max_browse_nodes", 500)),
        sampling_interval_ms=int(metadata.get("sampling_interval_ms", 1000)),
        max_buffer_rows=int(metadata.get("max_buffer_rows", 4096)),
    )
    config.validate()
    return config


def _environment_value(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise OPCUAConnectorError(f"required OPC-UA environment variable {name!r} is not set")
    return value


def resolved_auth(config: OPCUAConnectionConfig) -> dict[str, str | None]:
    """Resolve credentials at runtime without ever returning the environment-variable names as secrets."""

    config.validate()
    username = _environment_value(config.username_env)
    password = _environment_value(config.password_env)
    security_string = _environment_value(config.security_string_env)
    return {
        "username": username,
        "password": password,
        "security_string": security_string,
    }


def public_connection_metadata(config: OPCUAConnectionConfig) -> dict[str, Any]:
    """Return a persistence-safe connection description with no credential values."""

    return {
        "endpoint": config.endpoint,
        "browse_root": config.browse_root,
        "configured_node_count": len(config.node_ids or {}),
        "authentication": "environment_credentials" if config.username_env else "anonymous",
        "security": "environment_security_string" if config.security_string_env else "default",
        "request_timeout_seconds": config.request_timeout_seconds,
        "max_browse_depth": config.max_browse_depth,
        "max_browse_nodes": config.max_browse_nodes,
        "sampling_interval_ms": config.sampling_interval_ms,
        "max_buffer_rows": config.max_buffer_rows,
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


class AsyncuaTransport:
    """Lazy optional asyncua transport used by the live product layer.

    The dependency is imported only when an OPC-UA operation is requested, so
    the research/CPU-only package remains usable without industrial connector extras.
    """

    @staticmethod
    def _client_type():
        try:
            from asyncua import Client
        except ImportError as exc:
            raise OPCUAConnectorError(
                "live OPC-UA support requires the optional dependency; install industrial-tsfm[opcua]"
            ) from exc
        return Client

    @staticmethod
    def _configure_client(client: Any, config: OPCUAConnectionConfig) -> None:
        auth = resolved_auth(config)
        if auth["username"] is not None:
            client.set_user(auth["username"])
            client.set_password(auth["password"])
        if auth["security_string"] is not None:
            result = client.set_security_string(auth["security_string"])
            if hasattr(result, "__await__"):
                raise OPCUAConnectorError(
                    "the installed asyncua version exposes asynchronous security setup; "
                    "configure certificates with a compatible asyncua release"
                )

    async def _client(self, config: OPCUAConnectionConfig):
        Client = self._client_type()
        client = Client(url=config.endpoint, timeout=config.request_timeout_seconds)
        auth = resolved_auth(config)
        if auth["username"] is not None:
            client.set_user(auth["username"])
            client.set_password(auth["password"])
        if auth["security_string"] is not None:
            await client.set_security_string(auth["security_string"])
        return client

    async def browse(self, config: OPCUAConnectionConfig) -> list[OPCUATag]:
        config.validate()
        client = await self._client(config)
        rows: list[OPCUATag] = []
        async with client:
            root = client.get_node(config.browse_root)
            queue: list[tuple[Any, int, str | None]] = [(root, 0, None)]
            seen: set[str] = set()
            while queue and len(rows) < config.max_browse_nodes:
                node, depth, parent_node_id = queue.pop(0)
                node_id = node.nodeid.to_string()
                if node_id in seen:
                    continue
                seen.add(node_id)
                try:
                    browse_name_obj = await node.read_browse_name()
                    node_class_obj = await node.read_node_class()
                    browse_name = str(getattr(browse_name_obj, "Name", browse_name_obj))
                    node_class = str(node_class_obj)
                except Exception as exc:  # server ACLs may deny metadata on individual nodes
                    rows.append(
                        OPCUATag(
                            name=node_id,
                            node_id=node_id,
                            browse_name=None,
                            node_class=f"unreadable:{type(exc).__name__}",
                            parent_node_id=parent_node_id,
                        )
                    )
                    continue
                rows.append(
                    OPCUATag(
                        name=browse_name,
                        node_id=node_id,
                        browse_name=browse_name,
                        node_class=node_class,
                        parent_node_id=parent_node_id,
                    )
                )
                if depth >= config.max_browse_depth:
                    continue
                try:
                    children = await node.get_children()
                except Exception:
                    children = []
                for child in children:
                    queue.append((child, depth + 1, node_id))
        return rows

    async def read_snapshot(
        self,
        config: OPCUAConnectionConfig,
        nodes: dict[str, str],
    ) -> list[OPCUASample]:
        config.validate()
        if not nodes:
            raise OPCUAConnectorError("at least one OPC-UA node must be configured for snapshot reads")
        client = await self._client(config)
        received = datetime.now(timezone.utc).isoformat()
        rows: list[OPCUASample] = []
        async with client:
            for tag, node_id in nodes.items():
                node = client.get_node(node_id)
                try:
                    value = await node.read_data_value()
                    status_obj = getattr(value, "StatusCode", None)
                    status = str(status_obj) if status_obj is not None else "unknown"
                    is_good = bool(status_obj.is_good()) if hasattr(status_obj, "is_good") else True
                    variant = getattr(value, "Value", None)
                    raw = getattr(variant, "Value", variant)
                    rows.append(
                        OPCUASample(
                            tag=tag,
                            node_id=node_id,
                            value=raw,
                            source_timestamp=_iso(getattr(value, "SourceTimestamp", None)),
                            server_timestamp=_iso(getattr(value, "ServerTimestamp", None)),
                            received_timestamp=received,
                            status=status,
                            status_good=is_good,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        OPCUASample(
                            tag=tag,
                            node_id=node_id,
                            value=None,
                            source_timestamp=None,
                            server_timestamp=None,
                            received_timestamp=received,
                            status=f"read_error:{type(exc).__name__}",
                            status_good=False,
                        )
                    )
        return rows


async def browse_opcua_source(
    spec: DataSourceSpec,
    *,
    transport: OPCUATransport | None = None,
) -> dict[str, Any]:
    config = config_from_source(spec)
    active = transport or AsyncuaTransport()
    tags = await active.browse(config)
    return {
        "schema_version": "industrial_tsfm.opcua_browse.v1",
        "source_name": spec.name,
        "connection": public_connection_metadata(config),
        "count": len(tags),
        "tags": [tag.to_dict() for tag in tags],
    }


async def read_opcua_snapshot(
    spec: DataSourceSpec,
    *,
    node_ids: dict[str, str] | None = None,
    transport: OPCUATransport | None = None,
) -> dict[str, Any]:
    config = config_from_source(spec)
    nodes = dict(node_ids or config.node_ids or {})
    if not nodes:
        raise OPCUAConnectorError(
            "no OPC-UA node ids are configured; browse first and persist an explicit tag mapping"
        )
    active = transport or AsyncuaTransport()
    samples = await active.read_snapshot(config, nodes)
    return {
        "schema_version": "industrial_tsfm.opcua_snapshot.v1",
        "source_name": spec.name,
        "connection": public_connection_metadata(config),
        "sample_count": len(samples),
        "good_count": sum(int(sample.status_good) for sample in samples),
        "samples": [sample.to_dict() for sample in samples],
    }
