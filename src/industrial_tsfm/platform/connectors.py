from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contracts import DataSourceKind, DataSourceSpec


class ConnectorError(RuntimeError):
    """Raised when a product data source cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedDataSource:
    spec: DataSourceSpec
    frame: pd.DataFrame
    provenance: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_path(spec: DataSourceSpec) -> tuple[pd.DataFrame, dict[str, object]]:
    assert spec.location is not None
    path = Path(spec.location).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ConnectorError(f"data source path does not exist or is not a file: {path}")

    if spec.kind == DataSourceKind.CSV:
        frame = pd.read_csv(path)
    elif spec.kind == DataSourceKind.PARQUET:
        try:
            frame = pd.read_parquet(path)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ConnectorError(
                "Parquet loading requires a pandas parquet engine such as pyarrow or fastparquet"
            ) from exc
    else:
        raise ConnectorError(f"path loader does not support source kind {spec.kind.value}")

    return frame, {
        "resolved_location": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def load_data_source(
    spec: DataSourceSpec,
    *,
    memory_frames: Mapping[str, pd.DataFrame] | None = None,
    parse_timestamp: bool = True,
) -> LoadedDataSource:
    """Load one product data source without silently mutating its time order.

    V1 deliberately implements local CSV, local Parquet, and explicit in-memory
    sources. SQL, MQTT, and OPC-UA remain declared product contracts but fail
    closed until their connection/authentication semantics are implemented.
    """

    spec.validate()
    provenance: dict[str, object] = {
        "source_name": spec.name,
        "kind": spec.kind.value,
    }

    if spec.kind == DataSourceKind.MEMORY:
        frames = dict(memory_frames or {})
        if spec.name not in frames:
            raise ConnectorError(
                f"memory source {spec.name!r} was not supplied in memory_frames"
            )
        frame = frames[spec.name].copy()
        provenance["resolved_location"] = "memory"
        provenance["bytes"] = None
        provenance["sha256"] = None
    elif spec.kind in {DataSourceKind.CSV, DataSourceKind.PARQUET}:
        frame, path_provenance = _load_path(spec)
        provenance.update(path_provenance)
    else:
        raise ConnectorError(
            f"{spec.kind.value} is a declared connector contract but is not enabled in V1; "
            "use CSV/Parquet export or implement an authenticated connector adapter"
        )

    if frame.empty:
        raise ConnectorError(f"data source {spec.name!r} loaded an empty table")

    if parse_timestamp and spec.timestamp_column:
        if spec.timestamp_column not in frame.columns:
            raise ConnectorError(
                f"configured timestamp column {spec.timestamp_column!r} is missing"
            )
        frame = frame.copy()
        frame[spec.timestamp_column] = pd.to_datetime(
            frame[spec.timestamp_column], errors="coerce"
        )

    provenance.update(
        {
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_names": [str(column) for column in frame.columns],
        }
    )
    return LoadedDataSource(spec=spec, frame=frame, provenance=provenance)
