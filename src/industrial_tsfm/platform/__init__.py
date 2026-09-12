"""Product-facing platform primitives for IndusTSFM.

This package deliberately sits above the research runners. It provides stable
contracts for projects/tasks, data auditing, model routing, connectors, and
offline application replay without changing the existing experiment protocols.
"""

from .connectors import ConnectorError, LoadedDataSource, load_data_source
from .contracts import DataSourceKind, DataSourceSpec, ProjectSpec, TaskDefinition, TaskType
from .data_audit import audit_dataframe
from .model_router import ProductRoutingRequest, route_task
from .report import build_platform_report
from .runtime import DataReplayRuntime, ReplayBatch, ReplayConfig

__all__ = [
    "ConnectorError",
    "DataReplayRuntime",
    "DataSourceKind",
    "DataSourceSpec",
    "LoadedDataSource",
    "ProductRoutingRequest",
    "ProjectSpec",
    "ReplayBatch",
    "ReplayConfig",
    "TaskDefinition",
    "TaskType",
    "audit_dataframe",
    "build_platform_report",
    "load_data_source",
    "route_task",
]
