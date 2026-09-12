"""Product-facing platform primitives for IndusTSFM.

This package deliberately sits above the research runners. It provides stable
contracts for projects/tasks, data auditing, analytics, model routing,
connectors, model catalog metadata, anomaly diagnostics, deterministic replay,
and opt-in live industrial connector primitives without changing the existing
experiment protocols.
"""

from .analytics import LagAnalysisRequest, analyze_lagged_relationships
from .anomaly_engine import AnomalyDetectionRequest, run_pca_spe_anomaly_detection
from .application import PlatformApplication, build_platform_application
from .connectors import ConnectorError, LoadedDataSource, load_data_source
from .contracts import DataSourceKind, DataSourceSpec, ProjectSpec, TaskDefinition, TaskType
from .data_audit import audit_dataframe
from .feature_discovery import VariableDiscoveryRequest, discover_variables
from .model_catalog import ModelDescriptor, get_model_descriptor, model_catalog
from .model_router import ProductRoutingRequest, route_task
from .opcua import (
    OPCUAConnectionConfig,
    OPCUAConnectorError,
    OPCUASample,
    OPCUATag,
    browse_opcua_source,
    config_from_source as opcua_config_from_source,
    public_connection_metadata as opcua_public_connection_metadata,
    read_opcua_snapshot,
)
from .opcua_runtime import OPCUALiveRuntime, OPCUALiveState
from .report import build_platform_report
from .runtime import BoundedLiveBuffer, DataReplayRuntime, LiveSample, ReplayBatch, ReplayConfig
from .shift_analysis import (
    CrossSourceShiftRequest,
    DistributionShiftRequest,
    analyze_distribution_shift,
    compare_source_distributions,
)

__all__ = [
    "AnomalyDetectionRequest",
    "BoundedLiveBuffer",
    "ConnectorError",
    "CrossSourceShiftRequest",
    "DataReplayRuntime",
    "DataSourceKind",
    "DataSourceSpec",
    "DistributionShiftRequest",
    "LagAnalysisRequest",
    "LiveSample",
    "LoadedDataSource",
    "ModelDescriptor",
    "OPCUAConnectionConfig",
    "OPCUAConnectorError",
    "OPCUALiveRuntime",
    "OPCUALiveState",
    "OPCUASample",
    "OPCUATag",
    "PlatformApplication",
    "ProductRoutingRequest",
    "ProjectSpec",
    "ReplayBatch",
    "ReplayConfig",
    "TaskDefinition",
    "TaskType",
    "VariableDiscoveryRequest",
    "analyze_distribution_shift",
    "analyze_lagged_relationships",
    "audit_dataframe",
    "browse_opcua_source",
    "build_platform_application",
    "build_platform_report",
    "compare_source_distributions",
    "discover_variables",
    "get_model_descriptor",
    "load_data_source",
    "model_catalog",
    "opcua_config_from_source",
    "opcua_public_connection_metadata",
    "read_opcua_snapshot",
    "route_task",
    "run_pca_spe_anomaly_detection",
]
