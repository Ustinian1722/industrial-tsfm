"""Product-facing platform primitives for IndusTSFM.

This package deliberately sits above the research runners. It provides stable
contracts for projects/tasks, data auditing, analytics, model routing,
connectors, model catalog metadata, anomaly diagnostics, and offline
application replay without changing the existing experiment protocols.
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
from .report import build_platform_report
from .runtime import DataReplayRuntime, ReplayBatch, ReplayConfig
from .shift_analysis import DistributionShiftRequest, analyze_distribution_shift

__all__ = [
    "AnomalyDetectionRequest",
    "ConnectorError",
    "DataReplayRuntime",
    "DataSourceKind",
    "DataSourceSpec",
    "DistributionShiftRequest",
    "LagAnalysisRequest",
    "LoadedDataSource",
    "ModelDescriptor",
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
    "build_platform_application",
    "build_platform_report",
    "discover_variables",
    "get_model_descriptor",
    "load_data_source",
    "model_catalog",
    "route_task",
    "run_pca_spe_anomaly_detection",
]
