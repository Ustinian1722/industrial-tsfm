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
from .forecasting import (
    DeterministicPersistenceForecastModel,
    ForecastingRecord,
    ForecastingRequest,
    ForecastModelRuntimeAdapter,
    ForecastPoint,
    LiveForecastingApplication,
    RuntimeForecastModel,
)
from .forecasting_deployment import (
    ForecastingDeploymentSpec,
    build_forecasting_deployment_spec,
)
from .forecasting_registry import LiveForecastingRegistry
from .forecasting_uq import (
    ConformalForecastArtifact,
    apply_conformal_forecast,
    calibrate_conformal_forecast,
)
from .model_catalog import ModelDescriptor, get_model_descriptor, model_catalog
from .model_router import ProductRoutingRequest, route_task
from .online_anomaly import (
    OnlinePCASPEAnomalyDetector,
    PCASPEArtifact,
    fit_pca_spe_artifact,
    score_pca_spe_artifact,
)
from .opcua import (
    OPCUAConnectionConfig,
    OPCUAConnectorError,
    OPCUASample,
    OPCUATag,
    browse_opcua_source,
    read_opcua_snapshot,
)
from .opcua import config_from_source as opcua_config_from_source
from .opcua import public_connection_metadata as opcua_public_connection_metadata
from .opcua_registry import OPCUALiveRuntimeRegistry
from .opcua_runtime import OPCUALiveRuntime, OPCUALiveState
from .report import build_platform_report
from .runtime import (
    BoundedLiveBuffer,
    DataReplayRuntime,
    LiveSample,
    ReplayBatch,
    ReplayConfig,
    TimeSeriesWindow,
    WindowProvider,
)
from .shift_analysis import (
    CrossSourceShiftRequest,
    DistributionShiftRequest,
    analyze_distribution_shift,
    compare_source_distributions,
)
from .streaming import (
    DeterministicLastValuePredictor,
    PredictionRecord,
    Predictor,
    StreamingInferenceEngine,
)

__all__ = [
    "AnomalyDetectionRequest",
    "BoundedLiveBuffer",
    "ConformalForecastArtifact",
    "ConnectorError",
    "CrossSourceShiftRequest",
    "DataReplayRuntime",
    "DataSourceKind",
    "DataSourceSpec",
    "DeterministicLastValuePredictor",
    "DeterministicPersistenceForecastModel",
    "DistributionShiftRequest",
    "ForecastModelRuntimeAdapter",
    "ForecastPoint",
    "ForecastingDeploymentSpec",
    "ForecastingRecord",
    "ForecastingRequest",
    "LagAnalysisRequest",
    "LiveForecastingApplication",
    "LiveForecastingRegistry",
    "LiveSample",
    "LoadedDataSource",
    "ModelDescriptor",
    "OPCUAConnectionConfig",
    "OPCUAConnectorError",
    "OPCUALiveRuntime",
    "OPCUALiveRuntimeRegistry",
    "OPCUALiveState",
    "OPCUASample",
    "OPCUATag",
    "OnlinePCASPEAnomalyDetector",
    "PCASPEArtifact",
    "PlatformApplication",
    "PredictionRecord",
    "Predictor",
    "ProductRoutingRequest",
    "ProjectSpec",
    "ReplayBatch",
    "ReplayConfig",
    "RuntimeForecastModel",
    "StreamingInferenceEngine",
    "TaskDefinition",
    "TaskType",
    "TimeSeriesWindow",
    "VariableDiscoveryRequest",
    "WindowProvider",
    "analyze_distribution_shift",
    "analyze_lagged_relationships",
    "apply_conformal_forecast",
    "audit_dataframe",
    "browse_opcua_source",
    "build_forecasting_deployment_spec",
    "build_platform_application",
    "build_platform_report",
    "calibrate_conformal_forecast",
    "compare_source_distributions",
    "discover_variables",
    "fit_pca_spe_artifact",
    "get_model_descriptor",
    "load_data_source",
    "model_catalog",
    "opcua_config_from_source",
    "opcua_public_connection_metadata",
    "read_opcua_snapshot",
    "route_task",
    "run_pca_spe_anomaly_detection",
    "score_pca_spe_artifact",
]
