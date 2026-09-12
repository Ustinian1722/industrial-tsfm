from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    FORECASTING = "forecasting"
    ANOMALY = "anomaly_detection"
    DIAGNOSIS = "fault_diagnosis"
    RUL = "remaining_useful_life"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class DataSourceKind(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    SQL = "sql"
    MQTT = "mqtt"
    OPCUA = "opcua"
    MEMORY = "memory"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    kind: DataSourceKind
    location: str | None = None
    timestamp_column: str | None = None
    entity_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("data source name must be non-empty")
        if self.kind != DataSourceKind.MEMORY and not self.location:
            raise ValueError(f"location is required for data source kind {self.kind.value}")


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    task_type: TaskType
    target_columns: tuple[str, ...] = ()
    context_length: int | None = None
    horizon: int | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    business_kpi: str | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("task name must be non-empty")
        if self.context_length is not None and self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.task_type == TaskType.FORECASTING and not self.target_columns:
            raise ValueError("forecasting tasks require at least one target column")


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    description: str = ""
    data_sources: tuple[DataSourceSpec, ...] = ()
    tasks: tuple[TaskDefinition, ...] = ()
    tags: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("project name must be non-empty")
        if not self.data_sources:
            raise ValueError("project requires at least one data source")
        if not self.tasks:
            raise ValueError("project requires at least one task")
        source_names = [source.name for source in self.data_sources]
        task_names = [task.name for task in self.tasks]
        if len(set(source_names)) != len(source_names):
            raise ValueError("data source names must be unique within a project")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique within a project")
        for source in self.data_sources:
            source.validate()
        for task in self.tasks:
            task.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))
