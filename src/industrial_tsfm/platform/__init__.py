"""Product-facing platform primitives for IndusTSFM.

This package deliberately sits above the research runners.  It provides stable
contracts for projects/tasks and deployment-oriented data auditing without
changing the existing experiment protocols.
"""

from .contracts import DataSourceKind, DataSourceSpec, ProjectSpec, TaskDefinition, TaskType
from .data_audit import audit_dataframe
from .report import build_platform_report

__all__ = [
    "DataSourceKind",
    "DataSourceSpec",
    "ProjectSpec",
    "TaskDefinition",
    "TaskType",
    "audit_dataframe",
    "build_platform_report",
]
