from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_tsfm.platform import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
    audit_dataframe,
    build_platform_report,
)


def build_fixture(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    time = pd.date_range("2026-01-01", periods=rows, freq="10min")
    load = 60 + 8 * np.sin(np.linspace(0, 10, rows)) + rng.normal(0, 1.0, rows)
    temperature = 40 + 0.25 * load + rng.normal(0, 0.8, rows)
    pressure = 2.1 + 0.015 * load + rng.normal(0, 0.03, rows)
    target = np.roll(temperature, -1)
    target[-1] = target[-2]
    frame = pd.DataFrame(
        {
            "timestamp": time,
            "load": load,
            "temperature": temperature,
            "pressure": pressure,
            "temperature_next": target,
            "dead_tag": 1.0,
        }
    )
    frame.loc[25:34, "pressure"] = np.nan
    return frame


def main() -> None:
    output_dir = Path("results/platform-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)

    project = ProjectSpec(
        name="Tennessee-style Process Forecast Demo",
        description="Synthetic process data used only to validate the product-facing platform path.",
        data_sources=(
            DataSourceSpec(
                name="synthetic-process",
                kind=DataSourceKind.MEMORY,
                timestamp_column="timestamp",
            ),
        ),
        tasks=(
            TaskDefinition(
                name="temperature-forecast",
                task_type=TaskType.FORECASTING,
                target_columns=("temperature_next",),
                context_length=32,
                horizon=8,
                business_kpi="keep process temperature inside the operating envelope",
            ),
        ),
        tags=("platform-smoke", "industrial-ai"),
    )

    frame = build_fixture()
    audit = audit_dataframe(
        frame,
        timestamp_column="timestamp",
        target_columns=("temperature_next",),
    )
    (output_dir / "project.json").write_text(
        json.dumps(project.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = build_platform_report(project, audit, output_dir / "index.html")
    print(f"readiness={audit['summary']['readiness_score']}")
    print(f"usable_features={len(audit['usable_numeric_features'])}")
    print(report)


if __name__ == "__main__":
    main()
