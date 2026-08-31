from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = (
    "analysis_summary.json",
    "feature_error_summary.csv",
    "horizon_error_summary.csv",
    "entity_error_summary.csv",
    "ood_window_summary.csv",
    "ood_feature_summary.csv",
    "context_representation_summary.csv",
    "error_by_ood_bin.csv",
    "reliability_summary.csv",
    "reliability_feature_summary.csv",
)


def audit_analysis(run_dir: str | Path) -> None:
    run_dir = Path(run_dir).resolve()
    analysis_dir = run_dir / "analysis"
    summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    if not summary.get("artifacts"):
        raise AssertionError("Analysis summary does not list artifacts")
    frames: dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_FILES:
        path = analysis_dir / filename
        if not path.exists():
            raise AssertionError(f"Missing analysis artifact: {filename}")
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            if frame.empty:
                raise AssertionError(f"Analysis artifact is empty: {filename}")
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
                raise AssertionError(f"Non-finite numeric value in {filename}")
            frames[filename] = frame
    reliability = frames["reliability_summary.csv"]
    if ((reliability["coverage"] < 0) | (reliability["coverage"] > 1)).any():
        raise AssertionError("Reliability coverage is outside [0, 1]")
    if (reliability["nominal_coverage"] <= 0).any() or (reliability["nominal_coverage"] >= 1).any():
        raise AssertionError("Invalid nominal reliability coverage")
    print(f"ANALYSIS_AUDIT_OK {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit post-run analysis artifacts")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    audit_analysis(args.run_dir)


if __name__ == "__main__":
    main()
