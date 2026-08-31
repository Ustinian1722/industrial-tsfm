from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from industrial_tsfm.evaluation import summarize_tradeoffs


def report_tradeoffs(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    result_table = pd.read_csv(run_dir / "result_table.csv")
    tradeoff = summarize_tradeoffs(result_table)
    output_path = run_dir / "tradeoff_summary.csv"
    tradeoff.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a lower-is-better model trade-off report")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(report_tradeoffs(args.run_dir))


if __name__ == "__main__":
    main()
