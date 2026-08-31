from __future__ import annotations

import argparse
from pathlib import Path

from .cross_regime import run_cross_regime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C-MAPSS cross-regime generalization")
    parser.add_argument("--config", type=Path, default=Path("configs/cmapss_cross_regime.yaml"))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    args = parser.parse_args()
    requested_models = args.models.split(",") if args.models else None
    print(run_cross_regime(args.config, requested_models))


if __name__ == "__main__":
    main()
