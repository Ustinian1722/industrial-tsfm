from __future__ import annotations

import argparse
from pathlib import Path

from .cross_domain import run_cross_domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-domain industrial forecasting")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gas_turbine_cross_domain.yaml")
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    args = parser.parse_args()
    requested_models = args.models.split(",") if args.models else None
    print(run_cross_domain(args.config, requested_models))


if __name__ == "__main__":
    main()
