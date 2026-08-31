from __future__ import annotations

import argparse
from pathlib import Path

from .adaptation import run_adaptation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase-2 target adaptation experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/cmapss_fd001_adaptation.yaml"))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names")
    args = parser.parse_args()
    requested_models = args.models.split(",") if args.models else None
    output = run_adaptation(args.config, requested_models)
    print(output)


if __name__ == "__main__":
    main()
