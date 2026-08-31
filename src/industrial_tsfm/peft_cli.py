from __future__ import annotations

import argparse
from pathlib import Path

from .peft_experiment import run_peft


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official TimesFM Transformers+PEFT protocol"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cmapss_fd001_timesfm_peft.yaml")
    )
    args = parser.parse_args()
    print(run_peft(args.config))


if __name__ == "__main__":
    main()
