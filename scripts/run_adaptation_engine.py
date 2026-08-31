from __future__ import annotations

import argparse
from pathlib import Path

from industrial_tsfm.adaptation_engine import AdaptationRequest, plan_from_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validation/shift/budget adaptation plan")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--target-fraction", type=float, default=0.10)
    parser.add_argument("--support-windows", type=int, default=None)
    parser.add_argument("--shift-score", type=float, default=0.0)
    parser.add_argument("--max-adaptation-seconds", type=float, default=None)
    parser.add_argument("--max-parameters", type=int, default=None)
    parser.add_argument("--max-inference-seconds", type=float, default=None)
    parser.add_argument("--min-calibration-windows", type=int, default=2)
    parser.add_argument("--disable-calibration", action="store_true")
    args = parser.parse_args()
    request = AdaptationRequest(
        target_data_fraction=args.target_fraction,
        shift_score=args.shift_score,
        support_windows=args.support_windows,
        max_adaptation_seconds=args.max_adaptation_seconds,
        max_parameters=args.max_parameters,
        max_inference_seconds=args.max_inference_seconds,
        min_calibration_windows=args.min_calibration_windows,
        calibration_enabled=not args.disable_calibration,
    )
    print(plan_from_run(args.run_dir, request))


if __name__ == "__main__":
    main()
