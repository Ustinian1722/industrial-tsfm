from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, write_json
from .selection import rank_model_candidates


@dataclass(frozen=True)
class AdaptationRequest:
    """Runtime constraints used by the lightweight automatic adaptation engine."""

    target_data_fraction: float
    shift_score: float = 0.0
    support_windows: int | None = None
    max_adaptation_seconds: float | None = None
    max_parameters: int | None = None
    max_inference_seconds: float | None = None
    min_calibration_windows: int = 2
    calibration_enabled: bool = True
    metric: str = "normalized_rmse"

    def validate(self) -> None:
        if not 0.0 <= self.target_data_fraction <= 1.0:
            raise ValueError("target_data_fraction must be between zero and one")
        if self.shift_score < 0.0:
            raise ValueError("shift_score must be non-negative")
        if self.support_windows is not None and self.support_windows < 0:
            raise ValueError("support_windows must be non-negative")
        if self.max_adaptation_seconds is not None and self.max_adaptation_seconds < 0:
            raise ValueError("max_adaptation_seconds must be non-negative")
        if self.max_parameters is not None and self.max_parameters < 0:
            raise ValueError("max_parameters must be non-negative")
        if self.max_inference_seconds is not None and self.max_inference_seconds < 0:
            raise ValueError("max_inference_seconds must be non-negative")
        if self.min_calibration_windows < 1:
            raise ValueError("min_calibration_windows must be positive")


def _training_mode(row: pd.Series) -> str:
    return str(row.get("training_mode", "unknown"))


def _default_support_windows(request: AdaptationRequest) -> int:
    if request.support_windows is not None:
        return int(request.support_windows)
    if request.target_data_fraction <= 0.0:
        return 0
    # This is only a conservative policy fallback. The exact support count is
    # recorded by the adaptation runner when real target windows are built.
    return int(np.ceil(request.target_data_fraction * 100.0))


def _strategy_for_mode(
    training_mode: str,
    request: AdaptationRequest,
    estimated_support_windows: int,
) -> tuple[str, str]:
    if request.target_data_fraction <= 0.0 or estimated_support_windows <= 0:
        return "zero_shot", "no usable target support windows are available"
    if training_mode == "peft":
        return "target_peft", "the selected model exposes a PEFT adaptation path"
    if training_mode == "supervised":
        return "few_shot_finetune", "the selected model exposes a supervised adaptation path"
    if (
        request.calibration_enabled
        and estimated_support_windows >= request.min_calibration_windows
    ):
        if request.shift_score >= 1.0:
            return (
                "target_residual_calibration",
                "large source-standardized shift and sufficient observed support favor calibration",
            )
        return (
            "target_residual_calibration",
            "sufficient observed support is available for residual calibration",
        )
    return "target_scaling", "use frozen inference with target-support scaling under the support budget"


def _strategy_cost(
    strategy: str,
    strategy_costs: dict[str, dict[str, float]] | None,
) -> dict[str, float | None]:
    profile = (strategy_costs or {}).get(strategy, {})
    return {
        "estimated_adaptation_seconds": (
            float(profile["seconds"]) if "seconds" in profile else None
        ),
        "estimated_memory_mb": float(profile["memory_mb"]) if "memory_mb" in profile else None,
    }


def build_adaptation_plan(
    validation_table: pd.DataFrame,
    request: AdaptationRequest,
    model_constraints: dict[str, Any] | None = None,
    strategy_costs: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """Select a model from validation evidence and a strategy from runtime constraints.

    The model rank is empirical (validation only). The strategy is a transparent
    policy over the selected model's training contract, observed support amount,
    shift score, and optional measured strategy costs. It never reads target
    labels or final target metrics.
    """
    request.validate()
    constraints = dict(model_constraints or {})
    if request.max_parameters is not None:
        constraints["max_parameters"] = request.max_parameters
    if request.max_inference_seconds is not None:
        constraints["max_inference_seconds"] = request.max_inference_seconds
    ranking = rank_model_candidates(validation_table, metric=request.metric, constraints=constraints)
    selected_model = str(ranking.iloc[0]["model"])
    selected_row = validation_table[
        (validation_table["model"].astype(str) == selected_model)
        & (validation_table["status"].astype(str) == "complete")
    ].iloc[0]
    mode = _training_mode(selected_row)
    support_windows = _default_support_windows(request)
    strategy, reason = _strategy_for_mode(mode, request, support_windows)
    costs = _strategy_cost(strategy, strategy_costs)
    budget_status = "not_configured"
    if request.max_adaptation_seconds is not None:
        estimate = costs["estimated_adaptation_seconds"]
        budget_status = "unknown_cost" if estimate is None else (
            "within_budget" if estimate <= request.max_adaptation_seconds else "over_budget"
        )
        if budget_status == "over_budget" and strategy != "zero_shot":
            fallback_strategy = "zero_shot" if request.target_data_fraction <= 0.0 else "target_scaling"
            fallback_costs = _strategy_cost(fallback_strategy, strategy_costs)
            strategy = fallback_strategy
            reason = f"selected strategy exceeded adaptation budget; fallback to {fallback_strategy}"
            costs = fallback_costs
    selected_metric = float(ranking.iloc[0][f"{request.metric}_mean"])
    return {
        "engine": "industrial_tsfm_adaptation_engine_v1",
        "selection_evidence": "validation_only",
        "target_labels_used": False,
        "request": asdict(request),
        "selected_model": selected_model,
        "selected_training_mode": mode,
        "selected_validation_metric": selected_metric,
        "selected_strategy": strategy,
        "strategy_reason": reason,
        "estimated_support_windows": support_windows,
        "strategy_cost": costs,
        "adaptation_budget_status": budget_status,
        "model_ranking": ranking.to_dict(orient="records"),
    }


def _shift_score_from_artifact(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in payload:
        aggregate = payload["aggregate"]
        return float(aggregate.get("rms_standardized_mean_shift", 0.0))
    values = [
        float(item.get("rms_standardized_mean_shift", 0.0))
        for item in payload.values()
        if isinstance(item, dict) and "aggregate" in item
    ]
    return float(np.mean(values)) if values else 0.0


def plan_from_run(
    run_dir: str | Path,
    request: AdaptationRequest,
    output_path: str | Path | None = None,
) -> Path:
    """Build and persist a decision from an existing validated run directory."""
    run_dir = Path(run_dir).resolve()
    validation_path = run_dir / "validation_result_table.csv"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation evidence: {validation_path}")
    config_path = run_dir / "config.resolved.yaml"
    config = load_config(config_path) if config_path.exists() else {}
    if request.shift_score == 0.0:
        shift_path = run_dir / "shift_summary.json"
        if shift_path.exists():
            request = AdaptationRequest(
                **{
                    **asdict(request),
                    "shift_score": _shift_score_from_artifact(shift_path),
                }
            )
    strategy_config = config.get("adaptation", {}).get("strategy_selection", {})
    costs = strategy_config.get("strategy_costs", {})
    plan = build_adaptation_plan(
        pd.read_csv(validation_path),
        request,
        model_constraints=config.get("selection", {}).get("constraints", {}),
        strategy_costs=costs,
    )
    plan["run_dir"] = str(run_dir)
    plan["config_path"] = str(config_path) if config_path.exists() else None
    plan["source_shift_artifact"] = (
        str(run_dir / "shift_summary.json") if (run_dir / "shift_summary.json").exists() else None
    )
    output_path = Path(output_path) if output_path is not None else run_dir / "engine_decision.json"
    write_json(output_path, plan)
    return output_path
