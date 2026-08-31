import numpy as np
import pandas as pd

from industrial_tsfm.adaptation_engine import AdaptationRequest, build_adaptation_plan
from industrial_tsfm.evaluation import compute_distribution_shift, summarize_tradeoffs
from industrial_tsfm.selection import (
    compute_generalization_gap,
    recommend_adaptation_strategy,
    select_best_model,
    summarize_generalization_gap,
)


def test_selection_uses_validation_and_constraints() -> None:
    validation = pd.DataFrame(
        [
            {"model": "naive", "seed": 42, "status": "complete", "normalized_rmse": 2.0, "training_mode": "none", "total_parameters": 0},
            {"model": "patchtst", "seed": 42, "status": "complete", "normalized_rmse": 1.0, "training_mode": "supervised", "total_parameters": 100},
            {"model": "moirai2", "seed": 42, "status": "complete", "normalized_rmse": 1.2, "training_mode": "zero_shot", "total_parameters": 200},
        ]
    )
    selected = select_best_model(validation, constraints={"allowed_training_modes": ["zero_shot"]})
    assert selected["selected_model"] == "moirai2"
    assert selected["selection_split"] == "validation"


def test_generalization_gap_is_joined_by_model_and_seed() -> None:
    validation = pd.DataFrame(
        [{"model": "patchtst", "seed": 42, "status": "complete", "normalized_rmse": 1.0}]
    )
    target = pd.DataFrame(
        [{"model": "patchtst", "seed": 42, "status": "complete", "normalized_rmse": 1.5}]
    )
    gap = compute_generalization_gap(validation, target)
    assert gap.loc[0, "absolute_gap"] == 0.5
    summary = summarize_generalization_gap(gap)
    assert summary.loc[0, "relative_gap_mean"] == 0.5


def test_strategy_recommendation_is_support_aware_without_labels() -> None:
    recommendation = recommend_adaptation_strategy("moirai2", "zero_shot", 0.1, 4)
    assert recommendation["recommended_strategy"] == "target_residual_calibration"
    assert recommendation["selection_basis"] == "model_contract_and_observed_support_only"


def test_distribution_shift_uses_source_scale() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["source", "source", "target", "target"],
            "x": [0.0, 2.0, 4.0, 6.0],
            "y": [1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = compute_distribution_shift(frame, ["source"], ["target"], ["x", "y"], np.array([1.0, 1.0]))
    assert summary["aggregate"]["mean_abs_standardized_mean_shift"] == 2.0
    assert summary["features"][1]["std_ratio"] == 0.0


def test_tradeoff_report_marks_lower_is_better_pareto_front() -> None:
    table = pd.DataFrame(
        [
            {
                "model": "cheap",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 2.0,
                "fit_seconds": 1.0,
                "total_parameters": 1,
            },
            {
                "model": "accurate",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 1.0,
                "fit_seconds": 5.0,
                "total_parameters": 10,
            },
            {
                "model": "dominated",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 2.5,
                "fit_seconds": 2.0,
                "total_parameters": 20,
            },
        ]
    )
    report = summarize_tradeoffs(table)
    flags = dict(zip(report["model"], report["pareto_optimal"], strict=True))
    assert flags == {"accurate": True, "cheap": True, "dominated": False}


def test_adaptation_engine_uses_validation_shift_and_budget_inputs() -> None:
    validation = pd.DataFrame(
        [
            {
                "model": "patchtst",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 1.0,
                "training_mode": "supervised",
                "total_parameters": 100,
                "inference_seconds": 0.1,
            },
            {
                "model": "moirai2",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 1.2,
                "training_mode": "zero_shot",
                "total_parameters": 200,
                "inference_seconds": 0.2,
            },
        ]
    )
    plan = build_adaptation_plan(
        validation,
        AdaptationRequest(target_data_fraction=0.2, shift_score=1.5, support_windows=20),
    )
    assert plan["selected_model"] == "patchtst"
    assert plan["selected_strategy"] == "few_shot_finetune"
    assert plan["target_labels_used"] is False


def test_adaptation_engine_does_not_scale_without_support_windows() -> None:
    validation = pd.DataFrame(
        [
            {
                "model": "moirai2",
                "seed": 42,
                "status": "complete",
                "normalized_rmse": 1.0,
                "training_mode": "zero_shot",
                "total_parameters": 200,
            }
        ]
    )
    plan = build_adaptation_plan(
        validation,
        AdaptationRequest(target_data_fraction=0.1, support_windows=0),
    )
    assert plan["selected_strategy"] == "zero_shot"
