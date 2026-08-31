from __future__ import annotations

import numpy as np

from industrial_tsfm.evaluation import aggregate_result_table, compute_metrics
from industrial_tsfm.models.chronos2 import Chronos2Model
from industrial_tsfm.models.moirai2 import Moirai2Model
from industrial_tsfm.models.naive import NaiveModel
from industrial_tsfm.models.patchtst import PatchTSTModel


def test_naive_and_metrics() -> None:
    context = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    target = np.repeat(context[:, -1:, :], 2, axis=1)
    prediction = NaiveModel().configure(2).predict(context)
    metrics = compute_metrics(target, prediction, np.ones(3), ["test_unit_001", "test_unit_002"])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mase"] == 0.0


def test_normalized_metrics_use_source_scale() -> None:
    target = np.zeros((1, 2, 2), dtype=np.float32)
    prediction = np.asarray([[[2.0, 3.0], [2.0, 3.0]]], dtype=np.float32)
    metrics = compute_metrics(
        target,
        prediction,
        np.ones(2),
        ["test_unit_001"],
        normalized_scale=np.asarray([2.0, 3.0]),
    )
    assert metrics["normalized_mae"] == 1.0
    assert metrics["normalized_rmse"] == 1.0


def test_patchtst_fits_small_batch() -> None:
    rng = np.random.default_rng(0)
    train_context = rng.normal(size=(16, 16, 3)).astype(np.float32)
    train_target = rng.normal(size=(16, 4, 3)).astype(np.float32)
    model = PatchTSTModel(
        {
            "patch_len": 4,
            "stride": 2,
            "d_model": 16,
            "n_heads": 2,
            "e_layers": 1,
            "d_ff": 32,
            "epochs": 1,
            "batch_size": 8,
        },
        context_length=16,
        horizon=4,
        device="cpu",
    )
    info = model.fit(train_context, train_target, train_context[:4], train_target[:4], seed=0)
    prediction = model.predict(train_context[:3])
    assert info["fit_status"] == "complete"
    assert prediction.shape == (3, 4, 3)
    assert model.parameter_count() is not None


def test_chronos2_point_forecast_contract() -> None:
    predictions = [
        np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2),
        np.arange(3 * 3 * 2, 2 * 3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2),
    ]
    point = Chronos2Model._point_forecast(
        predictions,
        batch_size=2,
        n_features=3,
        horizon=2,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    assert point.shape == (2, 2, 3)
    np.testing.assert_array_equal(point[0], predictions[0][:, 1, :].T)


def test_moirai2_point_forecast_contract() -> None:
    class FakeForecast:
        def quantile(self, value: float) -> np.ndarray:
            assert value == 0.5
            return np.arange(2 * 3, dtype=np.float32).reshape(2, 3)

    point = Moirai2Model._point_forecast(FakeForecast(), horizon=2, n_features=3, quantile=0.5)
    assert point.shape == (2, 3)
    np.testing.assert_array_equal(point[0], np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_result_aggregation_reports_seed_spread_and_single_seed_ci_gap() -> None:
    table = np.asarray(
        [
            ("patchtst", 42, "complete", 1.0),
            ("patchtst", 43, "complete", 3.0),
            ("chronos2", 42, "complete", 2.0),
        ],
        dtype=object,
    )
    frame = __import__("pandas").DataFrame(
        table, columns=["model", "seed", "status", "mae"]
    )
    summary = aggregate_result_table(frame, ["model"], metrics=["mae"])
    patch = summary[summary["model"] == "patchtst"].iloc[0]
    chronos2 = summary[summary["model"] == "chronos2"].iloc[0]
    assert patch["n_seeds"] == 2
    assert patch["mae_mean"] == 2.0
    assert patch["mae_std"] > 0.0
    assert np.isfinite(patch["mae_ci95_low"])
    assert chronos2["n_seeds"] == 1
    assert chronos2["mae_std"] == 0.0
    assert np.isnan(chronos2["mae_ci95_low"])
