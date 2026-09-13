from __future__ import annotations

import numpy as np

from industrial_tsfm.platform.forecasting import ForecastingRecord, ForecastPoint
from industrial_tsfm.platform.forecasting_uq import (
    ConformalForecastArtifact,
    apply_conformal_forecast,
    calibrate_conformal_forecast,
)


def test_conformal_forecast_calibration_is_finite_sample_and_roundtrips() -> None:
    predicted = np.zeros((20, 2, 1), dtype=float)
    actual = np.zeros((20, 2, 1), dtype=float)
    actual[:, 0, 0] = np.arange(20, dtype=float)
    actual[:, 1, 0] = np.arange(20, dtype=float) * 2.0

    artifact = calibrate_conformal_forecast(
        actual,
        predicted,
        ("x",),
        alpha=0.1,
        min_calibration_rows=20,
    )
    restored = ConformalForecastArtifact.from_dict(artifact.to_dict())

    # n=20, alpha=.1 -> ceil(21*.9)=19th order statistic.
    assert restored.radii == ((18.0, 36.0),)
    assert restored.calibration_rows == 20
    assert restored.to_dict()["online_updates_allowed"] is False


def test_apply_conformal_forecast_never_updates_runtime_radii() -> None:
    artifact = ConformalForecastArtifact(
        target_columns=("x",),
        horizon=2,
        alpha=0.1,
        radii=((0.5, 1.0),),
        calibration_rows=64,
    )
    record = ForecastingRecord(
        generated_at="2026-01-01T00:00:00+00:00",
        source_kind="live",
        observed_through="2026-01-01T00:00:00Z",
        model={"name": "demo", "online_training": False},
        request={"target_columns": ["x"], "horizon": 2},
        forecasts=(
            ForecastPoint(target="x", step=1, value=10.0, timestamp=None),
            ForecastPoint(target="x", step=2, value=11.0, timestamp=None),
        ),
        context={"x": {"rows": 8}},
        quality={},
        latency_ms=1.0,
    )

    enriched = apply_conformal_forecast(artifact, record)

    assert enriched["forecasts"][0]["lower"] == 9.5
    assert enriched["forecasts"][0]["upper"] == 10.5
    assert enriched["forecasts"][1]["lower"] == 10.0
    assert enriched["forecasts"][1]["upper"] == 12.0
    assert enriched["uq"]["online_updates"] is False
    assert enriched["uq"]["calibration_rows"] == 64
    assert artifact.radii == ((0.5, 1.0),)


def test_conformal_apply_rejects_partial_forecast_shape() -> None:
    artifact = ConformalForecastArtifact(
        target_columns=("x",),
        horizon=2,
        alpha=0.1,
        radii=((0.5, 1.0),),
        calibration_rows=64,
    )
    record = {
        "forecasts": [
            {"target": "x", "step": 1, "value": 10.0, "timestamp": None},
        ]
    }

    try:
        apply_conformal_forecast(artifact, record)
    except ValueError as exc:
        assert "missing conformal points" in str(exc)
    else:
        raise AssertionError("partial forecast must not receive misleading intervals")


def test_conformal_artifact_rejects_malformed_or_unsafe_radii() -> None:
    malformed = ConformalForecastArtifact(
        target_columns=("x",),
        horizon=2,
        alpha=0.1,
        radii=((0.5,),),
        calibration_rows=64,
    )
    try:
        malformed.validate()
    except ValueError as exc:
        assert "horizon dimension" in str(exc)
    else:
        raise AssertionError("malformed conformal radii must be rejected")

    negative = ConformalForecastArtifact(
        target_columns=("x",),
        horizon=1,
        alpha=0.1,
        radii=((-0.1,),),
        calibration_rows=64,
    )
    try:
        negative.validate()
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative conformal radii must be rejected")
