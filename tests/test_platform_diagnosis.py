from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from industrial_tsfm.platform.diagnosis import DiagnosisRequest, LiveDiagnosisApplication
from industrial_tsfm.platform.online_anomaly import fit_pca_spe_artifact
from industrial_tsfm.platform.runtime import DataReplayRuntime, ReplayConfig


def _training_frame(rows: int = 40) -> pd.DataFrame:
    x = np.linspace(-1.0, 1.0, rows)
    y = 0.5 * x + 0.03 * np.sin(np.arange(rows, dtype=float))
    return pd.DataFrame({"x": x, "y": y})


def _artifact():
    return fit_pca_spe_artifact(
        _training_frame(),
        ("x", "y"),
        threshold_quantile=0.95,
        n_components=1,
        min_train_rows=20,
    )


def test_live_diagnosis_flags_excursion_without_online_updates() -> None:
    frame = _training_frame(12)
    frame.loc[len(frame) - 1, "x"] = 8.0
    frame.insert(
        0,
        "timestamp",
        pd.date_range("2026-01-01", periods=len(frame), freq="s", tz="UTC").astype(str),
    )
    replay = DataReplayRuntime(frame, ReplayConfig(timestamp_column="timestamp"))
    app = LiveDiagnosisApplication(
        _artifact(),
        DiagnosisRequest(
            context_observations=8,
            top_k_sensors=2,
            high_score_ratio=1.5,
            critical_score_ratio=3.0,
            persistent_points=3,
        ),
    )

    payload = app.infer(replay)

    assert payload["schema_version"] == "industrial_tsfm.live_diagnosis.v1"
    assert payload["source_kind"] == "replay"
    assert payload["status"] == "alert"
    assert payload["severity"] in {"warning", "high", "critical"}
    assert payload["score"]["ratio_to_threshold"] >= 1.0
    assert payload["event"]["latest_anomaly"] is True
    assert payload["contributors"]
    assert payload["claim_scope"].endswith("not_causal_root_cause")
    assert payload["online_training"] is False
    assert payload["threshold_updated_online"] is False
    assert payload["online_artifact_updates"] is False
    assert payload["control_actions_enabled"] is False


def test_live_diagnosis_rejects_invalid_frozen_artifact() -> None:
    artifact = replace(_artifact(), threshold=-1.0)

    try:
        LiveDiagnosisApplication(artifact, DiagnosisRequest())
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("invalid diagnosis artifact must be rejected")
