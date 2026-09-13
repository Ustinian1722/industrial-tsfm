from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform.online_anomaly import (
    OnlinePCASPEAnomalyDetector,
    PCASPEArtifact,
    fit_pca_spe_artifact,
    score_pca_spe_artifact,
)
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


def _training_frame() -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, 32)
    return pd.DataFrame({"pressure": x, "temperature": 2.0 * x + 0.05 * np.sin(8.0 * x)})


def test_pca_spe_artifact_roundtrip_and_threshold_are_frozen() -> None:
    artifact = fit_pca_spe_artifact(
        _training_frame(),
        ("pressure", "temperature"),
        threshold_quantile=0.95,
        n_components=1,
    )
    restored = PCASPEArtifact.from_dict(artifact.to_dict())
    threshold_before = restored.threshold

    scored = score_pca_spe_artifact(
        restored,
        pd.DataFrame({"pressure": [0.2, 8.0], "temperature": [0.4, -8.0]}),
    )

    assert restored.threshold == threshold_before
    assert scored["threshold"] == threshold_before
    assert scored["online_training"] is False
    assert scored["threshold_updated_online"] is False
    assert scored["latest_anomaly"] is True
    assert scored["anomaly_flags"][-1] is True


def test_online_pca_spe_consumes_live_window_without_recalibration() -> None:
    artifact = fit_pca_spe_artifact(
        _training_frame(),
        ("pressure", "temperature"),
        threshold_quantile=0.95,
        n_components=1,
    )
    live = BoundedLiveBuffer(max_rows=8)
    live.extend(
        [
            LiveSample(tag="pressure", value=0.2, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="temperature", value=0.4, timestamp="2026-01-01T00:00:00Z"),
            LiveSample(tag="pressure", value=8.0, timestamp="2026-01-01T00:00:01Z"),
            LiveSample(tag="temperature", value=-8.0, timestamp="2026-01-01T00:00:01Z"),
        ]
    )
    detector = OnlinePCASPEAnomalyDetector(artifact, context_observations=8)

    result = detector.infer(live)

    assert result["source_kind"] == "live"
    assert result["observed_through"] == "2026-01-01T00:00:01Z"
    assert result["latest_anomaly"] is True
    assert result["threshold"] == artifact.threshold
    assert result["artifact_fit_rows"] == 32
    assert result["threshold_updated_online"] is False
    assert result["window"]["metadata"]["timestamp_sort_applied"] is False
    assert result["window"]["metadata"]["interpolation"] is False
