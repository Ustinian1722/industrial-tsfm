from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_tsfm.platform.diagnosis import DiagnosisRequest, LiveDiagnosisApplication
from industrial_tsfm.platform.diagnosis_registry import LiveDiagnosisRegistry
from industrial_tsfm.platform.online_anomaly import fit_pca_spe_artifact
from industrial_tsfm.platform.runtime import DataReplayRuntime, ReplayConfig


def _application_and_provider():
    x = np.linspace(-1.0, 1.0, 40)
    y = 0.5 * x + 0.03 * np.sin(np.arange(40, dtype=float))
    artifact = fit_pca_spe_artifact(
        pd.DataFrame({"x": x, "y": y}),
        ("x", "y"),
        threshold_quantile=0.95,
        n_components=1,
        min_train_rows=20,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="s", tz="UTC").astype(str),
            "x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 6.0],
            "y": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        }
    )
    provider = DataReplayRuntime(frame, ReplayConfig(timestamp_column="timestamp"))
    application = LiveDiagnosisApplication(
        artifact,
        DiagnosisRequest(context_observations=8, top_k_sensors=2),
    )
    return application, provider


def test_diagnosis_registry_tracks_inference_without_online_mutation() -> None:
    registry = LiveDiagnosisRegistry()
    application, provider = _application_and_provider()
    status = registry.register(
        application_id="demo-diagnosis",
        runtime_id="replay-demo",
        project_id="demo",
        source_name="demo-source",
        task_name="detect-process",
        application=application,
        provider=provider,
    )

    assert status["inference_count"] == 0
    assert status["online_training"] is False
    assert status["threshold_updated_online"] is False
    assert status["control_actions_enabled"] is False

    payload = registry.infer("demo-diagnosis")
    after = registry.status("demo-diagnosis")
    assert after["inference_count"] == 1
    assert after["last_generated_at"] == payload["generated_at"]
    assert after["last_observed_through"] == payload["observed_through"]
    assert after["last_severity"] == payload["severity"]


def test_diagnosis_registry_rejects_duplicate_and_supports_remove() -> None:
    registry = LiveDiagnosisRegistry()
    application, provider = _application_and_provider()
    kwargs = {
        "application_id": "same",
        "runtime_id": "runtime-1",
        "project_id": "demo",
        "source_name": "plc",
        "task_name": "detect-process",
        "application": application,
        "provider": provider,
    }
    registry.register(**kwargs)
    try:
        registry.register(**kwargs)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate diagnosis application id must be rejected")

    registry.remove("same")
    assert registry.list() == []
