from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "industrial_tsfm" / "platform"
LIVE_MODULES = (
    ROOT / "forecasting.py",
    ROOT / "forecasting_deployment.py",
    ROOT / "forecasting_registry.py",
    ROOT / "forecasting_service.py",
    ROOT / "forecasting_api.py",
)


def _live_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in LIVE_MODULES)


def test_live_forecasting_path_cannot_fit_select_or_calibrate_online() -> None:
    source = _live_source()
    forbidden = (
        ".fit(",
        "route_task(",
        "calibrate_conformal_forecast(",
        ".write_value(",
        ".write_data_value(",
        ".write_attribute(",
        ".set_writable(",
    )
    for token in forbidden:
        assert token not in source


def test_live_forecasting_source_keeps_explicit_offline_boundaries() -> None:
    source = _live_source()
    assert '"online_training": False' in source
    assert '"online_calibration": False' in source
    assert "validation_only" in source
    assert "target_labels_used" in source
    assert "WindowProvider" in source
