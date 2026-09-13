from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "industrial_tsfm" / "platform"
LIVE_MODULES = (
    ROOT / "diagnosis.py",
    ROOT / "diagnosis_registry.py",
    ROOT / "diagnosis_service.py",
    ROOT / "diagnosis_api.py",
)


def _live_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in LIVE_MODULES)


def test_live_diagnosis_path_cannot_fit_or_write_online() -> None:
    source = _live_source()
    forbidden = (
        ".fit(",
        "fit_pca_spe_artifact(",
        ".write_value(",
        ".write_data_value(",
        ".write_attribute(",
        ".set_writable(",
    )
    for token in forbidden:
        assert token not in source


def test_live_diagnosis_keeps_noncausal_and_no_control_boundaries() -> None:
    source = _live_source()
    assert "statistical_reconstruction_contribution_not_causal_root_cause" in source
    assert '"online_training": False' in source
    assert '"threshold_updated_online": False' in source
    assert '"control_actions_enabled": False' in source
