from __future__ import annotations

from pathlib import Path

_PRODUCTION_FILES = (
    "src/industrial_tsfm/platform/decision_support.py",
    "src/industrial_tsfm/platform/decision_support_registry.py",
    "src/industrial_tsfm/platform/decision_support_api.py",
)


def test_decision_support_production_surface_has_no_training_or_ot_write_primitives() -> None:
    forbidden = (
        ".fit(",
        ".partial_fit(",
        ".write_value(",
        ".write_data_value(",
        ".write_attribute(",
        ".set_writable(",
        "write_setpoint",
        "send_command",
        "closed_loop",
    )
    for relative in _PRODUCTION_FILES:
        text = Path(relative).read_text(encoding="utf-8")
        lowered = text.lower()
        for token in forbidden:
            assert token not in lowered, f"unsafe primitive {token!r} found in {relative}"


def test_decision_support_contract_is_explicitly_advisory_only() -> None:
    combined = "\n".join(Path(path).read_text(encoding="utf-8") for path in _PRODUCTION_FILES)

    assert '"advisory_only": True' in combined
    assert '"operator_confirmation_required": True' in combined
    assert '"execution_enabled": False' in combined
    assert '"control_actions_enabled": False' in combined
    assert '"opcua_writes_enabled": False' in combined
    assert '"online_learning": False' in combined
    assert '"llm_actions_enabled": False' in combined
    assert "advisory_evidence_ranking_not_causal_or_control" in combined
    assert "deterministic_policy_rule_evaluation_not_process_simulation" in combined
