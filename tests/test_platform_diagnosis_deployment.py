from __future__ import annotations

from dataclasses import replace

from industrial_tsfm.platform.contracts import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.diagnosis_deployment import build_diagnosis_deployment_spec


def _project(task_type: TaskType = TaskType.ANOMALY) -> ProjectSpec:
    constraints = {"feature_columns": ["temperature", "pressure"]}
    return ProjectSpec(
        name="Demo Plant",
        data_sources=(
            DataSourceSpec(
                name="plc",
                kind=DataSourceKind.OPCUA,
                location="opc.tcp://127.0.0.1:4840/demo",
                metadata={
                    "node_ids": {
                        "temperature": "ns=2;s=temperature",
                        "pressure": "ns=2;s=pressure",
                    }
                },
            ),
        ),
        tasks=(
            TaskDefinition(
                name="detect-process",
                task_type=task_type,
                target_columns=(),
                constraints=constraints,
            ),
        ),
    )


def test_build_diagnosis_deployment_uses_project_and_task_contract() -> None:
    spec = build_diagnosis_deployment_spec(
        _project(),
        project_id="demo-plant-01",
        source_name="plc",
        task_name="detect-process",
        context_observations=48,
        top_k_sensors=2,
        artifact_ref="workspace://demo/pca-spe.json",
    )
    payload = spec.to_dict()

    assert payload["project_id"] == "demo-plant-01"
    assert payload["project_name"] == "Demo Plant"
    assert payload["feature_columns"] == ["temperature", "pressure"]
    assert payload["context_observations"] == 48
    assert payload["top_k_sensors"] == 2
    assert payload["fit_scope"] == "offline_training_or_calibration_only"
    assert payload["missing_value_policy"] == "frozen_training_median"
    assert payload["target_labels_used"] is False
    assert payload["online_training"] is False
    assert payload["threshold_updates"] is False
    assert payload["causal_root_cause_claim"] is False
    assert payload["control_actions_enabled"] is False


def test_diagnosis_deployment_rejects_non_anomaly_task() -> None:
    try:
        build_diagnosis_deployment_spec(
            _project(TaskType.REGRESSION),
            project_id="demo",
            source_name="plc",
            task_name="detect-process",
        )
    except ValueError as exc:
        assert "not an anomaly_detection task" in str(exc)
    else:
        raise AssertionError("non-anomaly task must not produce diagnosis deployment")


def test_diagnosis_deployment_rejects_online_or_label_guided_contract() -> None:
    spec = build_diagnosis_deployment_spec(
        _project(),
        project_id="demo",
        source_name="plc",
        task_name="detect-process",
    )
    for invalid in (
        replace(spec, target_labels_used=True),
        replace(spec, fit_scope="online_live_fit"),
        replace(spec, missing_value_policy="live_window_median"),
    ):
        try:
            invalid.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe diagnosis deployment must be rejected")
