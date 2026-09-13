from __future__ import annotations

from industrial_tsfm.platform.contracts import (
    DataSourceKind,
    DataSourceSpec,
    ProjectSpec,
    TaskDefinition,
    TaskType,
)
from industrial_tsfm.platform.forecasting_deployment import build_forecasting_deployment_spec


def _project() -> ProjectSpec:
    return ProjectSpec(
        name="Demo Plant",
        data_sources=(
            DataSourceSpec(
                name="plc",
                kind=DataSourceKind.OPCUA,
                location="opc.tcp://127.0.0.1:4840/demo",
                metadata={"node_ids": {"x": "ns=2;s=x"}},
            ),
        ),
        tasks=(
            TaskDefinition(
                name="forecast-x",
                task_type=TaskType.FORECASTING,
                target_columns=("x",),
                context_length=32,
                horizon=4,
            ),
        ),
    )


def test_build_forecasting_deployment_requires_validation_only_route() -> None:
    route = {
        "route_ready": True,
        "task": "forecast-x",
        "selected_model": "chronos",
        "selected_strategy": "zero_shot",
        "selection_evidence": "validation_only",
        "target_labels_used": False,
    }

    spec = build_forecasting_deployment_spec(
        _project(),
        source_name="plc",
        task_name="forecast-x",
        model_route=route,
        checkpoint_ref="amazon/chronos-t5-tiny",
    )
    payload = spec.to_dict()

    assert payload["selected_model"] == "chronos"
    assert payload["window_mode"] == "per_tag_univariate"
    assert payload["requires_aligned_rows"] is False
    assert payload["online_training"] is False
    assert payload["online_model_selection"] is False
    assert payload["target_labels_used"] is False


def test_native_multivariate_route_requires_strict_multivariate_windows() -> None:
    route = {
        "route_ready": True,
        "task": "forecast-x",
        "selected_model": "chronos2",
        "selected_strategy": "zero_shot",
        "selection_evidence": "validation_only",
        "target_labels_used": False,
    }
    spec = build_forecasting_deployment_spec(
        _project(),
        source_name="plc",
        task_name="forecast-x",
        model_route=route,
    )
    assert spec.window_mode == "strict_multivariate"
    assert spec.requires_aligned_rows is True

    try:
        build_forecasting_deployment_spec(
            _project(),
            source_name="plc",
            task_name="forecast-x",
            model_route=route,
            window_mode="per_tag_univariate",
        )
    except ValueError as exc:
        assert "native multivariate" in str(exc)
    else:
        raise AssertionError("native multivariate deployment must require aligned rows")


def test_deployment_rejects_target_label_guided_route() -> None:
    route = {
        "route_ready": True,
        "task": "forecast-x",
        "selected_model": "chronos",
        "selected_strategy": "zero_shot",
        "selection_evidence": "validation_only",
        "target_labels_used": True,
    }

    try:
        build_forecasting_deployment_spec(
            _project(),
            source_name="plc",
            task_name="forecast-x",
            model_route=route,
        )
    except ValueError as exc:
        assert "target-label-guided" in str(exc)
    else:
        raise AssertionError("target-label-guided route must be rejected")
