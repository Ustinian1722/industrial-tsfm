from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_support import DecisionSupportApplication
from .diagnosis_registry import LiveDiagnosisRegistry
from .forecasting_registry import LiveForecastingRegistry


@dataclass
class _DecisionSupportEntry:
    application_id: str
    project_id: str | None
    source_name: str
    runtime_id: str
    forecasting_application_id: str
    diagnosis_application_id: str
    application: DecisionSupportApplication
    forecasting_registry: LiveForecastingRegistry
    diagnosis_registry: LiveDiagnosisRegistry
    evaluation_count: int = 0
    last_generated_at: str | None = None
    last_status: str | None = None
    last_top_scenario_id: str | None = None


class DecisionSupportRegistry:
    """Process-local advisory registry joining forecast and diagnosis evidence.

    The registry owns no model, optimizer, controller, or write client. Every
    evaluation obtains fresh evidence from the already registered inference-only
    forecasting and diagnosis applications.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _DecisionSupportEntry] = {}

    @staticmethod
    def _validate_binding(
        *,
        project_id: str | None,
        source_name: str,
        runtime_id: str,
        forecasting_status: dict[str, Any],
        diagnosis_status: dict[str, Any],
    ) -> None:
        for label, status in (
            ("forecasting", forecasting_status),
            ("diagnosis", diagnosis_status),
        ):
            if status.get("project_id") != project_id:
                raise ValueError(f"{label} application project does not match decision support")
            if str(status.get("source_name")) != source_name:
                raise ValueError(f"{label} application source does not match decision support")
            if str(status.get("runtime_id")) != runtime_id:
                raise ValueError(f"{label} application runtime does not match decision support")
        if forecasting_status.get("online_training") is not False:
            raise ValueError("forecasting application must be inference-only")
        if diagnosis_status.get("online_training") is not False:
            raise ValueError("diagnosis application must be inference-only")
        if diagnosis_status.get("control_actions_enabled") is not False:
            raise ValueError("diagnosis application unexpectedly exposes control actions")

    def register(
        self,
        *,
        application_id: str,
        project_id: str | None,
        source_name: str,
        runtime_id: str,
        forecasting_application_id: str,
        diagnosis_application_id: str,
        application: DecisionSupportApplication,
        forecasting_registry: LiveForecastingRegistry,
        diagnosis_registry: LiveDiagnosisRegistry,
    ) -> dict[str, Any]:
        resolved_id = str(application_id).strip()
        if not resolved_id:
            raise ValueError("application_id must not be empty")
        if resolved_id in self._entries:
            raise RuntimeError(f"decision support application {resolved_id!r} already exists")
        resolved_source = str(source_name).strip()
        resolved_runtime = str(runtime_id).strip()
        forecast_id = str(forecasting_application_id).strip()
        diagnosis_id = str(diagnosis_application_id).strip()
        if not resolved_source or not resolved_runtime or not forecast_id or not diagnosis_id:
            raise ValueError("source/runtime/forecasting/diagnosis ids must not be empty")
        forecasting_status = forecasting_registry.status(forecast_id)
        diagnosis_status = diagnosis_registry.status(diagnosis_id)
        self._validate_binding(
            project_id=project_id,
            source_name=resolved_source,
            runtime_id=resolved_runtime,
            forecasting_status=forecasting_status,
            diagnosis_status=diagnosis_status,
        )
        application.policy.validate()
        self._entries[resolved_id] = _DecisionSupportEntry(
            application_id=resolved_id,
            project_id=project_id,
            source_name=resolved_source,
            runtime_id=resolved_runtime,
            forecasting_application_id=forecast_id,
            diagnosis_application_id=diagnosis_id,
            application=application,
            forecasting_registry=forecasting_registry,
            diagnosis_registry=diagnosis_registry,
        )
        return self.status(resolved_id)

    def _entry(self, application_id: str) -> _DecisionSupportEntry:
        try:
            return self._entries[application_id]
        except KeyError as exc:
            raise KeyError(application_id) from exc

    def status(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        return {
            "schema_version": "industrial_tsfm.decision_support_registry.v1",
            "application_id": entry.application_id,
            "project_id": entry.project_id,
            "source_name": entry.source_name,
            "runtime_id": entry.runtime_id,
            "forecasting_application_id": entry.forecasting_application_id,
            "diagnosis_application_id": entry.diagnosis_application_id,
            "policy": entry.application.policy.to_dict(),
            "evaluation_count": entry.evaluation_count,
            "last_generated_at": entry.last_generated_at,
            "last_status": entry.last_status,
            "last_top_scenario_id": entry.last_top_scenario_id,
            "advisory_only": True,
            "operator_confirmation_required": True,
            "execution_enabled": False,
            "control_actions_enabled": False,
            "opcua_writes_enabled": False,
            "online_learning": False,
            "process_local": True,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.status(application_id) for application_id in sorted(self._entries)]

    def evaluate(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        forecasting_status = entry.forecasting_registry.status(entry.forecasting_application_id)
        diagnosis_status = entry.diagnosis_registry.status(entry.diagnosis_application_id)
        self._validate_binding(
            project_id=entry.project_id,
            source_name=entry.source_name,
            runtime_id=entry.runtime_id,
            forecasting_status=forecasting_status,
            diagnosis_status=diagnosis_status,
        )
        forecast = entry.forecasting_registry.infer_payload(entry.forecasting_application_id)
        diagnosis = entry.diagnosis_registry.infer(entry.diagnosis_application_id)
        result = entry.application.evaluate(forecast, diagnosis)
        result["binding"] = {
            "application_id": entry.application_id,
            "project_id": entry.project_id,
            "source_name": entry.source_name,
            "runtime_id": entry.runtime_id,
            "forecasting_application_id": entry.forecasting_application_id,
            "diagnosis_application_id": entry.diagnosis_application_id,
        }
        entry.evaluation_count += 1
        entry.last_generated_at = str(result["generated_at"])
        entry.last_status = str(result["status"])
        top = result.get("top_recommendation")
        entry.last_top_scenario_id = None if not isinstance(top, dict) else str(top["scenario_id"])
        return result

    def remove(self, application_id: str) -> None:
        self._entry(application_id)
        del self._entries[application_id]
