from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .diagnosis import LiveDiagnosisApplication
from .runtime import WindowProvider


@dataclass
class _DiagnosisEntry:
    application_id: str
    runtime_id: str
    project_id: str | None
    source_name: str
    task_name: str
    application: LiveDiagnosisApplication
    provider: WindowProvider
    inference_count: int = 0
    last_generated_at: str | None = None
    last_observed_through: str | None = None
    last_severity: str | None = None


class LiveDiagnosisRegistry:
    """Process-local registry for frozen live diagnostic triage applications."""

    def __init__(self) -> None:
        self._entries: dict[str, _DiagnosisEntry] = {}

    def register(
        self,
        *,
        application_id: str,
        runtime_id: str,
        project_id: str | None,
        source_name: str,
        task_name: str,
        application: LiveDiagnosisApplication,
        provider: WindowProvider,
    ) -> dict[str, Any]:
        resolved_id = str(application_id).strip()
        if not resolved_id:
            raise ValueError("application_id must not be empty")
        if resolved_id in self._entries:
            raise RuntimeError(f"diagnosis application {resolved_id!r} already exists")
        if not str(runtime_id).strip():
            raise ValueError("runtime_id must not be empty")
        if not str(source_name).strip():
            raise ValueError("source_name must not be empty")
        if not str(task_name).strip():
            raise ValueError("task_name must not be empty")
        if not hasattr(provider, "materialize_window"):
            raise TypeError("provider must implement WindowProvider")
        self._entries[resolved_id] = _DiagnosisEntry(
            application_id=resolved_id,
            runtime_id=str(runtime_id),
            project_id=project_id,
            source_name=str(source_name),
            task_name=str(task_name),
            application=application,
            provider=provider,
        )
        return self.status(resolved_id)

    def _entry(self, application_id: str) -> _DiagnosisEntry:
        try:
            return self._entries[application_id]
        except KeyError as exc:
            raise KeyError(application_id) from exc

    def status(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        return {
            "schema_version": "industrial_tsfm.live_diagnosis_registry.v1",
            "application_id": entry.application_id,
            "runtime_id": entry.runtime_id,
            "project_id": entry.project_id,
            "source_name": entry.source_name,
            "task_name": entry.task_name,
            "feature_columns": list(entry.application.artifact.feature_columns),
            "request": {
                "context_observations": entry.application.request.context_observations,
                "top_k_sensors": entry.application.request.top_k_sensors,
                "high_score_ratio": entry.application.request.high_score_ratio,
                "critical_score_ratio": entry.application.request.critical_score_ratio,
                "persistent_points": entry.application.request.persistent_points,
            },
            "inference_count": entry.inference_count,
            "last_generated_at": entry.last_generated_at,
            "last_observed_through": entry.last_observed_through,
            "last_severity": entry.last_severity,
            "online_training": False,
            "threshold_updated_online": False,
            "online_artifact_updates": False,
            "control_actions_enabled": False,
            "process_local": True,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.status(application_id) for application_id in sorted(self._entries)]

    def infer(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        payload = entry.application.infer(entry.provider)
        entry.inference_count += 1
        entry.last_generated_at = str(payload["generated_at"])
        entry.last_observed_through = payload.get("observed_through")
        entry.last_severity = str(payload["severity"])
        return payload

    def remove(self, application_id: str) -> None:
        self._entry(application_id)
        del self._entries[application_id]
