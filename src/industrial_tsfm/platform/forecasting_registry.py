from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .forecasting import ForecastingRecord, LiveForecastingApplication
from .runtime import WindowProvider


@dataclass
class _ForecastingEntry:
    application_id: str
    project_id: str | None
    source_name: str
    task_name: str
    runtime_id: str
    application: LiveForecastingApplication
    provider: WindowProvider
    inference_count: int = 0
    last_generated_at: str | None = None
    last_observed_through: str | None = None
    last_latency_ms: float | None = None


class LiveForecastingRegistry:
    """Process-local registry binding frozen models to source-neutral runtimes.

    Registration receives an already prepared inference-only forecasting
    application. The registry does not fit, adapt, calibrate, or mutate models.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ForecastingEntry] = {}

    def register(
        self,
        *,
        application_id: str,
        runtime_id: str,
        source_name: str,
        task_name: str,
        application: LiveForecastingApplication,
        provider: WindowProvider,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_id = str(application_id).strip()
        if not resolved_id:
            raise ValueError("application_id must not be empty")
        if resolved_id in self._entries:
            raise RuntimeError(f"forecasting application {resolved_id!r} already exists")
        if not str(runtime_id).strip():
            raise ValueError("runtime_id must not be empty")
        if not str(source_name).strip():
            raise ValueError("source_name must not be empty")
        if not str(task_name).strip():
            raise ValueError("task_name must not be empty")
        if not hasattr(provider, "materialize_window"):
            raise TypeError("provider must implement WindowProvider")
        self._entries[resolved_id] = _ForecastingEntry(
            application_id=resolved_id,
            project_id=project_id,
            source_name=str(source_name),
            task_name=str(task_name),
            runtime_id=str(runtime_id),
            application=application,
            provider=provider,
        )
        return self.status(resolved_id)

    def _entry(self, application_id: str) -> _ForecastingEntry:
        try:
            return self._entries[application_id]
        except KeyError as exc:
            raise KeyError(application_id) from exc

    def status(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        request = asdict(entry.application.request)
        return {
            "schema_version": "industrial_tsfm.live_forecasting_registry.v1",
            "application_id": entry.application_id,
            "project_id": entry.project_id,
            "source_name": entry.source_name,
            "task_name": entry.task_name,
            "runtime_id": entry.runtime_id,
            "model": entry.application.model.metadata(),
            "request": request,
            "inference_count": entry.inference_count,
            "last_generated_at": entry.last_generated_at,
            "last_observed_through": entry.last_observed_through,
            "last_latency_ms": entry.last_latency_ms,
            "online_training": False,
            "process_local": True,
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.status(application_id) for application_id in sorted(self._entries)]

    def infer(self, application_id: str) -> ForecastingRecord:
        entry = self._entry(application_id)
        record = entry.application.infer(entry.provider)
        entry.inference_count += 1
        entry.last_generated_at = record.generated_at
        entry.last_observed_through = record.observed_through
        entry.last_latency_ms = record.latency_ms
        return record

    def remove(self, application_id: str) -> None:
        self._entry(application_id)
        del self._entries[application_id]
