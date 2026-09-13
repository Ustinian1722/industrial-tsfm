from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .forecasting import ForecastingRecord, LiveForecastingApplication
from .forecasting_uq import ConformalForecastArtifact, apply_conformal_forecast
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
    uq_artifact: ConformalForecastArtifact | None = None
    inference_count: int = 0
    last_generated_at: str | None = None
    last_observed_through: str | None = None
    last_latency_ms: float | None = None


class LiveForecastingRegistry:
    """Process-local registry binding frozen models to source-neutral runtimes.

    Registration receives an already prepared inference-only forecasting
    application and, optionally, an already calibrated conformal artifact. The
    registry does not fit, adapt, select, calibrate, or mutate either object.
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
        uq_artifact: ConformalForecastArtifact | None = None,
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
        if uq_artifact is not None:
            request = application.request
            if tuple(uq_artifact.target_columns) != tuple(request.target_columns):
                raise ValueError("UQ artifact targets do not match forecasting request")
            if int(uq_artifact.horizon) != int(request.horizon):
                raise ValueError("UQ artifact horizon does not match forecasting request")
        self._entries[resolved_id] = _ForecastingEntry(
            application_id=resolved_id,
            project_id=project_id,
            source_name=str(source_name),
            task_name=str(task_name),
            runtime_id=str(runtime_id),
            application=application,
            provider=provider,
            uq_artifact=uq_artifact,
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
        uq = None
        if entry.uq_artifact is not None:
            uq = {
                "method": "split_conformal_absolute_residual",
                "alpha": entry.uq_artifact.alpha,
                "nominal_coverage": 1.0 - entry.uq_artifact.alpha,
                "calibration_rows": entry.uq_artifact.calibration_rows,
                "calibration_scope": entry.uq_artifact.calibration_scope,
                "online_updates": False,
            }
        return {
            "schema_version": "industrial_tsfm.live_forecasting_registry.v1",
            "application_id": entry.application_id,
            "project_id": entry.project_id,
            "source_name": entry.source_name,
            "task_name": entry.task_name,
            "runtime_id": entry.runtime_id,
            "model": entry.application.model.metadata(),
            "request": request,
            "uq": uq,
            "inference_count": entry.inference_count,
            "last_generated_at": entry.last_generated_at,
            "last_observed_through": entry.last_observed_through,
            "last_latency_ms": entry.last_latency_ms,
            "online_training": False,
            "online_calibration": False,
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

    def infer_payload(self, application_id: str) -> dict[str, Any]:
        entry = self._entry(application_id)
        record = self.infer(application_id)
        if entry.uq_artifact is None:
            return record.to_dict()
        return apply_conformal_forecast(entry.uq_artifact, record)

    def remove(self, application_id: str) -> None:
        self._entry(application_id)
        del self._entries[application_id]
