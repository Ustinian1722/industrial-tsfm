from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from industrial_tsfm.platform.contracts import DataSourceKind, DataSourceSpec
from industrial_tsfm.platform.diagnosis import DiagnosisRequest
from industrial_tsfm.platform.diagnosis_deployment import DiagnosisDeploymentSpec
from industrial_tsfm.platform.diagnosis_registry import LiveDiagnosisRegistry
from industrial_tsfm.platform.diagnosis_service import register_prepared_live_diagnosis
from industrial_tsfm.platform.online_anomaly import fit_pca_spe_artifact
from industrial_tsfm.platform.opcua_registry import OPCUALiveRuntimeRegistry
from industrial_tsfm.platform.runtime import BoundedLiveBuffer, LiveSample


class FakeDiagnosisRuntime:
    def __init__(self, source: DataSourceSpec) -> None:
        self.source = source
        self.buffer = BoundedLiveBuffer(max_rows=32)
        samples: list[LiveSample] = []
        for index in range(8):
            timestamp = f"2026-01-01T00:00:{index:02d}Z"
            x = float(index) / 10.0
            y = 0.5 * x
            if index == 7:
                x = 6.0
            samples.extend(
                [
                    LiveSample(tag="x", value=x, timestamp=timestamp),
                    LiveSample(tag="y", value=y, timestamp=timestamp),
                ]
            )
        self.buffer.extend(samples)
        self.status_name = "created"

    async def run(self, stop_event: asyncio.Event) -> None:
        self.status_name = "running"
        await stop_event.wait()
        self.status_name = "stopped"

    def materialize_window(self, max_observations=None):
        return self.buffer.materialize_window(max_observations)

    def snapshot(self):
        return {
            "schema_version": "fake-diagnosis-runtime",
            "source_name": self.source.name,
            "state": {"status": self.status_name},
            "read_only": True,
            "writes_enabled": False,
        }


def _source() -> DataSourceSpec:
    return DataSourceSpec(
        name="plc",
        kind=DataSourceKind.OPCUA,
        location="opc.tcp://127.0.0.1:4840/demo",
        metadata={"node_ids": {"x": "ns=2;s=x", "y": "ns=2;s=y"}},
    )


def _artifact():
    x = np.linspace(-1.0, 1.0, 40)
    y = 0.5 * x + 0.03 * np.sin(np.arange(40, dtype=float))
    return fit_pca_spe_artifact(
        pd.DataFrame({"x": x, "y": y}),
        ("x", "y"),
        threshold_quantile=0.95,
        n_components=1,
        min_train_rows=20,
    )


def _deployment(project_id: str = "demo") -> DiagnosisDeploymentSpec:
    return DiagnosisDeploymentSpec(
        deployment_id="demo-diagnosis",
        project_name="Demo",
        project_id=project_id,
        source_name="plc",
        task_name="detect-process",
        feature_columns=("x", "y"),
        context_observations=16,
        top_k_sensors=2,
        artifact_ref="workspace://demo/pca-spe.json",
    )


def test_live_diagnosis_binding_is_inference_only_and_project_scoped() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeDiagnosisRuntime)
        started = await opcua.start(_source(), project_id="demo")
        diagnoses = LiveDiagnosisRegistry()
        status = register_prepared_live_diagnosis(
            deployment=_deployment(),
            runtime_id=started["runtime_id"],
            artifact=_artifact(),
            opcua_registry=opcua,
            diagnosis_registry=diagnoses,
            request=DiagnosisRequest(context_observations=16, top_k_sensors=2),
        )

        assert status["project_id"] == "demo"
        assert status["online_training"] is False
        assert status["threshold_updated_online"] is False
        payload = diagnoses.infer("demo-diagnosis")
        assert payload["status"] == "alert"
        assert payload["control_actions_enabled"] is False
        assert diagnoses.status("demo-diagnosis")["inference_count"] == 1
        await opcua.stop_all()

    asyncio.run(scenario())


def test_live_diagnosis_rejects_cross_project_runtime() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeDiagnosisRuntime)
        started = await opcua.start(_source(), project_id="other")
        diagnoses = LiveDiagnosisRegistry()
        try:
            register_prepared_live_diagnosis(
                deployment=_deployment(project_id="demo"),
                runtime_id=started["runtime_id"],
                artifact=_artifact(),
                opcua_registry=opcua,
                diagnosis_registry=diagnoses,
            )
        except ValueError as exc:
            assert "runtime project" in str(exc)
        else:
            raise AssertionError("cross-project diagnosis binding must be rejected")
        await opcua.stop_all()

    asyncio.run(scenario())


def test_live_diagnosis_rejects_stopped_runtime_instead_of_stale_buffer() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeDiagnosisRuntime)
        started = await opcua.start(_source(), project_id="demo")
        diagnoses = LiveDiagnosisRegistry()
        register_prepared_live_diagnosis(
            deployment=_deployment(),
            runtime_id=started["runtime_id"],
            artifact=_artifact(),
            opcua_registry=opcua,
            diagnosis_registry=diagnoses,
            request=DiagnosisRequest(context_observations=16, top_k_sensors=2),
        )
        diagnoses.infer("demo-diagnosis")
        await opcua.stop(started["runtime_id"])
        try:
            diagnoses.infer("demo-diagnosis")
        except RuntimeError as exc:
            assert "refusing stale live diagnosis" in str(exc)
        else:
            raise AssertionError("stopped runtime must not emit stale diagnosis")

    asyncio.run(scenario())


def test_live_diagnosis_rejects_artifact_feature_mismatch() -> None:
    async def scenario() -> None:
        opcua = OPCUALiveRuntimeRegistry(runtime_factory=FakeDiagnosisRuntime)
        started = await opcua.start(_source(), project_id="demo")
        diagnoses = LiveDiagnosisRegistry()
        deployment = DiagnosisDeploymentSpec(
            deployment_id="demo-diagnosis",
            project_name="Demo",
            project_id="demo",
            source_name="plc",
            task_name="detect-process",
            feature_columns=("y", "x"),
            context_observations=16,
            top_k_sensors=2,
        )
        try:
            register_prepared_live_diagnosis(
                deployment=deployment,
                runtime_id=started["runtime_id"],
                artifact=_artifact(),
                opcua_registry=opcua,
                diagnosis_registry=diagnoses,
            )
        except ValueError as exc:
            assert "artifact features" in str(exc)
        else:
            raise AssertionError("artifact/deployment feature mismatch must be rejected")
        await opcua.stop_all()

    asyncio.run(scenario())
