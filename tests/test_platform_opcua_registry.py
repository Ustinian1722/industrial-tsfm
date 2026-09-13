from __future__ import annotations

import asyncio

from industrial_tsfm.platform.contracts import DataSourceKind, DataSourceSpec
from industrial_tsfm.platform.opcua_registry import OPCUALiveRuntimeRegistry


class FakeRuntime:
    def __init__(self, source: DataSourceSpec) -> None:
        self.source = source
        self.samples = 0
        self.status_name = "created"

    async def run(self, stop_event: asyncio.Event) -> None:
        self.status_name = "running"
        while not stop_event.is_set():
            self.samples += 1
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
        self.status_name = "stopped"

    def snapshot(self):
        return {
            "schema_version": "fake",
            "source_name": self.source.name,
            "state": {"status": self.status_name, "samples_received": self.samples},
            "read_only": True,
            "writes_enabled": False,
        }

    def materialize_window(self, max_observations=None):
        return {"max_observations": max_observations, "source": self.source.name}


def _source(name: str = "plc") -> DataSourceSpec:
    return DataSourceSpec(
        name=name,
        kind=DataSourceKind.OPCUA,
        location="opc.tcp://127.0.0.1:4840/demo",
        metadata={"node_ids": {"x": "ns=2;s=x"}},
    )


def test_registry_start_stop_duplicate_guard_and_restart() -> None:
    async def scenario() -> None:
        registry = OPCUALiveRuntimeRegistry(runtime_factory=FakeRuntime)
        started = await registry.start(_source())
        runtime_id = started["runtime_id"]
        assert started["registry"] == "process_local_v1"
        assert started["project_id"] is None
        assert started["writes_enabled"] is False
        try:
            await registry.start(_source())
        except RuntimeError as exc:
            assert "already active" in str(exc)
        else:
            raise AssertionError("duplicate source runtime must be rejected")
        await asyncio.sleep(0.03)
        status = registry.status(runtime_id)
        assert status["state"]["samples_received"] > 0
        stopped = await registry.stop(runtime_id)
        assert stopped["state"]["status"] == "stopped"
        assert stopped["task_done"] is True

        restarted = await registry.start(_source())
        assert restarted["runtime_id"] != runtime_id
        assert restarted["task_done"] is False
        assert len(registry.list()) == 2
        await registry.stop(restarted["runtime_id"])

        registry.remove(runtime_id)
        registry.remove(restarted["runtime_id"])
        assert registry.list() == []

    asyncio.run(scenario())


def test_registry_scopes_same_source_name_by_project_and_exposes_window_provider() -> None:
    async def scenario() -> None:
        registry = OPCUALiveRuntimeRegistry(runtime_factory=FakeRuntime)
        left = await registry.start(_source(), project_id="project-a")
        right = await registry.start(_source(), project_id="project-b")

        assert left["project_id"] == "project-a"
        assert right["project_id"] == "project-b"
        assert left["runtime_id"] != right["runtime_id"]
        assert registry.runtime_scope(left["runtime_id"]) == {
            "runtime_id": left["runtime_id"],
            "project_id": "project-a",
            "source_name": "plc",
        }
        provider = registry.window_provider(left["runtime_id"])
        assert provider.materialize_window(7) == {"max_observations": 7, "source": "plc"}

        try:
            await registry.start(_source(), project_id="project-a")
        except RuntimeError as exc:
            assert "project-a" in str(exc)
        else:
            raise AssertionError("duplicate project/source runtime must be rejected")

        await registry.stop_all()
        assert all(row["task_done"] for row in registry.list())

    asyncio.run(scenario())
