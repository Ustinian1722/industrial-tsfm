from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from .contracts import DataSourceSpec
from .opcua_runtime import OPCUALiveRuntime


@dataclass
class _RuntimeEntry:
    runtime_id: str
    source_name: str
    runtime: Any
    stop_event: asyncio.Event
    task: asyncio.Task[Any]


class OPCUALiveRuntimeRegistry:
    """Process-local lifecycle manager for read-only OPC-UA subscriptions.

    Registry state is intentionally ephemeral in this first live slice. Project
    configuration remains persisted by WorkspaceStore, while active sockets and
    tasks are tied to the API process that owns them.
    """

    def __init__(
        self,
        runtime_factory: Callable[[DataSourceSpec], Any] = OPCUALiveRuntime,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._entries: dict[str, _RuntimeEntry] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    async def start(self, source: DataSourceSpec) -> dict[str, Any]:
        async with self._lock:
            existing = [entry for entry in self._entries.values() if entry.source_name == source.name]
            if existing:
                raise RuntimeError(f"OPC-UA runtime already active for source {source.name!r}")
            self._counter += 1
            runtime_id = f"opcua-{self._counter:04d}-{source.name}"
            runtime = self._runtime_factory(source)
            stop_event = asyncio.Event()
            task = asyncio.create_task(runtime.run(stop_event), name=runtime_id)
            self._entries[runtime_id] = _RuntimeEntry(
                runtime_id=runtime_id,
                source_name=source.name,
                runtime=runtime,
                stop_event=stop_event,
                task=task,
            )
        await asyncio.sleep(0)
        return self.status(runtime_id)

    def _entry(self, runtime_id: str) -> _RuntimeEntry:
        try:
            return self._entries[runtime_id]
        except KeyError as exc:
            raise KeyError(runtime_id) from exc

    def status(self, runtime_id: str) -> dict[str, Any]:
        entry = self._entry(runtime_id)
        snapshot = dict(entry.runtime.snapshot())
        snapshot.update(
            {
                "runtime_id": runtime_id,
                "task_done": entry.task.done(),
                "registry": "process_local_v1",
            }
        )
        if entry.task.done() and not entry.task.cancelled():
            error = entry.task.exception()
            if error is not None:
                snapshot["task_error"] = f"{type(error).__name__}: {error}"
        return snapshot

    def list(self) -> list[dict[str, Any]]:
        return [self.status(runtime_id) for runtime_id in sorted(self._entries)]

    async def stop(self, runtime_id: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        entry = self._entry(runtime_id)
        entry.stop_event.set()
        if not entry.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(entry.task), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                entry.task.cancel()
                try:
                    await entry.task
                except asyncio.CancelledError:
                    pass
        return self.status(runtime_id)

    async def stop_all(self) -> None:
        for runtime_id in list(self._entries):
            try:
                await self.stop(runtime_id)
            except Exception:
                entry = self._entries[runtime_id]
                if not entry.task.done():
                    entry.task.cancel()

    def remove(self, runtime_id: str) -> None:
        entry = self._entry(runtime_id)
        if not entry.task.done():
            raise RuntimeError("cannot remove a running OPC-UA runtime; stop it first")
        del self._entries[runtime_id]
