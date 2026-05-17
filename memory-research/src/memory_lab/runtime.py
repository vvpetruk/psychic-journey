from __future__ import annotations

from dataclasses import asdict, dataclass

from memory_lab.adapters.nested_learning_adapter import get_nested_learning_import_status
from memory_lab.adapters.titans_adapter import get_titans_import_status
from memory_lab.models.memory_mac import MemoryMAC


@dataclass
class RuntimeStatus:
    workspace_ready: bool
    titans_available: bool
    nested_learning_available: bool
    notes: list[str]


def get_runtime_status() -> RuntimeStatus:
    titans = get_titans_import_status()
    nested = get_nested_learning_import_status()
    notes = []
    if not titans.available:
        notes.append('Titans package source is present but dependency environment is not installed yet.')
    if not nested.available:
        notes.append('Nested Learning package source is present but dependency environment is not installed yet.')
    notes.append('Memory Workspace scaffold itself is importable and ready for environment setup.')
    return RuntimeStatus(
        workspace_ready=True,
        titans_available=titans.available,
        nested_learning_available=nested.available,
        notes=notes,
    )


def describe_runtime() -> dict:
    model = MemoryMAC()
    status = get_runtime_status()
    return {
        'runtime': asdict(status),
        'model': model.describe(),
    }
