from dataclasses import dataclass
from typing import Protocol, Any


@dataclass
class MemoryUpdateSchedule:
    fast_rate: float = 1.0
    medium_rate: float = 0.25
    slow_rate: float = 0.0625


class MemoryModule(Protocol):
    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, Any]: ...
    def retrieve(self, x: Any, state: Any | None = None) -> Any: ...
