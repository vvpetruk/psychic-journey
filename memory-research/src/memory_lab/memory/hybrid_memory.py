from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HybridMemoryConfig:
    use_multi_timescale_updates: bool = True
    use_continuum_memory: bool = False
    use_optimizer_coupling: bool = False
    fast_update_stride: int = 1
    medium_update_stride: int = 4
    slow_update_stride: int = 16
    max_context_length: int = 8192


class HybridMemory:
    def __init__(self, config: HybridMemoryConfig | None = None):
        self.config = config or HybridMemoryConfig()
        self.step_count = 0

    def should_update_fast(self) -> bool:
        return self.step_count % self.config.fast_update_stride == 0

    def should_update_medium(self) -> bool:
        return self.step_count % self.config.medium_update_stride == 0

    def should_update_slow(self) -> bool:
        return self.step_count % self.config.slow_update_stride == 0

    def step(self) -> dict[str, bool]:
        status = {
            'fast': self.should_update_fast(),
            'medium': self.should_update_medium(),
            'slow': self.should_update_slow(),
        }
        self.step_count += 1
        return status

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, dict[str, Any]]:
        update_status = self.step()
        return x, {
            'state': state,
            'update_status': update_status,
            'max_context_length': self.config.max_context_length,
        }

    def summary(self) -> dict:
        return {
            'use_multi_timescale_updates': self.config.use_multi_timescale_updates,
            'use_continuum_memory': self.config.use_continuum_memory,
            'use_optimizer_coupling': self.config.use_optimizer_coupling,
            'fast_update_stride': self.config.fast_update_stride,
            'medium_update_stride': self.config.medium_update_stride,
            'slow_update_stride': self.config.slow_update_stride,
            'max_context_length': self.config.max_context_length,
        }
