from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_lab.adapters.titans_adapter import get_titans_import_status, load_titans_symbols
from memory_lab.memory.hybrid_memory import HybridMemory, HybridMemoryConfig


@dataclass
class MemoryMACConfig:
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 32000
    max_context_length: int = 8192
    chunk_size: int = 512
    window_size: int = 512
    memory_decay: float = 0.001
    memory_lr: float = 0.1
    memory_momentum: float = 0.9


class MemoryMAC:
    def __init__(self, config: MemoryMACConfig | None = None, memory: HybridMemory | None = None):
        self.config = config or MemoryMACConfig()
        self.memory = memory or HybridMemory(HybridMemoryConfig(max_context_length=self.config.max_context_length))
        self.titans_status = get_titans_import_status()
        self.titans_symbols: dict[str, Any] | None = None
        self.live_backend = None

        if self.titans_status.available:
            self.titans_symbols = load_titans_symbols()
            TitansConfig = self.titans_symbols['TitansConfig']
            TitansMAC = self.titans_symbols['TitansMAC']
            titans_cfg = TitansConfig(
                dim=self.config.dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                vocab_size=self.config.vocab_size,
                max_seq_len=self.config.max_context_length,
                chunk_size=self.config.chunk_size,
                window_size=self.config.window_size,
                memory_decay=self.config.memory_decay,
                memory_lr=self.config.memory_lr,
                memory_momentum=self.config.memory_momentum,
            )
            self.live_backend = TitansMAC(titans_cfg)

    def forward(self, x: Any, state: Any | None = None) -> tuple[Any, dict[str, Any]]:
        # Non-executing safe default path. We do not invoke live Titans forward here yet.
        memory_out, memory_state = self.memory.forward(x, state=state)
        return memory_out, {
            'state': memory_state,
            'live_titans_backend_ready': self.live_backend is not None,
        }

    def describe(self) -> dict[str, Any]:
        return {
            'model': 'MemoryMAC',
            'dim': self.config.dim,
            'max_context_length': self.config.max_context_length,
            'memory': self.memory.summary(),
            'titans_available': self.titans_status.available,
            'titans_error': self.titans_status.error,
            'live_titans_backend_ready': self.live_backend is not None,
        }
