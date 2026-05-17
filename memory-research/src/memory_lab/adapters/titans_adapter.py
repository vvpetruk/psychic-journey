from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any


def add_titans_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    titans_src = root / 'titans-pytorch-mlx' / 'src'
    if titans_src.exists() and str(titans_src) not in sys.path:
        sys.path.insert(0, str(titans_src))
    return titans_src


@dataclass
class TitansImportStatus:
    available: bool
    source_path: str
    error: str | None = None


def get_titans_import_status() -> TitansImportStatus:
    src = add_titans_src_to_path()
    try:
        import titans  # noqa: F401
        return TitansImportStatus(True, str(src))
    except Exception as exc:
        return TitansImportStatus(False, str(src), error=repr(exc))


def load_titans_symbols() -> dict[str, Any]:
    add_titans_src_to_path()
    from titans.config import TitansConfig
    from titans.memory import NeuralLongTermMemory, MemoryState
    from titans.models import TitansMAC, TitansMAG, TitansMAL, TitansLMM

    return {
        'TitansConfig': TitansConfig,
        'NeuralLongTermMemory': NeuralLongTermMemory,
        'MemoryState': MemoryState,
        'TitansMAC': TitansMAC,
        'TitansMAG': TitansMAG,
        'TitansMAL': TitansMAL,
        'TitansLMM': TitansLMM,
    }
