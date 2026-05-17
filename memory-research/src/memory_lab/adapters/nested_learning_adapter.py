from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path


def add_nested_learning_src_to_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    nl_src = root / 'nested_learning' / 'src'
    if nl_src.exists() and str(nl_src) not in sys.path:
        sys.path.insert(0, str(nl_src))
    return nl_src


@dataclass
class NestedLearningImportStatus:
    available: bool
    source_path: str


def get_nested_learning_import_status() -> NestedLearningImportStatus:
    src = add_nested_learning_src_to_path()
    try:
        import nested_learning  # noqa: F401
        return NestedLearningImportStatus(True, str(src))
    except Exception:
        return NestedLearningImportStatus(False, str(src))
