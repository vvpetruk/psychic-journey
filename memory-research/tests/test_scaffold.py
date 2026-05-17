import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.adapters.titans_adapter import get_titans_import_status
from memory_lab.memory.hybrid_memory import HybridMemory
from memory_lab.models.memory_mac import MemoryMAC


def test_hybrid_memory_scaffold():
    m = HybridMemory()
    summary = m.summary()
    assert summary['use_multi_timescale_updates'] is True
    assert summary['max_context_length'] == 8192


def test_memory_mac_scaffold():
    model = MemoryMAC()
    desc = model.describe()
    assert desc['model'] == 'MemoryMAC'
    assert desc['max_context_length'] == 8192


def test_titans_adapter_path_resolution():
    status = get_titans_import_status()
    assert 'titans-pytorch-mlx/src' in status.source_path
