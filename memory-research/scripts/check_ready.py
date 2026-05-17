import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from memory_lab.runtime import describe_runtime


if __name__ == '__main__':
    print(describe_runtime())
