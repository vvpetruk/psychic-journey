from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path.cwd() / "src"))

    from memory_lab.models.memory_mac import MemoryMAC, MemoryMACConfig

    cfg = MemoryMACConfig(
        dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=50257,
        max_context_length=128,
        chunk_size=32,
        window_size=32,
        memory_decay=0.001,
        memory_lr=0.1,
        memory_momentum=0.9,
    )
    model = MemoryMAC(cfg)
    backend = model.live_backend
    if backend is None:
        raise RuntimeError(model.titans_status.error)

    params = list(backend.named_parameters())
    total = sum(param.numel() for _name, param in params)
    trainable = sum(param.numel() for _name, param in params if param.requires_grad)

    print("total", total)
    print("trainable", trainable)

    groups: dict[str, int] = {}
    for name, param in params:
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + param.numel()

    print("groups")
    for key, value in sorted(groups.items(), key=lambda item: item[1], reverse=True):
        print(key, value)

    print("blocks")
    for block_idx in range(cfg.num_layers):
        prefix = f"blocks.{block_idx}."
        subtotal = sum(param.numel() for name, param in params if name.startswith(prefix))
        print(block_idx, subtotal)
        subgroups: dict[str, int] = {}
        for name, param in params:
            if name.startswith(prefix):
                rest = name[len(prefix):]
                key = rest.split(".")[0]
                subgroups[key] = subgroups.get(key, 0) + param.numel()
        for key, value in sorted(subgroups.items(), key=lambda item: item[1], reverse=True):
            print(" ", key, value)


if __name__ == "__main__":
    main()
