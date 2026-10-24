"""Small shared helpers: seeding, device selection, timing."""
from __future__ import annotations

import os
import random
import time

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bf16_ok(device) -> bool:
    import torch
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def describe_hardware() -> str:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"CUDA: {p.name}, {p.total_memory / 2**30:.1f} GB, torch {torch.__version__}"
    return f"CPU only, torch {torch.__version__}"


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
