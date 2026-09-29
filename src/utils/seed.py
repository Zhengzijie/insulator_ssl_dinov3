"""Deterministic random-state helpers."""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def path_seed(relative_path: str) -> int:
    return zlib.crc32(relative_path.encode("utf-8")) & 0xFFFFFFFF

