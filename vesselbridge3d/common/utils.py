"""Misc utilities: seeding, random helpers, color palette, logging."""

from __future__ import annotations

import logging
import random
from colorsys import hsv_to_rgb

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rbool(p: float) -> bool:
    return bool(torch.rand(1) < p)


def _runif(a: float, b: float) -> float:
    return float(torch.empty(1).uniform_(a, b))


def make_palette(num_classes: int) -> np.ndarray:
    """
    Create a bright, high-contrast palette.
    0=background -> black, others evenly spaced hues in HSV (S=0.85, V=0.95).
    """
    num_classes = max(1, int(num_classes))
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    if num_classes <= 1:
        return palette
    for c in range(1, num_classes):
        h = (c - 1) / max(1, num_classes - 1)
        s, v = 0.85, 0.95
        r, g, b = hsv_to_rgb(h, s, v)
        palette[c] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def colorize(label_2d: np.ndarray, palette: np.ndarray) -> np.ndarray:
    label_safe = np.clip(label_2d.astype(np.int64), 0, len(palette) - 1)
    return palette[label_safe]


def get_logger(name: str = "vesselbridge3d") -> logging.Logger:
    """Return a process-wide logger that writes to stdout.

    Kept intentionally simple: a single stream handler with a plain format so
    messages read the same as the previous bare ``print`` statements.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
