"""Backward-compatible entry point for training.

The implementation now lives in the ``vesselbridge3d`` subpackages
(``models``, ``data``, ``losses``, ``engine``, ``cli``). This module is kept so
that ``python -m vesselbridge3d.train`` and existing imports keep working.
"""

from __future__ import annotations

from .cli import train_main
from .data import parse_img_size  # noqa: F401  (re-exported for compatibility)
from .engine import eval_epoch, train  # noqa: F401
from .models import SegModel3D_UNETRLite, build_model  # noqa: F401

if __name__ == '__main__':
    train_main()
