"""Backward-compatible entry point for inference.

The implementation now lives in ``vesselbridge3d.engine.inference`` and
``vesselbridge3d.cli``. This module is kept so that
``python -m vesselbridge3d.inference`` and existing imports keep working.
"""

from __future__ import annotations

from .cli import inference_main
from .data import parse_img_size, preprocess_with_z  # noqa: F401  (re-exported)
from .engine import run_inference, save_nifti  # noqa: F401

if __name__ == "__main__":
    inference_main()
