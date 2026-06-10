"""Inference entry point.

Run with ``python -m vesselbridge3d.inference ...`` or the
``vesselbridge3d-infer`` console script. This module owns the command-line
interface; the sliding-window inference and IO live in
``vesselbridge3d.engine.inference``.
"""

from __future__ import annotations

import argparse

from .common.config import add_config_arg, parse_args_with_config
from .common.constants import (
    CROP_D,
    DEFAULT_DECODER_UP_FACTOR,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_VIT_LAYERS,
)
from .engine import run_inference

# Re-exported for backward compatibility with `from vesselbridge3d.inference import ...`.
from .data import parse_img_size, preprocess_with_z  # noqa: F401
from .engine import save_nifti  # noqa: F401


def build_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE,
                        help='Model architecture key (see vesselbridge3d.models registry)')
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--img_size', type=str, default=DEFAULT_IMG_SIZE)

    parser.add_argument('--vit_layers', type=str, default=DEFAULT_VIT_LAYERS, help='Layers to use, e.g. "11,11,11" or "2,5,8,11"')

    parser.add_argument('--decoder_up_factor', type=int, default=DEFAULT_DECODER_UP_FACTOR)
    parser.add_argument('--vit_chunk_slices', type=int, default=8)
    parser.add_argument('--vit_amp', action='store_true')
    parser.add_argument('--crop_depth', type=int, default=CROP_D, help='Depth chunk size for the sliding window over Z.')

    return parser


def main(argv=None) -> None:
    parser = build_inference_parser()
    args = parse_args_with_config(parser)
    run_inference(args)


if __name__ == "__main__":
    main()
