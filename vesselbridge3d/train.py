"""Training entry point.

Run with ``python -m vesselbridge3d.train ...`` or the ``vesselbridge3d-train``
console script. This module owns the command-line interface; the heavy lifting
(model, data, training loop) lives in the ``vesselbridge3d`` subpackages
(``models``, ``data``, ``engine``).
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
from .common.utils import get_logger
from .data import parse_img_size, parse_layers
from .engine import train

# Re-exported for backward compatibility with `from vesselbridge3d.train import ...`.
from .engine import eval_epoch  # noqa: F401
from .models import SegModel3D_UNETRLite, build_model  # noqa: F401

logger = get_logger()


def build_train_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    # train/val lists
    ap.add_argument('--train_list', type=str, default=None, help='Path to training JSON list')
    ap.add_argument('--val_list',   type=str, default=None, help='Path to validation JSON list (defaults to train_list)')
    # Backward compatibility
    ap.add_argument('--list_json',  type=str, default=None, help='[Deprecated] Single JSON used for both train/val if train_list not provided')

    ap.add_argument('--model_type', type=str, default=DEFAULT_MODEL_TYPE,
                    help='Model architecture key (see vesselbridge3d.models registry)')

    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps to simulate larger batch size.')
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--min_lr', type=float, default=1e-4)
    ap.add_argument('--drop_empty', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bg_weight', type=float, default=0.05)
    ap.add_argument('--use_mfb', action='store_true')
    ap.add_argument('--min_fg_frac', type=float, default=0.0)
    ap.add_argument('--warmup_epochs', type=int, default=0)

    ap.add_argument('--init_prior_bias', action='store_true')
    ap.add_argument('--img_size', type=str, default=DEFAULT_IMG_SIZE, help='Input size e.g., 224 or 512. Must be multiples of 16')

    # augmentations
    ap.add_argument('--aug', action='store_true', help='Enable data augmentation on train set')
    ap.add_argument('--aug_rotate', type=float, default=15.0)
    ap.add_argument('--aug_scale_min', type=float, default=0.90)
    ap.add_argument('--aug_scale_max', type=float, default=1.10)
    ap.add_argument('--aug_translate', type=float, default=0.10, help='fraction of size (0-1)')
    ap.add_argument('--aug_hflip_p', type=float, default=0.0)
    ap.add_argument('--aug_vflip_p', type=float, default=0.0)
    ap.add_argument('--aug_noise_p', type=float, default=0.2)
    ap.add_argument('--aug_noise_std', type=float, default=0.01)
    ap.add_argument('--aug_bc_p', type=float, default=0.3, help='brightness/contrast prob')
    ap.add_argument('--aug_brightness', type=float, default=0.10)
    ap.add_argument('--aug_contrast', type=float, default=0.10)

    # 3D
    ap.add_argument('--use_3d', action='store_true', help='Use full 3D model (DINO 2D per-slice -> 3D aggregator -> 3D decoder)')
    ap.add_argument('--depth_min_fg_frac', type=float, default=0.0005, help='Per-depth foreground ratio threshold for 3D. Depth below this are masked out from CE/Dice. 0 to disable')
    ap.add_argument('--vit_chunk_slices', type=int, default=8)
    ap.add_argument('--vit_amp', action='store_true')

    # 3D UNETR-list
    ap.add_argument('--use_3d_unetr', action='store_true')
    ap.add_argument('--vit_layers', type=str, default=DEFAULT_VIT_LAYERS)

    # 3D Aggregator
    ap.add_argument('--decoder_up_factor', type=int, default=DEFAULT_DECODER_UP_FACTOR)

    # Z crop depth
    ap.add_argument('--crop_depth', type=int, default=CROP_D, help='Crop size in depth dimension (Z). Use 32 or 48 for high-res.')

    # Logger
    ap.add_argument('--log_dir', type=str, default='/path/to/log_dir', help='dir path to save logs per epoch')

    ap.add_argument('--save_dir', type=str,
                    default='/path/to/save_dir',
                    help='Directory to save best_checkpoint.pt and final_checkpoint.pt')

    return ap


def main(argv=None) -> None:
    ap = build_train_parser()
    args = parse_args_with_config(ap)

    # Resolve list paths (favor new args; fallback to legacy)
    if args.train_list is None:
        if args.list_json is None:
            raise ValueError("Please provide --train_list (and optionally --val_list)")
        train_list = args.list_json
        val_list = args.list_json
        logger.info("[WARN] Using legacy --list_json for both train and val.")
    else:
        train_list = args.train_list
        val_list = args.val_list if args.val_list is not None else args.train_list

    H, W = parse_img_size(args.img_size)
    vit_layers = parse_layers(args.vit_layers)

    train(train_list=train_list,
          val_list=val_list,
          num_classes=args.num_classes,
          epochs=args.epochs,
          batch_size=args.batch_size,
          accumulation_steps=args.accumulation_steps,
          lr=args.lr,
          min_lr=args.min_lr,
          drop_empty=args.drop_empty,
          seed=args.seed,
          bg_weight=args.bg_weight,
          use_mfb=args.use_mfb,
          min_fg_frac=args.min_fg_frac,
          init_prior_bias=args.init_prior_bias,
          img_size=(H, W),
          aug=args.aug,
          aug_rotate=args.aug_rotate,
          aug_scale_min=args.aug_scale_min, aug_scale_max=args.aug_scale_max,
          aug_translate=args.aug_translate,
          aug_hflip_p=args.aug_hflip_p, aug_vflip_p=args.aug_vflip_p,
          aug_noise_p=args.aug_noise_p, aug_noise_std=args.aug_noise_std,
          aug_bc_p=args.aug_bc_p, aug_brightness=args.aug_brightness, aug_contrast=args.aug_contrast,
          use_3d=args.use_3d,
          depth_min_fg_frac=args.depth_min_fg_frac,
          crop_depth=args.crop_depth,
          save_dir=args.save_dir,
          log_dir=args.log_dir,
          model_type=args.model_type,
          vit_layers=vit_layers,
          decoder_up_factor=args.decoder_up_factor,
          vit_chunk_slices=args.vit_chunk_slices,
          vit_amp=args.vit_amp,
          config_snapshot=vars(args))


if __name__ == '__main__':
    main()
