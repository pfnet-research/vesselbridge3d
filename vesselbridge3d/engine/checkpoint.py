"""Checkpoint loading helpers shared across inference / resuming."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.utils import get_logger

logger = get_logger()


def load_model_weights(model: nn.Module, checkpoint_path: str, device) -> nn.Module:
    """Load model weights from a checkpoint, stripping any ``module.`` prefix.

    Tries a strict load first and falls back to a non-strict load (matching the
    previous inference behavior).
    """
    logger.info(f"[INFO] Loading weights: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        logger.info(
            f"[WARN] Strict loading failed. Retrying with strict=False. Error: {e}"
        )
        model.load_state_dict(new_state_dict, strict=False)

    return model
