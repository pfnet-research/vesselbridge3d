"""Input-size parsing and inference-time volume preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_img_size(s: str) -> Tuple[int, int]:
    """
    '224' -> (224 ,224), '512' -> (512, 512)
    """
    s = str(s).strip().lower().replace('x', ',')
    parts = [p for p in s.split(',') if p]
    if len(parts) == 1:
        h = w = int(parts[0])
    elif len(parts) == 2:
        h, w = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Bad --img_size: {s}")
    return h, w


def parse_layers(s: str) -> Tuple[int, ...]:
    """'2,5,8,11' -> (2, 5, 8, 11)."""
    return tuple(int(t) for t in str(s).replace(' ', '').split(',') if t != '')


def preprocess_with_z(data_numpy, target_h, target_w):
    """
    Input: (H_orig, W_orig, D_orig)
    Output: Tensor (1, D_orig, 2, H_target, W_target) -> [B, D, C, H, W]
    """
    H_orig, W_orig, D_orig = data_numpy.shape

    z_indices = np.arange(0, D_orig, dtype=np.float32) / max(1.0, float(D_orig))
    # [D, 1, 1, 1]
    z_tensor = torch.from_numpy(z_indices).view(D_orig, 1, 1, 1)

    # (H, W, D) -> (D, 1, H, W)
    img_tensor = torch.from_numpy(data_numpy).float().permute(2, 0, 1).unsqueeze(1)

    # [D, 1, Ht, Wt]
    img_resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # [D, 1, Ht, Wt]
    z_channel = z_tensor.expand(D_orig, 1, target_h, target_w)

    # [D, 2, Ht, Wt] (Channel 0: Image, Channel 1: Z)
    x = torch.cat([img_resized, z_channel], dim=1)

    # [1, D, 2, Ht, Wt]
    x = x.unsqueeze(0)

    return x
