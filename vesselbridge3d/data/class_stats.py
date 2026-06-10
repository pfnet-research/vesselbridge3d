"""Class-imbalance utilities: histograms, class weights, prior-bias init."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# Utilities for imbalance
def compute_class_histogram(ds, num_classes: int) -> np.ndarray:
    """
    Compute label histogram over the dataset.

    Supports:
      - SingleCaseSliceDataset: uses ds.slices (list of (img2d, label2d))
      - StackedSliceDataset25D: uses ds.idxs + ds.segs (central-slice labels)
      - Fallback: iterate dataset and read y from __getitem__ (slower)
    """
    counts = np.zeros((num_classes,), dtype=np.int64)

    # Case 1: original 2D dataset
    if hasattr(ds, "slices"):
        for _, lb in ds.slices:
            counts += np.bincount(lb.reshape(-1), minlength=num_classes)
        return counts

    # Case 2: 2.5D dataset (late fusion)
    if hasattr(ds, "idxs") and hasattr(ds, "segs"):
        # count labels on the central slice for all sampled indices
        for vol_idx, z in ds.idxs:
            lb = ds.segs[vol_idx][:, :, z]
            counts += np.bincount(lb.reshape(-1), minlength=num_classes)
        return counts

    # Case 3: 3D VolumeDataset
    if hasattr(ds, "segs") and not hasattr(ds, "idxs"):
        for vol in ds.segs:
            counts += np.bincount(vol.reshape(-1), minlength=num_classes)
        return counts

    # Fallback: generic dataset (may be slower)
    for i in range(len(ds)):
        _, y = ds[i]  # y is [H,W]
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        counts += np.bincount(y.reshape(-1), minlength=num_classes)

    return counts

def make_class_weights(num_classes: int, counts: np.ndarray, bg_weight: float, use_mfb: bool) -> torch.Tensor:
    if use_mfb:
        freq = counts.astype(np.float64)
        freq = np.maximum(freq, 1.0)
        freq = freq / freq.sum()
        med = np.median(freq[freq > 0])
        weights = med / np.maximum(freq, 1e-12)
    else:
        weights = np.ones((num_classes,), dtype=np.float64)
    weights[0] = float(bg_weight)
    return torch.tensor(weights, dtype=torch.float32)


def init_prior_bias_for_head(head: nn.Module, num_classes: int, counts: np.ndarray):
    cls_conv = None
    for m in head.modules():
        # 2D or 3D
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if m.out_channels == num_classes:
                cls_conv = m
                break
    if cls_conv is None:
        raise RuntimeError(f"Could not find classifier Conv(out_channels={num_classes}) in head.")

    probs = counts.astype(np.float64)
    probs = np.maximum(probs, 1.0)
    probs = probs / probs.sum()
    bias = np.log(probs)

    with torch.no_grad():
        if cls_conv.bias is None:
            cls_conv.bias = nn.Parameter(torch.zeros(cls_conv.out_channels,
                                                     device=cls_conv.weight.device,
                                                     dtype=cls_conv.weight.dtype))
        cls_conv.bias.copy_(torch.from_numpy(bias).to(cls_conv.weight.device,
                                                      dtype=cls_conv.weight.dtype))

    return cls_conv
