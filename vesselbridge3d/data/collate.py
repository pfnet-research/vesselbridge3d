"""Collate functions for padding variable-depth 3D volumes into batches."""

from __future__ import annotations

import torch


def collate_pad_3d(batch):
    max_d = max(x.shape[0] for x, _ in batch)
    B = len(batch)
    # x: [D, C, H, W]
    C, H, W = batch[0][0].shape[1:]

    X = batch[0][0].new_zeros((B, max_d, C, H, W))
    Y = torch.zeros((B, max_d, H, W), dtype=batch[0][1].dtype)
    M = torch.zeros((B, max_d, H, W), dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        d = x.shape[0]
        X[i, :d] = x
        Y[i, :d] = y
        M[i, :d] = True
    return X, Y, M


def make_collate_pad_3d(depth_min_fg_frac: float | None = None):
    thr = None if depth_min_fg_frac is None else float(depth_min_fg_frac)

    def _collate(batch):
        X, Y, M = collate_pad_3d(batch)
        if thr is None or thr <= 0.0:
            return X, Y, M

        B, Dmax, _, H, W = X.shape
        for b in range(B):
            valid_depths = M[b, :, 0, 0]         # [Dmax] True/False
            d = int(valid_depths.sum().item())   # valid depth num
            if d == 0:
                continue

            yb = Y[b, :d]                        # [d,H,W]
            fg_counts = (yb > 0).float().view(d, -1).sum(dim=1)  # [d]
            frac = fg_counts / float(H * W)                       # [d]
            good = (frac >= thr)                                  # [d] bool

            M[b, :d] &= good.view(d, 1, 1).expand(d, H, W)

        return X, Y, M

    return _collate
