"""Segmentation losses: soft Dice and combined CE+Dice (+optional topology)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMultiClass(nn.Module):
    """
    Soft Dice over present classes (background excluded by default).
    Works for 2D or 3D.
    """

    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # logits: [B,K,H,W] or [B,K,D,H,W]
        # target: [B,H,W]   or [B,D,H,W]
        K = logits.shape[1]
        probs = logits.float().softmax(dim=1)
        # onehot: [...,K] -> [B,K,*]
        onehot = (
            F.one_hot(torch.clamp(target, min=0), num_classes=K).movedim(-1, 1).float()
        )

        start_c = 1 if self.exclude_bg and K > 1 else 0
        probs = probs[:, start_c:]
        onehot = onehot[:, start_c:]

        # optional mask (exclude padded voxels from sums)
        if mask is not None:
            m = mask.unsqueeze(1).float()  # [B,1,*] -> broadcast on class dim
            probs = probs * m
            onehot = onehot * m

        dims = tuple(range(2, probs.ndim))  # sum over spatial dims
        intersection = (probs * onehot).sum(dim=dims)
        probs_sum = probs.sum(dim=dims)
        onehot_sum = onehot.sum(dim=dims)

        dice_bc = (2.0 * intersection + self.eps) / (probs_sum + onehot_sum + self.eps)
        present_bc = (onehot_sum > 0).float()  # only classes present in GT contribute
        present_count = present_bc.sum()
        dice = (dice_bc * present_bc).sum() / torch.clamp(present_count, min=1.0)
        return 1.0 - dice


class DC_and_CE_loss(nn.Module):
    """
    Returns (for backward-compat):
      - ce_loss
      - dice_loss
      - total_loss = weight_ce * CE + weight_dice * Dice + (optional topology terms)

    If topology terms are enabled, this module also returns a 4th value (topology_loss).
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        use_topk: bool = False,
        topk_percent: float = 10.0,
        # topology
        use_cldice: bool = False,
        cldice_weight: float = 0.0,
        cldice_iters: int = 20,
        use_skelrecall: bool = False,
        skelrecall_weight: float = 0.0,
        skelrecall_iters: int = 20,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.dc = DiceLossMultiClass(num_classes=num_classes, exclude_bg=True)
        self.wce = float(weight_ce)
        self.wdc = float(weight_dice)
        self.use_topk = bool(use_topk)
        self.topk_percent = float(topk_percent)

        # topology
        self.use_cldice = bool(use_cldice)
        self.cldice_w = float(cldice_weight)
        self.use_skelrecall = bool(use_skelrecall)
        self.skelrecall_w = float(skelrecall_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        IGNORE = 255
        if mask is not None:
            target_ce = target.clone()
            target_ce[~mask] = IGNORE
        else:
            target_ce = target

        w = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        if self.use_topk:
            ce_loss = self.topk_ce(logits, target_ce, mask=mask)
        else:
            ce_loss = F.cross_entropy(
                logits.float(), target_ce, weight=w, ignore_index=IGNORE
            )
        # cast fp32
        dice_loss = self.dc(logits.float(), target, mask=mask)

        topo_loss = logits.sum() * 0.0  # default 0
        if self.use_cldice and self.cldice_w > 0.0:
            topo_loss = (
                topo_loss + self.cldice(logits, target, mask=mask) * self.cldice_w
            )
        if self.use_skelrecall and self.skelrecall_w > 0.0:
            topo_loss = (
                topo_loss
                + self.skelrecall(logits, target, mask=mask) * self.skelrecall_w
            )

        total_loss = self.wce * ce_loss + self.wdc * dice_loss + topo_loss
        if (self.use_cldice and self.cldice_w > 0.0) or (
            self.use_skelrecall and self.skelrecall_w > 0.0
        ):
            return ce_loss, dice_loss, topo_loss, total_loss
        else:
            return ce_loss, dice_loss, total_loss
