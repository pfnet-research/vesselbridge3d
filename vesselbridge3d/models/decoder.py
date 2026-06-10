"""UNETR-Lite 3D segmentation decoder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .blocks import UpBlock3DSkip


class SegDecoder3D_UNETRLite(nn.Module):
    """
    - input: bottom [B,Cb,D,Gh,Gw], skips: List[[B,Cs,D,Gh,Gw]] (all 1/16 resolution)
    - Internally interpolates each skip connection to the stage's resolution and concatenates them.
    - Creates 3 upsampling blocks for up_factor_hw=8.
    - (Reuse the last skip if there are too few; discard the first ones if there are too many.)
    """
    def __init__(self, c_in: int, c_skip: int, num_classes: int,
                 up_factor_hw: int = 4, base_channels: int = 128, use_se: bool = True):
        super().__init__()
        steps = []
        f = up_factor_hw
        while f > 1:
            steps.append(2); f //= 2
        self.num_stages = max(1, len(steps))

        self.stem = nn.Sequential(
            nn.Conv3d(c_in, base_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=False),
        )

        blocks = []
        for _ in range(self.num_stages):
            blocks.append(UpBlock3DSkip(base_channels, c_skip, base_channels, use_se=use_se))
        self.blocks = nn.ModuleList(blocks)

        self.refine = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(inplace=False),
        )
        self.classifier = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, bottom, skip_list: List[torch.Tensor]):
        h = self.stem(bottom)
        skips = list(skip_list)
        if len(skips) >= self.num_stages:
            skips = skips[-self.num_stages:]  # from deepest
        else:
            # when there are too few skip connections, repeat the last one
            while len(skips) < self.num_stages:
                skips.append(skips[-1])

        # low resolution -> high resolution
        for blk, sk in zip(self.blocks, reversed(skips)):
            h = blk(h, sk)

        h = self.refine(h)
        return self.classifier(h)  # [B,K,D,H,W]
