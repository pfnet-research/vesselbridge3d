"""3D pyramidal ConvNeXt adapter producing multi-scale 3D features."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .blocks import ConvNeXtAniso3DBlock


# 3D Pyramidal Adapter (ConvNeXt)
class AdapterPyramid3DConvNeXt(nn.Module):
    """
    input:  x_bd1hw = [B, D, 1, H, W]
    output:  F2_3d = [B, C2, D, H/2, W/2],  F3_3d = [B, C3, D, H/4, W/4]
    Each stage consists of 2 blocks.
    The first block performs downsampling with a stride of (1, 2, 2).
    Depthwise Convolution (DWConv) uses an anisotropic kernel of (3, 7, 7).
    """
    def __init__(self, c2: int = 48, c3: int = 64, in_ch: int = 1, drop_path: float = 0.05):
        super().__init__()
        # Stage2 (1/2)
        self.s2_down = ConvNeXtAniso3DBlock(in_ch, c2, kernel=(3, 7, 7), stride=(1, 2, 2),
                                            drop_path=drop_path, use_depth_branch=True)
        self.s2_blk  = ConvNeXtAniso3DBlock(c2, c2, kernel=(3, 7, 7), stride=(1, 1, 1),
                                            drop_path=drop_path, use_depth_branch=True)
        # Stage3 (1/4)
        self.s3_down = ConvNeXtAniso3DBlock(c2, c3, kernel=(3, 7, 7), stride=(1, 2, 2),
                                            drop_path=drop_path, use_depth_branch=True)
        self.s3_blk  = ConvNeXtAniso3DBlock(c3, c3, kernel=(3, 7, 7), stride=(1, 1, 1),
                                            drop_path=drop_path, use_depth_branch=True)

    def forward(self, x_bd2hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_bd2hw: [B,D,2,H,W] -> [B,1,D,H,W]
        B, D, C, H, W = x_bd2hw.shape
        assert C == 2
        x = x_bd2hw.permute(0, 2, 1, 3, 4).contiguous()

        f2 = self.s2_blk(self.s2_down(x))     # [B, C2, D, H/2, W/2]
        f3 = self.s3_blk(self.s3_down(f2))    # [B, C3, D, H/4, W/4]
        return f2, f3
