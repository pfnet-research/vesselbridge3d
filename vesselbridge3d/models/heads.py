"""Segmentation refinement heads (3D refine + 2D high-resolution)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SEBlock2D, SEBlock3D


class Refine3DHead(nn.Module):
    """
    input:  [B, (K + Cref), D, H/2, W/2]
    output:  [B, K, D, H/2, W/2]
    structure: (1,3,3) center of 2D path + lightweight 3D path made with (3,1,1)/(1,3,3) (DepthGate)
    """

    def __init__(
        self, in_ch: int, num_classes: int, mid: int = 128, use_se: bool = True
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(
            in_ch, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn_in = nn.GroupNorm(8, mid)
        self.act = nn.SiLU(inplace=False)

        # 2D branch (in-plane)
        self.conv2d = nn.Conv3d(
            mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn2d = nn.GroupNorm(8, mid)

        # 3D branch (shallow depth mixing, anisotropic)
        self.conv3d_z = nn.Conv3d(
            mid, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.conv3d_hw = nn.Conv3d(
            mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
        )
        self.gn3d = nn.GroupNorm(8, mid)

        self.depth_gate = nn.Conv3d(in_ch, 1, kernel_size=1)

        self.se = SEBlock3D(mid) if use_se else nn.Identity()
        self.proj = nn.Conv3d(mid, num_classes, kernel_size=1)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: [B, K+Cref, D, H/2, W/2]
        g = torch.sigmoid(self.depth_gate(x_cat))  # [B,1,D,H/2,W/2]

        h = self.act(self.gn_in(self.conv_in(x_cat)))

        h2d = self.act(self.gn2d(self.conv2d(h)))
        h3d = self.conv3d_hw(self.conv3d_z(h))
        h3d = self.act(self.gn3d(h3d))

        h = h2d + g * h3d
        h = self.se(h)
        return self.proj(h)  # [B,K,D,H/2,W/2]


class HRHead2D(nn.Module):
    """
    input (1/1): [B2D, K+64, H, W] (K=classes, 64=F2 channels)
    output (1/1): [B2D, K, H, W]
    3x3 → DW 3x3 → 1x1 (+SE) → 1x1(classifier)
    """

    def __init__(
        self, in_ch: int, num_classes: int, mid: int = 64, use_se: bool = True
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False)
        self.gn_in = nn.GroupNorm(8, mid)
        self.act = nn.SiLU(inplace=False)

        self.dw = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
        self.gn_dw = nn.GroupNorm(8, mid)

        self.pw = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
        self.gn_pw = nn.GroupNorm(8, mid)

        self.se = SEBlock2D(mid) if use_se else nn.Identity()
        self.cls = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn_in(self.conv_in(x)))
        h = self.act(self.gn_dw(self.dw(h)))
        h = self.act(self.gn_pw(self.pw(h)))
        h = self.se(h)
        out = self.cls(h)
        return out
