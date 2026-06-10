"""Low-level reusable neural building blocks (SE, DropPath, ConvNeXt, FFN)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth (per sample).
    Ref: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if (not self.training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: [B, 1, 1, 1, 1] to broadcast over [B,C,D,H,W]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.avg(x))


class SEBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(self.avg(x))


class UpBlock3DSkip(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, use_se: bool = True):
        super().__init__()
        self.proj_in = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False)
        self.proj_skip = nn.Conv3d(c_skip, c_out, kernel_size=1, bias=False)
        self.conv1 = nn.Conv3d(2 * c_out, c_out, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.gn1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.gn2 = nn.GroupNorm(8, c_out)
        self.act = nn.SiLU(inplace=False)
        self.se = SEBlock3D(c_out) if use_se else nn.Identity()

    def forward(self, x, skip):
        # upsample H/W 2x（Depth: keep same）
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        x = self.proj_in(x)
        skip = F.interpolate(skip, size=x.shape[-3:], mode="trilinear", align_corners=False)
        skip = self.proj_skip(skip)
        h = torch.cat([x, skip], dim=1)
        h = self.act(self.gn1(self.conv1(h)))
        h = self.act(self.gn2(self.conv2(h)))
        return self.se(h)


class ConvNeXtAniso3DBlock(nn.Module):
    """
    ConvNeXt-style 3D block with anisotropic DWConv3D,
    LayerNorm (channels-last), and PW-MLP (1x1x1).

    Additionally, a lightweight branch with DWConv (3,1,1) is added for depth mixing.
    - When stride=(1,2,2), it performs in-plane downsampling.
    - If input/output channels or spatial size differ, a 1x1x1 (with stride if needed)
        is used for the residual path.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int, int] = (3, 7, 7),
        stride: Tuple[int, int, int] = (1, 1, 1),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-6,
        use_depth_branch: bool = True,
    ):
        super().__init__()
        g = in_ch
        pad = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)

        # Depthwise
        self.dw = nn.Conv3d(in_ch, in_ch, kernel, stride=stride, padding=pad, groups=g, bias=True)

        self.use_depth_branch = bool(use_depth_branch)
        if self.use_depth_branch:
            kz = (3, 1, 1)
            self.dw_z = nn.Conv3d(in_ch, in_ch, kz, stride=stride,
                                   padding=(kz[0] // 2, 0, 0), groups=g, bias=True)

        # channels-last LayerNorm
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)

        # pointwise-MLP (1x1x1 conv → GELU → 1x1x1 conv)
        hidden = int(in_ch * mlp_ratio)
        self.pw1 = nn.Conv3d(in_ch, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, out_ch, kernel_size=1, bias=True)

        # residual path
        self.need_proj = (in_ch != out_ch) or (stride != (1, 1, 1))
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=True) if self.need_proj else nn.Identity()

        self.gamma = nn.Parameter(torch.ones((out_ch, 1, 1, 1)) * layer_scale_init)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DW
        y = self.dw(x)
        if self.use_depth_branch:
            y = y + self.dw_z(x)

        # channels-last LN
        y = y.permute(0, 2, 3, 4, 1)  # [B,D,H,W,C]
        y = self.norm(y)
        y = y.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]

        # PW-MLP
        y = self.pw2(self.act(self.pw1(y)))

        # Residual + DropPath + LayerScale
        y = self.drop_path(self.gamma * y) + self.proj(x)
        return y


class FFN3D_Pointwise(nn.Module):
    """
    pointwise-MLP (1x1x1) (Conv→GELU→Drop→Conv)
    """
    def __init__(self, c: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        h = int(hidden or (4 * c))
        self.fc1 = nn.Conv3d(c, h, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Conv3d(h, c, kernel_size=1, bias=True)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))
