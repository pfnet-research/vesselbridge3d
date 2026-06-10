"""
3D segmentation model based on DINOv3 ViT-S/16 (per-slice 2D encoder + 3D aggregator/decoder).

Changes in this version:
- Train/Val lists: accepts --train_list and --val_list (fallback to --list_json if needed).
- Always load DINOv3 ViT-S/16 from Hugging Face (no timm fallback).
- Explicitly DROP 5 special tokens (CLS=1 + Register=4) from last_hidden_state.
- Hybrid loss: CE + Dice with class weighting, foreground-slice filtering, and
  optional prior-bias initialization for the 1x1 head.
- Log CE, Dice, Topo, Total per train/eval.
- Save color (GT | Pred) image every 20 epochs to `vis_outputs/`.

Architecture:
  Frozen Adapter ([0,1] min-max + ImageNet norm, 2ch (Image + Z-coord) -> 3ch, NO learnable params)
    -> Frozen DINOv3 ViT-S/16 (2D encoder, outputs patch embeddings)
    -> 3D Pyramidal Adapter (ConvNeXt-style, anisotropic DWConv)
    -> Shared Parallel FFA Aggregator (Slice Self-Attn + Global Spatial Attn + FFN)
    -> UNETR-Lite Decoder (1/16 -> 1/4 with skip connections)
    -> Refine3DHead (DW (1,3,3) + GroupNorm + SiLU + SE + 1x1, 3D depth-gated)
    -> HRHead2D (DW 3x3 + GroupNorm + SiLU + SE + 1x1, per-slice chunked)

Example:
    uv run python train.py \
      --train_list /path/to/train.json \
      --val_list /path/to/val.json \
      --num_classes 16 \
      --epochs 300 \
      --batch_size 2 \
      --accumulation_steps 1 \
      --img_size 336 \
      --lr 5e-3 \
      --drop_empty \
      --min_fg_frac 0.0 \
      --bg_weight 0.05 \
      --use_mfb \
      --init_prior_bias \
      --aug \
      --use_3d \
      --use_3d_unetr \
      --depth_min_fg_frac 0.0005 
"""

from __future__ import annotations
import argparse, json, math, os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import nibabel as nib
from transformers import AutoModel
import imageio, random
from colorsys import hsv_to_rgb

from contextlib import nullcontext

import torch.utils.checkpoint as cp
import math

from datetime import datetime

from .config_utils import add_config_arg, parse_args_with_config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def _rbool(p: float) -> bool:
    return bool(torch.rand(1) < p)

def _runif(a:float, b:float) -> float:
    return float(torch.empty(1).uniform_(a, b))


def make_palette(num_classes: int) -> np.ndarray:
    """
    Create a bright, high-contrast palette.
    0=background -> black, others evenly spaced hues in HSV (S=0.85, V=0.95).
    """
    num_classes = max(1, int(num_classes))
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    if num_classes <= 1:
        return palette
    for c in range(1, num_classes):
        h = (c - 1) / max(1, num_classes - 1)
        s, v = 0.85, 0.95
        r, g, b = hsv_to_rgb(h, s, v)
        palette[c] = (int(r * 255), int(g * 255), int(b * 255))
    return palette

def colorize(label_2d: np.ndarray, palette: np.ndarray) -> np.ndarray:
    label_safe = np.clip(label_2d.astype(np.int64), 0, len(palette) - 1)
    return palette[label_safe]

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


class FrozenAdapter(nn.Module):
    """
    [B, 2, H, W] -> [B, 3, H, W]
    Channel 0: Image (min-max norm)
    Channel 1: Z-coord (from dataset)
    Output: R=Image, G=Image, B=Z-coord -> ImageNet Normalize
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', IMAGENET_MEAN)
        self.register_buffer('std', IMAGENET_STD)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        assert C == 2, f"Expected 2-channel input (Image + Z), got {C}"
        
        img = x[:, 0:1, :, :]   # [B, 1, H, W]
        z_map = x[:, 1:2, :, :] # [B, 1, H, W]
        
        x_min = img.amin(dim=(2,3), keepdim=True)
        x_max = img.amax(dim=(2,3), keepdim=True)
        img01 = (img - x_min) / (x_max - x_min + 1e-6)
        
        # (R=Img, G=Img, B=Z)
        x3 = torch.cat([img01, img01, z_map], dim=1) # [B, 3, H, W]
        
        x3 = (x3 - self.mean) / self.std
        return x3

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

class Refine3DHead(nn.Module):
    """
    input:  [B, (K + Cref), D, H/2, W/2]  
    output:  [B, K, D, H/2, W/2]
    structure: (1,3,3) center of 2D path + lightweight 3D path made with (3,1,1)/(1,3,3) (DepthGate)
    """
    def __init__(self, in_ch: int, num_classes: int, mid: int = 128, use_se: bool = True):
        super().__init__()
        self.conv_in = nn.Conv3d(in_ch, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn_in   = nn.GroupNorm(8, mid)
        self.act     = nn.SiLU(inplace=False)

        # 2D branch (in-plane)
        self.conv2d = nn.Conv3d(mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn2d   = nn.GroupNorm(8, mid)

        # 3D branch (shallow depth mixing, anisotropic)
        self.conv3d_z = nn.Conv3d(mid, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv3d_hw= nn.Conv3d(mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn3d     = nn.GroupNorm(8, mid)

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

class FrozenDINOv3ViTS16(nn.Module):
    def __init__(self, hf_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m", default_layers: Tuple[int, ...] = (2, 5, 8, 11)):
        super().__init__()
        self.vit = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        self.hidden_dim = int(getattr(self.vit.config, 'hidden_size', 384))
        self.patch = int(getattr(self.vit.config, 'patch_size', 16))
        self.num_special_tokens = 5  # CLS(1) + Register(4)
        self.default_layers = tuple(int(i) for i in default_layers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Gh, Gw = H // self.patch, W // self.patch
        outputs = self.vit(pixel_values=x, return_dict=True)
        tokens = outputs.last_hidden_state  # [B, seq, C]
        patch_tokens = tokens[:, self.num_special_tokens:self.num_special_tokens + Gh * Gw, :]
        return patch_tokens.transpose(1, 2).contiguous().view(B, self.hidden_dim, Gh, Gw)

    @torch.no_grad()
    def forward_multi(self, x: torch.Tensor, layers: Tuple[int, ...] | None = None) -> List[torch.Tensor]:
        """
        Returns output (patch tokens only) 
        from specified Transformer blocks as 2D grids.
        """
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0
        Gh, Gw = H // self.patch, W // self.patch
        N = Gh * Gw

        outputs = self.vit(pixel_values=x, output_hidden_states=True, return_dict=True)
        hiddens = outputs.hidden_states  # L+1

        L_total = len(hiddens) - 1  
        layers = tuple(self.default_layers if layers is None else layers)

        grids: List[torch.Tensor] = []
        for li in layers:
            idx = (L_total if li == -1 else int(li))
            # hidden_states は [0]=embeddings, [1..L_total]=each block
            idx = max(1, min(idx, L_total))
            tok = hiddens[idx]  # [B, seq, C]
            patch_tokens = tok[:, self.num_special_tokens:self.num_special_tokens + N, :]
            grid = patch_tokens.transpose(1, 2).contiguous().view(B, self.hidden_dim, Gh, Gw)
            grids.append(grid)
        return grids

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

class HRHead2D(nn.Module):
    """
    input (1/1): [B2D, K+64, H, W] (K=classes, 64=F2 channels)
    output (1/1): [B2D, K, H, W]
    3x3 → DW 3x3 → 1x1 (+SE) → 1x1(classifier) 
    """
    def __init__(self, in_ch: int, num_classes: int, mid: int = 64, use_se: bool = True):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, mid, kernel_size=3, padding=1, bias=False)
        self.gn_in   = nn.GroupNorm(8, mid)
        self.act     = nn.SiLU(inplace=False)

        self.dw      = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
        self.gn_dw   = nn.GroupNorm(8, mid)

        self.pw      = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
        self.gn_pw   = nn.GroupNorm(8, mid)

        self.se      = SEBlock2D(mid) if use_se else nn.Identity()
        self.cls     = nn.Conv2d(mid, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h  = self.act(self.gn_in(self.conv_in(x)))
        h  = self.act(self.gn_dw(self.dw(h)))
        h  = self.act(self.gn_pw(self.pw(h)))
        h  = self.se(h)
        out = self.cls(h)
        return out

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
    

# SegModel3D_UNETRLite (3D ConvNeXt Adapter + 3DGate + Refine3D) 
class SegModel3D_UNETRLite(nn.Module):
    """
    x: [B,D,1,H,W] ->
      Adapter3DConvNeXt (F2@1/2, F3@1/4)
      -> Frozen DINOv3 ViT (per-slice) -> multi-layer patch grids
      -> Shared Axial Aggregator 
      -> Decoder(UNETR-Lite): bottom = last layer, skips = others
      -> Up to 1/4 (H/4,W/4), then upsample to 1/2 and Refine3D
    """
    def __init__(self, num_classes: int,
                 vit_layers: Tuple[int, ...] = (2, 5, 8, 11),
                 decoder_base_channels: int = 128, decoder_up_factor: int = 4,
                 vit_chunk_slices: int = 8, vit_amp: bool = True, use_se: bool = True):
        super().__init__()
        # Frozen per-slice adapter for ViT encoder (min-max + ImageNet norm 1ch->3ch)
        self.adapter_slice = FrozenAdapter();  [p.requires_grad_(False) for p in self.adapter_slice.parameters()]
        self.encoder = FrozenDINOv3ViTS16(default_layers=vit_layers); [p.requires_grad_(False) for p in self.encoder.parameters()]
        C = int(self.encoder.hidden_dim)
        self.patch = int(self.encoder.patch)
        self.selected_layers = tuple(vit_layers)

        # 3D Pyramidal Adapter (ConvNeXt)
        self.adapter3d = AdapterPyramid3DConvNeXt(c2=48, c3=64, in_ch=2, drop_path=0.05)
        self.f2_proj_ref = nn.Conv3d(48, 64, kernel_size=1, bias=False)  # F2 -> 64 (refine用)
        self.f3_proj_C   = nn.Conv3d(64, C,   kernel_size=1, bias=False)  # F3 -> C (ViT hidden)
        self.f3_gate3d   = nn.Conv3d(C + C, 1, kernel_size=1)             # 3D gate σ([A5_up, F3pC])

        # Shared parallel axial aggregator 
        self.shared_agg = ParallelAggregatorSharedFFA(
            c=C, num_layers=len(vit_layers),
            n_blocks=2,           
            heads=6,
            attn_dim=C // 2,    
            kv_down=2,            
            dropout=0.0,
            drop_path=0.05,
            use_rope=True,
            use_pos_slice=True
        )

        # Decoder (UNETR-lite 1/16 -> 1/4)
        self.dec = SegDecoder3D_UNETRLite(
            c_in=C, c_skip=C, num_classes=num_classes,
            up_factor_hw=decoder_up_factor, base_channels=decoder_base_channels, use_se=use_se
        )

        # Refine head 
        self.head = Refine3DHead(in_ch=(num_classes + 64), num_classes=num_classes, mid=decoder_base_channels, use_se=True)
        self.hr2d = HRHead2D(in_ch=(num_classes + 16), num_classes=num_classes,
                            mid=decoder_base_channels, use_se=True)
        self.f2_hr_reduce3d = nn.Conv3d(64, 16, kernel_size=1, bias=False) # For HR
        self.hr2d_chunk = 16

        self.vit_chunk_slices = int(vit_chunk_slices)
        self.vit_amp = bool(vit_amp)

    def trainable_parameters(self):
        # Aggregator + Decoder + 3D Adapter + gates + Refine3D
        return (list(self.shared_agg.parameters())
                + list(self.dec.parameters())
                + list(self.adapter3d.parameters())
                + list(self.f3_proj_C.parameters()) + list(self.f3_gate3d.parameters())
                + list(self.f2_proj_ref.parameters()) + list(self.head.parameters())
                + list(self.hr2d.parameters()))

    def _stack_depth(self, feats_2d: List[torch.Tensor], B: int, D: int) -> List[torch.Tensor]:
        outs = []
        for f in feats_2d:
            _, C, Gh, Gw = f.shape
            outs.append(f.view(B, -1, C, Gh, Gw).permute(0, 2, 1, 3, 4).contiguous())
        return outs

    def forward(self, x_bdchw: torch.Tensor) -> torch.Tensor:
            # x: [B,D,2,H,W]
            B, D, C_in, H, W = x_bdchw.shape
            assert C_in == 2, "Input must have 2 channels (Image + Z)"

            # Per-slice ViT encoding (frozen)
            x_slices = x_bdchw.view(-1, 2, H, W)
            feats_per_layer: List[List[torch.Tensor]] = []
            
            # ViT inference
            for s in range(0, x_slices.size(0), self.vit_chunk_slices):
                xi = x_slices[s:s + self.vit_chunk_slices] # [chunk, 2, H, W]
                xi3 = self.adapter_slice(xi)  # [chunk,3,H,W]
                with torch.no_grad():
                    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if (self.vit_amp and xi3.is_cuda) else nullcontext()
                    with ctx:
                        grids = self.encoder.forward_multi(xi3, layers=self.selected_layers)
                for k, gi in enumerate(grids):
                    if len(feats_per_layer) <= k:
                        feats_per_layer.append([])
                    tgt_dtype = torch.bfloat16 if (self.vit_amp and xi3.is_cuda) else torch.float32
                    feats_per_layer[k].append(gi.to(tgt_dtype))
             
            del x_slices, xi, xi3

            feats_2d: List[torch.Tensor] = [torch.cat(vs, dim=0) for vs in feats_per_layer]
            del feats_per_layer 

            feats_3d: List[torch.Tensor] = self._stack_depth(feats_2d, B=B, D=D)
            del feats_2d 

            # Shared Aggregator -> Checkpoint
            if self.training:
                agg_outs = cp.checkpoint(self.shared_agg, feats_3d, use_reentrant=False)
            else:
                agg_outs = self.shared_agg(feats_3d)
            
            del feats_3d 

            *skips, bottom = agg_outs
            if len(skips) < 2:
                raise RuntimeError("Need at least two skip features for up_factor=4")
            A5, A8 = skips[-2], skips[-1]     # [B,C,D,H/16,W/16]
            A11    = bottom                   # [B,C,D,H/16,W/16]

            # 3D Adapter -> Checkpoint
            if self.training:
                F2_3d, F3_3d = cp.checkpoint(self.adapter3d, x_bdchw, use_reentrant=False)
            else:
                F2_3d, F3_3d = self.adapter3d(x_bdchw)
            
            # Projections
            F2p = self.f2_proj_ref(F2_3d)
            F3pC = self.f3_proj_C(F3_3d)

            # Gate fusion
            A5_up = F.interpolate(A5, size=F3pC.shape[-3:], mode="trilinear", align_corners=False)
            gate3d = torch.sigmoid(self.f3_gate3d(torch.cat([A5_up, F3pC], dim=1)))
            skip2 = A5_up + gate3d * F3pC

            # Decoder 
            if self.training:
                logits_56 = cp.checkpoint(self.dec, A11, [skip2, A8], use_reentrant=False)
            else:
                logits_56 = self.dec(A11, [skip2, A8])

            # Refine Head
            logits_112 = F.interpolate(logits_56, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            ref_in = torch.cat([logits_112, F2p], dim=1)
            
            if self.training:
                logits_ref = cp.checkpoint(self.head, ref_in, use_reentrant=False)
            else:
                logits_ref = self.head(ref_in)

            # HR Head 
            F2r_3d = self.f2_hr_reduce3d(F2p.detach()) 

            # Upsample to 1/1
            logits_1x1 = F.interpolate(logits_ref, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
            F2r_1x1    = F.interpolate(F2r_3d,     scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)

            # Output buffer
            B, K, D, H, W = logits_1x1.shape
            hr_logits = logits_1x1.new_empty((B, K, D, H, W))

            # Chunked HR processing
            for s in range(0, D, self.hr2d_chunk):
                e = min(D, s + self.hr2d_chunk)
                hr_in_chunk = torch.cat([logits_1x1[:, :, s:e], F2r_1x1[:, :, s:e]], dim=1)
                hr_in_2d = hr_in_chunk.permute(0, 2, 1, 3, 4).reshape(B * (e - s), K + 16, H, W)

                if self.training:
                    out2d = cp.checkpoint(self.hr2d, hr_in_2d, use_reentrant=False)
                else:
                    out2d = self.hr2d(hr_in_2d)

                out_chunk = out2d.view(B, (e - s), K, H, W).permute(0, 2, 1, 3, 4).contiguous()
                hr_logits[:, :, s:e] = out_chunk

            return hr_logits

# Layer-ID FiLM (per-layer scale/bias; weight-tying adapter) 
class LayerFiLM(nn.Module):
    def __init__(self, c: int, num_layers: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_layers, c))  # multiplicative
        self.beta  = nn.Parameter(torch.zeros(num_layers, c))  # additive
    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        g = (1.0 + self.gamma[layer_idx]).view(1, -1, 1, 1, 1)
        b = self.beta[layer_idx].view(1, -1, 1, 1, 1)
        return x * g + b


class RotaryPositionalEmbedding1D(nn.Module):
    """
    1D (Depth-only) Rotary Embedding.
    - Assume head_dim is even number
    """
    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        self.dim = dim
        self.base = float(base)
        self.scale = float(scale)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, D: int, device, dtype):
        # 0 ... D-1 integers for now
        t = torch.arange(D, device=device, dtype=torch.float32) * self.scale # fp32
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        cos = freqs.cos()[None, None, :, :] # [1, 1, D, dim/2]
        sin = freqs.sin()[None, None, :, :] # [1, 1, D, dim/2]
        # cast q/k's dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   [BHW, heads, D, dim] 
    cos: [1, 1,   D, dim/2]
    sin: [1, 1,   D, dim/2]
    """
    x_even = x[..., ::2]   # [BHW,h,D,dim/2]
    x_odd  = x[..., 1::2]  # [BHW,h,D,dim/2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd  = x_even * sin + x_odd * cos

    x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1)  # [..., dim/2, 2]
    x_rot = x_rot.flatten(-2)                             # [..., dim]
    return x_rot


# FFA core (Slice Attn + Global Spatial Attn + FFN, with pos_slice) 
class SliceSelfAttention1D(nn.Module):
    """
    input: [B,C,D,H,W]
    output: [B,C,D,H,W]
    apply Attn only Depth axis, keeping H/W as batch dimensions for efficiency.
    """
    def __init__(self, c: int, heads: int = 6, attn_dim: int | None = None, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        self.c = c
        self.heads = int(heads)
        self.attn_dim = int(attn_dim or c)
        assert self.attn_dim % self.heads == 0
        self.head_dim = self.attn_dim // self.heads
        self.use_rope = bool(use_rope)

        self.ln = nn.LayerNorm(c)
        self.q = nn.Linear(c, self.attn_dim, bias=False)
        self.k = nn.Linear(c, self.attn_dim, bias=False)
        self.v = nn.Linear(c, self.attn_dim, bias=False)
        self.proj = nn.Linear(self.attn_dim, c, bias=False)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryPositionalEmbedding1D(self.head_dim) if self.use_rope else None

    def forward(self, x_bcdhw: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x_bcdhw.shape
        x = x_bcdhw.permute(0, 3, 4, 2, 1).contiguous()      # [B,H,W,D,C]
        x = self.ln(x)
        x_seq = x.view(B * H * W, D, C)                      # [BHW, D, C]

        q = self.q(x_seq).view(B * H * W, D, self.heads, self.head_dim).transpose(1, 2)  # [BHW,h,D,d]
        k = self.k(x_seq).view(B * H * W, D, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_seq).view(B * H * W, D, self.heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope.get_cos_sin(D, device=q.device, dtype=q.dtype)
            q = apply_rope_1d(q, cos, sin)
            k = apply_rope_1d(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # [BHW,h,D,D]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                                # [BHW,h,D,d]
        out = out.transpose(1, 2).contiguous().view(B * H * W, D, self.attn_dim)
        out = self.proj(out)                                          # [BHW,D,C]
        out = self.drop(out)
        out = out.view(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()  # [B,C,D,H,W]
        return out


class GlobalSpatialAttention2D(nn.Module):
    """
    Performs global spatial attention within each slice.
    Uses all tokens (H x W) for Queries, and downsampled tokens (H/p x W/p) 
    via average pooling for Keys and Values.

    output: [B, C, D, H, W]
    """
    def __init__(self, c: int, heads: int = 6, attn_dim: int | None = None,
                 kv_down: int = 2, dropout: float = 0.0):
        super().__init__()
        self.c = c
        self.heads = int(heads)
        self.attn_dim = int(attn_dim or c)
        assert self.attn_dim % self.heads == 0
        self.head_dim = self.attn_dim // self.heads
        self.kv_down = int(kv_down)

        self.ln = nn.LayerNorm(c)

        self.q_conv = nn.Conv2d(c, self.attn_dim, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(c, self.attn_dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(c, self.attn_dim, kernel_size=1, bias=False)
        self.proj   = nn.Conv2d(self.attn_dim, c, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)

        self.pool = nn.AvgPool2d(kernel_size=self.kv_down, stride=self.kv_down) if self.kv_down > 1 else nn.Identity()

    def forward(self, x_bcdhw: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x_bcdhw.shape
        x = x_bcdhw.permute(0, 2, 1, 3, 4).contiguous()  # [B,D,C,H,W]
        x_ = x.permute(0, 1, 3, 4, 2).contiguous()       # [B,D,H,W,C] for LN
        x_ = self.ln(x_)
        x = x_.permute(0, 1, 4, 2, 3).contiguous()       # [B,D,C,H,W]

        BD = B * D
        x2d = x.view(BD, C, H, W)

        q = self.q_conv(x2d)  # [BD, attn_dim, H, W]
        k = self.k_conv(self.pool(x2d))  # [BD, attn_dim, H', W']
        v = self.v_conv(self.pool(x2d))  # [BD, attn_dim, H', W']

        Hq, Wq = H, W
        Hk, Wk = k.shape[-2], k.shape[-1]

        # [BD, attn_dim, H, W] -> [BD, H*W, heads, head_dim] -> [BD, heads, Nq, d]
        q = q.view(BD, self.heads, self.head_dim, Hq * Wq).transpose(2, 3).contiguous()  # [BD,h,Nq,d]
        k = k.view(BD, self.heads, self.head_dim, Hk * Wk).transpose(2, 3).contiguous()  # [BD,h,Nk,d]
        v = v.view(BD, self.heads, self.head_dim, Hk * Wk).transpose(2, 3).contiguous()  # [BD,h,Nk,d]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [BD,h,Nq,Nk]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                              # [BD,h,Nq,d]
        out = out.transpose(2, 3).contiguous().view(BD, self.attn_dim, Hq, Wq)

        out = self.proj(out)
        out = self.drop(out)
        out = out.view(B, D, C, Hq, Wq).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,D,H,W]
        return out


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


class FFABlock(nn.Module):
    """
    FFA 1block: [Slice Self-Attn (RoPE)] → [Global Spatial Attn(2D/each slice)] → [FFN]
    """
    def __init__(self, c: int, heads: int = 6, attn_dim: int | None = None,
                 kv_down: int = 2, dropout: float = 0.0, drop_path: float = 0.0,
                 use_rope: bool = True):
        super().__init__()
        self.slice_attn = SliceSelfAttention1D(c, heads=heads, attn_dim=attn_dim, dropout=dropout, use_rope=use_rope)
        self.global_attn = GlobalSpatialAttention2D(c, heads=heads, attn_dim=attn_dim, kv_down=kv_down, dropout=dropout)
        self.ffn = FFN3D_Pointwise(c, hidden=None, dropout=dropout)
        self.dp1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln3 = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W]
        x = x + self.dp1(self.slice_attn(x))
        x = x + self.dp2(self.global_attn(x))
        # FFNはchannels-lastでLN → 3D pointwise
        B, C, D, H, W = x.shape
        y = x.permute(0, 2, 3, 4, 1).contiguous()
        y = self.ln3(y).permute(0, 4, 1, 2, 3).contiguous()
        x = x + self.dp3(self.ffn(y))
        return x


class Aggregator3D_FFA(nn.Module):
    """
    FFA shared core. 
    Input/Output: [B,C,D,H,W] (1/16 res)
    """
    def __init__(self, c: int,
                 n_blocks: int = 2,
                 heads: int = 6,
                 attn_dim: int | None = None,
                 kv_down: int = 2,
                 dropout: float = 0.0,
                 drop_path: float = 0.05,
                 use_rope: bool = True,
                 use_pos_slice: bool = True,
                 max_depth: int = 512):
        super().__init__()
        self.use_pos_slice = bool(use_pos_slice)
        if self.use_pos_slice:
            self.pos_slice = nn.Embedding(max_depth, c)
            self.pos_gain = nn.Parameter(torch.ones(1))
        else:
            self.pos_slice = None

        self.pre = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=False),
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.SiLU(inplace=False),
        )

        blocks = []
        for i in range(int(n_blocks)):
            blocks.append(
                FFABlock(
                    c=c, heads=heads, attn_dim=(attn_dim or c // 2),
                    kv_down=kv_down, dropout=dropout,
                    drop_path=drop_path if n_blocks > 1 else 0.0,
                    use_rope=use_rope
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,D,H,W]
        if self.use_pos_slice and self.pos_slice is not None:
            B, C, D, H, W = x.shape
            idx = torch.arange(D, device=x.device)
            emb = self.pos_slice(idx).transpose(0, 1).view(1, C, D, 1, 1)  # [1,C,D,1,1]
            x = x + self.pos_gain * emb.to(dtype=x.dtype)

        x = self.pre(x)
        for blk in self.blocks:
            x = cp.checkpoint(blk, x, use_reentrant=False) if self.training else blk(x)
        return x


class ParallelAggregatorSharedFFA(nn.Module):
    """
    For 3D features from multiple ViT layers List[B, C, D, Gh, Gw],
    A shared Aggregator3D_FFA is applied across all layers.
    """
    def __init__(self, c: int, num_layers: int,
                 n_blocks: int = 2, heads: int = 6, attn_dim: int | None = None,
                 kv_down: int = 2, dropout: float = 0.0, drop_path: float = 0.05,
                 use_rope: bool = True, use_pos_slice: bool = True):
        super().__init__()
        self.core = Aggregator3D_FFA(
            c=c, n_blocks=n_blocks, heads=heads, attn_dim=attn_dim,
            kv_down=kv_down, dropout=dropout, drop_path=drop_path,
            use_rope=use_rope, use_pos_slice=use_pos_slice
        )
        self.film_in  = LayerFiLM(c, num_layers)
        self.film_out = LayerFiLM(c, num_layers)

    def forward(self, feats_3d_list: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for li, x in enumerate(feats_3d_list):
            xin  = self.film_in(x,  li)
            y    = self.core(xin)
            yout = self.film_out(y, li)
            outs.append(yout)
        return outs


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

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # logits: [B,K,H,W] or [B,K,D,H,W]
        # target: [B,H,W]   or [B,D,H,W]
        K = logits.shape[1]
        probs = logits.float().softmax(dim=1)
        # onehot: [...,K] -> [B,K,*]
        onehot = F.one_hot(torch.clamp(target, min=0), num_classes=K).movedim(-1, 1).float()

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
        probs_sum   = probs.sum(dim=dims)
        onehot_sum  = onehot.sum(dim=dims)

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
    def __init__(self, num_classes: int, class_weights: torch.Tensor,
                 weight_ce: float = 1.0, weight_dice: float = 1.0,
                 use_topk: bool = False, topk_percent: float = 10.0,
                 # topology
                 use_cldice: bool = False, cldice_weight: float = 0.0, cldice_iters: int = 20,
                 use_skelrecall: bool = False, skelrecall_weight: float = 0.0, skelrecall_iters: int = 20):
        super().__init__()
        self.register_buffer('class_weights', class_weights.float())
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

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
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
                logits.float(), target_ce,
                weight=w,
                ignore_index=IGNORE
            )
        # cast fp32
        dice_loss = self.dc(logits.float(), target, mask=mask)

        topo_loss = logits.sum() * 0.0  # default 0
        if self.use_cldice and self.cldice_w > 0.0:
            topo_loss = topo_loss + self.cldice(logits, target, mask=mask) * self.cldice_w
        if self.use_skelrecall and self.skelrecall_w > 0.0:
            topo_loss = topo_loss + self.skelrecall(logits, target, mask=mask) * self.skelrecall_w

        total_loss = self.wce * ce_loss + self.wdc * dice_loss + topo_loss
        if (self.use_cldice and self.cldice_w > 0.0) or (self.use_skelrecall and self.skelrecall_w > 0.0):
            return ce_loss, dice_loss, topo_loss, total_loss
        else:
            return ce_loss, dice_loss, total_loss

# 3D Volume Dataset (full volume; in-plane aug shared across slices)
class VolumeDataset3D(Dataset):
    """
    list_json: [{'volume','seg'}, ...]
    Returns:
      x: [D, 1, H, W]  (all slices; in-plane size is resized to img_size)
      y: [D, H, W]
    Augmentations are 2D in-plane and shared across all D slices (no z-translation).
    """
    def __init__(self, list_json: str, img_key: str='volume', seg_key: str='seg',
                 out_size: Tuple[int,int]=(224,224),
                 crop_depth: int=64, 
                 drop_empty: bool=True, min_fg_frac: float=0.0,
                 augment: bool=False,
                 # geometry
                 hflip_p: float=0.0, vflip_p: float=0.0,
                 rotate_deg: float=15.0,
                 scale_min: float=0.90, scale_max: float=1.10,
                 translate_frac: float=0.10,
                 # intensity
                 noise_p: float=0.2, noise_std: float=0.01,
                 bc_p: float=0.3, brightness: float=0.10, contrast: float=0.10):
        super().__init__()
        self.out_size = out_size
        self.crop_depth = int(crop_depth)
        self.augment = augment

        self.hflip_p, self.vflip_p = float(hflip_p), float(vflip_p)
        self.rotate_deg = float(rotate_deg)
        self.scale_min, self.scale_max = float(scale_min), float(scale_max)
        self.translate_frac = float(translate_frac)
        self.noise_p, self.noise_std = float(noise_p), float(noise_std)
        self.bc_p = float(bc_p); self.brightness = float(brightness); self.contrast = float(contrast)

        with open(list_json, 'r') as f:
            items = json.load(f)
        assert isinstance(items, list) and len(items) > 0

        self.imgs, self.segs = [], []
        for it in items:
            img = nib.load(it[img_key]).get_fdata(dtype=np.float32)
            seg = nib.load(it[seg_key]).get_fdata(dtype=np.float32).astype(np.int64)
            assert img.shape == seg.shape, f"Shape mismatch: {img.shape} vs {seg.shape}"
            if drop_empty:
                fg = (seg > 0).sum()
                if fg == 0 or (fg / seg.size) < min_fg_frac:
                    continue
            self.imgs.append(img); self.segs.append(seg)
        assert len(self.imgs) > 0, "No valid volumes found."

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i: int):
        img = self.imgs[i]  # [H0,W0,D]
        seg = self.segs[i]

        D_full = img.shape[2]

        if self.crop_depth > 0 and D_full > self.crop_depth:
            if self.augment:
                z_start = random.randint(0, D_full - self.crop_depth)
            else:
                z_start = (D_full - self.crop_depth) // 2
            
            z_end = z_start + self.crop_depth
            img = img[:, :, z_start:z_end]
            seg = seg[:, :, z_start:z_end]
        else:
            z_start = 0
            z_end = D_full

        current_depth = img.shape[2]
        z_indices = np.arange(z_start, z_start + current_depth, dtype=np.float32) / max(1.0, float(D_full))
        
        # -> torch, channel-last to channel-first per-slice
        x = torch.from_numpy(img).permute(2,0,1).unsqueeze(1)  # [D,1,H0,W0]
        y = torch.from_numpy(seg).permute(2,0,1)               # [D,H0,W0]
        # Z tensor [D, 1, 1, 1]
        z_tensor = torch.from_numpy(z_indices).view(current_depth, 1, 1, 1)

        # resize in-plane to multiples of 16
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)        # [D,1,H,W]
        y = F.interpolate(y.unsqueeze(1).float(), size=self.out_size, mode='nearest').long()  # [D,1,H,W] -> long
        y = y.squeeze(1)  # [D,H,W]

        H, W = self.out_size
        # [D, 1, H, W]
        z_channel = z_tensor.expand(current_depth, 1, H, W)
        
        # img(Ch0) + Z(Ch1) -> [D, 2, H, W]
        x = torch.cat([x, z_channel], dim=1)

        if self.augment:
          # flips
          if _rbool(self.hflip_p):
            x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
          if _rbool(self.vflip_p):
            x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])

          # shared affine (2D) across all D slices
          do_affine = (self.rotate_deg > 0) or (self.scale_min != 1.0 or self.scale_max != 1.0) or (self.translate_frac > 0)
          if do_affine:
            angle = _runif(-self.rotate_deg, self.rotate_deg)
            scale = _runif(self.scale_min, self.scale_max)
            tx = _runif(-self.translate_frac * W, self.translate_frac * W)
            ty = _runif(-self.translate_frac * H, self.translate_frac * H)
            rad = angle * math.pi / 180.0
            cos, sin = math.cos(rad), math.sin(rad)
            theta = torch.tensor([[
              [scale * cos, -scale * sin, 2.0 * tx / max(1.0, W)],
              [scale * sin,  scale * cos, 2.0 * ty / max(1.0, H)],
            ]], dtype=x.dtype, device=x.device)  # [1,2,3]
            theta_d = theta.repeat(x.shape[0], 1, 1)  # [D,2,3]
            grid = F.affine_grid(theta_d, size=(x.shape[0], 1, H, W), align_corners=False)

            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            y = F.grid_sample(y.unsqueeze(1).float(), grid, mode='nearest', padding_mode='zeros', align_corners=False).long().squeeze(1)

          img_part = x[:, 0:1]
          # intensity (x only)
          if _rbool(self.bc_p):
            alpha = _runif(1.0 - self.contrast, 1.0 + self.contrast)
            beta  = _runif(-self.brightness, self.brightness)
            img_part = img_part * alpha + beta
          if _rbool(self.noise_p):
            img_part = img_part + torch.randn_like(img_part) * self.noise_std
          
          x[:, 0:1] = img_part

        return x, y  # [D,2,H,W], [D,H,W]


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

def _resize_logits_to_target(logits: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    if logits.ndim == 5: # [B, K, D, H, W]
        if logits.shape[-3:] != y.shape[-3:]:
            logits = F.interpolate(logits, size=y.shape[-3:], mode='trilinear', align_corners=False)
    else: # [B, K, H, W]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
    return logits

# Eval (returns CE, Dice, Total averages) 
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_ce = total_dice = total_total = total_topo = 0.0
    count = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, m = batch
            m = m.to(device)
        else:
            x, y = batch
            m = None
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = _resize_logits_to_target(logits, y)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

        out = criterion(logits, y, mask=m)
        if isinstance(out, tuple) and len(out) == 4:
            ce_loss, dice_loss, topo_loss, total_loss = out
        else:
            ce_loss, dice_loss, total_loss = out
            topo_loss = torch.tensor(0.0, device=logits.device)

        bs = x.size(0)
        total_ce   += float(ce_loss.item())   * bs
        total_dice += float(dice_loss.item()) * bs
        total_topo += float(topo_loss.item()) * bs
        total_total+= float(total_loss.item())* bs
        count += bs

    return total_ce / count, total_dice / count, total_topo / count, total_total / count

# Training (with train_list / val_list) 
def train(train_list: str,
          val_list: str,
          num_classes: int,
          epochs: int = 50,
          batch_size: int = 16,
          accumulation_steps: int = 8,
          lr: float = 1e-4,
          min_lr: float = 1e-5,
          drop_empty: bool = True,
          seed: int = 42,
          bg_weight: float = 0.05,
          use_mfb: bool = False,
          min_fg_frac: float = 0.002,
          use_topk: bool = False,
          topk_percent: float = 10.0,
          init_prior_bias: bool = False,
          img_size: Tuple[int, int] = (224, 224),
          use_cldice: bool = False, cldice_weight: float = 0.1, cldice_iters: int = 20,
          use_skelrecall: bool = False, skelrecall_weight: float = 0.1, skelrecall_iters: int = 20,
          aug: bool = False,
          aug_rotate: float = 15.0,
          aug_scale_min: float = 0.90, aug_scale_max: float = 1.10,
          aug_translate: float = 0.10,
          aug_hflip_p: float = 0.0, aug_vflip_p: float = 0.0,
          aug_noise_p: float = 0.2, aug_noise_std: float = 0.01,
          aug_bc_p: float = 0.3, aug_brightness: float = 0.10, aug_contrast: float = 0.10,
          use_25d: bool = False,
          z_stack: int = 5,
          fuse_mode: str = 'conv3d-center',
          use_3d: bool = False,
          depth_min_fg_frac: float = 0.0,
          crop_depth: int = 64,
          save_dir="/path/to/save_dir"):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = getattr(args, 'log_dir',
        '/path/to/log_dir')
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(log_dir, f"train_2d_refine_head_{run_id}.log")
    print(f"[INFO] Logging to {log_file}")
    print(f"[INFO] Saving weights to existing directory: {save_dir}")
    
    # header
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# train_list={train_list}\n# val_list={val_list}\n")
        f.write(f"# num_classes={num_classes} epochs={epochs} batch_size={batch_size} "
                f"lr={lr} min_lr={min_lr} img_size={img_size} use_3d={use_3d} "
                f"depth_min_fg_frac={depth_min_fg_frac} crop_depth={crop_depth}\n")
        f.write(f"# save_dir={save_dir}\n")

    assert not (use_25d and use_3d), "use_25d and use_3d cannot be used at the same time!"

    def _parse_layers(s: str) -> Tuple[int, ...]:
        return tuple(int(t) for t in s.replace(' ', '').split(',') if t != '')


    # Datasets / Loaders
    ds_train = VolumeDataset3D(train_list, out_size=img_size,
                                drop_empty=drop_empty, min_fg_frac=min_fg_frac,
                                augment=aug,
                                hflip_p=aug_hflip_p, vflip_p=aug_vflip_p,
                                rotate_deg=aug_rotate, scale_min=aug_scale_min, scale_max=aug_scale_max,
                                translate_frac=aug_translate,
                                noise_p=aug_noise_p, noise_std=aug_noise_std,
                                bc_p=aug_bc_p, brightness=aug_brightness, contrast=aug_contrast,
                                crop_depth=crop_depth)
    ds_val = VolumeDataset3D(val_list if val_list is not None else train_list,
                                out_size=img_size, drop_empty=drop_empty, min_fg_frac=min_fg_frac,
                                augment=False, crop_depth=crop_depth)

    collate_fn_train = make_collate_pad_3d(depth_min_fg_frac)
    collate_fn_eval = make_collate_pad_3d(depth_min_fg_frac)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_fn_train
    )
    loader_eval = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn_eval
    )

    # Model
    vit_layers = _parse_layers(args.vit_layers)
    model = SegModel3D_UNETRLite(
                num_classes=num_classes,
                vit_layers=vit_layers,
                decoder_base_channels=128,
                decoder_up_factor=args.decoder_up_factor if hasattr(args, 'decoder_up_factor') else 4,
                vit_chunk_slices=args.vit_chunk_slices,
                vit_amp=args.vit_amp,
                use_se=True
            ).to(device)
    trainable_params = model.trainable_parameters() 

    # parameter counting helpers & prints 
    def count_params(module: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    def fmt(n: int) -> str:
        return f"{n:,}"

    # Granular components
    # Adapter: (a) slice adapter (frozen)  (b) 3D ConvNeXt adapter
    adapter_slice_total     = count_params(model.adapter_slice, trainable_only=False)
    adapter_slice_trainable = count_params(model.adapter_slice, trainable_only=True)

    adapter3d_total     = count_params(model.adapter3d, trainable_only=False)
    adapter3d_trainable = count_params(model.adapter3d, trainable_only=True)

    # Small adapter-related projection/gate layers
    proj_gate_modules = nn.ModuleList([
        model.f2_proj_ref,   # 48 -> 64 (3D)
        model.f3_proj_C,     # 64 -> C  (3D)
        model.f3_gate3d,     # gate conv
        model.f2_hr_reduce3d # for HR 2D head
    ])
    proj_gate_total     = count_params(proj_gate_modules, trainable_only=False)
    proj_gate_trainable = count_params(proj_gate_modules, trainable_only=True)

    # Aggregator (wrapper) and its FFA core
    aggregator_total     = count_params(model.shared_agg, trainable_only=False)
    aggregator_trainable = count_params(model.shared_agg, trainable_only=True)

    ffa_core_total     = count_params(model.shared_agg.core, trainable_only=False)
    ffa_core_trainable = count_params(model.shared_agg.core, trainable_only=True)

    # Segmentation heads: 3D refine head + 2D HR head
    seg_heads = nn.ModuleList([model.head, model.hr2d])
    seg_total     = count_params(seg_heads, trainable_only=False)
    seg_trainable = count_params(seg_heads, trainable_only=True)

    # ViT encoder (frozen, for reference)
    vit_total     = count_params(model.encoder, trainable_only=False)
    vit_trainable = count_params(model.encoder, trainable_only=True)

    # Model totals
    model_total_all     = count_params(model, trainable_only=False)
    model_total_trainable = count_params(model, trainable_only=True)

    print("===== Parameter Counts =====")
    print(f"[Adapter (slice/frozen)] total={fmt(adapter_slice_total)} | trainable={fmt(adapter_slice_trainable)}")
    print(f"[Adapter3D (ConvNeXt)]  total={fmt(adapter3d_total)} | trainable={fmt(adapter3d_trainable)}")
    print(f"[Adapter Proj/Gates]    total={fmt(proj_gate_total)} | trainable={fmt(proj_gate_trainable)}")
    print(f"[Aggregator (wrapper)]  total={fmt(aggregator_total)} | trainable={fmt(aggregator_trainable)}")
    print(f"[FFA core]              total={fmt(ffa_core_total)} | trainable={fmt(ffa_core_trainable)}")
    print(f"[Segmentation heads]    total={fmt(seg_total)} | trainable={fmt(seg_trainable)}")
    print(f"[ViT encoder (frozen)]  total={fmt(vit_total)} | trainable={fmt(vit_trainable)}")
    print("--------------------------------------------")
    print(f"[MODEL TOTAL]           total={fmt(model_total_all)} | trainable={fmt(model_total_trainable)}")
    print("============================================")      

    # Class weights & optional prior bias (TRAIN set)
    counts = compute_class_histogram(ds_train, num_classes)
    class_weights = make_class_weights(num_classes, counts, bg_weight=bg_weight, use_mfb=use_mfb)
    if init_prior_bias:
        cls_conv_3d = init_prior_bias_for_head(model.head,  num_classes, counts)
        cls_conv_2d = init_prior_bias_for_head(model.hr2d, num_classes, counts)
        print("[INFO] Initialized head bias (3D refine):", cls_conv_3d)
        print("[INFO] Initialized head bias (2D HR):",     cls_conv_2d)

    criterion = DC_and_CE_loss(
        num_classes=num_classes,
        class_weights=class_weights,
        weight_ce=1.0, weight_dice=1.0,
        use_topk=use_topk, topk_percent=topk_percent,
        use_cldice=use_cldice, cldice_weight=cldice_weight, cldice_iters=cldice_iters,
        use_skelrecall=use_skelrecall, skelrecall_weight=skelrecall_weight, skelrecall_iters=skelrecall_iters
    )

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.05)
    
    def lr_lambda(current_epoch):
        warmup_epochs = 50  
        total_epochs = epochs 
        
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            # Cosine Decay: 1.0 -> 0.0
            progress = float(current_epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # color palette
    palette = make_palette(num_classes)

    start_epoch = 1
    best_loss = float('inf')
    best_dice = float('inf')

    latest_ckpt_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    if os.path.isfile(latest_ckpt_path):
        print(f"[INFO] Found latest checkpoint: {latest_ckpt_path}")
        print("[INFO] Resuming training...")
        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(latest_ckpt_path, map_location=loc)
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_dice = checkpoint.get('best_dice', float('inf'))
        
        print(f"[INFO] Resumed from Epoch {checkpoint['epoch']}. Next Epoch: {start_epoch}")
    else:
        print("[INFO] No latest checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, epochs+1):
        model.train()
        
        tr_ce = tr_dice = tr_topo = tr_total = 0.0
        count = 0

        for i, batch_item in enumerate(loader):
            if use_3d:
                x, y, m = batch_item
                m = m.to(device, non_blocking=True)
            else:
                x, y = batch_item
                m = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                logits = _resize_logits_to_target(logits, y)
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

            # fp32
            out = criterion(logits.float(), y, mask=m)
            if isinstance(out, tuple) and len(out) == 4:
                ce_loss, dice_loss, topo_loss, total_loss = out
            else:
                ce_loss, dice_loss, total_loss = out
                topo_loss = torch.tensor(0.0, device=logits.device)

            loss_normalized = total_loss / accumulation_steps
            loss_normalized.backward()
            
            # Max norm 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)            

            bs = x.size(0)
            tr_ce    += float(ce_loss.item())   * bs
            tr_dice  += float(dice_loss.item()) * bs
            tr_topo  += float(topo_loss.item()) * bs
            tr_total += float(total_loss.item()) * bs
            count += bs

        if len(loader) % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        tr_ce   /= count; tr_dice /= count; tr_topo /= count; tr_total /= count

        ev_ce, ev_dice, ev_topo, ev_total = eval_epoch(model, loader_eval, criterion, device)

        cur_lr = optimizer.param_groups[0]["lr"]
        msg = (f"[Epoch {epoch:03d}] lr={cur_lr:.2e} "
               f"train: CE={tr_ce:.4f}, Dice={tr_dice:.4f}, Topo={tr_topo:.4f}, Total={tr_total:.4f} | "
               f"eval:  CE={ev_ce:.4f}, Dice={ev_dice:.4f}, Topo={ev_topo:.4f}, Total={ev_total:.4f}")
        print(msg)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception as e:
            print(f"[WARN] Failed to write log: {e}")

        # update LR per epoch
        scheduler.step()
        
        if epoch % 20 == 0:
            model.eval()
            vis_batch = next(iter(loader_eval))
            if use_3d:
                x_vis, y_vis, m_vis = vis_batch
                m_vis = m_vis.to(device)
            else:
                x_vis, y_vis = vis_batch
                m_vis = None
            x_vis, y_vis = x_vis.to(device), y_vis.to(device)
            with torch.no_grad():
                logits_vis = model(x_vis)
                logits_vis = _resize_logits_to_target(logits_vis, y_vis)
                preds = logits_vis.argmax(1).detach().cpu().numpy()

            if logits_vis.ndim == 4:
                # 2D/2.5D: [B,K,H,W] -> visualize first sample
                gt = y_vis[0].detach().cpu().numpy().astype(np.int64)
                pr = preds[0].astype(np.int64)
            else:
                # 3D: [B,K,D,H,W] -> central slice of first sample
                D = y_vis.shape[1]
                d = D // 2
                gt = y_vis[0, d].detach().cpu().numpy().astype(np.int64)
                pr = preds[0, d].astype(np.int64)

            gt_vis = colorize(gt, palette); pr_vis = colorize(pr, palette)
            concat = np.concatenate([gt_vis, pr_vis], axis=1)
            out_path = os.path.join("/path/to/vis_outputs/", f"epoch{epoch:03d}_sample0.png")
            imageio.imwrite(out_path, concat.astype(np.uint8))
            print(f"[INFO] Saved color visualization to {out_path}")

        if ev_dice < best_dice:
            best_dice = ev_dice
            best_dice_path = os.path.join(save_dir, 'minibaseline_best_dice.pt')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, 
                        best_dice_path)
            print(f"[INFO] Saved checkpoint: minibaseline_best_dice.pt (loss={best_loss:.4f}), dice={best_dice:.4f}")
        
        if ev_total < best_loss:
            best_loss = ev_total
            best_loss_path = os.path.join(save_dir, 'minibaseline_best_total.pt')
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, 
                        best_loss_path)
            print(f"[INFO] Saved checkpoint: minibaseline_best_total.pt (loss={best_loss:.4f}), dice={best_dice:.4f}")

        # Latest Checkpoint (For Resuming) - always save this
        latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_dice': best_dice,
            'args': vars(args)
        }, latest_path)        

    # Save Final Checkpoint
    final_path = os.path.join(save_dir, 'final_checkpoint.pt')
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'final_loss': ev_total, 
        'final_dice': ev_dice,
        'args': vars(args)
    }, final_path)
    print(f"[INFO] Saved FINAL checkpoint to: {final_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_config_arg(ap)
    # train/val lists
    ap.add_argument('--train_list', type=str, default=None, help='Path to training JSON list')
    ap.add_argument('--val_list',   type=str, default=None, help='Path to validation JSON list (defaults to train_list)')
    # Backward compatibility
    ap.add_argument('--list_json',  type=str, default=None, help='[Deprecated] Single JSON used for both train/val if train_list not provided')

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
    ap.add_argument('--img_size', type=str, default='336', help='Input size e.g., 224 or 512. Must be multiples of 16')

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
    ap.add_argument('--vit_layers', type=str, default='2,5,8,11')

    # 3D Aggregator
    ap.add_argument('--decoder_up_factor', type=int, default=4)

    # Z crop depth
    ap.add_argument('--crop_depth', type=int, default=64, help='Crop size in depth dimension (Z). Use 32 or 48 for high-res.')

    # Logger
    ap.add_argument('--log_dir', type=str, default='/path/to/log_dir', help='dir path to save logs per epoch')

    ap.add_argument('--save_dir', type=str, 
                    default='/path/to/save_dir',
                    help='Directory to save best_checkpoint.pt and final_checkpoint.pt')

    args = parse_args_with_config(ap)

    # Resolve list paths (favor new args; fallback to legacy)
    if args.train_list is None:
        if args.list_json is None:
            raise ValueError("Please provide --train_list (and optionally --val_list)")
        train_list = args.list_json
        val_list = args.list_json
        print("[WARN] Using legacy --list_json for both train and val.")
    else:
        train_list = args.train_list
        val_list = args.val_list if args.val_list is not None else args.train_list

    H, W = parse_img_size(args.img_size)

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
          crop_depth=args.crop_depth)
