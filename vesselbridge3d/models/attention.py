"""Attention / aggregation stack: RoPE, FFA blocks, shared aggregator, FiLM."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .blocks import DropPath, FFN3D_Pointwise


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
