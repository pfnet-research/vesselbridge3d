"""Frozen 2D encoders + input adapter.

This is the natural extension point for new architectures: a new model
(e.g. MedSAM, MedGemma) can add its own frozen encoder here exposing the same
``hidden_dim`` / ``patch`` attributes and a ``forward_multi(x, layers)`` method
returning per-layer patch grids ``[B, C, Gh, Gw]``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class FrozenAdapter(nn.Module):
    """
    [B, 2, H, W] -> [B, 3, H, W]
    Channel 0: Image (min-max norm)
    Channel 1: Z-coord (from dataset)
    Output: R=Image, G=Image, B=Z-coord -> ImageNet Normalize
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        assert C == 2, f"Expected 2-channel input (Image + Z), got {C}"

        img = x[:, 0:1, :, :]  # [B, 1, H, W]
        z_map = x[:, 1:2, :, :]  # [B, 1, H, W]

        x_min = img.amin(dim=(2, 3), keepdim=True)
        x_max = img.amax(dim=(2, 3), keepdim=True)
        img01 = (img - x_min) / (x_max - x_min + 1e-6)

        # (R=Img, G=Img, B=Z)
        x3 = torch.cat([img01, img01, z_map], dim=1)  # [B, 3, H, W]

        x3 = (x3 - self.mean) / self.std
        return x3


class FrozenDINOv3ViTS16(nn.Module):
    def __init__(
        self,
        hf_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        default_layers: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.vit = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        self.hidden_dim = int(getattr(self.vit.config, "hidden_size", 384))
        self.patch = int(getattr(self.vit.config, "patch_size", 16))
        self.num_special_tokens = 5  # CLS(1) + Register(4)
        self.default_layers = tuple(int(i) for i in default_layers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Gh, Gw = H // self.patch, W // self.patch
        outputs = self.vit(pixel_values=x, return_dict=True)
        tokens = outputs.last_hidden_state  # [B, seq, C]
        patch_tokens = tokens[
            :, self.num_special_tokens : self.num_special_tokens + Gh * Gw, :
        ]
        return (
            patch_tokens.transpose(1, 2).contiguous().view(B, self.hidden_dim, Gh, Gw)
        )

    @torch.no_grad()
    def forward_multi(
        self, x: torch.Tensor, layers: Tuple[int, ...] | None = None
    ) -> List[torch.Tensor]:
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
            idx = L_total if li == -1 else int(li)
            # hidden_states は [0]=embeddings, [1..L_total]=each block
            idx = max(1, min(idx, L_total))
            tok = hiddens[idx]  # [B, seq, C]
            patch_tokens = tok[
                :, self.num_special_tokens : self.num_special_tokens + N, :
            ]
            grid = (
                patch_tokens.transpose(1, 2)
                .contiguous()
                .view(B, self.hidden_dim, Gh, Gw)
            )
            grids.append(grid)
        return grids
