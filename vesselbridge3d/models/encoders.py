"""Frozen 2D encoders + input adapter.

This is the natural extension point for new architectures: a new model
(e.g. MedSAM, MedGemma) can add its own frozen encoder here exposing the same
``hidden_dim`` / ``patch`` attributes and a ``forward_multi(x, layers)`` method
returning per-layer patch grids ``[B, C, Gh, Gw]``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

# SigLIP / MedSigLIP expect inputs mapped from [0, 1] to [-1, 1].
SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)


class FrozenAdapter(nn.Module):
    """
    [B, 2, H, W] -> [B, 3, H, W]
    Channel 0: Image (min-max norm)
    Channel 1: Z-coord (from dataset)
    Output: R=Image, G=Image, B=Z-coord -> normalize with ``mean`` / ``std``.

    The default ImageNet statistics match how DINOv3 is fed. SAM's pixel
    statistics scaled to the [0, 1] range coincide with ImageNet, so the default
    is also correct for MedSAM. SigLIP-based encoders pass ``SIGLIP_MEAN`` /
    ``SIGLIP_STD`` to map the image into [-1, 1].
    """

    def __init__(
        self,
        mean: Sequence[float] | torch.Tensor | None = None,
        std: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        mean_t = (
            IMAGENET_MEAN
            if mean is None
            else torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        std_t = (
            IMAGENET_STD
            if std is None
            else torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

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


class FrozenMedSAMImageEncoder(nn.Module):
    """Frozen MedSAM (SAM ViT-B fine-tuned on medical images) image encoder.

    Exposes the same ``hidden_dim`` / ``patch`` / ``forward_multi`` contract as
    :class:`FrozenDINOv3ViTS16`. SAM keeps features in spatial ``[B, H', W', C]``
    layout; we transpose to ``[B, C, Gh, Gw]`` token grids.

    SAM was pretrained at 1024x1024 with a 64x64 absolute position embedding. For
    other input sizes we bilinearly interpolate that embedding to the current
    grid (decomposed relative-position tables are interpolated inside the HF
    attention itself). This is a deliberate caveat when running at 336/448.
    """

    def __init__(
        self,
        hf_id: str = "wanglab/medsam-vit-base",
        default_layers: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        from transformers import SamModel

        sam = SamModel.from_pretrained(hf_id)
        self.vision = sam.vision_encoder
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()
        cfg = self.vision.config
        self.hidden_dim = int(cfg.hidden_size)
        self.patch = int(cfg.patch_size)
        self.num_special_tokens = 0
        self.default_layers = tuple(int(i) for i in default_layers)

    def _pos_embed(self, Gh: int, Gw: int) -> torch.Tensor | None:
        pos = self.vision.pos_embed
        if pos is None:
            return None
        # pos: [1, P, P, C] -> interpolate to [1, Gh, Gw, C]
        if pos.shape[1] == Gh and pos.shape[2] == Gw:
            return pos
        p = pos.permute(0, 3, 1, 2)  # [1, C, P, P]
        p = F.interpolate(p, size=(Gh, Gw), mode="bicubic", align_corners=False)
        return p.permute(0, 2, 3, 1).contiguous()  # [1, Gh, Gw, C]

    @torch.no_grad()
    def forward_multi(
        self, x: torch.Tensor, layers: Tuple[int, ...] | None = None
    ) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0
        Gh, Gw = H // self.patch, W // self.patch
        layers = tuple(self.default_layers if layers is None else layers)

        # Call the patch projection directly: SamPatchEmbeddings.forward hard-asserts
        # the input matches the pretrain size (1024), but the conv itself is
        # size-agnostic. Output layout matches: [B, Gh, Gw, C].
        hidden = self.vision.patch_embed.projection(x).permute(0, 2, 3, 1)
        pos = self._pos_embed(Gh, Gw)
        if pos is not None:
            hidden = hidden + pos.to(dtype=hidden.dtype)

        n_layers = len(self.vision.layers)
        grids: List[torch.Tensor] = []
        # Map requested indices to per-block outputs (output after block i).
        wanted = {(n_layers - 1 if li == -1 else int(li)) for li in layers}
        wanted = {max(0, min(i, n_layers - 1)) for i in wanted}
        captured = {}
        for i, layer_module in enumerate(self.vision.layers):
            hidden = layer_module(hidden)
            if i in wanted:
                captured[i] = hidden
        for li in layers:
            idx = n_layers - 1 if li == -1 else int(li)
            idx = max(0, min(idx, n_layers - 1))
            grid = captured[idx].permute(0, 3, 1, 2).contiguous()  # [B, C, Gh, Gw]
            grids.append(grid)
        return grids


class FrozenMedGemmaVisionEncoder(nn.Module):
    """Frozen MedGemma vision tower (SigLIP / MedSigLIP) encoder.

    Same contract as :class:`FrozenDINOv3ViTS16`. SigLIP has no CLS / register
    tokens, so all sequence positions are patch tokens. Absolute position
    embeddings are interpolated via ``interpolate_pos_encoding=True`` so the
    encoder accepts arbitrary (patch-multiple) input sizes.
    """

    def __init__(
        self,
        hf_id: str = "google/medgemma-4b-it",
        default_layers: Tuple[int, ...] = (6, 13, 20, 26),
    ):
        super().__init__()
        self.vit = self._load_vision_tower(hf_id)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        cfg = self.vit.config
        self.hidden_dim = int(cfg.hidden_size)
        self.patch = int(cfg.patch_size)
        self.num_special_tokens = 0
        self.default_layers = tuple(int(i) for i in default_layers)

    @staticmethod
    def _load_vision_tower(hf_id: str) -> nn.Module:
        from transformers import SiglipVisionModel

        # Standalone SigLIP checkpoints (e.g. google/medsiglip-448) load directly.
        try:
            return SiglipVisionModel.from_pretrained(hf_id)
        except Exception:
            pass
        # Otherwise pull the vision tower out of the full MedGemma VLM.
        from transformers import AutoModelForImageTextToText

        full = AutoModelForImageTextToText.from_pretrained(hf_id)
        tower = getattr(full, "vision_tower", None)
        if tower is None:
            model = getattr(full, "model", full)
            tower = getattr(model, "vision_tower", None)
        if tower is None:
            raise RuntimeError(f"Could not locate a vision tower in {hf_id!r}")
        # Unwrap SiglipVisionModel -> keep the module that exposes a config.
        return tower

    @torch.no_grad()
    def forward_multi(
        self, x: torch.Tensor, layers: Tuple[int, ...] | None = None
    ) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, (
            f"Input {H}x{W} must be a multiple of patch size {self.patch}"
        )
        Gh, Gw = H // self.patch, W // self.patch
        N = Gh * Gw
        layers = tuple(self.default_layers if layers is None else layers)

        outputs = self.vit(
            pixel_values=x,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        hiddens = outputs.hidden_states  # tuple: [embeddings, block_1, ..., block_L]
        L_total = len(hiddens) - 1

        grids: List[torch.Tensor] = []
        for li in layers:
            idx = L_total if li == -1 else int(li)
            idx = max(1, min(idx, L_total))
            tok = hiddens[idx]  # [B, N, C]
            patch_tokens = tok[:, :N, :]
            grid = (
                patch_tokens.transpose(1, 2)
                .contiguous()
                .view(B, self.hidden_dim, Gh, Gw)
            )
            grids.append(grid)
        return grids
