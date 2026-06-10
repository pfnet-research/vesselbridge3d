"""MedGemma-based 3D UNETR-Lite comparison model and its registry builder.

Drop-in encoder swap: the frozen DINOv3 encoder is replaced by MedGemma's frozen
SigLIP vision tower; everything downstream is shared with the DINOv3 baseline.
The SigLIP hidden dim (1152) is projected to the common working dim (384).

SigLIP uses [-1, 1] normalization (mean=std=0.5) and a patch size of 14, so the
input size must be a multiple of 14 (336 and 448 both qualify). The encoder is
gated on the Hugging Face Hub: accept the license and provide ``HF_TOKEN``.
"""

from __future__ import annotations

from typing import Tuple

from . import register_model
from .encoders import (
    SIGLIP_MEAN,
    SIGLIP_STD,
    FrozenAdapter,
    FrozenMedGemmaVisionEncoder,
)
from .seg3d_base import SegModel3DBase


@register_model("medgemma_unetr")
def build_medgemma_unetr(
    *,
    num_classes: int,
    vit_layers: Tuple[int, ...] = (6, 13, 20, 26),
    decoder_base_channels: int = 128,
    decoder_up_factor: int = 4,
    vit_chunk_slices: int = 8,
    vit_amp: bool = True,
    use_se: bool = True,
    work_dim: int = 384,
    hf_id: str = "google/medgemma-4b-it",
) -> SegModel3DBase:
    encoder = FrozenMedGemmaVisionEncoder(hf_id=hf_id, default_layers=tuple(vit_layers))
    return SegModel3DBase(
        num_classes=num_classes,
        encoder=encoder,
        input_adapter=FrozenAdapter(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        vit_layers=vit_layers,
        work_dim=work_dim,
        decoder_base_channels=decoder_base_channels,
        decoder_up_factor=decoder_up_factor,
        vit_chunk_slices=vit_chunk_slices,
        vit_amp=vit_amp,
        use_se=use_se,
    )
