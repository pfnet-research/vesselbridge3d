"""MedSAM-based 3D UNETR-Lite comparison model and its registry builder.

Drop-in encoder swap: the frozen DINOv3 encoder is replaced by MedSAM's frozen
SAM ViT-B image encoder; everything downstream (3D adapter, aggregator, decoder,
heads) is shared with the DINOv3 baseline. MedSAM's hidden dim (768) is projected
to the common working dim (384) so trainable capacity matches the baseline.

MedSAM's pixel statistics scaled to [0, 1] coincide with ImageNet, so the default
``FrozenAdapter`` normalization is correct here.
"""

from __future__ import annotations

from typing import Tuple

from . import register_model
from .encoders import FrozenAdapter, FrozenMedSAMImageEncoder
from .seg3d_base import SegModel3DBase


@register_model("medsam_unetr")
def build_medsam_unetr(
    *,
    num_classes: int,
    vit_layers: Tuple[int, ...] = (2, 5, 8, 11),
    decoder_base_channels: int = 128,
    decoder_up_factor: int = 4,
    vit_chunk_slices: int = 8,
    vit_amp: bool = True,
    use_se: bool = True,
    work_dim: int = 384,
    hf_id: str = "wanglab/medsam-vit-base",
) -> SegModel3DBase:
    encoder = FrozenMedSAMImageEncoder(hf_id=hf_id, default_layers=tuple(vit_layers))
    return SegModel3DBase(
        num_classes=num_classes,
        encoder=encoder,
        input_adapter=FrozenAdapter(),  # ImageNet == SAM pixel stats on [0, 1]
        vit_layers=vit_layers,
        work_dim=work_dim,
        decoder_base_channels=decoder_base_channels,
        decoder_up_factor=decoder_up_factor,
        vit_chunk_slices=vit_chunk_slices,
        vit_amp=vit_amp,
        use_se=use_se,
    )
