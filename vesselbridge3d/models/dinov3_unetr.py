"""DINOv3-based 3D UNETR-Lite segmentation model and its registry builder."""

from __future__ import annotations

from typing import Tuple

from . import register_model
from .encoders import FrozenAdapter, FrozenDINOv3ViTS16
from .seg3d_base import SegModel3DBase


class SegModel3D_UNETRLite(SegModel3DBase):
    """DINOv3 ViT-S/16 encoder + 3D ConvNeXt adapter + UNETR-Lite decoder.

    Thin wrapper over :class:`SegModel3DBase` that wires in the frozen DINOv3
    encoder and ImageNet input normalization. ``work_dim`` matches the DINOv3
    hidden size (384), so the feature projection is an identity and behaviour is
    unchanged from the original single-model implementation.
    """

    def __init__(
        self,
        num_classes: int,
        vit_layers: Tuple[int, ...] = (2, 5, 8, 11),
        decoder_base_channels: int = 128,
        decoder_up_factor: int = 4,
        vit_chunk_slices: int = 8,
        vit_amp: bool = True,
        use_se: bool = True,
    ):
        encoder = FrozenDINOv3ViTS16(default_layers=vit_layers)
        super().__init__(
            num_classes=num_classes,
            encoder=encoder,
            input_adapter=FrozenAdapter(),
            vit_layers=vit_layers,
            work_dim=int(encoder.hidden_dim),
            decoder_base_channels=decoder_base_channels,
            decoder_up_factor=decoder_up_factor,
            vit_chunk_slices=vit_chunk_slices,
            vit_amp=vit_amp,
            use_se=use_se,
        )


@register_model("dinov3_unetr")
def build_dinov3_unetr(
    *,
    num_classes: int,
    vit_layers: Tuple[int, ...] = (2, 5, 8, 11),
    decoder_base_channels: int = 128,
    decoder_up_factor: int = 4,
    vit_chunk_slices: int = 8,
    vit_amp: bool = True,
    use_se: bool = True,
) -> SegModel3D_UNETRLite:
    return SegModel3D_UNETRLite(
        num_classes=num_classes,
        vit_layers=vit_layers,
        decoder_base_channels=decoder_base_channels,
        decoder_up_factor=decoder_up_factor,
        vit_chunk_slices=vit_chunk_slices,
        vit_amp=vit_amp,
        use_se=use_se,
    )
