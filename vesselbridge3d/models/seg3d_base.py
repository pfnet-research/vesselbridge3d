"""Shared 3D UNETR-Lite segmentation model with a pluggable 2D encoder.

The architecture is identical across DINOv3 / MedSAM / MedGemma comparisons:
a frozen 2D foundation encoder is applied per slice, its multi-layer patch grids
are projected to a common working dimension, stacked into depth, aggregated with
a shared FFA aggregator, and decoded with a UNETR-Lite decoder plus refinement
heads. Only the encoder (and its input normalization) differs between models,
which is what the controlled comparison isolates.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .adapter3d import AdapterPyramid3DConvNeXt
from .attention import ParallelAggregatorSharedFFA
from .decoder import SegDecoder3D_UNETRLite
from .heads import HRHead2D, Refine3DHead


class SegModel3DBase(nn.Module):
    """3D ConvNeXt Adapter + shared FFA aggregator + UNETR-Lite decoder + refine.

    Parameters
    ----------
    encoder:
        Frozen 2D encoder exposing ``hidden_dim`` / ``patch`` attributes and a
        ``forward_multi(x_3ch, layers) -> List[[B, C, Gh, Gw]]`` method.
    input_adapter:
        Module mapping ``[B, 2, H, W]`` (image + z-coord) to the 3-channel,
        encoder-appropriate normalized input.
    work_dim:
        Common channel width fed to the aggregator / decoder. Each encoder layer
        output is projected from ``encoder.hidden_dim`` to ``work_dim`` with a
        shared 1x1 conv (an identity when the dims already match, which keeps the
        DINOv3 baseline numerically unchanged).
    """

    def __init__(
        self,
        num_classes: int,
        encoder: nn.Module,
        input_adapter: nn.Module,
        vit_layers: Tuple[int, ...],
        work_dim: int = 384,
        decoder_base_channels: int = 128,
        decoder_up_factor: int = 4,
        vit_chunk_slices: int = 8,
        vit_amp: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        # Frozen per-slice input adapter (1ch image + z -> 3ch normalized)
        self.adapter_slice = input_adapter
        [p.requires_grad_(False) for p in self.adapter_slice.parameters()]
        # Frozen 2D foundation encoder
        self.encoder = encoder
        [p.requires_grad_(False) for p in self.encoder.parameters()]

        C = int(work_dim)
        self.work_dim = C
        self.patch = int(self.encoder.patch)
        self.selected_layers = tuple(vit_layers)
        # Downstream (3D adapter pyramid + decoder up-factors) assumes encoder
        # features live on a 1/16 grid. Encoders with a different patch size
        # (e.g. SigLIP's 14) are resampled to H/16 so the fusion stays aligned.
        # No-op for patch-16 encoders, keeping the DINOv3 baseline unchanged.
        self.grid_div = 16

        # Project encoder hidden_dim -> common work_dim (shared across layers).
        enc_dim = int(self.encoder.hidden_dim)
        if enc_dim == C:
            self.feat_proj: nn.Module = nn.Identity()
        else:
            self.feat_proj = nn.Conv2d(enc_dim, C, kernel_size=1, bias=False)

        # 3D Pyramidal Adapter (ConvNeXt)
        self.adapter3d = AdapterPyramid3DConvNeXt(c2=48, c3=64, in_ch=2, drop_path=0.05)
        self.f2_proj_ref = nn.Conv3d(48, 64, kernel_size=1, bias=False)  # F2 -> 64
        self.f3_proj_C = nn.Conv3d(64, C, kernel_size=1, bias=False)  # F3 -> C
        self.f3_gate3d = nn.Conv3d(C + C, 1, kernel_size=1)  # 3D gate

        # Shared parallel axial aggregator
        self.shared_agg = ParallelAggregatorSharedFFA(
            c=C,
            num_layers=len(vit_layers),
            n_blocks=2,
            heads=6,
            attn_dim=C // 2,
            kv_down=2,
            dropout=0.0,
            drop_path=0.05,
            use_rope=True,
            use_pos_slice=True,
        )

        # Decoder (UNETR-lite 1/16 -> 1/4)
        self.dec = SegDecoder3D_UNETRLite(
            c_in=C,
            c_skip=C,
            num_classes=num_classes,
            up_factor_hw=decoder_up_factor,
            base_channels=decoder_base_channels,
            use_se=use_se,
        )

        # Refine heads
        self.head = Refine3DHead(
            in_ch=(num_classes + 64),
            num_classes=num_classes,
            mid=decoder_base_channels,
            use_se=True,
        )
        self.hr2d = HRHead2D(
            in_ch=(num_classes + 16),
            num_classes=num_classes,
            mid=decoder_base_channels,
            use_se=True,
        )
        self.f2_hr_reduce3d = nn.Conv3d(64, 16, kernel_size=1, bias=False)  # For HR
        self.hr2d_chunk = 16

        self.vit_chunk_slices = int(vit_chunk_slices)
        self.vit_amp = bool(vit_amp)

    def trainable_parameters(self):
        # Aggregator + Decoder + 3D Adapter + gates + Refine3D + feature projection
        return (
            list(self.shared_agg.parameters())
            + list(self.dec.parameters())
            + list(self.adapter3d.parameters())
            + list(self.f3_proj_C.parameters())
            + list(self.f3_gate3d.parameters())
            + list(self.f2_proj_ref.parameters())
            + list(self.head.parameters())
            + list(self.hr2d.parameters())
            + list(self.feat_proj.parameters())
        )

    def _stack_depth(
        self, feats_2d: List[torch.Tensor], B: int, D: int
    ) -> List[torch.Tensor]:
        outs = []
        for f in feats_2d:
            _, C, Gh, Gw = f.shape
            outs.append(f.view(B, -1, C, Gh, Gw).permute(0, 2, 1, 3, 4).contiguous())
        return outs

    def forward(self, x_bdchw: torch.Tensor) -> torch.Tensor:
        # x: [B,D,2,H,W]
        B, D, C_in, H, W = x_bdchw.shape
        assert C_in == 2, "Input must have 2 channels (Image + Z)"

        # Per-slice encoding (frozen)
        x_slices = x_bdchw.view(-1, 2, H, W)
        feats_per_layer: List[List[torch.Tensor]] = []
        grid_hw = (H // self.grid_div, W // self.grid_div)

        for s in range(0, x_slices.size(0), self.vit_chunk_slices):
            xi = x_slices[s : s + self.vit_chunk_slices]  # [chunk, 2, H, W]
            xi3 = self.adapter_slice(xi)  # [chunk,3,H,W]
            with torch.no_grad():
                ctx = (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if (self.vit_amp and xi3.is_cuda)
                    else nullcontext()
                )
                with ctx:
                    grids = self.encoder.forward_multi(xi3, layers=self.selected_layers)
            for k, gi in enumerate(grids):
                if len(feats_per_layer) <= k:
                    feats_per_layer.append([])
                # Project encoder dim -> work_dim (identity for matching dims).
                gi = self.feat_proj(gi.float())
                # Resample to the canonical 1/16 grid if the patch size differs.
                if gi.shape[-2:] != grid_hw:
                    gi = F.interpolate(
                        gi, size=grid_hw, mode="bilinear", align_corners=False
                    )
                tgt_dtype = (
                    torch.bfloat16 if (self.vit_amp and xi3.is_cuda) else torch.float32
                )
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
        A5, A8 = skips[-2], skips[-1]  # [B,C,D,H/16,W/16]
        A11 = bottom  # [B,C,D,H/16,W/16]

        # 3D Adapter -> Checkpoint
        if self.training:
            F2_3d, F3_3d = cp.checkpoint(self.adapter3d, x_bdchw, use_reentrant=False)
        else:
            F2_3d, F3_3d = self.adapter3d(x_bdchw)

        # Projections
        F2p = self.f2_proj_ref(F2_3d)
        F3pC = self.f3_proj_C(F3_3d)

        # Gate fusion
        A5_up = F.interpolate(
            A5, size=F3pC.shape[-3:], mode="trilinear", align_corners=False
        )
        gate3d = torch.sigmoid(self.f3_gate3d(torch.cat([A5_up, F3pC], dim=1)))
        skip2 = A5_up + gate3d * F3pC

        # Decoder
        if self.training:
            logits_56 = cp.checkpoint(self.dec, A11, [skip2, A8], use_reentrant=False)
        else:
            logits_56 = self.dec(A11, [skip2, A8])

        # Refine Head
        logits_112 = F.interpolate(
            logits_56, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )
        ref_in = torch.cat([logits_112, F2p], dim=1)

        if self.training:
            logits_ref = cp.checkpoint(self.head, ref_in, use_reentrant=False)
        else:
            logits_ref = self.head(ref_in)

        # HR Head
        F2r_3d = self.f2_hr_reduce3d(F2p.detach())

        # Upsample to 1/1
        logits_1x1 = F.interpolate(
            logits_ref, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )
        F2r_1x1 = F.interpolate(
            F2r_3d, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
        )

        # Output buffer
        B, K, D, H, W = logits_1x1.shape
        hr_logits = logits_1x1.new_empty((B, K, D, H, W))

        # Chunked HR processing
        for s in range(0, D, self.hr2d_chunk):
            e = min(D, s + self.hr2d_chunk)
            hr_in_chunk = torch.cat([logits_1x1[:, :, s:e], F2r_1x1[:, :, s:e]], dim=1)
            hr_in_2d = hr_in_chunk.permute(0, 2, 1, 3, 4).reshape(
                B * (e - s), K + 16, H, W
            )

            if self.training:
                out2d = cp.checkpoint(self.hr2d, hr_in_2d, use_reentrant=False)
            else:
                out2d = self.hr2d(hr_in_2d)

            out_chunk = (
                out2d.view(B, (e - s), K, H, W).permute(0, 2, 1, 3, 4).contiguous()
            )
            hr_logits[:, :, s:e] = out_chunk

        return hr_logits
