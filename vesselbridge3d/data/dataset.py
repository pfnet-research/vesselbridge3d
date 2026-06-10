"""Full-volume 3D dataset with in-plane augmentation shared across slices."""

from __future__ import annotations

import json
import math
import random
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..common.utils import _rbool, _runif


# 3D Volume Dataset (full volume; in-plane aug shared across slices)
class VolumeDataset3D(Dataset):
    """
    list_json: [{'volume','seg'}, ...]
    Returns:
      x: [D, 1, H, W]  (all slices; in-plane size is resized to img_size)
      y: [D, H, W]
    Augmentations are 2D in-plane and shared across all D slices (no z-translation).
    """

    def __init__(
        self,
        list_json: str,
        img_key: str = "volume",
        seg_key: str = "seg",
        out_size: Tuple[int, int] = (224, 224),
        crop_depth: int = 64,
        drop_empty: bool = True,
        min_fg_frac: float = 0.0,
        augment: bool = False,
        # geometry
        hflip_p: float = 0.0,
        vflip_p: float = 0.0,
        rotate_deg: float = 15.0,
        scale_min: float = 0.90,
        scale_max: float = 1.10,
        translate_frac: float = 0.10,
        # intensity
        noise_p: float = 0.2,
        noise_std: float = 0.01,
        bc_p: float = 0.3,
        brightness: float = 0.10,
        contrast: float = 0.10,
    ):
        super().__init__()
        self.out_size = out_size
        self.crop_depth = int(crop_depth)
        self.augment = augment

        self.hflip_p, self.vflip_p = float(hflip_p), float(vflip_p)
        self.rotate_deg = float(rotate_deg)
        self.scale_min, self.scale_max = float(scale_min), float(scale_max)
        self.translate_frac = float(translate_frac)
        self.noise_p, self.noise_std = float(noise_p), float(noise_std)
        self.bc_p = float(bc_p)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

        with open(list_json, "r") as f:
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
            self.imgs.append(img)
            self.segs.append(seg)
        assert len(self.imgs) > 0, "No valid volumes found."

    def __len__(self):
        return len(self.imgs)

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
        z_indices = np.arange(z_start, z_start + current_depth, dtype=np.float32) / max(
            1.0, float(D_full)
        )

        # -> torch, channel-last to channel-first per-slice
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(1)  # [D,1,H0,W0]
        y = torch.from_numpy(seg).permute(2, 0, 1)  # [D,H0,W0]
        # Z tensor [D, 1, 1, 1]
        z_tensor = torch.from_numpy(z_indices).view(current_depth, 1, 1, 1)

        # resize in-plane to multiples of 16
        x = F.interpolate(
            x, size=self.out_size, mode="bilinear", align_corners=False
        )  # [D,1,H,W]
        y = F.interpolate(
            y.unsqueeze(1).float(), size=self.out_size, mode="nearest"
        ).long()  # [D,1,H,W] -> long
        y = y.squeeze(1)  # [D,H,W]

        H, W = self.out_size
        # [D, 1, H, W]
        z_channel = z_tensor.expand(current_depth, 1, H, W)

        # img(Ch0) + Z(Ch1) -> [D, 2, H, W]
        x = torch.cat([x, z_channel], dim=1)

        if self.augment:
            # flips
            if _rbool(self.hflip_p):
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if _rbool(self.vflip_p):
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])

            # shared affine (2D) across all D slices
            do_affine = (
                (self.rotate_deg > 0)
                or (self.scale_min != 1.0 or self.scale_max != 1.0)
                or (self.translate_frac > 0)
            )
            if do_affine:
                angle = _runif(-self.rotate_deg, self.rotate_deg)
                scale = _runif(self.scale_min, self.scale_max)
                tx = _runif(-self.translate_frac * W, self.translate_frac * W)
                ty = _runif(-self.translate_frac * H, self.translate_frac * H)
                rad = angle * math.pi / 180.0
                cos, sin = math.cos(rad), math.sin(rad)
                theta = torch.tensor(
                    [
                        [
                            [scale * cos, -scale * sin, 2.0 * tx / max(1.0, W)],
                            [scale * sin, scale * cos, 2.0 * ty / max(1.0, H)],
                        ]
                    ],
                    dtype=x.dtype,
                    device=x.device,
                )  # [1,2,3]
                theta_d = theta.repeat(x.shape[0], 1, 1)  # [D,2,3]
                grid = F.affine_grid(
                    theta_d, size=(x.shape[0], 1, H, W), align_corners=False
                )

                x = F.grid_sample(
                    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                y = (
                    F.grid_sample(
                        y.unsqueeze(1).float(),
                        grid,
                        mode="nearest",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    .long()
                    .squeeze(1)
                )

            img_part = x[:, 0:1]
            # intensity (x only)
            if _rbool(self.bc_p):
                alpha = _runif(1.0 - self.contrast, 1.0 + self.contrast)
                beta = _runif(-self.brightness, self.brightness)
                img_part = img_part * alpha + beta
            if _rbool(self.noise_p):
                img_part = img_part + torch.randn_like(img_part) * self.noise_std

            x[:, 0:1] = img_part

        return x, y  # [D,2,H,W], [D,H,W]
