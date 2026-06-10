"""Inference engine: sliding-window 3D prediction and NIfTI output."""

from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..constants import CROP_D, DEFAULT_DECODER_BASE_CHANNELS
from ..data import parse_img_size, parse_layers, preprocess_with_z
from ..models import build_model
from ..utils import get_logger
from .checkpoint import load_model_weights

logger = get_logger()


def save_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(out_img, out_path)


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    H_target, W_target = parse_img_size(args.img_size)
    logger.info(f"[INFO] XY Resize Target: {H_target}x{W_target}")
    logger.info(f"[INFO] Target ViT Layers: {args.vit_layers}")

    vit_layers = parse_layers(args.vit_layers)

    model = build_model(
        getattr(args, 'model_type', 'dinov3_unetr'),
        num_classes=args.num_classes,
        vit_layers=vit_layers,
        decoder_base_channels=DEFAULT_DECODER_BASE_CHANNELS,
        decoder_up_factor=args.decoder_up_factor,
        vit_chunk_slices=args.vit_chunk_slices,
        vit_amp=args.vit_amp,
        use_se=True
    ).to(device)

    load_model_weights(model, args.checkpoint, device)
    model.eval()

    with open(args.test_list, 'r') as f:
        test_data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    crop_d = int(getattr(args, 'crop_depth', CROP_D))

    with torch.no_grad():
        for case in tqdm(test_data):
            vol_path = case.get('volume') or case.get('image')
            case_id = os.path.basename(vol_path).replace(".nii.gz", "").replace(".nii", "")

            img = nib.load(vol_path)
            vol_data = img.get_fdata(dtype=np.float32)
            affine = img.affine
            header = img.header

            H_orig, W_orig, D_orig = vol_data.shape

            # input_tensor: [1, D_orig, 2, 336, 336]
            input_tensor = preprocess_with_z(vol_data, H_target, W_target).to(device)

            torch.cuda.empty_cache()

            final_pred_mask = np.zeros((D_orig, H_orig, W_orig), dtype=np.uint8)

            # Manual Sliding Window
            for z_start in range(0, D_orig, crop_d):
                z_end = min(z_start + crop_d, D_orig)
                current_depth = z_end - z_start

                # chunk: [1, current_depth, 2, H, W]
                chunk = input_tensor[:, z_start:z_end, :, :, :]

                # Padding
                if current_depth < crop_d:
                    pad_d = crop_d - current_depth
                    chunk = F.pad(chunk, (0,0, 0,0, 0,0, 0,pad_d))

                with torch.amp.autocast('cuda', enabled=args.vit_amp):
                    logits = model(chunk)

                if current_depth < crop_d:
                    logits = logits[:, :, :current_depth, :, :]

                logits_cpu = logits.cpu().float() # [1, K, current_depth, 336, 336]

                for i in range(current_depth):
                    slice_logits = logits_cpu[:, :, i, :, :] # (1, K, 336, 336)

                    # Argmax
                    slice_pred_336 = torch.argmax(slice_logits, dim=1).unsqueeze(1).float()

                    # Resize to original HW (Nearest Neighbor)
                    slice_resized = F.interpolate(
                        slice_pred_336,
                        size=(H_orig, W_orig),
                        mode='nearest'
                    )

                    final_pred_mask[z_start + i, :, :] = slice_resized.squeeze().numpy().astype(np.uint8)

            pred_save = np.transpose(final_pred_mask, (1, 2, 0))
            out_name = f"{case_id}.nii.gz"
            out_path = os.path.join(args.out_dir, out_name)
            save_nifti(pred_save, affine, header, out_path)

    logger.info("[INFO] Inference completed!")
