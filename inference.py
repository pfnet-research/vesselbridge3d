"""
uv run python inference.py \
  --test_list /path/to/test.json \
  --checkpoint /path/to/checkpoint.pt \
  --out_dir /path/to/output \
  --num_classes 16 \
  --img_size 336 
"""

import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm

from train import SegModel3D_UNETRLite, parse_img_size


def preprocess_with_z(data_numpy, target_h, target_w):
    """
    Input: (H_orig, W_orig, D_orig)
    Output: Tensor (1, D_orig, 2, H_target, W_target) -> [B, D, C, H, W]
    """
    H_orig, W_orig, D_orig = data_numpy.shape
    
    z_indices = np.arange(0, D_orig, dtype=np.float32) / max(1.0, float(D_orig))
    # [D, 1, 1, 1]
    z_tensor = torch.from_numpy(z_indices).view(D_orig, 1, 1, 1) 
    
    # (H, W, D) -> (D, 1, H, W)
    img_tensor = torch.from_numpy(data_numpy).float().permute(2, 0, 1).unsqueeze(1)
    
    # [D, 1, Ht, Wt]
    img_resized = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # [D, 1, Ht, Wt]
    z_channel = z_tensor.expand(D_orig, 1, target_h, target_w)
    
    # [D, 2, Ht, Wt] (Channel 0: Image, Channel 1: Z)
    x = torch.cat([img_resized, z_channel], dim=1)
    
    # [1, D, 2, Ht, Wt]
    x = x.unsqueeze(0)
    
    return x

def save_nifti(data, affine, header, out_path):
    out_img = nib.Nifti1Image(data.astype(np.uint8), affine, header)
    nib.save(out_img, out_path)

def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    H_target, W_target = parse_img_size(args.img_size)
    print(f"[INFO] XY Resize Target: {H_target}x{W_target}")
    print(f"[INFO] Target ViT Layers: {args.vit_layers}")
    
    vit_layers = tuple(int(t) for t in args.vit_layers.replace(' ', '').split(',') if t != '')
    
    model = SegModel3D_UNETRLite(
        num_classes=args.num_classes,
        vit_layers=vit_layers,  
        decoder_base_channels=128,
        decoder_up_factor=args.decoder_up_factor,
        vit_chunk_slices=args.vit_chunk_slices,
        vit_amp=args.vit_amp,
        use_se=True
    ).to(device)

    print(f"[INFO] Loading weights: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed. Retrying with strict=False. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.eval()

    with open(args.test_list, 'r') as f:
        test_data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    
    CROP_D = 64 

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
            for z_start in range(0, D_orig, CROP_D):
                z_end = min(z_start + CROP_D, D_orig)
                current_depth = z_end - z_start
                
                # chunk: [1, current_depth, 2, H, W]
                chunk = input_tensor[:, z_start:z_end, :, :, :]
                
                # Padding 
                if current_depth < CROP_D:
                    pad_d = CROP_D - current_depth
                    chunk = F.pad(chunk, (0,0, 0,0, 0,0, 0,pad_d))
                
                with torch.amp.autocast('cuda', enabled=args.vit_amp):
                    logits = model(chunk) 
                
                if current_depth < CROP_D:
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

    print("[INFO] Inference completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--img_size', type=str, default='336')
    
    parser.add_argument('--vit_layers', type=str, default='2,5,8,11', help='Layers to use, e.g. "11,11,11" or "2,5,8,11"')
    
    parser.add_argument('--decoder_up_factor', type=int, default=4)
    parser.add_argument('--vit_chunk_slices', type=int, default=8)
    parser.add_argument('--vit_amp', action='store_true')

    args = parser.parse_args()
    run_inference(args)