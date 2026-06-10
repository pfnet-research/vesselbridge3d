# Breaking the Data Barrier: Robust Few-Shot 3D Vessel Segmentation using Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.23782-b31b1b.svg)](https://arxiv.org/abs/2602.23782)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Model Overview

<p align="center">
  <img src="/assets/model-overview.png" alt="Model Overview" width="100%">
</p>

Our proposed architecture efficiently leverages robust 2D foundation models for 3D medical image segmentation, specifically designed to overcome data barriers in tasks like few-shot vessel segmentation. The pipeline consists of four key components:

1. **Frozen 2D Encoder (DINOv3):** Extracts high-quality, slice-wise 2D patch embeddings from the input volume without requiring fine-tuning.
2. **3D Pyramidal Adapter:** A ConvNeXt-style module that smoothly bridges the gap between 2D slices and 3D spatial representations.
3. **Parallel FFA Aggregator:** Captures inter-slice dependencies and global spatial contexts using highly efficient axial and spatial self-attention mechanisms.
4. **UNETR-Lite Decoder & Refinement Heads:** Reconstructs the 3D volume, utilizing a depth-gated 3D Refine Head and a chunked 2D High-Resolution Head for precise, high-fidelity mask generation.

## How to Run

### 1. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install dependencies from uv.lock
uv sync
```

> **Note**
> - Python 3.12 is used (managed automatically by uv via `.python-version`).
> - PyTorch is installed with CUDA 12.1 builds (`torch==2.5.1+cu121`). The CUDA index is configured in `pyproject.toml`.

Prefix the commands below with `uv run` (e.g. `uv run python train.py ...`), or activate the environment first with `source .venv/bin/activate`.

### 2. Prepare Data (JSON List Format)

Training and validation data are specified via JSON files.  
Each entry must contain `"volume"` (image) and `"seg"` (label) paths.  
```json
[
  {
    "volume": "/path/to/images/case_001_0000.nii.gz",
    "seg":    "/path/to/labels/case_001.nii.gz"
  },
  {
    "volume": "/path/to/images/case_002_0000.nii.gz",
    "seg":    "/path/to/labels/case_002.nii.gz"
  }
]
```

> **Note**  
> - Supported format: NIfTI (`.nii`, `.nii.gz`)  
> - `"volume"` and `"seg"` must share the same spatial shape  

### 3. Training

#### 3D Model 
```bash
python train.py \
  --train_list /path/to/train.json \
  --val_list /path/to/val.json \
  --log_dir /path/to/log_dir \
  --save_dir /path/to/save_dir \
  --num_classes 16 \
  --epochs 1000 \
  --batch_size 2 \
  --accumulation_steps 1 \
  --img_size 336 \
  --lr 5e-3 \
  --drop_empty \
  --min_fg_frac 0.0 \
  --bg_weight 0.05 \
  --use_mfb \
  --init_prior_bias \
  --aug \
  --use_3d \
  --use_3d_unetr \
  --depth_min_fg_frac 0.0005
```
### 4. Inference

Predictions are produced via a sliding window over the Z axis (chunk size = 64 slices).  
Each chunk is processed independently, and results are stitched back to the original volume shape.  
Output segmentation masks are saved as NIfTI files in `--out_dir`, preserving the original affine and header.

```bash
python inference.py \
  --test_list  /path/to/test.json \
  --checkpoint /path/to/checkpoint.pt \
  --out_dir    /path/to/output \
  --num_classes 16 \
  --img_size 336
```

> **Note**  
> - `--num_classes`, `--img_size`, `--vit_layers`, and `--decoder_up_factor` must match the values used during training  
> - Output masks are saved as `<case_id>.nii.gz` in `--out_dir`, with the same affine and header as the input volume

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_list` | *(required)* | Path to the training JSON list |
| `--val_list` | same as `train_list` | Path to the validation JSON list |
| `--num_classes` | *(required)* | Number of classes including background |
| `--epochs` | `1000` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--accumulation_steps` | `1` | Gradient accumulation steps (effective batch = `batch_size × accumulation_steps`) |
| `--img_size` | `336` | In-plane spatial resolution (must be a multiple of 16) |
| `--lr` | `5e-3` | Peak learning rate (Warmup + Cosine Decay) |
| `--crop_depth` | `64` | Number of slices to crop along the Z axis |
| `--depth_min_fg_frac` | `0.0005` | Per-slice foreground ratio threshold; slices below this are excluded from the loss |
| `--drop_empty` | `False` | Exclude volumes with no foreground voxels |
| `--use_mfb` | `False` | Enable Median Frequency Balancing for class weights |
| `--init_prior_bias` | `False` | Initialize head bias from class frequency priors |
| `--aug` | `False` | Enable data augmentation (rotation, scaling, noise, etc.) |
| `--use_3d` | `False` | Enable 3D mode (per-slice ViT + 3D aggregator + 3D decoder) |
| `--vit_chunk_slices` | `8` | Chunk size for ViT inference along the slice dimension (reduces VRAM usage) |
| `--vit_amp` | `False` | Run ViT inference in bfloat16 AMP |
| `--log_dir` | *(/path/to/log_dir)* | Directory to save training logs |

### Citation

If you find this repository or our paper useful in your research, please consider citing:

```bibtex
@misc{yoshihara2026fewshot3dsegmentation,
      title={Breaking the Data Barrier: Robust Few-Shot 3D Vessel Segmentation using Foundation Models}, 
      author={Kirato Yoshihara and Yohei Sugawara and Yuta Tokuoka and Lihang Hong},
      year={2026},
      eprint={2602.23782},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2602.23782}, 
}
