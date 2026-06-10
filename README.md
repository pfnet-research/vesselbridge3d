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

After `uv sync` (or `uv pip install -e .`), two console scripts are installed and used as the primary commands throughout this README:

```bash
vb3d-train ...   # equivalent to: python -m vesselbridge3d.train ...
vb3d-infer ...   # equivalent to: python -m vesselbridge3d.inference ...
```

The commands below run these scripts via `uv run` within the managed environment. Alternatively, activate the environment once with `source .venv/bin/activate` and then call `vb3d-train ...` / `vb3d-infer ...` directly (or `python -m vesselbridge3d.train ...`). Run all commands from the repository root.

The architecture is selected with `--model_type` (default `dinov3_unetr`). New models are added under `vesselbridge3d/models/` and registered in the model registry; the same training and inference commands then work for any registered `--model_type`.

#### Comparison models (MedSAM / MedGemma)

To benchmark the DINOv3 encoder against other medical foundation encoders, two
additional `--model_type` values swap **only** the frozen 2D encoder while keeping
the entire downstream stack (3D ConvNeXt adapter, shared FFA aggregator, UNETR-Lite
decoder, refine heads) identical. This isolates encoder quality as the only variable.

| `--model_type` | Frozen encoder | hidden dim | patch | preset config |
|---|---|---|---|---|
| `dinov3_unetr` | DINOv3 ViT-S/16 (`facebook/dinov3-vits16-pretrain-lvd1689m`) | 384 | 16 | `configs/train_3d_default.yaml` |
| `medsam_unetr` | MedSAM SAM ViT-B (`wanglab/medsam-vit-base`) | 768 | 16 | `configs/train_3d_medsam.yaml` |
| `medgemma_unetr` | MedGemma SigLIP vision tower (`google/medgemma-4b-it`) | 1152 | 14 | `configs/train_3d_medgemma.yaml` |

Each encoder's per-layer outputs are projected with a 1×1 conv to a common working
dimension (384) before the aggregator, so trainable capacity and memory stay
comparable across models (for DINOv3 the projection is an identity).

For a controlled comparison, train all three with the same `--train_list` /
`--val_list` and the same schedule, then evaluate on the same in-distribution and
OOD `--test_list`:

```bash
uv run vb3d-train --config configs/train_3d_medsam.yaml   --train_list train.json --val_list val.json --save_dir runs/medsam
uv run vb3d-train --config configs/train_3d_medgemma.yaml --train_list train.json --val_list val.json --save_dir runs/medgemma
```

> **Caveats**
> - **Gated weights**: MedGemma requires accepting its license on the Hugging Face Hub and authenticating (`huggingface-cli login` or `export HF_TOKEN=...`). MedGemma loads the 4B VLM and keeps only its SigLIP vision tower.
> - **Input size**: the default 336×336 is shared by all models for a fair comparison. It is a multiple of both patch sizes (336 = 16×21 = 14×24). Running off each encoder's native resolution (MedSAM 1024, MedSigLIP 448) means the absolute position embeddings are interpolated — MedSigLIP can optionally be run at its native `--img_size 448`.
> - **Patch size differs** (DINOv3/MedSAM = 16, SigLIP = 14): `--img_size` must be a multiple of the selected encoder's patch size, which is validated at runtime.

> **Layout**
> The package root contains only the two runnable entry points, `train.py` and `inference.py`. All library code lives in subpackages: `models/` (architectures + registry), `data/` (datasets, preprocessing), `engine/` (training loop, inference, losses), and `common/` (config, constants, utilities).

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
uv run vb3d-train \
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

Alternatively, load preset values from a YAML config (CLI args override the config):
```bash
uv run vb3d-train \
  --config configs/train_3d_default.yaml \
  --train_list /path/to/train.json \
  --save_dir /path/to/save_dir
```
### 4. Inference

Predictions are produced via a sliding window over the Z axis (chunk size = 64 slices).  
Each chunk is processed independently, and results are stitched back to the original volume shape.  
Output segmentation masks are saved as NIfTI files in `--out_dir`, preserving the original affine and header.

```bash
uv run vb3d-infer \
  --test_list  /path/to/test.json \
  --checkpoint /path/to/checkpoint.pt \
  --out_dir    /path/to/output \
  --num_classes 16 \
  --img_size 336
```

Or with a preset config:
```bash
uv run vb3d-infer \
  --config configs/inference_3d_default.yaml \
  --test_list /path/to/test.json \
  --checkpoint /path/to/checkpoint.pt \
  --out_dir /path/to/output
```

> **Note**  
> - `--num_classes`, `--img_size`, `--vit_layers`, and `--decoder_up_factor` must match the values used during training  
> - Output masks are saved as `<case_id>.nii.gz` in `--out_dir`, with the same affine and header as the input volume

#### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `dinov3_unetr` | Model architecture key (see `vesselbridge3d.models` registry) |
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
