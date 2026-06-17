"""Shared default constants for vesselbridge3d train / inference."""

from __future__ import annotations

# Default model architecture key (see vesselbridge3d.models registry).
DEFAULT_MODEL_TYPE = "dinov3_unetr"

# Default in-plane input size (parsed by parse_img_size; must be multiples of 16).
DEFAULT_IMG_SIZE = "336"

# Default ViT layers to extract patch grids from.
DEFAULT_VIT_LAYERS = "2,5,8,11"

# Decoder base channel width.
DEFAULT_DECODER_BASE_CHANNELS = 128

# Default decoder in-plane upsampling factor (1/16 -> 1/4).
DEFAULT_DECODER_UP_FACTOR = 4

# Depth chunk size for the inference sliding window over Z.
CROP_D = 64
