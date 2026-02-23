from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def mask_token_keep_indices(mask_img: Image.Image, token_grid: Tuple[int, int], keep_ratio: float) -> torch.Tensor:
    """Return boolean vector [ht*wt] selecting tokens whose mask coverage >= keep_ratio."""
    ht, wt = token_grid
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")

    H, W = mask_img.height, mask_img.width

    # pad mask so divisible by token grid
    pad_h = (ht - (H % ht)) % ht
    pad_w = (wt - (W % wt)) % wt
    if pad_h or pad_w:
        arr = np.array(mask_img)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
        mask_img = Image.fromarray(arr)
        H, W = mask_img.height, mask_img.width

    ps_h = H // ht
    ps_w = W // wt

    # force exact alignment
    if (mask_img.width, mask_img.height) != (wt * ps_w, ht * ps_h):
        mask_img = mask_img.resize((wt * ps_w, ht * ps_h), Image.NEAREST)
        H, W = mask_img.height, mask_img.width
        ps_h = H // ht
        ps_w = W // wt

    m = np.array(mask_img, dtype=np.float32) / 255.0
    mt = torch.from_numpy(m)[None, None]
    pooled = F.avg_pool2d(mt, kernel_size=(ps_h, ps_w), stride=(ps_h, ps_w)).flatten()
    keep = pooled >= keep_ratio
    return keep

def rotate_pair(img: Image.Image, mask: Image.Image, deg: int):
    if deg % 360 == 0:
        return img, mask
    return (
        img.rotate(deg, resample=Image.BICUBIC, expand=False),
        mask.rotate(deg, resample=Image.NEAREST, expand=False),
    )