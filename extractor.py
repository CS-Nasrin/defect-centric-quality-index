from __future__ import annotations
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

PATCH_SIZE = 14

class DinoV2Extractor(nn.Module):
    """Extract DINOv2 patch tokens from an image."""

    def __init__(self, arch: str = "dinov2_vits14", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.model = self._load_model(arch).to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _load_model(self, arch: str):
        # 1) Try repo-local loader (optional)
        try:
            from models.dinov2 import load_dinov2  # type: ignore
            print("[info] Using local models.dinov2 loader.")
            return load_dinov2(arch)
        except Exception:
            pass

        # 2) Torch hub fallback
        print("[info] Loading DINOv2 via torch.hub …")
        return torch.hub.load("facebookresearch/dinov2", arch)

    @torch.no_grad()
    def forward_tokens(self, pil_img: Image.Image, resize_shorter: int = 448) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Return patch tokens [1, N, D] and token grid (ht, wt)."""

        img = pil_img.convert("RGB")
        W0, H0 = img.size

        if resize_shorter > 0:
            scale = resize_shorter / float(min(W0, H0))
            W1, H1 = int(round(W0 * scale)), int(round(H0 * scale))
            img = img.resize((W1, H1), Image.BICUBIC)
        else:
            W1, H1 = W0, H0

        # pad to multiple of PATCH_SIZE
        pad_h = (PATCH_SIZE - (H1 % PATCH_SIZE)) % PATCH_SIZE
        pad_w = (PATCH_SIZE - (W1 % PATCH_SIZE)) % PATCH_SIZE
        if pad_h or pad_w:
            img = Image.fromarray(np.pad(np.array(img),
                                         ((0, pad_h), (0, pad_w), (0, 0)),
                                         mode='reflect'))
            W1 += pad_w
            H1 += pad_h

        x = self.transform(img).unsqueeze(0).to(self.device)
        out = self.model.forward_features(x)

        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            tokens = out["x_norm_patchtokens"]
        elif hasattr(out, "x_norm_patchtokens"):
            tokens = out.x_norm_patchtokens
        else:
            raise RuntimeError("forward_features did not return x_norm_patchtokens.")

        ht, wt = H1 // PATCH_SIZE, W1 // PATCH_SIZE
        return tokens, (ht, wt)