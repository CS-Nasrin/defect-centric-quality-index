from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, object
from PIL import Image
import torch
import torch.nn.functional as F

from .masks import mask_token_keep_indices, rotate_pair
from .knn import torch_knn_cosine_dist, faiss_knn_cosine_dist, has_faiss
from .io_utils import load_image, load_mask

@dataclass
class BankMeta:
    arch: str
    dim: int
    resize_shorter: int
    patch_size: int = 14
    normalized: bool = True

def build_defect_bank(extractor, pairs: List[Tuple[str, str]], resize_shorter: int, keep_ratio: float, rotations: List[int]):
    feats_all: List[torch.Tensor] = []
    for i, (ip, mp) in enumerate(pairs):
        img0 = load_image(ip)
        m0 = load_mask(mp)
        for deg in rotations:
            img, m = rotate_pair(img0, m0, deg)
            tokens, (ht, wt) = extractor.forward_tokens(img, resize_shorter)
            keep_idx = mask_token_keep_indices(m, (ht, wt), keep_ratio)

            tok = tokens[0]
            if keep_idx.sum().item() == 0:
                continue
            feats_all.append(tok[keep_idx])

    if len(feats_all) == 0:
        raise RuntimeError("No features collected for bank. Check masks/keep_ratio.")

    bank = torch.cat(feats_all, dim=0)
    bank = F.normalize(bank, dim=1)

    meta = BankMeta(arch="dinov2_vits14", dim=bank.shape[1], resize_shorter=resize_shorter)
    return bank, meta

def score_images(extractor, bank: torch.Tensor, pairs: List[Tuple[str, str]], resize_shorter: int,
                 keep_ratio: float, topk_percent: float, use_faiss: bool):
    results: List[Dict[str, object]] = []
    bank_n = F.normalize(bank, dim=1).to(extractor.device)

    for (ip, mp) in pairs:
        img = load_image(ip)
        m = load_mask(mp)
        tokens, (ht, wt) = extractor.forward_tokens(img, resize_shorter)
        keep_idx = mask_token_keep_indices(m, (ht, wt), keep_ratio)

        tok = tokens[0]
        if keep_idx.sum().item() == 0:
            results.append({"image": ip, "mask": mp, "n_kept": 0, "score": float("nan")})
            continue

        q = F.normalize(tok[keep_idx], dim=1)

        if use_faiss and has_faiss():
            dist = faiss_knn_cosine_dist(q, bank_n, use_gpu=True)
        else:
            dist = torch_knn_cosine_dist(q, bank_n)

        dist = dist.cpu()
        k = max(1, int(math.ceil(len(dist) * (topk_percent / 100.0))))
        score = float(torch.topk(dist, k).values.mean().item())

        results.append({"image": ip, "mask": mp, "n_kept": int(keep_idx.sum().item()), "score": score})

    return results