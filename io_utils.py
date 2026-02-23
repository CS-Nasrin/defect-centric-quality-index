from __future__ import annotations
import os, glob, csv
from typing import List, Tuple, Dict
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_image(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_paths(dir_or_glob: str) -> List[str]:
    if os.path.isdir(dir_or_glob):
        out = []
        for e in IMG_EXTS:
            out.extend(glob.glob(os.path.join(dir_or_glob, f"**/*{e}"), recursive=True))
        return sorted(out)
    return sorted(glob.glob(dir_or_glob, recursive=True))

def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def match_by_stem(imgs: List[str], masks: List[str], mask_suffix: str = "") -> List[Tuple[str, str]]:
    mask_map = {stem(m): m for m in masks}
    pairs = []
    for im in imgs:
        st = stem(im)
        if st in mask_map:
            pairs.append((im, mask_map[st])); continue
        if mask_suffix and (st + mask_suffix) in mask_map:
            pairs.append((im, mask_map[st + mask_suffix])); continue
        if mask_suffix and st.endswith(mask_suffix):
            base = st[: -len(mask_suffix)]
            if base in mask_map:
                pairs.append((im, mask_map[base])); continue
        print(f"[warn] No mask matched for image: {im}")
    return pairs

def load_image(p: str) -> Image.Image:
    return Image.open(p).convert("RGB")

def load_mask(p: str) -> Image.Image:
    return Image.open(p).convert("L")

def write_csv(rows: List[Dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)