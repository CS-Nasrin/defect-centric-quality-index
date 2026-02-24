import argparse, os, torch
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dqi import DinoV2Extractor, list_paths, match_by_stem, is_image
from dqi.bank import build_defect_bank

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-images", required=True)
    p.add_argument("--real-masks", required=True)
    p.add_argument("--mask-suffix", default="")
    p.add_argument("--bank-out", default="defect_bank.pt")
    p.add_argument("--resize-shorter", type=int, default=448)
    p.add_argument("--keep-ratio", type=float, default=0.7)
    p.add_argument("--rotations", default="0")
    p.add_argument("--shot", type=int, default=None)
    p.add_argument("--arch", default="dinov2_vits14")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    extractor = DinoV2Extractor(arch=args.arch, device=args.device)

    imgs = [x for x in list_paths(args.real_images) if is_image(x)]
    msks = [x for x in list_paths(args.real_masks) if is_image(x)]
    pairs = match_by_stem(imgs, msks, mask_suffix=args.mask_suffix)

    if args.shot:
        pairs = pairs[:args.shot]

    rotations = [int(x) for x in args.rotations.split(",") if x.strip() != ""]
    bank, meta = build_defect_bank(extractor, pairs, args.resize_shorter, args.keep_ratio, rotations)

    os.makedirs(os.path.dirname(args.bank_out) or ".", exist_ok=True)
    torch.save({"bank": bank.cpu(), "meta": meta.__dict__}, args.bank_out)
    print(f"[ok] Saved bank: {args.bank_out}  N={bank.shape[0]} D={bank.shape[1]}")

if __name__ == "__main__":
    main()
