import argparse, torch
import torch.nn.functional as F
from dqi import DinoV2Extractor, list_paths, match_by_stem, is_image
from dqi.bank import score_images, BankMeta
from dqi.io_utils import write_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn-images", required=True)
    p.add_argument("--syn-masks", required=True)
    p.add_argument("--mask-suffix", default="")
    p.add_argument("--bank-in", required=True)
    p.add_argument("--scores-out", default="scores.csv")
    p.add_argument("--resize-shorter", type=int, default=448)
    p.add_argument("--keep-ratio", type=float, default=0.7)
    p.add_argument("--topk-percent", type=float, default=1.0)
    p.add_argument("--use-faiss", action="store_true")
    p.add_argument("--arch", default="dinov2_vits14")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    extractor = DinoV2Extractor(arch=args.arch, device=args.device)

    pkg = torch.load(args.bank_in, map_location="cpu")
    bank = F.normalize(pkg["bank"], dim=1)
    _ = BankMeta(**pkg["meta"]) if "meta" in pkg else None

    imgs = [x for x in list_paths(args.syn_images) if is_image(x)]
    msks = [x for x in list_paths(args.syn_masks) if is_image(x)]
    pairs = match_by_stem(imgs, msks, mask_suffix=args.mask_suffix)

    rows = score_images(extractor, bank, pairs, args.resize_shorter, args.keep_ratio, args.topk_percent, args.use_faiss)
    write_csv(rows, args.scores_out)
    print(f"[ok] Wrote scores: {args.scores-out}")

if __name__ == "__main__":
    main()
