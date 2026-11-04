"""SAM automatic masks -> filter/merge into object-level masks for a thumbnail.

Pipeline:
  1. Generate all masks with SAM (Hugging Face `mask-generation` pipeline).
  2. Drop masks that are too small (parts, specks) or too large (background).
  3. De-duplicate near-identical masks (IoU NMS).
  4. Merge parts into parents: if a mask lies mostly inside another mask with its holes
     filled (eyes in a face, face in a hair ring, glasses on a head), absorb it.
  5. Rank survivors by size + centrality + SAM score; keep the top-k.

Usage:
    python experiments/sam/sam_test.py thumb.jpg [--top-k 4] [--save-masks] [--raw]

Needs: pip install transformers torch pillow numpy scipy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from transformers import pipeline


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def filter_and_merge(masks, scores, min_area, max_area, dedupe_iou, contain):
    h, w = masks[0].shape
    total = h * w
    items = [(m, s) for m, s in zip(masks, scores) if min_area <= m.sum() / total <= max_area]
    items.sort(key=lambda t: -t[0].sum())

    # IoU de-duplication (keep the larger / earlier one)
    deduped = []
    for m, s in items:
        if all(iou(m, k) < dedupe_iou for k, _ in deduped):
            deduped.append((m, s))

    # Merge parts into parents (largest first, so parents exist before their parts)
    kept = []  # [mask, score, n_merged]
    for m, s in deduped:
        parent = None
        for k in kept:
            filled = binary_fill_holes(k[0])
            if np.logical_and(m, filled).sum() / max(m.sum(), 1) >= contain:
                parent = k
                break
        if parent is not None:
            parent[0] = parent[0] | m
            parent[2] += 1
        else:
            kept.append([m, float(s), 0])
    return kept


def rank(kept, top_k):
    h, w = kept[0][0].shape if kept else (1, 1)
    ys, xs = np.mgrid[0:h, 0:w]
    out = []
    for m, s, n in kept:
        area = m.sum() / (h * w)
        cy, cx = ys[m].mean() / h, xs[m].mean() / w
        center = 1 - np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5)
        out.append((0.5 * np.sqrt(area) + 0.3 * center + 0.2 * s, m, n))
    out.sort(key=lambda t: -t[0])
    return out[:top_k]


def draw(img, items, labels=True):
    rng = np.random.default_rng(0)
    arr = np.asarray(img, dtype=np.float32).copy()
    for _, m, _ in items:
        arr[m] = 0.45 * arr[m] + 0.55 * rng.integers(40, 255, 3)
    out = Image.fromarray(arr.astype(np.uint8))
    if labels:
        d = ImageDraw.Draw(out)
        for i, (sc, m, n) in enumerate(items):
            ys, xs = np.where(m)
            d.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline="white", width=3)
            d.text((xs.min() + 6, ys.min() + 6), f"#{i} {sc:.2f} +{n}", fill="white")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--out", default=None, help="output overlay path")
    ap.add_argument("--points-per-batch", type=int, default=32)
    ap.add_argument("--min-area", type=float, default=0.02, help="min mask area as fraction of image")
    ap.add_argument("--max-area", type=float, default=0.75, help="max mask area (drops background)")
    ap.add_argument("--dedupe-iou", type=float, default=0.85)
    ap.add_argument("--contain", type=float, default=0.85, help="fraction inside a parent to be merged into it")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--save-masks", action="store_true", help="also save each kept mask as a PNG")
    ap.add_argument("--raw", action="store_true", help="also save the unfiltered overlay for comparison")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gen = pipeline("mask-generation", model=args.model, device=device)
    img = Image.open(args.image).convert("RGB")
    result = gen(img, points_per_batch=args.points_per_batch)
    masks = [np.asarray(m, dtype=bool) for m in result["masks"]]
    scores = result.get("scores", [1.0] * len(masks))
    print(f"{len(masks)} raw masks on {device}")

    base = Path(args.out) if args.out else Path(args.image).with_suffix(".sam.png")

    if args.raw:
        raw_items = [(0.0, m, 0) for m in masks]
        draw(img, raw_items, labels=False).save(base.with_name(base.stem + ".raw.png"))

    kept = filter_and_merge(masks, scores, args.min_area, args.max_area, args.dedupe_iou, args.contain)
    print(f"{len(kept)} after filter/dedupe/merge")
    if not kept:
        print("nothing survived; loosen --min-area/--max-area")
        return
    top = rank(kept, args.top_k)
    draw(img, top).save(base)
    print("saved", base)

    if args.save_masks:
        for i, (_, m, _) in enumerate(top):
            p = base.with_name(f"{base.stem}.mask{i}.png")
            Image.fromarray((m * 255).astype(np.uint8)).save(p)
            print("saved", p)


if __name__ == "__main__":
    main()
