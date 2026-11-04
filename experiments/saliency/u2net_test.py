"""U^2-Net saliency (via rembg) on a thumbnail: what is the focus of the image?

Saves, next to the input (or under --out-dir):
  <name>.sal.png      grayscale saliency map (0-255)
  <name>.heat.png     heatmap blended over the image
  <name>.subjects.png thresholded subjects: connected components, outlined + labeled
and prints per-subject stats (area fraction, mean saliency, bbox).

Usage:
    python experiments/saliency/u2net_test.py thumb.jpg [--threshold 0.5] [--min-area 0.01]
    python experiments/saliency/u2net_test.py a.jpg b.jpg c.jpg --model u2netp

Needs: pip install "rembg[cpu]" pillow numpy scipy matplotlib
Weights (~176 MB for u2net) download on first run.
Models: u2net (default), u2netp (small/fast), isnet-general-use (often sharper).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rembg import new_session, remove
from scipy.ndimage import label, binary_opening, binary_fill_holes


def saliency(img, session):
    m = remove(img, session=session, only_mask=True)  # L-mode image, 0-255
    return np.asarray(m, dtype=np.float32) / 255.0


def heatmap(img, sal, alpha=0.55):
    import matplotlib
    colors = (matplotlib.colormaps["jet"](sal)[..., :3] * 255).astype(np.float32)
    base = np.asarray(img, dtype=np.float32)
    w = alpha * sal[..., None]
    return Image.fromarray((base * (1 - w) + colors * w).astype(np.uint8))


def subjects(sal, threshold, min_area):
    mask = binary_fill_holes(binary_opening(sal >= threshold, iterations=2))
    lab, n = label(mask)
    h, w = sal.shape
    out = []
    for i in range(1, n + 1):
        m = lab == i
        area = m.sum() / (h * w)
        if area < min_area:
            continue
        ys, xs = np.where(m)
        out.append(dict(mask=m, area=area, mean_sal=float(sal[m].mean()),
                        bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    out.sort(key=lambda s: -s["area"] * s["mean_sal"])
    return out


def draw_subjects(img, subs):
    rng = np.random.default_rng(1)
    arr = np.asarray(img, dtype=np.float32).copy()
    for s in subs:
        arr[s["mask"]] = 0.5 * arr[s["mask"]] + 0.5 * rng.integers(40, 255, 3)
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, s in enumerate(subs):
        d.rectangle(s["bbox"], outline="white", width=3)
        d.text((s["bbox"][0] + 6, s["bbox"][1] + 6),
               f"#{i} area={s['area']:.2f} sal={s['mean_sal']:.2f}", fill="white")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="u2net")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.01)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    session = new_session(args.model)
    for path in args.images:
        p = Path(path)
        img = Image.open(p).convert("RGB")
        sal = saliency(img, session)
        subs = subjects(sal, args.threshold, args.min_area)

        d = Path(args.out_dir) if args.out_dir else p.parent
        d.mkdir(parents=True, exist_ok=True)
        stem = d / p.stem
        Image.fromarray((sal * 255).astype(np.uint8)).save(f"{stem}.sal.png")
        heatmap(img, sal).save(f"{stem}.heat.png")
        draw_subjects(img, subs).save(f"{stem}.subjects.png")

        print(f"{p.name}: focal fraction={float((sal >= args.threshold).mean()):.2f}, {len(subs)} subject(s)")
        for i, s in enumerate(subs):
            print(f"  #{i} area={s['area']:.3f} mean_sal={s['mean_sal']:.2f} bbox={s['bbox']}")


if __name__ == "__main__":
    main()
