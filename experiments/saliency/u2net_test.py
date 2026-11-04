"""U^2-Net saliency (via rembg) on a thumbnail: what is the focus of the image?

Saves to experiments/saliency/results/<kind>/<image name>.png (override with --out-dir):
  sal/<name>.png       grayscale saliency map (0-255)
  heat/<name>.png      heatmap blended over the image
  subjects/<name>.png  thresholded subjects: connected components, outlined + labeled
and prints per-subject stats (area fraction, mean saliency, bbox).

Usage:
    python experiments/saliency/u2net_test.py [images or folders ...] [--threshold 0.5] [--min-area 0.01]
  With no arguments it processes every image in experiments/Test Images/.

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

HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent / "Test Images"
RESULTS_DIR = HERE / "results"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(paths):
    """Expand files/folders into image files; no args -> the Test Images."""
    out = []
    for p in map(Path, paths or [TEST_DIR]):
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    return out



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
    ap.add_argument("images", nargs="*", help="image files and/or folders")
    ap.add_argument("--model", default="u2net")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.01)
    ap.add_argument("--out-dir", default=None, help="results root (default: experiments/saliency/results)")
    args = ap.parse_args()

    session = new_session(args.model)
    for p in collect_images(args.images):
        img = Image.open(p).convert("RGB")
        sal = saliency(img, session)
        subs = subjects(sal, args.threshold, args.min_area)

        root = Path(args.out_dir or RESULTS_DIR)
        for kind in ("sal", "heat", "subjects"):
            (root / kind).mkdir(parents=True, exist_ok=True)
        Image.fromarray((sal * 255).astype(np.uint8)).save(root / "sal" / f"{p.stem}.png")
        heatmap(img, sal).save(root / "heat" / f"{p.stem}.png")
        draw_subjects(img, subs).save(root / "subjects" / f"{p.stem}.png")

        print(f"{p.name}: focal fraction={float((sal >= args.threshold).mean()):.2f}, {len(subs)} subject(s)")
        for i, s in enumerate(subs):
            print(f"  #{i} area={s['area']:.3f} mean_sal={s['mean_sal']:.2f} bbox={s['bbox']}")


if __name__ == "__main__":
    main()
