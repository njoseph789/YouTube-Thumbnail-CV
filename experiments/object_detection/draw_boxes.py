"""Draw the detected boxes from detections.csv onto the 100-image labeled subset.

Writes, under results/annotated/ (gitignored with the rest of results/):
    <model>/<image>.jpg     one annotated image per model
    compare/<image>.jpg     tiny (left) and base (right) side by side

Boxes are colored by prompt category; each label shows "keyword score".

Usage (compviz venv), after run_baseline.py:
    python experiments/object_detection/draw_boxes.py
    python experiments/object_detection/draw_boxes.py --min-score 0.4 --models base
"""
import argparse
import colorsys
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE.parents[1] / "images" / "labeled_subset"
RESULTS = HERE / "results"


def category_colors(categories):
    cats = sorted(categories)
    return {
        c: tuple(int(v * 255) for v in colorsys.hsv_to_rgb(i / len(cats), 0.9, 1.0))
        for i, c in enumerate(cats)
    }


def load_detections(model, min_score):
    by_image = defaultdict(list)
    with open(RESULTS / model / "detections.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["score"]) >= min_score:
                by_image[r["image"]].append(r)
    return by_image


def get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(img, dets, colors):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(max(12, img.height // 45))
    width = max(2, img.height // 360)
    # draw low-confidence first so confident boxes end up on top
    for d in sorted(dets, key=lambda d: float(d["score"])):
        box = [float(d[k]) for k in ("x1", "y1", "x2", "y2")]
        color = colors[d["prompt_category"]]
        draw.rectangle(box, outline=color, width=width)
        text = f'{d["label"]} {float(d["score"]):.2f}'
        tb = draw.textbbox((box[0], box[1]), text, font=font)
        h = tb[3] - tb[1]
        y = box[1] - h - 4 if box[1] - h - 4 >= 0 else box[1]
        tb = draw.textbbox((box[0], y), text, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
        draw.text((box[0], y), text, fill="black", font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny", "base"])
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="hide detections below this score (tiny is very cluttered)")
    ap.add_argument("--out", type=Path, default=RESULTS / "annotated")
    args = ap.parse_args()

    dets = {m: load_detections(m, args.min_score) for m in args.models}
    cats = {d["prompt_category"] for m in dets.values() for ds in m.values() for d in ds}
    colors = category_colors(cats)
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    both = len(args.models) == 2

    for m in args.models:
        (args.out / m).mkdir(parents=True, exist_ok=True)
    if both:
        (args.out / "compare").mkdir(parents=True, exist_ok=True)

    for p in images:
        img = Image.open(p)
        rendered = []
        for m in args.models:
            a = annotate(img, dets[m].get(p.name, []), colors)
            a.save(args.out / m / p.name, quality=92)
            rendered.append(a)
        if both:
            w, h = rendered[0].size
            canvas = Image.new("RGB", (w * 2, h))
            canvas.paste(rendered[0], (0, 0))
            canvas.paste(rendered[1], (w, 0))
            ImageDraw.Draw(canvas).text((8, 8), f"{args.models[0]} | {args.models[1]}",
                                        fill="white", font=get_font(24))
            canvas.save(args.out / "compare" / p.name, quality=90)

    print(f"{len(images)} images -> {args.out}")
    print("category colors:", {c: "#%02x%02x%02x" % col for c, col in colors.items()})


if __name__ == "__main__":
    main()
