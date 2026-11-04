"""Copy N random images from a local folder tree (e.g. the synced Drive dataset) into Test Images/.

Usage:
    python experiments/sample_images.py /path/to/dataset_root [-n 100] [--seed 0] [--clear]

Files are prefixed with their channel folder (e.g. "WolfeyVGC__abc123.jpg") so names stay unique
and you can tell where each image came from. Same seed -> same sample.
"""
import argparse
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "Test Images"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", action="store_true", help="delete existing images in Test Images first")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    files = sorted(f for f in src.rglob("*") if f.suffix.lower() in EXTS and DEST not in f.parents)
    print(f"found {len(files)} images under {src}")
    pick = random.Random(args.seed).sample(files, min(args.n, len(files)))

    DEST.mkdir(exist_ok=True)
    if args.clear:
        for f in DEST.iterdir():
            if f.suffix.lower() in EXTS:
                f.unlink()
    for f in pick:
        rel = f.relative_to(src)
        prefix = f.parent.name if f.parent != src else "root"
        shutil.copy2(f, DEST / f"{prefix}__{f.name}")
    print(f"copied {len(pick)} images to {DEST}")


if __name__ == "__main__":
    main()
