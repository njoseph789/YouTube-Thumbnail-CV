"""Run baseline Grounding DINO models (Hugging Face) over the labeled subset.

For every model, every image and every category prompt in prompts.py it does one
forward pass and records:

* detections.csv      - long CSV, one row per (box, keyword) above the thresholds
                        (image, model, prompt_category, label, score, x1..y2 in
                        pixels, x1_norm..y2_norm, img_w, img_h)
* keyword_scores.csv  - threshold-free score for every keyword in every image
                        (image, model, category, keyword, score). This is what
                        image-level mAP is computed from.

Each model is one MLflow run in a single experiment so results compare directly.

Usage (compviz venv):
    python experiments/object_detection/run_baseline.py --limit 3 --models tiny
    python experiments/object_detection/run_baseline.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import mlflow
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import BOX_THRESHOLD, KEYWORDS, PROMPTS, TEXT_THRESHOLD  # noqa: E402
from tracking import setup_mlflow  # noqa: E402

IMAGE_DIR = HERE.parents[1] / "images" / "labeled_subset"
MODEL_IDS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}
DET_COLS = [
    "image", "model", "prompt_category", "label", "score",
    "x1", "y1", "x2", "y2",
    "x1_norm", "y1_norm", "x2_norm", "y2_norm", "img_w", "img_h",
]


def phrase_spans(input_ids, dot_id, n_expected):
    """Token spans (start, end) for each '.'-separated phrase of the prompt."""
    spans, start = [], 1  # position 0 is [CLS]
    for i, tok in enumerate(input_ids):
        if tok == dot_id:
            spans.append((start, i))
            start = i + 1
    if len(spans) != n_expected:
        raise ValueError(f"expected {n_expected} phrases, found {len(spans)}")
    return spans


@torch.inference_mode()
def run_model(name, args, device, image_paths):
    model_id = MODEL_IDS[name]
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    if args.fp16:
        model = model.half()
    dot_id = processor.tokenizer.convert_tokens_to_ids(".")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    det_rows, score_rows = [], []
    elapsed = 0.0
    for path in tqdm(image_paths, desc=name):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for cat, prompt in PROMPTS.items():
            kws = KEYWORDS[cat]
            inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
            if args.fp16:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            spans = phrase_spans(inputs["input_ids"][0].tolist(), dot_id, len(kws))
            out = model(**inputs)
            probs = out.logits[0].float().sigmoid()  # (queries, tokens)
            boxes = out.pred_boxes[0].float()  # (queries, 4) cx, cy, w, h normalized
            ph = torch.stack([probs[:, a:b].max(dim=1).values for a, b in spans], dim=1)

            img_scores = ph.max(dim=0).values.cpu().tolist()
            for kw, sc in zip(kws, img_scores):
                score_rows.append([path.name, name, cat, kw, f"{sc:.5f}"])

            keep = (ph.max(dim=1).values >= args.box_threshold).nonzero().flatten().tolist()
            for q in keep:
                cx, cy, bw, bh = boxes[q].tolist()
                xn1, yn1 = max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0)
                xn2, yn2 = min(cx + bw / 2, 1.0), min(cy + bh / 2, 1.0)
                for p, kw in enumerate(kws):
                    sc = ph[q, p].item()
                    if sc >= args.text_threshold:
                        det_rows.append([
                            path.name, name, cat, kw, f"{sc:.5f}",
                            f"{xn1 * w:.1f}", f"{yn1 * h:.1f}", f"{xn2 * w:.1f}", f"{yn2 * h:.1f}",
                            f"{xn1:.5f}", f"{yn1:.5f}", f"{xn2:.5f}", f"{yn2:.5f}", w, h,
                        ])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return det_rows, score_rows, elapsed, peak_gb


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tiny", "base"], choices=list(MODEL_IDS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    ap.add_argument("--fp16", action="store_true", help="half precision (less GPU memory)")
    ap.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--experiment", default="grounding_dino_baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[: args.limit]
    setup_mlflow(args.out, args.experiment)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={gpu}  images={len(image_paths)}  models={args.models}  fp16={args.fp16}")

    for name in args.models:
        det_rows, score_rows, elapsed, peak_gb = run_model(name, args, device, image_paths)
        model_dir = args.out / name
        model_dir.mkdir(exist_ok=True)
        det_csv, score_csv = model_dir / "detections.csv", model_dir / "keyword_scores.csv"
        write_csv(det_csv, DET_COLS, det_rows)
        write_csv(score_csv, ["image", "model", "category", "keyword", "score"], score_rows)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({
                "model": name,
                "model_id": MODEL_IDS[name],
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "n_images": len(image_paths),
                "prompt_categories": ",".join(PROMPTS),
                "dtype": "fp16" if args.fp16 else "fp32",
                "device": gpu,
            })
            mlflow.log_metrics({
                "n_detections": len(det_rows),
                "detections_per_image": len(det_rows) / max(len(image_paths), 1),
                "seconds_per_image": elapsed / max(len(image_paths), 1),
                "peak_gpu_mem_gb": peak_gb,
            })
            mlflow.log_artifact(str(det_csv))
            mlflow.log_artifact(str(score_csv))
            (model_dir / "run_id.txt").write_text(run.info.run_id)
        print(f"{name}: {len(det_rows)} detections, {elapsed / max(len(image_paths), 1):.2f}s/image, "
              f"peak GPU {peak_gb:.2f} GB -> {model_dir}")


if __name__ == "__main__":
    main()
