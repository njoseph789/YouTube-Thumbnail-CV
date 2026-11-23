"""Run the OWLv2 (ViT) open-vocabulary detector over the labeled subset.

Same outputs and column layout as run_baseline.py, so evaluate.py and draw_boxes.py
work on it unchanged (model name "owlv2"):

* results/owlv2/detections.csv      one row per (box, keyword) above --threshold
* results/owlv2/keyword_scores.csv  threshold-free max score per keyword per image

Differences from Grounding DINO worth knowing:
* OWLv2 scores each query independently, so all 153 keywords go in one forward pass
  (no per-category prompts). Queries are the plain keyword strings.
* Its sigmoid scores are lower and calibrated differently, hence the 0.10 default
  threshold. mAP does not depend on the threshold; F1 does.
* Boxes are relative to the image padded to a square (bottom/right padding), so they
  are scaled by max(w, h) and clipped to the real image.
* It emits many overlapping boxes per object, so detections get per-keyword NMS
  (IoU 0.3). keyword_scores.csv is taken before NMS and is unaffected.

Usage (compviz venv):
    python experiments/object_detection/run_owlv2.py --limit 3
    python experiments/object_detection/run_owlv2.py
"""
import argparse
import sys
import time
from pathlib import Path

import mlflow
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from transformers import AutoProcessor, Owlv2ForObjectDetection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KEYWORDS  # noqa: E402
from run_baseline import DET_COLS, IMAGE_DIR, write_csv  # noqa: E402
from tracking import setup_mlflow  # noqa: E402

MODEL_NAME = "owlv2"
MODEL_ID = "google/owlv2-base-patch16-ensemble"
KW_LIST = [(cat, kw) for cat, kws in KEYWORDS.items() for kw in kws]


@torch.inference_mode()
def run(args, device, image_paths):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
    if args.fp16:
        model = model.half()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    queries = [kw for _, kw in KW_LIST]

    det_rows, score_rows, elapsed = [], [], 0.0
    for path in tqdm(image_paths, desc=MODEL_NAME):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(text=[queries], images=img, return_tensors="pt").to(device)
        if args.fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        out = model(**inputs)
        probs = out.logits[0].float().sigmoid()  # (patches, keywords)
        boxes = out.pred_boxes[0].float()  # (patches, 4) cx, cy, w, h in padded square
        side = float(max(w, h))
        xyxy = torch.stack(
            [boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], dim=1) * side
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h)

        img_scores = probs.max(dim=0).values.cpu().tolist()
        for (cat, kw), sc in zip(KW_LIST, img_scores):
            score_rows.append([path.name, MODEL_NAME, cat, kw, f"{sc:.5f}"])

        for j, (cat, kw) in enumerate(KW_LIST):
            cand = (probs[:, j] >= args.threshold).nonzero().flatten()
            if cand.numel() == 0:
                continue
            keep = cand[nms(xyxy[cand], probs[cand, j], args.nms_iou)]
            for q in keep.tolist():
                x1, y1, x2, y2 = xyxy[q].tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                det_rows.append([
                    path.name, MODEL_NAME, cat, kw, f"{probs[q, j].item():.5f}",
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{x1 / w:.5f}", f"{y1 / h:.5f}", f"{x2 / w:.5f}", f"{y2 / h:.5f}", w, h,
                ])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
    return det_rows, score_rows, elapsed, peak_gb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    ap.add_argument("--fp16", action="store_true", help="half precision (less GPU memory)")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--experiment", default="grounding_dino_baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[: args.limit]
    setup_mlflow(args.out, args.experiment)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={gpu}  images={len(image_paths)}  model={MODEL_ID}  fp16={args.fp16}")

    det_rows, score_rows, elapsed, peak_gb = run(args, device, image_paths)
    model_dir = args.out / MODEL_NAME
    model_dir.mkdir(exist_ok=True)
    det_csv, score_csv = model_dir / "detections.csv", model_dir / "keyword_scores.csv"
    write_csv(det_csv, DET_COLS, det_rows)
    write_csv(score_csv, ["image", "model", "category", "keyword", "score"], score_rows)

    n = max(len(image_paths), 1)
    with mlflow.start_run(run_name=MODEL_NAME) as run_:
        mlflow.log_params({
            "model": MODEL_NAME, "model_id": MODEL_ID, "box_threshold": args.threshold,
            "nms_iou": args.nms_iou, "n_images": len(image_paths),
            "dtype": "fp16" if args.fp16 else "fp32", "device": gpu,
        })
        mlflow.log_metrics({
            "n_detections": len(det_rows), "detections_per_image": len(det_rows) / n,
            "seconds_per_image": elapsed / n, "peak_gpu_mem_gb": peak_gb,
        })
        mlflow.log_artifact(str(det_csv))
        mlflow.log_artifact(str(score_csv))
        (model_dir / "run_id.txt").write_text(run_.info.run_id)
    print(f"{MODEL_NAME}: {len(det_rows)} detections, {elapsed / n:.2f}s/image, "
          f"peak GPU {peak_gb:.2f} GB -> {model_dir}")


if __name__ == "__main__":
    main()
