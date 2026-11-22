"""
Face bounding boxes for YouTube thumbnails: run several detectors, compare them,
and write the final faces.csv (x/y/height etc.) for training + generation eval.

COLAB SETUP (run once in a cell):
    !pip -q install insightface onnxruntime-gpu ultralytics huggingface_hub facenet-pytorch mediapipe opencv-python-headless pandas tqdm
    (onnxruntime-gpu -> use `onnxruntime` if the runtime has no GPU)
    Optional: !pip -q install retina-face tf-keras

USAGE
    # 1) Run all detectors on the 100-image subset, write per-model CSVs + comparison + overlays
    python face_boxes.py run --images /content/drive/MyDrive/thumbs --out face_results

    # 2) (Optional) hand-label ~30-50 images, same CSV schema as detections, then score vs. truth
    python face_boxes.py run --images ... --out face_results --truth truth.csv

    # 3) Produce the final faces.csv from the chosen model at a chosen threshold
    python face_boxes.py final --out face_results --model scrfd --conf 0.5

CSV schema (all coords normalized 0-1 relative to image size):
    image, face_idx, conf, x, y, w, h, cx, cy
    x,y = top-left corner; h = box height (the "height" the team needs); cx,cy = center.
    truth.csv uses the same columns (conf can be 1).
"""
import argparse, math, os, time, urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LOW_CONF = 0.2  # detectors run with a low floor; threshold applied at scoring time
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


# ---------------------------------------------------------------- detectors
# Each detector: __call__(bgr_image) -> list of (x1, y1, x2, y2, conf) in pixels.
class SCRFD:  # InsightFace detector (buffalo_l = SCRFD-10G)
    def __init__(self, det_size=(960, 960)):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=LOW_CONF)

    def __call__(self, img):
        return [(*f.bbox, float(f.det_score)) for f in self.app.get(img)]


class YuNet:  # OpenCV, tiny
    def __init__(self):
        p = "face_detection_yunet_2023mar.onnx"
        if not os.path.exists(p):
            urllib.request.urlretrieve(YUNET_URL, p)
        self.det = cv2.FaceDetectorYN.create(p, "", (320, 320), LOW_CONF, 0.3, 5000)

    def __call__(self, img):
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return []
        return [(f[0], f[1], f[0] + f[2], f[1] + f[3], float(f[-1])) for f in faces]


class YoloFace:  # YOLOv8 face weights from HuggingFace
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        self.m = YOLO(hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt"))

    def __call__(self, img):
        r = self.m.predict(img, conf=LOW_CONF, imgsz=1280, verbose=False)[0]
        return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]


class MTCNNDet:
    def __init__(self):
        import torch
        from facenet_pytorch import MTCNN
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = MTCNN(keep_all=True, device=dev, thresholds=[0.5, 0.6, 0.6], min_face_size=12)

    def __call__(self, img):
        from PIL import Image
        boxes, probs = self.m.detect(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if boxes is None:
            return []
        return [(*b, float(p)) for b, p in zip(boxes, probs) if p >= LOW_CONF]


class MediaPipe:
    def __init__(self):
        import mediapipe as mp
        self.m = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                           min_detection_confidence=LOW_CONF)

    def __call__(self, img):
        h, w = img.shape[:2]
        res = self.m.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = []
        for d in res.detections or []:
            b = d.location_data.relative_bounding_box
            out.append((b.xmin * w, b.ymin * h, (b.xmin + b.width) * w,
                        (b.ymin + b.height) * h, float(d.score[0])))
        return out


class RetinaFaceDet:  # TF-based; most fragile install, so optional
    def __init__(self):
        from retinaface import RetinaFace
        self.rf = RetinaFace

    def __call__(self, img):
        r = self.rf.detect_faces(img, threshold=LOW_CONF)
        if not isinstance(r, dict):
            return []
        return [(*v["facial_area"], float(v["score"])) for v in r.values()]


DETECTORS = {"scrfd": SCRFD, "yunet": YuNet, "yolo": YoloFace,
             "mtcnn": MTCNNDet, "mediapipe": MediaPipe, "retinaface": RetinaFaceDet}


# ---------------------------------------------------------------- helpers
def to_rows(name, dets, W, H):
    rows = []
    dets = sorted(dets, key=lambda d: -d[4])
    for i, (x1, y1, x2, y2, c) in enumerate(dets):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        x, y, w, h = x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H
        rows.append(dict(image=name, face_idx=i, conf=round(c, 4), x=x, y=y, w=w, h=h,
                         cx=x + w / 2, cy=y + h / 2))
    return rows


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def boxes_of(df, img):
    d = df[df.image == img]
    return [(r.x, r.y, r.w, r.h, r.conf) for r in d.itertuples()]


def match(preds, gts, thr=0.5):
    """Greedy match by confidence. Returns TP, FP, FN."""
    used, tp = set(), 0
    for p in sorted(preds, key=lambda b: -b[4]):
        best, bj = 0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def consensus(dfs, model, img, min_votes=2, conf=0.5):
    """Pseudo-truth for `model`: boxes found by >= min_votes OTHER models."""
    pool = [(m, b) for m, d in dfs.items() if m != model
            for b in boxes_of(d, img) if b[4] >= conf]
    clusters = []
    for m, b in sorted(pool, key=lambda t: -t[1][4]):
        for c in clusters:
            if iou(c["box"], b) >= 0.5:
                c["models"].add(m)
                break
        else:
            clusters.append({"box": b, "models": {m}})
    return [c["box"] for c in clusters if len(c["models"]) >= min_votes]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def draw_montage(images, dfs, out_dir, conf=0.5, cell=420, n=24):
    colors = [(0, 200, 0), (0, 0, 255), (255, 120, 0), (200, 0, 200), (0, 200, 200), (120, 120, 120)]
    for m, df in dfs.items():
        tiles = []
        for p in images[:n]:
            img = cv2.imread(str(p))
            for r in df[(df.image == p.name) & (df.conf >= conf)].itertuples():
                H, W = img.shape[:2]
                cv2.rectangle(img, (int(r.x * W), int(r.y * H)),
                              (int((r.x + r.w) * W), int((r.y + r.h) * H)),
                              colors[list(dfs).index(m) % 6], max(2, W // 300))
            tiles.append(cv2.resize(img, (cell, int(cell * 9 / 16))))
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(out_dir / f"overlay_{m}.jpg"), np.vstack(rows))


# ---------------------------------------------------------------- commands
def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in EXTS)
    if a.limit:
        images = images[:a.limit]
    print(f"{len(images)} images")

    dfs, timing = {}, {}
    for name in a.models:
        try:
            det = DETECTORS[name]()
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        rows, t0 = [], time.time()
        for p in tqdm(images, desc=name):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            try:
                rows += to_rows(p.name, det(img), W, H)
            except Exception as e:
                print(f"  {name} failed on {p.name}: {e}")
        timing[name] = (time.time() - t0) / len(images)
        dfs[name] = pd.DataFrame(rows, columns=["image", "face_idx", "conf", "x", "y", "w", "h", "cx", "cy"])
        dfs[name].to_csv(out / f"faces_{name}.csv", index=False)

    truth = pd.read_csv(a.truth) if a.truth else None
    names = [p.name for p in images]
    report = []
    for m, df in dfs.items():
        for thr in (0.3, 0.5, 0.7, 0.9):
            sub = df[df.conf >= thr]
            row = dict(model=m, conf_thr=thr, sec_per_img=round(timing[m], 3),
                       faces_per_img=round(len(sub) / len(images), 2),
                       imgs_with_face=round(sub.image.nunique() / len(images), 3),
                       median_face_h=round(sub.h.median(), 3) if len(sub) else None)
            for tag, gt_fn in (("truth", (lambda im, mm: boxes_of(truth, im)) if truth is not None else None),
                               ("consensus", lambda im, mm: consensus(dfs, mm, im))):
                if gt_fn is None or (tag == "consensus" and len(dfs) < 3):
                    continue
                tp = fp = fn = 0
                for im in names:
                    r = match(boxes_of(sub, im), gt_fn(im, m))
                    tp, fp, fn = tp + r[0], fp + r[1], fn + r[2]
                row[f"{tag}_P"], row[f"{tag}_R"], row[f"{tag}_F1"] = [round(v, 3) for v in prf(tp, fp, fn)]
            report.append(row)
    rep = pd.DataFrame(report)
    rep.to_csv(out / "comparison.csv", index=False)
    print(rep.to_string(index=False))
    draw_montage(images, dfs, out, conf=0.5)
    print(f"\nWrote per-model CSVs, comparison.csv and overlay_*.jpg to {out}/")
    print("Rank by truth_F1 if you provided --truth; otherwise consensus_F1 is only a rough proxy.")


def cmd_final(a):
    df = pd.read_csv(Path(a.out) / f"faces_{a.model}.csv")
    df = df[df.conf >= a.conf].copy()
    df["face_idx"] = df.groupby("image").cumcount()
    df.to_csv(Path(a.out) / "faces.csv", index=False)
    print(f"faces.csv: {len(df)} faces, {df.image.nunique()} images (model={a.model}, conf>={a.conf})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--images", required=True)
    r.add_argument("--out", default="face_results")
    r.add_argument("--models", nargs="+", default=["scrfd", "yunet", "yolo", "mtcnn", "mediapipe"],
                   choices=list(DETECTORS))
    r.add_argument("--truth")
    r.add_argument("--limit", type=int)
    r.set_defaults(fn=cmd_run)
    f = sp.add_parser("final")
    f.add_argument("--out", default="face_results")
    f.add_argument("--model", required=True)
    f.add_argument("--conf", type=float, default=0.5)
    f.set_defaults(fn=cmd_final)
    a = ap.parse_args()
    a.fn(a)
