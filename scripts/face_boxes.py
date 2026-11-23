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


class AnimeFace:  # YOLOv8 anime-face detector (AGPL-3.0), catches drawn/anime faces
    def __init__(self):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        self.m = YOLO(hf_hub_download("Fuyucchi/yolov8_animeface", "yolov8x6_animeface.pt"))

    def __call__(self, img):
        r = self.m.predict(img, conf=LOW_CONF, imgsz=1280, verbose=False)[0]
        return [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]


class GDino:  # Grounding DINO open-vocabulary, prompted for real + cartoon faces
    def __init__(self, name="IDEA-Research/grounding-dino-tiny"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(name)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(self.dev).eval()
        self.text = "a face. a cartoon face. an anime face."

    def __call__(self, img):
        import torch
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=pil, text=self.text, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.m(**inp)
        kw = dict(text_threshold=0.25, target_sizes=[pil.size[::-1]])
        try:  # argument name differs across transformers versions
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.25, **kw)[0]
        except TypeError:
            res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.25, **kw)[0]
        return [(*b.tolist(), float(s)) for b, s in zip(res["boxes"], res["scores"])]


DETECTORS = {"scrfd": SCRFD, "yunet": YuNet, "yolo": YoloFace,
             "mtcnn": MTCNNDet, "mediapipe": MediaPipe, "retinaface": RetinaFaceDet,
             "animeface": AnimeFace, "gdino": GDino}


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


def draw_montage(images, dfs, out_dir, rel, conf=0.5, cell=420, n=24):
    colors = [(0, 200, 0), (0, 0, 255), (255, 120, 0), (200, 0, 200), (0, 200, 200), (120, 120, 120)]
    for m, df in dfs.items():
        tiles = []
        for p in images[:n]:
            img = cv2.imread(str(p))
            for r in df[(df.image == rel(p)) & (df.conf >= conf)].itertuples():
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
    root = Path(a.images)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if a.list:  # restrict to the files named in a CSV (first column, e.g. face_labels.csv source_path)
        keep = {Path(s).name for s in pd.read_csv(a.list).iloc[:, 0]}
        images = [p for p in images if p.name in keep]
    if a.limit and len(images) > a.limit:
        # Spread the sample evenly across channel subfolders (reproducible).
        import random
        from collections import defaultdict
        random.seed(0)
        by_ch = defaultdict(list)
        for p in images:
            by_ch[p.parent].append(p)
        for v in by_ch.values():
            random.shuffle(v)
        picked, chans = [], list(by_ch)
        while len(picked) < a.limit and chans:
            for c in list(chans):
                if by_ch[c]:
                    picked.append(by_ch[c].pop())
                    if len(picked) == a.limit:
                        break
                else:
                    chans.remove(c)
        images = sorted(picked)
    print(f"{len(images)} images from {len({p.parent for p in images})} folders")
    rel = lambda p: p.relative_to(root).as_posix()  # unique key: channel/file.jpg

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
                rows += to_rows(rel(p), det(img), W, H)
            except Exception as e:
                print(f"  {name} failed on {p.name}: {e}")
        timing[name] = (time.time() - t0) / len(images)
        dfs[name] = pd.DataFrame(rows, columns=["image", "face_idx", "conf", "x", "y", "w", "h", "cx", "cy"])
        dfs[name].to_csv(out / f"faces_{name}.csv", index=False)

    truth = pd.read_csv(a.truth) if a.truth else None
    names = [rel(p) for p in images]
    (out / "images.txt").write_text("\n".join(names))  # lets `ensemble` sample no-face images too
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
    draw_montage(images, dfs, out, rel, conf=0.5)
    print(f"\nWrote per-model CSVs, comparison.csv and overlay_*.jpg to {out}/")
    print("Rank by truth_F1 if you provided --truth; otherwise consensus_F1 is only a rough proxy.")


def cmd_final(a):
    df = pd.read_csv(Path(a.out) / f"faces_{a.model}.csv")
    df = df[df.conf >= a.conf].copy()
    df["face_idx"] = df.groupby("image").cumcount()
    df.to_csv(Path(a.out) / "faces.csv", index=False)
    print(f"faces.csv: {len(df)} faces, {df.image.nunique()} images (model={a.model}, conf>={a.conf})")


# ------------------------------------------- ensemble + manual review -> truth
ENS_THR = {"yolo": 0.5, "mtcnn": 0.9, "yunet": 0.7, "scrfd": 0.7,
           "animeface": 0.5, "gdino": 0.35}  # last two are starting guesses; tune from image-level scores  # per-model cutoffs from comparison.csv


def load_models(out):
    # Only per-detector outputs; faces_final / faces_ensemble are derived files and must not vote.
    return {p.stem[len("faces_"):]: pd.read_csv(p) for p in Path(out).glob("faces_*.csv")
            if p.stem[len("faces_"):] in DETECTORS}


def fuse(dfs, img):
    """Cluster boxes across models (IoU>=0.5, one box per model per cluster).
    Returns [(x, y, w, h, mean_conf, [models])] with confidence-weighted box averages."""
    pool = [(m, b) for m, d in dfs.items() for b in boxes_of(d, img) if b[4] >= ENS_THR.get(m, 0.5)]
    clusters = []
    for m, b in sorted(pool, key=lambda t: -t[1][4]):
        for c in clusters:
            if m not in c["models"] and iou(c["box"], b) >= 0.5:
                c["members"].append(b)
                c["models"].add(m)
                break
        else:
            clusters.append({"box": b, "members": [b], "models": {m}})
    res = []
    for c in clusters:
        arr = np.array([m[:4] for m in c["members"]])
        wts = np.array([m[4] for m in c["members"]])
        x, y, w, h = (arr * wts[:, None]).sum(0) / wts.sum()
        res.append((x, y, w, h, float(wts.mean()), sorted(c["models"])))
    return res


def spread(names, n, seed=1):
    """Sample n names spread evenly across channel folders (first path component)."""
    import random
    from collections import defaultdict
    random.seed(seed)
    g = defaultdict(list)
    for s in names:
        g[s.split("/")[0].split("__")[0]].append(s)  # channel = folder, or prefix before "__" in flat layouts
    for v in g.values():
        random.shuffle(v)
    picked, keys = [], list(g)
    while len(picked) < n and keys:
        for k in list(keys):
            if g[k]:
                picked.append(g[k].pop())
            else:
                keys.remove(k)
            if len(picked) == n:
                break
    return sorted(picked)


def cmd_ensemble(a):
    root, out = Path(a.images), Path(a.out)
    dfs = load_models(out)
    lst = out / "images.txt"
    if lst.exists():
        names = lst.read_text().split("\n")
    else:
        names = sorted(set().union(*[set(d.image) for d in dfs.values()]))
        print("WARNING: images.txt missing (re-run `run` to include no-face images in the sample).")
    sample = spread(names, a.n)
    (out / "reviewed_images.txt").write_text("\n".join(sample))
    rev_dir = out / "review_imgs"
    rev_dir.mkdir(exist_ok=True)
    rows = []
    for i, name in enumerate(sample):
        img = cv2.imread(str(root / name))
        if img is None:
            print("cannot read", name)
            continue
        H, W = img.shape[:2]
        grid = img.copy()
        for t in range(1, 10):  # 10% grid so missed faces can be located by eye
            cv2.line(grid, (int(W * t / 10), 0), (int(W * t / 10), H), (220, 220, 220), 1)
            cv2.line(grid, (0, int(H * t / 10)), (W, int(H * t / 10)), (220, 220, 220), 1)
            cv2.putText(grid, f".{t}", (int(W * t / 10) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(grid, f".{t}", (2, int(H * t / 10) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vis = cv2.addWeighted(grid, 0.4, img, 0.6, 0)
        fs, th = max(0.6, W / 1000), max(2, W // 400)
        for j, (x, y, w, h, c, ms) in enumerate(sorted(fuse(dfs, name), key=lambda f: -f[4])):
            keep = int(len(ms) >= 2)
            rows.append(dict(image=name, review_file=f"{i:02d}.jpg", box_id=j, votes=len(ms),
                             models="+".join(ms), conf=round(c, 3), x=x, y=y, w=w, h=h, keep=keep))
            col = (0, 200, 0) if keep else (0, 140, 255)  # green = default keep, orange = default drop
            p1, p2 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
            cv2.rectangle(vis, p1, p2, col, th)
            cv2.putText(vis, f"#{j} v{len(ms)}", (p1[0], max(p1[1] - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
        cv2.putText(vis, f"{i:02d} | {name}", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 255, 255), 2)
        s = min(1.0, 1280 / W)
        cv2.imwrite(str(rev_dir / f"{i:02d}.jpg"), cv2.resize(vis, (int(W * s), int(H * s))))
    pd.DataFrame(rows).to_csv(out / "ensemble_review.csv", index=False)
    if not (out / "missed_faces.csv").exists():
        pd.DataFrame(columns=["image", "x", "y", "w", "h"]).to_csv(out / "missed_faces.csv", index=False)
    print(f"{len(sample)} images, {len(rows)} candidate boxes -> {out}/ensemble_review.csv, review_imgs/")
    print("Edit `keep` (1/0) in ensemble_review.csv; add missed faces to missed_faces.csv; then run `truth`.")


def cmd_truth(a):
    out = Path(a.out)
    rev = pd.read_csv(out / "ensemble_review.csv")
    cols = ["image", "x", "y", "w", "h"]
    parts = [rev[rev.keep == 1][cols]]
    mp = out / "missed_faces.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        if len(m):
            parts.append(m[cols])
    t = pd.concat(parts, ignore_index=True)
    t["conf"] = 1.0
    t["face_idx"] = t.groupby("image").cumcount()
    t["cx"], t["cy"] = t.x + t.w / 2, t.y + t.h / 2
    t.to_csv(out / "truth.csv", index=False)
    n = len((out / "reviewed_images.txt").read_text().split("\n"))
    print(f"truth.csv: {len(t)} faces across {t.image.nunique()} of {n} reviewed images")


def cmd_score(a):
    out = Path(a.out)
    dfs = load_models(out)
    truth = pd.read_csv(out / "truth.csv")
    names = (out / "reviewed_images.txt").read_text().split("\n")
    rows = []
    ens = [dict(image=n, conf=len(f[5]), x=f[0], y=f[1], w=f[2], h=f[3]) for n in names for f in fuse(dfs, n)]
    cands = [(m, d, ENS_THR.get(m, 0.5)) for m, d in dfs.items()]
    cands += [("ensemble_votes>=1", pd.DataFrame(ens), 1), ("ensemble_votes>=2", pd.DataFrame(ens), 2)]
    for m, d, base in cands:
        thrs = [0.3, 0.5, 0.7, 0.9] if not m.startswith("ensemble") else [base]
        for thr in thrs:
            sub = d[d.conf >= thr]
            tp = fp = fn = 0
            for n in names:
                r = match(boxes_of(sub, n), boxes_of(truth, n))
                tp, fp, fn = tp + r[0], fp + r[1], fn + r[2]
            P, R, F = prf(tp, fp, fn)
            rows.append(dict(model=m, conf_thr=thr, TP=tp, FP=fp, FN=fn, P=round(P, 3), R=round(R, 3), F1=round(F, 3)))
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "truth_scores.csv", index=False)
    print(rep.to_string(index=False))


def cmd_imgscore(a):
    """Image-level check vs face_labels.csv: does each model say 'has a face' like the labels do?"""
    out = Path(a.out)
    dfs = load_models(out)
    lab = pd.read_csv(a.labels)
    lab["image"] = lab.source_path.map(lambda s: Path(s).name)
    names = list(lab.image)
    cols = ["image", "conf", "x", "y", "w", "h"]
    ens = pd.DataFrame([dict(image=n, conf=len(f[5]), x=f[0], y=f[1], w=f[2], h=f[3])
                        for n in names for f in fuse(dfs, n)], columns=cols)
    cands = [(m, d, [0.3, 0.5, 0.7, 0.9]) for m, d in dfs.items()]
    cands += [("ensemble_votes>=1", ens, [1]), ("ensemble_votes>=2", ens, [2])]
    rows = []
    for target in ("has_face", "has_real_face"):
        y = lab.set_index("image")[target].astype(int)
        for m, d, thrs in cands:
            for thr in thrs:
                pred = set(d[d.conf >= thr].image)
                tp = sum(1 for n in names if y[n] == 1 and n in pred)
                fp = sum(1 for n in names if y[n] == 0 and n in pred)
                fn = sum(1 for n in names if y[n] == 1 and n not in pred)
                P, R, F = prf(tp, fp, fn)
                rows.append(dict(target=target, model=m, conf_thr=thr, TP=tp, FP=fp, FN=fn,
                                 P=round(P, 3), R=round(R, 3), F1=round(F, 3)))
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "image_level_scores.csv", index=False)
    print(rep.to_string(index=False))
    pred2 = set(ens[ens.conf >= 2].image)
    lab["ensemble_pred"] = lab.image.isin(pred2).astype(int)
    mis = lab[lab.ensemble_pred != lab.has_real_face][["image", "has_face", "has_real_face", "ensemble_pred"]]
    mis.to_csv(out / "image_level_mismatches.csv", index=False)
    print(f"\n{len(mis)} images where ensemble (votes>=2) disagrees with has_real_face -> image_level_mismatches.csv")


CLIP_REAL = ["a photograph of a real person's face",
             "a close-up photo of a human face",
             "a photo of a person"]
CLIP_FAKE = ["a cartoon of a face",
             "an illustration or drawing of a character",
             "a 3D render of a character's face",
             "an anime or mascot character"]
CLIP_PROMPTS = CLIP_REAL + CLIP_FAKE


def clip_real_scores(root, df, pad=0.15, bs=32):
    """Zero-shot CLIP: P(real photographed face) for each box in df (crops padded by `pad`)."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = "openai/clip-vit-base-patch32"
    model, proc = CLIPModel.from_pretrained(name).to(dev).eval(), CLIPProcessor.from_pretrained(name)
    crops, cache = [], {}
    for r in df.itertuples():
        if r.image not in cache:
            cache = {r.image: Image.open(root / r.image).convert("RGB")}
        im = cache[r.image]
        W, H = im.size
        x1, y1 = max(0, (r.x - pad * r.w) * W), max(0, (r.y - pad * r.h) * H)
        x2, y2 = min(W, (r.x + r.w * (1 + pad)) * W), min(H, (r.y + r.h * (1 + pad)) * H)
        crops.append(im.crop((int(x1), int(y1), max(int(x2), int(x1) + 2), max(int(y2), int(y1) + 2))))
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            inp = proc(text=CLIP_PROMPTS, images=crops[i:i + bs], return_tensors="pt", padding=True).to(dev)
            lg = model(**inp).logits_per_image  # (n_crops, n_prompts)
            nr = len(CLIP_REAL)
            # group score = logsumexp over the group's prompts minus log(group size), so group sizes don't bias it
            real = torch.logsumexp(lg[:, :nr], dim=1) - math.log(nr)
            fake = torch.logsumexp(lg[:, nr:], dim=1) - math.log(len(CLIP_FAKE))
            out += torch.sigmoid(real - fake).cpu().tolist()
    return out


def cmd_realface(a):
    """Ensemble boxes (>=min_votes) + CLIP real-vs-cartoon score -> faces_final.csv."""
    out, root = Path(a.out), Path(a.images)
    dfs = load_models(out)
    names = (out / "images.txt").read_text().split("\n")
    rows = []
    for n in names:
        for j, (x, y, w, h, c, ms) in enumerate(sorted(fuse(dfs, n), key=lambda f: -f[4])):
            if len(ms) >= a.min_votes:
                rows.append(dict(image=n, x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2,
                                 conf=round(c, 3), votes=len(ms), models="+".join(ms)))
    df = pd.DataFrame(rows, columns=["image", "x", "y", "w", "h", "cx", "cy", "conf", "votes", "models"])
    df["p_real"] = 1.0 if a.no_clip else (clip_real_scores(root, df) if len(df) else [])
    df["is_real"] = (df.p_real >= a.real_thr).astype(int)
    df["face_idx"] = df.groupby("image").cumcount()
    df.to_csv(out / "faces_final.csv", index=False)
    print(f"faces_final.csv: {len(df)} boxes, {df.image.nunique()} images, "
          f"{int(df.is_real.sum())} marked real at p_real>={a.real_thr}")
    if a.labels:
        lab = pd.read_csv(a.labels)
        lab["image"] = lab.source_path.map(lambda s: Path(s).name)
        y = lab.set_index("image")["has_real_face"].astype(int)
        print("\nimage-level check vs has_real_face (any box with p_real >= thr):")
        for thr in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pred = set(df[df.p_real >= thr].image)
            tp = sum(1 for n in y.index if y[n] == 1 and n in pred)
            fp = sum(1 for n in y.index if y[n] == 0 and n in pred)
            fn = sum(1 for n in y.index if y[n] == 1 and n not in pred)
            P, R, F = prf(tp, fp, fn)
            print(f"  thr={thr:.1f}  TP={tp} FP={fp} FN={fn}  P={P:.3f} R={R:.3f} F1={F:.3f}")
        pred = set(df[df.is_real == 1].image)
        lab["pred_real"] = lab.image.isin(pred).astype(int)
        mis = lab[lab.pred_real != lab.has_real_face][["image", "has_face", "has_real_face", "pred_real"]]
        mis.to_csv(out / "final_mismatches.csv", index=False)
        print(f"\n{len(mis)} mismatches at thr={a.real_thr} -> final_mismatches.csv")


def cmd_boxcheck(a):
    """Contact sheets of final boxes for manual tightness review; --summarize tallies box_review.csv."""
    out, root = Path(a.out), Path(a.images)
    if a.summarize:
        r = pd.read_csv(out / "box_review.csv")
        r = r[r.ok.notna()].copy()
        r["ok"] = r.ok.astype(int)
        print(f"{len(r)} boxes reviewed, {r.ok.mean():.1%} good\n")
        print(r.groupby("votes").ok.agg(["count", "mean"]).rename(columns={"mean": "good_rate"}).round(3))
        print(r.groupby("is_real").ok.agg(["count", "mean"]).rename(columns={"mean": "good_rate"}).round(3))
        bad = r[r.ok == 0]
        if len(bad):
            print("\nissues:", bad.issue.fillna("unspecified").value_counts().to_dict())
            print(bad[["box_id", "image", "votes", "p_real", "issue"]].to_string(index=False))
        return
    df = pd.read_csv(out / "faces_final.csv")
    pri = df[(df.votes < 4) | (df.is_real == 0)]
    rest = df.drop(pri.index).sample(frac=1, random_state=1)
    pick = pd.concat([pri, rest]).head(a.n).sort_values(["image", "face_idx"]).reset_index(drop=True)
    pick["box_id"] = range(len(pick))
    tiles = []
    for r in pick.itertuples():
        img = cv2.imread(str(root / r.image))
        H, W = img.shape[:2]
        th = max(2, W // 250)
        for o in df[df.image == r.image].itertuples():  # context: other boxes thin gray
            cv2.rectangle(img, (int(o.x * W), int(o.y * H)), (int((o.x + o.w) * W), int((o.y + o.h) * H)), (160, 160, 160), 1)
        col = (0, 200, 0) if r.is_real else (0, 140, 255)
        cv2.rectangle(img, (int(r.x * W), int(r.y * H)), (int((r.x + r.w) * W), int((r.y + r.h) * H)), col, th)
        tile = cv2.resize(img, (480, 270))
        cv2.putText(tile, f"#{r.box_id} v{r.votes} p{r.p_real:.2f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    per, cols = 12, 3
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)])
        cv2.imwrite(str(out / f"boxcheck_{s // per + 1:02d}.jpg"), sheet)
    rev = pick[["box_id", "image", "face_idx", "votes", "models", "p_real", "is_real"]].copy()
    rev["ok"], rev["issue"] = "", ""
    rev.to_csv(out / "box_review.csv", index=False)
    print(f"{len(pick)} boxes -> {out}/boxcheck_*.jpg and box_review.csv")
    print("Fill ok (1 = tight/good, 0 = bad) and issue (loose/tight/offset/notface) in box_review.csv, then run with --summarize.")


def cmd_facecheck(a):
    """Contact sheet of images where the ensemble (>=min_votes) disagrees with has_face / has_real_face."""
    out, root = Path(a.out), Path(a.images)
    dfs = load_models(out)
    lab = pd.read_csv(a.labels)
    lab["image"] = lab.source_path.map(lambda s: Path(s).name)
    target = a.target
    rows, tiles = [], []
    for r in lab.itertuples():
        boxes = sorted(fuse(dfs, r.image), key=lambda f: -f[4])
        pred = int(any(len(f[5]) >= a.min_votes for f in boxes))
        truth = int(getattr(r, target))
        if pred == truth:
            continue
        kind = "FN (missed)" if truth == 1 else "FP (extra)"
        rows.append(dict(image=r.image, kind=kind, has_face=r.has_face, has_real_face=r.has_real_face,
                         n_boxes=len(boxes), max_votes=max([len(f[5]) for f in boxes], default=0),
                         models_seen="+".join(sorted({m for f in boxes for m in f[5]})), verdict="", note=""))
        img = cv2.imread(str(root / r.image))
        H, W = img.shape[:2]
        th = max(2, W // 250)
        for x, y, w, h, c, ms in boxes:
            col = (0, 200, 0) if len(ms) >= a.min_votes else (0, 140, 255)  # green = ensemble, orange = low-vote
            cv2.rectangle(img, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), col, th)
            cv2.putText(img, f"v{len(ms)}", (int(x * W), max(int(y * H) - 4, 16)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1200), col, 2)
        tile = cv2.resize(img, (480, 270))
        cv2.putText(tile, f"#{len(rows) - 1} {kind}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(tile, f"face={r.has_face} real={r.has_real_face}", (6, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
    pd.DataFrame(rows).to_csv(out / "facecheck.csv", index=False)
    for s in range(0, len(tiles), 12):
        chunk = tiles[s:s + 12]
        while len(chunk) % 3:
            chunk.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"facecheck_{s // 12 + 1:02d}.jpg"),
                    np.vstack([np.hstack(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]))
    print(f"{len(rows)} disagreements on {target} (ensemble votes>={a.min_votes}) -> facecheck.csv, facecheck_*.jpg")
    for r in rows:
        print(f"  {r['kind']:12s} {r['image']}  max_votes={r['max_votes']} models={r['models_seen']}")


def cmd_heldout(a):
    """Draw n fresh images (not in the labeled subset), copy them flat as channel__id.ext,
    and write numbered label sheets + a blank face_labels.csv to fill in BEFORE looking at model output."""
    import shutil
    root, out = Path(a.images), Path(a.out)
    seen = {Path(s).stem for s in pd.read_csv(a.exclude).iloc[:, 0]}
    cands = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTS and f"{p.parent.name}__{p.stem}" not in seen:
            cands[p.relative_to(root).as_posix()] = p
    print(f"{len(cands)} candidate images not in the labeled subset")
    pick = spread(list(cands), a.n, seed=a.seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for i, rel_name in enumerate(pick):
        p = cands[rel_name]
        flat = f"{p.parent.name}__{p.stem}{p.suffix.lower()}"
        shutil.copy(p, out / "images" / flat)
        rows.append(dict(source_path=f"images/heldout/{flat}", has_face="", has_real_face=""))
        img = cv2.imread(str(p))
        tile = cv2.resize(img, (480, 270))
        cv2.putText(tile, f"#{i}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        tiles.append(tile)
    pd.DataFrame(rows).to_csv(out / "face_labels.csv", index=False)
    for s in range(0, len(tiles), 12):
        chunk = tiles[s:s + 12]
        while len(chunk) % 3:
            chunk.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"label_sheet_{s // 12 + 1:02d}.jpg"),
                    np.vstack([np.hstack(chunk[i:i + 3]) for i in range(0, len(chunk), 3)]))
    print(f"{len(pick)} images -> {out}/images, face_labels.csv (blank), label_sheet_*.jpg")
    print("Fill has_face / has_real_face (1/0) for every row; row order = tile numbers (#0 = first row).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    ho = sp.add_parser("heldout")
    ho.add_argument("--images", required=True, help="root folder of all thumbnails (channel subfolders)")
    ho.add_argument("--exclude", required=True, help="face_labels.csv of the already-used subset")
    ho.add_argument("--out", default="heldout")
    ho.add_argument("--n", type=int, default=50)
    ho.add_argument("--seed", type=int, default=2)
    ho.set_defaults(fn=cmd_heldout)
    fc = sp.add_parser("facecheck")
    fc.add_argument("--images", required=True)
    fc.add_argument("--labels", required=True)
    fc.add_argument("--out", default="face_results")
    fc.add_argument("--target", default="has_face", choices=["has_face", "has_real_face"])
    fc.add_argument("--min-votes", type=int, default=2)
    fc.set_defaults(fn=cmd_facecheck)
    bc = sp.add_parser("boxcheck")
    bc.add_argument("--images", default=".")
    bc.add_argument("--out", default="face_results")
    bc.add_argument("--n", type=int, default=36)
    bc.add_argument("--summarize", action="store_true")
    bc.set_defaults(fn=cmd_boxcheck)
    rf = sp.add_parser("realface")
    rf.add_argument("--images", required=True)
    rf.add_argument("--out", default="face_results")
    rf.add_argument("--labels", help="face_labels.csv, to check the real/non-real filter")
    rf.add_argument("--min-votes", type=int, default=2)
    rf.add_argument("--real-thr", type=float, default=0.5)
    rf.add_argument("--no-clip", action="store_true", help="skip CLIP (for testing)")
    rf.set_defaults(fn=cmd_realface)
    i = sp.add_parser("imgscore")
    i.add_argument("--labels", required=True, help="path to face_labels.csv")
    i.add_argument("--out", default="face_results")
    i.set_defaults(fn=cmd_imgscore)
    e = sp.add_parser("ensemble")
    e.add_argument("--images", required=True)
    e.add_argument("--out", default="face_results")
    e.add_argument("--n", type=int, default=40)
    e.set_defaults(fn=cmd_ensemble)
    t = sp.add_parser("truth")
    t.add_argument("--out", default="face_results")
    t.set_defaults(fn=cmd_truth)
    s = sp.add_parser("score")
    s.add_argument("--out", default="face_results")
    s.set_defaults(fn=cmd_score)
    r = sp.add_parser("run")
    r.add_argument("--images", required=True)
    r.add_argument("--out", default="face_results")
    r.add_argument("--models", nargs="+", default=["scrfd", "yunet", "yolo", "mtcnn"],
                   choices=list(DETECTORS))
    r.add_argument("--truth")
    r.add_argument("--limit", type=int)
    r.add_argument("--list", help="CSV whose first column lists the images to use (e.g. face_labels.csv)")
    r.set_defaults(fn=cmd_run)
    f = sp.add_parser("final")
    f.add_argument("--out", default="face_results")
    f.add_argument("--model", required=True)
    f.add_argument("--conf", type=float, default=0.5)
    f.set_defaults(fn=cmd_final)
    a = ap.parse_args()
    a.fn(a)
