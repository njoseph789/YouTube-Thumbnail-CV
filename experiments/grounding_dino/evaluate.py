"""Score the baseline runs against ground_truth.csv and pick the model to build on.

Primary metric: image-level mAP. For each keyword, every image gets the model's
highest score for that keyword (keyword_scores.csv); average precision is computed
over the images against ground_truth.csv, then averaged over keywords that have at
least one positive and one negative image.

Secondary: mAP over keywords with >=3 positives, per-category mAP, and precision /
recall / F1 at the run's default thresholds (from detections.csv).

Selection rule (decisions.md): highest mAP wins; if a bootstrap over images says the
difference is within noise (95% interval includes 0), tiny wins.

Usage (compviz venv), after run_baseline.py:
    python experiments/grounding_dino/evaluate.py
"""
import argparse
import csv
import sys
from pathlib import Path

import mlflow
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KEYWORDS  # noqa: E402
from tracking import setup_mlflow  # noqa: E402

KW_CAT = {kw: cat for cat, kws in KEYWORDS.items() for kw in kws}
ALL_KW = list(KW_CAT)
MODEL_ORDER = ["tiny", "base"]


def average_precision(y, s):
    """AP = mean of precision at each positive's rank (ties broken by order)."""
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def load_truth(path, images):
    idx = {im: i for i, im in enumerate(images)}
    y = np.zeros((len(images), len(ALL_KW)), dtype=int)
    kw_idx = {k: j for j, k in enumerate(ALL_KW)}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image"] in idx:
                y[idx[r["image"]], kw_idx[r["label"]]] = 1
    return y


def load_scores(path, images):
    idx = {im: i for i, im in enumerate(images)}
    kw_idx = {k: j for j, k in enumerate(ALL_KW)}
    s = np.zeros((len(images), len(ALL_KW)))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image"] in idx:
                s[idx[r["image"]], kw_idx[r["keyword"]]] = float(r["score"])
    return s


def load_detected(path, images):
    idx = {im: i for i, im in enumerate(images)}
    kw_idx = {k: j for j, k in enumerate(ALL_KW)}
    d = np.zeros((len(images), len(ALL_KW)), dtype=int)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image"] in idx:
                d[idx[r["image"]], kw_idx[r["label"]]] = 1
    return d


def per_keyword_ap(y, s, rows=None):
    """AP per keyword (nan where it has no positive or no negative image)."""
    if rows is not None:
        y, s = y[rows], s[rows]
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        n = y[:, j].sum()
        if 0 < n < len(y):
            out[j] = average_precision(y[:, j], s[:, j])
    return out


def prf(y, d):
    tp = (y * d).sum(0)
    fp = ((1 - y) * d).sum(0)
    fn = (y * (1 - d)).sum(0)
    def f(t, p, n):
        pr = t / (t + p) if t + p else 0.0
        rc = t / (t + n) if t + n else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    micro = f(tp.sum(), fp.sum(), fn.sum())
    has_pos = y.sum(0) > 0
    macro_f1 = float(np.mean([f(tp[j], fp[j], fn[j])[2] for j in np.flatnonzero(has_pos)]))
    return micro, macro_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--truth", type=Path, default=HERE / "ground_truth.csv")
    ap.add_argument("--experiment", default="grounding_dino_baseline")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    models = [m for m in MODEL_ORDER if (args.results / m / "keyword_scores.csv").exists()]
    if not models:
        sys.exit(f"no keyword_scores.csv found under {args.results}; run run_baseline.py first")

    # Evaluate on the images every model scored (smaller than 100 for smoke tests).
    sets = []
    for m in models:
        with open(args.results / m / "keyword_scores.csv", encoding="utf-8") as f:
            sets.append({r["image"] for r in csv.DictReader(f)})
    images = sorted(set.intersection(*sets))
    y = load_truth(args.truth, images)
    print(f"{len(images)} images, models: {models}")

    scores = {m: load_scores(args.results / m / "keyword_scores.csv", images) for m in models}
    det = {m: load_detected(args.results / m / "detections.csv", images) for m in models}

    npos = y.sum(0)
    report, metrics = {}, {}
    for m in models:
        ap_kw = per_keyword_ap(y, scores[m])
        valid = ~np.isnan(ap_kw)
        ge3 = valid & (npos >= 3)
        (pr, rc, f1), macro_f1 = prf(y, det[m])
        cat_map = {}
        for cat, kws in KEYWORDS.items():
            cols = [ALL_KW.index(k) for k in kws if valid[ALL_KW.index(k)]]
            if cols:
                cat_map[cat] = float(np.mean(ap_kw[cols]))
        metrics[m] = {
            "mAP": float(np.nanmean(ap_kw)) if valid.any() else float("nan"),
            "mAP_ge3_positives": float(np.mean(ap_kw[ge3])) if ge3.any() else float("nan"),
            "keywords_scored": int(valid.sum()),
            "micro_precision": pr, "micro_recall": rc, "micro_f1": f1, "macro_f1": macro_f1,
            **{f"mAP_{c}": v for c, v in cat_map.items()},
        }
        report[m] = ap_kw

    # Per-keyword AP table.
    ap_csv = args.results / "per_keyword_ap.csv"
    with open(ap_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "keyword", "n_positive_images"] + [f"ap_{m}" for m in models])
        for j, kw in enumerate(ALL_KW):
            w.writerow([KW_CAT[kw], kw, int(npos[j])] + ["" if np.isnan(report[m][j]) else f"{report[m][j]:.4f}" for m in models])

    # Bootstrap over images: is the tiny/base difference within noise?
    verdict, ci = None, None
    if set(models) >= {"tiny", "base"} and len(images) >= 10:
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(args.bootstrap):
            rows = rng.integers(0, len(images), len(images))
            a = per_keyword_ap(y, scores["base"], rows)
            b = per_keyword_ap(y, scores["tiny"], rows)
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.any():
                diffs.append(np.mean(a[ok]) - np.mean(b[ok]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        ci = (float(lo), float(hi))
        if lo > 0:
            verdict = "base"
        elif hi < 0:
            verdict = "tiny"
        else:
            verdict = "tiny (tie: difference within noise)"

    # Print.
    print(f"\n{'metric':<24}" + "".join(f"{m:>10}" for m in models))
    for k in ["mAP", "mAP_ge3_positives", "keywords_scored", "micro_precision", "micro_recall", "micro_f1", "macro_f1"]:
        print(f"{k:<24}" + "".join(f"{metrics[m][k]:>10.3f}" if k != "keywords_scored" else f"{metrics[m][k]:>10d}" for m in models))
    print("\nper-category mAP")
    for cat in KEYWORDS:
        vals = [metrics[m].get(f"mAP_{cat}") for m in models]
        print(f"  {cat:<22}" + "".join(f"{'-' if v is None else f'{v:.3f}':>10}" for v in vals))
    if ci:
        print(f"\nmAP(base) - mAP(tiny): 95% bootstrap interval [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"verdict: {verdict}")
    print(f"\nper-keyword AP -> {ap_csv}")

    # Log back to each model's MLflow run.
    if not args.no_mlflow:
        setup_mlflow(args.results, args.experiment)
        for m in models:
            rid_file = args.results / m / "run_id.txt"
            if not rid_file.exists():
                continue
            with mlflow.start_run(run_id=rid_file.read_text().strip()):
                mlflow.log_metrics({k: v for k, v in metrics[m].items() if not np.isnan(v)})
                mlflow.log_artifact(str(ap_csv))
                if verdict:
                    mlflow.set_tag("selection_verdict", verdict)
                    mlflow.set_tag("bootstrap_ci_base_minus_tiny", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print("logged to MLflow")


if __name__ == "__main__":
    main()
