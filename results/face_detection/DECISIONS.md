# Face detection: design decisions

Why `scripts/face_boxes.py` uses an ensemble of detectors plus a CLIP real-vs-cartoon filter.
All scores are against `images/face_labels.csv` (100 images in `images/labeled_subset`), at image level.

## Goal

Face bounding boxes (x, y, w, h normalized 0-1) for thumbnails, used as training features and for Alex's generation evaluation (face placement).
Box convention: tight face, brow/hairline to chin, ear to ear, no hair.

## 1. Why an ensemble

- **No single detector won.** Candidates: YuNet, YOLOv8-face, SCRFD, MTCNN. Best single-detector F1 on `has_real_face` (best cutoff each): MTCNN@0.9 0.953, SCRFD@0.7 0.952, YuNet@0.7 0.948, YOLO@0.7 0.935. Differences are about one image.
- **Their weaknesses differ.** MTCNN: best recall, more false positives at low cutoffs. YOLO and SCRFD: precise but miss real faces at the cutoffs that keep precision high. YuNet: most false positives.
- **Agreement is informative.** A box found by 2+ detectors is much more likely a real face. Vote count also lets us review uncertain boxes (1-2 votes) first.
- **Result.** Ensemble at 2+ votes beat the best single detector on `has_face` (F1 0.959 vs 0.940, YOLO@0.3).

## 2. Why six detectors (animeface, gdino added)

The original four are trained on human faces and missed stylized faces (anime, 3D characters, cartoons) that the labels count as `has_face=1`.

- Added `animeface` (YOLOv8 anime-face model, Fuyucchi/yolov8_animeface) and `gdino` (Grounding DINO tiny, prompt "a face. a cartoon face. an anime face.").
- `has_face` missed faces at 2+ votes: 7 -> 4 (found 67 -> 70 of 74), false positives 1 -> 2.
- `gdino` is too noisy alone (at 0.3: all 74 faces found but 25 false positives; at 0.5 only 24 of 74 found). Kept only as one vote, cutoff 0.5.

## 3. Why a CLIP real-vs-cartoon filter

Detectors cannot tell real from cartoon faces, so adding cartoon-capable detectors lowers precision against `has_real_face` (false positives 4 -> 8 at 2+ votes).

- Zero-shot CLIP (ViT-B/32) scores each box crop: real-face prompts vs cartoon/illustration prompts, `p_real`, `is_real = p_real >= 0.3`.
- A first version (one real prompt vs three cartoon prompts in a single softmax) rejected too many real faces (F1 0.785 at 0.5). Grouping the prompts fixed it.
- Final: `has_real_face` F1 0.962 (63 of 65 found, 3 false positives). Stable for cutoffs 0.2-0.6 (F1 0.955-0.962).
- `faces_final.csv` keeps all boxes and flags `is_real`, so the team can choose the "any face" or "real face only" definition by filtering.

## 4. Box quality

A manual review of 36 boxes (all boxes with fewer than 4 votes or marked non-real, plus a random sample) found them tight and consistent. Image-level labels do not measure box position, so this is a spot check, not a metric.

## Caveats

- Cutoffs (`ENS_THR` in the script, `--real-thr 0.3`) were tuned on the same 100 images. A held-out check on new images is still to do (`face_boxes.py heldout` builds the sample).
- On 100 images, one image is about 1.5 F1 points; gains between options are small.
- A bug where `faces_final.csv` was counted as a voter inflated an intermediate ensemble result; fixed (only per-detector CSVs vote). Results here are from after the fix.
- Label questions are in `LABEL_NOTES.md`.
- `animeface` weights are AGPL-3.0.
