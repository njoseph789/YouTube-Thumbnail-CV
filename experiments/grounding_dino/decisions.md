# Grounding DINO decisions log

> Local tracking only. Do NOT commit this file.

## Environment
Run everything in the `compviz` venv: `C:\Users\njoseph5\venvs\compviz` (Python 3.14.8, also registered as the Jupyter kernel "Python (compviz)").

Checked 2026-11-17: has `torch` 2.14.1, `transformers` 5.19.0 (`AutoModelForZeroShotObjectDetection` imports fine), `mlflow` 3.17.1, `pandas` 3.0.6, `scikit-learn` 1.9.1, `pillow`, `supervision`. Nothing needs installing.
- The venv originally had `torch 2.14.1+cpu` (CPU-only build), so `torch.cuda.is_available()` was `False` even though the MX250 and driver were fine.
- **Fixed 2026-11-17:** the venv now has a matched CUDA stack from the PyTorch cu126 wheel index: `torch 2.11.0+cu126`, `torchvision 0.26.0+cu126`, `torchaudio 2.11.0+cu126`. cu126 still includes `sm_61`, the MX250's architecture.
- **Why 2.11 and not 2.14:** no `torchaudio` newer than 2.11.1 exists on PyPI or the PyTorch index, so a matched trio with torch 2.14.1 is impossible. I first installed `torch 2.14.1+cu126` (GPU worked), then moved the whole stack to 2.11 so torch, torchvision and torchaudio all match. `setuptools` was also pinned to `<82` (torch 2.11 requires it; the venv had 84.0.0).
- Verified: `cuda.is_available()` is `True`, device `NVIDIA GeForce MX250` (capability 6.1, 4.29 GB), fp32 and fp16 matmul run on the GPU, torchvision CUDA NMS works, a torchaudio op runs, `pip check` reports no broken requirements, and `transformers`/`accelerate`/`timm`/`supervision`/`mlflow` import fine.
- Rollback to the original CPU stack if ever needed: `pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0` from PyPI, and `setuptools` can go back up.
- Grounding DINO weights are not in the local Hugging Face cache; they download on first use. Not yet tested: loading a Grounding DINO model on the GPU (4 GB may be tight for base; the team GPUs remain the fallback).

## 1. Baseline Grounding DINO model
**Status:** decided

**Decision:** Run both `IDEA-Research/grounding-dino-tiny` (Swin-T) and `IDEA-Research/grounding-dino-base` (Swin-B) via Hugging Face `transformers`, on the 100-image labeled subset.

**Why:**
- They are the original authors' checkpoints converted to HF format, not reduced test versions, so there is no larger open model to graduate to.
- `transformers` is already in `requirements.txt`, and HF avoids the original repo's compiled CUDA ops (hard on Windows).
- 100 images is cheap on either model. (Earlier note that the laptop's MX250 4 GB GPU rules out larger models was wrong as a constraint: the team has access to proper GPUs for fine-tuning, so hardware is not a limiting factor. The tiny + base choice still stands as the baseline; larger open checkpoints could be added as candidates, see point 3.)
- Having both sizes gives the comparison needed for point 3.

**Ruled out:** Grounding DINO 1.5/1.6 (API only, cannot be fine-tuned). Original repo and MM-Grounding-DINO deferred to point 3 (fine-tuning tooling).

## 2. How to record results
**Status:** decided

**Decided so far:** Use MLflow so both models are compared directly in the same experiment (one experiment, one run per model). Run-level params and summary metrics go to MLflow.

**Decided:** Per-detection data is a long CSV (one row per detection), also logged to MLflow as an artifact. COCO export deferred to point 3 if fine-tuning tooling needs it.

**Why:** Matches repo CSV conventions (`labels.csv`, `face_labels.csv`), loads directly into pandas, joins to the feature table, and makes model comparison a simple groupby.

**Confirmed:**
- Columns: `image`, `model`, `prompt_category`, `label`, `score`, `x1`, `y1`, `x2`, `y2` (pixels), `x1_norm`, `y1_norm`, `x2_norm`, `y2_norm`, `img_w`, `img_h`.
- Keep raw detections; no merging across category prompts (dedupe at analysis time).
- Outputs in `experiments/grounding_dino/results/` (already gitignored via `experiments/*/results/`).
- Point MLflow tracking at a folder under `results/` (or add `mlruns/` to `.gitignore`) so run data is not committed.

**Ground truth (added to point 2):** Because no one will check detections by hand, Claude built an image-level ground truth by reviewing all 100 thumbnails against the `prompts.py` keywords. Presence only (no boxes). Script: `build_ground_truth.py` -> `ground_truth.csv` (`image, category, label`; 598 rows, 138 of 153 keywords appear at least once).

**Spot-check (24 of 100 images re-reviewed at 800px):** 16 images needed fixes: 26 labels added, 2 removed (a doubtful `sword` and `helmet` on ElectroBOOM `Se1NPlfUCss`). The ground truth is now 622 rows, 139/153 keywords used. Almost every error was a missed label rather than a wrong one: mainly `hand`, `hair`, and small background items (`tree`, `smartphone`/`phone`, `glasses`, `fence`, `flames`/`fire`, `toolbox`, `crowd of people`). The other 76 images were not re-checked and likely have the same under-labeling.

**Second pass (remaining 76 images, same 800px review):** 38 more labels added, none removed. The ground truth is now 660 rows, 140/153 keywords used, and every image has been reviewed at least twice. Again the misses were generic or small items (`hair`, `hand`, `beard`, `tree`, `head`, `microphone`, `headband`, `flames`/`fire`, `river`, `wires`, `glasses`, `basketball court`), not wrong labels.

**Caveats (counts below are from before the spot-check and second pass):**
- Single manual pass by Claude, so not independently verified; some common keywords (`hair`, `head`, `hand`) may be under-labeled, which understates precision.
- Image-level only: measures whether a keyword was found in the image, not whether the box is correct. Localization is not evaluated.
- Keywords came from the same review that produced the ground truth, so there is circularity; objects outside the vocabulary are not covered.
- 13 keywords have no positives in this set after both passes (`border outline`, `bowl`, `candy`, `circuit board`, `confetti`, `exclamation mark`, `hard hat`, `magnet`, `sparks`, `speech bubble`, `sword`, `video game console`, `welding mask`), so recall cannot be measured for them; any detections count as false positives.
- Image `hacksmith__bwtAmueLBS0` (text only) has no labels, so it contributes no rows to the CSV.

**Implementation notes (built 2026-11-17):**
- Scripts in `experiments/grounding_dino/`: `run_baseline.py` (run models, write CSVs, log to MLflow), `evaluate.py` (mAP, F1, bootstrap verdict, log back to MLflow), `tracking.py` (shared MLflow setup).
- MLflow 3.17 refuses the plain-folder backend, so tracking uses SQLite: `results/mlflow.db` with artifacts in `results/artifacts/` (replaces the earlier "folder under results/" idea; still gitignored via `experiments/*/results/`).
- Added `keyword_scores.csv` (image, model, category, keyword, score) next to `detections.csv`: a threshold-free score for every keyword in every image, taken as the max over the phrase's tokens and over all queries. mAP is computed from it.
- No `raw_label` column: phrases are mapped to keywords by token span, so labels are always exact keywords.
- Verified: smoke test on 3 images with tiny runs end to end on the GPU (2.22 GB peak); AP implementation matches scikit-learn on 200 random cases; the full evaluate and bootstrap path was exercised on synthetic score files (a less noisy model beat a noisier one, as it should). Not yet run on all 100 images with real models.
- Speed: about 36.8 s/image for tiny on the MX250 (12 prompts per image), so roughly 1 hour for 100 images; base will be slower. A team GPU is the better place for the full run.

## 3. How to choose the model to build on for transfer learning
**Status:** decided (for baseline-comparison scope; transfer-learning details deferred)

**(a) Primary metric: decided.** Image-level mAP against `ground_truth.csv`. For each image and keyword, the prediction score is the highest-confidence box for that keyword (0 if none). Average precision is computed per keyword over the 100 images, then averaged over keywords that have at least one positive.
- Why: threshold-free, so tiny and base are not penalized for calibrating confidence differently; rare keywords count equally.
- Secondary: F1 at the default thresholds (`BOX_THRESHOLD`/`TEXT_THRESHOLD`) and per-category breakdowns.
- Caveats: the 13 keywords with no positives are skipped; keywords with only 1-2 positives give noisy AP; box placement is not evaluated; the ground truth is still approximate, so only the tiny-vs-base comparison is meaningful, not the absolute values.

**Constraint correction:** GPUs are available to the team, so the laptop hardware (MX250, 4 GB) is not a constraint to build around for fine-tuning or inference. Model size matters only for per-image inference cost on the full dataset.

**(b) Selection rule: decided.** Highest image-level mAP wins between tiny and base. If the two are within noise of each other, pick tiny (faster inference over the full dataset). Compute is not a constraint, so size and speed are only a minor tiebreaker.

**Open items (tracked):**
- **Larger candidate comparison (deferred):** compare against a larger MM-Grounding-DINO checkpoint (e.g. Swin-L) later, after the tiny vs base baseline. Needs the mmdet install and a second model family; it also brings full training tooling.
- **(c) Fine-tuning tooling (deferred):** which framework to use (HF transformers, Open-GroundingDino, or MM-Grounding-DINO) and where to run it.
- **(d) Fine-tuning box labels (deferred):** where they come from, since the current ground truth is image-level only (one idea: baseline boxes filtered by ground-truth presence as pseudo-labels).

**Scope:** For now the work is only the baseline comparison (tiny vs base). Transfer-learning details (c, d) are out of scope until the baseline result is in.

**Status update:** Point 3 is decided for the current scope. Planning for the baseline comparison is complete.
