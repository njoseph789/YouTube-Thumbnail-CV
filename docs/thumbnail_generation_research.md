# Thumbnail Generation Conditioned on Success Features: Initial Research

Author: Alex Bacro. Draft of 2026-10-31, for team discussion.

## Contents

1. [Where the repo stands](#1-where-the-repo-stands)
2. [Defining "success" with the data we have](#2-defining-success-with-the-data-we-have)
3. [From success features to generation conditions](#3-from-success-features-to-generation-conditions)
4. [Candidate approaches](#4-candidate-approaches)
5. [Recommended champion and challenger](#5-recommended-champion-and-challenger)
6. [Evaluation](#6-evaluation)
7. [Minimal plan and milestones](#7-minimal-plan-and-milestones)
8. [Open questions for the team](#8-open-questions-for-the-team)
9. [Ethics, copyright and ToS](#9-ethics-copyright-and-tos)
10. [References](#10-references)

## 1. Where the repo stands

As of commit `6769fce` the repo holds only the data pipeline:

- `scraper.py` reads creators from the shared Creator Tracker sheet. For each channel it pulls all non-Shorts uploads (the `UULF` playlist), saves the best available thumbnail (`maxres` is 1280x720) to a Drive folder per creator as `<video_id>.jpg`, and writes metadata to BigQuery dataset `youtube-scraper-512714.youtube_thumbnails`.
- BigQuery tables: `videos` (title, description, `published_at`, `category_id`, `duration_seconds`, `definition`, `caption`, `licensed_content`, `made_for_kids`, `view_count`, `like_count`, `comment_count`, `has_paid_product_placement`, `drive_file_id`, `thumbnail_size`, `scraped_at`, ...), plus `video_tags`, `video_topics` (Wikipedia topic categories), `video_thumbnails`, `video_localizations`, `video_content_ratings` and `video_region_restrictions`.
- There are no image features, EDA notebooks, models or feature definitions yet. So the visual "success features" below are proposals that need to match whatever the analysis side builds.

Consequences for generation:

- **There is no CTR.** The public Data API exposes no impressions or click-through rate; only the channel owner gets those, through YouTube Analytics. Any "CTR predictor" is really a views predictor.
- **Thumbnails can change after upload.** The scraped image is the current thumbnail, which may not be the one that earned the views.
- **Snapshots are single.** `view_count` is captured once at `scraped_at`, so the video's age has to be controlled for.

## 2. Defining "success" with the data we have

"Creators with consistent audiences" suggests normalizing within each channel, so that the label reflects the thumbnail rather than the channel's size:

```
age_days   = scraped_at - published_at
perf       = log1p(view_count) - median over the channel's videos of similar age of log1p(view_count)
success    = perf in the channel's top tercile (or perf as a regression target)
```

Exclude videos younger than about 14 days, since their view counts haven't settled. Also exclude livestreams (`live_broadcast_content`) and very short or very long `duration_seconds` outliers. `like_count / view_count` and `comment_count / view_count` can serve as secondary engagement targets. This label serves two purposes: it is the target of the team's predictor (the other cognitive problem), and it is the "critic" for generation (section 6).

## 3. From success features to generation conditions

Each feature the analysis side is likely to compute maps to one conditioning channel. Proposed features and how they would drive generation:

| Success feature (proposed, analysis side) | How it is measured | How it conditions generation |
|---|---|---|
| Face present, count, size, position | Face detector (e.g. MediaPipe, RetinaFace) | Pose or face-box map for ControlNet/T2I-Adapter; prompt tokens such as "one face, close-up, left third" |
| Facial emotion (surprise, joy, ...) | FER classifier on face crops | Prompt ("shocked expression, open mouth"); OpenPose face landmarks for ControlNet |
| Creator identity | Channel ID; CLIP or face embedding of past thumbnails | IP-Adapter image prompt built from the creator's top thumbnails; per-creator LoRA |
| Text overlay present, word count, area, position | OCR (EasyOCR or PaddleOCR) | Text-box layout mask; text itself rendered with PIL after generation (section 4.5) |
| Color: dominant palette, saturation, brightness, contrast | k-means in Lab space, HSV stats | Prompt words ("high saturation, yellow and red"); low-res color-block map for T2I-Adapter "color"; palette loss for reranking |
| Composition: saliency center, rule of thirds, clutter | Saliency model, edge density | Saliency or segmentation map for ControlNet; layout boxes (GLIGEN-style) |
| Objects and scene | YOLO classes, CLIP zero-shot tags | Prompt nouns; object boxes |
| Topic and title semantics | `title`, `video_tags`, `video_topics`, `category_id` | Main text prompt; title-to-overlay-text candidates |

Write one "condition vector" per training thumbnail:
`{creator_id, topic, caption, face_boxes, emotion, text_boxes, palette, saliency_map, success}`. At training time it is computed from the real image (BLIP-2/Florence-2 captions plus the detectors above). At inference time it is set from what the analysis says works for that creator, for example "the top-tercile median: one large face on the right, 3 to 5 words top-left, high saturation". The generator learns "condition to image"; the analysis supplies the target condition. This keeps the two halves of the project decoupled.

To condition directly on success, add a token such as `<high_perf>` or `<low_perf>` to captions during LoRA training (classifier-free guidance can then push toward it). This works with the token-free approaches too.

## 4. Candidate approaches

Compute notes assume an Apple M5 Pro with 48 GB unified memory on PyTorch MPS, compared with a Colab or cloud NVIDIA GPU (L4 with 24 GB, A100 with 40 or 80 GB). MPS inference with diffusers works. MPS training works but is slow, and community reports say fp16 can give NaN losses there, so fp32 is safer ([diffusers discussion #13170](https://github.com/huggingface/diffusers/discussions/13170)). Kohya and ai-toolkit are mostly CUDA-first, and a FLUX.2 [dev] LoRA run on MPS failed to converge ([ai-toolkit #871](https://github.com/ostris/ai-toolkit/issues/871)). Plan to train on Colab and run inference and demos on the Mac.

### 4.1 Latent diffusion with LoRA fine-tuning

| Base model | Params | License (verified Oct 2026) | Notes |
|---|---|---|---|
| SD 1.5 ([HF, community mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)) | 0.9B | CreativeML OpenRAIL-M | Runway's original repo was removed in 2024. The cheapest option and trainable on MPS at 512 to 768 px. Biggest ControlNet ecosystem; weak image quality and text. |
| SDXL 1.0 ([HF](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)) | 3.5B | CreativeML OpenRAIL++-M (permissive with use restrictions) | Mature LoRA, ControlNet, T2I-Adapter and IP-Adapter support. Native 16:9 bucket of 1344x768. Text rendering is poor. |
| SD 3.5 Medium/Large ([license](https://stability.ai/license)) | 2.5B / 8B | Stability Community License: free under $1M annual revenue, research always free | Better prompt following and text than SDXL; smaller adapter ecosystem. |
| FLUX.2 [klein] 4B ([HF](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)) | 4B | Apache 2.0 | The newest fully open BFL model, about 13 GB VRAM, fast, with good text. The 9B klein and the 32B FLUX.2 [dev] are under the FLUX Non-Commercial License ([BFL licensing](https://bfl.ai/licensing)). |
| Qwen-Image ([HF](https://huggingface.co/Qwen/Qwen-Image)) | 20B | Apache 2.0 (the newer Qwen-Image-2.1 is reportedly non-commercial, unverified) | The best open model for text rendering, but too big to train comfortably; cloud only. |

- **Data:** a style LoRA needs a few hundred to a few thousand captioned thumbnails. We will have tens of thousands across creators. Use the top-tercile thumbnails per creator, or all of them with the success token.
- **Compute:** an SDXL LoRA (rank 16 to 32, 1024 px, cached latents) takes about 2 to 4 hours on an A100 for around 10k steps. On MPS it would be many times slower. An SD 1.5 LoRA is feasible on the Mac overnight, but the Mac is not available now. All models above run inference on 48 GB.
- **Pros:** highest quality; LoRA is the standard, documented path ([diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora)); combines with every adapter below.
- **Cons:** the code is library-heavy (little custom-code bonus); text overlays are unreliable; it can memorize and reproduce real thumbnails and faces.

### 4.2 Spatial control: ControlNet and T2I-Adapter

- **ControlNet** ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543); [diffusers docs](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)) adds a trainable copy of the encoder conditioned on a map (canny, depth, pose, segmentation). For SDXL, a "union" ControlNet such as `xinsir/controlnet-union-sdxl-1.0` covers several map types in one model.
- **T2I-Adapter** ([Mou et al. 2023](https://arxiv.org/abs/2302.08453); [diffusers docs](https://huggingface.co/docs/diffusers/using-diffusers/t2i_adapter)) is a lighter (about 77M parameter) side network. It includes a "color" adapter that suits palette conditioning.
- **Our use:** render the condition vector into maps: face and pose boxes, text-box masks (kept empty for later overlay), a saliency blob and a coarse color grid. Off-the-shelf pose or segmentation adapters work zero-shot. A custom "thumbnail layout" adapter trained on our maps is a good custom-code contribution, and needs a cloud GPU for a few hours.
- **License:** the adapter weights are mostly Apache 2.0 or OpenRAIL, so check each model card.

### 4.3 Identity and style: IP-Adapter

- **IP-Adapter** ([Ye et al. 2023](https://arxiv.org/abs/2308.06721); [diffusers docs](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)) conditions on CLIP image embeddings of reference images. Feeding it a creator's top thumbnails gives their style without per-creator training. Weights: [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter), Apache 2.0.
- **Face-identity variants** (IP-Adapter-FaceID, InstantID, PuLID) depend on InsightFace models (e.g. antelopev2), which are **licensed for non-commercial research only** ([InsightFace](https://www.insightface.ai/); [InstantID](https://github.com/instantx-research/instantid)). That is acceptable for a class project, but it raises a likeness concern (section 9). Recommendation: use style-only IP-Adapter, and generate generic faces rather than real creators.

### 4.4 Conditional GAN or VAE baseline (custom code)

- **Conditional DCGAN** ([Mirza & Osindero 2014](https://arxiv.org/abs/1411.1784)) or **CVAE** ([Sohn et al. 2015](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)), written from scratch in PyTorch at 128x72 or 256x144. The condition vector is concatenated into the generator and the projection discriminator. StyleGAN2-ADA ([Karras et al. 2020](https://arxiv.org/abs/2006.06676)) handles small datasets better, but its NVIDIA Source Code License is non-commercial and it is far less custom.
- **Data:** about 10k or more images; works at low resolution.
- **Compute:** small, at about 1 to 3 hours on a Colab T4/L4. It would also train on MPS when the GPU is free.
- **Pros:** earns the custom-code bonus; every line is explainable in the presentation; condition adherence is directly trainable (auxiliary classifier loss on palette and face count); no foundation-model licensing.
- **Cons:** low fidelity, mode collapse risk (GAN), blurriness (VAE). It will lose on FID, which is expected for a baseline.

### 4.5 Retrieval plus edit

- **Method:** embed all thumbnails with CLIP and retrieve the creator's top-performing thumbnail nearest to the new video's title or topic. Then edit it: SDXL img2img or inpainting to swap the subject or background, a palette shift, and new overlay text rendered with PIL in the creator's font style. Instruction editors such as FLUX.2 klein's editing mode or Qwen-Image-Edit can do this in one step.
- **Data:** no training at all.
- **Compute:** inference only; fine on the Mac.
- **Pros:** strong on-brand results, cheap, and makes a strong baseline.
- **Cons:** closest to copying the creator's work; low novelty; FID is artificially good because outputs sit near real images.

### 4.6 Layout-then-render pipeline

- **Stage 1:** generate a layout (face box, text boxes, object boxes, palette) from the condition vector and title. This can be a small transformer or VAE over boxes (custom code), an LLM producing JSON as in LayoutGPT ([Feng et al. 2023](https://arxiv.org/abs/2305.15393)), or OpenCOLE-style design plans ([Inoue et al. 2024](https://arxiv.org/abs/2406.08232); [code](https://github.com/CyberAgentAILab/OpenCOLE)).
- **Stage 2:** render the layout with ControlNet or GLIGEN ([Li et al. 2023](https://arxiv.org/abs/2301.07093)).
- **Stage 3:** composite typography programmatically, or use a glyph-aware model (AnyText, [Tuo et al. 2023](https://arxiv.org/abs/2311.03054); FLUX.2 klein; Qwen-Image).
- **Pros:** it maps one-to-one onto the analysis features, which makes it the most interpretable story; the layout model is custom code; text stays legible.
- **Cons:** more moving parts and more integration work, and errors compound across stages.

## 5. Recommended champion and challenger

**Champion: SDXL + thumbnail LoRA + layout control + post-hoc text.**

- Train a LoRA on high-performing thumbnails, with captions from Florence-2 or BLIP-2 plus feature tokens (emotion, palette words, `<high_perf>`).
- At inference, set the condition vector from the analysis. Render face and pose boxes and the saliency map as ControlNet-union or T2I-Adapter inputs, add a style-only IP-Adapter from the creator's top thumbnails, then overlay the title text with PIL into the planned text box.
- SDXL is chosen over FLUX.2 klein 4B because of the more mature, MPS-tested adapter ecosystem. Swapping in klein 4B (Apache 2.0) is a stretch goal if time allows.

**Challenger: a custom conditional generator trained from scratch.**

- A conditional GAN (projection discriminator plus auxiliary condition losses) at 256x144, using the same condition vector.
- It is the "classical" competitor and earns the custom-code bonus.
- If the team wants a closer contest, make retrieval-plus-edit (4.5) a third, zero-training baseline.

The comparison is fair because both models take the same condition vector and are scored by the same metrics at the same output size (downsample the champion's images for FID/KID).

## 6. Evaluation

Generate about 1,000 to 2,000 images per model from held-out condition vectors (videos not used in training).

| Aspect | Metric | Tool |
|---|---|---|
| Realism vs. real thumbnails | FID ([Heusel 2017](https://arxiv.org/abs/1706.08500)); **KID** ([Binkowski 2018](https://arxiv.org/abs/1801.01401)), which is less biased at N < 10k | [clean-fid](https://github.com/GaParmar/clean-fid) ([Parmar 2022](https://arxiv.org/abs/2104.11222)) or `torchmetrics` |
| Prompt and title adherence | CLIPScore ([Hessel 2021](https://arxiv.org/abs/2104.08718)) | `torchmetrics.multimodal.CLIPScore` |
| Condition adherence | Face count accuracy, face-box IoU, emotion accuracy, palette ΔE (CIEDE2000), OCR legibility of the overlay, text-area error | The analysis side's detectors, re-run on generated images |
| Diversity and memorization | Intra-set LPIPS; nearest-neighbor CLIP similarity to the training set (flag anything above about 0.95) | `lpips`, CLIP |
| **Predicted success** | Mean predicted `perf` / P(top tercile) from the team's predictor, relative to the real held-out thumbnails | The team's model as a "critic" |
| Human preference | Blind pairwise A/B ("which would you click?"), 20 to 50 pairs per rater | Simple form |

**Predictor gaming risk.** A generator tuned (or cherry-picked) against the same predictor used to score it will find that predictor's blind spots: oversaturation, giant faces everywhere, adversarial textures. Reward-tuned diffusion models are known to over-optimize and collapse to a few high-reward images ([DRaFT, Clark et al. 2024](https://arxiv.org/abs/2309.17400); [DDPO, Black et al. 2023](https://arxiv.org/abs/2305.13301); [Zhang et al. 2024](https://arxiv.org/abs/2402.08552)). Mitigations:

1. Never fine-tune or rerank against the evaluation critic. If we rerank best-of-N, use a different critic (another architecture, seed or data split) from the one that reports results.
2. Report the critic score next to realism, diversity and memorization, never alone.
3. Check the critic on out-of-distribution inputs (noise, oversaturated real thumbnails) to see whether it is exploitable.
4. Remember that the critic predicts relative views, not causal CTR, so a high score means "looks like this creator's hits", not "will get clicks".

## 7. Minimal plan and milestones

No GPU or MPS training until the Mac is free. Cloud training happens on Colab.

| Week | Milestone | Output |
|---|---|---|
| 1 | Agree on the success label and feature schema with the team; export a manifest (`video_id, channel, drive_file_id, perf, success`) from BigQuery; create a train/val/test split by video, stratified by creator | `data/manifest.parquet`, split file |
| 2 | Condition-vector extractor (captions, faces, emotion, OCR boxes, palette, saliency), shared with the analysis side's EDA; render the condition maps | `src/conditions.py`, sample visualizations for the EDA |
| 3 | Retrieval-plus-edit baseline (inference only); evaluation harness (KID/FID, CLIPScore, adherence, LPIPS, critic hook) | `src/eval_gen.py`, first numbers |
| 4 | Challenger: train the custom cGAN on Colab | Checkpoints, samples, metrics |
| 5 | Champion: SDXL LoRA on Colab; ControlNet/T2I/IP-Adapter inference pipeline; PIL text overlay | LoRA weights, pipeline |
| 6 | Full comparison on the held-out set, human A/B test, critic scoring with a held-out critic | Results tables |
| 7 | Model-ops: FastAPI endpoint `POST /generate {channel_id, title, overrides}` returning N candidates ranked by the critic; Gradio demo; MLflow tracking; monthly refresh (re-scrape, recompute labels, retrain the LoRA when KID or critic drift exceeds a threshold); deck slides | Deployment diagram, demo |

## 8. Open questions for the team

1. **Success label:** do we agree on views relative to the channel median, controlling for age? What is the minimum video age, and is it a binary top tercile or a regression target?
2. **Visual features:** who owns feature extraction (faces, emotion, OCR, palette, saliency)? Can we share one `conditions.py` so the analysis and generation use identical definitions?
3. **Two cognitive problems:** is the plan (1) success prediction (regression or classification) and (2) conditional generation? Who builds the predictor, and can it expose a `score(image, channel_id)` function for the critic?
4. **Creator list:** how many creators, and are they all in one niche? Generation quality depends a lot on stylistic consistency.
5. **Data access:** can we export thumbnails from Drive in bulk (e.g. a zip, or a script using `drive_file_id`) for Colab training?
6. **Compute:** does the team have Colab Pro or cloud credits?
7. **Data retention:** the YouTube API policies limit storing API data such as view counts to 30 days without refresh (section 9). Do we refresh, or document this as an academic-use limitation?
8. **Scraper:** `thumbnail_size` is left NULL when a thumbnail already existed in Drive. Is that intended? Also, `__pycache__/` is committed and the README link to `Scraper_SETUP.md` is malformed (`[Scraper_(SETUP.md)` has no closing bracket or URL).
9. **Faces:** are we OK generating real creators' faces at all, or should we restrict to generic or synthetic faces (my recommendation)?

## 9. Ethics, copyright and ToS

- **Copyright of thumbnails.** Thumbnails are the creators' copyrighted works. We use them for non-commercial academic research and do not redistribute the dataset (the course doesn't require uploading it). Don't publish the LoRA weights or a public demo that reproduces specific creators' thumbnails. Run the nearest-neighbor memorization check before showing any samples.
- **Creator likeness.** Generating identifiable people (especially with face-ID adapters) raises right-of-publicity and deepfake concerns. Default to style-only conditioning and generic faces. Never present generated images as a creator's real work. Label every output "AI-generated". YouTube itself asks creators to disclose realistic synthetic media.
- **YouTube ToS and API policies.** The [YouTube Terms of Service](https://www.youtube.com/t/terms) restrict collecting content outside approved methods. We use the official Data API, which is the approved route. The [API Developer Policies](https://developers.google.com/youtube/terms/developer-policies) say non-authorized statistics should not be stored for more than 30 days and that stored metadata must be refreshed or deleted (see also the [derived-metrics policy](https://developers.google.com/youtube/terms/derived-metrics-policy)). Our BigQuery snapshot and the derived `perf` metric sit in a grey area. Keep the snapshot private, and document it.
- **Model licenses.** SDXL (OpenRAIL++-M) and SD 1.5 (OpenRAIL-M) have use-based restrictions, for example no impersonation or deception. FLUX.2 klein 9B and dev, InsightFace models and StyleGAN2-ADA are non-commercial only. Fine for coursework, but a deployment plan for real creators would have to use Apache-2.0 components (FLUX.2 klein 4B, IP-Adapter, Qwen-Image) or buy licenses.
- **Bias and homogenization.** Optimizing for "what worked" can amplify clickbait conventions (exaggerated shock faces, misleading imagery) and penalize styles of underrepresented creators. Report this in the conclusion.

## 10. References

- Data and policies: [YouTube Data API](https://developers.google.com/youtube/v3); [Developer Policies](https://developers.google.com/youtube/terms/developer-policies); [YouTube ToS](https://www.youtube.com/t/terms). Possible extra open data: [l3afai/youtube-thumbnails](https://huggingface.co/datasets/l3afai/youtube-thumbnails) (about 164k thumbnails with titles, no performance labels; check its license).
- Models and licenses: [SD 1.5 mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5); [SDXL](https://arxiv.org/abs/2307.01952); [Stability Community License](https://stability.ai/license); [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B); [FLUX.2 repo](https://github.com/black-forest-labs/flux2); [BFL licensing](https://bfl.ai/licensing); [Qwen-Image](https://github.com/QwenLM/Qwen-Image).
- Methods: LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)); ControlNet ([2302.05543](https://arxiv.org/abs/2302.05543)); T2I-Adapter ([2302.08453](https://arxiv.org/abs/2302.08453)); IP-Adapter ([2308.06721](https://arxiv.org/abs/2308.06721)); GLIGEN ([2301.07093](https://arxiv.org/abs/2301.07093)); LayoutGPT ([2305.15393](https://arxiv.org/abs/2305.15393)); OpenCOLE ([2406.08232](https://arxiv.org/abs/2406.08232)); AnyText ([2311.03054](https://arxiv.org/abs/2311.03054)); cGAN ([1411.1784](https://arxiv.org/abs/1411.1784)); StyleGAN2-ADA ([2006.06676](https://arxiv.org/abs/2006.06676)).
- Evaluation: FID ([1706.08500](https://arxiv.org/abs/1706.08500)); KID ([1801.01401](https://arxiv.org/abs/1801.01401)); clean-fid ([2104.11222](https://arxiv.org/abs/2104.11222)); CLIPScore ([2104.08718](https://arxiv.org/abs/2104.08718)); DRaFT ([2309.17400](https://arxiv.org/abs/2309.17400)); DDPO ([2305.13301](https://arxiv.org/abs/2305.13301)); reward over-optimization ([2402.08552](https://arxiv.org/abs/2402.08552)).
- Tooling: [diffusers](https://huggingface.co/docs/diffusers); [diffusers on MPS](https://huggingface.co/docs/diffusers/optimization/mps); [peft](https://huggingface.co/docs/peft).
