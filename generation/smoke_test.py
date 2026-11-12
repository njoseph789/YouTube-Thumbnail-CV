"""Step 2 of docs/thumbnail_generation_research.md: end-to-end champion pipeline, no training.

SDXL base + ControlNet-union (OpenPose skeleton placed from the condition's face box)
+ IP-Adapter (style from a creator's reference thumbnails) + PIL text overlay.
Everything is off-the-shelf; the LoRA trained later on Colab plugs into the same pipeline.

Usage (from the repo root, with the class venv):
    python generation/smoke_test.py --creator colinfurze --topic "building a jet-powered go kart" \
        --text "IT ACTUALLY FLIES" --expression surprised -n 2

    --creator     channel name as in images/labeled_subset/labels.csv (style references come from there)
    --refs        or explicit reference image paths instead of --creator
    --face-side   left | right | none
    --ip-scale    how strongly to copy the reference style (0 = off, ~0.3-0.6 is typical)

Outputs go to generation/outputs/<timestamp>/ (git-ignored): each candidate, the pose map, a layout
debug view, and a contact sheet. Runs on Apple Silicon (MPS), CUDA (Colab) or CPU (very slow).
First run downloads ~12 GB of weights into the Hugging Face cache.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from conditions import ThumbnailCondition, build_prompt, draw_layout, extract_palette, render_pose_map
from text_overlay import overlay_text

REPO = Path(__file__).resolve().parent.parent
LABELED = REPO / "images" / "labeled_subset"
OUT_ROOT = Path(__file__).resolve().parent / "outputs"
SIZE = (1344, 768)       # SDXL's native 16:9 bucket; thumbnails are 1280x720
OPENPOSE_MODE = 0        # xinsir union control types: 0 openpose, 1 depth, 2 hed, 3 canny, 4 normal, 5 seg...

LAYOUTS = {  # face box and text box (x, y, w, h fractions) for each face side
    "right": ((0.60, 0.12, 0.22, 0.42), (0.04, 0.08, 0.48, 0.42)),
    "left": ((0.18, 0.12, 0.22, 0.42), (0.50, 0.08, 0.46, 0.42)),
    "none": (None, (0.05, 0.06, 0.90, 0.30)),
}


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16  # inference in fp16 is fine on MPS; it's training that NaNs
    return "cpu", torch.float32


def creator_refs(creator: str, k: int) -> list[Path]:
    """Reference thumbnails for a creator: from a local copy of the Drive "Thumbnail Images" folder
    (images/Thumbnail Images/<creator>/<video_id>.jpg, git-ignored) if present, else the labeled subset."""
    full = REPO / "images" / "Thumbnail Images" / creator
    if full.is_dir():
        files = sorted(full.glob("*.jpg"))
        return [files[i] for i in range(0, len(files), max(1, len(files) // k))][:k]  # spread across the folder
    with open(LABELED / "labels.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["channel"].lower() == creator.lower()]
    if not rows:
        raise SystemExit(f"no thumbnails for creator {creator!r} in {LABELED / 'labels.csv'}")
    return [LABELED / r["filename"] for r in rows[:k]]


def load_pipeline(dtype, device, ip_adapter: bool):
    from diffusers import (AutoencoderKL, ControlNetUnionModel, DPMSolverMultistepScheduler,
                           StableDiffusionXLControlNetUnionPipeline)

    repo = "xinsir/controlnet-union-sdxl-1.0"  # the "promax" weights: union of 10+ control types
    cfg = json.loads(Path(hf_hub_download(repo, "config_promax.json")).read_text())
    controlnet = ControlNetUnionModel.from_config(cfg)
    controlnet.load_state_dict(load_file(hf_hub_download(repo, "diffusion_pytorch_model_promax.safetensors")))
    controlnet = controlnet.to(dtype)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)  # stock VAE NaNs in fp16
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae,
        torch_dtype=dtype, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if ip_adapter:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipe.to(device)
    pipe.vae.enable_tiling()
    return pipe


def contact_sheet(images: list[Image.Image], cols: int = 2, width: int = 640) -> Image.Image:
    h = round(width * images[0].height / images[0].width)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * h), "white")
    for i, im in enumerate(images):
        sheet.paste(im.resize((width, h)), ((i % cols) * width, (i // cols) * h))
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--text", default="", help="overlay text, ideally 3-5 words")
    ap.add_argument("--creator")
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--n-refs", type=int, default=3)
    ap.add_argument("--expression", default="surprised")
    ap.add_argument("--face-side", choices=LAYOUTS, default="right")
    ap.add_argument("--ip-scale", type=float, default=0.4)
    ap.add_argument("--control-scale", type=float, default=0.7)
    ap.add_argument("-n", type=int, default=2, help="candidates to generate")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    refs = [Path(p) for p in args.refs] or (creator_refs(args.creator, args.n_refs) if args.creator else [])
    ref_imgs = [Image.open(p).convert("RGB") for p in refs]
    face_box, text_box = LAYOUTS[args.face_side]
    cond = ThumbnailCondition(topic=args.topic, overlay_text=args.text, face_box=face_box,
                              expression=args.expression, text_box=text_box if args.text else None,
                              palette=extract_palette(ref_imgs) if ref_imgs else [])
    prompt, negative = build_prompt(cond)
    pose = render_pose_map(cond.face_box, SIZE)

    out = OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    pose.save(out / "pose_map.png")
    use_ip = bool(ref_imgs) and args.ip_scale > 0
    meta = {"condition": asdict(cond), "prompt": prompt, "negative": negative, "refs": [str(p) for p in refs],
            "args": vars(args)}
    (out / "condition.json").write_text(json.dumps(meta, indent=2))
    print("prompt:", prompt)

    device, dtype = device_and_dtype()
    t0 = time.time()
    pipe = load_pipeline(dtype, device, use_ip)
    print(f"loaded pipeline on {device} in {time.time() - t0:.0f}s")

    kwargs = {}
    if use_ip:
        pipe.set_ip_adapter_scale(args.ip_scale)
        kwargs["ip_adapter_image"] = [ref_imgs]  # one adapter, several reference images
    # With no face, an all-black pose map would tell ControlNet "no person", so turn it off instead.
    control_scale = args.control_scale if cond.face_box is not None else 0.0

    finals = []
    for i in range(args.n):
        t = time.time()
        g = torch.Generator("cpu").manual_seed(args.seed + i)
        img = pipe(prompt=prompt, negative_prompt=negative, control_image=[pose], control_mode=[OPENPOSE_MODE],
                   controlnet_conditioning_scale=control_scale, width=SIZE[0], height=SIZE[1],
                   num_inference_steps=args.steps, guidance_scale=args.guidance, generator=g, **kwargs).images[0]
        img = img.resize((1280, 720), Image.LANCZOS)
        img.save(out / f"raw_{i}.png")
        final = overlay_text(img, cond.overlay_text, cond.text_box)
        final.save(out / f"thumb_{i}.jpg", quality=92)
        draw_layout(img, cond).save(out / f"layout_{i}.png")
        finals.append(final)
        print(f"candidate {i}: {time.time() - t:.0f}s")

    contact_sheet(ref_imgs[:2] + finals if ref_imgs else finals).save(out / "sheet.jpg", quality=90)
    print("saved to", out)


if __name__ == "__main__":
    main()
