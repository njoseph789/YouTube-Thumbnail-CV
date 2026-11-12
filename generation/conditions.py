"""Condition vector for thumbnail generation, and rendering it into control maps.

A ThumbnailCondition describes *what* a thumbnail should contain (face position and expression,
where the text goes, colour palette, topic). At training time it would be measured from a real
thumbnail; at inference time it is set from what the EDA / success predictor say works for a creator.
This module turns it into the inputs the diffusion pipeline needs:

  - render_pose_map():   an OpenPose-style head-and-shoulders skeleton at the face box, for ControlNet
  - extract_palette():   dominant colours of reference thumbnails (k-means in Lab space)
  - palette_words():     those colours as prompt words ("red, yellow, black")
  - build_prompt():      the text prompt from the condition
  - draw_layout():       a debug overlay of the planned boxes

Boxes are (x, y, w, h) as fractions of the image (0-1), so they work at any resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

Box = tuple[float, float, float, float]  # x, y, w, h as fractions of width/height


@dataclass
class ThumbnailCondition:
    topic: str                                # what the video is about, e.g. "building a giant robot arm"
    overlay_text: str = ""                    # 0-5 words drawn after generation (empty = no text)
    face_box: Box | None = (0.58, 0.14, 0.22, 0.42)  # None = no face
    expression: str = "surprised"             # matches facial_expression values in labels.csv
    text_box: Box | None = (0.04, 0.08, 0.46, 0.40)
    palette: list[tuple[int, int, int]] = field(default_factory=list)  # RGB, most dominant first
    style: str = "youtube thumbnail, high contrast, sharp focus, professional photo"


# --- OpenPose rendering ---------------------------------------------------------------------------
# Standard 18-keypoint OpenPose (COCO) layout and colours, as drawn by controlnet_aux, which is what
# the xinsir ControlNet-union model was trained on. Indices: 0 nose, 1 neck, 2 R shoulder, 3 R elbow,
# 4 R wrist, 5 L shoulder, 6 L elbow, 7 L wrist, 8-13 hips/legs, 14 R eye, 15 L eye, 16 R ear, 17 L ear.
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
          (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
           (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
           (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def bust_keypoints(face_box: Box, size: tuple[int, int]) -> dict[int, tuple[float, float]]:
    """Place head-and-shoulders keypoints (pixels) around a face box. The person faces the camera,
    so their right side is on the image's left."""
    W, H = size
    x, y, w, h = face_box[0] * W, face_box[1] * H, face_box[2] * W, face_box[3] * H
    cx, cy = x + w / 2, y + h / 2
    return {
        0: (cx, cy + 0.08 * h),                                         # nose
        14: (cx - 0.18 * w, cy - 0.08 * h), 15: (cx + 0.18 * w, cy - 0.08 * h),  # eyes
        16: (cx - 0.42 * w, cy), 17: (cx + 0.42 * w, cy),               # ears
        1: (cx, y + 1.25 * h),                                          # neck
        2: (cx - 1.05 * w, y + 1.35 * h), 5: (cx + 1.05 * w, y + 1.35 * h),  # shoulders
        3: (cx - 1.35 * w, y + 2.3 * h), 6: (cx + 1.35 * w, y + 2.3 * h),    # elbows (often off-frame)
    }


def render_pose_map(face_box: Box | None, size: tuple[int, int]) -> Image.Image:
    """Black image with an OpenPose skeleton at the face box (all black if there is no face)."""
    canvas = Image.new("RGB", size, "black")
    if face_box is None:
        return canvas
    kp = bust_keypoints(face_box, size)
    limbs = Image.new("RGB", size, "black")
    d = ImageDraw.Draw(limbs)
    stick = max(3, round(size[1] / 128))  # controlnet_aux uses 4 px at 512 px
    for i, (a, b) in enumerate(_LIMBS):
        if a in kp and b in kp:
            d.line([kp[a], kp[b]], fill=_COLORS[i], width=2 * stick)
    canvas = Image.blend(canvas, limbs, 0.6)  # limbs are drawn at 60% like the original
    d = ImageDraw.Draw(canvas)
    for i, (px, py) in kp.items():
        d.ellipse([px - stick, py - stick, px + stick, py + stick], fill=_COLORS[i])
    return canvas


# --- Colour ---------------------------------------------------------------------------------------
_NAMED = {"black": (20, 20, 20), "white": (240, 240, 240), "gray": (128, 128, 128), "red": (220, 30, 30),
          "orange": (245, 140, 20), "yellow": (250, 220, 30), "green": (40, 170, 60), "teal": (20, 160, 160),
          "blue": (30, 90, 220), "purple": (130, 50, 190), "pink": (240, 110, 180), "brown": (120, 75, 40)}


def extract_palette(images: list[Image.Image], k: int = 5, seed: int = 0) -> list[tuple[int, int, int]]:
    """Dominant colours across images: k-means on pixels in Lab space (perceptually even), sorted by
    how many pixels each cluster covers."""
    px = []
    for im in images:
        a = np.asarray(im.convert("RGB").resize((96, 54)), dtype=np.float64) / 255.0
        px.append(rgb2lab(a).reshape(-1, 3))
    lab = np.concatenate(px)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(lab)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    rgb = lab2rgb(km.cluster_centers_[order][None])[0]
    return [tuple(int(round(c * 255)) for c in col) for col in rgb]


def palette_words(palette: list[tuple[int, int, int]], n: int = 3) -> list[str]:
    """Nearest colour names (in Lab distance), de-duplicated, most dominant first."""
    names = list(_NAMED)
    ref = rgb2lab(np.array([_NAMED[c] for c in names], dtype=np.float64)[None] / 255.0)[0]
    out = []
    for col in palette:
        lab = rgb2lab(np.array(col, dtype=np.float64)[None, None] / 255.0)[0, 0]
        name = names[int(np.argmin(np.linalg.norm(ref - lab, axis=1)))]
        if name not in out:
            out.append(name)
    return out[:n]


def color_stats(im: Image.Image) -> dict[str, float]:
    """Mean saturation and brightness (HSV, 0-1) and RMS contrast of the luminance."""
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255.0
    lum = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
    return {"saturation": hsv[..., 1].mean(), "brightness": hsv[..., 2].mean(), "contrast": lum.std()}


# --- Prompt and debug view ------------------------------------------------------------------------
_EXPRESSION_WORDS = {
    "surprised": "shocked surprised expression, wide eyes, open mouth",
    "happy": "big smile, excited happy expression",
    "angry": "angry intense expression, furrowed brows",
    "sad": "sad disappointed expression",
    "skeptical": "skeptical expression, raised eyebrow",
    "smug": "smug confident smirk",
    "pained": "pained grimacing expression",
    "neutral": "calm neutral expression, looking at camera",
}


def build_prompt(c: ThumbnailCondition) -> tuple[str, str]:
    parts = [c.topic]
    if c.face_box is not None:
        parts.append(f"a person, head and shoulders, {_EXPRESSION_WORDS.get(c.expression, c.expression)}")
    if c.palette:
        parts.append(", ".join(palette_words(c.palette)) + " color scheme")
    else:
        parts.append("vibrant saturated colors")
    if c.text_box is not None:
        parts.append("clean uncluttered area for a title")
    parts.append(c.style)
    negative = ("text, letters, watermark, logo, signature, blurry, lowres, deformed face, extra fingers, "
                "jpeg artifacts, dull colors")
    return ", ".join(parts), negative


def draw_layout(im: Image.Image, c: ThumbnailCondition) -> Image.Image:
    """Copy of the image with the planned face (green) and text (yellow) boxes outlined."""
    out = im.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    W, H = out.size
    for box, color in ((c.face_box, "lime"), (c.text_box, "yellow")):
        if box is not None:
            x, y, w, h = box
            d.rectangle([x * W, y * H, (x + w) * W, (y + h) * H], outline=color, width=max(2, W // 300))
    return out
