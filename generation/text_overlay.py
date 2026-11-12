"""Draw thumbnail title text into a box after generation.

Diffusion models (SDXL especially) can't spell reliably, so the overlay is rendered with PIL:
bold font, auto-sized to fill the box, word-wrapped, with a thick dark outline for legibility.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",       # macOS
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",  # Linux / Colab with mscorefonts
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # always present on Colab
]


def _font_path() -> str | None:
    return next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


def _wrap(words: list[str], n_lines: int) -> list[str]:
    """Split words into n_lines lines with roughly equal character counts."""
    target = sum(len(w) for w in words) / n_lines
    lines, cur = [], []
    for w in words:
        if cur and len(" ".join(cur)) >= target and len(lines) < n_lines - 1:
            lines.append(" ".join(cur))
            cur = []
        cur.append(w)
    return lines + [" ".join(cur)]


def overlay_text(im: Image.Image, text: str, box, fill=(255, 230, 0), stroke=(0, 0, 0)) -> Image.Image:
    """Draw `text` (upper-cased) as large as fits inside box = (x, y, w, h) fractions."""
    out = im.convert("RGB").copy()
    if not text.strip() or box is None:
        return out
    W, H = out.size
    bx, by, bw, bh = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    words = text.upper().split()
    d = ImageDraw.Draw(out)
    path = _font_path()
    best = None
    for n_lines in range(1, min(3, len(words)) + 1):
        lines = _wrap(words, n_lines)
        size = int(bh / n_lines)
        while size > 10:
            font = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
            sw = max(2, size // 12)
            l, t, r, b = d.multiline_textbbox((0, 0), "\n".join(lines), font=font, stroke_width=sw,
                                              spacing=size // 10, align="left")
            if r - l <= bw and b - t <= bh:
                break
            size = int(size * 0.92)
        if best is None or size > best[0]:
            best = (size, lines, font, sw, (l, t))
    size, lines, font, sw, (l, t) = best
    d.multiline_text((bx - l, by - t), "\n".join(lines), font=font, fill=fill, stroke_width=sw,
                     stroke_fill=stroke, spacing=size // 10, align="left")
    return out
