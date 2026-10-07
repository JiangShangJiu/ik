"""A quiet, consistent visual language for reproducible IK figures.

Robot geometry is rendered by MuJoCo; Pillow only lays out type and panels.
No screenshots or previously generated figures are used as input.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = "#F8FAFC"
INK = "#203B38"
MUTED = "#687B75"
LINE = "#D6DDD4"
TEAL = "#167F73"
ORANGE = "#CE7439"
BLUE = "#537EAD"
RED = "#B95950"
WHITE = "#FFFFFF"


@lru_cache(maxsize=64)
def font(size=24, bold=False, mono=False):
    if mono:
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    else:
        weight = "Bold" if bold else "Regular"
        candidates = [
            f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text(im, xy, value, size=24, color=INK, bold=False, mono=False):
    ImageDraw.Draw(im).text(xy, str(value), font=font(size, bold, mono),
                           fill=color, anchor="lt", spacing=8)


def rule(im, xy, color=LINE, width=1):
    ImageDraw.Draw(im).line(xy, fill=color, width=width)


def chip(im, xy, label, color=TEAL, size=20):
    d = ImageDraw.Draw(im)
    f = font(size)
    w = int(d.textlength(label, font=f)) + 28
    x, y = xy
    d.rounded_rectangle((x, y, x + w, y + 38), radius=19, fill=color)
    d.text((x + 14, y + 7), label, font=f, fill=WHITE, anchor="lt")
    return w


def plate(number, title, subtitle, height=880, width=1440):
    im = Image.new("RGB", (width, height), BG)
    text(im, (48, 28), f"IK / KINEMATICS LAB                                      EXPERIMENT {number}",
         17, MUTED, mono=True)
    rule(im, (48, 66, width - 48, 66))
    text(im, (48, 92), title, 42, bold=True)
    text(im, (48, 156), subtitle, 23, MUTED)
    rule(im, (48, height - 49, width - 48, height - 49))
    text(im, (48, height - 31), "MuJoCo mesh / FK verified / fixed seeds / SI units", 15,
         MUTED, mono=True)
    text(im, (width - 250, height - 31), "INVERSE KINEMATICS", 15, MUTED, mono=True)
    return im


def inset(im, frame, box, rounded=12):
    """Fit a fresh render in its panel, without stretching the camera view."""
    x, y, w, h = box
    source = Image.fromarray(np.asarray(frame, dtype=np.uint8)) if not isinstance(frame, Image.Image) else frame
    source = source.resize((w, h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), rounded, fill=255)
    im.paste(source, (x, y), mask)


def save(im, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, optimize=True)
    print(f"saved: {path}", flush=True)


def metric(im, xy, value, label, color=TEAL, size=34):
    x, y = xy
    text(im, (x, y), value, size, color, mono=True)
    text(im, (x, y + size + 14), label, 20, MUTED)


def axes_style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=8)
    ax.grid(axis="y", color=LINE, alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_image(fig):
    from io import BytesIO
    out = BytesIO()
    fig.savefig(out, format="png", dpi=150, facecolor=BG)
    out.seek(0)
    return Image.open(out).convert("RGB").copy()
