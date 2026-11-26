"""Nimbus Roman typography and visual encodings for CETUS figures.

Widths are physical inches at the paper's final size. Font files are the
unmodified Nimbus Roman TrueType faces identified in fonts/nimbus-roman.
Models, folds and domains use Paul Tol's vibrant colors; map classes use his
bright scheme. Markers and labels also distinguish categories.
"""

from pathlib import Path

import numpy as np
from matplotlib import colormaps, font_manager, rcParams

WIDTH = 6.5
SMALL = 8
TEXT = 9
TITLE = 10
FONT = "Nimbus Roman"
FONT_DIR = Path(__file__).resolve().parents[1] / "fonts/nimbus-roman"
FOLD_COLORS = ("#0077BB", "#009988", "#EE7733", "#EE3377", "#33BBEE")
FOLD_MARKERS = ("o", "s", "^", "D", "v")
MODEL_COLORS = dict(dinov2="#0077BB", dofa="#EE3377", croma="#EE7733", random_init="#111111")
MODEL_MARKERS = dict(dinov2="o", dofa="s", croma="^", random_init="D")
CLASS_COLORS = ("#CCBB44", "#EE6677", "#228833", "#AA3377", "#66CCEE", "#4477AA")
DOMAIN_COLORS = dict(earth="#EE7733", titan="#0077BB", venus="#EE3377")
SEQUENTIAL = "YlOrBr"
INK = "#252525"


def configure():
    """Register the pinned TrueType faces, avoiding installed CFF substitutes."""
    manager = font_manager.fontManager
    # Matplotlib 3.10 can select an installed OpenType CFF face with the same
    # family name, then label its PDF stream as TrueType. Restrict this family
    # to the supplied TTFs; other installed families remain available.
    manager.ttflist[:] = [entry for entry in manager.ttflist if entry.name != FONT]
    for face in ("Regular", "Bold", "Italic", "BoldItalic"):
        manager.addfont(FONT_DIR / f"NimbusRoman-{face}.ttf")
    rcParams.update({
        "font.family": FONT, "font.size": TEXT,
        "axes.labelsize": TEXT, "axes.titlesize": TITLE,
        "xtick.labelsize": TEXT, "ytick.labelsize": TEXT,
        "legend.fontsize": SMALL, "figure.titlesize": TITLE,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.grid": False,
        "lines.linewidth": 1.1, "lines.markersize": 4,
        "lines.markeredgecolor": INK, "lines.markeredgewidth": .5,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "legend.frameon": False,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
        "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": FONT,
        "mathtext.fallback": None,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 220, "savefig.bbox": None, "savefig.pad_inches": 0,
    })


def cell_text_color(value, norm, cmap=SEQUENTIAL):
    """Choose black or white for the larger contrast against a filled cell."""
    rgb = np.asarray(colormaps[cmap](norm(value))[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = linear @ np.array([.2126, .7152, .0722])
    return "black" if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else "white"


def centered_bbox(figure, content_shift_pt=0):
    """Translate the export window without changing its size or any artist."""
    from matplotlib.transforms import Bbox

    figure.canvas.draw()
    bounds = figure.get_tightbbox(figure.canvas.get_renderer())
    width, height = figure.get_size_inches()
    return Bbox.from_bounds((bounds.x0 + bounds.x1 - width) / 2 - content_shift_pt / 72,
                           0, width, height)


def save_figure(figure, stem, *, center_content=False, content_shift_pt=0):
    """Keep the physical canvas fixed and embed searchable fonts in PDF."""
    stem = Path(stem)
    options = {"bbox_inches": centered_bbox(figure, content_shift_pt), "pad_inches": 0} if center_content else {}
    figure.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None}, **options)
    figure.savefig(stem.with_suffix(".png"), dpi=180, **options)
