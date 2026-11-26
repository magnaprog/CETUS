"""Set Matplotlib labels in the bundled 9-point Nimbus Sans face."""

from pathlib import Path

from matplotlib import font_manager, rcParams
from matplotlib.text import Text

FONT = "Nimbus Sans"
FONT_SIZE = 9
FONT_DIR = Path(__file__).resolve().parents[1] / "fonts/nimbus-sans"
FONT_FILE = FONT_DIR / "NimbusSans-Regular.ttf"
MAIN_WIDTH = 6.875
SUPPLEMENT_WIDTH = 6.5
COLUMN_WIDTH = (MAIN_WIDTH - 0.3125) / 2


def configure():
    """Select the bundled regular TTF, including math glyphs, without CFF fallback."""
    manager = font_manager.fontManager
    manager.ttflist[:] = [entry for entry in manager.ttflist if entry.name != FONT]
    manager.addfont(FONT_FILE)
    rcParams.update({
        "font.family": FONT, "font.size": FONT_SIZE,
        "font.weight": "normal", "font.style": "normal",
        "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
        "figure.titleweight": "normal", "text.usetex": False,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": FONT, "mathtext.bf": FONT, "mathtext.bfit": FONT,
        "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": FONT,
        "mathtext.fallback": None, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None, "savefig.pad_inches": 0,
    })


def prepare_figure(figure, name):
    """Restyle text before export at the manuscript's physical inclusion width."""
    configure()
    width = (COLUMN_WIDTH if name == "initialization" else
             MAIN_WIDTH if name in ("dataset", "geography", "fitting", "examples-extra")
             else SUPPLEMENT_WIDTH)
    figure.set_size_inches(width, figure.get_figheight(), forward=True)
    if name == "training-geography":
        figure.subplots_adjust(left=.155)
    elif name == "domain-examples":
        # Wrap only the row label; keep the 3x6 panel grid fixed for comparison.
        figure.axes[0].set_ylabel("Earth\n(Sentinel-1\nC-band)")
        for axes in figure.axes:
            for image in axes.images:
                image.set_interpolation("nearest")
    # Draw first to materialize tick labels; reapply after layout has settled.
    properties = font_manager.FontProperties(fname=FONT_FILE, size=FONT_SIZE,
                                              weight="normal", style="normal")
    for _ in range(2):
        for text in figure.findobj(Text):
            text.set_fontproperties(properties)
            text.set_math_fontfamily("custom")
        figure.canvas.draw()
    return text_metrics(figure)


def text_metrics(figure, export_bbox=None):
    """Record actual visible artist bounds and fonts; superscript sizes are PDF-level."""
    renderer = figure.canvas.get_renderer()
    origin_x, origin_y = ((export_bbox.x0, export_bbox.y0) if export_bbox is not None else (0, 0))
    width, height = figure.get_size_inches()
    tight = figure.get_tightbbox(renderer)
    # Locators retain labels beyond the view limits, but Axis.draw omits them.
    # Do not report those unused Text artists as clipped exported callouts.
    outside_ticks = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                if not low <= tick.get_loc() <= high:
                    outside_ticks.update((tick.label1, tick.label2))
    records = []
    for text in figure.findobj(Text):
        if text in outside_ticks or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        x0, x1 = box.x0 - origin_x * figure.dpi, box.x1 - origin_x * figure.dpi
        y0, y1 = box.y0 - origin_y * figure.dpi, box.y1 - origin_y * figure.dpi
        records.append(dict(text=text.get_text(), size_pt=text.get_fontsize(),
                            font_file=text.get_fontproperties().get_file(),
                            weight=text.get_fontweight(), style=text.get_fontstyle(),
                            bounds_px=[x0, y0, x1, y1],
                            inside_canvas=bool(x0 >= -.01 and y0 >= -.01
                                               and x1 <= width * figure.dpi + .01
                                               and y1 <= height * figure.dpi + .01)))
    return dict(width_inches=figure.get_figwidth(), height_inches=figure.get_figheight(),
                export_window_inches=[origin_x, origin_y, width, height],
                content_margins_pt=[72 * (tight.x0 - origin_x), 72 * (origin_x + width - tight.x1)],
                texts=records, clipped_text=[r["text"] for r in records if not r["inside_canvas"]])
