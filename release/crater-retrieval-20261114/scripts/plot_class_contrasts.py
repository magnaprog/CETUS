"""Display saved regional class AP contrasts; no scoring, fitting or NPZ input.

python -B -m scripts.plot_class_contrasts --report results.json \
    --report-sha256 SHA256 --output FRESH_PRIVATE_DIRECTORY

The reader verifies the report hash and complete contrast membership.
Both font variants copy the saved differences, multiplied by 100, without
reaveraging regional AP or the paired-seed metadata.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import math
from pathlib import Path
import sys

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.text import Text

from scripts import cvpr_figure_style as sans
from scripts import figure_style as roman
from scripts import plot_class_ranking as ranking
from scripts.plot_class_ranking import CLASSES, at, digest, mapped, read_report, require, write_json


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("dino_c1", "DINOv2: pretrained − random",
     "AP from one pretrained state minus the mean AP of five random states (42–46); logistic regression."),
    ("final", "ResNet18: ImageNet − scratch",
     "Mean of three seed-paired differences (0–2); respective validation-selected training procedures."),
)
LIMITS = (-50, 50)
TICKS = (-40, -20, 0, 20, 40)
REGION_OFFSETS = (-.28, -.14, 0, .14, .28)
PAIRED = "paired_seed_differences_0_1_2"


def difference(value):
    require(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1,
            "Invalid finite AP difference")


def validate_contrasts(report):
    """Check all 70 identities, including macro; do not recompute any contrast."""
    rows = report["summaries"].get("contrasts")
    require(isinstance(rows, list), "Missing contrast rows")
    expected = {(kind, fold, name) for kind, _, _ in PANELS
                for fold in range(5) for name in (*CLASSES, "macro")}
    index = {}
    for i, row in enumerate(rows):
        require(isinstance(row, dict), "Invalid contrast row")
        kind, fold, name = row.get("kind"), row.get("fold"), row.get("class")
        require(type(kind) is str and type(name) is str and type(fold) is int
                and (kind, fold, name) in expected, "Contrast identity differs")
        require(type(row.get("region")) is int and row["region"] == fold + 1,
                "Contrast region differs")
        key = kind, fold, name
        require(key not in index, "Duplicate contrast identity")
        difference(row.get("pretrained_minus_baseline"))
        require(PAIRED in row, "Missing paired-seed metadata")
        paired = row[PAIRED]
        if kind == "dino_c1":
            require(paired is None, "DINOv2 must not claim paired-seed differences")
        else:
            require(isinstance(paired, list) and len(paired) == 3,
                    "ResNet18 requires three paired-seed differences")
            for value in paired:
                difference(value)
        index[key] = f"/summaries/contrasts/{i}"
    require(set(index) == expected, "Complete 70-row contrast grid required, including macro")
    return index


def figure_data(report):
    index = validate_contrasts(report)
    panels = []
    for kind, title, estimand in PANELS:
        points = []
        for c, name in enumerate(CLASSES):
            for fold, offset in enumerate(REGION_OFFSETS):
                pointer = index[kind, fold, name]
                points.append({"kind": kind, "class": name, "class_index": c,
                               "fold": fold, "region": fold + 1, "y": c + offset,
                               "contrast": mapped(report, pointer + "/pretrained_minus_baseline"),
                               "paired_metadata": {"pointer": pointer + "/" + PAIRED,
                                                   "raw": at(report, pointer + "/" + PAIRED)}})
        panels.append({"kind": kind, "title": title, "estimand": estimand, "points": points})
    return panels


def draw_figure(panels, variant):
    require(variant in ("cvpr", "full"), "Unknown font variant")
    values = [p["contrast"]["display_value"] for panel in panels for p in panel["points"]]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
            "Display requires finite contrasts")
    require(all(LIMITS[0] <= v <= LIMITS[1] for v in values), "Clipped data artist")
    roman.configure()
    if variant == "cvpr":
        sans.configure()
    figure = Figure(figsize=(6.5, 4.1), dpi=180)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.135, right=.985, bottom=.21, top=.90, wspace=.62)
    records, lines = [], []
    for ax, panel in zip(axes, panels):
        ax.set(xlim=LIMITS, ylim=(5.6, -.6), xscale="linear",
               xlabel="AP difference (percentage points)")
        ax.set_xticks(TICKS)
        ax.set_yticks(range(6), [name.title() for name in CLASSES])
        ax.set_title(panel["title"], pad=8, fontsize=9)
        ax.axvline(0, color="#bbbbbb", linewidth=.7, zorder=1)
        for point in panel["points"]:
            fold = point["fold"]
            line, = ax.plot([point["contrast"]["display_value"]], [point["y"]],
                            linestyle="none", marker=roman.FOLD_MARKERS[fold],
                            color=roman.FOLD_COLORS[fold], ms=3.8, markeredgecolor="white",
                            markeredgewidth=.3, zorder=3)
            lines.append(line)
            records.append({"role": "class_contrast", "source": point})
    handles = [Line2D([], [], marker=roman.FOLD_MARKERS[f], color=roman.FOLD_COLORS[f],
                      linestyle="none", markersize=4, markeredgecolor="white", markeredgewidth=.3)
               for f in range(5)]
    legend = figure.legend(handles, [str(f + 1) for f in range(5)], title="Test region",
                           loc="lower center", bbox_to_anchor=(.5, .015), ncol=5,
                           columnspacing=1.5, handletextpad=.35, handlelength=1, frameon=False)
    if variant == "cvpr":
        qa = sans.prepare_figure(figure, "class-contrasts")
        font_file = sans.FONT_FILE
    else:
        font_file = roman.FONT_DIR / "NimbusRoman-Regular.ttf"
        properties = FontProperties(fname=font_file, size=9, weight="normal", style="normal")
        for _ in range(2):
            for text in figure.findobj(Text):
                text.set_fontproperties(properties)
            figure.canvas.draw()
        qa = sans.text_metrics(figure)
    require(not qa["clipped_text"], f"Clipped text: {qa['clipped_text']}")
    require(all(t["size_pt"] == 9 and t["weight"] == t["style"] == "normal"
                and Path(t["font_file"]) == font_file for t in qa["texts"]), "Font check failed")
    renderer = figure.canvas.get_renderer()
    require(not axes[0].title.get_window_extent(renderer).overlaps(
        axes[1].title.get_window_extent(renderer)), "Overlapping panel titles")
    for record, line in zip(records, lines):
        x, y = list(map(float, line.get_xdata())), list(map(float, line.get_ydata()))
        # Include marker extent, not just its center, when checking clipping.
        margin = (line.get_markersize() + line.get_markeredgewidth()) * figure.dpi / 144
        bounds = line.axes.bbox
        pixels = line.get_transform().transform(list(zip(x, y)))
        inside = all(bounds.x0 + margin <= px <= bounds.x1 - margin
                     and bounds.y0 + margin <= py <= bounds.y1 - margin for px, py in pixels)
        record.update(x=x, y=y, marker=line.get_marker(), color=line.get_color(),
                      inside_axes=inside)
    require(all(r["inside_axes"] for r in records), "Clipped data artist")
    qa["axes"] = [{"kind": panel["kind"], "xscale": ax.get_xscale(),
                   "xlim": list(ax.get_xlim()), "xticks": list(map(float, ax.get_xticks())),
                   "ylim": list(ax.get_ylim()), "yticks": list(map(float, ax.get_yticks()))}
                  for ax, panel in zip(axes, panels)]
    qa["legend"] = [{"region": t.get_text(), "marker": h.get_marker(), "color": h.get_color()}
                    for h, t in zip(legend.legend_handles, legend.get_texts())]
    return figure, {"artists": records, "qa": qa}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = read_report(args.report, args.report_sha256)
    panels = figure_data(report)
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Use a private directory outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    binding = {"report": str(args.report.resolve()), "report_sha256": args.report_sha256,
               "scientific_input": report["input"],
               "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                 (Path(__file__), Path(ranking.__file__), Path(sans.__file__),
                                  Path(roman.__file__))},
               "font_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                               (sans.FONT_FILE, roman.FONT_DIR / "NimbusRoman-Regular.ttf")},
               "python_executable": sys.executable, "python_version": sys.version,
               "python_executable_sha256": digest(Path(sys.executable)),
               "matplotlib": matplotlib.__version__, "numpy": version("numpy")}
    write_json(output / "started.json", binding)
    try:
        variants = {}
        for variant in ("cvpr", "full"):
            figure, evidence = draw_figure(panels, variant)
            roman.save_figure(figure, output / f"class-contrasts-{variant}")
            variants[variant] = evidence
            figure.clear()
        write_json(output / "artist-data.json", {
            "report_sha256": args.report_sha256,
            "units": "AP difference in percentage points; raw report fractions retained",
            "direction": "Positive favors pretrained initialization",
            "validated_contrast_rows": 70, "unplotted_class": "macro",
            "region_offsets": list(REGION_OFFSETS), "panels": panels, "variants": variants})
        write_json(output / "complete.json", {**binding, "status": "rendered",
                   "class_points_per_variant": 60, "validated_contrast_rows": 70,
                   "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                               for p in sorted(output.iterdir())}})
    except Exception as exc:
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
