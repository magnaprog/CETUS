"""Display authenticated saved Craters ranking endpoints, without rescoring AP.

python3 -B -m scripts.plot_crater_retrieval --report results.json \
    --report-sha256 SHA256 --curves craters-curves.npz \
    --curves-sha256 SHA256 --output FRESH_PRIVATE_DIRECTORY

Every whole-score-group endpoint is shown; no connectors imply attainable
counts inside ties. Larger symbols mark all and only groups that increase TP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from scripts import cvpr_figure_style as sans
from scripts import figure_style as roman
from scripts.plot_class_ranking import (
    ROOT, digest, expected_cells, read_report, require, write_json,
)


FIELDS = ("thresholds", "tp", "fp", "precision", "recall")
COUNT_LIMIT = 6500
ARCHIVE_LIMIT = 20 * 1024 * 1024
MEMBER_LIMIT = COUNT_LIMIT * 8 + 512  # One bounded vector plus its NPY header.
COLORS = {"random": "#B65E00", "scratch": "#B65E00", "pretrained": "#0072B2"}
DINO_KEYS = (("random", 42, "o", "Random 42"),
             ("random", 43, "s", "Random 43"),
             ("random", 44, "^", "Random 44"),
             ("random", 45, "D", "Random 45"),
             ("random", 46, "v", "Random 46"),
             ("pretrained", 42, "X", "Pretrained 42"))
CNN_KEYS = tuple((arm, seed, marker, f"{label} {seed}")
                 for arm, label in (("scratch", "Scratch"), ("pretrained", "ImageNet"))
                 for seed, marker in enumerate(("o", "s", "^")))
KEYS = {"dino_c1": DINO_KEYS, "final": CNN_KEYS}


def curve_index(report):
    """Bind prefixes to the writer's sorted 60-cell order, not index row order."""
    canonical = {cell: f"cell_{i:02d}" for i, cell in enumerate(sorted(expected_cells()))}
    entries = report["craters_curve_index"]
    require(isinstance(entries, list) and len(entries) == 60, "Curve grid must contain 60 cells")
    seen = set()
    for entry in entries:
        cell = entry["cell"]
        require(isinstance(cell, list) and len(cell) == 4
                and type(cell[0]) is str and type(cell[2]) is str
                and type(cell[1]) is int and type(cell[3]) is int,
                "Curve cell ID types differ")
        cell = tuple(cell)
        require(cell in canonical and cell not in seen, "Unknown or duplicate curve cell")
        require(type(entry["class_index"]) is int and entry["class_index"] == 5,
                "Curve class must be Craters (5)")
        require(entry["prefix"] == canonical[cell], "Curve prefix/cell mapping differs")
        seen.add(cell)
    require(seen == set(canonical), "Incomplete curve grid")
    return sorted(entries, key=lambda entry: entry["prefix"])


def read_curves(path, expected_sha256, report):
    """Read one hashed snapshot; validate the archive and saved count identities.

    The report must have passed read_report. These checks do not reconstruct
    probabilities or AP. Caller-supplied hashes bind accepted scientific inputs.
    """
    require(path.stat().st_size <= ARCHIVE_LIMIT, "Curve archive exceeds 20 MiB")
    data = path.read_bytes()
    require(len(data) <= ARCHIVE_LIMIT, "Curve archive exceeds 20 MiB")
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "Curves SHA256 differs")
    entries = curve_index(report)
    expected = {f"{e['prefix']}_{field}.npy" for e in entries for field in FIELDS}
    # Inspect raw ZIP membership before NumPy can hide duplicate names or aliases.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == 300 and {m.filename for m in members} == expected,
                "Curve archive must contain exactly 300 unique expected arrays")
        require(all(0 < m.file_size <= MEMBER_LIMIT for m in members),
                "Curve array exceeds bounded vector size")
    metrics = {tuple(row["cell"]): row for row in report["metrics"] if row["class_index"] == 5}
    curves = []
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        for entry in entries:
            prefix, cell = entry["prefix"], entry["cell"]
            row = metrics[tuple(cell)]
            n, k = row["n"], row["positives"]
            require(type(n) is int and type(k) is int and 0 < k < n <= COUNT_LIMIT,
                    "Curve support exceeds count axis or differs in type")
            arrays = {field: archive[f"{prefix}_{field}"] for field in FIELDS}
            tp, fp = arrays["tp"], arrays["fp"]
            require(all(a.ndim == 1 and a.shape == tp.shape for a in arrays.values())
                    and 1 <= tp.size <= n, "Curve vector shapes or group count differ")
            for name, limit in (("tp", k), ("fp", n - k)):
                a = arrays[name]
                require(a.dtype.kind in "iu" and a.dtype.itemsize <= 8,
                        f"{prefix}: {name} must contain integer counts")
                require(np.all((a >= 0) & (a <= limit))
                        and np.all(a[1:] >= a[:-1]) and a[-1] == limit,
                        f"{prefix}: {name} monotonic counts/endpoints differ")
            # Bounds are checked before int64 conversion/addition to avoid overflow.
            tp, fp = tp.astype(np.int64), fp.astype(np.int64)
            count = tp + fp
            require(count[0] >= 1 and count[-1] == n and np.all(count[1:] > count[:-1]),
                    f"{prefix}: retrieved group counts must strictly increase from 1..N to N")
            for name in ("thresholds", "precision", "recall"):
                a = arrays[name]
                require(a.dtype.kind == "f" and a.dtype.itemsize == 8 and np.isfinite(a).all()
                        and np.all((a >= 0) & (a <= 1)),
                        f"{prefix}: {name} must be finite float64 fractions")
            thresholds = arrays["thresholds"]
            require(np.all(thresholds[1:] < thresholds[:-1]),
                    f"{prefix}: thresholds must strictly descend")
            require(np.array_equal(arrays["recall"], tp / k)
                    and np.array_equal(arrays["precision"], tp / count),
                    f"{prefix}: saved recall/precision differs from integer counts")
            sizes = np.diff(np.r_[0, count])
            for name, value in (("distinct_scores", tp.size),
                                ("tied_groups", np.count_nonzero(sizes > 1)),
                                ("largest_tie", sizes.max())):
                require(type(row[name]) is int and row[name] == value,
                        f"{prefix}: saved {name} differs from groups")
            if k == 1:
                first = int(np.flatnonzero(tp)[0])
                single = row["single_positive"]
                require(single["rank_first"] == count[first] - sizes[first] + 1
                        and single["rank_last"] == count[first]
                        and single["tie_size"] == sizes[first],
                        f"{prefix}: single-positive group differs")
            zero = arrays["precision"] == 0
            require(np.all(arrays["recall"][zero] == 0), "Zero-precision groups must have zero recall")
            emphasis = np.flatnonzero(np.diff(np.r_[0, tp]) > 0)
            curves.append({"prefix": prefix, "cell": cell, "n": n, "positives": k,
                           "count": count, "recall_percent": 100 * arrays["recall"],
                           "emphasis_indices": emphasis,
                           "zero_precision_groups": int(np.count_nonzero(zero)),
                           "zero_recall_groups": int(np.count_nonzero(arrays["recall"] == 0))})
    return curves


def coordinate_digest(values):
    """Canonical little-endian float64 hash of every actual scalar coordinate."""
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def display_mapping(curve):
    return {key: curve[key] for key in ("prefix", "cell", "n", "positives",
                                       "zero_precision_groups", "zero_recall_groups")} | {
        "point_count": len(curve["count"]),
        "first_count": int(curve["count"][0]), "last_count": int(curve["count"][-1]),
        "tp_increase_endpoints": len(curve["emphasis_indices"]),
        "arrays": {name: f"{curve['prefix']}_{name}" for name in FIELDS},
    }


def draw_figure(curves, variant):
    require(variant in ("cvpr", "full"), "Unknown font variant")
    roman.configure()
    if variant == "cvpr":
        sans.configure()
    # Keep all points in the paths; no data reduction, even for dense plateaus.
    matplotlib.rcParams["path.simplify"] = False
    figure = Figure(figsize=(6.5, 7.4), dpi=180)
    FigureCanvasAgg(figure)
    axes = figure.subplots(5, 2, sharex=True, sharey=True)
    figure.subplots_adjust(left=.105, right=.985, bottom=.08, top=.825,
                           hspace=.45, wspace=.15)
    artists = []
    for curve in curves:
        kind, fold, arm, seed = curve["cell"]
        ax = axes[fold, 0 if kind == "dino_c1" else 1]
        marker = next(m for a, s, m, _ in KEYS[kind] if (a, s) == (arm, seed))
        x, y, selected = curve["count"], curve["recall_percent"], curve["emphasis_indices"]
        background, = ax.plot(x, y, linestyle="none", marker=".", markersize=1.1,
                              markeredgewidth=0, alpha=.3, color=COLORS[arm],
                              rasterized=True, clip_on=False, zorder=2)
        emphasis, = ax.plot(x[selected], y[selected], linestyle="none", marker=marker,
                            markersize=3.2, markeredgewidth=.35, alpha=.9,
                            color=COLORS[arm], rasterized=False, clip_on=False, zorder=3)
        artists.append((curve, background, emphasis))
    for fold, row in enumerate(axes):
        support = next(c for c in curves if c["cell"][1] == fold)
        for ax in row:
            ax.set(xscale="log", xlim=(1, COUNT_LIMIT), ylim=(-5, 105))
            ax.set_yticks([0, 50, 100])
            ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
            ax.minorticks_off()
            ax.grid(axis="y", color="#dddddd", linewidth=.4)
            ax.set_title(f"Region {fold + 1}: {support['positives']} / {support['n']:,}",
                         fontsize=9, pad=5)
    figure.supylabel("Recall of Craters map labels (%)", x=.015, y=.45)
    figure.supxlabel("Retrieved test tiles (log scale)", x=.545, y=.02)
    legends = []
    for col, (kind, title) in enumerate((("dino_c1", "DINOv2 + logistic regression"),
                                        ("final", "ResNet18 full training"))):
        center = (axes[0, col].get_position().x0 + axes[0, col].get_position().x1) / 2
        figure.text(center, .977, title, ha="center", va="top")
        handles = [Line2D([], [], linestyle="none", marker=marker, color=COLORS[arm],
                          markersize=3.2, markeredgewidth=.35, label=label)
                   for arm, seed, marker, label in KEYS[kind]]
        legends.append(figure.legend(handles=handles, loc="upper center", ncol=3,
                                     bbox_to_anchor=(center, .947), frameon=False,
                                     handlelength=.7, handletextpad=.3, columnspacing=.6,
                                     borderaxespad=0, labelspacing=.5))
    if variant == "cvpr":
        qa = sans.prepare_figure(figure, "crater-retrieval")
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
    boxes = [obj.get_window_extent(renderer)
             for obj in [*figure.texts, *(ax.title for ax in axes.flat), *legends]]
    require(not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]),
            "Overlapping headings/keys")
    require(all(ax.title.get_window_extent(renderer).y0 > ax.bbox.y1 for ax in axes.flat),
            "Region title overlaps data")
    require(all(legend.get_window_extent(renderer).y0 > axes[0, 0].bbox.y1
                for legend in legends), "Seed key overlaps data")
    records = []
    for curve, background, emphasis in artists:
        record = {"prefix": curve["prefix"], "cell": curve["cell"],
                  "tp_increase_group_indices": curve["emphasis_indices"].tolist()}
        for name, line, selected in (("all_groups", background, slice(None)),
                                      ("tp_increases", emphasis, curve["emphasis_indices"])):
            x, y = line.get_xdata(), line.get_ydata()
            require(np.array_equal(x, curve["count"][selected])
                    and np.array_equal(y, curve["recall_percent"][selected]),
                    "Actual artist coordinates differ from saved endpoints")
            require(line.get_linestyle() == "None" and line.get_markevery() is None
                    and not line.get_path().should_simplify, "Artist connects or thins groups")
            require(np.all((x >= 1) & (x <= COUNT_LIMIT) & (y >= 0) & (y <= 100)),
                    "Data outside common axes")
            record[name] = {"point_count": len(x), "x_sha256": coordinate_digest(x),
                            "y_sha256": coordinate_digest(y), "marker": line.get_marker(),
                            "color": line.get_color(), "linestyle": line.get_linestyle(),
                            "drawstyle": line.get_drawstyle(), "rasterized": line.get_rasterized(),
                            "path_simplify": line.get_path().should_simplify}
        records.append(record)
    qa["axes"] = [{"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                   "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                   "title_above_axes": True} for ax in axes.flat]
    qa["legend"] = [[{"label": t.get_text(), "marker": h.get_marker(), "color": h.get_color()}
                     for h, t in zip(legend.legend_handles, legend.get_texts())] for legend in legends]
    return figure, {"artists": records, "qa": qa}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--curves", required=True, type=Path)
    parser.add_argument("--curves-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Use a private directory outside the checkout")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    report = read_report(args.report, args.report_sha256)
    curves = read_curves(args.curves, args.curves_sha256, report)
    binding = {"report_sha256": args.report_sha256, "curves_sha256": args.curves_sha256,
               "source_sha256": digest(Path(__file__)), "python_version": sys.version,
               "numpy": np.__version__, "matplotlib": matplotlib.__version__,
               "dependencies_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                   Path(sans.__file__), Path(roman.__file__),
                   ROOT / "scripts/plot_class_ranking.py", sans.FONT_FILE,
                   roman.FONT_DIR / "NimbusRoman-Regular.ttf")}}
    output.mkdir(parents=True, exist_ok=False)
    variants = {}
    for variant in ("cvpr", "full"):
        figure, evidence = draw_figure(curves, variant)
        with matplotlib.rc_context({"savefig.dpi": 180}):
            roman.save_figure(figure, output / f"crater-retrieval-{variant}")
        variants[variant] = evidence
        # Direct Figure instances have no pyplot registry to close. Clearing
        # shared log axes here only triggers Matplotlib's reset-limit warnings.
        del figure
    write_json(output / "display.json", {
        **binding, "schema_version": "crater-retrieval-display-v2",
        "x": "saved TP + FP, retrieved test tiles; common log scale 1..6500",
        "y": "100 * saved recall; linear percent, padded -5..105 to show 0/100 marks",
        "emphasis": "all and only TP increases; previous TP starts at zero",
        "rendering": "All saved endpoints rasterized at 180 dpi; TP-increase markers vector. No connectors or thinning.",
        "coordinate_hash_encoding": "contiguous little-endian float64 scalar coordinates",
        "curves": [display_mapping(curve) for curve in curves], "variants": variants,
    })
    # Saving and actual-artist QA must both succeed. A partial run has no completion marker.
    names = [f"crater-retrieval-{variant}.{ext}" for variant in ("cvpr", "full")
             for ext in ("pdf", "png")] + ["display.json"]
    require(all((output / name).stat().st_size > 0 for name in names), "Empty render output")
    write_json(output / "complete.json", {
        **binding, "status": "rendered", "curves_per_variant": 60,
        "outputs": {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                    for name in names},
    })


if __name__ == "__main__":
    main()
