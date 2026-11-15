"""Present an accepted class-ranking report; never read probabilities or score AP.

python -B -m scripts.plot_class_ranking --report results.json \
    --report-sha256 SHA256 --output FRESH_PRIVATE_DIRECTORY
python -B -m scripts.plot_class_ranking --report results.json \
    --report-sha256 SHA256 --check-table FRESH_PRIVATE_DIRECTORY/class-ranking.tex

The reader verifies the report hash and complete display membership.
Plots and tables copy the saved scores and averages.
Both font variants use the same data, all seeds, and a common logarithmic axis.
The log axis is a post-results display choice; means remain saved arithmetic AP.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
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
from scripts.prepare_cvpr_tables import table


CLASSES = ("plains", "dunes", "hummocky", "labyrinths", "lakes", "craters")
ARMS = (
    ("dino_c1", "random", (42, 43, 44, 45, 46), "DINOv2\nrandom"),
    ("dino_c1", "pretrained", (42,), "DINOv2\npretrained"),
    ("final", "scratch", (0, 1, 2), "ResNet18\nscratch"),
    ("final", "pretrained", (0, 1, 2), "ResNet18\nImageNet"),
)
ROOT = Path(__file__).resolve().parents[1]
LOG_LIMITS = (.01, 50)
LOG_TICKS = (.01, .1, 1, 10, 50)
MARKERS = ("o", "s", "^", "D")
TABLE_NOTE = (
    "AP (%), mean +/- sample SD across five equally weighted geographic regions, "
    "after averaging saved AP over seeds within each region. Macro AP includes all six "
    "classes. SD describes geography, not independent seed uncertainty. "
    "DINOv2 uses logistic regression with matched random/pretrained encoders; "
    "ResNet18 uses full training. DINOv2 random has five seeds, pretrained one; "
    "each ResNet18 arm has three. No winner highlighting."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "Invalid AP/display fraction")


def at(report, pointer):
    value = report
    for part in pointer.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def expected_cells():
    return {(kind, fold, arm, seed) for kind, arm, seeds, _ in ARMS
            for fold in range(5) for seed in seeds}


def validate(report):
    """Check display joins, not upstream score arithmetic or scientific admission."""
    require(report["schema_version"] == "class-ranking-v1"
            and report["status"] == "computed_pending_publication", "Report status differs")
    require(report["class_order"] == list(CLASSES), "Class order differs")
    require(len(report["fold_counts"]) == 5, "Five regions required")
    for counts in report["fold_counts"]:
        test = counts["test"]
        require(type(test["total"]) is int and len(test["class_counts"]) == 6
                and all(type(k) is int and k > 0 for k in test["class_counts"])
                and sum(test["class_counts"]) == test["total"], "Test denominators differ")
    cells = expected_cells()
    provenance = [tuple(r["cell"]) for r in report["cells"]]
    require(len(provenance) == 60 and set(provenance) == cells, "Cell grid differs")
    metrics = {}
    for i, row in enumerate(report["metrics"]):
        cell, c = tuple(row["cell"]), row["class_index"]
        require(cell in cells and type(c) is int and 0 <= c < 6, "Metric identity differs")
        require(type(cell[1]) is int and type(cell[3]) is int, "Cell ID types differ")
        require(row["class"] == CLASSES[c] and row["region"] == cell[1] + 1,
                "Metric class/region differs")
        key = (*cell, row["class"])
        require(key not in metrics, "Duplicate metric identity")
        test = report["fold_counts"][cell[1]]["test"]
        n, k = test["total"], test["class_counts"][c]
        require(row["n"] == n and row["positives"] == k and row["negatives"] == n - k,
                "Metric support differs")
        require(row["prevalence_all_tied_reference"] == k / n, "Prevalence differs")
        require(row["undefined_reason"] is None, "Undefined AP in displayed grid")
        number(row["ap"])
        single = row["single_positive"]
        if k == 1:
            require(isinstance(single, dict), "Missing single-positive rank")
            first, last = single["rank_first"], single["rank_last"]
            require(type(first) is int and type(last) is int and 1 <= first <= last <= n
                    and single["tie_size"] == last - first + 1, "Invalid rank/tie metadata")
        else:
            require(single is None, "Unexpected single-positive metadata")
        metrics[key] = f"/metrics/{i}"
    require(len(metrics) == 360, "Complete 360-row grid required")
    regional, geographic = {}, {}
    arms = {(k, a) for k, a, _, _ in ARMS}
    for name, index in (("regional", regional), ("geographic", geographic)):
        for i, row in enumerate(report["summaries"][name]):
            if "initialization" not in row:
                require(name == "geographic" and "mean_difference" in row,
                        "Unexpected summary row")
                continue  # Saved contrasts are not part of this display.
            arm = row["kind"], row["initialization"]
            require(arm in arms and row["class"] in (*CLASSES, "macro"), "Summary identity differs")
            key = (*arm, row["class"])
            if name == "regional":
                fold = row["fold"]
                require(type(fold) is int and 0 <= fold < 5 and row["region"] == fold + 1,
                        "Summary region differs")
                key = (*key, fold)
                number(row["seed_mean_ap"])
            else:
                number(row["mean_ap"])
                number(row["regional_sd"])
            require(key not in index, "Duplicate summary identity")
            index[key] = f"/summaries/{name}/{i}"
    require(len(regional) == 140 and len(geographic) == 28, "Summary grid incomplete")
    return metrics, regional, geographic


def read_report(path, expected_sha256):
    require(path.stat().st_size <= 5 * 1024 * 1024, "Display report exceeds 5 MiB")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "Report SHA256 differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    report = json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)
    validate(report)
    return report


def mapped(report, pointer):
    raw = at(report, pointer)
    return {"pointer": pointer, "raw": raw, "scale": 100, "display_value": 100 * raw}


def render_table(report):
    _, _, geographic = validate(report)
    rows, mapping = [], []
    for name in (*CLASSES, "macro"):
        label = name.title() + (" AP" if name == "macro" else "")
        row = [label]
        for kind, arm, _, _ in ARMS:
            pointer = geographic[kind, arm, name]
            mean = mapped(report, pointer + "/mean_ap")
            sd = mapped(report, pointer + "/regional_sd")
            text = rf"${mean['display_value']:.2f} \pm {sd['display_value']:.2f}$"
            row.append(text)
            mapping.append({"class": name, "kind": kind, "initialization": arm,
                            "text": text, "mean": mean, "sd": sd})
        rows.append(row)
    header = (r"Class & \multicolumn{2}{c}{DINOv2} & \multicolumn{2}{c}{ResNet18} \\" + "\n"
              + r"AP (\%) & Random & Pretrained & Scratch & ImageNet")
    tex = "% " + TABLE_NOTE + "\n" + table("lrrrr", header, rows)
    return tex, mapping


def check_table(report, path):
    expected, _ = render_table(report)
    require(path.read_bytes() == expected.encode(), "Table differs from accepted report mapping")


def markdown_table(mapping):
    lines = [TABLE_NOTE, "", "| Class | DINOv2 random | DINOv2 pretrained | ResNet18 scratch | ResNet18 ImageNet |",
             "|---|---:|---:|---:|---:|"]
    for c, name in enumerate((*CLASSES, "macro")):
        label = name.title() + (" AP" if name == "macro" else "")
        values = [m["text"].strip("$").replace(r"\pm", "±") for m in mapping[4*c:4*c+4]]
        lines.append("| " + " | ".join([label, *values]) + " |")
    return "\n".join(lines) + "\n"


def figure_data(report):
    """Only convert saved values to percent and assign deterministic seed x offsets."""
    metrics, regional, _ = validate(report)
    panels = []
    for fold in range(5):
        points, means = [], []
        for arm_index, (kind, arm, seeds, _) in enumerate(ARMS):
            for j, seed in enumerate(seeds):
                pointer = metrics[kind, fold, arm, seed, "craters"]
                row = at(report, pointer)
                points.append({"cell": row["cell"], "arm_index": arm_index,
                               "x": arm_index + .22 * (j - (len(seeds) - 1) / 2),
                               "ap": mapped(report, pointer + "/ap"),
                               "single_positive": row["single_positive"]})
            pointer = regional[kind, arm, "craters", fold]
            means.append({"kind": kind, "initialization": arm, "x": arm_index,
                          "ap": mapped(report, pointer + "/seed_mean_ap")})
        pointer = metrics["dino_c1", fold, "random", 42, "craters"]
        row = at(report, pointer)
        panels.append({"region": fold + 1, "n": row["n"], "positives": row["positives"],
                       "support_pointer": pointer, "points": points, "means": means,
                       "reference": mapped(report, pointer + "/prevalence_all_tied_reference")})
    return panels


def figure_caption(panels):
    text = (
        "Craters average precision (AP, %) in five held-out geographic regions. "
        "All panels share a logarithmic axis from 0.01% to 50%; this axis was chosen "
        "after inspecting the results to show low AP values and the Region 2 extreme "
        "on one common scale. Symbols show every saved seed (60 points); solid bars "
        "show the 20 saved arithmetic seed means, without log averaging. "
        "The four arms occupy the same left-to-right positions as the shared legend. "
        "Horizontal offsets identify seeds in increasing order: DINOv2 random 42–46, "
        "DINOv2 pretrained 42, and each ResNet18 arm 0–2. Vertical positions are not jittered. "
        "DINOv2 uses matched encoder features with logistic regression; ResNet18 uses full training. "
        "N is the number of test tiles and K the number labeled Craters. "
        "Dashed lines mark 100K/N on the percent axis, the AP of an all-tied ranking, "
        "not the finite-sample expected AP of a random ranking. "
    )
    for panel in panels:
        if panel["positives"] == 1:
            text += f"Region {panel['region']} has only one positive. "
        for point in panel["points"]:
            single = point["single_positive"]
            if panel["region"] == 2 and single and single["rank_first"] == single["rank_last"] == 3:
                arm = ARMS[point["arm_index"]][3].replace("\n", " ")
                text += (f"In Region 2, {arm} seed {point['cell'][3]} ranks the sole positive third "
                         f"without a tie (AP {point['ap']['display_value']:.2f}%). ")
    return text + (
        "With one positive, grouped AP is the reciprocal of its tie-group end rank. "
        "Seeds do not add independent geographic regions. These scores describe map-label "
        "agreement, not independent geological accuracy. No test threshold was selected; "
        "scores were not pooled across folds, and between-fold calibration is not established.\n"
    )


def draw_figure(panels, variant):
    require(variant in ("cvpr", "full"), "Unknown font variant")
    values = [item["ap"]["display_value"] for panel in panels for item in (*panel["points"], *panel["means"])]
    values.extend(panel["reference"]["display_value"] for panel in panels)
    require(all(math.isfinite(v) and v > 0 for v in values),
            "Log display requires strictly positive finite AP and K/N; no zero substitution")
    roman.configure()
    if variant == "cvpr":
        sans.configure()
    figure = Figure(figsize=(6.5, 2.7), dpi=180)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 5, sharey=True)
    figure.subplots_adjust(left=.09, right=.985, bottom=.25, top=.76, wspace=.16)
    records, lines, annotations = [], [], []

    def retain(line, role, region, source):
        records.append({"role": role, "region": region, "source": source})
        lines.append(line)

    for ax, panel in zip(axes.flat, panels):
        region = panel["region"]
        ax.set(xlim=(-.65, 3.65), yscale="log", ylim=LOG_LIMITS)
        ax.set_yticks(LOG_TICKS, ["0.01", "0.1", "1", "10", "50"])
        ax.minorticks_off()
        ax.set_xticks([])
        ax.grid(axis="y", color="#dddddd", linewidth=.4)
        if region == 1:
            ax.set_ylabel("AP (%)")
        count = f"N = {panel['n']:,}\nK = {panel['positives']}"
        ax.set_title(f"Region {region}\n{count}", pad=6, fontsize=9)
        ref = panel["reference"]["display_value"]
        line, = ax.plot([-.5, 3.5], [ref, ref], color="#777777", ls="--", lw=.8, zorder=1)
        retain(line, "all_tied_reference", region, panel["reference"])
        for mean in panel["means"]:
            x, y = mean["x"], mean["ap"]["display_value"]
            line, = ax.plot([x - .32, x + .32], [y, y], color=roman.INK, lw=1, zorder=2)
            retain(line, "seed_mean", region, mean)
        for point in panel["points"]:
            a, y = point["arm_index"], point["ap"]["display_value"]
            line, = ax.plot([point["x"]], [y], linestyle="none", marker=MARKERS[a],
                            color=roman.FOLD_COLORS[a], ms=3.2, markeredgecolor="white",
                            markeredgewidth=.3, zorder=3)
            retain(line, "seed", region, point)
            single = point["single_positive"]
            if region == 2 and single and single["rank_first"] == single["rank_last"] == 3:
                annotations.append(ax.annotate("Rank 3", xy=(point["x"], y), xytext=(-6, 0),
                                              textcoords="offset points", ha="right", va="center"))
    handles = [Line2D([], [], marker=MARKERS[a], color=roman.FOLD_COLORS[a], linestyle="none",
                      markersize=4, markeredgecolor="white", markeredgewidth=.3) for a in range(4)]
    legend = figure.legend(handles, [a[3] for a in ARMS], loc="lower center", bbox_to_anchor=(.54, .01),
                           ncol=4, columnspacing=1.5, handletextpad=.35, handlelength=1, frameon=False)
    # Use the existing pinned faces; keep every exported callout at regular 9 pt.
    if variant == "cvpr":
        qa = sans.prepare_figure(figure, "class-ranking")
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
    headings = [*figure.texts, *(ax.title for ax in list(axes.flat)[:5])]
    overlaps = [(a.get_text(), b.get_text()) for i, a in enumerate(headings) for b in headings[i+1:]
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    require(not overlaps, f"Overlapping headings: {overlaps}")
    # Annotation.get_window_extent includes the arrow. Use Text's box to check
    # that the callout text itself cannot conceal a seed, mean or reference.
    annotation_boxes = [Text.get_window_extent(a, renderer) for a in annotations]
    qa["overlapping_headings"] = overlaps
    qa["annotation_text_bounds_px"] = [list(b.extents) for b in annotation_boxes]
    qa["annotations"] = [{"text": a.get_text(), "xy": list(a.xy),
                           "offset_points": list(a.get_position())} for a in annotations]
    qa["legend"] = [{"label": t.get_text(), "marker": h.get_marker(), "color": h.get_color()}
                    for h, t in zip(legend.legend_handles, legend.get_texts())]
    # Retain actual artist coordinates, not merely intended plot inputs.
    for record, line in zip(records, lines):
        x, y = list(map(float, line.get_xdata())), list(map(float, line.get_ydata()))
        record.update(x=x, y=y, marker=line.get_marker(), color=line.get_color())
        # Include marker extent when checking data clipping, with all artists drawn.
        margin = line.get_markersize() * figure.dpi / 144 if record["role"] == "seed" else 0
        bounds = line.axes.bbox
        pixels = line.get_transform().transform(list(zip(x, y)))
        record["inside_axes"] = all(bounds.x0 + margin <= px <= bounds.x1 - margin
                                    and bounds.y0 + margin <= py <= bounds.y1 - margin for px, py in pixels)
        padding = margin if record["role"] == "seed" else line.get_linewidth() * figure.dpi / 144
        x0, y0 = pixels.min(axis=0)
        x1, y1 = pixels.max(axis=0)
        record["clear_of_annotation_text"] = all(
            x1 + padding < box.x0 or x0 - padding > box.x1 or
            y1 + padding < box.y0 or y0 - padding > box.y1 for box in annotation_boxes)
        require(record["clear_of_annotation_text"], f"Annotation text overlaps {record['role']}")
    require(all(r["inside_axes"] for r in records), "Clipped data artist")
    qa["axes"] = [{"region": panel["region"], "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                   "yscale": ax.get_yscale(), "yticks": list(ax.get_yticks())}
                  for ax, panel in zip(axes.flat, panels)]
    return figure, {"artists": records, "qa": qa}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--check-table", type=Path)
    args = parser.parse_args(argv)
    report = read_report(args.report, args.report_sha256)
    if args.check_table:
        check_table(report, args.check_table)
        print("Table matches the complete saved aggregate mapping.")
        return
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Use a private directory outside the checkout")
    output.mkdir(parents=True, exist_ok=False)
    binding = {"report": str(args.report.resolve()), "report_sha256": args.report_sha256,
               "scientific_input": report["input"], "source_sha256": digest(Path(__file__)),
               "python_executable": sys.executable, "python_version": sys.version,
               "matplotlib": matplotlib.__version__, "numpy": version("numpy"),
               "style_sha256": {str(p.relative_to(ROOT)): digest(p)
                                for p in (Path(sans.__file__), Path(roman.__file__))}}
    write_json(output / "started.json", binding)
    try:
        tex, mapping = render_table(report)
        (output / "class-ranking.tex").write_text(tex)
        (output / "class-ranking.md").write_text(markdown_table(mapping))
        panels, variants = figure_data(report), {}
        (output / "caption.md").write_text(figure_caption(panels))
        for variant in ("cvpr", "full"):
            figure, evidence = draw_figure(panels, variant)
            roman.save_figure(figure, output / f"craters-ap-{variant}")
            variants[variant] = evidence
        write_json(output / "artist-data.json", {"units": "AP percent; raw report fractions retained",
                   "table_note": TABLE_NOTE, "table": mapping, "panels": panels, "variants": variants})
        write_json(output / "complete.json", {**binding, "status": "rendered",
                   "seed_points_per_variant": 60, "table_cells": 28,
                   "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                               for p in sorted(output.iterdir())}})
    except Exception as exc:
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
