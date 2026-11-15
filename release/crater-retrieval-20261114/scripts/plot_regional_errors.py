"""Render regional SGD results from a byte-authenticated saved report.

python -B -m scripts.plot_regional_errors --report results.json \
    --report-sha256 SHA256 --output FRESH_DIRECTORY

Panel A displays the five saved classifier seeds and their mean and range.
Panel B shows eligible class/fold error differences and the saved equal-group mean.
The original absolute endpoint means remain in the unmodified report for callers.
Both full-paper and CVPR PDF/PNG variants are written to a new directory.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import cvpr_figure_style as typography
from scripts import figure_style as style

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dinov2", "dofa", "croma", "random_init")
DISPLAY = ("DINOv2", "DOFA", "CROMA", "Untrained ViT")
CLASSES = ("Plains", "Dunes", "Hummocky", "Labyrinths", "Lakes", "Craters")
WEIGHTING = "equal eligible class/fold strata; average head error within each tile"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "Expected a finite fraction in [0, 1]")


def validate_report(report):
    """Validate display inputs and return eligible groups, without scoring."""
    require(isinstance(report, dict) and report.get("schema_version") == "1.0.0",
            "Report schema differs")
    require(report.get("classes") == list(CLASSES), "Class order differs")
    models = report.get("models")
    require(isinstance(models, dict) and set(models) == set(MODELS), "Model membership differs")
    groups = {}
    try:
        for model in MODELS:
            data = models[model]
            folds = data["folds"]
            require(isinstance(folds, list) and len(folds) == 5, "Expected five model folds")
            for index, fold in enumerate(folds):
                require(type(fold["fold"]) is int and fold["fold"] == index, "Fold order differs")
                values = fold["macro_by_seed"]["f1"]
                require(isinstance(values, list) and len(values) == 5, "Expected five classifier seeds")
                for value in values:
                    fraction(value)
                    require(value <= .5, "Classifier F1 exceeds the fixed 50% display limit")
            context = data["context"]["modal_fraction"]
            fraction(context["low_below"])
            fraction(context["high_at_least"])
            require(context["low_below"] == .75 and context["high_at_least"] == .90
                    and type(context["minimum_each_group"]) is int
                    and context["minimum_each_group"] == 20
                    and context["weighting"] == WEIGHTING, "Modal-fraction definitions differ")
            strata = context["strata"]
            require(isinstance(strata, list), "Invalid class/fold strata")
            seen = set()
            groups[model] = {}
            for row in strata:
                require(isinstance(row, dict), "Invalid class/fold stratum")
                c, f = row["class_index"], row["fold"]
                require(type(c) is int and 0 <= c < len(CLASSES)
                        and type(f) is int and 0 <= f < 5, "Invalid class/fold identity")
                key = (c, f)
                require(key not in seen, "Duplicate class/fold group")
                seen.add(key)
                for bin_name in ("low", "high"):
                    count, error = row[f"{bin_name}_count"], row[f"{bin_name}_error"]
                    require(type(count) is int and count >= 0, "Invalid bin support")
                    if count == 0:
                        require(error is None, "Empty bin must have null error")
                    else:
                        fraction(error)
                eligible = row["eligible"]
                require(type(eligible) is bool and eligible ==
                        (min(row["low_count"], row["high_count"]) >= context["minimum_each_group"]),
                        "Eligibility differs from bin support")
                if eligible:
                    groups[model][key] = row
            summary = context["summary"]
            if summary is None:
                require(not groups[model], "Missing summary for eligible groups")
            else:
                require(isinstance(summary, dict), "Invalid modal-fraction summary")
                count = summary["strata"]
                require(type(count) is int and count == len(groups[model]) and count > 0,
                        "Invalid eligible-group count")
                fraction(summary["low_error"])
                fraction(summary["high_error"])
                difference = summary["low_minus_high"]
                require(type(difference) in (int, float) and math.isfinite(difference)
                        and -1 <= difference <= 1, "Invalid mean difference")
                require(math.isclose(difference, summary["low_error"] - summary["high_error"],
                                     rel_tol=0, abs_tol=1e-12), "Summary endpoints disagree")
        supports = [{key: (row["low_count"], row["high_count"]) for key, row in group.items()}
                    for group in groups.values()]
        require(all(support == supports[0] for support in supports[1:]),
                "Eligible groups or support differ across models")
        folds = report["majority_reference"]["folds"]
        require(isinstance(folds, list) and len(folds) == 5, "Expected five baseline folds")
        for index, fold in enumerate(folds):
            require(type(fold["fold"]) is int and fold["fold"] == index, "Baseline fold order differs")
            fraction(fold["macro"]["f1"])
            require(fold["macro"]["f1"] <= .5, "Baseline F1 exceeds the fixed 50% display limit")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing or malformed regional report fields") from exc
    return groups


def read_report(path, expected_sha256):
    """Authenticate the original bytes before parsing or creating outputs."""
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256), "Invalid SHA256")
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Report SHA256 differs")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), f"Nonfinite JSON value: {value}")
        return result

    report = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant,
                        parse_float=finite_float)
    validate_report(report)
    return report


def draw_figure(report):
    """Build an undrawn regional figure; the caller configures style and export."""
    groups = validate_report(report)
    keys = sorted(groups[MODELS[0]])
    n = len(keys)
    # Preserve row spacing at larger support counts; leave room for the x label.
    height = max(6.0, 6.0 * (n + 4) / 24)
    colors = [style.MODEL_COLORS[model] for model in MODELS]
    fig, (left,right) = plt.subplots(1,2,figsize=(style.WIDTH, height))
    left.set_position([.075, .85 - 2.652 / height, .415, 2.652 / height])
    right.set_position([.715, .13, .26, .72])
    for index,(model,name,color) in enumerate(zip(MODELS,DISPLAY,colors)):
        data = report["models"][model]
        values=np.array([f["macro_by_seed"]["f1"] for f in data["folds"]])*100
        xs=np.arange(5)+(index-1.5)*.22
        left.vlines(xs,values.min(1),values.max(1),colors=color,lw=.65,zorder=1)
        for seed in range(5):
            left.scatter(xs+(seed-2)*.042,values[:,seed],s=2.25,color=color,
                         linewidths=0,zorder=3)
        left.plot(xs,values.mean(1),marker=style.MODEL_MARKERS[model],color=color,
                  label=name,ls="",ms=7.5,mfc="none",mew=.7,zorder=2)
    majority=np.array([f["macro"]["f1"] for f in report["majority_reference"]["folds"]])*100
    left.plot(range(5),majority,color="#666666",ls="",marker="_",ms=12,mew=1.2,
              label="Most frequent training class")
    left.set_xticks(range(5),[str(f + 1) for f in range(5)])
    left.set(xlabel="Test region",ylabel="Six-class macro F1 (%)",ylim=(0,50),xlim=(-.55,4.55))
    left.set_title("A  Macro F1 by test region\nFive classifier seeds per model",loc="left")
    legend = left.legend(loc="upper left",bbox_to_anchor=(0,-.27),fontsize=style.SMALL,ncol=2,frameon=False)
    # Color the legend proxies explicitly; do not change A's open mean artists.
    for model, handle in zip(MODELS, legend.legend_handles):
        handle.set_markerfacecolor(style.MODEL_COLORS[model])
        handle.set_markeredgecolor(style.MODEL_COLORS[model])
        handle.set_markersize(5)
    for label in legend.get_texts():
        if label.get_text() == "DOFA":
            label.set_text("DOFA (13.78)")
    counts = [sum(c == index for c, _ in keys) for index in range(len(CLASSES))]
    craters = "no Craters" if counts[-1] == 0 else f"Craters groups: {counts[-1]}"
    right.set_title(f"B  Error difference by class and region\n{n} groups; ≥20 tiles per bin; {craters}",
                    loc="left", x=-.73, pad=9)
    if n:
        differences = {model: [100 * (groups[model][key]["low_error"] -
                                     groups[model][key]["high_error"]) for key in keys]
                       for model in MODELS}
        means = {model: 100 * report["models"][model]["context"]["modal_fraction"]["summary"]["low_minus_high"]
                 for model in MODELS}
        values = [0, *means.values(), *(value for row in differences.values() for value in row)]
        low = 20 * math.floor((min(values) - 1) / 20)
        high = 20 * math.ceil((max(values) + 1) / 20)
        tick_step = 20 if high - low <= 120 else 50
        right.set(xlim=(low, high), ylim=(n + 3, -1),
                  xlabel="Mixed − high-fraction error\n(percentage points)")
        right.set_xticks(np.arange(math.ceil(low / tick_step) * tick_step, high + 1, tick_step))
        right.set_yticks(range(n), [str(fold + 1) for _, fold in keys])
        right.tick_params(axis="y", length=0, pad=4)
        right.spines["left"].set_visible(False)
        right.axvline(0, color="#999999", linewidth=.65, zorder=1)
        for index, count in enumerate(counts):
            if not count:
                continue
            positions = [i for i, (c, _) in enumerate(keys) if c == index]
            if index % 2 == 0:
                right.axhspan(min(positions) - .5, max(positions) + .5, color="#f0f0f0", zorder=0)
            right.text(-.16, np.mean(positions), f"{CLASSES[index]} ({count})",
                       transform=right.get_yaxis_transform(), va="center", ha="right")
        mean_y = n + 1.6
        right.axhline(n + .4, color="#bbbbbb", linewidth=.6)
        right.text(-.16, mean_y, "Equal-group\nmean", transform=right.get_yaxis_transform(),
                   va="center", ha="right")
        for index, (model, color) in enumerate(zip(MODELS, colors)):
            # The diamond is sqrt(2) taller than the other nominal-size glyphs.
            # Reserve extra space on both sides, including the next group row.
            offset = (-.385, -.155, .075, .345)[index]
            right.plot(differences[model], np.arange(n) + offset, linestyle="none", markersize=2.8,
                       marker=style.MODEL_MARKERS[model], color=color, markeredgewidth=0)
            right.plot([means[model]], [mean_y + (index - 1.5) * .34], linestyle="none", markersize=3.5,
                       marker=style.MODEL_MARKERS[model], color=color, markeredgewidth=0)
    else:
        right.set_axis_off()
        right.text(.5, .5, "No eligible class–region groups\nEqual-group mean unavailable",
                   transform=right.transAxes, va="center", ha="center")
    fig.text(.075, .025, "B: eligible groups per class in parentheses; positive = more errors in mixed tiles.",
             va="bottom")
    fig.suptitle('SGD classifiers; encoder weights unchanged', fontsize=style.TITLE, y=.98)
    return fig


def render_figures(report, output):
    """Export both font variants with the fixed canvas and draw sequence."""
    layouts = {}
    with plt.rc_context():
        style.configure()
        for variant in ("full", "cvpr"):
            with plt.rc_context():
                with plt.rc_context():
                    style.configure()
                    figure = draw_figure(report)
                    plt.close(figure)
                try:
                    if variant == "cvpr":
                        typography.prepare_figure(figure, "regional-errors")
                    bbox = style.centered_bbox(figure)
                    layouts[variant] = typography.text_metrics(figure, bbox)
                    for ext in ("pdf", "png"):
                        options = ({"metadata": {"CreationDate": None, "ModDate": None}}
                                   if ext == "pdf" else {"dpi": 220})
                        figure.savefig(output / f"{variant}-regional-errors.{ext}",
                                       bbox_inches=bbox, pad_inches=0, **options)
                finally:
                    plt.close(figure)
    return layouts


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = read_report(args.report, args.report_sha256)
    require(not args.output.exists() and not args.output.is_symlink(), "Output path already exists")
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Use an output directory outside the checkout")
    binding = {
        "report_sha256": args.report_sha256,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                          (Path(__file__), Path(style.__file__), Path(typography.__file__))},
        "font_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                        [typography.FONT_FILE, *sorted(style.FONT_DIR.glob("*.ttf"))]},
        "python": sys.version, "matplotlib": matplotlib.__version__,
        "numpy": np.__version__, "pillow": version("Pillow"),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "started.json", binding)
    try:
        layouts = render_figures(report, output)
        files = [output / f"{variant}-regional-errors.{ext}"
                 for variant in ("full", "cvpr") for ext in ("pdf", "png")]
        write_json(output / "complete.json", {
            **binding, "status": "rendered", "text_layout": layouts,
            "outputs": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in files},
        })
    except Exception as exc:
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
