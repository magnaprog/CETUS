"""Render regional SGD results from a byte-authenticated saved report.

python -B -m scripts.plot_regional_errors --report results.json \
    --report-sha256 SHA256 --output FRESH_DIRECTORY

Panel A displays the five saved classifier seeds and their mean and range.
Panel B copies the saved error summaries without reaggregating class/fold groups.
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
from matplotlib.lines import Line2D
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
    """Validate consumed fields and fixed label definitions, without scoring."""
    require(isinstance(report, dict) and report.get("schema_version") == "1.0.0",
            "Report schema differs")
    require(report.get("classes") == list(CLASSES), "Class order differs")
    models = report.get("models")
    require(isinstance(models, dict) and set(models) == set(MODELS), "Model membership differs")
    group_counts = []
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
            summary = context["summary"]
            if summary is None:
                group_counts.append(0)
            else:
                require(isinstance(summary, dict), "Invalid modal-fraction summary")
                count = summary["strata"]
                require(type(count) is int and 1 <= count <= 30, "Invalid eligible-group count")
                group_counts.append(count)
                fraction(summary["low_error"])
                fraction(summary["high_error"])
        require(len(set(group_counts)) == 1, "Eligible-group counts differ across models")
        folds = report["majority_reference"]["folds"]
        require(isinstance(folds, list) and len(folds) == 5, "Expected five baseline folds")
        for index, fold in enumerate(folds):
            require(type(fold["fold"]) is int and fold["fold"] == index, "Baseline fold order differs")
            fraction(fold["macro"]["f1"])
            require(fold["macro"]["f1"] <= .5, "Baseline F1 exceeds the fixed 50% display limit")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing or malformed regional report fields") from exc


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
    colors = [style.MODEL_COLORS[model] for model in MODELS]
    fig, (left,right) = plt.subplots(1,2,figsize=(style.WIDTH, 4.6),layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
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
        context=data["context"]["modal_fraction"]["summary"]
        if context:
            low, high=100*context["low_error"],100*context["high_error"]
            right.plot([low,high],[index,index],color=color,lw=1.8)
            right.scatter(low,index,marker="o",s=48,facecolors="white",edgecolors=style.INK,linewidths=.5,zorder=3)
            right.scatter(high,index,marker="o",s=40,color=color,edgecolors=style.INK,linewidths=.5,zorder=3)
        else:
            right.text(.02,index,"Insufficient matched support",transform=right.get_yaxis_transform())
    majority=np.array([f["macro"]["f1"] for f in report["majority_reference"]["folds"]])*100
    left.plot(range(5),majority,color="#666666",ls="",marker="_",ms=12,mew=1.2,
              label="Most frequent training class")
    left.set_xticks(range(5),[str(f + 1) for f in range(5)])
    left.set(xlabel="Test region",ylabel="Six-class macro F1 (%)",ylim=(0,50),xlim=(-.55,4.55))
    left.set_title("A  Macro F1 by test region\nFive classifier seeds per model",loc="left")
    left.legend(loc="upper left",bbox_to_anchor=(0,-.27),fontsize=style.SMALL,ncol=2,frameon=False)
    right.set_yticks(range(4),DISPLAY)
    right.invert_yaxis()
    right.set(xlabel="Mean error rate across\nclass and fold groups (%)",xlim=(0,100))
    right.legend(handles=[Line2D([],[],marker="o",mfc="white",mec="#333",ls="",label="Modal fraction < 0.75"),
                          Line2D([],[],marker="o",color="#333",ls="",label="Modal fraction ≥ 0.90")],
                 loc="upper left",bbox_to_anchor=(0,-.27),fontsize=style.SMALL,frameon=False)
    selected=report["models"]["dinov2"]["context"]["modal_fraction"]["summary"]
    n=selected["strata"] if selected else 0
    right.set_title(f"B  Error by modal map fraction\n{n} eligible class and region groups", loc="left")
    fig.suptitle('SGD classifiers; encoder weights unchanged', fontsize=style.TITLE)
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
