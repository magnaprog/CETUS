"""Synthetic checks of saved regional values, rendering and input boundaries."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import numpy as np
import pytest

from scripts import plot_regional_errors as plot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report():
    result = {"schema_version": "1.0.0", "classes": list(plot.CLASSES), "models": {},
              "majority_reference": {"folds": [
                  {"fold": f, "macro": {"f1": (f + 1) / 100}} for f in range(5)]},
              "test_support": [10, 20, 30, 40, 50, 60]}
    for index, model in enumerate(plot.MODELS):
        result["models"][model] = {
            "folds": [{"fold": f, "macro_by_seed": {"f1": [
                (4 * (index + 1) + 3 * f + s) / 100 for s in (2, 8, 1, 5, 4)]}}
                      for f in range(5)],
            "macro": {"f1": {"mean": .99}},
            "mean_row_confusion": np.eye(6).tolist(),
            "context": {"modal_fraction": {
                "low_below": .75, "high_at_least": .9, "minimum_each_group": 20,
                "weighting": plot.WEIGHTING,
                # Deliberately different from summary: the renderer must use summary.
                "strata": [{"low_error": .01, "high_error": .99}],
                "summary": {"strata": 3, "low_error": (.71, .63, .55, .47)[index],
                            "high_error": (.21, .22, .23, .24)[index]}}},
        }
    return result


def write_report(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return path, plot.digest(path)


def arguments(path, sha, output):
    return ["--report", str(path), "--report-sha256", sha, "--output", str(output)]


def test_exact_artists_and_undrawn_figure(report, monkeypatch):
    before = copy.deepcopy(report)
    monkeypatch.setattr(FigureCanvasAgg, "draw", lambda *a, **k: pytest.fail("Unexpected draw"))
    with plt.rc_context():
        plot.style.configure()
        fig = plot.draw_figure(report)
    try:
        assert tuple(plot.MODELS) == ("dinov2", "dofa", "croma", "random_init")
        assert tuple(plot.DISPLAY) == ("DINOv2", "DOFA", "CROMA", "Untrained ViT")
        assert tuple(plot.CLASSES) == ("Plains", "Dunes", "Hummocky", "Labyrinths", "Lakes", "Craters")
        left, right = fig.axes
        assert len(left.lines) == 5 and len(left.collections) == 24
        assert len(right.lines) == 4 and len(right.collections) == 8
        for index, (model, shift) in enumerate(zip(plot.MODELS, (-.33, -.11, .11, .33))):
            centers = np.arange(5) + shift
            base = 4 * (index + 1) + 3 * np.arange(5)
            color = plot.style.MODEL_COLORS[model]
            values = np.array([f["macro_by_seed"]["f1"] for f in report["models"][model]["folds"]]) * 100
            mean = left.lines[index]
            np.testing.assert_allclose(mean.get_xdata(), centers, atol=1e-15, rtol=0)
            np.testing.assert_array_equal(mean.get_ydata(), values.mean(1))
            np.testing.assert_allclose(mean.get_ydata(), base + 4, atol=1e-14, rtol=0)
            assert mean.get_linestyle() == "None" and mean.get_markerfacecolor() == "none"
            assert mean.get_marker() == plot.style.MODEL_MARKERS[model]
            assert mean.get_color() == color and mean.get_markersize() == 7.5
            assert mean.get_markeredgewidth() == .7 and mean.get_zorder() == 2
            ranges = left.collections[6 * index]
            np.testing.assert_allclose(ranges.get_segments(), [
                [[x, b + 1], [x, b + 8]] for x, b in zip(centers, base)], atol=1e-14, rtol=0)
            np.testing.assert_array_equal(ranges.get_colors(), [to_rgba(color)])
            np.testing.assert_array_equal(ranges.get_linewidths(), [.65])
            assert ranges.get_zorder() == 1
            for seed, offset in enumerate((-.084, -.042, 0, .042, .084)):
                dots = left.collections[6 * index + seed + 1]
                np.testing.assert_allclose(dots.get_offsets()[:, 0], centers + offset, atol=1e-15, rtol=0)
                np.testing.assert_array_equal(dots.get_offsets()[:, 1], values[:, seed])
                np.testing.assert_array_equal(dots.get_facecolors(), [to_rgba(color)])
                np.testing.assert_array_equal(dots.get_sizes(), [2.25])
                np.testing.assert_array_equal(dots.get_linewidths(), [0])
                assert dots.get_zorder() == 3
            low, high = (71, 63, 55, 47)[index], (21, 22, 23, 24)[index]
            np.testing.assert_allclose(right.lines[index].get_xdata(), [low, high], atol=1e-14, rtol=0)
            np.testing.assert_array_equal(right.lines[index].get_ydata(), [index, index])
            assert right.lines[index].get_color() == color
            for offset, value in enumerate((low, high)):
                dots = right.collections[2 * index + offset]
                np.testing.assert_allclose(dots.get_offsets(), [[value, index]], atol=1e-14, rtol=0)
                np.testing.assert_array_equal(dots.get_facecolors(), [to_rgba("white" if offset == 0 else color)])
                np.testing.assert_array_equal(dots.get_edgecolors(), [to_rgba(plot.style.INK)])
                np.testing.assert_array_equal(dots.get_sizes(), [48 if offset == 0 else 40])
        baseline = left.lines[-1]
        np.testing.assert_array_equal(baseline.get_xdata(), range(5))
        np.testing.assert_array_equal(baseline.get_ydata(), [1, 2, 3, 4, 5])
        assert baseline.get_marker() == "_" and baseline.get_linestyle() == "None"
        assert baseline.get_color() == "#666666"
        assert baseline.get_markersize() == 12 and baseline.get_markeredgewidth() == 1.2
        assert [t.get_text() for t in left.get_xticklabels()] == ["1", "2", "3", "4", "5"]
        assert [t.get_text() for t in right.get_yticklabels()] == list(plot.DISPLAY)
        assert left.get_ylim() == (0, 50) and left.get_xlim() == (-.55, 4.55)
        assert right.get_xlim() == (0, 100) and right.yaxis_inverted()
        assert left.get_xlabel() == "Test region" and left.get_ylabel() == "Six-class macro F1 (%)"
        assert left.get_title(loc="left") == "A  Macro F1 by test region\nFive classifier seeds per model"
        assert right.get_title(loc="left") == "B  Error by modal map fraction\n3 eligible class and region groups"
        assert right.get_xlabel() == "Mean error rate across\nclass and fold groups (%)"
        assert [t.get_text() for t in left.get_legend().get_texts()] == [*plot.DISPLAY, "Most frequent training class"]
        assert [t.get_text() for t in right.get_legend().get_texts()] == ["Modal fraction < 0.75", "Modal fraction ≥ 0.90"]
        assert fig._suptitle.get_text() == "SGD classifiers; encoder weights unchanged"
        np.testing.assert_array_equal(fig.get_size_inches(), [6.5, 4.6])
        assert report == before
    finally:
        plt.close(fig)


def test_missing_support_branch(report):
    for data in report["models"].values():
        data["context"]["modal_fraction"]["summary"] = None
    plot.validate_report(report)
    fig = plot.draw_figure(report)
    try:
        right = fig.axes[1]
        assert len(right.lines) == len(right.collections) == 0
        assert [text.get_text() for text in right.texts] == ["Insufficient matched support"] * 4
        assert right.get_title(loc="left").endswith("0 eligible class and region groups")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("field,value", [
    (("schema_version",), "2.0.0"), (("classes", 0), "Dunes"),
    (("models", "dinov2", "folds", 0, "fold"), True),
    (("models", "dinov2", "folds", 0, "fold"), 1),
    (("models", "dinov2", "folds"), []),
    (("models", "dinov2", "folds", 0, "macro_by_seed", "f1"), [0] * 4),
    (("models", "dinov2", "folds", 0, "macro_by_seed", "f1", 0), True),
    (("models", "dinov2", "folds", 0, "macro_by_seed", "f1", 0), 1.01),
    (("models", "dinov2", "folds", 0, "macro_by_seed", "f1", 0), .5001),
    (("models", "dinov2", "context", "modal_fraction", "summary", "low_error"), -.01),
    (("models", "dinov2", "context", "modal_fraction", "summary", "strata"), 4),
    (("models", "dinov2", "context", "modal_fraction", "low_below"), .7),
    (("models", "dinov2", "context", "modal_fraction", "weighting"), "pooled tiles"),
    (("models", "dinov2", "context", "modal_fraction", "minimum_each_group"), 20.0),
    (("majority_reference", "folds", 0, "fold"), False),
    (("majority_reference", "folds", 0, "macro", "f1"), None),
    (("majority_reference", "folds", 0, "macro", "f1"), .5001),
])
def test_schema_denied_before_output(report, tmp_path, monkeypatch, field, value):
    target = report
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    monkeypatch.setattr(plot, "render_figures", lambda *a: pytest.fail("Rendered invalid input"))
    with pytest.raises(ValueError):
        plot.main(arguments(path, sha, output))
    assert not output.exists()


def test_missing_model_and_fields(report):
    del report["models"]["dofa"]
    with pytest.raises(ValueError, match="Model membership"):
        plot.validate_report(report)
    report["models"]["dofa"] = {}
    with pytest.raises(ValueError, match="Missing or malformed"):
        plot.validate_report(report)


@pytest.mark.parametrize("raw", [b'{"schema_version":1,"schema_version":2}', b'{"x":NaN}',
                                   b'{"x":Infinity}', b'{"x":1e999}'])
def test_json_rejection_before_output(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Duplicate|Nonfinite"):
        plot.main(arguments(path, plot.digest(path), output))
    assert not output.exists()


def test_hash_precedes_parsing_and_rendering(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"not JSON")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Report SHA256 differs"):
        plot.main(arguments(path, "0" * 64, output))
    assert not output.exists()
    with pytest.raises(ValueError, match="Invalid SHA256"):
        plot.read_report(path, "0")


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_existing_output_denied(report, tmp_path, monkeypatch, kind):
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    target = tmp_path / "target"
    if kind == "directory":
        output.mkdir()
        (output / "keep").write_bytes(b"keep")
    elif kind == "file":
        output.write_bytes(b"keep")
    else:
        if kind == "symlink":
            target.mkdir()
        output.symlink_to(target)
    monkeypatch.setattr(plot, "render_figures", lambda *a: pytest.fail("Overwrote existing output"))
    with pytest.raises(ValueError, match="Output path already exists"):
        plot.main(arguments(path, sha, output))
    if kind == "directory":
        assert (output / "keep").read_bytes() == b"keep"
        assert len(list(output.iterdir())) == 1
    elif kind == "file":
        assert output.read_bytes() == b"keep"
    else:
        assert output.is_symlink()
        assert target.is_dir() if kind == "symlink" else not target.exists()


def test_output_inside_checkout_denied(report, tmp_path, monkeypatch):
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    monkeypatch.setattr(plot, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside the checkout"):
        plot.main(arguments(path, sha, output))
    assert not output.exists()


def test_export_failure_keeps_partial_output(report, tmp_path, monkeypatch):
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    calls = []
    before = set(plt.get_fignums())

    def fail_on_png(figure, destination, **options):
        calls.append(Path(destination).name)
        if str(destination).endswith(".png"):
            raise OSError("synthetic export failure")
        Path(destination).write_bytes(b"partial PDF")

    monkeypatch.setattr(Figure, "savefig", fail_on_png)
    with pytest.raises(OSError, match="synthetic export failure"):
        plot.main(arguments(path, sha, output))
    assert calls == ["full-regional-errors.pdf", "full-regional-errors.png"]
    assert (output / "full-regional-errors.pdf").read_bytes() == b"partial PDF"
    assert (output / "started.json").exists() and not (output / "complete.json").exists()
    assert json.loads((output / "failure.json").read_text())["error"] == "OSError"
    assert set(plt.get_fignums()) == before


def test_export_draw_sequence_and_options(report, tmp_path, monkeypatch):
    events = []
    figures = []
    original_draw = FigureCanvasAgg.draw
    original_build = plot.draw_figure
    original_prepare = plot.typography.prepare_figure
    original_center = plot.style.centered_bbox
    original_metrics = plot.typography.text_metrics
    original_family = list(plt.rcParams["font.family"])

    def build(value):
        events.append("build")
        figure = original_build(value)
        figures.append(figure)
        return figure

    def canvas_draw(canvas):
        events.append("draw")
        return original_draw(canvas)

    def prepare(figure, name):
        events.append("prepare")
        assert plt.rcParams["font.family"] == [plot.style.FONT]
        assert name == "regional-errors"
        return original_prepare(figure, name)

    def center(figure):
        events.append("center")
        return original_center(figure)

    def metrics(figure, *args):
        events.append("metrics")
        return original_metrics(figure, *args)

    def save(figure, destination, **options):
        events.append(Path(destination).suffix[1:])
        assert figure is figures[-1]
        assert options["pad_inches"] == 0
        np.testing.assert_allclose([options["bbox_inches"].width, options["bbox_inches"].height], [6.5, 4.6])
        if Path(destination).suffix == ".pdf":
            assert options["metadata"] == {"CreationDate": None, "ModDate": None}
            assert "dpi" not in options
        else:
            assert options["dpi"] == 220 and "metadata" not in options

    monkeypatch.setattr(plot, "draw_figure", build)
    monkeypatch.setattr(FigureCanvasAgg, "draw", canvas_draw)
    monkeypatch.setattr(plot.typography, "prepare_figure", prepare)
    monkeypatch.setattr(plot.style, "centered_bbox", center)
    monkeypatch.setattr(plot.typography, "text_metrics", metrics)
    monkeypatch.setattr(Figure, "savefig", save)
    plot.render_figures(report, tmp_path)
    assert events == ["build", "center", "draw", "metrics", "pdf", "png",
                      "build", "prepare", "draw", "draw", "metrics", "center", "draw", "metrics", "pdf", "png"]
    assert len(figures) == 2 and figures[0] is not figures[1]
    assert plt.rcParams["font.family"] == original_family


def test_synthetic_cli_and_lightweight_imports(report, tmp_path):
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-B", "-m", "scripts.plot_regional_errors",
                             *arguments(path, sha, output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    complete = json.loads((output / "complete.json").read_text())
    assert complete["status"] == "rendered" and complete["report_sha256"] == sha
    assert len(complete["outputs"]) == 4
    for name, record in complete["outputs"].items():
        content = (output / name).read_bytes()
        assert content.startswith(b"%PDF-" if name.endswith(".pdf") else b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        assert len(content) == record["bytes"]
    assert path.read_text() == json.dumps(report)
    forbidden = ["scripts.render_cvpr_figures", "scripts.generate_figures", "scripts.analyze_repaired_predictions",
                 "scripts.render_full_paper_figures", "scipy", "sklearn", "torch"]
    check = subprocess.run([sys.executable, "-B", "-c",
                            "import sys; from scripts import plot_regional_errors; "
                            f"assert not set({forbidden!r}) & sys.modules.keys()"],
                           cwd=ROOT, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
