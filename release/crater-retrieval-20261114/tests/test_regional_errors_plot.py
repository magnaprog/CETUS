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
from matplotlib.markers import MarkerStyle
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
        low, high = (.71, .63, .55, .47)[index], (.21, .22, .23, .24)[index]
        result["models"][model] = {
            "folds": [{"fold": f, "macro_by_seed": {"f1": [
                (4 * (index + 1) + 3 * f + s) / 100 for s in (2, 8, 1, 5, 4)]}}
                      for f in range(5)],
            "macro": {"f1": {"mean": .99}},
            "mean_row_confusion": np.eye(6).tolist(),
            "context": {"modal_fraction": {
                "low_below": .75, "high_at_least": .9, "minimum_each_group": 20,
                "weighting": plot.WEIGHTING,
                # Unsorted identities, unequal supports, and a zero-support bin.
                # The saved summary is the equal-group mean, not a tile-weighted mean.
                "strata": [
                    {"class_index": 4, "fold": 3, "low_count": 20, "high_count": 70,
                     "low_error": low - .2, "high_error": high + .1, "eligible": True},
                    {"class_index": 0, "fold": 2, "low_count": 200, "high_count": 2000,
                     "low_error": low + .2, "high_error": high - .2, "eligible": True},
                    {"class_index": 5, "fold": 1, "low_count": 0, "high_count": 8,
                     "low_error": None, "high_error": .5, "eligible": False},
                    {"class_index": 0, "fold": 0, "low_count": 36, "high_count": 20,
                     "low_error": low, "high_error": high + .1, "eligible": True}],
                "summary": {"strata": 3, "low_error": low, "high_error": high,
                            "low_minus_high": low - high}}},
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
        assert len(right.lines) == 10 and len(right.collections) == 0
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
            context = report["models"][model]["context"]["modal_fraction"]
            rows = [context["strata"][i] for i in (3, 1, 0)]
            dots, group_mean = right.lines[2 + 2 * index:4 + 2 * index]
            np.testing.assert_array_equal(dots.get_xdata(), [
                100 * (row["low_error"] - row["high_error"]) for row in rows])
            np.testing.assert_allclose(dots.get_ydata(), np.arange(3) + (-.385, -.155, .075, .345)[index], atol=1e-15, rtol=0)
            np.testing.assert_array_equal(group_mean.get_xdata(), [100 * context["summary"]["low_minus_high"]])
            np.testing.assert_allclose(group_mean.get_ydata(), [4.6 + (index - 1.5) * .34], atol=1e-15, rtol=0)
            weighted = 100 * (np.average([row["low_error"] for row in rows], weights=[row["low_count"] for row in rows])
                              - np.average([row["high_error"] for row in rows], weights=[row["high_count"] for row in rows]))
            assert abs(group_mean.get_xdata()[0] - weighted) > 10
            for artist, size in ((dots, 2.8), (group_mean, 3.5)):
                assert artist.get_color() == color and artist.get_marker() == plot.style.MODEL_MARKERS[model]
                assert artist.get_linestyle() == "None" and artist.get_markersize() == size
                assert artist.get_markeredgewidth() == 0
            key = left.get_legend().legend_handles[index]
            assert key.get_marker() == plot.style.MODEL_MARKERS[model]
            assert key.get_markerfacecolor() == key.get_markeredgecolor() == color
            assert key.get_markersize() == 5
        baseline = left.lines[-1]
        np.testing.assert_array_equal(baseline.get_xdata(), range(5))
        np.testing.assert_array_equal(baseline.get_ydata(), [1, 2, 3, 4, 5])
        assert baseline.get_marker() == "_" and baseline.get_linestyle() == "None"
        assert baseline.get_color() == "#666666"
        assert baseline.get_markersize() == 12 and baseline.get_markeredgewidth() == 1.2
        assert [t.get_text() for t in left.get_xticklabels()] == ["1", "2", "3", "4", "5"]
        assert [t.get_text() for t in right.get_yticklabels()] == ["1", "3", "4"]
        assert [t.get_text() for t in right.texts] == ["Plains (2)", "Lakes (1)", "Equal-group\nmean"]
        assert left.get_ylim() == (0, 50) and left.get_xlim() == (-.55, 4.55)
        assert right.get_xlim() == (-20, 100) and right.yaxis_inverted()
        assert left.get_xlabel() == "Test region" and left.get_ylabel() == "Six-class macro F1 (%)"
        assert left.get_title(loc="left") == "A  Macro F1 by test region\nFive classifier seeds per model"
        assert right.get_title(loc="left") == "B  Error difference by class and region\n3 groups; ≥20 tiles per bin; no Craters"
        assert right.get_xlabel() == "Mixed − high-fraction error\n(percentage points)"
        assert [t.get_text() for t in left.get_legend().get_texts()] == [
            "DINOv2", "DOFA (13.78)", "CROMA", "Untrained ViT", "Most frequent training class"]
        assert right.get_legend() is None
        assert fig._suptitle.get_text() == "SGD classifiers; encoder weights unchanged"
        np.testing.assert_array_equal(fig.get_size_inches(), [6.5, 6.0])
        assert report == before
    finally:
        plt.close(fig)


def test_missing_support_branch(report):
    for data in report["models"].values():
        context = data["context"]["modal_fraction"]
        context["summary"] = None
        context["strata"] = [row for row in context["strata"] if not row["eligible"]]
    plot.validate_report(report)
    fig = plot.draw_figure(report)
    try:
        right = fig.axes[1]
        assert len(right.lines) == len(right.collections) == 0
        assert not right.axison
        assert [text.get_text() for text in right.texts] == [
            "No eligible class–region groups\nEqual-group mean unavailable"]
        assert right.get_title(loc="left").endswith("0 groups; ≥20 tiles per bin; no Craters")
    finally:
        plt.close(fig)


@pytest.mark.parametrize("keys", [
    [(5, 4)], [(0, 3), (2, 1), (5, 4)],
    [(c, f) for c in (0, 2, 4, 5) for f in range(5)],
    [(c, f) for c in range(6) for f in range(5)],
])
def test_layout_uses_actual_groups_and_signed_endpoints(report, keys):
    for model_index, data in enumerate(report["models"].values()):
        context = data["context"]["modal_fraction"]
        rows = [{"class_index": c, "fold": f, "low_count": 20 + i, "high_count": 40 + i,
                 "low_error": float(i % 2), "high_error": float(1 - i % 2), "eligible": True}
                for i, (c, f) in enumerate(keys)]
        # Every model may serialize the same groups in a different order.
        context["strata"] = rows[model_index:] + rows[:model_index]
        low, high = np.mean([row["low_error"] for row in rows]), np.mean([row["high_error"] for row in rows])
        context["summary"] = {"strata": len(keys), "low_error": float(low), "high_error": float(high),
                              "low_minus_high": float(low - high)}
    before = copy.deepcopy(report)
    with plt.rc_context():
        plot.style.configure()
        fig = plot.draw_figure(report)
        try:
            left, right = fig.axes
            assert [t.get_text() for t in right.get_yticklabels()] == [str(f + 1) for _, f in keys]
            counts = [sum(c == i for c, _ in keys) for i in range(6)]
            assert [t.get_text() for t in right.texts] == [
                f"{name} ({n})" for name, n in zip(plot.CLASSES, counts) if n] + ["Equal-group\nmean"]
            assert right.get_title(loc="left").endswith(f"{len(keys)} groups; ≥20 tiles per bin; Craters groups: {counts[-1]}")
            for index in range(4):
                points = right.lines[2 + 2 * index]
                np.testing.assert_array_equal(points.get_xdata(), [100 * (2 * (i % 2) - 1) for i in range(len(keys))])
                np.testing.assert_allclose(points.get_ydata(), np.arange(len(keys)) + (-.385, -.155, .075, .345)[index])
                assert right.get_xlim()[0] < min(points.get_xdata()) <= max(points.get_xdata()) < right.get_xlim()[1]
            assert fig.get_figwidth() == 6.5
            np.testing.assert_allclose(left.get_position().height * fig.get_figheight(), 2.652)
            if len(keys) == 30:
                assert fig.get_figheight() > 6.0
            # Bound actual glyph heights, including D's sqrt(2) extent, not nominal sizes.
            # This checks separation even for coincident x values and adjacent groups.
            row_height_pt = right.get_position().height * fig.get_figheight() * 72 / (len(keys) + 4)
            spans = []
            for artist in right.lines[2:]:
                marker = MarkerStyle(artist.get_marker())
                glyph_height = marker.get_path().transformed(marker.get_transform()).get_extents().height
                half_height = glyph_height * artist.get_markersize() / 2
                spans.extend((y * row_height_pt - half_height, y * row_height_pt + half_height)
                             for y in artist.get_ydata())
            spans.sort()
            assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
            plot.typography.prepare_figure(fig, "regional-errors")
            layout = plot.typography.text_metrics(fig, plot.style.centered_bbox(fig))
            assert not layout["clipped_text"]
            assert report == before
        finally:
            plt.close(fig)


@pytest.mark.parametrize("field,value", [
    ("class_index", True), ("class_index", -1), ("class_index", 6),
    ("fold", 1.0), ("fold", 5),
    ("low_count", True), ("low_count", 20.0), ("low_count", -1), ("low_count", 19),
    ("high_count", 0), ("high_count", None),
    ("low_error", -.01), ("low_error", 1.01), ("low_error", None),
    ("high_error", True), ("high_error", float("nan")), ("high_error", float("inf")),
    ("eligible", False), ("eligible", 1),
])
def test_invalid_stratum_denied_before_output(report, tmp_path, monkeypatch, field, value):
    report["models"]["dinov2"]["context"]["modal_fraction"]["strata"][0][field] = value
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "output"
    monkeypatch.setattr(plot, "render_figures", lambda *a: pytest.fail("Rendered invalid strata"))
    with pytest.raises(ValueError):
        plot.main(arguments(path, sha, output))
    assert not output.exists()


@pytest.mark.parametrize("case,message", [
    ("duplicate_eligible", "Duplicate"), ("duplicate_ineligible", "Duplicate"),
    ("different_identity", "Eligible groups or support differ"),
    ("different_support", "Eligible groups or support differ"),
    ("empty_bin_error", "Empty bin"), ("empty_bin_eligible", "Eligibility"),
    ("missing_summary", "Missing summary"), ("empty_with_summary", "eligible-group count"),
    ("no_groups_one_model", "Eligible groups or support differ"),
])
def test_group_consistency(report, case, message):
    context = report["models"]["dinov2"]["context"]["modal_fraction"]
    if case.startswith("duplicate"):
        context["strata"].append(copy.deepcopy(context["strata"][0 if case.endswith("_eligible") else 2]))
    elif case == "different_identity":
        context["strata"][0]["fold"] = 4  # Same count, different class/region identities.
    elif case == "different_support":
        context["strata"][0]["high_count"] += 1
    elif case == "empty_bin_error":
        context["strata"][2]["low_error"] = 0
    elif case == "empty_bin_eligible":
        context["strata"][2]["eligible"] = True
    elif case == "missing_summary":
        context["summary"] = None
    else:
        context["strata"] = []
        if case == "no_groups_one_model":
            context["summary"] = None
    with pytest.raises(ValueError, match=message):
        plot.validate_report(report)


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
    (("models", "dinov2", "context", "modal_fraction", "summary", "strata"), 3.0),
    (("models", "dinov2", "context", "modal_fraction", "summary", "low_minus_high"), .4),
    (("models", "dinov2", "context", "modal_fraction", "summary", "low_minus_high"), 1.01),
    (("models", "dinov2", "context", "modal_fraction", "summary", "low_minus_high"), True),
    (("models", "dinov2", "context", "modal_fraction", "summary", "low_minus_high"), float("inf")),
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
        np.testing.assert_allclose([options["bbox_inches"].width, options["bbox_inches"].height], [6.5, 6.0])
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
