"""Synthetic saved contrasts: membership, sign, artists and offline CLI evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import plot_class_contrasts as p
from tests.test_class_ranking_plot import report as base_report, write_report  # noqa: F401


NAMES = ("plains", "dunes", "hummocky", "labyrinths", "lakes", "craters")


@pytest.fixture
def report(base_report):
    """Contrasts deliberately disagree with per-arm and paired-seed sentinels."""
    rows = []
    for kind in ("dino_c1", "final"):
        for fold in range(5):
            for c, name in enumerate((*NAMES, "macro")):
                value = (-.31 + .045 * c + .012 * fold if kind == "dino_c1"
                         else .38 - .05 * c - .009 * fold)
                rows.append({"kind": kind, "fold": fold, "region": fold + 1, "class": name,
                             "pretrained_minus_baseline": value,
                             "paired_seed_differences_0_1_2": None if kind == "dino_c1"
                             else [-.2, .01, .1]})
    rows[0]["pretrained_minus_baseline"] = 0
    rows[4]["pretrained_minus_baseline"] = -.4291240298718917
    rows[66]["pretrained_minus_baseline"] = .4085977390559033
    base_report["summaries"]["contrasts"] = rows
    return base_report


def test_complete_membership_saved_values_sign_and_pointers(report):
    assert len(p.validate_contrasts(report)) == 70
    panels = p.figure_data(report)
    points = [point for panel in panels for point in panel["points"]]
    expected = {(kind, fold, name) for kind in ("dino_c1", "final")
                for fold in range(5) for name in NAMES}
    assert len(points) == 60
    assert {(r["kind"], r["fold"], r["class"]) for r in points} == expected
    assert [r["class"] for r in panels[0]["points"]] == [name for name in NAMES for _ in range(5)]
    for point in points:
        contrast = point["contrast"]
        assert p.at(report, contrast["pointer"]) == contrast["raw"]
        assert contrast["scale"] == 100
        assert contrast["display_value"] == 100 * contrast["raw"]
        assert point["y"] == NAMES.index(point["class"]) + (-.28, -.14, 0, .14, .28)[point["fold"]]
        paired = point["paired_metadata"]
        assert p.at(report, paired["pointer"]) == paired["raw"]
        assert paired["raw"] == (None if point["kind"] == "dino_c1" else [-.2, .01, .1])
    by_key = {(r["kind"], r["fold"], r["class"]): r["contrast"]["display_value"] for r in points}
    assert by_key["dino_c1", 0, "plains"] == 0
    assert by_key["dino_c1", 0, "lakes"] == 100 * -.4291240298718917
    assert by_key["final", 4, "labyrinths"] == 100 * .4085977390559033
    # A change to saved arm means or paired metadata cannot replace the saved contrast.
    report["summaries"]["regional"][0]["seed_mean_ap"] = .99
    report["summaries"]["contrasts"][35]["paired_seed_differences_0_1_2"] = [.8, .7, .6]
    assert [r["contrast"] for panel in p.figure_data(report) for r in panel["points"]] == [
        r["contrast"] for r in points]


def test_row_permutation_rebinds_pointers_and_saved_mutation_propagates(report):
    before = p.figure_data(report)
    report["summaries"]["contrasts"].reverse()
    after = p.figure_data(report)
    for a, b in zip(before, after):
        assert [(r["class"], r["fold"], r["contrast"]["raw"]) for r in a["points"]] == [
            (r["class"], r["fold"], r["contrast"]["raw"]) for r in b["points"]]
        for point in b["points"]:
            assert p.at(report, point["contrast"]["pointer"]) == point["contrast"]["raw"]
    row = next(r for r in report["summaries"]["contrasts"]
               if (r["kind"], r["fold"], r["class"]) == ("dino_c1", 0, "plains"))
    row["pretrained_minus_baseline"] = -.1234
    assert p.figure_data(report)[0]["points"][0]["contrast"]["display_value"] == -.1234 * 100


@pytest.mark.parametrize("mutation", ["missing_class", "missing_macro", "duplicate_class",
    "duplicate_macro", "kind", "class", "fold", "bool_fold", "float_fold", "region",
    "bool_region", "float_region", "missing_value", "missing_paired", "dino_paired",
    "cnn_none", "cnn_short", "cnn_nonfinite", "cnn_bool", "cnn_outside"])
def test_rejects_incomplete_or_invalid_contrast_grid(report, mutation):
    rows = report["summaries"]["contrasts"]
    row = rows[0]
    if mutation == "missing_class": rows.pop(0)
    elif mutation == "missing_macro": rows.pop(6)
    elif mutation == "duplicate_class": rows[1] = copy.deepcopy(row)
    elif mutation == "duplicate_macro": rows[13] = copy.deepcopy(rows[6])
    elif mutation == "kind": row["kind"] = "dofa"
    elif mutation == "class": row["class"] = "Craters"
    elif mutation == "fold": row["fold"] = 5
    elif mutation == "bool_fold": row["fold"] = False
    elif mutation == "float_fold": row["fold"] = 0.0
    elif mutation == "region": row["region"] = 5
    elif mutation == "bool_region": row["region"] = True
    elif mutation == "float_region": row["region"] = 1.0
    elif mutation == "missing_value": del row["pretrained_minus_baseline"]
    elif mutation == "missing_paired": del row["paired_seed_differences_0_1_2"]
    elif mutation == "dino_paired": row["paired_seed_differences_0_1_2"] = [0, 0, 0]
    else:
        rows[35]["paired_seed_differences_0_1_2"] = {
            "cnn_none": None, "cnn_short": [0, 0], "cnn_nonfinite": [0, float("inf"), 0],
            "cnn_bool": [0, True, 0], "cnn_outside": [0, 1.01, 0]}[mutation]
    with pytest.raises(ValueError):
        p.figure_data(report)


@pytest.mark.parametrize("index", [0, 6, 35, 41])  # Both families, plotted and macro.
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0", None, 1.01, -1.01])
def test_every_contrast_must_be_finite_difference(report, index, value):
    report["summaries"]["contrasts"][index]["pretrained_minus_baseline"] = value
    with pytest.raises(ValueError, match="finite AP difference"):
        p.figure_data(report)


@pytest.mark.parametrize("variant", ["sans", "roman"])
def test_actual_sixty_artists_common_linear_axes_regions_and_fonts(report, variant):
    # Even exactly equal contrasts must leave five individually visible region marks.
    for row in report["summaries"]["contrasts"]:
        if row["class"] == "craters":
            row["pretrained_minus_baseline"] = 0
    panels = p.figure_data(report)
    figure, evidence = p.draw_figure(panels, variant)
    records = evidence["artists"]
    assert len(figure.axes) == 2 and len(records) == 60
    for ax, panel in zip(figure.axes, panels):
        points = [line for line in ax.lines if line.get_linestyle() == "None"]
        assert len(points) == 30 and len(ax.lines) == 31  # Only the zero line is added.
        assert [label.get_text() for label in ax.get_yticklabels()] == [name.title() for name in NAMES]
        assert ax.get_title() == panel["title"]
        assert ax.get_xlabel() == "AP difference (percentage points)"
        for line, point in zip(points, panel["points"]):
            assert list(line.get_xdata()) == [point["contrast"]["raw"] * 100]
            assert list(line.get_ydata()) == [point["y"]]
        for a, b in zip(points[-5:-1], points[-4:]):
            ya = a.get_transform().transform((0, a.get_ydata()[0]))[1]
            yb = b.get_transform().transform((0, b.get_ydata()[0]))[1]
            radii = sum(line.get_markersize() + line.get_markeredgewidth() for line in (a, b))
            assert abs(ya - yb) > radii * figure.dpi / 144
    for artist, point in zip(records, [r for panel in panels for r in panel["points"]]):
        assert artist["source"] == point
        assert artist["x"] == [point["contrast"]["display_value"]]
        assert artist["y"] == [point["y"]]
        assert artist["marker"] == ("o", "s", "^", "D", "v")[point["fold"]]
        assert artist["color"] == ("#0077BB", "#009988", "#EE7733", "#EE3377", "#33BBEE")[point["fold"]]
        assert artist["inside_axes"]
    qa = evidence["qa"]
    assert qa["width_inches"] == 6.5 and not qa["clipped_text"]
    assert all(a["xscale"] == "linear" and a["xlim"] == [-50, 50]
               and a["xticks"] == [-40, -20, 0, 20, 40] and a["yticks"] == list(range(6))
               for a in qa["axes"])
    font = p.sans.FONT_FILE if variant == "sans" else p.roman.FONT_DIR / "NimbusRoman-Regular.ttf"
    assert all(t["size_pt"] == 9 and Path(t["font_file"]) == font and t["inside_canvas"]
               for t in qa["texts"])
    assert [entry["region"] for entry in qa["legend"]] == ["1", "2", "3", "4", "5"]
    assert figure.legends[0].get_title().get_text() == "Test region"


@pytest.mark.parametrize("value", [-.51, .51, -.5, .5])
def test_rejects_axis_and_marker_edge_clipping(report, value):
    report["summaries"]["contrasts"][0]["pretrained_minus_baseline"] = value
    with pytest.raises(ValueError, match="Clipped data artist"):
        p.draw_figure(p.figure_data(report), "sans")


def test_cli_rejects_hash_duplicate_keys_nonfinite_and_output_reuse(tmp_path, report):
    path, sha = write_report(tmp_path, report)
    args = ["--report", str(path), "--report-sha256", sha]
    with pytest.raises(FileExistsError):
        p.main([*args, "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="outside the checkout"):
        p.main([*args, "--output", str(p.ROOT / "rejected-class-contrasts")])
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="SHA256"):
        p.main(["--report", str(path), "--report-sha256", "f" * 64, "--output", str(output)])
    for data, match in ((b'{"a":1,"a":2}', "Duplicate JSON"), (b'{"a":NaN}', "Nonfinite")):
        path.write_bytes(data)
        with pytest.raises(ValueError, match=match):
            p.main(["--report", str(path), "--report-sha256", hashlib.sha256(data).hexdigest(),
                    "--output", str(output)])
    assert not output.exists()


def test_cli_exports_bound_artifacts_without_models_fitting_or_array_inputs(tmp_path, report):
    path, sha = write_report(tmp_path, report)
    output = tmp_path / "display"
    # Start fresh so the guard also covers import-time work, not just draw_figure.
    code = '''
import builtins, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'timm', 'sklearn', 'transformers'} or name.startswith('scripts.analyze'):
        raise AssertionError('Unexpected model/scoring import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import numpy
def no_arrays(*args, **kwargs):
    raise AssertionError('Unexpected array input')
numpy.load = no_arrays
from scripts import plot_class_contrasts
plot_class_contrasts.main(sys.argv[1:])
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, "--report", str(path),
                             "--report-sha256", sha, "--output", str(output)],
                            cwd=p.ROOT, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    complete = json.loads((output / "complete.json").read_text())
    assert complete["report_sha256"] == sha
    assert complete["validated_contrast_rows"] == 70 and complete["class_points_per_variant"] == 60
    assert complete["status"] == "rendered"
    for name, entry in complete["outputs"].items():
        assert entry == {"sha256": p.digest(output / name), "bytes": (output / name).stat().st_size}
    assert set(complete["outputs"]) == {"started.json", "artist-data.json",
        "class-contrasts-sans.pdf", "class-contrasts-sans.png",
        "class-contrasts-roman.pdf", "class-contrasts-roman.png"}
    for field in ("source_sha256", "font_sha256"):
        for name, digest in complete[field].items():
            assert p.digest(p.ROOT / name) == digest
    assert p.digest(Path(sys.executable)) == complete["python_executable_sha256"]
    display = json.loads((output / "artist-data.json").read_text())
    assert display["report_sha256"] == sha and display["direction"] == "Positive favors pretrained initialization"
    for variant in ("sans", "roman"):
        assert (output / f"class-contrasts-{variant}.pdf").read_bytes().startswith(b"%PDF")
        assert (output / f"class-contrasts-{variant}.png").read_bytes().startswith(b"\x89PNG")
        assert len(display["variants"][variant]["artists"]) == 60
    assert p.digest(path) == sha


def test_failed_render_preserves_started_and_failure_without_complete(tmp_path, report, monkeypatch):
    path, sha = write_report(tmp_path, report)
    def broken(*args):
        raise RuntimeError("synthetic render failure")
    monkeypatch.setattr(p, "draw_figure", broken)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        p.main(["--report", str(path), "--report-sha256", sha, "--output", str(output)])
    assert (output / "started.json").exists() and (output / "failure.json").exists()
    assert not (output / "complete.json").exists()
