"""Saved endpoints only: synthetic count identities, artists and CLI boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import plot_crater_retrieval as p
from tests.test_class_ranking_plot import report  # noqa: F401 -- shared full-report fixture


@pytest.fixture
def ranking(report):
    """Unequal group counts, first-group ties/TP, zero recall and an all-tied fit.

    Saved AP and summaries stay sentinel values: the consumer cannot rescore.
    """
    report["craters_curve_index"] = []
    arrays = {}
    rows = {tuple(r["cell"]): r for r in report["metrics"] if r["class_index"] == 5}
    for i, cell in enumerate(sorted(tuple(r["cell"]) for r in report["cells"])):
        row = rows[cell]
        n, k = row["n"], row["positives"]
        counts, tp = (([3, 5, n], [0, 1, k]), ([1, 4, 8, n], [0, 0, min(2, k), k]),
                      ([2, n], [1, k]), ([n], [k]))[i % 4]
        counts, tp = np.array(counts, dtype=np.int64), np.array(tp, dtype=np.int64)
        prefix = f"cell_{i:02d}"
        report["craters_curve_index"].append({"prefix": prefix, "cell": list(cell), "class_index": 5})
        sizes = np.diff(np.r_[0, counts])
        row.update(distinct_scores=len(tp), tied_groups=int((sizes > 1).sum()),
                   largest_tie=int(sizes.max()))
        if k == 1:
            first = int(np.flatnonzero(tp)[0])
            row["single_positive"] = {"rank_first": int(counts[first] - sizes[first] + 1),
                                      "rank_last": int(counts[first]),
                                      "tie_size": int(sizes[first]), "tile_id": "SYNTHETIC"}
        arrays.update({f"{prefix}_tp": tp, f"{prefix}_fp": counts - tp,
                       f"{prefix}_thresholds": np.linspace(1, 0, len(tp), dtype=np.float64),
                       f"{prefix}_precision": tp / counts, f"{prefix}_recall": tp / k})
    return report, arrays


def inputs(tmp_path, ranking):
    report, arrays = ranking
    report_path, curve_path = tmp_path / "results.json", tmp_path / "curves.npz"
    report_path.write_text(json.dumps(report, allow_nan=False))
    np.savez(curve_path, **arrays)
    return ["--report", str(report_path), "--report-sha256", p.digest(report_path),
            "--curves", str(curve_path), "--curves-sha256", p.digest(curve_path)]


def load(tmp_path, ranking):
    args = inputs(tmp_path, ranking)
    report = p.read_report(Path(args[1]), args[3])
    return p.read_curves(Path(args[5]), args[7], report)


def test_complete_mapping_unequal_groups_ties_zero_and_singleton(tmp_path, ranking):
    curves = load(tmp_path, ranking)
    report, arrays = ranking
    assert len(curves) == 60
    assert {len(c["count"]) for c in curves} == {1, 2, 3, 4}
    assert curves[0]["count"][0] == 3  # No invented point at count 1.
    assert curves[0]["recall_percent"][0] == 0
    assert curves[2]["emphasis_indices"][0] == 0  # Previous TP is zero, not first TP.
    assert curves[3]["count"].tolist() == [curves[3]["n"]]  # All-tied ranking.
    for curve in curves:
        prefix = curve["prefix"]
        tp, fp = arrays[prefix + "_tp"], arrays[prefix + "_fp"]
        np.testing.assert_array_equal(curve["count"], tp + fp)
        np.testing.assert_array_equal(curve["recall_percent"], 100 * arrays[prefix + "_recall"])
        np.testing.assert_array_equal(curve["emphasis_indices"], np.flatnonzero(tp > np.r_[0, tp[:-1]]))
        assert curve["count"][-1] == curve["n"] and curve["recall_percent"][-1] == 100
        mapping = p.display_mapping(curve)
        assert mapping["zero_precision_groups"] == np.count_nonzero(tp == 0)
        assert mapping["zero_recall_groups"] == mapping["zero_precision_groups"]
        assert mapping["tp_increase_endpoints"] == len(curve["emphasis_indices"])
        if curve["cell"][1] == 1:
            assert set(curve["recall_percent"]) <= {0, 100}
    # Index/provenance/metric row order is not curve identity.
    for name in ("craters_curve_index", "cells", "metrics"):
        report[name].reverse()
    after = load(tmp_path, ranking)
    assert [p.display_mapping(c) for c in after] == [p.display_mapping(c) for c in curves]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "swap_cells", "swap_prefixes",
                                     "wrong_class", "wrong_seed", "bool_fold", "float_seed"])
def test_reject_bad_curve_membership_before_output(tmp_path, ranking, mutation):
    index = ranking[0]["craters_curve_index"]
    if mutation == "missing": index.pop()
    elif mutation == "duplicate": index[-1] = index[0].copy()
    elif mutation == "swap_cells": index[0]["cell"], index[1]["cell"] = index[1]["cell"], index[0]["cell"]
    elif mutation == "swap_prefixes": index[0]["prefix"], index[1]["prefix"] = index[1]["prefix"], index[0]["prefix"]
    elif mutation == "wrong_class": index[0]["class_index"] = 4
    elif mutation == "wrong_seed": index[0]["cell"][3] = 99
    elif mutation == "bool_fold": index[0]["cell"][1] = False
    else: index[0]["cell"][3] = 42.0
    args = inputs(tmp_path, ranking)
    with pytest.raises(ValueError):
        p.main([*args, "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "swap_arrays", "duplicate_zip", "alias"])
def test_reject_archive_membership_and_swapped_contents(tmp_path, ranking, mutation):
    arrays = ranking[1]
    if mutation == "missing": arrays.pop("cell_00_tp")
    elif mutation == "extra": arrays["surprise"] = np.zeros(1)
    elif mutation == "swap_arrays": arrays["cell_00_tp"], arrays["cell_01_tp"] = arrays["cell_01_tp"], arrays["cell_00_tp"]
    elif mutation == "alias": arrays["cell_00_tp.npy"] = arrays.pop("cell_00_tp")
    args = inputs(tmp_path, ranking)
    if mutation == "duplicate_zip":
        with zipfile.ZipFile(args[5], "a") as archive:
            data = archive.read("cell_00_tp.npy")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("cell_00_tp.npy", data)
        args[7] = p.digest(Path(args[5]))
    with pytest.raises(ValueError):
        p.main([*args, "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mutation", ["tp_float", "fp_bool", "overflow", "negative", "tp_decrease",
    "fp_decrease", "wrong_tp_end", "wrong_fp_end", "zero_count", "repeat_count", "shape", "empty",
    "too_many_groups", "float32", "nan", "inf", "recall_mismatch", "precision_mismatch",
    "threshold_tie", "threshold_up", "threshold_negative", "threshold_over_one", "wrong_distinct",
    "wrong_ties", "wrong_largest_tie", "wrong_single_rank"])
def test_reject_bad_saved_vectors(tmp_path, ranking, mutation):
    report, arrays = ranking
    tp, fp = arrays["cell_00_tp"], arrays["cell_00_fp"]
    row = next(r for r in report["metrics"] if r["cell"] == report["craters_curve_index"][0]["cell"]
               and r["class_index"] == 5)
    if mutation == "tp_float": arrays["cell_00_tp"] = tp.astype(float)
    elif mutation == "fp_bool": arrays["cell_00_fp"] = fp.astype(bool)
    elif mutation == "overflow": arrays["cell_00_fp"] = np.full(3, 2**64 - 1, dtype=np.uint64)
    elif mutation == "negative": tp[0] = -1
    elif mutation == "tp_decrease": tp[:2] = [2, 1]
    elif mutation == "fp_decrease": fp[:2] = [4, 3]
    elif mutation == "wrong_tp_end": tp[-1] -= 1
    elif mutation == "wrong_fp_end": fp[-1] -= 1
    elif mutation == "zero_count": fp[0] = 0
    elif mutation == "repeat_count": tp[:2] = 0; fp[:2] = 3
    elif mutation == "shape": arrays["cell_00_tp"] = tp.reshape(1, -1)
    elif mutation in ("empty", "too_many_groups"):
        for field in p.FIELDS:
            arrays["cell_00_" + field] = np.zeros(0 if mutation == "empty" else row["n"] + 1)
    elif mutation == "float32": arrays["cell_00_recall"] = arrays["cell_00_recall"].astype(np.float32)
    elif mutation == "nan": arrays["cell_00_recall"][0] = np.nan
    elif mutation == "inf": arrays["cell_00_precision"][0] = np.inf
    elif mutation == "recall_mismatch": arrays["cell_00_recall"][0] = .001
    elif mutation == "precision_mismatch": arrays["cell_00_precision"][0] = .001
    elif mutation == "threshold_tie": arrays["cell_00_thresholds"][:2] = .5
    elif mutation == "threshold_up": arrays["cell_00_thresholds"][:] = [0, .5, 1]
    elif mutation == "threshold_negative": arrays["cell_00_thresholds"][-1] = -.1
    elif mutation == "threshold_over_one": arrays["cell_00_thresholds"][0] = 1.1
    elif mutation == "wrong_distinct": row["distinct_scores"] += 1
    elif mutation == "wrong_ties": row["tied_groups"] += 1
    elif mutation == "wrong_largest_tie": row["largest_tie"] += 1
    else:
        single = next(r["single_positive"] for r in report["metrics"]
                      if r["cell"][1] == 1 and r["class_index"] == 5)
        single["rank_first"] += 1; single["rank_last"] += 1
    args = inputs(tmp_path, ranking)
    with pytest.raises(ValueError):
        p.main([*args, "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("which", ["report", "curves"])
def test_hash_rejected_before_decoding_or_output(tmp_path, ranking, monkeypatch, which):
    args = inputs(tmp_path, ranking)
    path, hash_index = (Path(args[1]), 3) if which == "report" else (Path(args[5]), 7)
    path.write_bytes(b"not JSON or ZIP")
    args[hash_index] = "0" * 64
    if which == "report":
        monkeypatch.setattr(json, "loads", lambda *a, **kw: pytest.fail("decoded unauthenticated JSON"))
    else:
        monkeypatch.setattr(zipfile, "ZipFile", lambda *a, **kw: pytest.fail("decoded unauthenticated ZIP"))
        monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("decoded unauthenticated arrays"))
    with pytest.raises(ValueError, match="SHA256"):
        p.main([*args, "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


def test_full_report_validator_is_used(tmp_path, ranking):
    # An unrelated class summary is invalid: a Craters-only partial check would miss it.
    ranking[0]["summaries"]["geographic"].pop()
    with pytest.raises(ValueError, match="Summary grid incomplete"):
        p.main([*inputs(tmp_path, ranking), "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


def test_bounded_archive_and_member(tmp_path, ranking):
    args = inputs(tmp_path, ranking)
    path = Path(args[5])
    with path.open("wb") as stream:
        stream.truncate(p.ARCHIVE_LIMIT + 1)
    with pytest.raises(ValueError, match="20 MiB"):
        p.read_curves(path, p.digest(path), ranking[0])
    # Use a larger member than the small permitted header allowance.
    ranking[1]["cell_00_tp"] = np.zeros(6600, dtype=np.int64)
    args = inputs(tmp_path, ranking)
    with pytest.raises(ValueError, match="bounded vector"):
        p.read_curves(Path(args[5]), args[7], ranking[0])


@pytest.mark.parametrize("field", ["thresholds", "precision", "recall"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_all_float_vectors_reject_nonfinite(tmp_path, ranking, field, value):
    ranking[1]["cell_00_" + field][0] = value
    with pytest.raises(ValueError, match="finite float64"):
        load(tmp_path, ranking)


def test_fixed_count_axis_rejects_larger_region_before_output(tmp_path, ranking):
    report = ranking[0]
    counts = report["fold_counts"][0]["test"]
    counts["class_counts"][0] += 6501 - counts["total"]
    counts["total"] = 6501
    for row in report["metrics"]:
        if row["cell"][1] == 0:
            row.update(n=6501, positives=counts["class_counts"][row["class_index"]])
            row.update(negatives=6501 - row["positives"],
                       prevalence_all_tied_reference=row["positives"] / 6501)
    with pytest.raises(ValueError, match="support exceeds count axis"):
        p.main([*inputs(tmp_path, ranking), "--output", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("variant", ["cvpr", "full"])
def test_exact_artists_all_endpoints_and_only_tp_increases(tmp_path, ranking, variant):
    curves = load(tmp_path, ranking)
    fig, receipt = p.draw_figure(curves, variant)
    assert len(fig.axes) == 10 and len(receipt["artists"]) == 60
    assert sum(len(ax.lines) for ax in fig.axes) == 120
    expected_markers = {("dino_c1", "random", s): m for s, m in zip(range(42, 47), ("o", "s", "^", "D", "v"))}
    expected_markers["dino_c1", "pretrained", 42] = "X"
    expected_markers.update({("final", a, s): m for a in ("scratch", "pretrained")
                            for s, m in enumerate(("o", "s", "^"))})
    line_pairs = {ax: iter(ax.lines) for ax in fig.axes}
    for curve, record in zip(curves, receipt["artists"]):
        kind, fold, arm, seed = curve["cell"]
        ax = fig.axes[2 * fold + (kind == "final")]
        background, emphasis = next(line_pairs[ax]), next(line_pairs[ax])
        prefix = curve["prefix"]
        tp, fp = ranking[1][prefix + "_tp"], ranking[1][prefix + "_fp"]
        x, y = tp + fp, ranking[1][prefix + "_recall"] * 100
        selected = np.flatnonzero(tp > np.r_[0, tp[:-1]])
        np.testing.assert_array_equal(background.get_xdata(), x)
        np.testing.assert_array_equal(background.get_ydata(), y)
        np.testing.assert_array_equal(emphasis.get_xdata(), x[selected])
        np.testing.assert_array_equal(emphasis.get_ydata(), y[selected])
        assert record["tp_increase_group_indices"] == selected.tolist()
        assert not background.get_rasterized() and not emphasis.get_rasterized()
        assert emphasis.get_marker() == expected_markers[kind, arm, seed]
        assert background.get_color() == emphasis.get_color()
        for role, line in (("all_groups", background), ("tp_increases", emphasis)):
            assert line.get_linestyle() == "None" and line.get_drawstyle() == "default"
            assert not line.get_path().should_simplify and line.get_markevery() is None
            assert record[role]["x_sha256"] == hashlib.sha256(np.asarray(line.get_xdata(), dtype="<f8").tobytes()).hexdigest()
            assert record[role]["y_sha256"] == hashlib.sha256(np.asarray(line.get_ydata(), dtype="<f8").tobytes()).hexdigest()
    qa = receipt["qa"]
    assert qa["width_inches"] == 6.5 and qa["height_inches"] <= 7.4 and not qa["clipped_text"]
    assert all(t["size_pt"] == 9 and t["inside_canvas"] for t in qa["texts"])
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        assert ax.get_xscale() == "log" and ax.get_yscale() == "linear"
        assert ax.get_xlim() == (1, 6500) and ax.get_ylim() == (-5, 105)
        assert ax.get_yticks().tolist() == [0, 50, 100]
        assert ax.title.get_window_extent(renderer).y0 > ax.bbox.y1
        assert ax.get_title().startswith("Region ") and ": " in ax.get_title()
    assert {t.get_text() for t in fig.texts} >= {"Retrieved test tiles (log scale)", "Recall of Craters map labels (%)"}
    assert [r["label"] for r in qa["legend"][0]] == [*(f"Random {s}" for s in range(42, 47)), "Pretrained"]
    assert [r["label"] for r in qa["legend"][1]] == [f"{a} {s}" for a in ("Scratch", "ImageNet") for s in range(3)]
    assert not fig.legends[0].get_window_extent(renderer).overlaps(fig.legends[1].get_window_extent(renderer))


@pytest.mark.parametrize("mutation", ["connector", "step", "thin", "coordinate"])
def test_artist_negative_controls(tmp_path, ranking, monkeypatch, mutation):
    from matplotlib.axes import Axes

    curves = load(tmp_path, ranking)
    original = Axes.plot
    def changed(self, *args, **kwargs):
        if mutation == "connector": kwargs["linestyle"] = "-"
        elif mutation == "step": kwargs.update(linestyle="-", drawstyle="steps-post")
        elif mutation == "thin": kwargs["markevery"] = 2
        else: args = (np.asarray(args[0]) + 1, *args[1:])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Axes, "plot", changed)
    with pytest.raises(ValueError, match="connects or thins|coordinates differ"):
        p.draw_figure(curves, "cvpr")


def test_cli_actual_saves_and_last_completion_receipt(tmp_path, ranking, monkeypatch):
    output = tmp_path / "render"
    original = p.write_json
    writes = []
    def checked_write(path, value):
        writes.append(path.name)
        if path.name == "complete.json":
            assert (output / "display.json").exists()
            assert all((output / name).exists() for name in value["outputs"])
        original(path, value)
    monkeypatch.setattr(p, "write_json", checked_write)
    args = inputs(tmp_path, ranking)
    p.main([*args, "--output", str(output)])
    assert writes == ["display.json", "complete.json"]
    assert {f.name for f in output.iterdir()} == {"display.json", "complete.json",
        "crater-retrieval-cvpr.pdf", "crater-retrieval-full.pdf",
        "crater-retrieval-cvpr.png", "crater-retrieval-full.png"}
    receipt = json.loads((output / "complete.json").read_text())
    display = json.loads((output / "display.json").read_text())
    assert len(display["curves"]) == 60 and set(display["variants"]) == {"cvpr", "full"}
    assert display["report_sha256"] == args[3] and display["curves_sha256"] == args[7]
    for name, item in receipt["outputs"].items():
        assert p.digest(output / name) == item["sha256"]
        assert (output / name).stat().st_size == item["bytes"] > 0
    for variant in ("cvpr", "full"):
        data = (output / f"crater-retrieval-{variant}.pdf").read_bytes()
        assert data.startswith(b"%PDF") and str(tmp_path).encode() not in data
    with pytest.raises(FileExistsError):
        p.main([*args, "--output", str(output)])


@pytest.mark.parametrize("failure", ["draw", "save", "display"])
def test_partial_failures_never_write_complete(tmp_path, ranking, monkeypatch, failure):
    def broken(*a, **kw):
        raise RuntimeError("synthetic failure")
    if failure == "draw": monkeypatch.setattr(p, "draw_figure", broken)
    elif failure == "save": monkeypatch.setattr(p.roman, "save_figure", broken)
    else: monkeypatch.setattr(p, "write_json", broken)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        p.main([*inputs(tmp_path, ranking), "--output", str(output)])
    assert output.exists() and not (output / "complete.json").exists()


def test_output_fresh_outside_checkout_and_no_symlink_reuse(tmp_path, ranking):
    args = inputs(tmp_path, ranking)
    for existing in (tmp_path, tmp_path / "results.json"):
        with pytest.raises(FileExistsError):
            p.main([*args, "--output", str(existing)])
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nonexistent", target_is_directory=True)
    with pytest.raises(FileExistsError):
        p.main([*args, "--output", str(link)])
    with pytest.raises(ValueError, match="outside the checkout"):
        p.main([*args, "--output", str(p.ROOT / "rejected-retrieval")])


@pytest.mark.parametrize("variant", ["cvpr", "full"])
def test_display_sparse_identifiers_keep_actual_tie_endpoints(tmp_path, ranking, variant):
    curves = load(tmp_path, ranking)
    original = [(c["count"].copy(), c["recall_percent"].copy()) for c in curves]
    fig, record = p.draw_figure(curves, variant, class_name="Invented", dense=True)
    axis_lines = {ax: iter(ax.lines) for ax in fig.axes}
    for c, rec, (x, y) in zip(curves, record["artists"], original):
        np.testing.assert_array_equal(c["count"], x)
        np.testing.assert_array_equal(c["recall_percent"], y)
        indices = np.asarray(rec["identity_group_indices"], dtype=int)
        assert len(indices) >= 1 and np.all(np.diff(indices) > 0)
        assert indices.min() >= 0 and indices.max() < len(x)
        kind, fold, arm, seed = c["cell"]
        ax = fig.axes[fold * 2 + (kind == "final")]
        background, emphasis, identity = (next(axis_lines[ax]) for _ in range(3))
        np.testing.assert_array_equal(background.get_xdata(), x)
        np.testing.assert_array_equal(background.get_ydata(), y)
        np.testing.assert_array_equal(identity.get_xdata(), x[indices])
        np.testing.assert_array_equal(identity.get_ydata(), y[indices])
        assert len(emphasis.get_xdata()) == 0
        for line in (background, identity):
            assert line.get_linestyle() == "None" and line.get_markevery() is None
            assert not line.get_path().should_simplify
        assert not background.get_rasterized() and not identity.get_rasterized()
        assert identity.get_markerfacecolor() == ("none" if arm == "scratch" else p.COLORS[arm])
        assert identity.get_markeredgecolor() == "#222222"
        if c["positives"] == 1:
            assert set(identity.get_ydata()).issubset({0., 100.})
        if len(x) == 1:  # An all-tied fit has no attainable earlier endpoint.
            assert identity.get_xdata().tolist() == [c["n"]]
            assert identity.get_ydata().tolist() == [100.]
        assert rec["all_groups"]["point_count"] == len(x)
        assert rec["state_identity"]["x_sha256"] == p.coordinate_digest(x[indices])
        assert rec["state_identity"]["y_sha256"] == p.coordinate_digest(y[indices])
    assert len(record["artists"]) == 60 and record["identity_spacing_pt"] == 36


@pytest.mark.parametrize("variant", ["cvpr", "full"])
def test_display_reference_legend_and_craters_increment_semantics(tmp_path, ranking, variant):
    curves = load(tmp_path, ranking)
    fig, record = p.draw_figure(curves, variant)
    for ax, ref in zip(fig.axes, record["uniform_order_references"]):
        collection, = ax.collections
        coordinates, = collection.get_segments()
        assert collection.get_gid() == "uniform_order_expectation"
        assert collection.get_zorder() < min(line.get_zorder() for line in ax.lines)
        assert collection.get_linestyles()[0][1] is not None
        np.testing.assert_array_equal(coordinates[:, 0], np.arange(1, ref["n"] + 1))
        np.testing.assert_array_equal(coordinates[:, 1], 100 * coordinates[:, 0] / ref["n"])
        assert coordinates[-1].tolist() == [ref["n"], 100.]
        assert ax.get_xlim() == (1, 6500) and ax.get_ylim() == (-5, 105)
        assert ax.get_xticks().tolist() == [1, 10, 100, 1000, 5000]
    for c, rec in zip(curves, record["artists"]):
        assert rec["tp_increase_group_indices"] == c["emphasis_indices"].tolist()
        assert rec["identity_group_indices"] == [] and "state_identity" not in rec
        assert rec["tp_increases"]["point_count"] == len(c["emphasis_indices"])
    # Check the actual key's two rows, independently of its QA serialization.
    legend = fig.legends[1]
    texts, handles = legend.get_texts(), legend.legend_handles
    for row, arm in enumerate(("Scratch", "ImageNet")):
        positions = []
        for column, seed in enumerate(range(3)):
            i = 2 * column + row
            assert texts[i].get_text() == f"{arm} {seed}"
            assert handles[i].get_markerfacecolor() == ("none" if row == 0 else p.COLORS["pretrained"])
            assert handles[i].get_markeredgecolor() == "#222222"
            positions.append(texts[i].get_transform().transform(texts[i].get_position()))
        assert np.all(np.diff(np.asarray(positions)[:, 0]) > 0)
        np.testing.assert_allclose(np.asarray(positions)[:, 1], positions[0][1], atol=.1)
    assert [x["label"] for x in record["qa"]["legend"][1]] == [
        f"{arm} {seed}" for arm in ("Scratch", "ImageNet") for seed in range(3)]
    assert "expectation" in record["qa"]["reference_legend"]
    # Holding N fixed while changing K must not move the theoretical reference.
    changed = copy.deepcopy(curves)
    for c in changed:
        c["positives"] += 1
    _, other = p.draw_figure(changed, variant, class_name="Invented other", dense=True)
    assert record["uniform_order_references"] == other["uniform_order_references"]


@pytest.mark.parametrize("dpi", [90, 180, 300])
def test_display_spacing_uses_physical_points_without_moving_data(dpi):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6.5, 7.4), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.1, .1, .4, .12])
    ax.set(xscale="log", xlim=(1, 6500), ylim=(-5, 105))
    x, y = np.arange(1, 6001), np.linspace(0, 100, 6000)
    original = x.copy(), y.copy()
    # Physical path distance is an independent oracle for selected endpoints.
    xy = ax.transData.transform(np.column_stack((x, y))) / dpi * 72
    distance = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    selected = []
    for key in range(6):
        targets = np.arange(36 * (key + .5) / 6, distance[-1], 36)
        expected = np.unique(np.searchsorted(distance, targets))
        actual = p.spaced_endpoint_indices(ax, x, y, key)
        np.testing.assert_array_equal(actual, expected)
        assert 4 <= len(actual) <= 10  # Reject empty/dense/all-endpoint shortcuts.
        selected.append(actual)
    fig.set_dpi(dpi * 2)
    for key, indices in enumerate(selected):
        np.testing.assert_array_equal(p.spaced_endpoint_indices(ax, x, y, key), indices)
    np.testing.assert_array_equal(x, original[0])
    np.testing.assert_array_equal(y, original[1])


def test_display_wrong_reference_coordinates_are_rejected(tmp_path, ranking, monkeypatch):
    curves = load(tmp_path, ranking)
    original = p.LineCollection
    def altered(segments, **kwargs):
        bad = np.array(segments)
        bad[0, -1, 1] = 99
        return original(bad, **kwargs)
    monkeypatch.setattr(p, "LineCollection", altered)
    with pytest.raises(ValueError, match="Uniform-order reference"):
        p.draw_figure(curves, "full")


def test_display_interpolated_identity_marker_is_rejected(tmp_path, ranking, monkeypatch):
    from matplotlib.axes import Axes
    curves = load(tmp_path, ranking)
    original = Axes.plot
    def altered(ax, x, y, *args, **kwargs):
        if kwargs.get("markersize") == 3.2 and len(x) and "alpha" not in kwargs:
            x = np.asarray(x, dtype=float).copy()
            x[0] += .5  # Impossible between-group retrieval count.
        return original(ax, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, "plot", altered)
    with pytest.raises(ValueError, match="coordinates"):
        p.draw_figure(curves, "full", class_name="Invented", dense=True)
