"""Synthetic presentation tests: no prediction files, AP estimator, or models."""
import copy
import hashlib
import json

import pytest

from scripts import plot_class_ranking as p


# Explicit expected ordering is independent of the renderer's constants.
ARM_GRID = (("dino_c1", "random", (42, 43, 44, 45, 46)),
            ("dino_c1", "pretrained", (42,)),
            ("final", "scratch", (0, 1, 2)),
            ("final", "pretrained", (0, 1, 2)))
NAMES = ("plains", "dunes", "hummocky", "labyrinths", "lakes", "craters")


@pytest.fixture
def report():
    """Sentinel stored values, deliberately not recomputed from any scores.

    Aggregates are distinct sentinels too: accidentally averaging rows instead
    of copying a saved aggregate must fail the mapping assertions.
    """
    r = {"schema_version": "class-ranking-v1", "status": "computed_pending_publication",
         "class_order": list(NAMES), "fold_counts": [], "cells": [], "metrics": [],
         "input": {"sha256": "0" * 64, "bytes": 0},
         "summaries": {"regional": [], "geographic": []}}
    for fold in range(5):
        counts = [20 + fold, 18, 12, 11, 10, 1 if fold == 1 else 5 + fold]
        n = sum(counts)
        r["fold_counts"].append({"test": {"total": n, "class_counts": counts}})
        for a, (kind, arm, seeds) in enumerate(ARM_GRID):
            for j, seed in enumerate(seeds):
                cell = [kind, fold, arm, seed]
                r["cells"].append({"cell": cell})
                for c, name in enumerate(NAMES):
                    value = .01 + .01 * a + .04 * fold + .002 * j + .0001 * c
                    single = None
                    if counts[c] == 1:
                        rank = 3 if (kind, arm, seed) == ("final", "scratch", 2) else 5 + j
                        single = {"rank_first": rank, "rank_last": rank, "tie_size": 1,
                                  "tile_id": "SYNTHETIC"}
                        if rank == 3:
                            value = 1 / 3
                    r["metrics"].append({"cell": cell, "region": fold + 1, "class_index": c,
                                         "class": name, "n": n, "positives": counts[c],
                                         "negatives": n - counts[c],
                                         "prevalence_all_tied_reference": counts[c] / n,
                                         "ap": value, "undefined_reason": None,
                                         "single_positive": single})
            for c, name in enumerate((*NAMES, "macro")):
                r["summaries"]["regional"].append({"kind": kind, "initialization": arm,
                    "fold": fold, "region": fold + 1, "class": name,
                    "seed_mean_ap": .2 + .01 * a + .001 * fold + .0001 * c})
    for a, (kind, arm, _) in enumerate(ARM_GRID):
        for c, name in enumerate((*NAMES, "macro")):
            r["summaries"]["geographic"].append({"kind": kind, "initialization": arm,
                "class": name, "mean_ap": .4 + .1 * a + .01 * c,
                "regional_sd": .01 + .001 * a + .0001 * c})
    return r


def write_report(tmp_path, report):
    path = tmp_path / "SYNTHETIC.json"
    path.write_text(json.dumps(report, allow_nan=False))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_table_all_classes_arms_and_saved_aggregate_sentinels(report):
    tex, cells = p.render_table(report)
    assert len(cells) == 28
    for c, name in enumerate((*NAMES, "macro")):
        for a, (kind, arm, _) in enumerate(ARM_GRID):
            cell = cells[4 * c + a]
            assert (cell["class"], cell["kind"], cell["initialization"]) == (name, kind, arm)
            assert cell["mean"]["raw"] == .4 + .1 * a + .01 * c
            assert cell["sd"]["raw"] == .01 + .001 * a + .0001 * c
            for key in ("mean", "sd"):
                assert p.at(report, cell[key]["pointer"]) == cell[key]["raw"]
                assert cell[key]["display_value"] == 100 * cell[key]["raw"]
    assert "$40.00 \\pm 1.00$" in tex
    assert "Macro AP" in tex and "AP (\\%)" in tex
    assert "sample SD" in tex and "five equally weighted" in tex
    assert r"\textbf" not in tex and r"\mathbf" not in tex
    markdown = p.markdown_table(cells)
    assert "40.00 ± 1.00" in markdown and "Macro AP" in markdown
    assert len(markdown.splitlines()) == 11


def test_table_stale_value_class_and_arm_mapping_mutations_fail(report, tmp_path):
    path = tmp_path / "table.tex"
    path.write_text(p.render_table(report)[0])
    p.check_table(report, path)
    for mutation in ("mean", "sd", "class_swap", "arm_swap"):
        changed = copy.deepcopy(report)
        rows = changed["summaries"]["geographic"]
        if mutation == "mean": rows[0]["mean_ap"] += .01
        elif mutation == "sd": rows[0]["regional_sd"] += .01
        elif mutation == "class_swap": rows[0]["class"], rows[1]["class"] = rows[1]["class"], rows[0]["class"]
        else:
            for a, b in zip(rows[:7], rows[7:14]):
                a["initialization"], b["initialization"] = b["initialization"], a["initialization"]
        with pytest.raises(ValueError, match="Table differs"):
            p.check_table(changed, path)


def test_row_order_invariance_and_pointer_rebinding(report):
    tex = p.render_table(report)[0]
    before = p.figure_data(report)
    for rows in (report["cells"], report["metrics"], *report["summaries"].values()):
        rows.reverse()
    assert p.render_table(report)[0] == tex
    after = p.figure_data(report)
    for a, b in zip(before, after):
        assert [(v["cell"], v["x"], v["ap"]["raw"]) for v in a["points"]] == [
            (v["cell"], v["x"], v["ap"]["raw"]) for v in b["points"]]
        for point in b["points"]:
            assert p.at(report, point["ap"]["pointer"]) == point["ap"]["raw"]


def test_all_seed_coordinates_means_and_denominators(report):
    panels = p.figure_data(report)
    assert len(panels) == 5
    for fold, panel in enumerate(panels):
        assert panel["region"] == fold + 1
        assert panel["n"] == report["fold_counts"][fold]["test"]["total"]
        assert panel["positives"] == (1 if fold == 1 else 5 + fold)
        assert len(panel["points"]) == 12 and len(panel["means"]) == 4
        assert panel["reference"]["raw"] == panel["positives"] / panel["n"]
        points = iter(panel["points"])
        for a, (kind, arm, seeds) in enumerate(ARM_GRID):
            for j, seed in enumerate(seeds):
                point = next(points)
                assert point["cell"] == [kind, fold, arm, seed]
                assert point["x"] == a + .22 * (j - (len(seeds) - 1) / 2)
                raw = p.at(report, point["ap"]["pointer"])
                assert point["ap"]["display_value"] == 100 * raw
            assert panel["means"][a]["ap"]["raw"] == .2 + .01 * a + .001 * fold + .0005
    extreme = next(v for v in panels[1]["points"] if v["cell"] == ["final", 1, "scratch", 2])
    assert extreme["ap"]["display_value"] == 100 * (1 / 3)
    assert extreme["single_positive"]["rank_last"] == 3


def test_seed_and_region_mapping_mutations_propagate_without_reaggregation(report):
    before = p.figure_data(report)
    crater_rows = [r for r in report["metrics"] if r["class"] == "craters"]
    a, b = crater_rows[0], crater_rows[1]
    a["ap"], b["ap"] = b["ap"], a["ap"]
    after = p.figure_data(report)
    assert after[0]["points"][0]["ap"]["raw"] == before[0]["points"][1]["ap"]["raw"]
    assert after[0]["points"][1]["ap"]["raw"] == before[0]["points"][0]["ap"]["raw"]
    assert after[0]["means"] == before[0]["means"]  # Only saved aggregates may supply means.
    row = next(r for r in report["summaries"]["regional"]
               if r["kind"] == "final" and r["initialization"] == "pretrained"
               and r["fold"] == 4 and r["class"] == "craters")
    row["seed_mean_ap"] = .765
    assert p.figure_data(report)[4]["means"][3]["ap"]["display_value"] == 76.5


@pytest.mark.parametrize("variant", ["sans", "roman"])
def test_actual_artists_all_seeds_common_log_axes_fonts_clipping_and_extreme(report, variant):
    panels = p.figure_data(report)
    figure, evidence = p.draw_figure(panels, variant)
    artists = evidence["artists"]
    assert len(artists) == 85
    points = [a for a in artists if a["role"] == "seed"]
    assert len(points) == 60
    for a, expected in zip(points, [v for panel in panels for v in panel["points"]]):
        assert a["source"]["cell"] == expected["cell"]
        assert a["x"] == [expected["x"]]
        assert a["y"] == [expected["ap"]["display_value"]]
        assert a["marker"] == ("o", "s", "^", "D")[expected["arm_index"]]
        assert a["color"] == ("#0077BB", "#009988", "#EE7733", "#EE3377")[expected["arm_index"]]
    for a in artists:
        if a["role"] == "seed_mean":
            assert a["y"] == [a["source"]["ap"]["display_value"]] * 2
        elif a["role"] == "all_tied_reference":
            assert a["y"] == [a["source"]["display_value"]] * 2
    qa = evidence["qa"]
    assert qa["width_inches"] == 6.5 and not qa["clipped_text"]
    assert qa["height_inches"] == 2.7
    assert all(r["inside_axes"] for r in artists)
    assert all(r["clear_of_annotation_text"] for r in artists)
    assert not qa["overlapping_headings"]
    assert all(t["size_pt"] == 9 and t["inside_canvas"] for t in qa["texts"])
    assert all(ax["yscale"] == "log" and ax["ylim"] == [.01, 50] for ax in qa["axes"])
    assert all(ax["yticks"] == [.01, .1, 1, 10, 50] for ax in qa["axes"])
    texts = [t["text"] for t in qa["texts"]]
    assert any(t.endswith("K = 1") for t in texts)
    assert not any("K = 1*" in t for t in texts)
    assert "Rank 3" in texts
    assert not figure.texts  # No overall title, footer or prose panel.
    assert [r["label"] for r in qa["legend"]] == ["DINOv2\nrandom", "DINOv2\npretrained",
                                                   "ResNet18\nscratch", "ResNet18\nImageNet"]
    assert [r["marker"] for r in qa["legend"]] == ["o", "s", "^", "D"]
    caption = p.figure_caption(panels)
    assert "after inspecting the results" in caption and "arithmetic seed means" in caption
    assert "33.33%" in caption and "only one positive" in caption
    assert "100K/N on the percent axis" in caption and "asterisk" not in caption
    extreme = next(p for p in panels[1]["points"] if p["cell"] == ["final", 1, "scratch", 2])
    assert qa["annotations"] == [{"text": "Rank 3", "xy": [extreme["x"], extreme["ap"]["display_value"]],
                                  "offset_points": [-6, 0]}]
    # Fixed coordinates, no random jitter, and deterministic rendering in one runtime.
    pixels = bytes(figure.canvas.buffer_rgba())
    figure2, evidence2 = p.draw_figure(panels, variant)
    assert bytes(figure2.canvas.buffer_rgba()) == pixels
    assert evidence2 == evidence


@pytest.mark.parametrize("role", ["point", "mean", "reference"])
@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_log_axis_never_substitutes_nonpositive_or_nonfinite_values(report, role, value):
    panels = p.figure_data(report)
    target = (panels[0]["points"][0]["ap"] if role == "point" else
              panels[0]["means"][0]["ap"] if role == "mean" else panels[0]["reference"])
    target["display_value"] = value
    with pytest.raises(ValueError, match="strictly positive"):
        p.draw_figure(panels, "sans")
    assert target["display_value"] is value  # Failed rendering cannot substitute a floor.


def test_fixed_log_axis_rejects_clipped_values(report):
    panels = p.figure_data(report)
    panels[0]["points"][0]["ap"]["display_value"] = 70
    with pytest.raises(ValueError, match="Clipped data artist"):
        p.draw_figure(panels, "sans")


def test_rank_label_cannot_move_onto_mean_bar(report, monkeypatch):
    from matplotlib.axes import Axes

    panels = p.figure_data(report)
    panels[1]["means"][2]["ap"]["display_value"] = 11.2
    _, evidence = p.draw_figure(panels, "sans")
    assert all(a["clear_of_annotation_text"] for a in evidence["artists"])
    original = Axes.annotate

    def misplaced(self, text, *args, **kwargs):
        if text == "Rank 3":
            kwargs.update(xytext=(-6, -8), va="top")  # Reproduce proposal02's misleading label.
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", misplaced)
    with pytest.raises(ValueError, match="Annotation text overlaps seed_mean"):
        p.draw_figure(panels, "sans")


@pytest.mark.parametrize("mutation", ["missing_metric", "duplicate_metric", "wrong_seed", "class_index",
    "region", "counts", "prevalence", "nonfinite", "negative", "missing_summary", "duplicate_summary",
    "summary_region", "missing_rank", "bad_tie", "class_order", "status", "missing_cell"])
def test_rejects_malformed_display_inputs(report, mutation):
    row = report["metrics"][0]
    if mutation == "missing_metric": report["metrics"].pop()
    elif mutation == "duplicate_metric": report["metrics"][-1] = copy.deepcopy(row)
    elif mutation == "wrong_seed": row["cell"] = ["dino_c1", 0, "random", 99]
    elif mutation == "class_index": row["class_index"] = 1
    elif mutation == "region": row["region"] = 4
    elif mutation == "counts": row["n"] += 1
    elif mutation == "prevalence": row["prevalence_all_tied_reference"] += .01
    elif mutation == "nonfinite": row["ap"] = float("nan")
    elif mutation == "negative": row["ap"] = -.1
    elif mutation == "missing_summary": report["summaries"]["geographic"].pop()
    elif mutation == "duplicate_summary": report["summaries"]["regional"][1] = report["summaries"]["regional"][0]
    elif mutation == "summary_region": report["summaries"]["regional"][0]["region"] = 5
    elif mutation in ("missing_rank", "bad_tie"):
        single = next(r for r in report["metrics"] if r["single_positive"] is not None)
        if mutation == "missing_rank": single["single_positive"] = None
        else: single["single_positive"]["tie_size"] = 99
    elif mutation == "class_order": report["class_order"].reverse()
    elif mutation == "status": report["status"] = "incomplete"
    else: report["cells"].pop()
    with pytest.raises(ValueError):
        p.validate(report)


def test_input_hash_duplicate_json_keys_and_nonfinite(tmp_path, report):
    path, sha = write_report(tmp_path, report)
    assert p.read_report(path, sha) == report
    with pytest.raises(ValueError, match="SHA256"):
        p.read_report(path, "f" * 64)
    for data, match in ((b'{"a":1,"a":2}', "Duplicate JSON"), (b'{"a":NaN}', "Nonfinite")):
        path.write_bytes(data)
        with pytest.raises(ValueError, match=match):
            p.read_report(path, hashlib.sha256(data).hexdigest())


def test_cli_check_table_is_read_only_and_output_refuses_reuse(tmp_path, report):
    path, sha = write_report(tmp_path, report)
    table = tmp_path / "table.tex"
    table.write_text(p.render_table(report)[0])
    args = ["--report", str(path), "--report-sha256", sha]
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    p.main([*args, "--check-table", str(table)])
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    with pytest.raises(FileExistsError):
        p.main([*args, "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="outside the checkout"):
        p.main([*args, "--output", str(p.ROOT / "rejected-preview")])


def test_failure_retains_evidence_without_complete(tmp_path, report, monkeypatch):
    path, sha = write_report(tmp_path, report)
    def broken(*args):
        raise RuntimeError("synthetic render failure")
    monkeypatch.setattr(p, "draw_figure", broken)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        p.main(["--report", str(path), "--report-sha256", sha, "--output", str(output)])
    assert (output / "started.json").exists() and (output / "failure.json").exists()
    assert not (output / "complete.json").exists()
