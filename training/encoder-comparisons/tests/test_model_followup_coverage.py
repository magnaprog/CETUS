"""Synthetic saved-vector arithmetic; no model, fit or historical analysis."""
from collections import Counter
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts import analyze_model_followup as upstream
from scripts import analyze_model_followup_coverage as c
from tests.test_analyze_model_followup import bind, make_grid, saved_scores, write


def replace_json(records, field, value):
    records["plan"][field] = write(Path(records["plan"][field]["path"]), value)


def publish_membership(records, tiles):
    """Small integer count reference, with deliberately unreadable old vectors."""
    root = records["root"]
    package = root / "historical.zip"
    with zipfile.ZipFile(package, "w") as bundle:
        bundle.writestr("data/catalog.json", json.dumps({"class_names": upstream.CLASSES, "tiles": tiles}))
        bundle.writestr("results/common_predictions.json.gz", b"must never be opened")
        bundle.writestr("results/partial_training_predictions.json.gz", b"must never be opened")
    records["plan"]["coverage_package"] = bind(package)
    populations, by_fold = [], []
    for fold in range(5):
        selected = [t for t in tiles if t["region"] == fold]
        groups = {"all": selected, "complete": [t for t in selected if t["valid_fraction"] == 1],
                  "partial": [t for t in selected if t["valid_fraction"] < 1]}
        populations.append({"fold": fold, "tile_ids": {g: [t["tile_id"] for t in rows] for g, rows in groups.items()}})
        by_fold.append({"fold": fold, "groups": {g: {"n": len(rows), "class_counts": [
            sum(t["label"] == k for t in rows) for k in range(6)]} for g, rows in groups.items()}})
    historical = {"class_names": upstream.CLASSES, "populations": populations, "test_images": len(tiles),
                  "execution": {"package_sha256": bind(package)["sha256"]}}
    records["plan"]["coverage_report"] = write(root / "historical.json", historical)
    acceptance = {"status": "accepted_with_interpretation_limits", "package_sha256": bind(package)["sha256"],
                  "result_sha256": records["plan"]["coverage_report"]["sha256"], "by_fold": by_fold,
                  "population_counts": {g: {"n": sum(r["groups"][g]["n"] for r in by_fold),
                      "class_counts": [sum(r["groups"][g]["class_counts"][k] for r in by_fold) for k in range(6)]}
                      for g in c.coverage.GROUPS}}
    records["plan"]["coverage_acceptance"] = write(root / "historical-acceptance.json", acceptance)


@pytest.fixture
def records(tmp_path):
    plan = make_grid(tmp_path / "grid")
    report = upstream.analyze(write(tmp_path / "upstream-plan.json", plan))
    records = {"root": tmp_path, "plan": {"schema_version": c.SCHEMA, "purpose": "synthetic_cpu_test",
               "followup_report": write(tmp_path / "upstream-report.json", report)}}
    tiles = upstream.read_json(plan["catalog"]["path"])["tiles"]
    for t in tiles:
        t["valid_fraction"] = 1 if t["tile_id"].endswith("-n0") else 0.75
    publish_membership(records, tiles)
    return records


def analyze(records):
    return c.analyze(write(records["root"] / "coverage-plan.json", records["plan"]))


def refresh_historical(records, historical):
    replace_json(records, "coverage_report", historical)
    accepted = upstream.read_json(records["plan"]["coverage_acceptance"]["path"])
    accepted["result_sha256"] = records["plan"]["coverage_report"]["sha256"]
    replace_json(records, "coverage_acceptance", accepted)


def change_saved(records, mutation):
    report = upstream.read_json(records["plan"]["followup_report"]["path"])
    row = next(r for r in report["cells"] if r["cell"][0] == "dino_c1")
    previous = row["result"]
    result = upstream.read_json(previous["path"])
    mutation(result)
    result["specification_sha256"] = upstream.digest(result["specification"])
    row["result"] = write(Path(previous["path"]), result)
    plan = upstream.read_json(report["plan"]["path"])
    plan["accepted_results"] = [row["result"] if b == previous else b for b in plan["accepted_results"]]
    report["plan"] = write(Path(report["plan"]["path"]), plan)
    replace_json(records, "followup_report", report)


def oracle(labels, predictions):
    counts = Counter(zip(labels, predictions))
    matrix = [[counts[y, p] for p in range(6)] for y in range(6)]
    support = [sum(row) for row in matrix]
    called = [sum(row[p] for row in matrix) for p in range(6)]
    ratios = {}
    for name in ("precision", "recall", "f1"):
        top = [matrix[k][k] * (2 if name == "f1" else 1) for k in range(6)]
        bottom = called if name == "precision" else support if name == "recall" else [a + b for a, b in zip(support, called)]
        ratios[name] = [float(Fraction(a, b)) if b else None for a, b in zip(top, bottom)]
    return matrix, support, called, ratios


def test_complete60_confusions_counts_and_ratios(records, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical/grid admission or model artifact path was rerun")
    monkeypatch.setattr(c.coverage, "analyze", forbidden)
    monkeypatch.setattr(c.coverage, "load_package", forbidden)
    monkeypatch.setattr(upstream, "analyze", forbidden)
    monkeypatch.setattr(upstream.Analysis, "bundle", forbidden)
    result = analyze(records)
    assert result["prediction_vectors"] == len(result["results"]) == 60
    assert len(result["populations"]) == 5
    assert result["verification_scope"]["scientific_acceptance"] is False
    for row in result["results"]:
        saved_result = upstream.read_json(row["result"]["path"])
        saved = saved_result["predictions"]["test"] if row["cell"][0] == "dino_c1" else saved_result["test"]
        indexed = dict(zip(saved["tile_ids"], zip(saved["labels"], saved["predictions"])))
        for group, scores in row["scores"].items():
            ids = result["populations"][row["cell"][1]]["tile_ids"][group]
            pairs = [indexed[i] for i in ids]
            matrix, support, called, ratios = oracle([p[0] for p in pairs], [p[1] for p in pairs])
            assert scores["confusion"] == matrix and scores["support"] == support
            assert scores["predicted_counts"] == called and scores["n"] == len(ids)
            assert scores["false_positive"] == [called[k] - matrix[k][k] for k in range(6)]
            for name, values in ratios.items():
                assert scores[name] == values
            if all(support):
                assert scores["macro_f1"] == pytest.approx(sum(ratios["f1"]) / 6, abs=1e-15, rel=0)
            else:
                assert scores["macro_f1"] is None
        assert np.array_equal(row["scores"]["all"]["confusion"],
                              np.array(row["scores"]["complete"]["confusion"]) + row["scores"]["partial"]["confusion"])


def test_seed_then_equal_region_and_both_contrasts(records):
    result = analyze(records)
    cells = {tuple(r["cell"]): r for r in result["results"]}
    for group in ("all", "complete"):
        for kind, arm, seeds in c.ARMS:
            means = [sum(cells[(kind, f, arm, s)]["scores"][group]["macro_f1"] for s in seeds) / len(seeds) for f in range(5)]
            expected_mean = sum(means) / 5
            sd = math.sqrt(sum((x - expected_mean) ** 2 for x in means) / 4)
            actual = result["cohorts"][group]["arms"][f"{kind}:{arm}"]
            assert actual["equal_region"]["mean"] == pytest.approx(expected_mean, rel=0, abs=1e-15)
            assert actual["equal_region"]["sample_sd"] == pytest.approx(sd, rel=0, abs=1e-15)
        for contrast in result["cohorts"][group]["contrasts"]:
            kind = "dino_c1" if contrast["single_pretrained_state"] else "final"
            other = "random" if kind == "dino_c1" else "scratch"
            for region in contrast["regions"]:
                f = region["fold"]
                differences = [cells[(kind, f, "pretrained", 42 if kind == "dino_c1" else s)]["scores"][group]["macro_f1"]
                               - cells[(kind, f, other, s)]["scores"][group]["macro_f1"] for s in region["seed_ids"]]
                assert region["seed_differences"] == differences
                assert region["within_region"]["mean"] == pytest.approx(sum(differences) / len(differences), rel=0, abs=1e-15)
            means = [r["within_region"]["mean"] for r in contrast["regions"]]
            assert contrast["equal_region"]["mean"] == pytest.approx(sum(means) / 5, rel=0, abs=1e-15)
    random = result["cohorts"]["all"]["arms"]["dino_c1:random"]
    sizes = [p["counts"]["all"]["n"] for p in result["populations"]]
    weighted = sum(n * r["within_region"]["mean"] for n, r in zip(sizes, random["regions"])) / sum(sizes)
    assert abs(weighted - random["equal_region"]["mean"]) > 1e-5
    pooled = np.sum([cells[("dino_c1", 0, "random", s)]["scores"]["all"]["confusion"] for s in range(42, 47)], axis=0)
    pooled_f1 = float(np.mean(2 * pooled.diagonal() / (pooled.sum(0) + pooled.sum(1))))
    assert abs(pooled_f1 - random["regions"][0]["within_region"]["mean"]) > 1e-5


def test_partial_support_withheld_without_dropping_region(records):
    result = analyze(records)
    for row in list(result["cohorts"]["partial"]["arms"].values()) + result["cohorts"]["partial"]["contrasts"]:
        assert row["unsupported_fold_ids"] == [0]
        assert row["equal_region"] == {"mean": None, "sample_sd": None, "n": 5}
        assert len(row["regions"]) == 5
    random = next(r for r in result["results"] if r["cell"] == ["dino_c1", 0, "random", 42])
    partial = random["scores"]["partial"]
    assert partial["support"][0] == 0 and partial["false_positive"][0] > 0
    assert partial["recall"][0] is None and partial["precision"][0] == partial["f1"][0] == 0


def test_empty_cohort_and_absent_unpredicted_class():
    truth = {"tile_ids": [str(i) for i in range(12)], "labels": list(range(6)) * 2}
    saved = saved_scores(truth, list(range(6)) + [0] * 6)
    groups = {"all": truth["tile_ids"], "complete": truth["tile_ids"][:6] + ["11"], "partial": truth["tile_ids"][6:11]}
    _, scores = c.joined_scores(saved, truth, groups)
    assert scores["partial"]["support"][5] == scores["partial"]["predicted_counts"][5] == 0
    assert scores["partial"]["f1"][5] is None
    groups.update(complete=truth["tile_ids"], partial=[])
    _, scores = c.joined_scores(saved, truth, groups)
    assert scores["partial"]["n"] == 0 and scores["partial"]["macro_f1"] is None
    assert scores["partial"]["f1"] == [None] * 6


def test_permuted_catalog_and_cohort_rows_preserve_arithmetic(records):
    before = analyze(records)
    with zipfile.ZipFile(records["plan"]["coverage_package"]["path"]) as bundle:
        tiles = json.loads(bundle.read("data/catalog.json"))["tiles"]
    publish_membership(records, list(reversed(tiles)))
    historical = upstream.read_json(records["plan"]["coverage_report"]["path"])
    historical["populations"].reverse()
    refresh_historical(records, historical)
    after = analyze(records)
    for key in ("populations", "results", "cohorts"):
        assert before[key] == after[key]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unexpected_seed", "duplicate_path", "tuning_test", "status", "class_order", "helper_source", "summary"])
def test_upstream_complete_grid_rejections(records, fault):
    report = upstream.read_json(records["plan"]["followup_report"]["path"])
    if fault == "missing": report["cells"].pop()
    elif fault == "duplicate": report["cells"][-1] = report["cells"][0]
    elif fault == "unexpected_seed": report["cells"][0]["cell"][3] = 999
    elif fault == "duplicate_path": report["cells"][0]["result"] = report["cells"][1]["result"]
    elif fault == "tuning_test": next(r for r in report["cells"] if r["cell"][0] == "tuning")["test"] = {}
    elif fault == "status": report["status"] = "running"
    elif fault == "class_order": report["class_order"].reverse()
    elif fault == "helper_source":
        path = records["root"] / "other-helper.py"
        path.write_text("# Different source bytes; never imported\n")
        report["analysis_source"] = bind(path)
    else: report["arms"]["dino_c1:random"]["equal_region"]["macro_f1"]["mean"] += .01
    replace_json(records, "followup_report", report)
    with pytest.raises(ValueError): analyze(records)


@pytest.mark.parametrize("fault", ["duplicate_id", "missing_id", "extra_id", "label", "seed", "float_prediction", "native_order"])
def test_saved_vector_semantic_rejections_with_refreshed_bindings(records, fault):
    def mutate(result):
        saved = result["predictions"]["test"]
        if fault == "duplicate_id": saved["tile_ids"][0] = saved["tile_ids"][1]
        elif fault == "missing_id": saved["tile_ids"].pop()
        elif fault == "extra_id": saved["tile_ids"].append("not-a-test-tile")
        elif fault == "label": saved["labels"][0] = 5
        elif fault == "seed": result["identity"]["encoder_seed"] = 99
        elif fault == "float_prediction": saved["predictions"] = [float(p) for p in saved["predictions"]]
        else:
            for key in ("tile_ids", "labels", "predictions", "logits", "probabilities"):
                saved[key].reverse()
    change_saved(records, mutate)
    with pytest.raises(ValueError): analyze(records)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "overlap", "cross_fold", "wrong_group", "counts", "bad_acceptance"])
def test_membership_and_count_rejections(records, fault):
    historical = upstream.read_json(records["plan"]["coverage_report"]["path"])
    groups = historical["populations"][0]["tile_ids"]
    if fault == "duplicate": groups["complete"].append(groups["complete"][0])
    elif fault == "missing": groups["partial"].pop()
    elif fault == "overlap": groups["partial"].append(groups["complete"][0])
    elif fault == "cross_fold": groups["all"][0] = historical["populations"][1]["tile_ids"]["all"][0]
    elif fault == "wrong_group":
        groups["complete"][1], groups["partial"][0] = groups["partial"][0], groups["complete"][1]
    refresh_historical(records, historical)
    acceptance = upstream.read_json(records["plan"]["coverage_acceptance"]["path"])
    if fault == "counts": acceptance["by_fold"][0]["groups"]["complete"]["class_counts"][0] += 1
    elif fault == "bad_acceptance": acceptance["status"] = "pending"
    replace_json(records, "coverage_acceptance", acceptance)
    with pytest.raises(ValueError): analyze(records)


def test_complete_cohort_missing_class_rejected(records):
    with zipfile.ZipFile(records["plan"]["coverage_package"]["path"]) as bundle:
        tiles = json.loads(bundle.read("data/catalog.json"))["tiles"]
    for tile in tiles:
        if tile["region"] == 0 and tile["label"] == 5:
            tile["valid_fraction"] = 0.75
    publish_membership(records, tiles)
    with pytest.raises(ValueError, match="All/complete cohort lacks a truth class"):
        analyze(records)


@pytest.mark.parametrize("fault", ["label", "fraction", "split_fold", "changed_result", "historical_pin"])
def test_catalog_split_and_byte_bindings(records, fault):
    if fault in ("label", "fraction"):
        path = Path(records["plan"]["coverage_package"]["path"])
        with zipfile.ZipFile(path) as z: catalog = json.loads(z.read("data/catalog.json"))
        catalog["tiles"][0]["label" if fault == "label" else "valid_fraction"] = 5 if fault == "label" else -1
        with zipfile.ZipFile(path, "w") as z: z.writestr("data/catalog.json", json.dumps(catalog))
        records["plan"]["coverage_package"] = bind(path)
        historical = upstream.read_json(records["plan"]["coverage_report"]["path"])
        historical["execution"]["package_sha256"] = bind(path)["sha256"]
        refresh_historical(records, historical)
        accepted = upstream.read_json(records["plan"]["coverage_acceptance"]["path"])
        accepted["package_sha256"] = bind(path)["sha256"]
        replace_json(records, "coverage_acceptance", accepted)
    elif fault == "split_fold":
        report = upstream.read_json(records["plan"]["followup_report"]["path"])
        plan = upstream.read_json(report["plan"]["path"])
        split = upstream.read_json(plan["splits"][0]["path"])
        split["policy"]["test_fold"] = 1
        plan["splits"][0] = write(Path(plan["splits"][0]["path"]), split)
        report["plan"] = write(Path(report["plan"]["path"]), plan)
        replace_json(records, "followup_report", report)
    elif fault == "changed_result":
        report = upstream.read_json(records["plan"]["followup_report"]["path"])
        path = Path(report["cells"][0]["result"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    else: records["plan"]["purpose"] = "titan_followup"
    with pytest.raises(ValueError): analyze(records)


def test_changed_input_during_arithmetic(records, monkeypatch):
    original = c.summarize
    def mutate(cells):
        result = original(cells)
        path = Path(records["plan"]["coverage_acceptance"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(c, "summarize", mutate)
    with pytest.raises(ValueError, match="changed during analysis"): analyze(records)


def test_cli_cpu_imports_and_no_overwrite(records):
    binding = write(records["root"] / "coverage-plan.json", records["plan"])
    output = records["root"] / "new-report.json"
    code = ("import sys; from scripts.analyze_model_followup_coverage import main; "
            "assert main(sys.argv[1:]) == 0; "
            "assert not {'torch', 'torchvision', 'sklearn'}.intersection(sys.modules)")
    command = [sys.executable, "-B", "-c", code, "--plan", binding["path"], "--plan-sha256", binding["sha256"], "--output", str(output)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["prediction_vectors"] == 60
    original = output.read_bytes()
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and "Report already exists" in failed.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("raw", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}'])
def test_bad_plan_json_rejected(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(ValueError): c.analyze(bind(path))


def test_publication_failure_has_no_result(records, monkeypatch):
    def fail(_): raise OSError("injected fsync failure")
    output_dir = records["root"] / "publication"
    output_dir.mkdir()
    binding = write(records["root"] / "coverage-plan.json", records["plan"])
    monkeypatch.setattr(c.coverage.os, "fsync", fail)
    with pytest.raises(OSError, match="injected fsync"):
        c.main(["--plan", binding["path"], "--plan-sha256", binding["sha256"], "--output", str(output_dir / "report.json")])
    assert not list(output_dir.iterdir())
