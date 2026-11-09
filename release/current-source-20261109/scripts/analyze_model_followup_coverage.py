"""Coverage arithmetic for the 60 test vectors in an accepted follow-up report.

The externally sealed plan binds followup_report, coverage_report,
coverage_package and coverage_acceptance (absolute path/sha256 pairs).
Parent acceptance of supervision and state replay is a prerequisite, not a
claim made by this consumer. Only stdlib/NumPy and saved JSON are evaluated.
"""
import argparse
import json
from pathlib import Path
import platform
import zipfile

import numpy as np

from scripts import analyze_model_followup as followup
from scripts import analyze_sar_coverage as coverage

SCHEMA = "model-followup-coverage-v1"
HISTORICAL = {
    "coverage_report": "3170d9e92e88e32812b848df7448df8f04af0aa3cb717acb3978680af3fb8de3",
    "coverage_package": "67b6aa65be5bab9b7cbe83df90aafcf7b479117291562471652f59ed7f182a1d",
    "coverage_acceptance": "f9010bc0fd88e80b7c54b8eefe13a1fe214528494bb58a64a276ba7bd7be217f",
}
ARMS = (("dino_c1", "random", tuple(range(42, 47))), ("dino_c1", "pretrained", (42,)),
        ("final", "scratch", (0, 1, 2)), ("final", "pretrained", (0, 1, 2)))
require = followup.require


def _ids(values):
    require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
            and len(values) == len(set(values)), "Invalid/duplicate tile IDs")
    return set(values)


def membership(plan, analysis, files):
    """Bind existing membership, joining anonymous and original catalogs by ID."""
    historical = files.json(plan["coverage_report"])
    accepted = files.json(plan["coverage_acceptance"])
    require(accepted["status"] == "accepted_with_interpretation_limits"
            and accepted["result_sha256"] == plan["coverage_report"]["sha256"]
            and accepted["package_sha256"] == plan["coverage_package"]["sha256"]
            and historical["execution"]["package_sha256"] == accepted["package_sha256"],
            "Historical coverage acceptance links differ")
    package = files.bound(plan["coverage_package"])
    with zipfile.ZipFile(package) as bundle:
        require(len(bundle.namelist()) == len(set(bundle.namelist())), "Duplicate ZIP members")
        catalog = json.loads(bundle.read("data/catalog.json"), object_pairs_hook=followup._object,
                             parse_float=followup._number, parse_constant=followup._number)
    require(catalog["class_names"] == historical["class_names"] == followup.CLASSES,
            "Coverage class order differs")
    original = {t["tile_id"]: t["label"] for t in analysis.catalog["tiles"]}
    entries = catalog["tiles"]
    require(_ids([t["tile_id"] for t in entries]) == set(original), "Anonymous catalog IDs differ")
    for tile in entries:
        require(type(tile["label"]) is int and tile["label"] == original[tile["tile_id"]],
                "Anonymous catalog label differs")
        v = tile["valid_fraction"]
        require(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1, "Invalid SAR fraction")
    fraction = {t["tile_id"]: t["valid_fraction"] for t in entries}
    for rows in (historical["populations"], accepted["by_fold"]):
        require(all(type(r["fold"]) is int for r in rows), "Invalid coverage fold")
    populations = coverage.unique(historical["populations"], ("fold",), [(f,) for f in range(5)], "Populations")
    reference = coverage.unique(accepted["by_fold"], ("fold",), [(f,) for f in range(5)], "Accepted counts")
    result, seen = [], set()
    for fold in range(5):
        groups = populations[(fold,)]["tile_ids"]
        require(set(groups) == set(coverage.GROUPS), "Coverage groups differ")
        sets = {g: _ids(groups[g]) for g in coverage.GROUPS}
        truth = analysis.truth[fold]["test"]
        require(sets["all"] == set(truth["tile_ids"]) and not seen.intersection(sets["all"]),
                "Coverage test fold IDs differ or overlap")
        seen.update(sets["all"])
        require(not sets["complete"].intersection(sets["partial"])
                and sets["complete"] | sets["partial"] == sets["all"], "Coverage partition differs")
        require(sets["complete"] == {i for i in sets["all"] if fraction[i] == 1}
                and sets["partial"] == {i for i in sets["all"] if fraction[i] < 1}, "SAR membership differs")
        ordered, counts = {}, {}
        for group in coverage.GROUPS:
            ordered[group] = [i for i in truth["tile_ids"] if i in sets[group]]
            support = [sum(original[i] == c for i in ordered[group]) for c in range(6)]
            counts[group] = {"n": len(ordered[group]), "class_counts": support}
            require(counts[group] == reference[(fold,)]["groups"][group], "Accepted cohort counts differ")
            if group != "partial":
                require(all(support), "All/complete cohort lacks a truth class")
        result.append({"fold": fold, "tile_ids": ordered, "counts": counts})
    for group in coverage.GROUPS:
        total = {"n": sum(r["counts"][group]["n"] for r in result), "class_counts": [
            sum(r["counts"][group]["class_counts"][c] for r in result) for c in range(6)]}
        require(all(total[k] == accepted["population_counts"][group][k] for k in total),
                "Accepted population totals differ")
    require(len(seen) == historical["test_images"], "Historical test union differs")
    if plan["purpose"] == "titan_followup":
        require([sum(r["counts"][g]["n"] for r in result) for g in coverage.GROUPS]
                == [23246, 20603, 2643], "Titan coverage totals differ")
    return result


def joined_scores(saved, truth, groups):
    """Keep native prediction order; match all other inputs by explicit tile ID."""
    ids = saved["tile_ids"]
    require(_ids(ids) == _ids(truth["tile_ids"]), "Prediction test ID set differs")
    labels = dict(zip(truth["tile_ids"], truth["labels"]))
    checked = followup.prediction_metrics(saved, {"tile_ids": ids, "labels": [labels[i] for i in ids]})
    positions = {tile_id: i for i, tile_id in enumerate(ids)}
    y, p = np.asarray(saved["labels"], dtype=np.int64), np.asarray(saved["predictions"], dtype=np.int64)
    scores = {}
    for group in coverage.GROUPS:
        indices = np.asarray([positions[i] for i in groups[group]], dtype=np.int64)
        score = coverage.scores(y[indices], p[indices])
        cm = np.asarray(score["confusion"])
        score["false_positive"] = (cm.sum(0) - cm.diagonal()).tolist()
        scores[group] = score
    for key in ("n", "confusion", "support", "predicted_counts"):
        require(np.array_equal(np.asarray(scores["all"][key]),
                               np.asarray(scores["complete"][key]) + scores["partial"][key]),
                "Cohort count partition differs")
    return checked, scores


def _summary(values):
    # n is the declared input count, including undefined values; never drop one.
    return ({"mean": None, "sample_sd": None, "n": len(values)} if any(v is None for v in values)
            else followup._summary(values))


def summarize(cells):
    result = {}
    for group in coverage.GROUPS:
        arms, contrasts = {}, []
        def value(kind, fold, arm, seed):
            return cells[(kind, fold, arm, seed)]["scores"][group]["macro_f1"]
        for kind, arm, seeds in ARMS:
            regions = [{"fold": f, "seed_ids": list(seeds),
                        "within_region": _summary([value(kind, f, arm, s) for s in seeds])} for f in range(5)]
            arms[f"{kind}:{arm}"] = {"regions": regions,
                "equal_region": _summary([r["within_region"]["mean"] for r in regions]),
                "unsupported_fold_ids": [r["fold"] for r in regions if r["within_region"]["mean"] is None]}
        for kind, name, other, seeds in (
                ("dino_c1", "pretrained_dino_minus_mean_five_random", "random", range(42, 47)),
                ("final", "imagenet_cnn_minus_random_paired_three_seeds", "scratch", range(3))):
            regions = []
            for fold in range(5):
                pairs = [(value(kind, fold, "pretrained", 42 if kind == "dino_c1" else s),
                          value(kind, fold, other, s)) for s in seeds]
                differences = [a - b if a is not None and b is not None else None for a, b in pairs]
                within = _summary(differences)
                if kind == "dino_c1" and within["mean"] is not None:
                    within["mean"] = pairs[0][0] - float(np.mean([b for _, b in pairs]))
                regions.append({"fold": fold, "seed_ids": list(seeds), "seed_differences": differences,
                                "within_region": within})
            contrasts.append({"name": name, "single_pretrained_state": kind == "dino_c1", "regions": regions,
                "equal_region": _summary([r["within_region"]["mean"] for r in regions]),
                "unsupported_fold_ids": [r["fold"] for r in regions if r["within_region"]["mean"] is None]})
        result[group] = {"arms": arms, "contrasts": contrasts}
    return result


def analyze(plan_binding):
    files = followup.Files()
    plan = files.json(plan_binding)
    require(set(plan) == {"schema_version", "purpose", "followup_report", *HISTORICAL}
            and plan["schema_version"] == SCHEMA, "Unknown coverage plan")
    require(plan["purpose"] in ("titan_followup", "synthetic_cpu_test"), "Unknown purpose")
    if plan["purpose"] == "titan_followup":
        require(all(plan[k]["sha256"] == v for k, v in HISTORICAL.items()), "Wrong historical coverage binding")
    report = files.json(plan["followup_report"])
    require(report["schema_version"] == followup.SCHEMA and report["status"] == "complete_grid_verified"
            and report["purpose"] == plan["purpose"] and report["class_order"] == followup.CLASSES
            and report["grid_counts"] == {"dino_c1": 30, "cnn_tuning": 30, "cnn_final": 30},
            "Incomplete/wrong upstream report")
    files.bound(report["analysis_source"])
    require(files.hash(Path(followup.__file__).resolve()) == report["analysis_source"]["sha256"],
            "Imported analysis helper differs")
    upstream = files.json(report["plan"])
    require(upstream["purpose"] == plan["purpose"] and upstream["provenance"] == report["provenance"],
            "Upstream plan/report provenance differs")
    analysis = followup.Analysis(upstream, files)
    require(report["fold_class_counts"] == analysis.counts, "Upstream role counts differ")
    rows = report["cells"]
    for row in rows:
        c = row["cell"]
        require(isinstance(c, list) and len(c) == 4 and type(c[1]) is int
                and type(c[3]) is (float if c[0] == "tuning" else int), "Invalid cell identity")
        require((c[0] == "tuning") == ("test" not in row), "Tuning/test role differs")
    keyed = {tuple(r["cell"]): r for r in rows}
    require(len(rows) == len(keyed) == 90 and set(keyed) == followup._expected_grid(), "Complete unique grid required")
    for bindings in ([r["result"] for r in rows], upstream["accepted_results"]):
        require(len(bindings) == 90 and all(set(b) == {"path", "sha256"} for b in bindings)
                and len({b["path"] for b in bindings}) == 90, "Duplicate/missing result bindings")
    require({(r["result"]["path"], r["result"]["sha256"]) for r in rows}
            == {(b["path"], b["sha256"]) for b in upstream["accepted_results"]}, "Upstream result bindings differ")
    populations = membership(plan, analysis, files)
    cells = {}
    for cell, row in sorted(keyed.items()):
        kind, fold, arm, seed = cell
        if kind == "tuning":
            continue
        result = files.json(row["result"])
        spec = result["specification"]
        require(result["status"] == "complete" and result["purpose"] == plan["purpose"]
                and result["fold"] == spec["fold"] == fold
                and result["specification_sha256"] == followup.digest(spec), "Saved result identity differs")
        if kind == "dino_c1":
            require(result["kind"] == spec["kind"] == kind and result["identity"]["initialization"] == arm
                    and result["identity"]["encoder_seed"] == seed, "DINO cell differs")
            saved = result["predictions"]["test"]
        else:
            require(result["kind"] == spec["kind"] == "resnet18" and spec["phase"] == "final"
                    and spec["model"]["initialization"] == arm and spec["seed"] == seed, "CNN final cell differs")
            saved = result["test"]
        metrics, scores = joined_scores(saved, analysis.truth[fold]["test"], populations[fold]["tile_ids"])
        for key, expected in metrics.items():
            if key in followup.METRICS:
                followup._close(row["test"][key], expected, "Upstream test metric differs")
            else:
                require(row["test"][key] == expected, "Upstream test counts/IDs differ")
        cells[cell] = {"cell": list(cell), "result": row["result"], "scores": scores,
                       "ordered_ids_sha256": metrics["ordered_ids_sha256"], "labels_sha256": metrics["labels_sha256"]}
    require(len(cells) == 60, "Exactly 60 test vectors required")
    summaries = summarize(cells)
    for key, arm in summaries["all"]["arms"].items():
        for field in ("mean", "sample_sd"):
            followup._close(arm["equal_region"][field], report["arms"][key]["equal_region"]["macro_f1"][field],
                            "All-test arm summary differs")
    original_contrasts = {r["name"]: r for r in report["contrasts"]}
    require(set(original_contrasts) == {c["name"] for c in summaries["all"]["contrasts"]}
            and len(report["contrasts"]) == 2, "Upstream contrasts differ")
    for contrast in summaries["all"]["contrasts"]:
        for field in ("mean", "sample_sd"):
            followup._close(contrast["equal_region"][field], original_contrasts[contrast["name"]]["equal_region"][field],
                            "All-test contrast differs")
    sources = {Path(m.__file__).name: {"path": str(Path(m.__file__).resolve()),
               "sha256": files.hash(Path(m.__file__).resolve())} for m in (followup, coverage)}
    sources[Path(__file__).name] = {"path": str(Path(__file__).resolve()), "sha256": files.hash(Path(__file__).resolve())}
    files.unchanged()
    return {"schema_version": SCHEMA, "status": "coverage_verified_from_accepted_report", "purpose": plan["purpose"],
            "plan": plan_binding, "inputs": {k: plan[k] for k in ("followup_report", *HISTORICAL)},
            "class_order": followup.CLASSES, "test_images": sum(p["counts"]["all"]["n"] for p in populations),
            "prediction_vectors": 60, "populations": populations, "results": list(cells.values()), "cohorts": summaries,
            "execution": {"sources": sources, "python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
            "verification_scope": {"saved_prediction_arithmetic": True, "state_replay": False, "estimator_fits": False,
                "model_execution": False, "scientific_acceptance": False, "prior_acceptance": "Parent prerequisite",
                "interpretation": "Seeds precede five equal-weight regions; SD is descriptive. Coverage changes class, "
                    "geographic and acquisition composition, not a causal missing-pixel effect or independent geological truth."}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Report already exists")
    result = analyze({"path": str(args.plan), "sha256": args.plan_sha256})
    coverage.write_result(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": coverage.digest(args.output), "prediction_vectors": 60}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
