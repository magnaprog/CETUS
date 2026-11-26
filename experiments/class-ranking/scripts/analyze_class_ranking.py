"""CPU ranking of the authenticated initialization companion; no model imports.

Scientific execution requires prior independent source/contract review and parent
authorization. Synthetic tests do not open the accepted scientific archive.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np

from scripts import analyze_model_followup as primary
from scripts.analyze_sar_coverage import write_result


SCHEMA = "class-ranking-v1"
ZIP_SHA256 = "ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486"
CLASSES = primary.CLASSES
require = primary.require
ROLE_SUPPORTS = [
    [[9700, 2363, 1962, 314, 141, 123], [2763, 1051, 345, 91, 91, 1], [2486, 492, 372, 51, 251, 8]],
    [[8277, 1671, 1934, 187, 358, 115], [3729, 1130, 382, 156, 9, 14], [2842, 1065, 380, 109, 100, 1]],
    [[8171, 2383, 1174, 188, 375, 57], [2755, 305, 1125, 88, 97, 59], [3956, 1164, 403, 175, 9, 14]],
    [[8882, 2672, 1103, 320, 350, 23], [3034, 858, 443, 55, 28, 48], [2992, 349, 1142, 102, 102, 61]],
    [[9600, 2538, 1875, 363, 192, 76], [2340, 486, 334, 44, 241, 8], [3130, 888, 460, 61, 33, 48]],
]


def decode(data):
    return json.loads(data, object_pairs_hook=primary._object,
                      parse_float=primary._number, parse_constant=primary._number)


def load_package(path):
    """Authenticate the bytes actually decoded; never execute archive members."""
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == ZIP_SHA256, "Initialization ZIP SHA256 differs")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        manifest = decode(archive.read("manifest.json"))
        require(set(names) == set(manifest["files"]) | {"manifest.json", "SHA256SUMS"},
                "ZIP inventory differs")

        def member(name):
            data = archive.read(name)
            binding = manifest["files"][name]
            require(len(data) == binding["bytes"]
                    and hashlib.sha256(data).hexdigest() == binding["sha256"], "Member SHA256/size differs")
            return data

        catalog_bytes = member("data/catalog.json")
        catalog = decode(catalog_bytes)
        splits = [decode(member(f"data/splits/titan_contiguous_fold_{f}.json")) for f in range(5)]
        require(all(s["catalog_sha256"] == hashlib.sha256(catalog_bytes).hexdigest() for s in splits),
                "Split/catalog SHA256 differs")
        with gzip.GzipFile(fileobj=io.BytesIO(member("data/records.json.gz"))) as stream:
            records = json.load(stream, object_pairs_hook=primary._object,
                                parse_float=primary._number, parse_constant=primary._number)
    require(records["purpose"] == "titan_followup", "CLI requires the scientific package purpose")
    return records, catalog, splits, {"sha256": ZIP_SHA256, "bytes": len(payload)}


def role_truth(records, catalog, splits):
    require(catalog["class_names"] == CLASSES and len(splits) == len(records["fold_counts"]) == 5,
            "Catalog classes or fold count differ")
    tiles = catalog["tiles"]
    ids = [t["tile_id"] for t in tiles]
    require(all(type(i) is str and i for i in ids) and len(ids) == len(set(ids)), "Invalid/duplicate catalog IDs")
    require(all(type(t["label"]) is int and 0 <= t["label"] < 6 for t in tiles), "Invalid catalog label")
    truth, seen = [], set()
    for fold, split in enumerate(splits):
        assignments = split["assignments"]
        require(type(split["policy"]["test_fold"]) is int and split["policy"]["test_fold"] == fold
                and set(assignments) == set(ids)
                and set(assignments.values()) <= set(primary.ROLES) | {"buffer_excluded", "selk_holdout"},
                "Split fold/IDs/roles differ")
        require(set(records["fold_counts"][fold]) == set(primary.ROLES), "Role count keys differ")
        roles = {}
        for ri, role in enumerate(primary.ROLES):
            rows = [t for t in tiles if assignments[t["tile_id"]] == role]
            labels = [t["label"] for t in rows]
            support = [labels.count(c) for c in range(6)]
            count = records["fold_counts"][fold][role]
            require(type(count["total"]) is int and all(type(n) is int for n in count["class_counts"])
                    and count == {"total": len(rows), "class_counts": support} and all(support),
                    "Role support differs or a primary class is absent")
            if records["purpose"] == "titan_followup":
                require(support == ROLE_SUPPORTS[fold][ri], "Fixed scientific support differs")
            roles[role] = {"tile_ids": [t["tile_id"] for t in rows], "labels": labels}
        require(not seen.intersection(roles["test"]["tile_ids"]), "Repeated test ID across folds")
        seen.update(roles["test"]["tile_ids"])
        truth.append(roles)
    if records["purpose"] == "titan_followup":
        require(len(ids) == 23380 and len(seen) == 23246, "Catalog/test population differs")
    return truth


def check_selection(row, validation_truth):
    """Check saved selection metadata; never select using ranking or test data."""
    recipe, history = row["recipe"], row["validation_history"]
    require({k: v for k, v in recipe.items() if k != "lr"} == primary.RECIPE
            and recipe["lr"] in primary.LRS and len(history) == 100, "CNN recipe/history differs")
    scores = []
    for epoch, entry in enumerate(history, 1):
        h, metrics = entry["history"], entry["metrics"]
        require(type(h["epoch"]) is int and h["epoch"] == epoch
                and type(h["optimizer_steps"]) is int and h["optimizer_steps"] > 0
                and type(h["val_macro_f1"]) in (int, float)
                and np.isfinite(h["val_macro_f1"]) and 0 <= h["val_macro_f1"] <= 1,
                "Invalid validation selection history")
        require(metrics["support_per_class"] == [validation_truth["labels"].count(c) for c in range(6)]
                and metrics["ordered_ids_sha256"] == primary.digest(validation_truth["tile_ids"])
                and metrics["labels_sha256"] == primary.digest(validation_truth["labels"]),
                "Validation role support differs")
        primary._close(metrics["macro_f1"], h["val_macro_f1"], "Validation F1 metadata differs")
        scores.append(h["val_macro_f1"])
    chosen = scores.index(max(scores)) + 1
    require(type(row["selected_epoch"]) is int and row["selected_epoch"] == chosen
            and row["selected_score"] == scores[chosen-1]
            and row["validation"] == history[chosen-1]["metrics"], "Original epoch selection differs")
    for key in ("selected_state_sha256", "final_state_sha256"):
        primary._sha(row[key])


def validate_records(records, catalog, splits):
    """Validate the entire fixed grid before calculating any ranking metric."""
    require(records["schema_version"] == "model-followup-numeric-v1"
            and records["purpose"] in ("titan_followup", "synthetic_cpu_test")
            and records["class_order"] == CLASSES, "Record schema/purpose/classes differ")
    truth = role_truth(records, catalog, splits)
    cells, result_ids, inputs, states = {}, set(), {}, {}
    expected = primary._expected_grid()
    for row in records["cells"]:
        c = row["cell"]
        require(type(c) is list and len(c) == 4 and type(c[0]) is str and type(c[2]) is str
                and type(c[1]) is int and type(c[3]) is (float if c[0] == "tuning" else int),
                "Invalid cell identity")
        c = tuple(c)
        require(c in expected and c not in cells, "Duplicate/unexpected cell")
        kind, fold, arm, seed = c
        result_id = primary._sha(row["result_sha256"])
        require(result_id not in result_ids, "Duplicate source result hash")
        result_ids.add(result_id)
        binding = primary._sha(row["inputs_sha256"])
        require(inputs.setdefault(fold, binding) == binding, "Fold input binding differs")
        identity = row["identity"]
        require(identity["initialization"] == arm and identity["transform"] == primary.TRANSFORM
                and identity["dtype"] == "float32", "Model identity differs")
        primary._sha(identity["initial_state_sha256"])
        identity_fields = ("backend constructor dtype encoder_seed initial_state_sha256 initialization model revision "
                           "source_python_inventory_sha256 transform checkpoint_sha256" if kind == "dino_c1" else
                           "backbone_seed dtype head_seed head_state_sha256 initial_state_sha256 initialization model "
                           "num_classes resnet_source_sha256 torchvision_version transform weights "
                           "zero_initialized_legacy_batch_counters checkpoint_sha256")
        require(set(identity) == set(identity_fields.split()), "Complete model identity fields differ")
        validation = row["validation"]
        require(validation["ordered_ids_sha256"] == primary.digest(truth[fold]["val"]["tile_ids"])
                and validation["labels_sha256"] == primary.digest(truth[fold]["val"]["labels"])
                and validation["support_per_class"] == [truth[fold]["val"]["labels"].count(c) for c in range(6)],
                "Validation role identity differs")
        if kind == "dino_c1":
            require(identity["model"] == f"dinov2_{arm}_matched" and type(identity["encoder_seed"]) is int
                    and identity["encoder_seed"] == seed and identity["revision"] == primary.DINO_REVISION
                    and identity["constructor"] == primary.DINO_CONSTRUCTOR
                    and identity["backend"] == "torch_sdpa_no_xformers"
                    and identity["source_python_inventory_sha256"] == primary.DINO_SOURCE_SHA,
                    "Matched DINO identity differs")
            primary._sha(identity["checkpoint_sha256"])
            if arm == "pretrained":
                require(identity["checkpoint_sha256"] == primary.DINO_SHA, "DINO checkpoint differs")
            p = row["classifier_parameters"]
            require(all(p[k] == v for k, v in {"C": 1., "solver": "lbfgs", "max_iter": 2000,
                    "tol": 1e-4, "class_weight": "balanced", "random_state": 0, "fit_intercept": True}.items())
                    and p["penalty"] in ("l2", "deprecated") and p["l1_ratio"] in (None, 0, 0.)
                    and row["scaler_parameters"] == {"copy": True, "with_mean": True, "with_std": True},
                    "Fixed DINO readout differs")
        else:
            actual_seed = 100 if kind == "tuning" else seed
            primary._sha(identity["head_state_sha256"])
            primary._sha(identity["resnet_source_sha256"])
            require(type(identity["torchvision_version"]) is str and identity["torchvision_version"],
                    "Missing CNN implementation version")
            require(identity["model"] == "resnet18" and identity["num_classes"] == 6
                    and all(type(row[k]) is int and row[k] == v for k, v in
                            {"seed": actual_seed, "head_seed": 20000+actual_seed, "loader_seed": 10000+actual_seed}.items())
                    and identity["backbone_seed"] == actual_seed and identity["head_seed"] == row["head_seed"],
                    "CNN seed/model identity differs")
            require(identity["checkpoint_sha256"] == (primary.RESNET_SHA if arm == "pretrained" else None)
                    and identity["weights"] == ("ResNet18_Weights.IMAGENET1K_V1" if arm == "pretrained" else None),
                    "CNN initialization weights differ")
            check_selection(row, truth[fold]["val"])
            if kind == "tuning":
                require("test" not in row and row["recipe"]["lr"] == seed
                        and row["selection"] is None and row["tuning_results"] == [], "Tuning role differs")
        state_key = ("dino" if kind == "dino_c1" else "cnn", arm,
                     seed if kind != "tuning" else 100)
        state = primary.digest(identity)
        require(states.setdefault(state_key, state) == state, "Model state identity differs across folds")
        if kind != "tuning":
            saved = row["test"]
            require(all(type(v) is int for k in ("labels", "predictions", "class_order") for v in saved[k]),
                    "Prediction indices must be integers")
            require(type(saved["probability_argmax_disagreements"]) is int, "Invalid argmax disagreement count")
            for name in ("probabilities", "logits"):
                values = saved[name]
                require(all(type(v) in (int, float) for r in values for v in r), "Invalid numeric score")
                require(saved[name+"_dtype"] in ("float32", "float64"), "Invalid native dtype")
                array = np.asarray(values)
                require(np.array_equal(array, array.astype(saved[name+"_dtype"]).astype(np.float64)),
                        "Scores are not exact in their declared native dtype")
            primary.prediction_metrics(saved, truth[fold]["test"])
        cells[c] = row
    require(set(cells) == expected, "Missing cells")
    for fold in range(5):
        for arm in ("scratch", "pretrained"):
            tuning = [cells["tuning", fold, arm, lr] for lr in primary.LRS]
            winner = max(tuning, key=lambda r: (r["selected_score"], -r["recipe"]["lr"]))
            by_hash = {r["result_sha256"]: r for r in tuning}
            for seed in range(3):
                row = cells["final", fold, arm, seed]
                refs = row["tuning_results"]
                require(len(refs) == 3 and set(refs) == set(by_hash), "Tuning bindings differ")
                selection = {"lr": winner["recipe"]["lr"], "candidates": [
                    {"lr": by_hash[h]["recipe"]["lr"], "score": by_hash[h]["selected_score"],
                     "result_sha256": h} for h in refs], "rule": "maximum validation macro_f1, then lower LR"}
                require(row["recipe"]["lr"] == winner["recipe"]["lr"]
                        and row["selection"] == selection, "Original learning-rate selection differs")
    return [cells[c] for c in sorted(cells) if c[0] != "tuning"]


def binary_ranking(labels, scores):
    """Non-interpolated AP with whole exact-score ties; no selected threshold."""
    y, s = np.asarray(labels), np.asarray(scores)
    require(y.ndim == s.ndim == 1 and y.shape == s.shape and y.size > 0
            and y.dtype.kind in "biu" and np.isin(y, [0, 1]).all()
            and s.dtype.kind in "iuf" and np.isfinite(s).all(), "Invalid binary ranking vectors")
    order = np.argsort(s, kind="stable")[::-1]
    s, y = s[order], y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    sizes = np.diff(np.r_[-1, ends])
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    precision = tp / (ends+1)
    k, n = int(y.sum()), len(y)
    recall = tp/k if k else None
    ap = float(np.sum(np.diff(np.r_[0, tp]) * precision) / k) if k else None
    single = None
    if k == 1:
        g = int(np.flatnonzero(tp)[0])
        single = {"rank_first": int(ends[g]+2-sizes[g]), "rank_last": int(ends[g]+1),
                  "tie_size": int(sizes[g])}
    metrics = {"n": n, "positives": k, "negatives": n-k, "prevalence_all_tied_reference": k/n,
               "ap": ap, "undefined_reason": "no_positive_labels" if not k else None,
               "all_positive": k == n, "distinct_scores": len(ends),
               "tied_groups": int(np.sum(sizes > 1)), "largest_tie": int(sizes.max()),
               "single_positive": single}
    return metrics, {"thresholds": s[ends], "tp": tp, "fp": fp, "precision": precision, "recall": recall}


def summarize(rows):
    """Seed means first; geographic means/SD never pool or ensemble scores."""
    lookup = {(tuple(r["cell"]), r["class_index"]): r["ap"] for r in rows}
    arms = [("dino_c1", "random", range(42, 47)), ("dino_c1", "pretrained", [42]),
            ("final", "scratch", range(3)), ("final", "pretrained", range(3))]

    def mean(values):
        return None if any(v is None for v in values) else float(np.mean(values))

    regional, values = [], {}
    for kind, arm, seeds in arms:
        for fold in range(5):
            for c in range(7):
                per_seed = [lookup[((kind, fold, arm, seed), c)] if c < 6 else
                            mean([lookup[((kind, fold, arm, seed), j)] for j in range(6)]) for seed in seeds]
                value = mean(per_seed)
                values[kind, arm, fold, c] = value
                regional.append({"kind": kind, "initialization": arm, "fold": fold, "region": fold+1,
                                 "class": CLASSES[c] if c < 6 else "macro", "seed_mean_ap": value})
    contrasts = []
    for kind, baseline in (("dino_c1", "random"), ("final", "scratch")):
        for fold in range(5):
            for c in range(7):
                a, b = values[kind, "pretrained", fold, c], values[kind, baseline, fold, c]
                paired = None
                difference = None if a is None or b is None else a-b
                if kind == "final":
                    paired = []
                    for seed in range(3):
                        per_arm = [lookup[((kind, fold, arm, seed), c)] if c < 6 else
                                   mean([lookup[((kind, fold, arm, seed), j)] for j in range(6)])
                                   for arm in ("pretrained", baseline)]
                        pa, pb = per_arm
                        paired.append(None if pa is None or pb is None else pa-pb)
                    difference = mean(paired)
                contrasts.append({"kind": kind, "fold": fold, "region": fold+1,
                                  "class": CLASSES[c] if c < 6 else "macro",
                                  "pretrained_minus_baseline": difference,
                                  "paired_seed_differences_0_1_2": paired})
    geographic = []
    for kind, arm, _ in arms:
        for c in range(7):
            v = [values[kind, arm, f, c] for f in range(5)]
            geographic.append({"kind": kind, "initialization": arm, "class": CLASSES[c] if c < 6 else "macro",
                               "mean_ap": mean(v), "regional_sd": None if None in v else float(np.std(v, ddof=1))})
    for kind in ("dino_c1", "final"):
        for label in CLASSES + ["macro"]:
            v = [r["pretrained_minus_baseline"] for r in contrasts if r["kind"] == kind and r["class"] == label]
            geographic.append({"kind": kind, "class": label, "mean_difference": mean(v),
                               "regional_difference_sd": None if None in v else float(np.std(v, ddof=1))})
    return {"regional": regional, "contrasts": contrasts, "geographic": geographic}


def analyze(records, catalog, splits):
    cells = validate_records(records, catalog, splits)
    rows, arrays, index, provenance = [], {}, [], []
    for ci, row in enumerate(cells):
        cell, saved = row["cell"], row["test"]
        y, probabilities = np.asarray(saved["labels"]), np.asarray(saved["probabilities"])
        provenance.append({k: row[k] for k in ("cell", "identity", "result_sha256", "inputs_sha256")})
        if cell[0] == "final":
            provenance[-1].update({k: row[k] for k in ("selected_epoch", "selected_score", "selected_state_sha256", "selection")})
        for c in range(6):
            metrics, curve = binary_ranking((y == c).astype(np.int64), probabilities[:, c])
            if metrics["single_positive"] is not None:
                metrics["single_positive"]["tile_id"] = saved["tile_ids"][int(np.flatnonzero(y == c)[0])]
            rows.append({"cell": cell, "region": cell[1]+1, "class_index": c, "class": CLASSES[c],
                         "result_sha256": row["result_sha256"], **metrics})
            if c == 5:
                prefix = f"cell_{ci:02d}"
                arrays.update({prefix+"_"+k: v for k, v in curve.items()})
                index.append({"prefix": prefix, "cell": cell, "class_index": 5})
    require(len(rows) == 360 and len(index) == 60, "Ranking grid differs")
    return {"class_order": CLASSES, "fold_counts": records["fold_counts"], "metrics": rows,
            "cells": provenance, "craters_curve_index": index, "summaries": summarize(rows)}, arrays


def runtime_record():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA with CUDA_VISIBLE_DEVICES=''")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(key) == "1", f"Require {key}=1")
    forbidden = {"torch", "torchvision", "timm", "transformers", "tensorflow", "jax", "sklearn"}
    require(not forbidden.intersection(n.split(".")[0] for n in sys.modules), "Model/estimator module loaded")
    return {"python": platform.python_version(), "numpy": np.__version__,
            "environment": {k: os.environ[k] for k in
                            ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def write_curves(path, arrays):
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            os.unlink(name)


def write_completion(path, record):
    """The exclusive link commits completion; later cleanup cannot revoke it."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    name, committed = None, False
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
        committed = True
    finally:
        if name is not None:
            try:
                os.unlink(name)
            except Exception:
                # A residual temporary name does not invalidate a committed receipt.
                if not committed:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-zip", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), "Use a fresh output directory")
    files = primary.Files()
    contract = {"path": str(args.contract.absolute()), "sha256": args.contract_sha256}
    files.bound(contract)  # Prospective document is checked before scientific input access.
    dependencies = [Path(__file__).absolute(), Path(primary.__file__).absolute(),
                    Path(sys.modules[write_result.__module__].__file__).absolute()]
    source = {p.name: files.hash(p) for p in dependencies}
    runtime = runtime_record()
    args.output.mkdir()
    try:
        started = {"schema_version": SCHEMA, "status": "started", "contract_sha256": contract["sha256"],
                   "source_sha256": source, "runtime": runtime}
        write_result(args.output / "started.json", started)
        records, catalog, splits, binding = load_package(args.initialization_zip)
        result, arrays = analyze(records, catalog, splits)
        result.update({"schema_version": SCHEMA, "status": "computed_pending_publication",
                       "execution": started, "input": binding,
                       "estimand": "Within-fold class AP; seed means before five equal regions; no pooled scores or chosen threshold"})
        files.unchanged()
        runtime_record()
        write_curves(args.output / "craters-curves.npz", arrays)
        write_result(args.output / "results.json", result)
        outputs = {name: {"sha256": files.hash((args.output/name).absolute()), "bytes": (args.output/name).stat().st_size}
                   for name in ("results.json", "craters-curves.npz")}
        files.unchanged()
        write_completion(args.output / "complete.json", {"schema_version": SCHEMA, "status": "complete",
                     "contract_sha256": contract["sha256"], "input": binding, "outputs": outputs,
                     "metric_rows": 360, "craters_curves": 60})
    except Exception as error:
        write_result(args.output / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                                   "error": str(error)})
        raise
    print(json.dumps({"status": "complete", "metric_rows": 360, "craters_curves": 60}))


if __name__ == "__main__":
    main()
