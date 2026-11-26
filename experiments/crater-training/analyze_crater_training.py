"""Reproduce descriptive training availability and Craters map/prediction joins.

Only the three accepted public input byte sequences are admitted. No model is
loaded, fitted or scored; saved census counts are reused without map resampling.
"""

import argparse
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import zipfile

import numpy as np


INPUT_SHA256 = {
    "initialization_zip": "ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486",
    "map_census": "d6511e2ffb01cd6fc5f5e78b68f4d9ae7fe7ecf6176fb67aea4f0e756359ab96",
    "census_metadata": "302f12413c0a92cf0b8152383bf30765adac57d05719ff2609852efdab037126",
}
CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
MODES = ["baseline", "weighted", "all_four", "any_valid"]
ROLES = ["train", "val", "test", "buffer_excluded", "selk_holdout"]
ARRAY_KEYS = ("tile_ids", "modal_labels", "roles", "counts", "unknown")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, name, bindings):
    """Authenticate the exact bytes returned, before any format decoding."""
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    require(actual == INPUT_SHA256[name], f"Input SHA256 differs: {name}")
    bindings[name] = {"sha256": actual, "bytes": len(data)}
    return data


def integer_labels(values, length):
    require(len(values) == length and all(type(v) is int and 0 <= v < 6 for v in values),
            "Class indices must be integers in 0..5 with the expected length")
    return np.asarray(values, dtype=np.int64)


def batch_availability(n, k, batch_size):
    """Exact hypergeometric expectation, including the final partial batch."""
    require(all(type(v) is int for v in (n, k, batch_size)) and n > 0
            and 0 <= k <= n and batch_size > 0, "Invalid class/batch counts")
    sizes = [batch_size] * (n // batch_size)
    if n % batch_size:
        sizes.append(n % batch_size)
    probabilities = [Fraction(1) - Fraction(math.comb(n-k, b) if n-k >= b else 0,
                                           math.comb(n, b)) for b in sizes]
    expected = sum(probabilities, Fraction(0))
    upper = Fraction(min(k, len(sizes)), len(sizes))
    return {"images": n, "class_images": k, "batch_size": batch_size,
            "batches": len(sizes), "last_batch_size": sizes[-1],
            "max_positive_batches": min(k, len(sizes)),
            "loss_coefficient_mass_upper_bound": float(upper),
            "expected_positive_batches_uniform_shuffle": float(expected),
            "expected_positive_batch_fraction_uniform_shuffle": float(expected/len(sizes)),
            "bound_fraction": [upper.numerator, upper.denominator]}


def validate_census(metadata, census):
    require(type(metadata.get("schema_version")) is int and metadata["schema_version"] == 1
            and metadata.get("map_cells_npz_sha256") == INPUT_SHA256["map_census"]
            and metadata.get("original_results_sha256")
            == "bb38930b99034148b53f4376bac4e465fe7d5d1f6a1a36a60279b13e498682d6",
            "Public census metadata schema/lineage differs")
    schema = metadata.get("array_conventions", {})
    require(isinstance(schema, dict) and schema.get("class_order") == CLASSES
            and schema.get("mode_order") == MODES and schema.get("role_order") == ROLES,
            "Census array interpretation differs")
    ids, modal, roles, counts, unknown = (census[k] for k in ARRAY_KEYS)
    n = len(ids)
    require(ids.shape == (n,) and ids.dtype.kind == "U"
            and all(ids) and len(set(ids.tolist())) == n
            and modal.shape == (n,) and modal.dtype.kind in "iu"
            and ((modal >= 0) & (modal < 6)).all()
            and roles.shape == (n, 5) and roles.dtype.kind in "iu"
            and ((roles >= 0) & (roles < len(ROLES))).all()
            and counts.shape == (n, 4, 6) and unknown.shape == (n, 4)
            and counts.dtype == np.int64 and unknown.dtype == np.int64
            and (counts >= 0).all() and (unknown >= 0).all(), "Unexpected census schema")
    require(((roles == ROLES.index("test")).sum(axis=1) <= 1).all(),
            "Repeated test ID across folds")


def load_inputs(initialization_zip, map_census, census_metadata):
    bindings = {}
    archive_bytes = read_bound(initialization_zip, "initialization_zip", bindings)
    census_bytes = read_bound(map_census, "map_census", bindings)
    metadata_bytes = read_bound(census_metadata, "census_metadata", bindings)
    metadata = json.loads(metadata_bytes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        records = json.loads(gzip.decompress(archive.read("data/records.json.gz")))
        catalog_bytes = archive.read("data/catalog.json")
        catalog = json.loads(catalog_bytes)
        splits = [json.loads(archive.read(f"data/splits/titan_contiguous_fold_{f}.json"))
                  for f in range(5)]
    require(all(s["catalog_sha256"] == hashlib.sha256(catalog_bytes).hexdigest()
                for s in splits), "Split/catalog binding differs")
    require(records["schema_version"] == "model-followup-numeric-v1"
            and records["purpose"] == "titan_followup" and len(records["cells"]) == 90,
            "Unexpected initialization package schema")
    with np.load(io.BytesIO(census_bytes), allow_pickle=False) as arrays:
        census = {k: arrays[k].copy() for k in ARRAY_KEYS}
    validate_census(metadata, census)
    require(len(census["tile_ids"]) == 23380, "Incomplete census")
    return records, catalog, splits, census, bindings


def select_cells(records):
    require(records["class_order"] == CLASSES, "Input class names differ")
    cells = []
    for row in records["cells"]:
        cell = row["cell"]
        require(isinstance(cell, list) and len(cell) == 4, "Invalid cell identity")
        if cell[0] == "final" or (cell[0] == "dino_c1" and cell[2] == "pretrained"):
            require(type(cell[1]) is int and type(cell[3]) is int, "Invalid fold/seed")
            cells.append(row)
    expected = {("final", f, arm, s) for f in range(5)
                for arm in ("scratch", "pretrained") for s in range(3)}
    expected |= {("dino_c1", f, "pretrained", 42) for f in range(5)}
    require(len(cells) == 35 and {tuple(c["cell"]) for c in cells} == expected,
            "Missing, duplicate or unexpected accepted cell")
    return sorted(cells, key=lambda c: tuple(c["cell"]))


def fold_inputs(records, catalog, splits, census):
    """Recover ordered roles from the public catalog; cross-check every census ID."""
    require(catalog["class_names"] == CLASSES, "Catalog class order differs")
    ids = [t["tile_id"] for t in catalog["tiles"]]
    require(all(isinstance(t, str) and t for t in ids) and len(ids) == len(set(ids))
            and set(ids) == set(census["tile_ids"].tolist()), "Catalog IDs differ")
    labels = integer_labels([t["label"] for t in catalog["tiles"]], len(ids))
    lookup = {t: j for j, t in enumerate(census["tile_ids"].tolist())}
    indices = [lookup[t] for t in ids]
    require(np.array_equal(labels, census["modal_labels"][indices]), "Catalog labels differ")
    require(len(splits) == len(records["fold_counts"]) == 5, "Five folds required")
    ordered = []
    for fold, split in enumerate(splits):
        assignments = split["assignments"]
        require(type(split["policy"]["test_fold"]) is int
                and split["policy"]["test_fold"] == fold
                and set(assignments) == set(ids)
                and set(assignments.values()) <= set(ROLES), "Split fold/IDs/roles differ")
        require([ROLES.index(assignments[t]) for t in ids]
                == census["roles"][indices, fold].tolist(), "Census/split roles differ")
        roles = {}
        for role in ("train", "val", "test"):
            positions = [j for j, t in enumerate(ids) if assignments[t] == role]
            truth = labels[positions].tolist()
            count = records["fold_counts"][fold][role]
            require(type(count["total"]) is int and count["total"] == len(positions)
                    and len(count["class_counts"]) == 6
                    and all(type(v) is int for v in count["class_counts"])
                    and count["class_counts"] == np.bincount(truth, minlength=6).tolist(),
                    "Fold class counts differ")
            roles[role] = {"tile_ids": [ids[j] for j in positions], "labels": truth}
        ordered.append(roles)
    return ordered


def join_test(test, ordered, census, fold):
    ids = test["tile_ids"]
    require(ids == ordered["tile_ids"] and test["labels"] == ordered["labels"],
            "Saved test ordering/labels differ from accepted inputs")
    require(len(ids) == len(set(ids)), "Duplicate test ID")
    truth = integer_labels(test["labels"], len(ids))
    prediction = integer_labels(test["predictions"], len(ids))
    require(test["class_order"] == list(range(6))
            and all(type(v) is int for v in test["class_order"]), "Prediction class order differs")
    test_ids = census["tile_ids"][census["roles"][:, fold] == ROLES.index("test")]
    require(set(ids) == set(test_ids.tolist()), "Test membership differs from census roles")
    lookup = {t: j for j, t in enumerate(census["tile_ids"].tolist())}
    indices = np.asarray([lookup[t] for t in ids], dtype=np.int64)
    require(np.array_equal(census["modal_labels"][indices], truth), "Modal labels differ at joined IDs")
    return indices, truth, prediction


def map_rows(model, fold, seed, mode, indices, predictions, census):
    """Join each modal-Craters image to its known map-cell denominator."""
    mi = MODES.index(mode)
    rows = []
    for index, prediction in zip(indices, predictions):
        k, pred = int(index), int(prediction)
        counts = census["counts"][k, mi].tolist()
        total = sum(counts)
        rows.append({"model": model, "fold": fold, "seed": seed,
                     "tile_id": str(census["tile_ids"][k]), "mode": mode,
                     "prediction": pred, "correct": pred == 5,
                     "class_counts": counts, "known_cells": total,
                     "unknown_cells": int(census["unknown"][k, mi]),
                     "craters_fraction": counts[5]/total if total else None,
                     "predicted_class_fraction": counts[pred]/total if total else None,
                     "all_known_cells_craters": bool(total and counts[5] == total)})
    errors = [r for r in rows if not r["correct"]]
    endpoint = [r for r in rows if r["all_known_cells_craters"]]
    summary = {"model": model, "fold": fold, "seed": seed, "mode": mode,
               "craters_test_images": len(rows), "correct": sum(r["correct"] for r in rows),
               "undefined_map_fraction": sum(r["known_cells"] == 0 for r in rows),
               "errors": len(errors), "errors_with_predicted_class_present": sum(
                   r["class_counts"][r["prediction"]] > 0 for r in errors),
               "all_known_cells_craters": len(endpoint),
               "correct_all_known_cells_craters": sum(r["correct"] for r in endpoint)}
    return rows, summary


def analyze(records, catalog, splits, census):
    cells = select_cells(records)
    ordered = fold_inputs(records, catalog, splits, census)
    input_hashes, train_weights, image_rows, cohorts = {}, {}, [], []
    for cell in cells:
        kind, fold, arm, seed = cell["cell"]
        require(cell["identity"]["initialization"] == arm, "Initialization differs")
        if fold in input_hashes:
            require(cell["inputs_sha256"] == input_hashes[fold], "Fold input binding differs")
        input_hashes[fold] = cell["inputs_sha256"]
        indices, truth, prediction = join_test(cell["test"], ordered[fold]["test"], census, fold)
        if kind == "final":
            require(type(cell["recipe"]["batch_size"]) is int
                    and cell["recipe"]["batch_size"] == 32 and cell["seed"] == seed,
                    "CNN batch size or seed differs")
            weights = cell["training_class_weights"]
            require(len(weights) == 6 and all(type(w) in (int, float)
                    and math.isfinite(w) and w > 0 for w in weights), "Invalid training weights")
            if fold in train_weights:
                require(weights == train_weights[fold], "CNN training weights differ across states")
            train_weights[fold] = weights
        crater = np.flatnonzero(truth == 5)
        require(len(crater) == [8, 1, 14, 61, 48][fold], "Craters test cohort differs")
        model = "resnet18_" + arm if kind == "final" else "dinov2_pretrained"
        for mode in ("baseline", "all_four"):
            rows, summary = map_rows(model, fold, seed, mode, indices[crater], prediction[crater], census)
            image_rows.extend(rows)
            cohorts.append(summary)
    exposure = []
    for fold, counts in enumerate(records["fold_counts"]):
        train = counts["train"]
        for c, count in enumerate(train["class_counts"]):
            exposure.append({"fold": fold, "class": CLASSES[c],
                             **batch_availability(train["total"], count, 32)})
    return {"training_batch_availability": exposure,
            "map_prediction_cohorts": cohorts, "image_rows": image_rows}


def write_output(output, record):
    """Reserve a new directory and publish only a completely encoded JSON file."""
    encoded = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    output.mkdir()  # Exclusive: existing directories, files and symlinks fail.
    temporary = output / "results.json.tmp"
    with temporary.open("xb") as stream:
        stream.write(encoded)
    temporary.rename(output / "results.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-zip", required=True, type=Path)
    parser.add_argument("--map-census", required=True, type=Path)
    parser.add_argument("--census-metadata", required=True, type=Path, help="Public census counts.json")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args(argv)
    require(not args.output.exists() and not args.output.is_symlink(), "Use a new output directory")
    records, catalog, splits, census, bindings = load_inputs(
        args.initialization_zip, args.map_census, args.census_metadata)
    result = {"status": "complete", "python": platform.python_version(), "numpy": np.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs": bindings, "class_order": CLASSES,
              **analyze(records, catalog, splits, census),
              "scope": "Saved-output joins and class-count algebra only; no model, image, mask or map resampling.",
              "limits": [
                  "The coefficient bound concerns the equal-step average of normalized target-class CE coefficients, not logged epoch loss, numerical loss share, gradients or performance.",
                  "Expected batch counts assume a uniform random permutation, not a replay of actual batch orders.",
                  "Map fractions concern known sampled map cells, not physical purity or unique landforms; unknown counts remain separate.",
                  "Empty known-cell domains have null fractions and do not belong to the all-known-cells-Craters endpoint.",
                  "Error-presence counts have no rate denominator; any rate needs errors with known_cells > 0 and separate empty-domain errors.",
                  "Repeated seed predictions are not independent geographic samples; map composition repeats across seeds."]}
    terminal = json.dumps({"status": "complete", "cells": 35,
                           "cohorts": len(result["map_prediction_cohorts"]),
                           "image_rows": len(result["image_rows"]),
                           "class_count_rows": len(result["training_batch_availability"])}, allow_nan=False)
    write_output(args.output, result)
    print(terminal)


if __name__ == "__main__":
    main()
