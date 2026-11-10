"""Check saved loss-comparison predictions and validation choices on CPU."""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import zipfile

SCHEMA = "loss-normalization-numeric-v1"
A_ZIP = "ada744e9278f4fbac5f5f953b5688132ad0b080b35cff32ed23b7f02c3192486"
A_PRODUCER = "1f3fc8fe0e831777e3188d70caf2c5450b656333"
B_PRODUCER = "11033ff9f6fb21db81b3cc91b18bff4f9a58d02b"
CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
CELLS = {(fold, seed) for fold in range(5) for seed in range(3)}
ATOL = 8 * 2**-53
COUNTS = "confusion_matrix support_per_class true_positive false_positive false_negative".split()
SCORES = "precision_per_class recall_per_class f1_per_class".split()
EVAL = "tile_ids labels predictions logits logits_dtype class_order".split() + COUNTS + SCORES + ["macro_f1"]
HISTORY = "epoch examples optimizer_steps finite_gradients lr val_confusion_matrix val_macro_f1 loss_numerator loss_denominator train_loss val_loss_numerator val_loss_denominator val_loss training_order_sha256".split()
ROW = "fold seed result_sha256 control_result_sha256 specification_sha256 selected_epoch selected_score selected_state_sha256 validation_history validation test".split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid finite number")
    return value


def hex_sha(value):
    require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "Invalid SHA256")


def decode(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def floating(token):
        return finite(float(token))
    return json.loads(data, object_pairs_hook=pairs, parse_float=floating, parse_constant=floating)


def read_json(path, limit=64 * 1024**2):
    with Path(path).open("rb") as stream:
        if str(path).endswith(".gz"):
            with gzip.GzipFile(fileobj=stream) as zipped:
                data = zipped.read(limit + 1)
        else:
            data = stream.read(limit + 1)
    require(len(data) <= limit, "JSON size exceeds limit")
    return decode(data)


def equal(actual, expected, name="value"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ": keys differ")
        for key in expected:
            equal(actual[key], expected[key], name + "/" + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), name + ": shape differs")
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, name + "/" + str(index))
    elif type(expected) is float:
        require(abs(finite(actual) - expected) <= ATOL, name + ": metric differs")
    else:
        require(type(actual) is type(expected) and actual == expected, name + ": exact value differs")


def metrics(matrix):
    require(isinstance(matrix, list) and len(matrix) == 6, "Confusion rows differ")
    require(all(isinstance(row, list) and len(row) == 6 and
                all(type(x) is int and x >= 0 for x in row) for row in matrix), "Invalid confusion counts")
    support = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(6)]
    tp = [matrix[c][c] for c in range(6)]
    require(all(support), "Missing truth class")
    precision = [tp[c] / predicted[c] if predicted[c] else 0.0 for c in range(6)]
    recall = [tp[c] / support[c] for c in range(6)]
    f1 = [2 * tp[c] / (support[c] + predicted[c]) for c in range(6)]
    return {"confusion_matrix": matrix, "support_per_class": support,
            "predicted_per_class": predicted, "true_positive": tp,
            "false_positive": [predicted[c] - tp[c] for c in range(6)],
            "false_negative": [support[c] - tp[c] for c in range(6)],
            "precision_per_class": precision, "recall_per_class": recall,
            "f1_per_class": f1, "macro_f1": math.fsum(f1) / 6}


def predictions(record, truth):
    require(record["tile_ids"] == truth["tile_ids"] and record["labels"] == truth["labels"],
            "Ordered test IDs or labels differ")
    require(record["class_order"] == list(range(6)) and
            all(type(c) is int for c in record["class_order"]), "Class order differs")
    require(record["logits_dtype"] == "float32", "Logit dtype differs")
    n = len(truth["tile_ids"])
    require(len(record["predictions"]) == len(record["logits"]) == n, "Test vector length differs")
    matrix = [[0] * 6 for _ in range(6)]
    for label, pred, logits in zip(record["labels"], record["predictions"], record["logits"]):
        require(type(label) is int and 0 <= label < 6 and type(pred) is int and 0 <= pred < 6,
                "Invalid class index")
        require(isinstance(logits, list) and len(logits) == 6, "Logit shape differs")
        for value in logits:
            finite(value)
            require(struct.unpack("<f", struct.pack("<f", value))[0] == value,
                    "Logit does not retain float32 value")
        require(pred == max(range(6), key=logits.__getitem__), "Native logit argmax differs")
        matrix[label][pred] += 1
    result = metrics(matrix)
    equal({k: record[k] for k in COUNTS + SCORES + ["macro_f1"]},
          {k: result[k] for k in COUNTS + SCORES + ["macro_f1"]}, "Native test metrics")
    return result


def populations(catalog, splits, real):
    require(catalog["class_names"] == CLASSES and len(splits) == 5, "Catalog/classes/folds differ")
    tiles = catalog["tiles"]
    ids = [t["tile_id"] for t in tiles]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), "Invalid catalog IDs")
    require(all(type(t["label"]) is int and 0 <= t["label"] < 6 for t in tiles), "Invalid catalog label")
    result, seen = [], set()
    for fold, split in enumerate(splits):
        require(type(split["policy"]["test_fold"]) is int and split["policy"]["test_fold"] == fold,
                "Split fold differs")
        roles = split["assignments"]
        require(set(roles) == set(ids) and set(roles.values()) <=
                {"train", "val", "test", "buffer_excluded", "selk_holdout"}, "Split membership differs")
        by_role = {}
        for role in ("train", "val", "test"):
            rows = [t for t in tiles if roles[t["tile_id"]] == role]
            labels = [t["label"] for t in rows]
            counts = [labels.count(c) for c in range(6)]
            require(all(counts), "Role lacks a truth class")
            by_role[role] = {"tile_ids": [t["tile_id"] for t in rows], "labels": labels, "counts": counts}
        require(not seen.intersection(by_role["test"]["tile_ids"]), "Repeated test IDs across folds")
        seen.update(by_role["test"]["tile_ids"])
        result.append(by_role)
    if real:
        require(len(tiles) == 23380 and len(seen) == 23246, "Wrong Titan population")
    return result


def selection(row, truth, original=None):
    entries = row["validation_history"]
    require(len(entries) == 100, "Incomplete validation history")
    scores = []
    for epoch, entry in enumerate(entries, 1):
        h = entry["history"]
        hex_sha(entry["source_epoch_sha256"])
        require(type(h["epoch"]) is int and h["epoch"] == epoch and h["finite_gradients"] is True,
                "Epoch order or finite flag differs")
        n = len(truth["train"]["tile_ids"])
        require(type(h["examples"]) is int and h["examples"] == n and
                type(h["optimizer_steps"]) is int and h["optimizer_steps"] == (n + 31) // 32,
                "Training image/update count differs")
        m = metrics(h["val_confusion_matrix"])
        require(m["support_per_class"] == truth["val"]["counts"], "Validation support differs")
        score = finite(h["val_macro_f1"])
        equal(score, m["macro_f1"], "Validation macro F1")
        scores.append(score)
        if original is not None:
            require(set(entry) == {"history", "source_epoch_sha256"}, "B history entry fields differ")
            require(set(h) == set(HISTORY), "B compact history fields differ")
            require(finite(h["lr"]) == original[epoch - 1]["history"]["lr"], "Inherited learning rate differs")
            hex_sha(h["training_order_sha256"])
            for key, numerator, denominator in (("train_loss", "loss_numerator", "loss_denominator"),
                                                ("val_loss", "val_loss_numerator", "val_loss_denominator")):
                require(finite(h[numerator]) >= 0 and finite(h[denominator]) > 0, "Invalid loss operands")
                require(finite(h[key]) == h[numerator] / h[denominator], "Stored diagnostic ratio differs")
    chosen = max(range(100), key=scores.__getitem__)
    require(type(row["selected_epoch"]) is int and row["selected_epoch"] == chosen + 1 and
            finite(row["selected_score"]) == scores[chosen], "Earliest unrounded selection differs")
    selected = metrics(entries[chosen]["history"]["val_confusion_matrix"])
    equal({k: row["validation"][k] for k in COUNTS + SCORES + ["macro_f1"]},
          {k: selected[k] for k in COUNTS + SCORES + ["macro_f1"]}, "Selected validation")


def keyed(rows):
    result = {}
    for row in rows:
        fold, seed = row["fold"], row["seed"]
        key = fold, seed
        require(type(fold) is int and type(seed) is int and key in CELLS and key not in result,
                "Unexpected or duplicate fold/seed")
        result[key] = row
    require(set(result) == CELLS, "Missing fold/seed")
    return result


def aggregate(rows):
    by_key = keyed(rows)
    regions = []
    for fold in range(5):
        region = {"fold": fold, "region": fold + 1}
        for name in ("craters_f1_delta", "macro_f1_delta"):
            region[name] = math.fsum(by_key[fold, seed][name] for seed in range(3)) / 3
        regions.append(region)
    return {"cells": [by_key[k] for k in sorted(CELLS)], "regions": regions,
            "equal_region_delta": {name: math.fsum(r[name] for r in regions) / 5
                                   for name in ("craters_f1_delta", "macro_f1_delta")}}


def reconstruct(data, baseline, catalog, splits, expected, real=True):
    require(set(data) == {"schema_version", "purpose", "class_order", "a_archive_sha256", "producer",
                          "control_producer", "model", "initialization", "endpoint", "cells"},
            "Public record fields differ")
    require(data["schema_version"] == SCHEMA and data["class_order"] == CLASSES and
            data["a_archive_sha256"] == A_ZIP and data["producer"] == B_PRODUCER and
            data["control_producer"] == A_PRODUCER and data["endpoint"] == "validation_selected" and
            data["model"] == "resnet18" and data["initialization"] == "pretrained",
            "Study identity differs")
    require(data["purpose"] == ("titan_followup" if real else "synthetic_cpu_test"), "Wrong data purpose")
    require(baseline["schema_version"] == "model-followup-numeric-v1", "Wrong A schema")
    truth = populations(catalog, splits, real)
    old = {}
    for row in baseline["cells"]:
        cell = row["cell"]
        if isinstance(cell, list) and len(cell) == 4 and cell[0] == "final" and cell[2] == "pretrained":
            require(type(cell[1]) is int and type(cell[3]) is int, "Invalid A cell index")
            key = cell[1], cell[3]
            require(key in CELLS and key not in old, "Unexpected or duplicate A cell")
            old[key] = row
    require(set(old) == CELLS, "Missing A cell")
    new = keyed(data["cells"])
    rows = []
    for fold, seed in sorted(CELLS):
        a, b = old[fold, seed], new[fold, seed]
        require(set(b) == set(ROW) and set(b["test"]) == set(EVAL), "B record fields differ")
        require(set(b["validation"]) == set(COUNTS + SCORES + ["macro_f1"]), "B validation fields differ")
        for name in ("result_sha256", "control_result_sha256", "specification_sha256", "selected_state_sha256"):
            hex_sha(b[name])
        require(b["control_result_sha256"] == a["result_sha256"], "A result binding differs")
        selection(a, truth[fold])
        selection(b, truth[fold], a["validation_history"])
        ma = predictions(a["test"], truth[fold]["test"])
        mb = predictions(b["test"], truth[fold]["test"])
        require(ma["true_positive"][5] == 0, "Original Craters premise differs")
        rows.append({"fold": fold, "seed": seed, "craters_index": 5,
                     "selected_epochs": {"A": a["selected_epoch"], "B": b["selected_epoch"]},
                     "A": ma, "B": mb, "craters_f1_delta": mb["f1_per_class"][5] - ma["f1_per_class"][5],
                     "macro_f1_delta": mb["macro_f1"] - ma["macro_f1"],
                     "identities": {"A_result_sha256": a["result_sha256"], "B_result_sha256": b["result_sha256"],
                                    "B_specification_sha256": b["specification_sha256"]}})
    result = aggregate(rows)
    require(set(expected) == {"schema_version", "producer", "independent_report_sha256", "comparison"}
            and expected["schema_version"] == SCHEMA and expected["producer"] == B_PRODUCER,
            "Wrong expected-result identity")
    hex_sha(expected["independent_report_sha256"])
    equal(expected["comparison"], result, "Independent targets")
    return result


def old_archive(path):
    require(sha(path) == A_ZIP, "A archive SHA256 differs")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP names")
        manifest = decode(archive.read("manifest.json"))
        require(set(names) == set(manifest["files"]) | {"manifest.json", "SHA256SUMS"}, "A member set differs")
        def member(name):
            raw = archive.read(name)
            row = manifest["files"][name]
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                    "A member hash/size differs")
            return raw
        raw_catalog = member("data/catalog.json")
        catalog = decode(raw_catalog)
        splits = [decode(member(f"data/splits/titan_contiguous_fold_{f}.json")) for f in range(5)]
        require(all(s["catalog_sha256"] == hashlib.sha256(raw_catalog).hexdigest() for s in splits),
                "A catalog/split binding differs")
        with gzip.GzipFile(fileobj=io.BytesIO(member("data/records.json.gz"))) as stream:
            records = stream.read(512 * 1024**2 + 1)
        require(len(records) <= 512 * 1024**2, "A decoded records too large")
    require(sha(path) == A_ZIP, "A archive changed during read")
    return decode(records), catalog, splits


def package(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == SCHEMA, "Package schema differs")
    entries = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in entries), "Symlink package member")
    actual = {p.relative_to(root).as_posix() for p in entries if p.is_file()}
    require(actual == set(manifest["files"]) | {"manifest.json", "SHA256SUMS"}, "Package members differ")
    for name, row in manifest["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe member path")
        require((root / name).stat().st_size == row["bytes"] and sha(root / name) == row["sha256"],
                "Package member identity differs")
    supplied = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        value, name = line.split("  ", 1)
        require(name not in supplied, "Duplicate checksum entry")
        supplied[name] = value
    require(supplied == {n: sha(root / n) for n in actual if n != "SHA256SUMS"}, "Checksums differ")
    return sha(root / "manifest.json")


def publish(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".partial")
    require(not path.exists() and not path.is_symlink(), "Output already exists")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    require(not args.output.resolve().is_relative_to(root), "Output must be outside package")
    pin = package(root)
    baseline, catalog, splits = old_archive(args.initialization_zip)
    data = read_json(root / "records.json.gz")
    expected = read_json(root / "expected.json")
    result = reconstruct(data, baseline, catalog, splits, expected)
    require(package(root) == pin, "Package changed during reconstruction")
    output = {"schema_version": SCHEMA, "status": "numeric_reconstruction_pass", "comparison": result,
              "a_archive_sha256": A_ZIP, "manifest_sha256": pin,
              "checks": {"selected_test_records": 30, "compact_histories": 3000, "regional_differences": 5},
              "metric_atol": ATOL, "rtol": 0,
              "scope": {"training_or_inference": False, "state_replay": False,
                        "epoch_prediction_vectors": False, "batch_arithmetic": False,
                        "probability_generation": False, "convergence_claim": False}}
    publish(args.output, output)
    print(json.dumps({"status": output["status"], "output": str(args.output), "sha256": sha(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
