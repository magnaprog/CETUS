"""Check saved DOFA probabilities, scalar-history selection and prediction metrics.

Standard library only. Package integrity is not proof of scientific execution.
Parsing/publication patterns adapt the MIT-licensed CETUS loss companion.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct

SCHEMA = "dofa-partial-scores-numeric-v1"
MARKER = "synthetic-only-not-scientific-acceptance"
PRODUCER = "3a91cc7f19ab6c3f91b9015f5e9e09b97364aaf0"
CONTRACT_SHA = "7eecb0d70e467c222a501b96150c502ab606ec318a207569729426f354ee5d39"
CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
BANDS = [13.78, 3.75]
ARMS = ["13.78", "3.75", "difference_3.75_minus_13.78"]
ATOL = 1e-12
FILE_LIMIT = 64 * 1024**2  # Inherited operational cap, never enlarged by a manifest.
META_LIMIT = 1024**2
SUBSTANTIVE = {"README.md", "LICENSE", "NOTICE.md", "reconstruct.py", "test_reconstruct.py",
               "records.json.gz", "expected.json", "provenance.json"}
MEMBERS = SUBSTANTIVE | {"manifest.json", "SHA256SUMS"}
LOCATORS = {"tile_ids": "/test/tile_ids", "labels": "/test/labels", "predictions": "/test/predictions",
            "selected_epoch": "/cell_rows/{cell}/selection_amp/selected_epoch",
            "cell_metrics": "/cells/{cell}/recomputed_test",
            "summaries": ["/paired_seed_rows", "/fold_band_rows", "/paired_fold_rows", "/overall"],
            "probabilities": "/test/probabilities", "history": "/history", "selection": "/selection",
            "best_validation_macro_recall": "/best_validation_macro_recall"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names):
    require(type(value) is dict and set(value) == set(names), "Unexpected fields: " + repr(names))


def integer(value, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), "Invalid integer identity/count")


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid finite metric")
    return value


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def hex_sha(value):
    require(type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Invalid SHA256")


def identity(actual, expected):
    """Exact JSON identity, preserving integer/float/bool distinctions."""
    require(type(actual) is type(expected), "Identity type differs")
    if type(expected) is dict:
        require(actual.keys() == expected.keys(), "Identity fields differ")
        for key in expected:
            identity(actual[key], expected[key])
    elif type(expected) is list:
        require(len(actual) == len(expected), "Identity length differs")
        for a, e in zip(actual, expected):
            identity(a, e)
    else:
        require(actual == expected, "Identity/order differs")


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_float=lambda s: finite(float(s)), parse_constant=lambda s: finite(float(s)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def truth_digest(tile_ids, labels):
    return sha256(canonical({"tile_ids": tile_ids, "labels": labels}))


def marker(value, synthetic, required):
    fields(value, set(required) | ({"fixture_purpose"} if synthetic else set()))
    if synthetic:
        require(value["fixture_purpose"] == MARKER, "Exact synthetic marker required")


def header(value, schema, synthetic, extra):
    fixed = {"schema_version": schema, "class_order": CLASSES, "band_order": BANDS,
             "fold_order": list(range(5)), "seed_order": list(range(5)), "units": "fractions"}
    marker(value, synthetic, set(fixed) | set(extra))
    identity({k: value[k] for k in fixed}, fixed)


def binding(value):
    fields(value, {"sha256", "bytes"})
    hex_sha(value["sha256"]); integer(value["bytes"], 1)


def read_bounded(path, limit):
    require(stat.S_ISREG(path.lstat().st_mode), "Package member is not a regular file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, "File exceeds operational cap or is not regular")
        raw = stream.read(limit + 1)
    require(len(raw) <= limit and len(raw) == info.st_size, "File size changed/exceeded cap")
    return raw


def package(root, synthetic):
    require({p.name for p in root.iterdir()} == MEMBERS, "Exactly ten package members required")
    manifest_raw = read_bounded(root / "manifest.json", META_LIMIT)
    manifest = decode(manifest_raw)
    marker(manifest, synthetic, {"schema_version", "files"})
    require(manifest["schema_version"] == "dofa-partial-scores-package-v1", "Manifest schema differs")
    fields(manifest["files"], SUBSTANTIVE)
    hashes, data = {}, {}
    for name in sorted(SUBSTANTIVE):
        bound = manifest["files"][name]
        binding(bound)
        require(bound["bytes"] <= FILE_LIMIT, "Declared size exceeds operational cap")
        raw = read_bounded(root / name, FILE_LIMIT)
        require(len(raw) == bound["bytes"] and sha256(raw) == bound["sha256"], "Member bytes/hash differ: " + name)
        hashes[name] = bound["sha256"]
        if name in {"records.json.gz", "expected.json", "provenance.json"}:
            data[name] = raw
    hashes["manifest.json"] = sha256(manifest_raw)
    sums = "".join(hashes[name] + "  " + name + "\n" for name in sorted(hashes)).encode()
    require(read_bounded(root / "SHA256SUMS", META_LIMIT) == sums, "Checksum coverage/content differs")
    with gzip.GzipFile(fileobj=io.BytesIO(data.pop("records.json.gz"))) as stream:
        decoded = stream.read(FILE_LIMIT + 1)
    require(len(decoded) <= FILE_LIMIT, "Decoded records exceed operational cap")
    return decode(decoded), decode(data["expected.json"]), decode(data["provenance.json"]), hashes["manifest.json"]


def provenance(value, synthetic):
    marker(value, synthetic, {"schema_version", "producer_commit", "contract", "checker_report", "aggregate", "exporter",
                              "fixed_test_sources", "cell_sources", "field_locators", "accepted_freeze", "reader"})
    require(value["schema_version"] == "dofa-partial-scores-provenance-v1" and value["producer_commit"] == PRODUCER, "Producer identity differs")
    for key in ("contract", "checker_report", "aggregate", "exporter", "accepted_freeze", "reader"):
        binding(value[key])
    identity(value["contract"], {"sha256": CONTRACT_SHA, "bytes": 43796})
    require(type(value["fixed_test_sources"]) is list and len(value["fixed_test_sources"]) == 5, "Five truth sources required")
    for f, row in enumerate(value["fixed_test_sources"]):
        fields(row, {"fold", "catalog", "split"}); identity(row["fold"], f)
        binding(row["catalog"]); binding(row["split"])
    require(type(value["cell_sources"]) is list and len(value["cell_sources"]) == 50, "Fifty cell sources required")
    for i, row in enumerate(value["cell_sources"]):
        fields(row, {"cell", "result", "replay", "history", "selected_epoch"}); identity(row["cell"], i)
        for key in ("result", "replay", "history"):
            binding(row[key])
        integer(row["selected_epoch"], 1, 50)
    identity(value["field_locators"], LOCATORS)


def metrics(labels, predictions):
    require(len(labels) == len(predictions) and labels, "Prediction length differs or population is empty")
    counts = [[0]*6 for _ in range(6)]
    for label, prediction in zip(labels, predictions):
        integer(label, 0, 5); integer(prediction, 0, 5)
        counts[label][prediction] += 1
    support = [sum(row) for row in counts]
    predicted = [sum(row[c] for row in counts) for c in range(6)]
    tp = [counts[c][c] for c in range(6)]
    ratio = lambda a, b: a / b if b else 0.0
    pc = {"precision": [ratio(tp[c], predicted[c]) for c in range(6)],
          "recall": [ratio(tp[c], support[c]) for c in range(6)],
          "f1": [ratio(2*tp[c], support[c]+predicted[c]) for c in range(6)]}
    return {"confusion_counts": counts, "support": support, "prediction_counts": predicted,
            "scores": {"macro_f1": math.fsum(pc["f1"])/6, "balanced_accuracy": math.fsum(pc["recall"])/6, "per_class": pc}}


def componentwise(values, operation):
    return {"macro_f1": operation([v["macro_f1"] for v in values]),
            "balanced_accuracy": operation([v["balanced_accuracy"] for v in values]),
            "per_class": {k: [operation([v["per_class"][k][c] for v in values]) for c in range(6)]
                          for k in ("precision", "recall", "f1")}}


def fold_summary(values):
    require(len(values) == 5, "Five geographic values required")
    mean = math.fsum(values) / 5
    return {"mean": mean, "sample_std": math.sqrt(math.fsum((x-mean)**2 for x in values)/4), "fold_values": values}


def compare(actual, expected, path="comparison"):
    """The computed structure fixes the allowed expected fields and leaf types."""
    if type(actual) is dict:
        fields(expected, actual)
        for key in actual:
            compare(actual[key], expected[key], path + "/" + key)
    elif type(actual) is list:
        require(type(expected) is list and len(expected) == len(actual), "Expected list differs: " + path)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path + "/" + str(i))
    elif type(actual) is float and not path.endswith("/band"):
        require(abs(actual - finite(expected)) <= ATOL, "Expected metric differs: " + path)
    else:
        identity(expected, actual)


def check_probabilities(probabilities, predictions):
    """Check supplied maxima, not logit argmax order or native projection fidelity."""
    require(type(probabilities) is list and len(probabilities) == len(predictions), "Probability rows differ")
    ties = 0
    for row, prediction in zip(probabilities, predictions):
        require(type(row) is list and len(row) == 6, "Six probability columns required")
        integer(prediction, 0, 5)
        for value in row:
            require(0 <= finite(value) <= 1, "Probability outside unit interval")
            require(struct.unpack("!f", struct.pack("!f", value))[0] == value,
                    "Probability is not lossless binary32")
        require(abs(math.fsum(row) - 1) <= 1e-6, "Probability simplex differs")
        maximum = max(row)
        require(row[prediction] == maximum, "Prediction is not a probability maximum")
        ties += int(sum(value == maximum for value in row) > 1)
    return ties


def check_history(cell, train_rows):
    """Verify policy against retained scalars; do not reconstruct validation metrics."""
    integer(train_rows, 1)
    history = cell["history"]
    fields(history, {"train_loss", "val_balanced_accuracy", "optimizer_steps", "amp_skipped_steps", "loss_scale"})
    for key, values in history.items():
        require(type(values) is list and len(values) == 50, "Fifty history epochs required")
        for value in values:
            if key in ("optimizer_steps", "amp_skipped_steps"):
                integer(value, int(key == "optimizer_steps"))
            else:
                require(finite(value) >= 0, "Negative history scalar")
                require(key != "val_balanced_accuracy" or value <= 1, "Validation recall outside unit interval")
                require(key != "loss_scale" or value > 0, "Nonpositive loss scale")
    steps, skipped = history["optimizer_steps"], history["amp_skipped_steps"]
    attempts = (train_rows + 31) // 32
    require(all(a+b == attempts for a, b in zip(steps, skipped)), "Per-epoch update budget differs")
    values = history["val_balanced_accuracy"]
    # All epochs have positive updates in this release; max retains the earliest exact tie.
    chosen = max(range(50), key=lambda i: values[i])
    selected = {"selected_epoch": chosen+1, "epoch_indexing": "one based",
                "optimizer_attempts_per_epoch": attempts, "optimizer_steps_total": sum(steps),
                "amp_skipped_steps_total": sum(skipped), "training_example_presentations": 50*train_rows}
    identity(cell["selection"], selected)
    best = finite(cell["best_validation_macro_recall"])
    require(0 <= best <= 1 and abs(best-values[chosen]) <= ATOL, "Best validation scalar differs")
    return {**selected, "selected_validation_macro_recall": float(values[chosen])}


def reconstruct(records, expected, source, synthetic=False):
    header(records, SCHEMA, synthetic, {"folds", "cells"})
    header(expected, "dofa-partial-scores-expected-v1", synthetic, {"fixed_tests", "comparison", "score_checks", "selection_checks"})
    provenance(source, synthetic)
    require(type(records["folds"]) is list and len(records["folds"]) == 5
            and type(expected["fixed_tests"]) is list and len(expected["fixed_tests"]) == 5, "Five ordered folds required")
    for f, (fold, fixed) in enumerate(zip(records["folds"], expected["fixed_tests"])):
        fields(fold, {"fold", "region", "tile_ids", "labels", "train_rows"})
        fields(fixed, {"fold", "region", "rows", "ordered_truth_sha256", "train_rows"})
        integer(fold["train_rows"], 1); identity(fold["train_rows"], fixed["train_rows"])
        for row in (fold, fixed):
            identity(row["fold"], f); identity(row["region"], f+1)
        ids, labels = fold["tile_ids"], fold["labels"]
        require(type(ids) is list and ids and all(type(i) is str and i for i in ids)
                and len(set(ids)) == len(ids), "Invalid/duplicate IDs")
        require(type(labels) is list and len(labels) == len(ids), "Truth length differs")
        for label in labels:
            integer(label, 0, 5)
        identity(fixed["rows"], len(ids)); hex_sha(fixed["ordered_truth_sha256"])
        require(truth_digest(ids, labels) == fixed["ordered_truth_sha256"], "Ordered truth digest differs")
    require(type(records["cells"]) is list and len(records["cells"]) == 50, "Fifty ordered cells required")
    require(all(type(expected[k]) is list and len(expected[k]) == 50
                for k in ("score_checks", "selection_checks")), "Fifty expected check rows required")
    rows, score_checks, selection_checks = [], [], []
    for i, cell in enumerate(records["cells"]):
        fields(cell, {"cell", "fold", "seed", "band", "predictions", "probabilities", "history",
                      "selection", "best_validation_macro_recall"})
        key = {"cell": i, "fold": i//10, "seed": (i//2)%5, "band": BANDS[i%2]}
        identity({k: cell[k] for k in key}, key)
        require(type(cell["predictions"]) is list, "Prediction list required")
        rows.append({**key, **metrics(records["folds"][i//10]["labels"], cell["predictions"])})
        score = {**key, "probability_tie_rows": check_probabilities(cell["probabilities"], cell["predictions"])}
        selection = {**key, **check_history(cell, records["folds"][i//10]["train_rows"])}
        identity(selection["selected_epoch"], source["cell_sources"][i]["selected_epoch"])
        compare(score, expected["score_checks"][i], "score_checks")
        compare(selection, expected["selection_checks"][i], "selection_checks")
        score_checks.append(score); selection_checks.append(selection)
    pairs = [{"fold": j//5, "seed": j%5, "cells": [2*j, 2*j+1],
              "difference": componentwise([rows[2*j]["scores"], rows[2*j+1]["scores"]], lambda v: v[1]-v[0])} for j in range(25)]
    folds, differences = [], []
    for f in range(5):
        for arm, band in enumerate(BANDS):
            cells = [10*f+2*s+arm for s in range(5)]
            folds.append({"fold": f, "band": band, "seeds": list(range(5)), "cells": cells,
                          "scores": componentwise([rows[i]["scores"] for i in cells], lambda v: math.fsum(v)/5)})
        differences.append({"fold": f, "seeds": list(range(5)),
                            "difference": componentwise([p["difference"] for p in pairs[5*f:5*f+5]], lambda v: math.fsum(v)/5)})
    overall = {str(band): componentwise([r["scores"] for r in folds if r["band"] == band], fold_summary) for band in BANDS}
    overall[ARMS[2]] = componentwise([r["difference"] for r in differences], fold_summary)
    result = {"cell_rows": rows, "paired_seed_rows": pairs, "fold_band_rows": folds, "paired_fold_rows": differences, "overall": overall}
    compare(result, expected["comparison"])
    return {"comparison": result, "score_checks": score_checks, "selection_checks": selection_checks}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_new(path, data):
    stream = path.open("xb")
    try:
        require(stream.write(data) == len(data), "Short output write")
        stream.flush()
        os.fsync(stream.fileno())
    except BaseException as error:
        try:
            stream.close()
        except Exception as closing:
            error.add_note("Close also failed: " + repr(closing))
        raise
    else:
        stream.close()


def run(root, output, synthetic=False):
    root, output = root.resolve(), output.absolute()
    partial, failure = Path(str(output)+".partial"), Path(str(output)+".failure.json")
    require(not output.resolve().is_relative_to(root), "Output must be outside package")
    require(all(not p.exists() and not p.is_symlink() for p in (output, partial, failure)), "Fresh output/attempt paths required")
    try:
        records, expected, source, manifest_sha = package(root, synthetic)
        reader_raw = read_bounded(root / "reconstruct.py", FILE_LIMIT)
        identity(source["reader"], {"sha256": sha256(reader_raw), "bytes": len(reader_raw)})
        result = reconstruct(records, expected, source, synthetic)
        require(package(root, synthetic)[3] == manifest_sha, "Package changed during reconstruction")
        report = {"schema_version": "dofa-partial-scores-reconstruction-v1",
                  "status": "synthetic_saved_record_checks_pass" if synthetic else "saved_record_checks_pass",
                  "synthetic": synthetic, "package_manifest_sha256": manifest_sha, "metric_atol": ATOL, "metric_rtol": 0,
                  "units": "fractions", **result, "provenance": source,
                  "scope": {"saved_prediction_metrics": True, "seed_then_region_aggregation": True,
                            "probabilities_regenerated": False, "probability_maxima_checked": True,
                            "selection_policy_from_scalar_history_checked": True,
                            "validation_metrics_recomputed": False, "average_precision_computed": False,
                            "native_projection_authenticated": False, "training_or_inference": False,
                            "state_replay": False, "execution_authenticated": False}}
        if synthetic:
            report["fixture_purpose"] = MARKER
        write_new(partial, encoded(report))
        os.link(partial, output)  # Commit only the closed complete file; refuse an existing name.
    except BaseException as error:
        try:
            write_new(failure, encoded({"status": "failed", "synthetic": synthetic, "error_type": type(error).__name__, "error": str(error)}))
        except Exception as diagnostic:
            error.add_note("Failure receipt could not be written: " + repr(diagnostic))
        raise
    try:
        partial.unlink()
    except OSError:
        pass  # Residual identical staging bytes do not revoke committed publication.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)
    result = run(Path(__file__).resolve().parent, args.output, args.synthetic)
    print(json.dumps({"schema_version": result["schema_version"], "status": result["status"], "synthetic": args.synthetic,
                      "fixture_purpose": result.get("fixture_purpose"), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
