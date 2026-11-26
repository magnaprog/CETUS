"""Count Craters predictions from logistic regression on eight feature sets.

Count correct predictions on the 39 test tiles where every known map cell
carries the Craters label, both before and after restricting cells to those
with four valid radar pixels. Across all five test regions, also count correct
Craters predictions among 132 Craters tiles and false alarms among 23,114 tiles
with other map labels. Write results.json and a LaTeX table.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


INPUT_SHA256 = {
    "crater_results": "c0f1d60faaedc9b083be115e1468e97f25f40c676d0e9a25651076dec1759716",
    "predictions": "fa6f8a56a67c8f3a36020724b6aa0f4a32a933598089ca45f6bbbe692c2636b9",
}
CRATER_SOURCE_SHA256 = "984020ced2828392d0793734924922001a950400cac636f313b6421460b41736"
CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
TEST_COUNTS = [3660, 4497, 5721, 4748, 4620]
CRATER_COUNTS = [8, 1, 14, 61, 48]
SUBSET_COUNTS = [0, 0, 0, 23, 16]
FAMILIES = [
    ("common", "dinov2", "DINOv2 (primary extraction)"),
    ("common", "dofa", "DOFA (13.78)"),
    ("common", "croma", "CROMA"),
    ("common", "random_init", "Primary untrained ViT"),
    ("classical", "raw_intensity_statistics", "Intensity statistics"),
    ("classical", "intensity_histogram", "Intensity histogram"),
    ("classical", "texture_glcm_lbp_gabor_hog_wavelet", "Texture"),
    ("classical", "combined_classical", "Combined classical"),
]
MODES = ("baseline", "all_four")
ARMS = [("dinov2_pretrained", 42)] + [
    (model, seed) for model in ("resnet18_pretrained", "resnet18_scratch")
    for seed in range(3)
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(value, names, context):
    require(isinstance(value, dict) and set(value) == set(names),
            f"Unexpected {context} schema")


def class_vector(values, length, context):
    require(isinstance(values, list) and len(values) == length
            and all(type(x) is int and 0 <= x < 6 for x in values),
            f"Invalid {context}: expected {length} integer class indices")


def load_inputs(crater_results, predictions):
    """Check both byte identities before decoding either input; no hash override."""
    raw = {"crater_results": Path(crater_results).read_bytes(),
           "predictions": Path(predictions).read_bytes()}
    bindings = {}
    for name, data in raw.items():
        digest = hashlib.sha256(data).hexdigest()
        require(digest == INPUT_SHA256[name], f"Input SHA256 differs: {name}")
        bindings[name] = {"sha256": digest, "bytes": len(data)}
    return (json.loads(raw["crater_results"]),
            json.loads(gzip.decompress(raw["predictions"])), bindings)


def validate_predictions(predictions):
    exact_keys(predictions, ("class_names", "folds", "schema_version", "source_audit_sha256"),
               "primary predictions")
    require(predictions["schema_version"] == "1.0.0"
            and predictions["class_names"] == CLASSES, "Prediction schema/class order differs")
    folds = predictions["folds"]
    require(isinstance(folds, list) and len(folds) == 5, "Expected all five folds")
    seen = set()
    for f, fold in enumerate(folds):
        exact_keys(fold, ("classical", "common", "controls", "fold", "test_ids", "true_labels"),
                   "prediction fold")
        require(type(fold["fold"]) is int and fold["fold"] == f, "Fold order differs")
        ids, labels = fold["test_ids"], fold["true_labels"]
        require(isinstance(ids, list) and len(ids) == TEST_COUNTS[f]
                and all(isinstance(t, str) and t for t in ids), "Test ID count/type differs")
        require(ids == sorted(ids) and len(set(ids)) == len(ids), "Test ID order/uniqueness differs")
        require(not seen.intersection(ids), "Test IDs overlap across folds")
        seen.update(ids)
        class_vector(labels, len(ids), "true labels")
        require(labels.count(5) == CRATER_COUNTS[f], "Modal-Craters denominator differs")
        for group in ("common", "classical"):
            exact_keys(fold[group], [name for g, name, _ in FAMILIES if g == group], group)
            for name, values in fold[group].items():
                class_vector(values, len(ids), f"{name} predictions")
    return folds


def derive_subset(crater, folds):
    """Verify every historical cohort, but derive eligibility from map counts."""
    exact_keys(crater, ("status", "python", "numpy", "source_sha256", "inputs", "class_order",
                       "training_batch_availability", "map_prediction_cohorts", "image_rows",
                       "scope", "limits"), "Craters result")
    require(crater["status"] == "complete" and crater["class_order"] == CLASSES
            and crater["source_sha256"] == CRATER_SOURCE_SHA256, "Craters source/class order differs")
    crater_ids = [{t for t, y in zip(f["test_ids"], f["true_labels"]) if y == 5} for f in folds]
    expected = {(model, f, seed, mode) for model, seed in ARMS
                for f in range(5) for mode in MODES}
    rows, cohorts = crater["image_rows"], crater["map_prediction_cohorts"]
    require(isinstance(rows, list) and len(rows) == 1848
            and isinstance(cohorts, list) and len(cohorts) == 70, "Historical cohort size differs")
    groups, maps, saved_predictions = {}, {}, {}
    for row in rows:
        exact_keys(row, ("model", "fold", "seed", "tile_id", "mode", "prediction", "correct",
                         "class_counts", "known_cells", "unknown_cells", "craters_fraction",
                         "predicted_class_fraction", "all_known_cells_craters"), "map row")
        model, f, seed, mode = (row[k] for k in ("model", "fold", "seed", "mode"))
        require(isinstance(model, str) and isinstance(mode, str)
                and type(f) is int and type(seed) is int, "Invalid cohort identity types")
        key = (model, f, seed, mode)
        require(key in expected, "Unexpected historical cohort")
        tid = row["tile_id"]
        require(isinstance(tid, str) and tid in crater_ids[f], "Map ID/label does not match primary truth")
        group = groups.setdefault(key, {})
        require(tid not in group, "Duplicate map ID within cohort")
        counts, known, unknown, pred = (row[k] for k in
                                       ("class_counts", "known_cells", "unknown_cells", "prediction"))
        require(isinstance(counts, list) and len(counts) == 6
                and all(type(x) is int and x >= 0 for x in counts), "Invalid map class counts")
        require(type(known) is int and type(unknown) is int and known >= 0 and unknown >= 0
                and sum(counts) == known, "Map known-cell denominator differs")
        class_vector([pred], 1, "historical prediction")
        eligible = known > 0 and counts[5] == known
        require(row["all_known_cells_craters"] is eligible and row["correct"] is (pred == 5),
                "Map eligibility/correct flag disagrees with counts/prediction")
        for field, numerator in (("craters_fraction", counts[5]), ("predicted_class_fraction", counts[pred])):
            value = row[field]
            require((type(value) in (int, float) and value == numerator / known) if known else value is None,
                    "Map fractions disagree with counts/denominator")
        map_key, value = (f, tid, mode), (tuple(counts), known, unknown, eligible)
        require(maps.setdefault(map_key, value) == value, "Map counts differ across model/seed cohorts")
        prediction_key = (model, f, seed, tid)
        require(saved_predictions.setdefault(prediction_key, pred) == pred,
                "Historical prediction differs between map modes")
        group[tid] = (pred, counts, known, eligible)
    require(set(groups) == expected, "Missing historical cohort")
    seen = set()
    for cohort in cohorts:
        exact_keys(cohort, ("model", "fold", "seed", "mode", "craters_test_images", "correct",
                            "undefined_map_fraction", "errors", "errors_with_predicted_class_present",
                            "all_known_cells_craters", "correct_all_known_cells_craters"), "cohort summary")
        require(isinstance(cohort["model"], str) and isinstance(cohort["mode"], str)
                and type(cohort["fold"]) is int and type(cohort["seed"]) is int,
                "Invalid summary identity types")
        key = tuple(cohort[k] for k in ("model", "fold", "seed", "mode"))
        require(key in expected and key not in seen, "Missing/duplicate/unexpected cohort summary")
        seen.add(key)
        group = groups[key]
        require(set(group) == crater_ids[key[1]], "Incomplete historical cohort IDs")
        values = list(group.values())
        actual = {
            "craters_test_images": len(values),
            "correct": sum(p == 5 for p, c, n, e in values),
            "undefined_map_fraction": sum(n == 0 for p, c, n, e in values),
            "errors": sum(p != 5 for p, c, n, e in values),
            "errors_with_predicted_class_present": sum(p != 5 and c[p] > 0 for p, c, n, e in values),
            "all_known_cells_craters": sum(e for p, c, n, e in values),
            "correct_all_known_cells_craters": sum(e and p == 5 for p, c, n, e in values),
        }
        require(all(type(cohort[k]) is int and cohort[k] == v for k, v in actual.items()),
                "Cohort summary disagrees with image rows")
    require(seen == expected, "Incomplete cohort summaries")
    subsets = []
    for f in range(5):
        by_mode = [{tid for tid in crater_ids[f] if maps[f, tid, mode][3]} for mode in MODES]
        require(by_mode[0] == by_mode[1], "Map modes select different subset IDs")
        require(len(by_mode[0]) == SUBSET_COUNTS[f], "Single-map-class denominator differs")
        for tid in crater_ids[f]:
            base, masked = maps[f, tid, "baseline"], maps[f, tid, "all_four"]
            require(all(a <= b for a, b in zip(masked[0], base[0]))
                    and masked[2] <= base[2], "Masked counts exceed baseline counts")
        subsets.append(by_mode[0])
    return subsets


def summarize(crater, predictions):
    """Validate decoded inputs and compute counts; production entry is load_inputs."""
    folds = validate_predictions(predictions)
    subsets = derive_subset(crater, folds)
    rows = []
    for group, family, name in FAMILIES:
        regional = []
        for f, fold in enumerate(folds):
            triples = list(zip(fold["test_ids"], fold["true_labels"], fold[group][family]))
            tp = sum(y == 5 and p == 5 for t, y, p in triples)
            fp = sum(y != 5 and p == 5 for t, y, p in triples)
            regional.append({"subset_correct": sum(t in subsets[f] and p == 5 for t, y, p in triples),
                             "true_positives": tp, "false_positives": fp, "craters_predictions": tp + fp})
        rows.append({"family": family, "display_name": name,
                     "region_4": regional[3], "region_5": regional[4],
                     "pooled": {k: sum(r[k] for r in regional) for k in regional[0]}})
    return {
        "schema_version": 1,
        "scope": "Exploratory CPU join on the same observed test sets; no causal claim or new geographic validation.",
        "class_order": CLASSES,
        "recipe": "All eight primary standardized class-balanced C=1 LBFGS logistic-regression feature families.",
        "inherited_provenance": "Catalog/split/census correspondence and fitting recipe are validated externally for the exact saved packages; this script reads only the two prediction/result files, not a catalog, NPZ or fit metadata.",
        "classifier_seed": 0,
        "primary_untrained_vit_encoder_seed": 42,
        "dino_identity": "Primary frozen extraction, separate from matched-initialization pretrained DINOv2; primary untrained ViT is not matched-random DINOv2.",
        "population": {"test_tiles": sum(TEST_COUNTS), "modal_craters": sum(CRATER_COUNTS),
                       "non_craters": sum(TEST_COUNTS) - sum(CRATER_COUNTS), "subset": sum(SUBSET_COUNTS)},
        "regions": [{"region": f + 1, "saved_fold": f, "test_tiles": TEST_COUNTS[f],
                     "modal_craters": CRATER_COUNTS[f], "non_craters": TEST_COUNTS[f] - CRATER_COUNTS[f],
                     "subset": len(subsets[f]), "subset_ids": sorted(subsets[f])} for f in range(5)],
        "definitions": {"subset": "Known cells > 0 and all six class counts sum to the Craters count; same IDs in baseline and all_four. Unknown cells remain separate.",
                        "true_positives": "Craters predictions on modal-Craters test tiles; pooled denominator 132.",
                        "false_positives": "Craters predictions on other modal map classes; pooled denominator 23114.",
                        "pooled": "Integer sums across all five disjoint test regions, not regional mean F1 or physical-crater counts."},
        "rows": rows,
    }


def render_tex(result):
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Feature family & \shortstack{Region 4\\ /23} & \shortstack{Region 5\\ /16} & \shortstack{Total\\ /39} & \shortstack{True positives\\ /132} & \shortstack{False positives\\ /23,114} \\",
             r"\midrule"]
    for row in result["rows"]:
        pooled = row["pooled"]
        values = [row["region_4"]["subset_correct"], row["region_5"]["subset_correct"],
                  pooled["subset_correct"], pooled["true_positives"], pooled["false_positives"]]
        lines.append(row["display_name"] + " & " + " & ".join(str(x) for x in values) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def write_outputs(output_dir, result):
    """Serialize before reserving a new directory; never overwrite an existing path."""
    payloads = {"results.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                "crater-map-primary-lr.tex": render_tex(result)}
    output_dir = Path(output_dir)
    output_dir.mkdir()  # Exclusive, including existing directories or dangling symlinks.
    created = []
    try:
        for name, content in payloads.items():
            path = output_dir / name
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(content)
    except BaseException:
        # Leave any files another process created in this directory alone.
        for path in created:
            path.unlink(missing_ok=True)
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crater-results", required=True, type=Path,
                        help="results.json from the Craters map-composition analysis")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="experiments/regional-predictions/results/predictions.json.gz")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New output directory inside an existing parent directory")
    args = parser.parse_args(argv)
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "Output directory must be new")
    crater, predictions, bindings = load_inputs(args.crater_results, args.predictions)
    result = summarize(crater, predictions)
    result["inputs"] = bindings
    result["adapter_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_outputs(args.output_dir, result)
    print(json.dumps({"status": "complete", "families": len(result["rows"]),
                      "population": result["population"]}, sort_keys=True))


if __name__ == "__main__":
    main()
