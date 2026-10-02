"""Export the verified September 29 frozen Titan evaluation, without operational records.

Fixed scientific schemas only. No upload, code export, image tiles or fine-tuning.
The output directory must be new. --verify-only checks only the package itself.
"""

import argparse
import copy
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TRACK = "TitanSAR-HiSAR-IMG-longitude-review-20260929"
CATALOG_SHA256 = "2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec"
RESULT_SHA256 = "db352133220ce64b71f68e79269f1f93407d1906ad6545c06797eadd24a092b2"
PROTOCOL_SHA256 = "ff554e9ee599eebc11753b0a798b288e1daeffbbf35da13a25a982d1e06c01e2"
MODELS = ("dinov2", "dofa", "croma", "random_init")
CLASSES = ("plains", "dunes", "hummocky", "labyrinths", "lakes", "craters")
ROLES = ("train", "val", "test", "selk_holdout", "buffer_excluded")
METRICS = ("precision", "recall", "f1")
SOURCES = {
    "frozen_results": "audit/REPAIRED_TITAN_RESULTS_2026-09-29.json",
    "protocol": "audit/gpu_execution/repaired_splits/protocol.json",
    "comparison": "audit/gpu_execution/catalog_comparison.json",
    "labels": "audit/LABEL_PROVENANCE_2026-09-29.json",
    "crater_support": "audit/REPAIRED_CRATER_SUPPORT_2026-09-29.json",
}
CATALOG_FIELDS = "schema_version benchmark_track domain source_product tile_size footprint_size_m target_pixel_size_m num_classes class_names labels_available intensity_space"
TILE_FIELDS = "tile_id tile_sha256 center_lat center_lon valid_fraction mean_display_dn std_display_dn label label_confidence mean_incidence_angle has_vims_display_sample vims_display_sample has_tb brightness_temp"
PROVENANCE_FIELDS = "hisar_source_sha256 label_source_sha256 vims_source_sha256 label_sampling"
SPEC_FIELDS = "catalog_sha256 evaluation_scope geometry folds buffer_policy role_rule boundary_rule buffer_rule class_order num_classes encoders execution probe normalization reporting prospective_constraint earth_transfer"
RESULT_FIELDS = "schema_version benchmark_track class_names classifier_outputs comparison_scope dofa_identifier_ghz evaluation_scope aggregation uncertainty zero_division metric_atol probability_sum_atol random_encoder_seed random_state_reference_sha256 validation limitations source_file_sha256"
COMPARISON_FIELDS = "schema_version status scope old new added removed retained_count all_retained_pixels_and_centers_identical retained_label_transitions changed_labels changed_modal_fractions new_label_sampling"
SAVED_LABEL_FIELDS = "status raw_grid_centers_regenerated raw_grid_center_digest raw_grid_center_digest_matches_remote_independent_validation catalog_tiles_checked label_mismatches modal_fraction_mismatches center_mismatch_ids invalid_or_duplicate_ids catalog_tile_hash_vs_manifest_mismatches class_counts retained_count added_count removed_count changed_label_count changed_modal_fraction_count changed_label_ids_match_prior_direct_raster_prediction changed_modal_fraction_ids_match_prior_direct_raster_prediction saved_catalog_expectation_digest saved_catalog_expectation_digest_format"
PAYLOADS = {
    "README.md", "catalogs/titan_catalog.json", "splits/protocol.json",
    "results/frozen_results.json", "results/predictions.json.gz",
    "evidence/catalog_comparison.json", "evidence/label_verification.json",
    "evidence/crater_support.json", "provenance.json",
    *(f"splits/titan_contiguous_fold_{f}.json" for f in range(5)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value).rstrip(b"\n")).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    value = json.loads(raw, object_pairs_hook=pairs)
    encoded(value)  # Reject nonfinite numbers even in deeply nested arrays.
    return value


def pick(value, fields):
    """Select an explicit schema; missing scientific fields are an error."""
    return {key: copy.deepcopy(value[key]) for key in fields.split()}


def check_public(value):
    # Secondary fail-closed check, not a substitute for the fixed field selections.
    text = json.dumps(value)
    for marker in ("/home/", "/Users/", "/mnt/", "/tmp/", "gpu-", "ssh "):
        require(marker not in text, f"Operational marker in scientific export: {marker}")


def write_json(root, name, value):
    check_public(value)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(value)
    if name.endswith(".gz"):
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
            stream.write(raw)
        raw = buffer.getvalue()
    path.write_bytes(raw)


def inventory(root, exclude=()):
    files = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), "Symlinks are not release payloads")
        if ".git" in p.relative_to(root).parts or not p.is_file():
            continue
        name = p.relative_to(root).as_posix()
        if name not in exclude:
            files[name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return files


def write_indexes(root, metadata):
    """Index excludes itself and SHA256SUMS; checksums exclude only themselves."""
    files = inventory(root, ("file_index.json", "SHA256SUMS"))
    write_json(root, "file_index.json", {**metadata, "files": files})
    files = inventory(root, ("SHA256SUMS",))
    (root / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in files.items()))


def verify_indexes(root):
    index = read_json(root / "file_index.json")
    require(index["files"] == inventory(root, ("file_index.json", "SHA256SUMS")), "Index inventory/hash mismatch")
    lines = (root / "SHA256SUMS").read_text().splitlines()
    expected = {name: row["sha256"] for name, row in inventory(root, ("SHA256SUMS",)).items()}
    actual = {}
    for line in lines:
        hash_value, name = line.split("  ", 1)
        require(name not in actual, "Duplicate checksum path")
        actual[name] = hash_value
    require(actual == expected, "Checksum inventory/hash mismatch")


def metrics(counts):
    a = np.asarray(counts)
    require(a.shape == (6, 6) and a.dtype.kind in "iu" and (a >= 0).all(), "Invalid confusion counts")
    support, predicted, tp = a.sum(1), a.sum(0), a.diagonal()
    require((support > 0).all(), "Missing class support")
    precision = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
    recall = tp / support
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(6), where=support + predicted > 0)
    n = a.sum()
    expected = float(np.dot(support.astype(float), predicted) / n ** 2)
    observed = float(tp.sum() / n)
    return dict(support=support.tolist(), prediction_counts=predicted.tolist(),
                macro={k: float(v.mean()) for k, v in zip(METRICS, (precision, recall, f1))},
                per_class={k: v.tolist() for k, v in zip(METRICS, (precision, recall, f1))},
                kappa=(observed - expected) / (1 - expected) if expected != 1 else float(observed == 1))


def close(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=1e-12, rtol=0), message)


def verify_package(root):
    """Recompute membership, label votes, 120 confusion matrices and all aggregates."""
    root = Path(root)
    verify_indexes(root)
    require(set(inventory(root)) == PAYLOADS | {"file_index.json", "SHA256SUMS"}, "Unexpected package files")
    for name in PAYLOADS:
        if name.endswith((".json", ".json.gz")):
            check_public(read_json(root / name))
    catalog = read_json(root / "catalogs/titan_catalog.json")
    protocol = read_json(root / "splits/protocol.json")
    results = read_json(root / "results/frozen_results.json")
    predictions = read_json(root / "results/predictions.json.gz")
    catalog_hash = sha256(root / "catalogs/titan_catalog.json")
    require(catalog["benchmark_track"] == TRACK and catalog["class_names"] == list(CLASSES), "Catalog semantics differ")
    rows = catalog["tiles"]
    ids = [r["tile_id"] for r in rows]
    require(len(ids) == len(set(ids)) and all(re.fullmatch(r"titan_\d{6}", i) for i in ids), "Invalid/duplicate tile IDs")
    labels = {r["tile_id"]: r["label"] for r in rows}
    require(all(type(v) is int and 0 <= v < 6 for v in labels.values()), "Invalid labels")
    require(protocol["catalog_sha256"] == results["catalog_sha256"] == predictions["catalog_sha256"] == catalog_hash, "Catalog binding differs")
    spec_hash = digest(protocol["specification"])
    require(protocol["specification_sha256"] == spec_hash and protocol["specification"]["catalog_sha256"] == catalog_hash, "Specification hash differs")
    require(results["protocol_file_sha256"] == sha256(root / "splits/protocol.json") and results["protocol_specification_sha256"] == spec_hash, "Results protocol binding differs")
    require(predictions["frozen_results_sha256"] == sha256(root / "results/frozen_results.json"), "Prediction result binding differs")
    require([f["fold"] for f in protocol["splits"]] == list(range(5)) and [f["fold"] for f in results["folds"]] == list(range(5)), "Fold identities differ")
    split_ids, union = {}, set()
    splits = []
    for f, item in enumerate(protocol["splits"]):
        require(item["path"] == f"titan_contiguous_fold_{f}.json", "Unexpected split path")
        split_path = root / "splits" / item["path"]
        require(item["sha256"] == sha256(split_path) == results["folds"][f]["split_sha256"], "Split hash differs")
        split = read_json(split_path)
        splits.append(split)
        require(split["catalog_sha256"] == catalog_hash and split["policy"]["protocol_specification_sha256"] == spec_hash, "Split binding differs")
        assignment = split["assignments"]
        require(set(assignment) == set(ids) and set(assignment.values()) <= set(ROLES), "Split membership differs")
        require(dict(Counter(assignment.values())) == split["counts"], "Split role counts differ")
        for role in ROLES:
            selected = [i for i in ids if assignment[i] == role]
            counts = np.bincount([labels[i] for i in selected], minlength=6).tolist()
            require(results["folds"][f]["counts"][role] == {"total": len(selected), "class_counts": counts}, "Fold class counts differ")
        selected = [i for i in ids if assignment[i] == "test"]
        require(selected == results["folds"][f]["test_ids"] and not union.intersection(selected), "Test membership/order differs")
        union.update(selected)
        split_ids[f] = selected
        normalization = results["folds"][f]["normalization_provenance"]
        require(all(assignment[i] == "train" for i in normalization["sampling"]["selected_tile_ids"]), "Normalization selected non-training tile")
    require(len(union) == protocol["test_union_count"], "Test union count differs")
    require(hashlib.sha256(json.dumps(sorted(union)).encode()).hexdigest() == protocol["test_union_sorted_ids_sha256"], "Test union digest differs")
    oracle = read_json(root / "evidence/label_verification.json")
    candidate_counts = oracle["candidate_class_counts"]
    for row in rows:
        counts = np.asarray(candidate_counts[int(row["tile_id"].split("_")[1])])
        require(counts.shape == (6,) and counts.dtype.kind in "iu" and (counts >= 0).all() and counts.sum() > 0, "Invalid map vote counts")
        require(int(counts.argmax()) == row["label"] and round(float(counts.max() / counts.sum()), 4) == row["label_confidence"], "Catalog label/modal fraction differs")
    jobs = {(j["model"], j["fold"]): j for j in predictions["jobs"]}
    require(len(jobs) == len(predictions["jobs"]) == 20 and set(jobs) == {(m, f) for m in MODELS for f in range(5)}, "Missing/duplicate prediction jobs")
    require(set(results["models"]) == set(MODELS) and results["validation"]["status"] == "pass", "Incomplete frozen results")
    for model in MODELS:
        block = results["models"][model]
        require(re.fullmatch(r"[0-9a-f]{64}", block["weights_sha256"]) and block["model_revision"], "Missing model pins")
        require([f["fold"] for f in block["folds"]] == list(range(5)), "Model folds differ")
        per_fold = {task: [] for task in ("linear_probe", "knn")}
        for f, fold in enumerate(block["folds"]):
            job = jobs[model, f]
            truth = np.asarray([labels[i] for i in split_ids[f]])
            require(job["tile_ids"] == split_ids[f] and job["true_labels"] == truth.tolist() and job["seeds"] == list(range(5)), "Prediction IDs/labels/seeds differ")
            require(len(job["probe_predictions"]) == len(fold["probe_heads"]) == 5, "Probe head count differs")
            require([h["seed"] for h in fold["probe_heads"]] == list(range(5)), "Probe seeds differ")
            computed = []
            for predicted, record in zip([*job["probe_predictions"], job["knn_predictions"]], [*fold["probe_heads"], fold["knn"]]):
                predicted = np.asarray(predicted)
                require(predicted.shape == truth.shape and predicted.dtype.kind in "iu" and np.isin(predicted, range(6)).all(), "Invalid prediction array")
                counts = np.bincount(6 * truth + predicted, minlength=36).reshape(6, 6)
                require(counts.tolist() == record["confusion_counts"], "Prediction confusion counts differ")
                checked = metrics(counts)
                for key in ("support", "prediction_counts", "kappa"):
                    close(record[key], checked[key], f"{key} differs")
                for scope in ("macro", "per_class"):
                    for metric in METRICS:
                        close(record[scope][metric], checked[scope][metric], f"{scope} {metric} differs")
                computed.append(checked)
            require(fold["support"] == computed[-1]["support"], "Model support differs")
            for task, heads in (("linear_probe", computed[:5]), ("knn", computed[5:])):
                per_fold[task].append({scope: {metric: np.mean([h[scope][metric] for h in heads], axis=0)
                                             for metric in METRICS} for scope in ("macro", "per_class")})
        for task in per_fold:
            for scope in ("macro", "per_class"):
                for metric in METRICS:
                    values = np.asarray([v[scope][metric] for v in per_fold[task]])
                    series = [values] if scope == "macro" else values.T
                    records = [block[task][scope][metric]] if scope == "macro" else block[task][scope][metric]
                    require(len(series) == len(records), "Class aggregate count differs")
                    for vector, record in zip(series, records):
                        close(record["fold_values"], vector, "Fold values differ")
                        close(record["mean"], vector.mean(), "Aggregate mean differs")
                        close(record["sample_std"], vector.std(ddof=1), "Aggregate SD differs")
    crater = read_json(root / "evidence/crater_support.json")
    for row in crater["tiles"]:
        require(row["tile_id"] in labels and row["label"] == labels[row["tile_id"]], "Crater tile label differs")
        require(row["sample_class_counts"] == candidate_counts[int(row["tile_id"].split("_")[1])], "Crater class sample counts differ")
        for f in range(5):
            require(row["fold_roles"][str(f)] == splits[f]["assignments"][row["tile_id"]], "Crater role differs")
    return dict(status="pass", catalog_tiles=len(rows), distinct_test_tiles=len(union), probe_heads=100, knn_vectors=20,
                prediction_rows=sum(6 * len(j["tile_ids"]) for j in jobs.values()))


README = """# Repaired Titan frozen evaluation

This data-only package contains the 23,380-tile repaired Titan catalog and a
verified frozen-encoder evaluation. It contains no image tiles, code, encoder
weights, embeddings, or fine-tuning results. The repository is private at
preparation; no public tile archive endpoint or project DOI is assigned here.

Use `catalogs/titan_catalog.json`, the five `splits/titan_contiguous_fold_*.json`
manifests, and `splits/protocol.json` together. Assignments in the manifests
define membership. Inline legacy splits were removed from the exported catalog.
The catalog's SHA-256 changes because private paths and legacy split fields
were removed; all other selected scientific values and tile order are retained.
Source file hashes in `provenance.json` identify the inputs before sanitization.

`results/frozen_results.json` retains all four model identities, 100 probe-head
confusion matrices, 20 kNN confusion matrices, normalization records, class
support, per-class metrics, and fold summaries. `results/predictions.json.gz`
contains the corresponding test IDs, true labels, five probe prediction vectors
and one kNN vector for every model/fold. Probabilities and feature arrays are
not included; their original validation is recorded, not repeated here.

Reconstruct a 6 by 6 confusion matrix with true classes on rows and predicted
classes on columns. For each class, precision is TP / predicted support,
recall is TP / true support, and F1 is 2 TP / (predicted + true support).
Undefined ratios are zero. Average each metric over all six classes. Average
the five probe heads within each fold, then report the equal-weight mean and
sample standard deviation (ddof=1) of the five fold values. kNN has one vector
per fold and k=20. Cohen's kappa uses the confusion-matrix marginals. Saved
metrics are fractions, not percentages. Fold SD is descriptive variation,
not a confidence interval; overlapping training sets are not independent trials.

`evidence/catalog_comparison.json` records membership and label changes from
v1. `evidence/label_verification.json` retains the original candidate-grid class
vote counts: numeric tile-ID suffixes index that array; the modal class is the
smallest class index attaining the maximum, and its fraction is rounded to four
decimals. These votes include valid map pixels without a SAR-validity mask.
`evidence/crater_support.json` records source polygon parts, sample support,
and split roles. Parts do not establish independent physical craters. Fold 1
has only one test Craters tile. Labels reproduce an expert map, not independent
geological truth. Lakes includes filled and empty basins under that ontology.

The repaired population and geographic partitions both differ from v1; score
changes cannot isolate either change. Earth transfer remains outside this
evaluation and requires expert source-label curation. No new Earth labels are
assigned here. Software export and a usable tile archive remain pending.
Source-product terms are in the repository's DATA_LICENSES.md and
THIRD_PARTY_NOTICES.md; software licensing does not replace those terms.

Run `sha256sum -c SHA256SUMS` from this directory. `file_index.json` covers every
payload except itself and SHA256SUMS; SHA256SUMS also covers file_index.json.
There are no timestamps or checkout paths in generated package metadata.
"""


def build_package(source_repo, catalog_path, predictions_path, output):
    source_repo, catalog_path, predictions_path, output = map(Path, (source_repo, catalog_path, predictions_path, output))
    require(not output.exists(), "Output directory must be new")
    paths = {k: source_repo / v for k, v in SOURCES.items()}
    paths.update(catalog=catalog_path, predictions=predictions_path)
    input_hashes = {k: sha256(p) for k, p in paths.items()}
    for key, expected in (("catalog", CATALOG_SHA256), ("frozen_results", RESULT_SHA256), ("protocol", PROTOCOL_SHA256)):
        require(input_hashes[key] == expected, f"Unverified source identity: {key}")
    source = {k: read_json(p) for k, p in paths.items()}
    raw_catalog, raw_protocol, raw_results = (source[k] for k in ("catalog", "protocol", "frozen_results"))
    require(raw_results["catalog_sha256"] == raw_protocol["catalog_sha256"] == CATALOG_SHA256, "Source catalog bindings differ")
    require(raw_results["protocol_file_sha256"] == PROTOCOL_SHA256 and raw_results["validation"]["status"] == "pass", "Unverified frozen evaluation")
    require(source["predictions"]["verified_audit_sha256"] == RESULT_SHA256, "Prediction audit binding differs")
    require(hashlib.sha256(source["predictions"]["catalog_json"].encode()).hexdigest() == CATALOG_SHA256, "Prediction catalog binding differs")
    require(digest(raw_protocol["specification"]) == raw_protocol["specification_sha256"], "Source protocol specification differs")
    require(source["comparison"]["status"] == "complete" and source["crater_support"]["independent_local_verification"]["status"] == "pass", "Incomplete scientific evidence")
    require(source["labels"]["saved_catalog_label_and_grid_validation"]["status"] == "pass", "Incomplete label verification")
    require(source["comparison"]["inputs"]["rebuilt"]["catalog_sha256"] == CATALOG_SHA256, "Comparison catalog binding differs")
    crater_inputs = source["crater_support"]["inputs"]
    require(crater_inputs["catalog"]["sha256"] == CATALOG_SHA256
            and crater_inputs["protocol"]["sha256"] == PROTOCOL_SHA256
            and crater_inputs["global_oracle"]["sha256"] == input_hashes["labels"], "Crater evidence bindings differ")
    label_inputs = source["labels"]["saved_catalog_label_and_grid_validation"]["inputs"]
    require(label_inputs["catalog"]["sha256"] == CATALOG_SHA256
            and label_inputs["main_catalog_comparison"]["sha256"] == input_hashes["comparison"], "Label evidence bindings differ")
    require(digest(source["labels"]["deployed_global_sampler_validation"]["expected_class_counts"])
            == label_inputs["expected_class_counts"]["sha256"], "Label vote array hash differs")
    expected_jobs = {(j["model"], j["fold"]): j for j in raw_results["jobs"]}
    for job in source["predictions"]["jobs"]:
        require(job["source_identity"] == expected_jobs[job["model"], job["fold"]], "Prediction producer binding differs")
    catalog = pick(raw_catalog, CATALOG_FIELDS)
    catalog.update(split_policy="external_contiguous_manifest", provenance=pick(raw_catalog["provenance"], PROVENANCE_FIELDS),
                   tiles=[pick(t, TILE_FIELDS) for t in raw_catalog["tiles"]])
    output.mkdir(parents=True)
    (output / ".INCOMPLETE").write_text("Export has not passed verification.\n")
    write_json(output, "catalogs/titan_catalog.json", catalog)
    catalog_hash = sha256(output / "catalogs/titan_catalog.json")
    spec = pick(raw_protocol["specification"], SPEC_FIELDS)
    spec["catalog_sha256"] = catalog_hash
    spec_hash = digest(spec)
    protocol = pick(raw_protocol, "schema_version test_union_count test_union_sorted_ids_sha256 specification_digest_format")
    protocol.update(catalog_sha256=catalog_hash, specification=spec, specification_sha256=spec_hash,
                    status="predeclared scientific protocol; frozen results verified separately", splits=[])
    for f, item in enumerate(raw_protocol["splits"]):
        require(item["fold"] == f and item["path"] == f"titan_contiguous_fold_{f}.json", "Source split order/path differs")
        paths[f"split_{f}"] = paths["protocol"].parent / item["path"]
        input_hashes[f"split_{f}"] = sha256(paths[f"split_{f}"])
        require(input_hashes[f"split_{f}"] == item["sha256"], "Source split hash differs")
        split = pick(read_json(paths[f"split_{f}"]), "schema_version policy counts assignments")
        split["catalog_sha256"] = catalog_hash
        split["policy"]["protocol_specification_sha256"] = spec_hash
        name = "splits/" + item["path"]
        write_json(output, name, split)
        protocol["splits"].append(dict(fold=f, path=item["path"], sha256=sha256(output / name)))
    write_json(output, "splits/protocol.json", protocol)
    results = pick(raw_results, RESULT_FIELDS)
    results.update(catalog_sha256=catalog_hash, protocol_file_sha256=sha256(output / "splits/protocol.json"),
                   protocol_specification_sha256=spec_hash, folds=[], models={})
    for fold in raw_results["folds"]:
        row = pick(fold, "fold counts test_ids normalization_provenance")
        row["split_sha256"] = protocol["splits"][row["fold"]]["sha256"]
        results["folds"].append(row)
    for model in MODELS:
        results["models"][model] = pick(raw_results["models"][model], "model_revision weights_sha256 folds linear_probe knn")
    write_json(output, "results/frozen_results.json", results)
    jobs = [pick(j, "model fold tile_ids true_labels seeds probe_predictions knn_predictions") for j in source["predictions"]["jobs"]]
    write_json(output, "results/predictions.json.gz", dict(schema_version="1.0.0", catalog_sha256=catalog_hash,
               frozen_results_sha256=sha256(output / "results/frozen_results.json"), jobs=sorted(jobs, key=lambda j: (j["model"], j["fold"]))))
    write_json(output, "evidence/catalog_comparison.json", pick(source["comparison"], COMPARISON_FIELDS))
    labels = source["labels"]
    write_json(output, "evidence/label_verification.json", dict(
        schema_version="1.0.0", class_crosswalk=labels["class_crosswalk"],
        label_raster=pick(labels["label_raster"], "bounds transform crs_wkt nodata"),
        candidate_grid=labels["deployed_global_sampler_validation"]["grid"],
        candidate_sampling_summary=labels["deployed_global_sampler_validation"]["summary"],
        candidate_class_counts=labels["deployed_global_sampler_validation"]["expected_class_counts"],
        saved_catalog_validation=pick(labels["saved_catalog_label_and_grid_validation"], SAVED_LABEL_FIELDS),
        primary_sources=[pick(r, "id url locator supports") for r in labels["primary_sources"]]))
    crater = pick(source["crater_support"], "schema_version definitions summary source_parts tiles named_center_associations verification")
    crater["verified_role_totals"] = source["crater_support"]["independent_local_verification"]["role_totals"]
    vector_files = crater_inputs["source_vector_files"]
    names = {Path(p).name for p in vector_files}
    expected_names = {"Craters." + suffix for suffix in ("cpg", "dbf", "lyr", "prj", "sbn", "sbx", "shp", "shp.xml", "shx")}
    require(names == expected_names and len(names) == len(vector_files), "Unexpected Craters source files")
    crater["source_vector_sha256"] = {Path(p).name: h for p, h in vector_files.items()}
    write_json(output, "evidence/crater_support.json", crater)
    write_json(output, "provenance.json", dict(schema_version="1.0.0", benchmark_track=TRACK,
        bundle_type="repaired_titan_frozen_scientific_metadata", code_included=False, tile_arrays_included=False,
        finetuning_included=False, source_file_sha256=dict(sorted(input_hashes.items())),
        transformations=dict(catalog="Fixed scientific fields, original order; omit machine paths and inline legacy splits",
            protocol="Remove operational execution approval; bind sanitized catalog, specification and split hashes",
            results="Preserve scientific arrays, counts, model pins and metrics; rebind exported catalog/protocol",
            evidence="Fixed scientific fields only; omit hosts, paths, commands, embedded scripts and private revisions")))
    (output / "README.md").write_text(README)
    (output / ".INCOMPLETE").unlink()
    write_indexes(output, dict(schema_version="1.0.0", benchmark_track=TRACK))
    try:
        verification = verify_package(output)
        require(all(sha256(p) == input_hashes[k] for k, p in paths.items()), "Scientific inputs changed during export")
    except Exception:
        (output / ".INCOMPLETE").write_text("Package verification failed.\n")
        raise
    # Returned mapping belongs in the private dev audit, never in the release.
    return dict(verification=verification,
                inputs={k: {"path": str(p.resolve()), "sha256": input_hashes[k]} for k, p in sorted(paths.items())},
                exported_files=inventory(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_package(args.output_dir), sort_keys=True))
    else:
        if not all((args.source_repo, args.catalog, args.predictions)):
            parser.error("Export requires --source-repo, --catalog, and --predictions")
        result = build_package(args.source_repo, args.catalog, args.predictions, args.output_dir)
        print(json.dumps(result["verification"], sort_keys=True))


if __name__ == "__main__":
    main()
