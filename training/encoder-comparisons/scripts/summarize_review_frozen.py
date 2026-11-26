"""Validate the completed frozen-baseline jobs without importing their runners.

Requires NumPy, Git, the result tree, and the producer repository. Writes only a
new requested report. No model, optimizer, GPU, or remote service is invoked.
"""

import argparse
import ast
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "f8c02e8011a60f3cee3f47b4370edec3d4f77a07"
MODELS = ("dinov2", "dofa", "croma", "random_init")
CLASS_NAMES = ("plains", "dunes", "hummocky", "labyrinths", "lakes", "craters")
SUPPORTED = [0, 1, 3, 4, 5]
METRICS = ("precision", "recall", "f1")
STEMS = {"titan_train": ("titan", "train"), "titan_test": ("titan", "test"),
         "earth_train": ("earth", "train"), "selk_holdout": ("titan", "selk_holdout")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def number(value):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def read_json(path):
    return parse_json(Path(path).read_bytes())


def relative_file(root, name):
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its root: {name}")
    return path


def verify_files(root, expected):
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError(f"File inventory mismatch: {root}")
    for name, digest in expected.items():
        if sha256(relative_file(root, name)) != digest:
            raise ValueError(f"Artifact hash mismatch: {root / name}")


def verify_manifest(root, source_commit):
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete job: {root}")
    git = manifest["git"]
    if git["commit"] != source_commit or git["clean"] is not True or git["status"]:
        raise ValueError(f"Job source identity is not the expected clean commit: {root}")
    verify_files(root, {**manifest["outputs"], "run_manifest.json": sha256(root / "run_manifest.json")})
    return manifest


def option(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Expected one command option {flag}")
    result = []
    for word in command[command.index(flag) + 1:]:
        if word.startswith("--"):
            break
        result.append(word)
    return result


def source_blob(repository, relative):
    return subprocess.check_output(["git", "-C", str(repository), "show", f"{SOURCE_COMMIT}:{relative}"])


def ordered_seeds(records):
    if (len(records) != 5 or any(type(r.get("seed")) is not int for r in records)
            or sorted(r["seed"] for r in records) != list(range(5))):
        raise ValueError("Expected each seed 0 through 4 exactly once")
    return sorted(records, key=lambda r: r["seed"])


def metrics_from_confusion(confusion, classes):
    counts = np.asarray(confusion, dtype=np.float64)
    if (counts.shape != (6, 6) or not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()):
        raise ValueError("Invalid six-output confusion counts")
    if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(range(6)):
        raise ValueError("Invalid scored classes")
    support, predicted = counts.sum(axis=1), counts.sum(axis=0)
    if any(support[k] for k in set(range(6)) - set(classes)):
        raise ValueError("Confusion matrix includes excluded true labels")
    tp = counts.diagonal()
    per_class = {}
    for name, numerator, denominator in zip(METRICS, (tp, tp, 2 * tp), (predicted, support, support + predicted)):
        values = np.divide(numerator, denominator, out=np.zeros(6), where=denominator > 0)
        per_class[name] = [float(values[k]) if k in classes else None for k in range(6)]
    return {"macro": {m: float(np.mean([per_class[m][k] for k in classes])) for m in METRICS},
            "per_class": per_class, "support": support.astype(int).tolist(),
            "prediction_counts": predicted.astype(int).tolist()}


def checked_predictions(record, labels, classes):
    labels = np.asarray(labels)
    predictions, probabilities = np.asarray(record["predictions"]), np.asarray(record["probabilities"])
    if (labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels)
            or not np.isin(labels, classes).all() or predictions.shape != labels.shape
            or predictions.dtype.kind not in "iu" or not np.isin(predictions, range(6)).all()):
        raise ValueError("Prediction labels or row count disagree")
    if (probabilities.shape != (len(labels), 6) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("Invalid six-output probabilities")
    maxima = probabilities.max(axis=1)
    if not np.array_equal(probabilities[np.arange(len(labels)), predictions], maxima):
        raise ValueError("Predictions disagree with probability maxima")
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    if not np.array_equal(confusion, np.asarray(record["confusion_matrix"])):
        raise ValueError("Predictions do not reproduce confusion in the bound cache row order")
    metrics = metrics_from_confusion(confusion, classes)
    if not np.isclose(metrics["macro"]["recall"], record["macro_accuracy"], atol=1e-12, rtol=0):
        raise ValueError("Saved recall disagrees with predictions")
    for k in range(6):
        expected = metrics["per_class"]["recall"][k]
        actual = record["per_class_accuracy"][k]
        if (expected is None and actual is not None) or (expected is not None and
                (actual is None or not np.isclose(expected, actual, atol=1e-12, rtol=0))):
            raise ValueError("Saved class recall disagrees with predictions")
    normalized = confusion / np.maximum(1, confusion.sum(axis=1, keepdims=True))
    if not np.allclose(normalized, record["confusion_matrix_normalized"], atol=1e-12, rtol=0):
        raise ValueError("Normalized confusion disagrees with integer counts")
    if "source_supported_macro_accuracy" in record:
        value = np.mean([metrics["per_class"]["recall"][k] for k in SUPPORTED])
        if not np.isclose(value, record["source_supported_macro_accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Supported-class recall disagrees")
    return {**metrics, "confusion_counts": confusion.tolist(),
            "probability_tie_rows": int(((probabilities == maxima[:, None]).sum(axis=1) > 1).sum())}


def describe_folds(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Expected five finite fold means")
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)), "fold_values": values.tolist()}


def check_cache(directory, model, stem, expected_entries, input_hashes, pinned):
    metadata = read_json(directory / f"{stem}_feats.meta.json")
    if metadata["model"] != model or metadata["array"] != f"{stem}_feats":
        raise ValueError("Cache model or array identity mismatch")
    source = "random_init" if model == "random_init" else "pretrained"
    if metadata["weights_source"] != source:
        raise ValueError("Unexpected cache weight source")
    provenance = metadata["provenance"]
    for key, digest in input_hashes.items():
        if provenance[f"{key}_sha256"] != digest:
            raise ValueError(f"Cache input binding mismatch: {key}")
    for suffix, key in (("feats", "feature_sha256"), ("ids", "tile_ids_sha256"), ("labels", "labels_sha256")):
        if sha256(directory / f"{stem}_{suffix}.npy") != provenance[key]:
            raise ValueError("Cache companion hash mismatch")
    features = np.load(directory / f"{stem}_feats.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(directory / f"{stem}_ids.npy", allow_pickle=False)
    labels = np.load(directory / f"{stem}_labels.npy", allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in "US" or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid or duplicate cache IDs")
    if ids.astype(str).tolist() != [t["tile_id"] for t in expected_entries]:
        raise ValueError("Cache IDs differ from catalog/partition order")
    expected_labels = [t["label"] for t in expected_entries]
    if labels.dtype.kind not in "iu" or labels.shape != (len(ids),) or labels.tolist() != expected_labels:
        raise ValueError("Cache labels disagree with bound catalog rows")
    if (features.shape != (len(ids), 768) or features.shape != (metadata["n_features"], metadata["feature_dim"])
            or features.dtype != np.float32 or not np.isfinite(features).all()):
        raise ValueError("Cache feature shape, dtype, or finiteness mismatch")
    revision, state = provenance["model_revision"], provenance["weights_sha256"]
    if model != "random_init":
        if state != pinned[f"{model.upper()}_SHA256"]:
            raise ValueError("Checkpoint identity disagrees with producer code")
        expected_revision = pinned[f"{model.upper()}_REVISION"] if model != "croma" else "local:titansar.models.croma_model"
        if revision != expected_revision:
            raise ValueError("Model revision disagrees with producer code")
    elif not revision.endswith(":vit_base_patch14_dinov2.lvd142m:seed=42") or len(state) != 64:
        raise ValueError("Random encoder architecture, seed, or state identity is missing")
    if model == "dofa" and provenance.get("band_identifier_ghz") != (5.405 if stem.startswith("earth") else 13.78):
        raise ValueError("DOFA domain identifier is missing or incorrect")
    return ids.astype(str).tolist(), labels, (revision, state)


def summarize(results_root, source_repo, auditor_sha256):
    index = parse_json(source_blob(source_repo, "release/v1/artifact_index.json"))
    archive_raw = source_blob(source_repo, "audit/REVIEW_METRICS_2026-09-29.json")
    archive = parse_json(archive_raw)
    source_paths = ("scripts/run_probing.py", "titansar/models/probing.py", "titansar/models/foundation_models.py",
                    "titansar/models/croma_model.py", "titansar/data/dataset.py", "titansar/configs/defaults.py",
                    "titansar/reproducibility.py")
    source_hashes = {p: hashlib.sha256(source_blob(source_repo, p)).hexdigest() for p in source_paths}
    tree = ast.parse(source_blob(source_repo, "titansar/models/foundation_models.py"))
    pinned = {node.targets[0].id: node.value.value for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, ast.Constant)}
    report = {"schema_version": "1.0.0", "source_commit": SOURCE_COMMIT, "auditor_sha256": auditor_sha256,
              "source_file_sha256": source_hashes, "results_root": str(results_root),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__},
              "population_scope": {"interpretation": "v1_conditional_diagnostic",
                                   "known_defect": "longitude-wrap label nodata omits valid western-hemisphere footprints",
                                   "evidence": "audit/LABEL_PROVENANCE_2026-09-29.md",
                                   "earth_source_fully_curated": False,
                                   "earth_source_evidence": "audit/EARTH_SITE_REVIEW_2026-09-29.md",
                                   "earth_source_caveat": "marine Arctic nominal footprints and Earth validation/training footprint overlap remain unresolved",
                                   "repaired_global_population_evaluated": False},
              "archive_comparison_source": {"path": "audit/REVIEW_METRICS_2026-09-29.json",
                                            "sha256": hashlib.sha256(archive_raw).hexdigest()},
              "historical_dofa_provenance": {"evidence": "audit/GPU_ASSET_AUDIT_2026-09-29.md",
                                             "scope": "original five-fold probe caches",
                                             "earth_identifier_ghz": 13.78, "titan_identifier_ghz": 13.78,
                                             "basis": "independent server audit of cache hashes, extraction logs, and source"},
              "protocol": {"classifier_outputs": 6, "class_names": list(CLASS_NAMES),
                           "titan_to_titan_scored_classes": list(range(6)), "earth_to_titan_scored_classes": SUPPORTED,
                           "aggregation": "mean five seeds within fold, then equal-weight mean and sample SD of five folds",
                           "uncertainty": "descriptive spread only; folds and seeds are not independent geographical replicates",
                           "probability_sum_atol": 1e-6, "metric_atol": 1e-12, "zero_division": 0,
                           "prediction_identity": "rows inherit verified cache ID order; predictions do not embed their own IDs",
                           "matched_comparison": False, "wadi_rum_excluded": False},
              "jobs": [], "models": {}, "validation": {"complete_manifests": 0, "output_hashes": 0,
                                                        "cache_arrays": 0, "prediction_vectors": 0, "prediction_rows": 0}}
    common_test_ids = {}
    for model in MODELS:
        folds_by_scenario = {name: [] for name in ("titan_to_titan", "earth_to_titan")}
        identities, all_test_ids = set(), []
        for fold in range(5):
            analysis_root = results_root / "frozen-analysis" / model / f"fold-{fold}"
            feature_root = results_root / "frozen-features" / model / f"fold-{fold}"
            analysis, feature = [verify_manifest(p, SOURCE_COMMIT) for p in (analysis_root, feature_root)]
            report["validation"]["complete_manifests"] += 2
            report["validation"]["output_hashes"] += len(analysis["outputs"]) + len(feature["outputs"])
            for manifest, root in ((analysis, analysis_root), (feature, feature_root)):
                command = manifest["command"]
                if (option(command, "--models") != [model] or option(command, "--seeds") != [str(k) for k in range(5)]
                        or option(command, "--output_dir") != [str(root)] or "--require_real_weights" not in command):
                    raise ValueError("Producer command identity mismatch")
            if "--features_only" not in feature["command"] or option(analysis["command"], "--device") != ["cpu"]:
                raise ValueError("Unexpected feature/analysis execution modes")
            cache = analysis["inputs"]["cached_features"]
            if Path(cache["path"]).resolve() != (feature_root / "features").resolve():
                raise ValueError("Analysis consumes a different feature job")
            if option(analysis["command"], "--cached_features_dir") != [cache["path"]]:
                raise ValueError("Analysis command/cache manifest disagreement")
            verify_files(Path(cache["path"]), cache["files"])
            if {f"features/{k}": v for k, v in cache["files"].items()} != {
                    k: v for k, v in feature["outputs"].items() if k.startswith("features/")}:
                raise ValueError("Producer/consumer cache hash bindings differ")
            catalogs, splits, input_hashes = {}, {}, {}
            for domain in ("earth", "titan"):
                for role in ("catalog", "split_manifest"):
                    key = f"{domain}_{role}"
                    item = feature["inputs"][key]
                    if analysis["inputs"][key] != item or sha256(item["path"]) != item["sha256"]:
                        raise ValueError("Producer/consumer input identity disagreement")
                    if any(option(m["command"], f"--{key}") != [item["path"]] for m in (analysis, feature)):
                        raise ValueError("Command and input manifest paths disagree")
                    input_hashes[key] = item["sha256"]
                    value = read_json(item["path"])
                    if role == "catalog":
                        if item["sha256"] != index["external_pending_archive"][f"catalogs/{domain}_catalog.json"]["sha256"]:
                            raise ValueError("Catalog differs from the released input")
                        catalogs[domain] = value
                    else:
                        name = "earth_transfer_grouped_validation.json" if domain == "earth" else f"titan_spatial_fold_{fold}.json"
                        if item["sha256"] != index["included"][f"release/v1/splits/{name}"]:
                            raise ValueError("Split differs from the released fold")
                        splits[domain] = value
                entries = catalogs[domain]["tiles"]
                ids = [t["tile_id"] for t in entries]
                if len(ids) != len(set(ids)) or set(splits[domain]["assignments"]) != set(ids):
                    raise ValueError("Catalog/split identity coverage mismatch")
                if splits[domain]["catalog_sha256"] != input_hashes[f"{domain}_catalog"]:
                    raise ValueError("Split/catalog binding mismatch")
                if any(type(t["label"]) is not int or t["label"] not in range(6) for t in entries):
                    raise ValueError("Invalid catalog labels")
            if catalogs["titan"]["class_names"] != list(CLASS_NAMES) or splits["titan"]["policy"]["test_fold"] != fold:
                raise ValueError("Class or fold identity mismatch")
            arrays = {}
            for stem, (domain, split) in STEMS.items():
                rows = [t for t in catalogs[domain]["tiles"] if splits[domain]["assignments"][t["tile_id"]] == split]
                ids, labels, identity = check_cache(feature_root / "features" / model, model, stem, rows, input_hashes, pinned)
                identities.add(identity)
                arrays[stem] = (ids, labels)
                report["validation"]["cache_arrays"] += 1
            ids, target_labels = arrays["titan_test"]
            if fold in common_test_ids and ids != common_test_ids[fold]:
                raise ValueError("Models evaluate different test ID sequences")
            common_test_ids[fold] = ids
            all_test_ids.extend(ids)
            if np.unique(arrays["earth_train"][1]).tolist() != SUPPORTED:
                raise ValueError("Unexpected source-supported classes")
            payload = read_json(analysis_root / "probing_results.json")
            if set(payload) != {model} or payload[model]["model"] != model:
                raise ValueError("Result model identity mismatch")
            payload = payload[model]
            if payload["weights_source"] != ("random_init" if model == "random_init" else "pretrained"):
                raise ValueError("Result weight source mismatch")
            for scenario, scored in (("titan_to_titan", list(range(6))), ("earth_to_titan", SUPPORTED)):
                block = payload["linear_probe"][scenario]
                labels = target_labels if scenario == "titan_to_titan" else target_labels[np.isin(target_labels, SUPPORTED)]
                seeds = ordered_seeds(block["seed_results"])
                checked = [checked_predictions(seed, labels, scored) for seed in seeds]
                recalls = [c["macro"]["recall"] for c in checked]
                if (not np.isclose(np.mean(recalls), block["macro_accuracy_mean"], atol=1e-12, rtol=0)
                        or not np.isclose(np.std(recalls, ddof=1), block["macro_accuracy_std"], atol=1e-12, rtol=0)):
                    raise ValueError("Saved seed aggregate disagrees with reconstructed recall")
                if block["source_supported_evaluated_class_indices"] != SUPPORTED:
                    raise ValueError("Reported evaluated class set differs")
                fold_result = {"fold": fold, "support": checked[0]["support"],
                               "macro": {m: float(np.mean([c["macro"][m] for c in checked])) for m in METRICS},
                               "per_class": {m: [float(np.mean([c["per_class"][m][k] for c in checked]))
                                                 if k in scored else None for k in range(6)] for m in METRICS},
                               "seeds": [{"seed": s["seed"], "macro": c["macro"],
                                          "confusion_counts": c["confusion_counts"], "probability_tie_rows": c["probability_tie_rows"]}
                                         for s, c in zip(seeds, checked)]}
                folds_by_scenario[scenario].append(fold_result)
                report["validation"]["prediction_vectors"] += 5
                report["validation"]["prediction_rows"] += 5 * len(labels)
            report["jobs"].append({"model": model, "fold": fold,
                                   "analysis_manifest_sha256": sha256(analysis_root / "run_manifest.json"),
                                   "feature_manifest_sha256": sha256(feature_root / "run_manifest.json"),
                                   "result_sha256": analysis["outputs"]["probing_results.json"],
                                   "input_sha256": input_hashes,
                                   "train_counts": {d: np.bincount(arrays[f"{d}_train"][1], minlength=6).tolist() for d in ("earth", "titan")}})
        if len(identities) != 1 or len(all_test_ids) != len(set(all_test_ids)):
            raise ValueError("Model state changes across caches, or test IDs repeat across folds")
        model_result = {"model_revision": next(iter(identities))[0], "weights_sha256": next(iter(identities))[1]}
        for scenario, folds in folds_by_scenario.items():
            macro = {m: describe_folds([f["macro"][m] for f in folds]) for m in METRICS}
            model_result[scenario] = {"folds": folds, "macro": macro,
                                      "archive_macro": archive["models"][model][scenario]["macro"],
                                      "delta_from_archive_percentage_points": {
                                          m: 100 * (macro[m]["mean"] - archive["models"][model][scenario]["macro"][m]["mean"])
                                          for m in METRICS}}
        report["models"][model] = model_result
    report["validation"]["distinct_test_tiles"] = sum(map(len, common_test_ids.values()))
    report["validation"]["status"] = "pass"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report file; refuses overwrite")
    parser.add_argument("--auditor-sha256", help="Source digest when executing this script through stdin")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.results_root.resolve()):
        parser.error("Output must be new and outside the immutable results tree")
    own_path = Path(__file__)
    code_hash = sha256(own_path) if own_path.is_file() else args.auditor_sha256
    if not code_hash or len(code_hash) != 64 or any(c not in "0123456789abcdef" for c in code_hash):
        parser.error("A valid auditor SHA256 is required for stdin execution")
    report = summarize(args.results_root.resolve(), args.source_repo.resolve(), code_hash)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
