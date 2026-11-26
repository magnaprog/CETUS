"""Authenticate and summarize the complete, fixed model-followup grid on CPU.

Run on the original host, where every recorded absolute path still exists::

    python -m scripts.analyze_model_followup --plan PLAN.json \
        --plan-sha256 SHA256 --output REPORT.json

PLAN (model-followup-analysis-v1) contains purpose, catalog (path/sha256),
splits (five bindings in fold order), provenance (dino_features, dino_c1,
resnet18, each with exact source/runtime objects), accepted_results (90
result.json bindings), and failed_attempts (explicit failed.json bindings).
The caller seals that plan independently of the bundles. No glob discovers
accepted results, no partial grid is summarized, and publication never replaces
an existing report. Synthetic fixtures have the same grid and a smaller budget.

This is saved-prediction arithmetic and provenance verification. State files
are hashed, never deserialized. It does not replay models, probabilities,
normalization, optimization, or convergence. Only stdlib and NumPy are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA = "model-followup-analysis-v1"
ROLES = ("train", "val", "test")
CLASSES = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
LRS = (0.0001, 0.001, 0.01)
CATALOG_SHA = "2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec"
SPLIT_SHAS = (
    "1079b1c63588ee9af674184b7dd6937ef01d3a354c8773e8ee9caad4ab8cc388",
    "f9bb93592f5bb9e111118f796c4cdc4e2478cb1ef18f1d61d7044525d0a820d2",
    "7862dd95df2eed01d0252a7a37068ee780509d98dd9e0105e1930ad0ffa75610",
    "8317384eb1945fc2af628e0051b8181637525d0412c2f2c1b38b7b2967fee9b6",
    "25e66ac42fd246d824fb957547e30d2e9f453dfd039130a4adf0b5cd0d796d09",
)
RECIPE = {
    "optimizer": "AdamW", "epochs": 100, "batch_size": 32,
    "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
    "scheduler": "CosineAnnealingLR", "dtype": "float32", "amp": False,
    "selection": "earliest maximum unrounded validation macro_f1 with optimizer updates",
}
TRANSFORM = {
    "name": "cetus_full_tile_rgb224_v1", "input_shape": [1, 128, 128],
    "dtype": "float32", "resize": [224, 224], "mode": "bilinear",
    "align_corners": False, "antialias": False,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_SOURCE_SHA = "929c0b25f4287511f065c8696667b39d2b9499c68f284bcab30b3e27099a488f"
DINO_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
RESNET_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
DINO_CONSTRUCTOR = {
    "img_size": 518, "patch_size": 14, "init_values": 1.0, "ffn_layer": "mlp",
    "block_chunks": 0, "num_register_tokens": 0,
    "interpolate_antialias": False, "interpolate_offset": 0.1,
}
METRICS = ("macro_f1", "balanced_accuracy", "kappa", "precision_per_class",
           "recall_per_class", "f1_per_class")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Invalid SHA256")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite JSON number")
    return result


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream, object_pairs_hook=_object, parse_float=_number,
                         parse_constant=_number)


class Files:
    """Hash each immutable file once, retaining stat guards for repeat reads."""

    def __init__(self):
        self.seen = {}

    def hash(self, path):
        path = Path(path)
        require(path.is_absolute() and not path.is_symlink() and path.is_file(),
                f"Expected original absolute regular file: {path}")
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = str(path.resolve(strict=True))
        if key in self.seen:
            before, sha = self.seen[key]
            require(before == stamp, f"File changed during analysis: {path}")
            return sha
        hashed = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hashed.update(block)
        after = path.stat()
        require(stamp == (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns, after.st_ctime_ns), f"File changed while hashing: {path}")
        sha = hashed.hexdigest()
        self.seen[key] = (stamp, sha)
        return sha

    def bound(self, binding):
        require(set(binding) == {"path", "sha256"}, "Expected explicit path/SHA binding")
        path = Path(binding["path"])
        require(self.hash(path) == _sha(binding["sha256"]), f"File SHA differs: {path}")
        return path

    def json(self, binding):
        path = self.bound(binding)
        result = read_json(path)
        self.hash(path)
        return result

    def unchanged(self):
        for path in list(self.seen):
            self.hash(Path(path))


def _inventory(root):
    """Walk only the explicitly bound bundle, to reject omitted/extra artifacts."""
    names = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            require(not (Path(directory) / name).is_symlink(), "Bundle symlink is forbidden")
        names.update(str((Path(directory) / name).relative_to(root)) for name in files)
    return names


def _close(actual, expected, message, atol=1e-15):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "iuf" and b.dtype.kind in "iuf"
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=0), message)


def prediction_metrics(saved, truth):
    """Reconstruct integer confusion and six-class metrics independently."""
    require(saved["tile_ids"] == truth["tile_ids"] and saved["labels"] == truth["labels"],
            "Prediction IDs/labels differ from catalog and split")
    labels, predicted = np.asarray(saved["labels"]), np.asarray(saved["predictions"])
    require(labels.ndim == predicted.ndim == 1 and labels.shape == predicted.shape
            and labels.dtype.kind in "iu" and predicted.dtype.kind in "iu"
            and ((predicted >= 0) & (predicted < 6)).all(), "Invalid predictions")
    require(saved["class_order"] == list(range(6)), "Prediction class order differs")
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    support, called, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    require((support > 0).all(), "Every truth class is required")
    precision = np.divide(tp, called, out=np.zeros(6), where=called != 0)
    recall = tp / support
    f1 = 2 * tp / (support + called)
    n = int(support.sum())
    observed, expected = float(tp.sum()) / n, float(support @ called) / n**2
    metrics = {
        "confusion_matrix": matrix.tolist(), "support_per_class": support.tolist(),
        "true_positive": tp.tolist(), "false_positive": (called - tp).tolist(),
        "false_negative": (support - tp).tolist(), "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(), "f1_per_class": f1.tolist(),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
        "kappa": (observed - expected) / (1 - expected),
    }
    for key, value in metrics.items():
        if key in METRICS:
            _close(saved[key], value, f"Saved {key} differs from predictions")
        else:
            a = np.asarray(saved[key])
            require(a.dtype.kind in "iu" and np.array_equal(a, value), f"Saved {key} differs")
    logits, probabilities = np.asarray(saved["logits"]), np.asarray(saved["probabilities"])
    require(saved["logits_dtype"] in ("float32", "float64")
            and saved["probabilities_dtype"] in ("float32", "float64"), "Invalid output dtype")
    require(logits.shape == probabilities.shape == (n, 6)
            and np.isfinite(logits).all() and np.isfinite(probabilities).all()
            and np.array_equal(predicted, logits.argmax(1)), "Invalid saved logit decisions")
    tolerance = 1e-6 if saved["probabilities_dtype"] == "float32" else 1e-12
    require(((probabilities >= 0) & (probabilities <= 1)).all()
            and np.allclose(probabilities.sum(1), 1, atol=tolerance, rtol=0), "Invalid probabilities")
    require(saved["probability_argmax_disagreements"] == int(np.sum(predicted != probabilities.argmax(1))),
            "Probability decision disagreement count differs")
    if "loss" in saved:
        require(saved["loss_denominator"] > 0 and saved["loss_numerator"] >= 0,
                "Invalid evaluation loss denominator/numerator")
        _close(saved["loss"], saved["loss_numerator"] / saved["loss_denominator"],
               "Evaluation loss ratio differs", 1e-12)
    return {**metrics, "n": n, "ordered_ids_sha256": digest(truth["tile_ids"]),
            "labels_sha256": digest(truth["labels"])}


def _expected_grid():
    cells = set()
    for fold in range(5):
        cells.add(("dino_c1", fold, "pretrained", 42))
        cells.update(("dino_c1", fold, "random", seed) for seed in range(42, 47))
        for arm in ("scratch", "pretrained"):
            cells.update(("tuning", fold, arm, lr) for lr in LRS)
            cells.update(("final", fold, arm, seed) for seed in range(3))
    return cells


class Analysis:
    def __init__(self, plan, files):
        self.plan, self.files = plan, files
        require(plan["schema_version"] == SCHEMA, "Unknown analysis plan schema")
        self.purpose = plan["purpose"]
        require(self.purpose in ("titan_followup", "synthetic_cpu_test"), "Unknown purpose")
        require(set(plan["provenance"]) == {"dino_features", "dino_c1", "resnet18"},
                "Specify all three provenance classes")
        for lock in plan["provenance"].values():
            require(set(lock) == {"source", "runtime"} and lock["source"] and lock["runtime"],
                    "Empty provenance lock")
        require(len(plan["splits"]) == 5, "Exactly five split bindings required")
        self.catalog = files.json(plan["catalog"])
        require(self.catalog["intensity_space"] == "hisar_log_dn", "Wrong input intensity space")
        if self.purpose == "titan_followup":
            require(plan["catalog"]["sha256"] == CATALOG_SHA
                    and len(self.catalog["tiles"]) == 23380
                    and self.catalog["class_names"] == CLASSES and self.catalog["num_classes"] == 6
                    and self.catalog["labels_available"] is True, "Wrong current Titan catalog")
        else:
            require(self.catalog["benchmark_track"] == "synthetic_fixture", "Synthetic catalog required")
        entries = self.catalog["tiles"]
        ids = [row["tile_id"] for row in entries]
        require(all(isinstance(i, str) and i and Path(i).name == i for i in ids)
                and len(ids) == len(set(ids)), "Invalid/duplicate catalog IDs")
        require(all(type(row["label"]) is int and row["label"] in range(6) for row in entries),
                "Invalid catalog labels")
        self.truth, self.counts, self.tile_inventories = [], [], []
        for fold, binding in enumerate(plan["splits"]):
            split = files.json(binding)
            require(type(split["policy"]["test_fold"]) is int and split["policy"]["test_fold"] == fold,
                    "Split fold identity differs")
            require(split["catalog_sha256"] == plan["catalog"]["sha256"]
                    and set(split["assignments"]) == set(ids)
                    and set(split["assignments"].values()) <= set(ROLES) | {"buffer_excluded", "selk_holdout"},
                    "Split catalog/role coverage differs")
            if self.purpose == "titan_followup":
                require(binding["sha256"] == SPLIT_SHAS[fold], "Wrong current Titan split")
            truth, counts, tiles = {}, {}, {}
            for role in ROLES:
                rows = [e for e in entries if split["assignments"][e["tile_id"]] == role]
                truth[role] = {"tile_ids": [e["tile_id"] for e in rows], "labels": [e["label"] for e in rows]}
                histogram = np.bincount(truth[role]["labels"], minlength=6)
                require((histogram > 0).all(), f"Missing {role} truth class")
                counts[role] = {"total": len(rows), "class_counts": histogram.tolist()}
                tiles.update({e["tile_id"]: _sha(e.get("tile_sha256", e.get("sha256"))) for e in rows})
            self.truth.append(truth)
            self.counts.append(counts)
            self.tile_inventories.append(digest(tiles))
        self.inputs_by_fold, self.features, self.state_groups = {}, {}, {}

    def spec(self, spec):
        require(spec["schema_version"] == "model-followup-v1" and spec["purpose"] == self.purpose,
                "Specification schema/purpose differs")
        require(type(spec["fold"]) is int and spec["fold"] in range(5), "Invalid job fold")
        kind = spec["kind"]
        require(kind in self.plan["provenance"], "Unknown job kind")
        require({k: spec[k] for k in ("source", "runtime")} == self.plan["provenance"][kind],
                f"Source/runtime mixture within {kind}")
        device = spec["device"]
        cuda = (isinstance(device, str) and device.startswith("cuda:")
                and device[5:].isascii() and device[5:].isdigit())
        require(device == "cpu" or cuda, "Invalid device")
        if kind == "dino_c1" or self.purpose == "synthetic_cpu_test":
            require(spec["device"] == "cpu", "Expected CPU job")

    def bundle(self, binding):
        path = self.files.bound(binding)
        require(path.name == "result.json", "Accepted completion must bind result.json")
        result = self.files.json(binding)
        require(result["status"] == "complete", "Accepted job is not complete")
        spec = result["specification"]
        self.spec(spec)
        require(result["specification_sha256"] == digest(spec), "Specification seal differs")
        require(type(result["fold"]) is int
                and all(result[k] == spec[k] for k in ("kind", "fold", "purpose")), "Result job identity differs")
        root = path.parent
        require(Path(spec["output_dir"]) == root, "Result output path/job binding differs")
        names = _inventory(root)
        require(names == set(result["outputs"]) | {"result.json"}, "Bundle inventory coverage differs")
        require("result.json" not in result["outputs"] and "failed.json" not in names,
                "Complete/failed status conflict")
        require({"started.json", "preparation.json", "initial_state.pt" if spec["kind"] != "dino_c1"
                 else "estimator.npz"} <= names, "Required bundle artifacts missing")
        for name, sha in result["outputs"].items():
            member = root / name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and member.resolve().is_relative_to(root.resolve()), "Inventory path escapes bundle")
            require(self.files.hash(member) == _sha(sha), f"Inventory SHA differs: {name}")
        started, prepared = (read_json(root / name) for name in ("started.json", "preparation.json"))
        require(started["status"] == "started" and started["specification"] == spec
                and started["specification_sha256"] == digest(spec), "Started specification differs")
        require(prepared["status"] == "prepared_no_model_execution" and prepared["ready"] is True
                and prepared["missing_prerequisites"] == []
                and prepared["specification_sha256"] == digest(spec)
                and prepared["inputs"] == result["inputs"]
                and prepared["source"]["commit"] == spec["source"]["commit"]
                and prepared["source_files"] == spec["source"]["files"]
                and prepared["runtime"] == spec["runtime"], "Preparation binding differs")
        if self.purpose == "titan_followup":
            require(prepared["source"]["clean"] is True, "Scientific producer was not clean")
        self.inputs(result["inputs"], spec)
        return root, result, prepared

    def inputs(self, inputs, spec):
        fold = spec["fold"]
        require(inputs["catalog"] == self.plan["catalog"] and inputs["split"] == self.plan["splits"][fold]
                and inputs["catalog_count"] == len(self.catalog["tiles"])
                and inputs["class_order"] == CLASSES
                and inputs["benchmark_track"] == self.catalog["benchmark_track"]
                and inputs["ordered"] == self.truth[fold] and inputs["counts"] == self.counts[fold]
                and inputs["tile_inventory_sha256"] == self.tile_inventories[fold],
                "Input role/ID/label/count/catalog binding differs")
        norm = inputs["normalization"]
        require(digest(norm) == inputs["normalization_sha256"] and norm["enabled"] is True
                and norm["fallback_used"] is False and norm["sampling"]["sample_count"] > 0
                and not norm["sampling"]["missing_tile_ids"] and not norm["sampling"]["empty_tile_ids"]
                and norm["training_ids_sha256"] == digest(self.truth[fold]["train"]["tile_ids"])
                and norm["intensity_space"] == "hisar_log_dn", "Normalization binding differs")
        inverse = 1 / np.asarray(self.counts[fold]["train"]["class_counts"], dtype=float)
        weights = (inverse / inverse.sum() * 6).astype(np.float32).tolist()
        require(inputs["training_class_weights"] == weights, "Training class weights differ")
        if spec["kind"] != "dino_c1":
            require(all(spec["inputs"][k] == inputs[k] for k in
                        ("catalog", "split", "benchmark_track", "counts", "normalization_sha256")),
                    "Specification input binding differs")
        signature = digest(inputs)
        require(fold not in self.inputs_by_fold or self.inputs_by_fold[fold] == signature,
                "Input/normalization mixture within fold")
        self.inputs_by_fold[fold] = signature

    def same_state(self, group, identity):
        signature = digest(identity)
        require(group not in self.state_groups or self.state_groups[group] == signature,
                f"Model/state varies across paired cells: {group}")
        self.state_groups[group] = signature

    def feature(self, binding):
        key = (binding["path"], binding["sha256"])
        if key in self.features:
            return self.features[key]
        root, result, _ = self.bundle(binding)
        spec, identity = result["specification"], result["identity"]
        require(spec["kind"] == "dino_features", "Wrong feature producer kind")
        arm, seed = identity["initialization"], identity["encoder_seed"]
        require(arm in ("random", "pretrained") and type(seed) is int
                and seed in (range(42, 47) if arm == "random" else (42,)), "Unexpected DINO cell")
        require(identity["model"] == f"dinov2_{arm}_matched" and identity["transform"] == TRANSFORM
                and identity["dtype"] == "float32" and identity["backend"] == "torch_sdpa_no_xformers"
                and identity["revision"] == DINO_REVISION and identity["constructor"] == DINO_CONSTRUCTOR,
                "DINO implementation/transform identity differs")
        model = spec["model"]
        require(all(identity[k] == model[k] for k in
                    ("initialization", "encoder_seed", "checkpoint", "initial_state_sha256"))
                and model["source_revision"] == DINO_REVISION
                and digest(model["source_files"]) == identity["source_python_inventory_sha256"],
                "DINO model specification differs")
        if self.purpose == "titan_followup":
            require(identity["source_python_inventory_sha256"] == DINO_SOURCE_SHA,
                    "Wrong official DINO source inventory")
        _sha(identity["initial_state_sha256"])
        self.files.bound(model["checkpoint"])
        if arm == "pretrained" and self.purpose == "titan_followup":
            require(model["checkpoint"]["sha256"] == DINO_SHA, "Wrong pretrained DINO release")
        require(set(spec["recipe"]) == {"batch_size"} and type(spec["recipe"]["batch_size"]) is int
                and spec["recipe"]["batch_size"] > 0, "Invalid feature recipe")
        self.same_state(("dino", arm, seed), identity)
        require(set(result["caches"]) == set(ROLES), "Feature role coverage differs")
        for role in ROLES:
            expected = {"identity": identity, "inputs": result["inputs"], "fold": spec["fold"],
                        "role": role, "transform": TRANSFORM, "purpose": self.purpose,
                        **self.truth[spec["fold"]][role]}
            require(result["caches"][role] == expected, "Feature cache role binding differs")
            stem = f"features/titan_{role}"
            meta = read_json(root / f"{stem}_feats.meta.json")
            provenance = meta["provenance"]
            require(meta["schema_version"] == "model-followup-v1" and meta["model"] == identity["model"]
                    and meta["weights_source"] == arm and meta["array"] == f"titan_{role}_feats"
                    and meta["n_features"] == len(expected["tile_ids"]) and meta["feature_dim"] == 768
                    and all(provenance[k] == v for k, v in expected.items()), "Feature metadata differs")
            for suffix, field in (("feats", "feature_sha256"), ("ids", "tile_ids_sha256"),
                                  ("labels", "labels_sha256")):
                path = root / f"{stem}_{suffix}.npy"
                require(provenance[field] == self.files.hash(path), "Feature companion binding differs")
                # Features/states are integrity checked only. Read small role companions.
                if suffix != "feats":
                    array = np.load(path, allow_pickle=False)
                    require(array.ndim == 1 and array.dtype.kind in ("U" if suffix == "ids" else "iu")
                            and array.tolist() == expected["tile_ids" if suffix == "ids" else "labels"],
                            "Feature companion IDs/labels differ")
        summary = {"binding": binding, "identity": identity, "fold": spec["fold"],
                   "inputs_sha256": digest(result["inputs"])}
        self.features[key] = summary
        return summary

    def dino(self, binding, root, result):
        spec = result["specification"]
        feature = self.feature(spec["features"])
        require(feature["fold"] == spec["fold"] and feature["identity"] == result["identity"]
                and feature["identity"] == spec["feature_identity"]
                and feature["inputs_sha256"] == digest(result["inputs"]), "Readout/feature job identity differs")
        parameters = result["classifier_parameters"]
        require(all(parameters[k] == v for k, v in
                    {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4,
                     "class_weight": "balanced", "random_state": 0, "fit_intercept": True}.items())
                and parameters.get("penalty") in ("l2", "deprecated")
                and parameters.get("l1_ratio") in (None, 0, 0.0), "C=1 classifier recipe differs")
        require(result["scaler_parameters"] == {"copy": True, "with_mean": True, "with_std": True}
                and result["convergence_claim"] is False, "Scaler/convergence contract differs")
        require(set(result["predictions"]) == {"val", "test"}, "Readout role coverage differs")
        # The NPZ is inventoried, but estimator arrays are left to the separate replay review.
        require("estimator.npz" in result["outputs"] and result["feature_dtype"] == "float32"
                and result["coefficient_dtype"] in ("float32", "float64")
                and len(result["n_iter"]) == 1 and type(result["n_iter"][0]) is int
                and 0 < result["n_iter"][0] <= 2000 and isinstance(result["warnings"], list),
                "Readout fitted-state metadata differs")
        identity = result["identity"]
        cell = ("dino_c1", spec["fold"], identity["initialization"], identity["encoder_seed"])
        return cell, {"result": binding, "feature_result": spec["features"], "cell": list(cell),
                      "identity": identity, "n_iter": result["n_iter"], "warnings": result["warnings"],
                      "validation": prediction_metrics(result["predictions"]["val"], self.truth[spec["fold"]]["val"]),
                      "test": prediction_metrics(result["predictions"]["test"], self.truth[spec["fold"]]["test"])}

    def cnn(self, binding, root, result, prepared):
        spec, identity = result["specification"], result["identity"]
        fold, arm, phase, seed = spec["fold"], spec["model"]["initialization"], spec["phase"], spec["seed"]
        require(arm in ("scratch", "pretrained") and phase in ("tuning", "final") and type(seed) is int
                and (seed == 100 if phase == "tuning" else seed in range(3)), "Unexpected CNN cell")
        require(identity["model"] == "resnet18" and identity["transform"] == TRANSFORM
                and identity["num_classes"] == 6 and identity["dtype"] == "float32", "CNN identity differs")
        require(all(identity[k] == spec["model"].get(k) for k in
                    ("initialization", "backbone_seed", "head_seed", "torchvision_version",
                     "resnet_source_sha256", "checkpoint", "weights"))
                and identity["backbone_seed"] == seed and identity["head_seed"] == seed + 20000
                and spec["loader_seed"] == seed + 10000, "CNN model/seed identity differs")
        if self.purpose == "titan_followup":
            require(identity["torchvision_version"] == spec["runtime"]["versions"]["torchvision"]
                    and identity["resnet_source_sha256"]
                    == spec["runtime"]["modules"]["torchvision.models.resnet"]["sha256"],
                    "CNN implementation differs from runtime lock")
        shared_model = {k: v for k, v in spec["model"].items() if k not in
                        ("initialization", "backbone_seed", "head_seed", "weights", "checkpoint")}
        self.same_state(("cnn_common_model",), shared_model)
        for key in ("initial_state_sha256", "head_state_sha256"):
            _sha(identity[key])
        for key in ("selected_state_sha256", "final_state_sha256"):
            _sha(result[key])
        if arm == "pretrained":
            require(identity["weights"] == "ResNet18_Weights.IMAGENET1K_V1", "Wrong CNN weights")
            self.files.bound(identity["checkpoint"])
            if self.purpose == "titan_followup":
                require(identity["checkpoint"]["sha256"] == RESNET_SHA, "Wrong pretrained CNN release")
        else:
            require(identity["checkpoint"] is None and identity["weights"] is None, "Scratch CNN has weights")
        self.same_state(("cnn", arm, seed), identity)
        self.same_state(("cnn_head", seed), identity["head_state_sha256"])
        recipe = spec["recipe"]
        expected = dict(RECIPE)
        if self.purpose == "synthetic_cpu_test":
            require(type(recipe["epochs"]) is int and recipe["epochs"] in range(1, 4)
                    and type(recipe["batch_size"]) is int and recipe["batch_size"] in range(1, 7),
                    "Invalid synthetic budget")
            expected.update(epochs=recipe["epochs"], batch_size=recipe["batch_size"])
        require({k: v for k, v in recipe.items() if k != "lr"} == expected and recipe["lr"] in LRS,
                "CNN recipe differs")
        require({"initial_state.pt", "selected.pt", "final.pt"} <= set(result["outputs"]), "Missing CNN state")
        require(result["training_class_weights"] == result["inputs"]["training_class_weights"],
                "CNN training class weights differ")
        history = result["history"]
        require(len(history) == recipe["epochs"], "Incomplete epoch budget")
        require({n for n in result["outputs"] if n.startswith("epoch-")}
                == {f"epoch-{e:03d}.json" for e in range(1, recipe["epochs"] + 1)}, "Epoch file coverage differs")
        chosen, best, selected_digest, validation_history = None, -1.0, None, []
        for epoch, row in enumerate(history, 1):
            require(row["epoch"] == epoch and type(row["epoch"]) is int, "Epoch order differs")
            saved = read_json(root / f"epoch-{epoch:03d}.json")
            require(saved["history"] == row, "Epoch/history identity differs")
            metrics = prediction_metrics(saved["validation"], self.truth[fold]["val"])
            _close(row["val_macro_f1"], metrics["macro_f1"], "Epoch validation score differs")
            require(row["val_confusion_matrix"] == metrics["confusion_matrix"], "Epoch confusion differs")
            for key in ("loss", "loss_numerator", "loss_denominator"):
                require(row[f"val_{key}"] == saved["validation"][key], "Epoch validation loss differs")
            require(type(row["optimizer_steps"]) is int
                    and row["optimizer_steps"] == math.ceil(self.counts[fold]["train"]["total"] / recipe["batch_size"])
                    and row["examples"] == self.counts[fold]["train"]["total"]
                    and row["finite_gradients"] is True, "Training update/example budget differs")
            for prefix, numerator, denominator in (("train", "loss_numerator", "loss_denominator"),
                                                   ("val", "val_loss_numerator", "val_loss_denominator")):
                require(row[denominator] > 0 and row[numerator] >= 0, "Invalid loss denominator/numerator")
                _close(row[f"{prefix}_loss"], row[numerator] / row[denominator], "Loss ratio differs", 1e-12)
            rate = recipe["lr"] * (1 + math.cos(math.pi * (epoch - 1) / recipe["epochs"])) / 2
            _close(row["lr"], rate, "Cosine learning rate differs", 1e-14)
            # Compare the checked original unrounded score, preserving producer tie rules.
            if row["val_macro_f1"] > best:
                chosen, best, selected_digest = epoch, row["val_macro_f1"], digest(saved["validation"])
            validation_history.append({"history": row, "metrics": metrics})
        require(type(result["selected_epoch"]) is int and result["selected_epoch"] == chosen
                and digest(result["selected_validation"]) == selected_digest, "Selected epoch/validation differs")
        require(prepared["selection"] == result["selection"], "Prepared selection differs")
        if phase == "tuning":
            require("test" not in result and result["selection"] is None, "Tuning must not score test data")
        cell = (phase, fold, arm, recipe["lr"] if phase == "tuning" else seed)
        summary = {"result": binding, "cell": list(cell), "identity": identity, "recipe": recipe,
                   "model": spec["model"], "inputs_sha256": digest(result["inputs"]),
                   "selected_epoch": chosen, "selected_score": best,
                   "selected_state_sha256": result["selected_state_sha256"],
                   "final_state_sha256": result["final_state_sha256"],
                   "validation": prediction_metrics(result["selected_validation"], self.truth[fold]["val"]),
                   "validation_history": validation_history, "selection": result["selection"],
                   "tuning_results": spec.get("tuning_results", [])}
        if phase == "final":
            summary["test"] = prediction_metrics(result["test"], self.truth[fold]["test"])
        return cell, summary

    def failed(self, bindings, accepted_paths):
        failures, seen = [], set()
        for binding in bindings:
            path = self.files.bound(binding)
            require(path.name == "failed.json" and str(path) not in seen, "Invalid/duplicate failed attempt")
            seen.add(str(path))
            require(not (path.parent / "result.json").exists()
                    and str(path.parent / "result.json") not in accepted_paths, "Failed/complete status conflict")
            failed = self.files.json(binding)
            started_path = path.parent / "started.json"
            started_binding = {"path": str(started_path), "sha256": self.files.hash(started_path)}
            started = self.files.json(started_binding)
            spec = started["specification"]
            self.spec(spec)
            require(started["status"] == "started" and failed["status"] == "failed"
                    and started["specification_sha256"] == failed["specification_sha256"] == digest(spec)
                    and Path(spec["output_dir"]) == path.parent, "Failed attempt specification binding differs")
            failures.append({"failed": binding, "started": started_binding, "specification": spec,
                             "receipt": failed})
        return failures


def _summary(values):
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), "Nonfinite summary")
    return {"mean": a.mean(0).tolist(), "sample_sd": a.std(0, ddof=1).tolist() if len(a) > 1 else None,
            "n": len(a)}


def aggregate(cells):
    """Seeds first; five equally weighted regions. Never pool confusions."""
    arms = {}
    for kind, arm, seeds in (("dino_c1", "random", range(42, 47)), ("dino_c1", "pretrained", (42,)),
                             ("final", "scratch", range(3)), ("final", "pretrained", range(3))):
        regions = []
        for fold in range(5):
            scores = [cells[(kind, fold, arm, seed)]["test"] for seed in seeds]
            regions.append({"fold": fold, "seed_ids": list(seeds),
                            "class_counts": scores[0]["support_per_class"],
                            "seeds": [cells[(kind, fold, arm, seed)]["cell"] for seed in seeds],
                            "metrics": {k: _summary([s[k] for s in scores]) for k in METRICS}})
        arms[f"{kind}:{arm}"] = {"regions": regions,
            "equal_region": {k: _summary([r["metrics"][k]["mean"] for r in regions]) for k in METRICS}}
    contrasts = []
    for kind, name, seeds in (("dino_c1", "pretrained_dino_minus_mean_five_random", range(42, 47)),
                               ("final", "imagenet_cnn_minus_random_paired_three_seeds", range(3))):
        regions = []
        for fold in range(5):
            random_arm = "random" if kind == "dino_c1" else "scratch"
            differences = [cells[(kind, fold, "pretrained", 42 if kind == "dino_c1" else seed)]["test"]["macro_f1"]
                           - cells[(kind, fold, random_arm, seed)]["test"]["macro_f1"] for seed in seeds]
            within = _summary(differences)
            if kind == "dino_c1":
                # One pretrained observation, compared with the mean of five random states.
                within["mean"] = cells[(kind, fold, "pretrained", 42)]["test"]["macro_f1"] - float(np.mean([
                    cells[(kind, fold, "random", seed)]["test"]["macro_f1"] for seed in seeds]))
            regions.append({"fold": fold, "seed_ids": list(seeds), "seed_differences": differences,
                            "within_region": within})
        contrasts.append({"name": name, "regions": regions,
                          "single_pretrained_state": kind == "dino_c1",
                          "equal_region": _summary([r["within_region"]["mean"] for r in regions])})
    return {"arms": arms, "contrasts": contrasts, "comparison_questions": len(contrasts),
            "regional_differences": sum(len(c["regions"]) for c in contrasts)}


def analyze(plan_binding):
    files = Files()
    plan = files.json(plan_binding)
    analysis = Analysis(plan, files)
    bindings = plan["accepted_results"]
    require(len(bindings) == 90, "Complete grid requires exactly 90 accepted results")
    paths = [b["path"] for b in bindings]
    require(len(paths) == len(set(paths)), "Duplicate accepted result path")
    cells = {}
    for binding in bindings:
        root, result, prepared = analysis.bundle(binding)
        kind = result["kind"]
        require(kind in ("dino_c1", "resnet18"), "Unexpected accepted job kind")
        cell, row = (analysis.dino(binding, root, result) if kind == "dino_c1"
                     else analysis.cnn(binding, root, result, prepared))
        require(cell in _expected_grid(), f"Unexpected grid cell: {cell}")
        require(cell not in cells, f"Duplicate grid cell: {cell}")
        cells[cell] = row
    require(set(cells) == _expected_grid(), "Missing/unexpected completed grid cells")
    for cell, row in cells.items():
        if cell[0] != "final":
            continue
        candidates = [cells[("tuning", cell[1], cell[2], lr)] for lr in LRS]
        expected = {(c["result"]["path"], c["result"]["sha256"]): c for c in candidates}
        declared = [(b["path"], b["sha256"]) for b in row["tuning_results"]]
        require(len(declared) == 3 and set(declared) == set(expected), "Final tuning candidate bindings differ")
        stripped = lambda model: {k: v for k, v in model.items() if k not in ("backbone_seed", "head_seed")}
        for candidate in candidates:
            require(candidate["inputs_sha256"] == row["inputs_sha256"]
                    and stripped(candidate["model"]) == stripped(row["model"])
                    and {k: v for k, v in candidate["recipe"].items() if k != "lr"}
                    == {k: v for k, v in row["recipe"].items() if k != "lr"}, "Tuning/final contract mixture")
        chosen = max(candidates, key=lambda c: (c["selected_score"], -c["recipe"]["lr"]))
        require(row["recipe"]["lr"] == chosen["recipe"]["lr"], "Wrong fold-local tuning LR selection")
        selection = {"lr": chosen["recipe"]["lr"],
                     "candidates": [{"lr": expected[b]["recipe"]["lr"],
                                     "score": expected[b]["selected_score"], "result": expected[b]["result"]}
                                    for b in declared],
                     "rule": "maximum validation macro_f1, then lower LR"}
        require(row["selection"] == selection, "Saved tuning selection differs")
    failures = analysis.failed(plan["failed_attempts"], set(paths))
    consumer = Path(__file__).resolve()
    consumer_binding = {"path": str(consumer), "sha256": files.hash(consumer)}
    files.unchanged()
    return {"schema_version": SCHEMA, "status": "complete_grid_verified", "purpose": plan["purpose"],
            "plan": plan_binding, "analysis_source": consumer_binding, "numpy_version": np.__version__,
            "provenance": plan["provenance"],
            "grid_counts": {"dino_c1": 30, "cnn_tuning": 30, "cnn_final": 30},
            "cells": [cells[c] for c in sorted(cells)], "failed_attempts": failures,
            "class_order": CLASSES, "fold_class_counts": analysis.counts, **aggregate(cells),
            "verification_scope": {
                "complete_bundle_file_hashes": True, "catalog_split_ids_labels": True,
                "saved_prediction_metrics": True, "all_epoch_validation_selection": True,
                "full_model_or_probability_replay": False, "state_deserialization": False,
                "raw_image_or_normalization_replay": False, "estimator_fits": False,
                "convergence_claim": False,
                "limitations": "Descriptive five-region spread; seeds do not add independent regions. "
                    "Scores measure map-label agreement. State/probability replay requires separate review."}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Report already exists")
    result = analyze({"path": str(args.plan), "sha256": args.plan_sha256})
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Like the runner, retain staging on failure and never replace prior evidence.
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}.",
                                     suffix=".partial", delete=False) as stream:
        staging = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staging, args.output)
    staging.unlink()
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "sha256": hashlib.sha256(data).hexdigest(), "grid_counts": result["grid_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
