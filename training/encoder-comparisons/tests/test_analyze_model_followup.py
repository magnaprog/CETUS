"""Complete-grid synthetic tests: no models, Torch, fitting, or real-data claim."""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import analyze_model_followup as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return bind(path)


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def saved_scores(truth, predicted):
    """Small scalar/Fraction oracle, independent of the consumer arithmetic."""
    labels = truth["labels"]
    matrix = [[0] * 6 for _ in range(6)]
    for y, p in zip(labels, predicted):
        matrix[y][p] += 1
    support = [sum(row) for row in matrix]
    called = [sum(matrix[i][j] for i in range(6)) for j in range(6)]
    tp = [matrix[i][i] for i in range(6)]
    precision = [float(Fraction(tp[i], called[i])) if called[i] else 0.0 for i in range(6)]
    recall = [float(Fraction(tp[i], support[i])) for i in range(6)]
    f1 = [float(Fraction(2 * tp[i], support[i] + called[i])) for i in range(6)]
    n = len(labels)
    observed = Fraction(sum(tp), n)
    expected = sum(Fraction(s * p, n * n) for s, p in zip(support, called))
    logits = [[1.0 if j == p else 0.0 for j in range(6)] for p in predicted]
    probabilities = [[1.0 if j == p else 0.0 for j in range(6)] for p in predicted]
    return {**truth, "predictions": predicted, "confusion_matrix": matrix,
            "support_per_class": support, "true_positive": tp,
            "false_positive": [p - t for p, t in zip(called, tp)],
            "false_negative": [s - t for s, t in zip(support, tp)],
            "precision_per_class": precision, "recall_per_class": recall, "f1_per_class": f1,
            "macro_f1": sum(f1) / 6, "balanced_accuracy": sum(recall) / 6,
            "kappa": float((observed - expected) / (1 - expected)),
            "logits": logits, "probabilities": probabilities, "logits_dtype": "float32",
            "probabilities_dtype": "float32", "class_order": list(range(6)),
            "probability_argmax_disagreements": 0, "loss": 1.0,
            "loss_numerator": float(n), "loss_denominator": float(n)}


def publish(root, spec, inputs, extra, selection=None, files=None):
    root.mkdir(parents=True, exist_ok=True)
    spec = {**spec, "output_dir": str(root)}
    write(root / "started.json", {"status": "started", "specification": spec,
                                  "specification_sha256": a.digest(spec)})
    write(root / "preparation.json", {
        "status": "prepared_no_model_execution", "ready": True, "missing_prerequisites": [],
        "specification_sha256": a.digest(spec), "inputs": inputs,
        "source": {"commit": spec["source"]["commit"], "clean": True},
        "source_files": spec["source"]["files"], "runtime": spec["runtime"], "selection": selection})
    for name, data in (files or {}).items():
        (root / name).write_bytes(data)
    result = {"status": "complete", "kind": spec["kind"], "purpose": spec["purpose"],
              "fold": spec["fold"], "specification": spec, "specification_sha256": a.digest(spec),
              "inputs": inputs, **extra}
    return republish(root, result)


def republish(root, result):
    result["outputs"] = {str(p.relative_to(root)): bind(p)["sha256"]
                         for p in root.rglob("*") if p.is_file() and p.name != "result.json"}
    return write(root / "result.json", result)


def make_grid(tmp_path):
    provenance = {kind: {"source": {"commit": "c" * 40, "files": {"runner.py": "a" * 64}},
                         "runtime": {"synthetic": kind}}
                  for kind in ("dino_features", "dino_c1", "resnet18")}
    catalog = {"intensity_space": "hisar_log_dn", "benchmark_track": "synthetic_fixture", "tiles": []}
    for region in range(5):
        for label in range(6):
            for repeat in range(label + 1 + region):
                catalog["tiles"].append({"tile_id": f"r{region}-c{label}-n{repeat}",
                                         "label": label, "tile_sha256": "a" * 64, "region": region})
    catalog_binding = write(tmp_path / "catalog.json", catalog)
    splits, inputs_by_fold = [], []
    for fold in range(5):
        assignments = {e["tile_id"]: "test" if e["region"] == fold else
                       "val" if e["region"] == (fold + 1) % 5 else "train" for e in catalog["tiles"]}
        split = write(tmp_path / f"split-{fold}.json", {"catalog_sha256": catalog_binding["sha256"],
                      "policy": {"test_fold": fold}, "assignments": assignments})
        splits.append(split)
        ordered, counts = {}, {}
        for role in a.ROLES:
            rows = [e for e in catalog["tiles"] if assignments[e["tile_id"]] == role]
            ordered[role] = {"tile_ids": [e["tile_id"] for e in rows], "labels": [e["label"] for e in rows]}
            counts[role] = {"total": len(rows), "class_counts": [ordered[role]["labels"].count(i) for i in range(6)]}
        normalization = {"enabled": True, "fallback_used": False, "intensity_space": "hisar_log_dn",
                         "sampling": {"sample_count": 1, "missing_tile_ids": [], "empty_tile_ids": []},
                         "training_ids_sha256": a.digest(ordered["train"]["tile_ids"])}
        inverse = 1 / np.array(counts["train"]["class_counts"], dtype=float)
        inputs_by_fold.append({"catalog": catalog_binding, "split": split,
            "catalog_count": len(catalog["tiles"]), "class_order": a.CLASSES, "counts": counts, "ordered": ordered,
            "benchmark_track": "synthetic_fixture", "tile_inventory_sha256": a.digest({e["tile_id"]: "a" * 64 for e in catalog["tiles"]}),
            "normalization": normalization, "normalization_sha256": a.digest(normalization),
            "training_class_weights": (inverse / inverse.sum() * 6).astype(np.float32).tolist()})

    def base(kind, fold):
        return {"schema_version": "model-followup-v1", "kind": kind, "purpose": "synthetic_cpu_test",
                "fold": fold, "device": "cpu", **provenance[kind]}

    def predict(truth, arm, seed):
        labels = truth["labels"]
        # Deliberately different called-class distributions: pooling F1 is wrong.
        return [y if arm == "pretrained" or i % (seed % 3 + 2) else 0 for i, y in enumerate(labels)]

    accepted = []
    source_files = {"synthetic.py": "a" * 64}
    checkpoint = tmp_path / "synthetic-weights.bin"
    checkpoint.write_bytes(b"synthetic weights; deliberately not a loadable state")
    for fold, inputs in enumerate(inputs_by_fold):
        for arm, seeds in (("random", range(42, 47)), ("pretrained", (42,))):
            for seed in seeds:
                identity = {"model": f"dinov2_{arm}_matched", "initialization": arm, "encoder_seed": seed,
                    "transform": a.TRANSFORM, "dtype": "float32", "backend": "torch_sdpa_no_xformers",
                    "revision": a.DINO_REVISION, "constructor": a.DINO_CONSTRUCTOR,
                    "source_python_inventory_sha256": a.digest(source_files), "checkpoint": bind(checkpoint),
                    "initial_state_sha256": a.digest([arm, seed])}
                model = {k: identity[k] for k in ("initialization", "encoder_seed", "checkpoint", "initial_state_sha256")}
                model.update(source_revision=a.DINO_REVISION, source_files=source_files)
                root = tmp_path / f"features-{fold}-{arm}-{seed}"
                (root / "features").mkdir(parents=True)
                caches = {}
                for role in a.ROLES:
                    expected = {"identity": identity, "inputs": inputs, "fold": fold, "role": role,
                                "transform": a.TRANSFORM, "purpose": "synthetic_cpu_test", **inputs["ordered"][role]}
                    caches[role] = expected
                    hashes = {}
                    for suffix, field, values in (("feats", "feature_sha256", b"not deserialized by this check"),
                            ("ids", "tile_ids_sha256", np.array(expected["tile_ids"])),
                            ("labels", "labels_sha256", np.array(expected["labels"]))):
                        path = root / "features" / f"titan_{role}_{suffix}.npy"
                        if suffix == "feats":
                            path.write_bytes(values)
                        else:
                            np.save(path, values)
                        hashes[field] = bind(path)["sha256"]
                    write(root / "features" / f"titan_{role}_feats.meta.json", {
                        "schema_version": "model-followup-v1", "model": identity["model"], "weights_source": arm,
                        "array": f"titan_{role}_feats", "n_features": len(expected["tile_ids"]), "feature_dim": 768,
                        "provenance": {**expected, **hashes}})
                producer = publish(root, {**base("dino_features", fold), "inputs": inputs,
                    "model": model, "recipe": {"batch_size": 32}}, inputs,
                    {"identity": identity, "caches": caches}, files={"initial_state.pt": b"not a Torch object"})
                root = tmp_path / f"dino-{fold}-{arm}-{seed}"
                accepted.append(publish(root, {**base("dino_c1", fold), "features": producer,
                    "feature_identity": identity}, inputs, {"identity": identity,
                    "classifier_parameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4,
                        "class_weight": "balanced", "random_state": 0, "fit_intercept": True, "penalty": "l2"},
                    "scaler_parameters": {"copy": True, "with_mean": True, "with_std": True},
                    "feature_dtype": "float32", "coefficient_dtype": "float64", "n_iter": [7], "warnings": [],
                    "convergence_claim": False, "predictions": {role: saved_scores(inputs["ordered"][role],
                        predict(inputs["ordered"][role], arm, seed)) for role in ("val", "test")}},
                    files={"estimator.npz": b"separate estimator replay required"}))
        for arm in ("scratch", "pretrained"):
            candidates = []
            for phase, settings in (("tuning", [(100, lr) for lr in a.LRS]),
                                    ("final", [(seed, a.LRS[0]) for seed in range(3)])):
                for seed, lr in settings:
                    model = {"initialization": arm, "backbone_seed": seed, "head_seed": seed + 20000,
                             "torchvision_version": "synthetic", "resnet_source_sha256": "a" * 64}
                    if arm == "pretrained":
                        model.update(weights="ResNet18_Weights.IMAGENET1K_V1", checkpoint=bind(checkpoint))
                    identity = {**model, "model": "resnet18", "transform": a.TRANSFORM, "num_classes": 6,
                        "dtype": "float32", "initial_state_sha256": a.digest([arm, seed]),
                        "head_state_sha256": a.digest(seed), "checkpoint": model.get("checkpoint"),
                        "weights": model.get("weights")}
                    root = tmp_path / f"cnn-{fold}-{arm}-{phase}-{seed}-{lr}"
                    root.mkdir()
                    history = []
                    validation = saved_scores(inputs["ordered"]["val"], inputs["ordered"]["val"]["labels"])
                    for epoch in range(1, 3):
                        row = {"epoch": epoch, "lr": lr * (1 + math.cos(math.pi * (epoch - 1) / 2)) / 2,
                               "val_macro_f1": validation["macro_f1"], "val_confusion_matrix": validation["confusion_matrix"],
                               "val_loss": 1.0, "val_loss_numerator": validation["loss_numerator"],
                               "val_loss_denominator": validation["loss_denominator"], "train_loss": 1.0,
                               "loss_numerator": float(inputs["counts"]["train"]["total"]),
                               "loss_denominator": float(inputs["counts"]["train"]["total"]),
                               "optimizer_steps": math.ceil(inputs["counts"]["train"]["total"] / 6),
                               "examples": inputs["counts"]["train"]["total"], "finite_gradients": True}
                        history.append(row)
                        write(root / f"epoch-{epoch:03d}.json", {"history": row, "validation": validation})
                    selection = None if phase == "tuning" else {"lr": a.LRS[0], "candidates": candidates,
                        "rule": "maximum validation macro_f1, then lower LR"}
                    extra = {"identity": identity, "history": history, "selected_epoch": 1,
                             "selected_validation": validation, "selection": selection,
                             "selected_state_sha256": "b" * 64, "final_state_sha256": "c" * 64,
                             "training_class_weights": inputs["training_class_weights"]}
                    spec = {**base("resnet18", fold), "model": model, "inputs": inputs, "phase": phase,
                            "seed": seed, "loader_seed": seed + 10000,
                            "recipe": {**a.RECIPE, "epochs": 2, "batch_size": 6, "lr": lr}}
                    if phase == "final":
                        spec["tuning_results"] = [c["result"] for c in candidates]
                        extra["test"] = saved_scores(inputs["ordered"]["test"], predict(inputs["ordered"]["test"], arm, seed))
                    binding = publish(root, spec, inputs, extra, selection,
                                      {name: b"not a Torch object" for name in ("initial_state.pt", "selected.pt", "final.pt")})
                    accepted.append(binding)
                    if phase == "tuning":
                        candidates.append({"lr": lr, "score": validation["macro_f1"], "result": binding})
    return {"schema_version": a.SCHEMA, "purpose": "synthetic_cpu_test", "catalog": catalog_binding,
            "splits": splits, "provenance": provenance, "accepted_results": accepted, "failed_attempts": []}


@pytest.fixture
def grid(tmp_path):
    return tmp_path, make_grid(tmp_path)


def run(grid):
    root, plan = grid
    return a.analyze(write(root / "plan.json", plan))


def find_result(plan, kind="dino_c1", phase=None):
    for i, binding in enumerate(plan["accepted_results"]):
        result = a.read_json(binding["path"])
        if result["kind"] == kind and (phase is None or result["specification"]["phase"] == phase):
            return i, Path(binding["path"]).parent, result
    raise AssertionError("fixture cell missing")


def refresh(plan, index, root, result, spec_changed=False):
    if spec_changed:
        spec = result["specification"]
        result["specification_sha256"] = a.digest(spec)
        started = a.read_json(root / "started.json")
        started.update(specification=spec, specification_sha256=a.digest(spec))
        write(root / "started.json", started)
        prepared = a.read_json(root / "preparation.json")
        prepared.update(specification_sha256=a.digest(spec))
        write(root / "preparation.json", prepared)
    plan["accepted_results"][index] = republish(root, result)


def test_complete_grid_and_seed_first_estimator(grid):
    report = run(grid)
    assert report["grid_counts"] == {"dino_c1": 30, "cnn_tuning": 30, "cnn_final": 30}
    assert report["comparison_questions"] == 2 and report["regional_differences"] == 10
    assert report["verification_scope"]["full_model_or_probability_replay"] is False
    random = [c["test"] for c in report["cells"] if c["cell"][:3] == ["dino_c1", 0, "random"]]
    mean_f1 = sum(c["macro_f1"] for c in random) / 5
    matrix = np.sum([c["confusion_matrix"] for c in random], axis=0)
    pooled = np.mean(2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1)))
    actual = report["arms"]["dino_c1:random"]["regions"][0]["metrics"]["macro_f1"]["mean"]
    assert actual == pytest.approx(mean_f1) and abs(actual - pooled) > 1e-5
    for contrast in report["contrasts"]:
        values = [r["within_region"]["mean"] for r in contrast["regions"]]
        assert contrast["equal_region"]["mean"] == pytest.approx(sum(values) / 5)
        assert contrast["equal_region"]["sample_sd"] == pytest.approx(np.std(values, ddof=1))
    assert report["arms"]["dino_c1:pretrained"]["regions"][0]["metrics"]["macro_f1"]["sample_sd"] is None


@pytest.mark.parametrize("fault,match", [("missing", "exactly 90"), ("duplicate_path", "Duplicate accepted"),
                                       ("duplicate_cell", "Duplicate grid"), ("unexpected", "Unexpected CNN")])
def test_exact_grid_rejects_missing_duplicate_unexpected(grid, fault, match):
    _, plan = grid
    if fault == "missing":
        plan["accepted_results"].pop()
    elif fault == "duplicate_path":
        plan["accepted_results"][-1] = plan["accepted_results"][0]
    else:
        i, root, result = find_result(plan, "resnet18", "final")
        if fault == "unexpected":
            result["specification"]["seed"] = 77
        else:
            # A second path for an existing cell is still a duplicate.
            other = root.parent / "duplicate-attempt"
            import shutil
            shutil.copytree(root, other)
            root = other
            result["specification"]["output_dir"] = str(other)
            i = len(plan["accepted_results"]) - 1
        refresh(plan, i, root, result, spec_changed=True)
    with pytest.raises(ValueError, match=match):
        run(grid)


@pytest.mark.parametrize("fault,match", [
    ("labels", "IDs/labels"), ("fold", "job identity"), ("confusion", "confusion_matrix"),
    ("source", "Source/runtime"), ("runtime", "Source/runtime"),
    ("state_file", "Inventory SHA"), ("extra_file", "inventory coverage"),
    ("epoch_tie", "Selected epoch"), ("missing_epoch", "Epoch file coverage"),
    ("lr_choice", "Wrong fold-local"), ("candidate", "candidate bindings"),
    ("zero_updates", "update/example"), ("test_tuning", "Tuning must not"),
])
def test_refreshed_hash_semantic_mutations(grid, fault, match):
    _, plan = grid
    cnn = fault in {"epoch_tie", "missing_epoch", "lr_choice", "candidate", "zero_updates", "test_tuning"}
    phase = "tuning" if fault in {"zero_updates", "test_tuning"} else "final"
    i, root, result = find_result(plan, "resnet18" if cnn else "dino_c1", phase if cnn else None)
    spec_changed = False
    if fault == "labels":
        result["predictions"]["test"]["labels"][0] = 1
    elif fault == "fold":
        result["fold"] = 4
    elif fault == "confusion":
        result["predictions"]["test"]["confusion_matrix"][0][0] += 1
    elif fault in ("source", "runtime"):
        result["specification"][fault]["unapproved_change"] = True
        spec_changed = True
    elif fault == "state_file":
        (root / "estimator.npz").write_bytes(b"changed")
        plan["accepted_results"][i] = write(root / "result.json", result)
    elif fault == "extra_file":
        (root / "extra.bin").write_bytes(b"undeclared")
    elif fault == "epoch_tie":
        result["selected_epoch"] = 2
    elif fault == "missing_epoch":
        (root / "epoch-002.json").unlink()
    elif fault == "lr_choice":
        result["specification"]["recipe"]["lr"] = .01
        for row in result["history"]:
            row["lr"] *= 100
            path = root / f"epoch-{row['epoch']:03d}.json"
            epoch = a.read_json(path)
            epoch["history"] = row
            write(path, epoch)
        spec_changed = True
    elif fault == "candidate":
        result["specification"]["tuning_results"][2] = result["specification"]["tuning_results"][0]
        spec_changed = True
    elif fault == "zero_updates":
        result["history"][0]["optimizer_steps"] = 0
        path = root / "epoch-001.json"
        epoch = a.read_json(path)
        epoch["history"] = result["history"][0]
        write(path, epoch)
    elif fault == "test_tuning":
        result["test"] = result["selected_validation"]
    if fault not in ("state_file", "extra_file"):
        refresh(plan, i, root, result, spec_changed)
    with pytest.raises(ValueError, match=match):
        run(grid)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_rejected_with_updated_plan_hash(grid, value):
    _, plan = grid
    i, root, result = find_result(plan)
    path = root / "result.json"
    path.write_text(json.dumps(result)[:-1] + ', "bad_number": ' + value + '}')
    plan["accepted_results"][i] = bind(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        run(grid)


def test_feature_label_mutation_cannot_be_hidden_by_refreshed_hashes(grid):
    _, plan = grid
    i, root, result = find_result(plan)
    feature_path = Path(result["specification"]["features"]["path"])
    feature = a.read_json(feature_path)
    labels_path = feature_path.parent / "features/titan_test_labels.npy"
    labels = np.load(labels_path)
    labels[0] = 1
    np.save(labels_path, labels)
    meta_path = feature_path.parent / "features/titan_test_feats.meta.json"
    meta = a.read_json(meta_path)
    meta["provenance"]["labels_sha256"] = bind(labels_path)["sha256"]
    write(meta_path, meta)
    result["specification"]["features"] = republish(feature_path.parent, feature)
    refresh(plan, i, root, result, spec_changed=True)
    with pytest.raises(ValueError, match="companion IDs/labels"):
        run(grid)


def test_failed_attempt_retained_and_digest_joined(grid):
    root, plan = grid
    _, _, result = find_result(plan)
    spec = copy.deepcopy(result["specification"])
    spec["output_dir"] = str(root / "failed-attempt")
    started = Path(spec["output_dir"]) / "started.json"
    write(started, {"status": "started", "specification": spec, "specification_sha256": a.digest(spec)})
    failure = started.with_name("failed.json")
    plan["failed_attempts"] = [write(failure, {"status": "failed", "specification_sha256": a.digest(spec),
                                             "error_type": "RuntimeError", "error": "synthetic failure"})]
    report = run(grid)
    assert len(report["cells"]) == 90 and len(report["failed_attempts"]) == 1
    assert report["failed_attempts"][0]["specification"] == spec
    bad = a.read_json(failure)
    bad["specification_sha256"] = "f" * 64
    plan["failed_attempts"] = [write(failure, bad)]
    with pytest.raises(ValueError, match="Failed attempt specification"):
        run(grid)


def test_selection_uses_later_improvement_all_candidates_and_fold_local_scores(grid):
    _, plan = grid
    candidates = []
    # Only scratch fold 0 changes: 1e-3 wins at epoch 2; other folds retain 1e-4.
    for index, binding in enumerate(plan["accepted_results"]):
        root = Path(binding["path"]).parent
        result = a.read_json(binding["path"])
        spec = result["specification"]
        if spec["kind"] != "resnet18" or spec["fold"] != 0 or spec["model"]["initialization"] != "scratch":
            continue
        if spec["phase"] != "tuning":
            continue
        for row in result["history"]:
            truth = result["inputs"]["ordered"]["val"]
            predictions = truth["labels"] if spec["recipe"]["lr"] == .001 and row["epoch"] == 2 else [0] * len(truth["labels"])
            validation = saved_scores(truth, predictions)
            row.update(val_macro_f1=validation["macro_f1"], val_confusion_matrix=validation["confusion_matrix"])
            write(root / f"epoch-{row['epoch']:03d}.json", {"history": row, "validation": validation})
        result["selected_epoch"] = 2 if spec["recipe"]["lr"] == .001 else 1
        result["selected_validation"] = a.read_json(root / f"epoch-{result['selected_epoch']:03d}.json")["validation"]
        refresh(plan, index, root, result)
        candidates.append({"lr": spec["recipe"]["lr"], "score": result["selected_validation"]["macro_f1"],
                           "result": plan["accepted_results"][index]})
    selection = {"lr": .001, "candidates": candidates, "rule": "maximum validation macro_f1, then lower LR"}
    for index, binding in enumerate(plan["accepted_results"]):
        root = Path(binding["path"]).parent
        result = a.read_json(binding["path"])
        spec = result["specification"]
        if spec["kind"] != "resnet18" or spec["fold"] != 0 or spec["model"]["initialization"] != "scratch" or spec["phase"] != "final":
            continue
        spec["recipe"]["lr"] = .001
        spec["tuning_results"] = [c["result"] for c in candidates]
        result["selection"] = selection
        prepared = a.read_json(root / "preparation.json")
        prepared["selection"] = selection
        write(root / "preparation.json", prepared)
        for row in result["history"]:
            row["lr"] *= 10
            epoch_path = root / f"epoch-{row['epoch']:03d}.json"
            epoch = a.read_json(epoch_path)
            epoch["history"] = row
            write(epoch_path, epoch)
        refresh(plan, index, root, result, spec_changed=True)
    report = run(grid)
    finals = [row for row in report["cells"] if row["cell"][0] == "final" and row["cell"][2] == "scratch"]
    assert all(row["recipe"]["lr"] == (.001 if row["cell"][1] == 0 else .0001) for row in finals)
    chosen = [row for row in report["cells"] if row["cell"] == ["tuning", 0, "scratch", .001]][0]
    assert chosen["selected_epoch"] == 2


def test_cli_no_torch_import_and_no_overwrite(grid, capsys):
    root, plan = grid
    binding = write(root / "plan.json", plan)
    output = root / "report.json"
    assert a.main(["--plan", binding["path"], "--plan-sha256", binding["sha256"], "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete_grid_verified"
    with pytest.raises(ValueError, match="already exists"):
        a.main(["--plan", binding["path"], "--plan-sha256", binding["sha256"], "--output", str(output)])
    subprocess.run([sys.executable, "-c", "import sys; from scripts import analyze_model_followup; "
                    "assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules"], check=True)


def test_input_plan_seal_and_duplicate_json_keys(grid):
    root, plan = grid
    binding = write(root / "plan.json", plan)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="File SHA differs"):
        a.analyze(binding)
    path = root / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        a.read_json(path)
