"""Fit one declared CPU classifier to authenticated CETUS feature arrays.

This study varies standardization and L2 in the same finite LBFGS recipe.
It does not change accepted experiments or tune settings on test scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import warnings

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax as sklearn_softmax
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("dinov2", "dofa", "croma", "random_init")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Publish complete JSON without replacing a prior result.

    A failed filesystem operation retains its staging file for inspection.
    Serialization errors create no file.
    """
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        staging = Path(stream.name)
    os.link(staging, path)
    staging.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def runtime_identity():
    """Identify imported numerical code and the callable objects used here."""
    names = ("numpy", "numpy._core._multiarray_umath", "scipy", "sklearn", "joblib",
             "threadpoolctl", "sklearn.linear_model._logistic", "sklearn.linear_model._linear_loss",
             "sklearn._loss.loss", "sklearn._loss._loss", "sklearn.preprocessing._data",
             "sklearn.utils.extmath", "scipy.optimize._minimize", "scipy.optimize._lbfgsb_py",
             "scipy.optimize._lbfgsb", "scipy.optimize._differentiable_functions", "scipy.optimize._optimize")
    modules = {name: importlib.import_module(name) for name in names}

    def file_identity(path):
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": sha(path)}

    logistic = modules["sklearn.linear_model._logistic"]
    minimize = modules["scipy.optimize._minimize"]
    used = {"LogisticRegression": LogisticRegression, "LogisticRegression.fit": LogisticRegression.fit,
            "StandardScaler": StandardScaler, "StandardScaler.fit": StandardScaler.fit,
            "LinearModelLoss": logistic.LinearModelLoss, "HalfMultinomialLoss": logistic.HalfMultinomialLoss,
            "diagnostic_LinearModelLoss": LinearModelLoss, "diagnostic_HalfMultinomialLoss": HalfMultinomialLoss,
            "prediction_softmax": sklearn_softmax,
            "minimize": logistic.optimize.minimize, "lbfgs": minimize._minimize_lbfgsb}
    return {
        "modules": {name: file_identity(module.__file__) for name, module in modules.items()},
        "used": {name: {**file_identity(inspect.getfile(inspect.unwrap(obj))),
                        "module": obj.__module__, "qualname": obj.__qualname__}
                 for name, obj in used.items()},
        "numerical_libraries": {str(Path(p["filepath"]).resolve()): sha(p["filepath"])
                                for p in threadpool_info()},
    }


def verify_runtime(expected):
    actual = runtime_identity()
    require(actual == expected, "Imported runtime implementation differs")
    return actual


def validate_cells(cells):
    require(len(cells) == len({c["id"] for c in cells}) == 80, "Expected all 80 distinct cells")
    require({(c["model"], c["fold"], c["standardize"], c["regularization"]) for c in cells}
            == {(m, f, s, p) for m in MODELS for f in range(5) for s in (False, True) for p in ("none", "l2")},
            "Cell coverage differs")
    for cell in cells:
        require(type(cell["fold"]) is int and type(cell["standardize"]) is bool, "Invalid setting types")
        scale = "standardized" if cell["standardize"] else "raw"
        require(cell["id"] == f"{cell['model']}-fold-{cell['fold']}-{scale}-{cell['regularization']}",
                "Cell ID differs from its settings")


def balanced_weights(labels, classes=6):
    counts = np.bincount(labels, minlength=classes)
    require(len(counts) == classes and np.all(counts > 0), "Every training class must occur")
    return len(labels) / (classes * counts[labels].astype(np.float64))


def loss_gradient(x, y, coef, intercept, regularization):
    """Independent float64 mathematical calculation, distinct from solver arithmetic."""
    x, coef, intercept = (np.asarray(a, dtype=np.float64) for a in (x, coef, intercept))
    weights = balanced_weights(y, len(intercept))
    total = weights.sum()
    logits = x @ coef.T + intercept
    probabilities = softmax(logits, axis=1)
    data_loss = np.dot(weights, logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]) / total
    require(regularization in ("none", "l2"), "Unknown regularization")
    strength = 0.0 if regularization == "none" else 1.0 / total
    penalty = 0.5 * strength * np.square(coef).sum()
    residual = probabilities.copy()
    residual[np.arange(len(y)), y] -= 1
    residual *= (weights / total)[:, None]
    return {
        "data_loss": float(data_loss), "penalty": float(penalty),
        "objective": float(data_loss + penalty), "weight_sum": float(total),
        "gradient_coef": residual.T @ x + strength * coef,
        "gradient_intercept": residual.sum(axis=0),
        "probabilities": probabilities, "logits": logits,
    }


def solver_oracle_at_saved_state(x, y, coef, intercept, regularization):
    """Replay the pinned solver's value/gradient calculation at saved parameters.

    This uses the library's loss implementation and is not an independent
    calculation or a captured optimizer termination record. Fresh fits forward
    parameters in X.dtype through SciPy's ScalarFunction, including for L2.
    """
    require(x.dtype in (np.dtype("float32"), np.dtype("float64")), "Invalid fitting dtype")
    require(regularization in ("none", "l2"), "Unknown regularization")
    q = balanced_weights(y, len(intercept)).astype(x.dtype)
    total = float(q.sum())
    strength = 0.0 if regularization == "none" else 1.0 / total
    theta = np.asfortranarray(np.column_stack((coef, intercept)), dtype=x.dtype)
    loss = LinearModelLoss(HalfMultinomialLoss(n_classes=len(intercept)), fit_intercept=True)
    value, gradient = loss.loss_gradient(theta, x, y.astype(x.dtype), sample_weight=q,
                                         l2_reg_strength=strength, n_threads=1)
    penalty = loss.l2_penalty(theta[:, :-1], strength)
    return {"objective": float(value), "penalty": float(penalty), "data_loss": float(value - penalty),
            "weight_sum": total, "regularization_strength": strength,
            "gradient_coef": gradient[:, :-1], "gradient_intercept": gradient[:, -1],
            "input_dtype": str(x.dtype), "weight_dtype": str(q.dtype),
            "parameter_evaluation_dtype": str(theta.dtype), "gradient_dtype": str(gradient.dtype)}


def native_prediction_replay(x, coef, intercept):
    logits = x @ coef.T + intercept
    return logits.argmax(axis=1), sklearn_softmax(logits, copy=False)


def valid_probability_rows(probabilities):
    tolerance = 1e-6 if probabilities.dtype == np.float32 else 1e-12
    return bool(np.isfinite(probabilities).all() and (probabilities >= 0).all()
                and (probabilities <= 1).all()
                and np.allclose(probabilities.sum(axis=1, dtype=np.float64), 1, rtol=0, atol=tolerance))


def class_metrics(labels, predictions):
    counts = np.zeros((6, 6), dtype=np.int64)
    np.add.at(counts, (labels, predictions), 1)
    hits = np.diag(counts)
    actual, predicted = counts.sum(axis=1), counts.sum(axis=0)
    precision = np.divide(hits, predicted, out=np.zeros(6), where=predicted != 0)
    recall = np.divide(hits, actual, out=np.zeros(6), where=actual != 0)
    f1 = np.divide(2 * hits, actual + predicted, out=np.zeros(6), where=actual + predicted != 0)
    return {"confusion": counts.tolist(), "support": actual.tolist(),
            "predicted_counts": predicted.tolist(), "precision": precision.tolist(),
            "recall": recall.tolist(), "f1": f1.tolist(),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean())}


def load_inputs(plan, cell):
    protocol_path = Path(plan["accepted_protocol"])
    require(sha(protocol_path) == plan["accepted_protocol_sha256"], "Accepted input specification changed")
    spec = json.loads(protocol_path.read_text())["specification"]
    matches = [r for r in spec["caches"] if (r["model"], r["fold"]) == (cell["model"], cell["fold"])]
    require(len(matches) == 1, "Expected one accepted model/fold cache")
    cache = matches[0]
    root = Path(plan["feature_root"]) / cell["model"] / f"fold-{cell['fold']}"
    require(sha(root / "run_manifest.json") == cache["manifest_sha256"], "Extraction manifest changed")
    for relative, expected in cache["files_sha256"].items():
        require(sha(root / relative) == expected, f"Accepted cache changed: {relative}")
    arrays = {}
    for role in ("train", "test"):
        for suffix in ("feats", "ids", "labels"):
            path = root / "features" / cell["model"] / f"titan_{role}_{suffix}.npy"
            arrays[role + "_" + suffix] = np.load(path, allow_pickle=False)
        x, ids, y = (arrays[role + "_" + s] for s in ("feats", "ids", "labels"))
        require(x.dtype == np.float32 and x.shape == (len(y), 768) and np.isfinite(x).all(), "Invalid features")
        require(ids.shape == y.shape and ids.dtype.kind in "US" and len(set(ids.tolist())) == len(ids), "Invalid IDs")
        require(y.dtype.kind in "iu" and set(np.unique(y)) == set(range(6)), "Invalid labels")
        bound = spec["folds"][cell["fold"]]
        require(np.bincount(y, minlength=6).tolist() == bound["counts"][role]["class_counts"], "Class counts changed")
    require(not set(arrays["train_ids"]).intersection(arrays["test_ids"]), "Training/test overlap")
    return arrays, cache


def fit_cell(plan, cell, plan_sha256):
    require(cell["model"] in MODELS and cell["fold"] in range(5), "Invalid model/fold")
    require(type(cell["standardize"]) is bool and cell["regularization"] in ("none", "l2"), "Invalid fitting cell")
    scale_name = "standardized" if cell["standardize"] else "raw"
    require(cell["id"] == f"{cell['model']}-fold-{cell['fold']}-{scale_name}-{cell['regularization']}", "Cell ID differs from its settings")
    arrays, cache = load_inputs(plan, cell)
    output = Path(plan["output_root"]) / cell["id"]
    for protected in (ROOT, Path(plan["feature_root"])):
        require(not output.resolve().is_relative_to(protected.resolve()), "Output overlaps protected inputs")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    initial = {"cell": cell, "started_utc": started, "producer_commit": plan["producer_commit"],
               "plan_sha256": plan_sha256, "command": [sys.executable, *sys.argv],
               "runtime": {k: plan[k] for k in ("python_version", "package_versions", "runtime_identity")},
               "source_sha256": plan["source_sha256"], "cache": cache,
               "restored_cache": str(Path(plan["feature_root"]) / cell["model"] / f"fold-{cell['fold']}")}
    try:
        write_json(output / "started.json", initial)
        return fit_arrays(arrays, cell, output, initial)
    except BaseException as error:
        try:
            write_json(output / "failed.json", {
                **initial, "status": "failed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__, "error": str(error),
                "process_exit": "Recorded by the launcher after this process exits",
            })
        except Exception as receipt_error:
            error.add_note(f"Failure receipt could not be written: {receipt_error!r}")
        raise


def fit_arrays(arrays, cell, output, initial):
    train, test = arrays["train_feats"], arrays["test_feats"]
    reference = StandardScaler().fit(train)
    if cell["standardize"]:
        train, test = reference.transform(train), reference.transform(test)
    c = np.inf if cell["regularization"] == "none" else 1.0
    classifier = LogisticRegression(C=c, l1_ratio=0.0, class_weight="balanced", solver="lbfgs",
                                    fit_intercept=True, max_iter=2000, tol=1e-4,
                                    random_state=0, warm_start=False)
    before = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(train, arrays["train_labels"])
    elapsed = time.perf_counter() - before
    require(classifier.classes_.tolist() == list(range(6)), "Unexpected class order")
    require(np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all(), "Nonfinite fitted state")
    calculation = loss_gradient(train, arrays["train_labels"], classifier.coef_, classifier.intercept_, cell["regularization"])
    oracle = solver_oracle_at_saved_state(train, arrays["train_labels"], classifier.coef_,
                                         classifier.intercept_, cell["regularization"])
    pred_train, pred_test = classifier.predict(train), classifier.predict(test)
    prob_train, prob_test = classifier.predict_proba(train), classifier.predict_proba(test)
    for x, pred, prob in ((train, pred_train, prob_train), (test, pred_test, prob_test)):
        replay_pred, replay_prob = native_prediction_replay(x, classifier.coef_, classifier.intercept_)
        require(np.array_equal(pred, replay_pred) and np.array_equal(prob, replay_prob), "Native prediction replay differs")
        require(valid_probability_rows(prob), "Invalid probability rows")
    state = {"coef": classifier.coef_, "intercept": classifier.intercept_, "classes": classifier.classes_,
             "reference_mean": reference.mean_, "reference_scale": reference.scale_, "reference_var": reference.var_,
             "train_ids": arrays["train_ids"], "test_ids": arrays["test_ids"],
             "train_labels": arrays["train_labels"], "test_labels": arrays["test_labels"],
             "train_predictions": pred_train, "test_predictions": pred_test,
             "train_probabilities": prob_train, "test_probabilities": prob_test}
    warning_rows = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    parameters = classifier.get_params()
    parameters["C"] = "infinity" if np.isinf(c) else c
    concerns = []
    if warning_rows:
        concerns.append("warnings")
    if int(classifier.n_iter_.max()) >= 2000:
        concerns.append("iteration_limit")
    diagnostics = {}
    for name, values in (("mathematical_float64", calculation), ("solver_oracle", oracle)):
        g, gb = (values[k].astype(np.float64) for k in ("gradient_coef", "gradient_intercept"))
        common_g = g if cell["standardize"] else (g - gb[:, None] * reference.mean_) / reference.scale_
        residual = float(max(np.abs(common_g).max(), np.abs(gb).max()))
        diagnostics[name] = {k: v for k, v in values.items() if not isinstance(v, np.ndarray)}
        diagnostics[name].update(gradient_native_inf=float(max(np.abs(g).max(), np.abs(gb).max())),
                                 gradient_reference_inf=residual)
        state.update({name + "_gradient_coef": values["gradient_coef"],
                      name + "_gradient_intercept": values["gradient_intercept"],
                      name + "_gradient_reference_coef": common_g})
        if residual > 1e-4:
            concerns.append(name + "_reference_gradient_above_1e-4")
    test_logits64 = test.astype(np.float64) @ classifier.coef_.astype(np.float64).T + classifier.intercept_.astype(np.float64)
    diagnostic_difference = {
        "objective_absolute": abs(calculation["objective"] - oracle["objective"]),
        "gradient_coef_max_absolute": float(np.max(np.abs(calculation["gradient_coef"] - oracle["gradient_coef"]))),
        "gradient_intercept_max_absolute": float(np.max(np.abs(calculation["gradient_intercept"] - oracle["gradient_intercept"]))),
        "train_probability_max_absolute": float(np.max(np.abs(prob_train - calculation["probabilities"]))),
        "train_argmax_changes": int(np.count_nonzero(pred_train != calculation["logits"].argmax(axis=1))),
        "test_probability_max_absolute": float(np.max(np.abs(prob_test - softmax(test_logits64, axis=1)))),
        "test_argmax_changes": int(np.count_nonzero(pred_test != test_logits64.argmax(axis=1))),
    }
    np.savez_compressed(output / "fitted_arrays.npz", **state)
    with (output / "fitted_arrays.npz").open("rb") as stream:
        os.fsync(stream.fileno())
    result = {**initial, "status": "finite_fit", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "fit_seconds": elapsed, "parameters": parameters, "warnings": warning_rows,
              "iterations": classifier.n_iter_.tolist(), "numerical_concerns": concerns,
              "termination_reason": None, "function_evaluations": None, "trajectory": None,
              "unavailable_fields_reason": "Not returned by the public LogisticRegression estimator",
              "diagnostics": diagnostics, "float64_vs_native": diagnostic_difference,
              "diagnostic_limits": "Float64 is an independent mathematical calculation; solver_oracle reuses the pinned library at saved parameters. Neither captures the termination reason. Reference residuals transform each computed gradient, including its own penalty; they are screening quantities.",
              "coefficient_norm": float(np.linalg.norm(classifier.coef_)),
              "intercept_norm": float(np.linalg.norm(classifier.intercept_)),
              "train": class_metrics(arrays["train_labels"], pred_train),
              "test": class_metrics(arrays["test_labels"], pred_test),
              "transformed_dtype": str(train.dtype), "coefficient_dtype": str(classifier.coef_.dtype),
              "threadpools": threadpool_info(), "arrays_sha256": sha(output / "fitted_arrays.npz"),
              "limits": "Finite fitting result on previously examined tests. Residuals are evaluated at saved coefficients; small residuals do not prove existence or uniqueness of an MLE."}
    write_json(output / "result.json", result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--cell", required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, "Study specification changed")
    plan = json.loads(args.plan.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == plan["producer_commit"], "Producer commit differs")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True), "Scientific source must be clean")
    for relative, expected in plan["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "Scientific source bytes changed")
    for name, expected in plan["package_versions"].items():
        require(importlib.metadata.version(name) == expected, f"Package differs: {name}")
    require(platform.python_version() == plan["python_version"], "Python version differs")
    verify_runtime(plan["runtime_identity"])
    cells = plan["cells"]
    validate_cells(cells)
    selected = [c for c in cells if c["id"] == args.cell]
    require(len(selected) == 1, "Unknown study cell")
    with threadpool_limits(limits=1):
        require(all(p["num_threads"] == 1 for p in threadpool_info()), "Numerical thread limit differs")
        output = fit_cell(plan, selected[0], args.plan_sha256)
    print(json.dumps({"status": "finite_fit", "output": str(output)}))


if __name__ == "__main__":
    main()
