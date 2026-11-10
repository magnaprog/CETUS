"""One offline preparation, DINO extraction/C=1 readout, or ResNet18 fit.

CLI: python -m scripts.run_model_followup --spec JOB.json [--execute]
Default is read-only preparation. JOB.json contains specification and its
canonical specification_sha256. See prepare() for readiness, not a model claim.
No downloads, grid launch, automatic retries or budget extension are provided.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings
from uuid import UUID

import numpy as np
import torch

from scripts import model_followup_support as support
from scripts import loss_comparison
from titansar.models.baselines import evaluate_predictions, fit_classical_probe
from titansar.models.followup import (
    DINOV2_SHA256, RESNET18_SHA256, RESNET_WEIGHTS, TRANSFORM, build_resnet18, check_file,
    load_matched_dino, require, state_digest, verify_dino_source, verify_resnet_source,
)
from titansar.models.image_training import evaluate_images, train_epoch
from titansar.models.probing import extract_features
from titansar.reproducibility import artifact_hashes, git_state, sha256_file


def _gpu_uuid(value):
    return "GPU-" + str(UUID(str(value).removeprefix("GPU-")))


def _validate_spec(spec):
    if "loss_comparison" in spec:
        loss_comparison.validate_spec(spec)
    require(spec["schema_version"] == "model-followup-v1", "Unknown specification schema")
    require(spec["kind"] in ("dino_features", "dino_c1", "resnet18"), "Unknown job kind")
    require(spec["purpose"] in ("synthetic_cpu_test", "titan_followup"), "Unknown purpose")
    require(type(spec["fold"]) is int and spec["fold"] in range(5), "Invalid fold")
    require(torch.get_num_threads() <= 2, "Limit Torch to at most two CPU threads")
    require(torch.are_deterministic_algorithms_enabled(), "Enable deterministic algorithms before sealing runtime")
    require(torch.get_default_dtype() == torch.float32, "Models require the float32 default dtype")
    device = spec["device"]
    cuda = (isinstance(device, str) and device.startswith("cuda:")
            and device[5:].isascii() and device[5:].isdigit())
    require(device == "cpu" or cuda, "Explicit CPU or indexed CUDA device required")
    if cuda:
        # Host-side configuration checks only: preparation must not initialize CUDA.
        require(torch.get_num_interop_threads() <= 2, "Limit Torch inter-op threads to at most two")
        require(not torch.is_deterministic_algorithms_warn_only_enabled(), "CUDA determinism must raise, not warn")
        require(torch.backends.cuda.matmul.fp32_precision == "ieee", "CUDA matmul must use IEEE FP32")
        require(torch.backends.cudnn.conv.fp32_precision == "ieee", "cuDNN convolution must use IEEE FP32")
        require(not torch.is_autocast_enabled("cuda"), "CUDA autocast must be disabled")
        require(not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
                "Require deterministic cuDNN with benchmarking disabled")
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the producer")
        require(all(os.environ.get(name) in (None, "0") for name in
                    ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")),
                "TF32 environment overrides must be unset or zero")
        execution = spec.get("execution", {})
        expected_uuid = execution.get("gpu_uuid")
        require(isinstance(expected_uuid, str) and expected_uuid.startswith("GPU-")
                and expected_uuid == _gpu_uuid(expected_uuid), "Declare a canonical full GPU UUID")
        require(device == "cuda:0" and os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid,
                "Expose exactly the declared GPU UUID as logical cuda:0")
        receipt = execution.get("cuda_dependencies")
        require(isinstance(receipt, dict) and set(receipt) == {"path", "sha256"},
                "Bind the external CUDA dependency receipt")
    if spec["purpose"] == "synthetic_cpu_test" or spec["kind"] == "dino_c1":
        require(device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU jobs require hidden CUDA")
    if spec["kind"] == "resnet18":
        recipe = dict(spec["recipe"])
        lr = recipe.pop("lr")
        require(lr in support.CANDIDATE_LRS, "LR outside declared three-candidate budget")
        expected = dict(support.RECIPE)
        if spec["purpose"] == "synthetic_cpu_test":
            require(type(recipe["epochs"]) is int and 1 <= recipe["epochs"] <= 3
                    and type(recipe["batch_size"]) is int and 1 <= recipe["batch_size"] <= 6,
                    "Synthetic fit must be at most three epochs and batch size six")
            expected.update(epochs=recipe["epochs"], batch_size=recipe["batch_size"])
        require(recipe == expected, "Recipe differs from fixed AdamW budget")
        require(spec["phase"] in ("tuning", "final"), "Unknown training phase")
        seed = spec["seed"]
        require(type(seed) is int and (seed == 100 if spec["phase"] == "tuning" else seed in (0, 1, 2)),
                "Tuning seed 100 is separate from final seeds 0..2")
        require(spec["model"]["backbone_seed"] == seed
                and spec["model"]["head_seed"] == 20000 + seed
                and spec["loader_seed"] == 10000 + seed, "Independent seed schedule differs")
    elif spec["kind"] == "dino_features":
        require(type(spec["model"]["encoder_seed"]) is int and spec["model"]["encoder_seed"] in range(42, 47),
                "DINO encoder seed must be 42..46")
        require(spec["recipe"] == {"batch_size": spec["recipe"]["batch_size"]}
                and type(spec["recipe"]["batch_size"]) is int and spec["recipe"]["batch_size"] > 0,
                "Feature recipe is only a positive batch size")


def _completed(binding, kind):
    path = check_file(binding)
    result = json.loads(path.read_text())
    require(result["status"] == "complete" and result["kind"] == kind, "Incomplete or wrong producer")
    require(artifact_hashes(path.parent, exclude=[path.name]) == result["outputs"], "Producer inventory differs")
    for name, expected in result["outputs"].items():
        member = (path.parent / name).resolve(strict=True)
        require(member.is_relative_to(path.parent.resolve()) and sha256_file(member) == expected,
                f"Producer output differs: {name}")
    return path.parent, result


def _feature_inputs(spec):
    root, result = _completed(spec["features"], "dino_features")
    require(result["fold"] == spec["fold"] and result["purpose"] == spec["purpose"]
            and result["identity"] == spec["feature_identity"], "Feature producer identity differs")
    if spec["purpose"] == "titan_followup":
        support.require_followup_inputs(result["inputs"], spec["fold"])
        require(result["inputs"]["catalog_count"] == 23380 and result["inputs"]["class_order"] == support.CLASS_ORDER,
                "Feature catalog class/count contract differs")
    arrays = {}
    for role in support.ROLES:
        expected = result["caches"][role]
        require(expected["inputs"] == result["inputs"] and expected["identity"] == result["identity"]
                and expected["role"] == role and expected["fold"] == spec["fold"]
                and expected["purpose"] == spec["purpose"]
                and expected["tile_ids"] == result["inputs"]["ordered"][role]["tile_ids"]
                and expected["labels"] == result["inputs"]["ordered"][role]["labels"], "Feature role binding differs")
        arrays[role] = support.load_followup_cache(root / "features" / f"titan_{role}_feats.npy", expected)
    return result, arrays


def _selected_lr(spec, inputs):
    candidates = spec.get("tuning_results", [])
    if spec["purpose"] == "synthetic_cpu_test" and not candidates:
        return {"synthetic_only": True, "lr": spec["recipe"]["lr"]}
    require(len(candidates) == 3, "All three complete tuning candidates are required")
    rows = []
    for candidate in candidates:
        _, result = _completed(candidate, "resnet18")
        prior = result["specification"]
        require(prior["phase"] == "tuning" and prior["seed"] == 100 and prior["fold"] == spec["fold"]
                and prior["purpose"] == spec["purpose"] and prior["source"] == spec["source"]
                and prior["runtime"] == spec["runtime"] and result["inputs"] == inputs,
                "Tuning candidate source/task differs")
        require(prior["model"]["initialization"] == spec["model"]["initialization"], "Candidate initialization differs")
        require(prior.get("execution", {}).get("cuda_dependencies")
                == spec.get("execution", {}).get("cuda_dependencies"), "Candidate CUDA dependency receipt differs")
        # All model settings other than the declared backbone/head seeds must match.
        strip = lambda model: {k: v for k, v in model.items() if k not in ("backbone_seed", "head_seed")}
        require(strip(prior["model"]) == strip(spec["model"]), "Candidate model contract differs")
        require({k: v for k, v in prior["recipe"].items() if k != "lr"}
                == {k: v for k, v in spec["recipe"].items() if k != "lr"}, "Candidate recipe differs")
        require(len(result["history"]) == prior["recipe"]["epochs"], "Incomplete tuning budget")
        required = {"selected.pt", "final.pt", "initial_state.pt", "started.json", "preparation.json"}
        required.update(f"epoch-{epoch:03d}.json" for epoch in range(1, prior["recipe"]["epochs"] + 1))
        require(required <= set(result["outputs"]), "Tuning artifacts are incomplete")
        epoch = support.select_epoch(result["history"])
        require(epoch == result["selected_epoch"], "Candidate checkpoint selection differs")
        rows.append({"lr": prior["recipe"]["lr"], "score": result["history"][epoch - 1]["val_macro_f1"],
                     "result": candidate})
    require(sorted(r["lr"] for r in rows) == support.CANDIDATE_LRS, "Missing/duplicate LR candidate")
    chosen = max(rows, key=lambda r: (r["score"], -r["lr"]))
    require(chosen["lr"] == spec["recipe"]["lr"], "Final LR does not match fold-local validation selection")
    return {"lr": chosen["lr"], "candidates": rows, "rule": "maximum validation macro_f1, then lower LR"}


def prepare(spec):
    """Check bindings and report missing assets without constructing a model."""
    _validate_spec(spec)
    support.checked_output(spec)
    missing = []
    source = git_state(support.ROOT)
    if spec["purpose"] == "titan_followup":
        require(source["clean"], "Scientific execution requires clean source")
    current_files = {name: sha256_file(support.ROOT / name) for name in support.SOURCE_FILES}
    support.verify_source_imports(current_files)
    if not spec.get("source"):
        missing.append("source commit/file lock")
    else:
        require(spec["source"] == {"commit": source["commit"], "files": current_files}, "Source lock differs")
    runtime = support.runtime_identity()
    if not spec.get("runtime"):
        missing.append("runtime lock")
    else:
        require(runtime == spec["runtime"], "Imported runtime implementation differs")
    cuda_dependencies = None
    if spec["device"].startswith("cuda:"):
        try:
            receipt = spec["execution"]["cuda_dependencies"]
            path = check_file(receipt)
            cuda_dependencies = {"path": str(path), "sha256": receipt["sha256"]}
        except FileNotFoundError as error:
            missing.append(str(error))
    inputs = None
    try:
        if spec["kind"] == "dino_c1":
            feature_result, _ = _feature_inputs(spec)
            inputs = feature_result["inputs"]
        else:
            inputs = support.validate_inputs(spec)
            if not spec["inputs"].get("normalization_sha256"):
                missing.append("training normalization SHA256")
    except FileNotFoundError as error:
        missing.append(str(error))
    model = spec.get("model", {})
    if spec["kind"] != "dino_c1":
        try:
            if spec["kind"] == "dino_features":
                verify_dino_source(model)
                require(os.environ.get("XFORMERS_DISABLED") == "1", "Set XFORMERS_DISABLED=1")
                require(model["initialization"] in ("random", "pretrained"), "Unknown DINO initialization")
                if not model.get("checkpoint"):
                    missing.append("dinov2_vitb14_pretrain.pth" if model["initialization"] == "pretrained"
                                   else f"official DINO initial state for seed {model['encoder_seed']}")
                if not model.get("initial_state_sha256"):
                    missing.append("DINO initial tensor-state SHA256")
            else:
                verify_resnet_source(model)
                require(model["initialization"] in ("scratch", "pretrained"), "Unknown ResNet initialization")
                if model["initialization"] == "pretrained":
                    require(model.get("weights") == RESNET_WEIGHTS, "Explicit ImageNet weight enum required")
                    if not model.get("checkpoint"):
                        missing.append("resnet18-f37072fd.pth and its full SHA256")
            if model.get("checkpoint"):
                check_file(model["checkpoint"])
                if spec["kind"] == "dino_features" and model["initialization"] == "pretrained":
                    require(model["checkpoint"]["sha256"] == DINOV2_SHA256, "DINO checkpoint release differs")
                if spec["kind"] == "resnet18" and model["initialization"] == "pretrained":
                    require(model["checkpoint"]["sha256"] == RESNET18_SHA256, "ResNet checkpoint release differs")
        except FileNotFoundError as error:
            missing.append(str(error))
    selection = comparison = None
    if spec["kind"] == "resnet18" and spec["phase"] == "final" and inputs is not None:
        if "loss_comparison" in spec:
            comparison = loss_comparison.prepare(spec, inputs, _completed)
            selection = comparison["selection"]
        else:
            selection = _selected_lr(spec, inputs)
    return {"status": "prepared_no_model_execution", "ready": not missing,
            "missing_prerequisites": missing, "specification_sha256": support.digest(spec),
            "source": source, "source_files": current_files, "runtime": runtime,
            "inputs": inputs, "selection": selection, "cuda_dependencies": cuda_dependencies,
            "model_checks_executed": False, "scientific_accuracy_measured": False,
            **({"loss_comparison": comparison} if comparison is not None else {})}


def _attempt(spec, work):
    output = support.checked_output(spec)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    gpu = None
    support.publish_json(output / "started.json", {"status": "started", "specification": spec,
                        "specification_sha256": support.digest(spec), "command": sys.argv,
                        "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        prepared = prepare(spec)
        support.publish_json(output / "preparation.json", prepared)
        require(prepared["ready"], "Missing prerequisites: " + "; ".join(prepared["missing_prerequisites"]))
        if spec["device"].startswith("cuda:"):
            # Execution only: CPU preparation/tests never query a GPU.
            require(not torch.cuda.is_initialized(), "Each CUDA job requires a fresh process")
            require(torch.cuda.device_count() == 1, "CUDA execution requires exactly one visible GPU")
            torch.cuda.set_device(spec["device"])
            properties = torch.cuda.get_device_properties(spec["device"])
            require(getattr(properties, "uuid", None) is not None, "GPU UUID unavailable")
            actual_uuid = _gpu_uuid(properties.uuid)
            require(actual_uuid == spec["execution"]["gpu_uuid"], "Actual GPU UUID differs from declared lane")
            gpu = {"device": spec["device"], "name": properties.name, "uuid": actual_uuid,
                   "cuda_dependencies": prepared["cuda_dependencies"],
                   "total_memory_bytes": properties.total_memory, "torch_cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version()}
            torch.cuda.reset_peak_memory_stats(spec["device"])
            support.publish_json(output / "gpu-started.json", gpu)
        result = work(spec, prepared, output)
        if gpu is not None:
            gpu = _gpu_finished(gpu)
            result["gpu_execution"] = gpu
        # Inputs, source and runtime are rechecked before publishing completion.
        after = prepare(spec)
        require(all(after[k] == prepared[k] for k in
                    ("ready", "missing_prerequisites", "source_files", "runtime", "inputs", "selection", "cuda_dependencies")),
                "Bindings changed during execution")
        if "loss_comparison" in spec:
            require(after["loss_comparison"] == prepared["loss_comparison"],
                    "Loss comparison binding changed during execution")
        result.update(status="complete", kind=spec["kind"], purpose=spec["purpose"], fold=spec["fold"],
                      specification=spec, specification_sha256=support.digest(spec), inputs=prepared["inputs"],
                      elapsed_seconds=time.monotonic() - start, outputs=artifact_hashes(output))
        support.publish_json(output / "result.json", result)
        return result
    except BaseException as error:
        if gpu is not None:
            try:
                gpu = _gpu_finished(gpu)
            except Exception as recording_error:
                error.add_note(f"GPU failure metrics unavailable: {recording_error!r}")
        try:
            support.publish_json(output / "failed.json", {"status": "failed", "error_type": type(error).__name__,
                                "error": str(error), "specification_sha256": support.digest(spec),
                                "elapsed_seconds": time.monotonic() - start, "gpu_execution": gpu})
        except Exception as recording_error:
            error.add_note(f"Failed to publish failure receipt: {recording_error!r}")
        raise


def _gpu_finished(record):
    device = record["device"]
    torch.cuda.synchronize(device)
    return {**record, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _extract(spec, prepared, output):
    wrapper, identity = load_matched_dino(spec["model"], spec["device"])
    support.publish_artifact(output / "initial_state.pt", lambda f: torch.save(_cpu_copy(wrapper.model.state_dict()), f))
    directory = output / "features"
    directory.mkdir()
    caches = {}
    loaders = support.make_loaders(spec, shuffle_train=False, seed=0)
    for role, loader in loaders.items():
        features, labels, ids = extract_features(wrapper, loader, spec["device"])
        stem = f"titan_{role}"
        for suffix, array in (("feats", features), ("labels", labels), ("ids", np.asarray(ids, dtype=str))):
            support.publish_artifact(directory / f"{stem}_{suffix}.npy", lambda f, a=array: np.save(f, a, allow_pickle=False))
        expected = {"identity": identity, "inputs": prepared["inputs"], "fold": spec["fold"], "role": role,
                    "transform": TRANSFORM, "purpose": spec["purpose"], **prepared["inputs"]["ordered"][role]}
        provenance = {**expected, **{key: sha256_file(directory / f"{stem}_{suffix}.npy") for suffix, key in
                      (("feats", "feature_sha256"), ("labels", "labels_sha256"), ("ids", "tile_ids_sha256"))}}
        support.publish_json(directory / f"{stem}_feats.meta.json", {
            "schema_version": "model-followup-v1", "model": identity["model"], "array": f"{stem}_feats",
            "weights_source": identity["initialization"], "n_features": len(ids), "feature_dim": 768,
            "provenance": provenance})
        support.load_followup_cache(directory / f"{stem}_feats.npy", expected)
        caches[role] = expected
    require(state_digest(wrapper.model.state_dict()) == identity["initial_state_sha256"], "Frozen encoder state changed")
    return {"identity": identity, "caches": caches}


def extract_dino(spec):
    require(spec["kind"] == "dino_features", "Expected DINO extraction job")
    return _attempt(spec, _extract)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_copy(v) for v in value)
    return value


def _snapshot(model, optimizer, scheduler, loader, epoch, validation, identity, spec):
    return {"model_state_dict": _cpu_copy(model.state_dict()), "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
            "scheduler_state_dict": _cpu_copy(scheduler.state_dict()), "epoch": epoch,
            "validation": validation, "identity": identity, "specification_sha256": support.digest(spec),
            "model_state_sha256": state_digest(model.state_dict()), "torch_rng": torch.get_rng_state().clone(),
            "loader_rng": loader.generator.get_state().clone(), "python_rng": random.getstate(),
            "numpy_rng": [v.tolist() if isinstance(v, np.ndarray) else v for v in np.random.get_state()],
            "cuda_rng": torch.cuda.get_rng_state(spec["device"]).cpu() if spec["device"].startswith("cuda:") else None}


def _train(spec, prepared, output):
    model, identity = build_resnet18(spec["model"], spec["device"])
    normalizer = None
    if "loss_comparison" in spec:
        comparison = prepared["loss_comparison"]
        loss_comparison.verify_initial_state(model, identity, comparison)
        normalizer = comparison["normalizer"]
    # Reset run/head/order seeds after constructor/loading has consumed its own RNG.
    torch.manual_seed(spec["seed"])
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    loaders = support.make_loaders(spec, roles=("train", "val"), shuffle_train=True, seed=spec["loader_seed"])
    weights = loaders["train"].dataset.get_class_weights()
    if normalizer is not None:
        require(weights.dtype == torch.float32 and weights.tolist() == normalizer["native_weights"],
                "Training weights differ from the loss comparison normalizer")
    recipe = spec["recipe"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], betas=tuple(recipe["betas"]),
                                 eps=recipe["eps"], weight_decay=recipe["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe["epochs"])
    support.publish_artifact(output / "initial_state.pt", lambda f: torch.save(_cpu_copy(model.state_dict()), f))
    history, best = [], None
    for epoch in range(1, recipe["epochs"] + 1):
        rate = optimizer.param_groups[0]["lr"]
        if normalizer is None:
            training = train_epoch(model, loaders["train"], optimizer, class_weights=weights, device=spec["device"])
        else:
            training = train_epoch(model, loaders["train"], optimizer, class_weights=weights, device=spec["device"],
                                   population_mean_weight=normalizer["m32"], record_batches=True)
            support.publish_json(output / f"batches-{epoch:03d}.json",
                                 {"epoch": epoch, "batch_records": training.pop("batch_records")})
        validation = evaluate_images(model, loaders["val"], class_weights=weights, device=spec["device"])
        scheduler.step()
        history.append({"epoch": epoch, "lr": rate, **training, "val_macro_f1": validation["macro_f1"],
                        "val_loss": validation["loss"], "val_loss_numerator": validation["loss_numerator"],
                        "val_loss_denominator": validation["loss_denominator"],
                        "val_confusion_matrix": validation["confusion_matrix"]})
        support.publish_json(output / f"epoch-{epoch:03d}.json", {"history": history[-1], "validation": validation})
        if support.select_epoch(history) == epoch:
            best = _snapshot(model, optimizer, scheduler, loaders["train"], epoch, validation, identity, spec)
    final = _snapshot(model, optimizer, scheduler, loaders["train"], epoch, validation, identity, spec)
    for name, state in (("selected", best), ("final", final)):
        support.publish_artifact(output / f"{name}.pt", lambda f, s=state: torch.save(s, f))
    selected = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
    require(selected["specification_sha256"] == support.digest(spec)
            and state_digest(selected["model_state_dict"]) == selected["model_state_sha256"], "Selected checkpoint binding differs")
    model.load_state_dict(selected["model_state_dict"], strict=True)
    replay = evaluate_images(model, loaders["val"], class_weights=weights, device=spec["device"])
    require(replay == selected["validation"], "Selected validation predictions do not replay")
    result = {"identity": identity, "history": history, "selected_epoch": selected["epoch"],
              "selected_validation": replay, "selection": prepared["selection"],
              "selected_state_sha256": selected["model_state_sha256"],
              "final_state_sha256": final["model_state_sha256"],
              "training_class_weights": weights.tolist(), "budget_disposition": "complete finite training recipe; no convergence claim"}
    if normalizer is not None:
        result["normalizer"] = normalizer
    if spec["phase"] == "final":
        test = support.make_loaders(spec, roles=("test",), seed=spec["loader_seed"])["test"]
        result["test"] = evaluate_images(model, test, class_weights=weights, device=spec["device"])
    return result


def train_resnet(spec):
    require(spec["kind"] == "resnet18", "Expected ResNet training job")
    return _attempt(spec, _train)


def _readout(spec, prepared, output):
    feature_result, arrays = _feature_inputs(spec)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline = fit_classical_probe(arrays["train"]["feats"], arrays["train"]["labels"], seed=0)
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    parameters = classifier.get_params()
    require(parameters["C"] == 1.0 and parameters["solver"] == "lbfgs" and parameters["max_iter"] == 2000
            and parameters["tol"] == 1e-4 and parameters["class_weight"] == "balanced"
            and parameters.get("penalty") in ("l2", "deprecated")
            and parameters.get("l1_ratio") in (None, 0, 0.0), "Existing common C=1 recipe differs")
    require(classifier.classes_.tolist() == list(range(6)), "Classifier class order differs")
    state = {"coef": classifier.coef_, "intercept": classifier.intercept_, "classes": classifier.classes_,
             "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "scaler_var": scaler.var_,
             "scaler_samples": np.asarray(scaler.n_samples_seen_), "n_iter": classifier.n_iter_}
    support.publish_artifact(output / "estimator.npz", lambda f: np.savez_compressed(f, **state))
    results = {}
    for role in ("val", "test"):
        data = arrays[role]
        logits = pipeline.decision_function(data["feats"])
        probabilities = pipeline.predict_proba(data["feats"])
        predicted = pipeline.predict(data["feats"])
        require(np.isfinite(logits).all() and np.isfinite(probabilities).all()
                and np.array_equal(predicted, logits.argmax(axis=1)), "Native LR outputs differ")
        scores = evaluate_predictions(data["labels"], predicted, 6, probabilities)
        matrix = np.asarray(scores["confusion_matrix"])
        require(((probabilities >= 0) & (probabilities <= 1)).all()
                and np.allclose(probabilities.sum(1), 1, atol=1e-12 if probabilities.dtype == np.float64 else 1e-6, rtol=0),
                "Invalid native LR probabilities")
        scores.update(tile_ids=data["ids"].tolist(), labels=data["labels"].tolist(), logits=logits.tolist(), class_order=list(range(6)),
                      logits_dtype=str(logits.dtype), probabilities_dtype=str(probabilities.dtype),
                      probability_argmax_disagreements=int(np.sum(predicted != probabilities.argmax(1))),
                      true_positive=matrix.diagonal().tolist(), false_positive=(matrix.sum(0) - matrix.diagonal()).tolist(),
                      false_negative=(matrix.sum(1) - matrix.diagonal()).tolist())
        results[role] = scores
    return {"identity": feature_result["identity"], "readout": "training-only StandardScaler; fixed balanced C=1 LR",
            "classifier_parameters": parameters, "scaler_parameters": scaler.get_params(),
            "feature_dtype": str(arrays["train"]["feats"].dtype), "coefficient_dtype": str(classifier.coef_.dtype),
            "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
            "n_iter": classifier.n_iter_.tolist(), "convergence_claim": False, "predictions": results}


def readout_c1(spec):
    require(spec["kind"] == "dino_c1", "Expected C=1 readout job")
    return _attempt(spec, _readout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    support.configure_runtime()
    envelope = json.loads(args.spec.read_text())
    spec = envelope["specification"]
    require(support.digest(spec) == envelope["specification_sha256"], "Specification seal differs")
    result = ({"dino_features": extract_dino, "dino_c1": readout_c1, "resnet18": train_resnet}[spec["kind"]](spec)
              if args.execute else prepare(spec))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
