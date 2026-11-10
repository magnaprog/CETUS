"""Seeded partial fine-tuning for the TitanSAR image-domain benchmark."""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from titansar.configs.defaults import TitanSARConfig
from titansar.data.dataset import TitanSARDataset
from titansar.models.foundation_models import get_model
from titansar.reproducibility import complete_run_manifest, sha256_file, start_run_manifest
from titansar.models.probing import (
    LinearProbe,
    _cohens_kappa,
    evaluate_linear_probe,
    extract_features,
    load_bound_feature_cache,
    train_finetuned_model,
    train_linear_probe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPAIRED_RECIPE = {
    "unfreeze_blocks": 2, "epochs": 50, "microbatch_size": 8,
    "grad_accum_steps": 4, "optimizer": "AdamW", "backbone_lr": 1e-4,
    "head_lr": 1e-3, "weight_decay": 0.01, "scheduler": "CosineAnnealingLR",
    "T_max": 50, "amp": True, "deterministic": True,
    "class_weights": "inverse training frequency, normalized to mean one",
    "selection": "earliest highest validation macro recall among epochs with optimizer updates",
    "head_initialization": "fresh LinearProbe, torch seed reset after encoder load",
    "comparison": "adaptation recipe comparison; no causal isolation of unfreezing",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _check_prediction(result, dataset):
    # Pure NumPy verification helpers; no auditor main function or GPU is invoked.
    from scripts.summarize_review_frozen import checked_predictions

    ids = [e["tile_id"] for e in dataset.entries]
    labels = np.array([e["label"] for e in dataset.entries])
    truth = np.asarray(result.get("true_labels", result.get("labels")))
    _require(result["tile_ids"] == ids and truth.dtype.kind in "iu"
             and np.array_equal(truth, labels), "Prediction IDs or labels differ from the test split")
    metrics = checked_predictions(result, labels, list(range(6)))
    _require(np.isclose(result["kappa"], _cohens_kappa(labels, np.asarray(result["predictions"]), 6),
                        atol=1e-12, rtol=0), "Prediction kappa differs")
    return metrics


def prepare_repaired_run(args):
    """Validate the sealed reference and raw normalization before any model load."""
    from scripts.summarize_review_frozen import ordered_seeds, read_json, verify_manifest

    _require(len(args.models) == len(args.seeds) == 1 and args.models[0] in ("dinov2", "dofa", "croma")
             and args.seeds[0] in range(5), "Repaired jobs require one pretrained model and one seed 0..4")
    _require(args.unfreeze_blocks == [2] and args.epochs == 50 and args.batch_size == 8
             and args.grad_accum_steps == 4 and args.lr == 1e-4 and args.num_workers >= 0
             and not args.allow_nondeterministic and not args.disable_amp and args.device.startswith("cuda"),
             "Repaired job arguments differ from the sealed recipe")
    protocol = read_json(args.repaired_protocol)
    spec = protocol["specification"]
    _require(_json_digest(spec) == protocol["specification_sha256"], "Fine-tuning protocol seal differs")
    _require(spec["recipe"] == REPAIRED_RECIPE, "Protocol recipe differs from the supported implementation")
    repository = Path(__file__).resolve().parents[1]
    for name, expected in spec["scientific_source_sha256"].items():
        path = (repository / name).resolve()
        _require(path.is_relative_to(repository) and sha256_file(path) == expected,
                 f"Scientific source changed after protocol sealing: {name}")
    _require(sha256_file(Path(args.titan_catalog)) == spec["catalog_sha256"], "Catalog differs from repaired protocol")
    split_hash = sha256_file(Path(args.titan_split_manifest))
    folds = [f for f in spec["folds"] if f["split_sha256"] == split_hash]
    _require(len(folds) == 1, "Split is absent or ambiguous in repaired protocol")
    fold = folds[0]
    model_name, seed = args.models[0], args.seeds[0]
    jobs = [j for j in spec["frozen_jobs"] if j["model"] == model_name and j["fold"] == fold["fold"]]
    _require(len(jobs) == 1, "Frozen reference job is absent or ambiguous")
    job = jobs[0]
    reference_path, features_dir = Path(args.frozen_reference), Path(args.frozen_features_dir)
    analysis_root, feature_root = reference_path.parent, features_dir.parent
    _require(reference_path.name == "probing_results.json" and features_dir.name == "features",
             "Expected probing_results.json and its producer's features directory")
    _require(sha256_file(reference_path) == job["result_sha256"], "Frozen reference result hash differs")
    manifests = {}
    for kind, root in (("analysis", analysis_root), ("extraction", feature_root)):
        _require(sha256_file(root / "run_manifest.json") == job[f"{kind}_manifest_sha256"],
                 f"Frozen {kind} manifest hash differs")
        manifests[kind] = verify_manifest(root, spec["frozen_source_commit"])
        for key, expected in (("titan_catalog", spec["catalog_sha256"]), ("titan_split_manifest", split_hash)):
            _require(manifests[kind]["inputs"][key]["sha256"] == expected, "Frozen input binding differs")
    producer_files = {k.removeprefix("features/"): v for k, v in manifests["extraction"]["outputs"].items()
                      if k.startswith("features/")}
    _require(manifests["analysis"]["inputs"]["cached_features"]["files"] == producer_files,
             "Frozen analysis did not consume the bound producer arrays")

    loaders = make_loaders(args.titan_catalog, args.titan_split_manifest, TitanSARConfig(),
                           args.batch_size, args.num_workers, seed)
    train, train_loader, val_loader, test_loader = loaders
    normalization = train.normalization_provenance
    sampling = normalization.get("sampling", {})
    _require(normalization.get("enabled") is True and sampling.get("sample_count", 0) > 0
             and sampling.get("missing_tile_ids") == [] and sampling.get("empty_tile_ids") == [],
             "Repaired normalization requires complete nonempty training samples")
    _require(_json_digest(normalization) == fold["normalization_sha256"], "Normalization differs from verified reference")
    counts = {}
    for role, dataset in (("train", train), ("val", val_loader.dataset), ("test", test_loader.dataset)):
        ids = [e["tile_id"] for e in dataset.entries]
        labels = np.array([e["label"] for e in dataset.entries])
        _require(len(ids) > 0 and len(ids) == len(set(ids)) and np.unique(labels).tolist() == list(range(6)),
                 f"Incomplete six-class {role} membership")
        counts[role] = {"total": len(ids), "class_counts": np.bincount(labels, minlength=6).tolist()}
        _require(counts[role] == fold["counts"][role], f"{role} class counts differ from protocol")
        _require(dataset.benchmark_track == spec["benchmark_track"]
                 and dataset.normalization_provenance == normalization, "Partition track or normalization differs")
    entries = {e["tile_id"]: e for e in train.entries}
    for tile_id in sampling["used_tile_ids"]:
        entry = entries[tile_id]
        _require(sha256_file(train.tiles_dir / f"{tile_id}.npy") == entry.get("tile_sha256", entry.get("sha256")),
                 f"Normalization tile hash differs: {tile_id}")
    identity = spec["encoders"][model_name]
    source = {"titan_catalog_sha256": spec["catalog_sha256"], "earth_catalog_sha256": None,
              "titan_split_manifest_sha256": split_hash, "earth_split_manifest_sha256": None,
              "evaluation_scope": "within_titan_only", "benchmark_track": train.benchmark_track,
              "normalization_provenance": normalization, **identity}
    for stem, dataset in (("titan_train", train), ("titan_test", test_loader.dataset)):
        expected = {**source, **({"band_identifier_ghz": 13.78} if model_name == "dofa" else {})}
        features, labels, ids, _ = load_bound_feature_cache(
            features_dir / model_name, f"{stem}_feats", model_name, expected,
            {e["tile_id"]: e["label"] for e in dataset.entries})
        _require(ids == [e["tile_id"] for e in dataset.entries]
                 and features.dtype == np.float32 and features.shape == (len(dataset), 768),
                 "Frozen cache order, dtype or feature dimension differs")
    payload = read_json(reference_path)
    _require(set(payload) == {model_name}, "Frozen reference model inventory differs")
    reference = payload[model_name]
    _require(reference["weights_source"] == "pretrained" and reference["source_metadata"] == source
             and reference["evaluation_scope"] == "within_titan_only"
             and set(reference["linear_probe"]) == {"titan_to_titan"}, "Frozen reference source metadata differs")
    heads = ordered_seeds(reference["linear_probe"]["titan_to_titan"]["seed_results"])
    for head in heads:
        _check_prediction(head, test_loader.dataset)
    selected = {**heads[seed], "labels": heads[seed]["true_labels"]}
    provenance = {**source, "fold": fold["fold"], "seed": seed, "recipe": REPAIRED_RECIPE,
                  "class_counts": counts, "protocol_sha256": sha256_file(Path(args.repaired_protocol)),
                  "protocol_specification_sha256": protocol["specification_sha256"],
                  "frozen_reference": {**job, "source_commit": spec["frozen_source_commit"],
                                       "matched_head_seed": seed, "validated_head_seeds": list(range(5))},
                  "dofa_band_identifier_ghz": 13.78 if model_name == "dofa" else None}
    return {"loaders": loaders, "frozen_head": selected, "provenance": provenance}


def repaired_training_metadata(model, seed):
    """Expose the exact block parameters and independently seeded initial head."""
    blocks = model.get_transformer_blocks()
    _require(len(blocks) >= 2, "Encoder has fewer than two transformer blocks")
    for parameter in model.model.parameters():
        parameter.requires_grad = False
    for block in blocks[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    names = {id(module): name for name, module in model.model.named_modules()}
    parameters = {name: p for name, p in model.model.named_parameters() if p.requires_grad}
    _require(bool(parameters), "Two-block selection contains no trainable parameters")
    seed_everything(seed)
    head = LinearProbe(model.embedding_dim, 6)
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.numpy().tobytes())
    return head, {"head_initialization_seed": seed, "initial_head_state_sha256": digest.hexdigest(),
                  "state_digest_format": "sorted name:dtype:shape newline followed by contiguous CPU tensor bytes",
                  "unfrozen_blocks": [names[id(block)] for block in blocks[-2:]],
                  "unfrozen_parameter_names": list(parameters),
                  "trainable_backbone_parameters": sum(p.numel() for p in parameters.values()),
                  "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
                  "total_backbone_parameters": sum(p.numel() for p in model.model.parameters())}


def repaired_history_metadata(history, train_count, best_val):
    expected = (train_count + 31) // 32
    for name in ("train_loss", "val_balanced_accuracy", "optimizer_steps", "amp_skipped_steps", "loss_scale"):
        values = np.asarray(history[name])
        _require(values.shape == (50,) and np.isfinite(values).all(), f"Invalid fine-tuning history: {name}")
    steps, skipped = np.asarray(history["optimizer_steps"]), np.asarray(history["amp_skipped_steps"])
    _require(steps.dtype.kind in "iu" and skipped.dtype.kind in "iu" and (steps > 0).all()
             and (skipped >= 0).all() and (steps + skipped == expected).all()
             and (np.asarray(history["loss_scale"]) > 0).all(), "Fine-tuning update budget or AMP history differs")
    validation = np.asarray(history["val_balanced_accuracy"])
    _require(((validation >= 0) & (validation <= 1)).all()
             and np.isclose(best_val, validation.max(), atol=1e-12, rtol=0), "Selected validation score differs")
    return {"selected_epoch": int(validation.argmax()) + 1, "epoch_indexing": "one based",
            "optimizer_attempts_per_epoch": expected, "optimizer_steps_total": int(steps.sum()),
            "amp_skipped_steps_total": int(skipped.sum()), "training_example_presentations": 50 * train_count}


def seed_everything(seed: int, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loaders(catalog, split_manifest, config, batch_size, num_workers, seed):
    train = TitanSARDataset(catalog, split="train", config=config, split_manifest_path=split_manifest)
    val = TitanSARDataset(catalog, split="val", config=config, split_manifest_path=split_manifest)
    test = TitanSARDataset(catalog, split="test", config=config, split_manifest_path=split_manifest)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    return (
        train,
        DataLoader(train, shuffle=True, drop_last=False, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def evaluate_finetuned(model, probe, test_loader, num_classes, device):
    features, labels, tile_ids = extract_features(model, test_loader, device)
    result = evaluate_linear_probe(probe, features, labels, num_classes, device)
    result["tile_ids"] = tile_ids
    result["labels"] = labels
    return result


def compact_result(result, seed, best_val_accuracy=None, history=None, elapsed=None):
    output = {
        "seed": seed,
        "macro_accuracy": result["macro_accuracy"],
        "kappa": result["kappa"],
        "per_class_accuracy": result["per_class_accuracy"],
        "confusion_matrix": result["confusion_matrix"],
        "confusion_matrix_normalized": result["confusion_matrix_normalized"],
        "predictions": result["predictions"],
        "probabilities": result["probabilities"],
    }
    if "tile_ids" in result:
        output["tile_ids"] = result["tile_ids"]
        output["labels"] = result["labels"]
    if best_val_accuracy is not None:
        output["best_val_accuracy"] = best_val_accuracy
    if history is not None:
        output["history"] = history
    if elapsed is not None:
        output["time_seconds"] = elapsed
    return output


def summarize_runs(runs):
    accuracies = np.array([run["macro_accuracy"] for run in runs], dtype=float)
    kappas = np.array([run["kappa"] for run in runs], dtype=float)
    return {
        "macro_accuracy_mean": float(accuracies.mean()),
        "macro_accuracy_std": float(accuracies.std(ddof=1)) if len(runs) > 1 else 0.0,
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std(ddof=1)) if len(runs) > 1 else 0.0,
        "runs": runs,
    }


def save_checkpoint(path, model, probe, seed, unfreeze_blocks, best_val_accuracy, provenance=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "probe_state_dict": probe.state_dict(),
            "seed": seed,
            "unfreeze_blocks": unfreeze_blocks,
            "best_val_accuracy": best_val_accuracy,
            **({"provenance": provenance} if provenance is not None else {}),
        },
        path,
    )


def run_model(args, model_name, repaired=None):
    config = TitanSARConfig()
    if getattr(args, "repaired_protocol", None) and repaired is None:
        repaired = prepare_repaired_run(args)
    strict = args.require_real_weights or os.environ.get(
        "TITANSAR_REQUIRE_REAL_WEIGHTS", ""
    ).lower() in {"1", "true", "yes"} or repaired is not None
    results = {
        "model": model_name,
        "seeds": args.seeds,
        "deterministic": not args.allow_nondeterministic,
        "configurations": {},
    }

    if repaired is not None:
        results.update(weights_source="pretrained", evaluation_scope="within_titan_only",
                       provenance=repaired["provenance"])
        results["configurations"]["frozen"] = summarize_runs(
            [compact_result(repaired["frozen_head"], args.seeds[0])])
    else:
        seed_everything(args.seeds[0], not args.allow_nondeterministic)
        train_dataset, train_loader, _, test_loader = make_loaders(
            args.titan_catalog, args.titan_split_manifest, config,
            args.batch_size, args.num_workers, args.seeds[0],
        )
        model = get_model(model_name, device=args.device, allow_placeholder=not strict)
        results["weights_source"] = model.weights_source
        train_features, train_labels, _ = extract_features(model, train_loader, args.device)
        test_features, test_labels, test_ids = extract_features(model, test_loader, args.device)
        frozen_runs = []
        for seed in args.seeds:
            seed_everything(seed, not args.allow_nondeterministic)
            probe = train_linear_probe(
                train_features, train_labels, num_classes=config.num_classes,
                class_weights=train_dataset.get_class_weights(), config=config,
                seed=seed, device=args.device,
            )
            result = evaluate_linear_probe(
                probe, test_features, test_labels, config.num_classes, args.device
            )
            result["tile_ids"] = test_ids
            result["labels"] = test_labels
            frozen_runs.append(compact_result(result, seed))
        results["configurations"]["frozen"] = summarize_runs(frozen_runs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for depth in args.unfreeze_blocks:
        runs = []
        for seed in args.seeds:
            seed_everything(seed, not args.allow_nondeterministic)
            if repaired is None:
                loaders = make_loaders(args.titan_catalog, args.titan_split_manifest, config,
                                       args.batch_size, args.num_workers, seed)
            else:
                loaders = repaired["loaders"]
            train_dataset, train_loader, val_loader, test_loader = loaders
            model = get_model(model_name, device=args.device, allow_placeholder=not strict)
            extra, provenance = {}, None
            if repaired is not None:
                binding = repaired["provenance"]
                _require(model.weights_source == "pretrained"
                         and model.model_revision == binding["model_revision"]
                         and model.weights_sha256 == binding["weights_sha256"], "Loaded encoder identity differs")
                if model_name == "dofa":
                    _require(model.wavelength == 13.78, "Loaded DOFA identifier differs")
                head, trainable = repaired_training_metadata(model, seed)
                extra["init_probe"] = head
                provenance = {**binding, **trainable,
                              "training_class_weights": train_dataset.get_class_weights().tolist()}
            started = time.time()
            ft_model, ft_probe, best_val, history = train_finetuned_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                num_classes=config.num_classes, unfreeze_blocks=depth,
                lr=args.lr, epochs=args.epochs,
                class_weights=train_dataset.get_class_weights(), device=args.device,
                use_amp=not args.disable_amp and args.device.startswith("cuda"),
                grad_accum_steps=args.grad_accum_steps,
                return_history=True,
                **extra,
            )
            result = evaluate_finetuned(
                ft_model, ft_probe, test_loader, config.num_classes, args.device
            )
            run = compact_result(result, seed, best_val, history, time.time() - started)
            if repaired is not None:
                run["metrics"] = _check_prediction(result, test_loader.dataset)
                provenance.update(repaired_history_metadata(history, len(train_dataset), best_val))
                run["provenance"] = provenance
                run["true_labels"] = run["labels"]
            runs.append(run)
            save_checkpoint(
                Path(args.output_dir) / "checkpoints" / model_name / f"ft_{depth}block_seed_{seed}.pt",
                ft_model, ft_probe, seed, depth, best_val, provenance=provenance,
            )
            del model, ft_model, ft_probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["configurations"][f"ft_{depth}block"] = summarize_runs(runs)
    return results


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(key): sanitize_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--titan_catalog", required=True)
    parser.add_argument("--titan_split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=["dinov2", "dofa", "croma"])
    parser.add_argument("--unfreeze_blocks", nargs="+", type=int, default=[2])
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Optimizer update interval in microbatches; default effective batch is 32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--require_real_weights", action="store_true")
    parser.add_argument("--allow_nondeterministic", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--repaired_protocol", help="Sealed repaired fine-tuning protocol; requires both frozen paths")
    parser.add_argument("--frozen_reference", help="Verified same-fold probing_results.json with all five head seeds")
    parser.add_argument("--frozen_features_dir", help="Its feature producer's features directory, containing the model subdirectory")
    args = parser.parse_args()
    repaired_paths = (args.repaired_protocol, args.frozen_reference, args.frozen_features_dir)
    if any(repaired_paths) and not all(repaired_paths):
        parser.error("--repaired_protocol, --frozen_reference and --frozen_features_dir must be supplied together")
    return args


def main():
    args = parse_args()
    repaired = prepare_repaired_run(args) if args.repaired_protocol else None
    output_dir = Path(args.output_dir)
    inputs = {"titan_catalog": args.titan_catalog, "titan_split_manifest": args.titan_split_manifest}
    if repaired is not None:
        inputs.update(repaired_protocol=args.repaired_protocol, frozen_reference=args.frozen_reference,
                      frozen_analysis_manifest=Path(args.frozen_reference).parent / "run_manifest.json",
                      frozen_feature_manifest=Path(args.frozen_features_dir).parent / "run_manifest.json",
                      frozen_features=args.frozen_features_dir)
    manifest_path = start_run_manifest(
        output_dir,
        sys.argv,
        inputs,
    )
    results_path = output_dir / "finetuning_results.json"

    all_results = {}
    for model_name in args.models:
        logger.info("Fine-tuning %s", model_name)
        all_results[model_name] = run_model(args, model_name, repaired=repaired)
        results_path.write_text(json.dumps(sanitize_json(all_results), indent=2) + "\n")
    complete_run_manifest(manifest_path)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
