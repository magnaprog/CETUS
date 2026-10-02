"""Run foundation model probing experiments on TitanSAR.

This script:
1. Loads the TitanSAR and Earth-analog datasets
2. Extracts features from all tiles using each foundation model
3. Runs linear probing and k-NN experiments
4. Computes domain gap metrics
5. Saves all results

Use --within_titan_only for Titan geographic evaluation without Earth inputs.
It supports both --features_only and --cached_features_dir. Cached analysis
still requires Titan training tiles to reproduce normalization provenance.
New-mode sidecars and source_metadata bind evaluation_scope, benchmark_track,
and the full dataset normalization_provenance object. File hashes are SHA256
of file bytes; normalization sample and membership formats are recorded by
the dataset. Unused Earth catalog and split hashes are null.

Usage:
    # Sequential (single GPU):
    python scripts/run_probing.py --titan_catalog data/titan_sar/catalog.json \
                                  --earth_catalog data/earth_analog_real/catalog.json \
                                  --output_dir outputs/probing \
                                  --models dinov2 dofa croma random_init \
                                  --device cuda

    # Parallel (one model per GPU, auto-assigned):
    python scripts/run_probing.py --titan_catalog data/titan_sar/catalog.json \
                                  --earth_catalog data/earth_analog_real/catalog.json \
                                  --output_dir outputs/probing \
                                  --models dinov2 dofa croma random_init \
                                  --parallel

    # Parallel with explicit GPU assignment:
    python scripts/run_probing.py --titan_catalog ... --earth_catalog ... \
                                  --parallel --gpu_ids 0 1 2 3
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from titansar.configs.defaults import CASSINI_FREQ_GHZ, SENTINEL1_FREQ_GHZ, TitanSARConfig
from titansar.data.dataset import TitanSARDataset, EarthAnalogDataset
from titansar.models.foundation_models import get_model
from titansar.models.probing import (
    extract_features,
    train_linear_probe,
    evaluate_linear_probe,
    knn_classify,
    compute_silhouette_score,
    save_feature_metadata,
    load_bound_feature_cache,
)
from titansar.reproducibility import complete_run_manifest, start_run_manifest
from titansar.evaluation.metrics import (
    compute_centroid_distances,
    compute_group_resampling_mmd,
    compute_mmd_permutation_test,
    compute_mmd_repeated,
    compute_proxy_a_distance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _spatial_group(lat, lon, radius_m=2_574_700.0, block_size_m=250_000.0):
    lat_band = int(np.floor((np.radians(lat) + np.pi / 2.0) * radius_m / block_size_m))
    lat_center = -np.pi / 2.0 + (lat_band + 0.5) * block_size_m / radius_m
    longitude_blocks = max(
        1, round(2.0 * np.pi * radius_m * max(np.cos(lat_center), 1e-6) / block_size_m)
    )
    lon_band = int(np.floor((lon % 360.0) / 360.0 * longitude_blocks))
    return f"{lat_band}:{longitude_blocks}:{lon_band}"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_single_model(
    model_name: str,
    device: str,
    titan_catalog: str,
    earth_catalog: str,
    titan_split_manifest: str,
    earth_split_manifest: str,
    output_dir: str,
    batch_size: int,
    seeds: list[int],
    num_workers: int = 4,
    allow_placeholder: bool = True,
    features_only: bool = False,
    cached_features_dir: str = None,
    within_titan_only: bool = False,
) -> dict:
    """Run the full probing pipeline for a single foundation model.

    This function is designed to be called in a separate process for multi-GPU
    parallelism. It creates its own datasets, dataloaders, and model instance
    so that nothing non-picklable crosses process boundaries.

    Args:
        model_name: One of "dinov2", "dofa", "croma", "random_init".
        device: CUDA device string, e.g. "cuda:0".
        titan_catalog: Path to Titan SAR catalog JSON.
        earth_catalog: Path to Earth analog catalog JSON.
        output_dir: Directory to save features and results.
        batch_size: Batch size for feature extraction.
        seeds: List of random seeds for linear probe training.
        num_workers: Number of DataLoader workers.

    Returns:
        Results dict (JSON-serializable) with linear probe, k-NN, silhouette,
        and domain gap metrics.
    """
    if not within_titan_only and (not earth_catalog or not earth_split_manifest):
        raise ValueError("Earth catalog and split manifest are required for cross-domain probing")
    if features_only and cached_features_dir:
        raise ValueError("features_only and cached_features_dir are mutually exclusive")
    if within_titan_only:
        earth_catalog = earth_split_manifest = None

    # Set up logging with model/device prefix for parallel runs
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{model_name}@{device}] %(levelname)s %(message)s",
        force=True,
    )
    log = logging.getLogger(f"titansar.probing.{model_name}")

    config = TitanSARConfig()
    output_path = Path(output_dir)

    log.info(f"Starting {model_name} on {device}")
    t_start = time.time()

    # Load datasets (must be created inside the worker process)
    titan_train = TitanSARDataset(
        titan_catalog, split="train", config=config,
        split_manifest_path=titan_split_manifest,
    )
    titan_test = TitanSARDataset(
        titan_catalog, split="test", config=config,
        split_manifest_path=titan_split_manifest,
    )
    earth_train = None if within_titan_only else EarthAnalogDataset(
        earth_catalog, split="train", config=config,
        split_manifest_path=earth_split_manifest,
    )
    selk_dataset = TitanSARDataset(
        titan_catalog, split="selk_holdout", config=config,
        split_manifest_path=titan_split_manifest,
    )
    expected_provenance = {
        "titan_catalog_sha256": _sha256(titan_catalog),
        "earth_catalog_sha256": _sha256(earth_catalog) if earth_catalog else None,
        "titan_split_manifest_sha256": _sha256(titan_split_manifest),
        "earth_split_manifest_sha256": _sha256(earth_split_manifest) if earth_split_manifest else None,
    }
    if within_titan_only:
        if not len(titan_train) or not len(titan_test):
            raise ValueError("Titan training and test partitions must be nonempty")
        normalization = titan_train.normalization_provenance
        if not titan_train.benchmark_track:
            raise ValueError("Titan catalog must declare benchmark_track")
        sampler = normalization.get("sampling")
        if sampler is not None and (
            sampler["missing_tile_ids"] or sampler["empty_tile_ids"] or sampler["sample_count"] <= 0
        ):
            raise ValueError("Titan normalization requires all sampled tiles to contain usable pixels")
        for dataset in (titan_test, selk_dataset):
            if (dataset.normalization_provenance != normalization
                    or dataset.benchmark_track != titan_train.benchmark_track):
                raise ValueError("Titan partitions must share training-fit normalization and benchmark track")
        expected_provenance.update({
            "evaluation_scope": "within_titan_only",
            "benchmark_track": titan_train.benchmark_track,
            "normalization_provenance": normalization,
        })
    earth_train_feats = earth_train_labels = earth_train_ids = None
    if cached_features_dir:
        def read_cache(stem, dataset):
            return load_bound_feature_cache(
                Path(cached_features_dir) / model_name, stem, model_name,
                expected_provenance,
                {entry["tile_id"]: entry["label"] for entry in dataset.entries},
            )
        titan_train_feats, titan_train_labels, titan_train_ids, meta = read_cache("titan_train_feats", titan_train)
        titan_test_feats, titan_test_labels, titan_test_ids, test_meta = read_cache("titan_test_feats", titan_test)
        other_metadata = [test_meta]
        if earth_train is not None:
            earth_train_feats, earth_train_labels, earth_train_ids, earth_meta = read_cache("earth_train_feats", earth_train)
            other_metadata.append(earth_meta)
        for other in other_metadata:
            for key in ("model_revision", "weights_sha256"):
                expected = meta["provenance"].get(key)
                if not expected or other["provenance"].get(key) != expected:
                    raise RuntimeError(f"Feature caches do not share a verified {key}")
        weights_source = meta["weights_source"]
        encoder_identity = {key: meta["provenance"][key] for key in ("model_revision", "weights_sha256")}
    else:
        titan_train_loader = DataLoader(titan_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        titan_test_loader = DataLoader(titan_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        earth_train_loader = (DataLoader(earth_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                              if earth_train is not None else None)

        # Load model onto assigned GPU
        model = get_model(model_name, device=device, allow_placeholder=allow_placeholder)

        # Extract features
        log.info("Extracting Titan training features...")
        t0 = time.time()
        titan_train_feats, titan_train_labels, titan_train_ids = extract_features(
            model, titan_train_loader, device
        )
        log.info(f"  Done in {time.time()-t0:.1f}s. Shape: {titan_train_feats.shape}")

        log.info("Extracting Titan test features...")
        titan_test_feats, titan_test_labels, titan_test_ids = extract_features(
            model, titan_test_loader, device
        )

        if earth_train_loader is not None:
            log.info("Extracting Earth training features...")
            if model_name == "dofa":
                model.wavelength = SENTINEL1_FREQ_GHZ
            try:
                earth_train_feats, earth_train_labels, earth_train_ids = extract_features(
                    model, earth_train_loader, device
                )
            finally:
                if model_name == "dofa":
                    model.wavelength = CASSINI_FREQ_GHZ

        # Extract Selk holdout features (for anomaly scoring consistency)
        if len(selk_dataset) > 0:
            selk_loader = DataLoader(selk_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            log.info(f"Extracting Selk holdout features ({len(selk_dataset)} tiles)...")
            selk_feats, selk_labels, selk_ids = extract_features(model, selk_loader, device)
        else:
            selk_feats, selk_labels, selk_ids = None, None, None

        # Save features
        feat_dir = output_path / "features" / model_name
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.save(feat_dir / "titan_train_feats.npy", titan_train_feats)
        np.save(feat_dir / "titan_train_labels.npy", titan_train_labels)
        np.save(feat_dir / "titan_train_ids.npy", np.array(titan_train_ids))
        np.save(feat_dir / "titan_test_feats.npy", titan_test_feats)
        np.save(feat_dir / "titan_test_labels.npy", titan_test_labels)
        np.save(feat_dir / "titan_test_ids.npy", np.array(titan_test_ids))
        if earth_train_feats is not None:
            np.save(feat_dir / "earth_train_feats.npy", earth_train_feats)
            np.save(feat_dir / "earth_train_labels.npy", earth_train_labels)
            np.save(feat_dir / "earth_train_ids.npy", np.array(earth_train_ids))
        if selk_feats is not None:
            np.save(feat_dir / "selk_holdout_feats.npy", selk_feats)
            np.save(feat_dir / "selk_holdout_labels.npy", selk_labels)
            np.save(feat_dir / "selk_holdout_ids.npy", np.array(selk_ids))

        encoder_identity = {"model_revision": model.model_revision, "weights_sha256": model.weights_sha256}
        provenance = {
            **expected_provenance,
            "benchmark_track": expected_provenance.get("benchmark_track", "TitanSAR-HiSAR-IMG-review-2026-09"),
            **encoder_identity,
        }
        # Per-array provenance sidecars so every cache is bound to its catalogs and
        # immutable split manifests.
        for _stem, _arr in [
            ("titan_train_feats", titan_train_feats),
            ("titan_test_feats", titan_test_feats),
            ("earth_train_feats", earth_train_feats),
            ("selk_holdout_feats", selk_feats),
        ]:
            if _arr is not None:
                id_stem = _stem.replace("_feats", "_ids")
                array_provenance = {
                    **provenance,
                    "feature_sha256": _sha256(feat_dir / f"{_stem}.npy"),
                    "tile_ids_sha256": _sha256(feat_dir / f"{id_stem}.npy"),
                    "labels_sha256": _sha256(
                        feat_dir / f"{_stem.replace('_feats', '_labels')}.npy"
                    ),
                }
                if model_name == "dofa":
                    array_provenance["band_identifier_ghz"] = (
                        SENTINEL1_FREQ_GHZ if _stem.startswith("earth_") else CASSINI_FREQ_GHZ
                    )
                save_feature_metadata(
                    feat_dir, _stem, model_name, model.weights_source,
                    n_features=_arr.shape[0], feature_dim=_arr.shape[1],
                    provenance=array_provenance,
                )

        weights_source = model.weights_source
        if features_only:
            return {"model": model_name, "weights_source": weights_source,
                    "feature_extraction_only": True, "time_seconds": time.time() - t_start,
                    **({"evaluation_scope": "within_titan_only",
                        "source_metadata": {**expected_provenance, **encoder_identity}} if within_titan_only else {})}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_results = {
        "model": model_name,
        "device": device,
        "weights_source": weights_source,
        "linear_probe": {},
        "knn": {},
    }
    if within_titan_only:
        model_results.update(evaluation_scope="within_titan_only",
                             source_metadata={**expected_provenance, **encoder_identity})
    else:
        source_classes = np.unique(earth_train_labels)
        evaluated_source_classes = np.intersect1d(source_classes, np.unique(titan_test_labels))
        if not len(evaluated_source_classes):
            raise RuntimeError("No Titan test labels overlap the Earth source classes")
        source_support = {
            "source_supported_evaluated_class_indices": evaluated_source_classes.tolist(),
            "source_supported_n_classes": int(len(evaluated_source_classes)),
            "source_supported_absent_target_class_indices": np.setdiff1d(
                source_classes, evaluated_source_classes
            ).tolist(),
        }

    # --- Linear Probing: Train on Titan, test on Titan ---
    log.info("Linear probing: Titan -> Titan")
    probe_results = []
    for seed in seeds:
        probe = train_linear_probe(
            titan_train_feats, titan_train_labels,
            num_classes=config.num_classes,
            class_weights=titan_train.get_class_weights(),
            config=config, seed=seed, device=device,
        )
        result = evaluate_linear_probe(
            probe, titan_test_feats, titan_test_labels,
            num_classes=config.num_classes, device=device,
        )
        if within_titan_only:
            result.update(tile_ids=list(titan_test_ids), true_labels=titan_test_labels.tolist())
        else:
            result["source_supported_macro_accuracy"] = float(np.mean([
                result["per_class_accuracy"][int(class_index)]
                for class_index in evaluated_source_classes
            ]))
            result.update(source_support)
        probe_results.append(result)

    # Aggregate across seeds
    macro_accs = [r["macro_accuracy"] for r in probe_results]
    kappas = [r["kappa"] for r in probe_results]
    model_results["linear_probe"]["titan_to_titan"] = {
        "training_class_indices": np.unique(titan_train_labels).tolist(),
        "macro_accuracy_mean": float(np.mean(macro_accs)),
        "macro_accuracy_std": float(np.std(macro_accs, ddof=1)),
        "kappa_mean": float(np.mean(kappas)),
        "kappa_std": float(np.std(kappas, ddof=1)),
        "per_class_accuracy": [
            float(np.mean([r["per_class_accuracy"][c] for r in probe_results]))
            for c in range(config.num_classes)
        ],
        "seed_results": [
            {"seed": seed, **result} for seed, result in zip(seeds, probe_results)
        ],
    }
    if not within_titan_only:
        supported_within = [result["source_supported_macro_accuracy"] for result in probe_results]
        model_results["linear_probe"]["titan_to_titan"].update({
            "source_supported_class_indices": source_classes.tolist(), **source_support,
            "source_supported_macro_accuracy_mean": float(np.mean(supported_within)),
            "source_supported_macro_accuracy_std": float(np.std(supported_within, ddof=1)),
        })

    log.info("k-NN classification: Titan -> Titan")
    knn_result = knn_classify(
        titan_train_feats, titan_train_labels, titan_test_feats, titan_test_labels,
        k=config.knn_k, num_classes=config.num_classes,
    )
    model_results["knn"]["titan_to_titan"] = {
        "macro_accuracy": knn_result["macro_accuracy"],
        "kappa": knn_result["kappa"],
        "per_class_accuracy": knn_result["per_class_accuracy"],
    }
    if within_titan_only:
        model_results["knn"]["titan_to_titan"].update(
            predictions=knn_result["predictions"], k=knn_result["k"],
            tile_ids=list(titan_test_ids), true_labels=titan_test_labels.tolist(),
        )
        model_results["silhouette_titan"] = compute_silhouette_score(titan_train_feats, titan_train_labels)
        log.info("%s Titan-only evaluation completed in %.1fs", model_name, time.time() - t_start)
        return model_results

    # --- Linear Probing: Earth -> Titan (cross-domain) ---
    # Primary transfer metrics cover only classes represented in the Earth source.
    cross_mask = np.isin(titan_test_labels, source_classes)
    if not cross_mask.any():
        raise RuntimeError("No Titan test labels overlap the Earth source classes")
    log.info(
        "Linear probing: Earth -> Titan on source-supported classes %s",
        source_classes.tolist(),
    )
    probe_results_cross = []
    for seed in seeds:
        probe = train_linear_probe(
            earth_train_feats, earth_train_labels,
            num_classes=config.num_classes,
            class_weights=earth_train.get_class_weights(),
            config=config, seed=seed, device=device,
        )
        result = evaluate_linear_probe(
            probe, titan_test_feats[cross_mask], titan_test_labels[cross_mask],
            num_classes=config.num_classes, device=device,
        )
        result.update(source_support)
        probe_results_cross.append(result)

    macro_accs_cross = [r["macro_accuracy"] for r in probe_results_cross]
    kappas_cross = [r["kappa"] for r in probe_results_cross]
    model_results["linear_probe"]["earth_to_titan"] = {
        "supported_class_indices": source_classes.tolist(),
        **source_support,
        "excluded_target_count": int((~cross_mask).sum()),
        "evaluated_target_count": int(cross_mask.sum()),
        "macro_accuracy_mean": float(np.mean(macro_accs_cross)),
        "macro_accuracy_std": float(np.std(macro_accs_cross, ddof=1)),
        "kappa_mean": float(np.mean(kappas_cross)),
        "kappa_std": float(np.std(kappas_cross, ddof=1)),
        "per_class_accuracy": [
            (
                float(np.mean([
                    r["per_class_accuracy"][c] for r in probe_results_cross
                    if not np.isnan(r["per_class_accuracy"][c])
                ]))
                if any(
                    not np.isnan(r["per_class_accuracy"][c])
                    for r in probe_results_cross
                )
                else float("nan")
            )
            for c in range(config.num_classes)
        ],
        "seed_results": [
            {"seed": seed, **result}
            for seed, result in zip(seeds, probe_results_cross)
        ],
    }

    log.info("k-NN classification: Earth -> Titan")
    knn_cross = knn_classify(
        earth_train_feats, earth_train_labels,
        titan_test_feats[cross_mask], titan_test_labels[cross_mask],
        k=config.knn_k, num_classes=config.num_classes,
    )
    model_results["knn"]["earth_to_titan"] = {
        "macro_accuracy": knn_cross["macro_accuracy"],
        "kappa": knn_cross["kappa"],
        "per_class_accuracy": knn_cross["per_class_accuracy"],
    }

    # --- Silhouette Score ---
    log.info("Computing silhouette score...")
    sil_titan = compute_silhouette_score(titan_train_feats, titan_train_labels)
    model_results["silhouette_titan"] = sil_titan

    # --- Domain Gap Metrics ---
    log.info("Computing domain gap metrics...")
    mmd_result = compute_mmd_repeated(
        earth_train_feats, titan_train_feats, repeats=100,
        estimator="biased", equal_sample_size=True,
    )
    mmd_unbiased = compute_mmd_repeated(
        earth_train_feats, titan_train_feats, repeats=100,
        estimator="unbiased", equal_sample_size=True,
    )
    permutation = compute_mmd_permutation_test(
        earth_train_feats, titan_train_feats, permutations=1000, max_samples=500,
    )
    earth_site_by_id = {
        entry["tile_id"]: entry["site"] for entry in earth_train.entries
    }
    titan_group_by_id = {
        entry["tile_id"]: _spatial_group(entry["center_lat"], entry["center_lon"])
        for entry in titan_train.entries
    }
    earth_groups = np.array([earth_site_by_id[str(tile_id)] for tile_id in earth_train_ids])
    titan_groups = np.array([titan_group_by_id[str(tile_id)] for tile_id in titan_train_ids])
    site_jackknife = compute_group_resampling_mmd(
        earth_train_feats,
        titan_train_feats,
        earth_groups,
        titan_groups,
        mode="source_jackknife",
        max_samples=500,
    )
    spatial_bootstrap = compute_group_resampling_mmd(
        earth_train_feats,
        titan_train_feats,
        earth_groups,
        titan_groups,
        repeats=1000,
        mode="target_bootstrap",
        max_samples=500,
    )
    a_dist = compute_proxy_a_distance(earth_train_feats, titan_train_feats)
    centroids = compute_centroid_distances(
        earth_train_feats, earth_train_labels,
        titan_train_feats, titan_train_labels,
        num_classes=config.num_classes,
    )

    model_results["domain_gap"] = {
        "earth_to_titan": {
            "mmd_biased": mmd_result,
            "mmd_unbiased_sensitivity": mmd_unbiased,
            "mmd_permutation_test": permutation,
            "earth_site_jackknife": site_jackknife,
            "titan_spatial_block_bootstrap": spatial_bootstrap,
            "a_distance_appendix": a_dist,
            "centroid_distances_within_model_appendix": centroids,
        }
    }

    elapsed = time.time() - t_start
    log.info(
        f"\n{model_name} completed in {elapsed:.1f}s:\n"
        f"  Titan->Titan linear probe: {model_results['linear_probe']['titan_to_titan']['macro_accuracy_mean']:.3f} "
        f"+/- {model_results['linear_probe']['titan_to_titan']['macro_accuracy_std']:.3f}\n"
        f"  Earth->Titan linear probe: {model_results['linear_probe']['earth_to_titan']['macro_accuracy_mean']:.3f}\n"
        f"  k-NN: {model_results['knn']['titan_to_titan']['macro_accuracy']:.3f}\n"
        f"  Silhouette: {sil_titan:.3f}\n"
        f"  MMD: {mmd_result['mean']:.4f}, A-dist: {a_dist:.3f}"
    )

    return model_results


def run_models_on_gpu(model_names, device, kwargs):
    """Keep one process responsible for each GPU, even when jobs finish unevenly."""
    results, failed = {}, []
    for name in model_names:
        try:
            results[name] = run_single_model(name, device, **kwargs)
        except Exception:
            logger.exception("Model %s failed on %s", name, device)
            failed.append(name)
    return results, failed


def parse_args():
    parser = argparse.ArgumentParser(description="TitanSAR foundation model probing")
    parser.add_argument("--titan_catalog", type=str, required=True)
    parser.add_argument("--earth_catalog", type=str)
    parser.add_argument("--titan_split_manifest", type=str, required=True)
    parser.add_argument("--earth_split_manifest", type=str)
    parser.add_argument("--within_titan_only", action="store_true",
                        help="Evaluate Titan only; Earth inputs are unused. Bind actual catalog track and training-fit normalization.")
    parser.add_argument("--output_dir", type=str, default="outputs/probing")
    parser.add_argument("--models", nargs="+", default=["dinov2", "dofa", "croma", "random_init"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--require_real_weights", action="store_true",
        help="Fail instead of falling back to random placeholder weights when a "
             "model's real checkpoint is unavailable (use for evaluation runs; D8).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--features_only", action="store_true",
                      help="Extract and bind caches, leaving CPU analyses to a separate job")
    mode.add_argument("--cached_features_dir",
                      help="Analyze verified caches without loading an encoder")
    # Multi-GPU options
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run models in parallel across multiple GPUs (one model per GPU).",
    )
    parser.add_argument(
        "--gpu_ids", nargs="+", type=int, default=None,
        help="GPU IDs to use in parallel mode. Default: auto-assign GPUs 0..N-1.",
    )
    args = parser.parse_args()
    if not args.within_titan_only and (not args.earth_catalog or not args.earth_split_manifest):
        parser.error("--earth_catalog and --earth_split_manifest are required unless --within_titan_only is set")
    return args


def main():
    args = parse_args()
    if len(args.models) != len(set(args.models)):
        raise ValueError('Requested models must be unique')
    output_dir = Path(args.output_dir)
    manifest_path = start_run_manifest(
        output_dir,
        sys.argv,
        {
            "titan_catalog": args.titan_catalog,
            "titan_split_manifest": args.titan_split_manifest,
            **({"earth_catalog": args.earth_catalog,
                "earth_split_manifest": args.earth_split_manifest} if not args.within_titan_only else {}),
            **({"cached_features": args.cached_features_dir} if args.cached_features_dir else {}),
        },
    )

    all_results = {}

    if args.parallel:
        # --- Multi-GPU parallel execution ---
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("--parallel requires CUDA GPUs but none were found.")

        gpu_ids = args.gpu_ids or list(range(num_gpus))
        if len(set(gpu_ids)) != len(gpu_ids) or any(i < 0 or i >= num_gpus for i in gpu_ids):
            raise ValueError("gpu_ids must name distinct available CUDA devices")
        gpu_groups = [(f"cuda:{gpu}", args.models[index::len(gpu_ids)])
                      for index, gpu in enumerate(gpu_ids)]
        gpu_groups = [(device, names) for device, names in gpu_groups if names]
        kwargs = dict(
            titan_catalog=args.titan_catalog, earth_catalog=args.earth_catalog,
            titan_split_manifest=args.titan_split_manifest,
            earth_split_manifest=args.earth_split_manifest,
            output_dir=args.output_dir, batch_size=args.batch_size, seeds=args.seeds,
            num_workers=max(1, 4 // len(gpu_groups)),
            allow_placeholder=not args.require_real_weights,
            features_only=args.features_only, cached_features_dir=args.cached_features_dir,
            within_titan_only=args.within_titan_only,
        )
        t_start = time.time()
        failed = []
        with ProcessPoolExecutor(
            max_workers=len(gpu_groups), mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = [(names, executor.submit(run_models_on_gpu, names, device, kwargs))
                       for device, names in gpu_groups]
            for names, future in futures:
                try:
                    results, errors = future.result()
                    all_results.update(results)
                    failed.extend(errors)
                except Exception:
                    logger.exception("GPU worker failed for models %s", names)
                    failed.extend(names)

        logger.info(f"All models completed in {time.time() - t_start:.1f}s")
        strict = args.require_real_weights or os.environ.get(
            "TITANSAR_REQUIRE_REAL_WEIGHTS", "").lower() in ("1", "true", "yes")
        if failed and strict:
            raise RuntimeError(
                f"Strict mode: {len(failed)} requested model(s) failed and were "
                f"not run: {failed}. Refusing to emit a partial result set."
            )

    else:
        # --- Sequential execution ---
        # Reuses run_single_model for each model to avoid code duplication.
        # Datasets are reloaded per model (acceptable overhead vs maintaining
        # two divergent code paths).
        for model_name in args.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name} (sequential on {args.device})")
            logger.info(f"{'='*60}")
            all_results[model_name] = run_single_model(
                model_name=model_name,
                device=args.device,
                titan_catalog=args.titan_catalog,
                earth_catalog=args.earth_catalog,
                titan_split_manifest=args.titan_split_manifest,
                earth_split_manifest=args.earth_split_manifest,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                seeds=args.seeds,
                num_workers=4,
                allow_placeholder=not args.require_real_weights,
                features_only=args.features_only,
                cached_features_dir=args.cached_features_dir,
                within_titan_only=args.within_titan_only,
            )

    # Save all results
    results_path = output_dir / "probing_results.json"

    def _sanitize_for_json(obj):
        """Recursively convert numpy types and replace NaN/Inf with None for valid JSON."""
        if isinstance(obj, dict):
            return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize_for_json(obj.tolist())
        return obj

    with open(results_path, "w") as f:
        json.dump(_sanitize_for_json(all_results), f, indent=2)
    complete_run_manifest(manifest_path)
    logger.info(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
