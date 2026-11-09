"""Descriptive error analysis for TitanSAR map predictions.

Summarize agreement by location, product intensity, reference class and modal
map fraction. These associations do not identify physical causes of errors.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def align_ids_to_index(feat_ids, id_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Join feature rows to catalog metadata rows by tile id.

    Feature arrays are saved in DataLoader order, while catalog metadata is often
    built in a different order (e.g. sorted by tile id). Pairing them positionally
    can misalign per-tile geography, intensity and map-fraction records.
    This maps both sides through the shared tile id instead (review D17).

    Args:
        feat_ids: per-feature-row tile ids (length = number of feature rows)
        id_to_idx: mapping from tile id to its row index in the metadata arrays

    Returns:
        (keep, order): ``keep[j]`` indexes the j-th usable FEATURE row;
        ``order[j]`` indexes the matching METADATA row for the same tile id.
        Feature rows whose tile id is absent from ``id_to_idx`` are dropped.
    """
    keep, order = [], []
    for i, tid in enumerate(feat_ids):
        key = str(tid)
        idx = id_to_idx.get(key)
        if idx is not None:
            keep.append(i)
            order.append(idx)
    return np.asarray(keep, dtype=int), np.asarray(order, dtype=int)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson correlation that returns None on degenerate input.

    np.corrcoef yields NaN (not an error) when either input has zero variance,
    and that NaN would silently leak into the saved JSON/logs. Guard against it
    and against fewer than two points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def geographic_error_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    num_classes: int = 6,
) -> dict:
    """Summarize prediction correctness within fixed geographic bands.

    These diagnostic regions are separate from the evaluation folds:
    - North polar (lat > 60)
    - South polar (lat < -60)
    - Northern mid-latitudes (30 < lat <= 60)
    - Southern mid-latitudes (-60 <= lat < -30)
    - Equatorial (|lat| <= 30)

    Returns dict with per-region accuracy stats.
    """
    correct = (predictions == labels).astype(float)

    regions = {
        "north_polar": (lats > 60),
        "south_polar": (lats < -60),
        "north_mid": (lats > 30) & (lats <= 60),
        "south_mid": (lats >= -60) & (lats < -30),
        "equatorial": (np.abs(lats) <= 30),
    }

    # Also split by longitude hemisphere
    lon_regions = {
        "western_hemisphere": (lons >= 180) & (lons <= 360),
        "eastern_hemisphere": (lons >= 0) & (lons < 180),
    }

    results = {"latitude_regions": {}, "longitude_regions": {}, "overall": {}}

    # Overall accuracy
    results["overall"]["accuracy"] = float(correct.mean()) if correct.size else None
    results["overall"]["n_tiles"] = int(len(labels))

    # Per latitude region
    for name, mask in regions.items():
        n = int(mask.sum())
        if n == 0:
            results["latitude_regions"][name] = {
                "accuracy": None,
                "n_tiles": 0,
            }
            continue

        region_acc = float(correct[mask].mean())
        # Reference-class recall within region
        per_class = {}
        for c in range(num_classes):
            class_mask = mask & (labels == c)
            nc = int(class_mask.sum())
            if nc > 0:
                per_class[c] = {
                    "accuracy": float(correct[class_mask].mean()),
                    "n_tiles": nc,
                }
            else:
                per_class[c] = {"accuracy": None, "n_tiles": 0}

        results["latitude_regions"][name] = {
            "accuracy": region_acc,
            "n_tiles": n,
            "per_class": per_class,
        }

    # Per longitude region
    for name, mask in lon_regions.items():
        n = int(mask.sum())
        if n == 0:
            results["longitude_regions"][name] = {
                "accuracy": None,
                "n_tiles": 0,
            }
            continue
        results["longitude_regions"][name] = {
            "accuracy": float(correct[mask].mean()),
            "n_tiles": n,
        }

    logger.info(
        "Geographic error analysis: %s",
        {k: v.get("accuracy") for k, v in results["latitude_regions"].items()},
    )

    return results


def intensity_binned_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    mean_product_intensity: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Bin tiles by product-native mean intensity and report accuracy per bin.

    Quantiles define bin edges. Ties can produce unequal or empty bins; a
    constant finite sample forms one bin. Missing intensity values are omitted.
    An empty finite sample has zero bins and an undefined correlation.

    Returns dict with bin edges, per-bin accuracy, and per-bin counts.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    valid_mask = np.isfinite(mean_product_intensity)
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    mean_product_intensity = mean_product_intensity[valid_mask]

    correct = (predictions == labels).astype(float)

    # Quantile-based bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = (np.percentile(mean_product_intensity, quantiles)
                 if mean_product_intensity.size else np.array([]))
    # Collapse tied edges, preserving a single bin for a constant sample.
    bin_edges = np.unique(bin_edges)
    if bin_edges.size == 1:
        bin_edges = np.repeat(bin_edges, 2)
    actual_n_bins = max(0, len(bin_edges) - 1)

    bins = []
    for i in range(actual_n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < actual_n_bins - 1:
            mask = (mean_product_intensity >= low) & (mean_product_intensity < high)
        else:
            mask = (mean_product_intensity >= low) & (mean_product_intensity <= high)

        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None

        bins.append({
            "bin_low_product_intensity": float(low),
            "bin_high_product_intensity": float(high),
            "accuracy": acc,
            "n_tiles": n,
            "mean_product_intensity": float(mean_product_intensity[mask].mean()) if n > 0 else None,
        })

    # Correlation between product-native intensity and correctness
    corr = _safe_corr(mean_product_intensity, correct)

    results = {
        "bins": bins,
        "n_bins_actual": actual_n_bins,
        "product_intensity_correctness_correlation": corr,
        "n_valid_tiles": int(valid_mask.sum()),
    }

    logger.info(
        "Intensity-binned accuracy: correlation=%s, bins=%d",
        f"{corr:.4f}" if corr is not None else "N/A",
        actual_n_bins,
    )

    return results


def confusion_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    top_n: int = 5,
) -> dict:
    """Compute confusion matrix and identify top-N most common errors.

    Returns raw and normalized confusion matrices, class precision, recall, F1,
    and a ranked list of the most frequent misclassification patterns.
    """
    num_classes = len(class_names)

    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        confusion[int(true), int(pred)] += 1

    # Normalized confusion matrix (by true class)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums_safe = np.maximum(row_sums, 1)
    confusion_norm = confusion / row_sums_safe

    # Per-class precision and recall
    per_class_metrics = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[class_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[c, :].sum()),
        }

    # Top-N most common errors
    errors = []
    for true_c in range(num_classes):
        for pred_c in range(num_classes):
            if true_c != pred_c and confusion[true_c, pred_c] > 0:
                errors.append({
                    "true_class": class_names[true_c],
                    "predicted_class": class_names[pred_c],
                    "count": int(confusion[true_c, pred_c]),
                    "error_rate": float(confusion_norm[true_c, pred_c]),
                })

    errors.sort(key=lambda x: x["count"], reverse=True)
    top_errors = errors[:top_n]

    results = {
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": confusion_norm.tolist(),
        "class_names": class_names,
        "per_class_metrics": per_class_metrics,
        "top_errors": top_errors,
        "overall_accuracy": float((predictions == labels).mean()) if labels.size else None,
    }

    logger.info("Top confusion errors:")
    for err in top_errors:
        logger.info(
            "  %s classified as %s: %d (%.1f%%)",
            err["true_class"],
            err["predicted_class"],
            err["count"],
            err["error_rate"] * 100,
        )

    return results


def label_confidence_correlation(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> dict:
    """Summarize map agreement as a function of modal map fraction.

    The legacy ``confidences`` argument contains catalog ``label_confidence``:
    the modal class count divided by the count of valid map samples. It measures
    map composition within a footprint. Expert reliability remains unmeasured.

    Return correctness rates above each fraction threshold and within fraction
    quartiles, plus Pearson correlation with binary prediction correctness.
    Missing fractions are omitted. An empty finite sample has empty quartile
    records and undefined rates and correlation. Legacy result keys are retained.
    """
    if thresholds is None:
        thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]

    valid_mask = np.isfinite(confidences)
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    confidences = confidences[valid_mask]

    correct = (predictions == labels).astype(float)

    # Accuracy among tiles at or above each modal fraction threshold.
    threshold_results = []
    for thresh in sorted(thresholds):
        mask = confidences >= thresh
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None
        threshold_results.append({
            "threshold": float(thresh),
            "accuracy": acc,
            "n_tiles": n,
            "fraction_of_data": float(n / len(labels)) if len(labels) > 0 else 0.0,
        })

    # Pearson correlation
    corr = _safe_corr(confidences, correct)

    # Fraction quartiles retain ties, so some bins may be empty.
    quartile_edges = (np.percentile(confidences, [0, 25, 50, 75, 100])
                      if confidences.size else np.array([]))
    quartiles = []
    for i in range(len(quartile_edges) - 1):
        low, high = quartile_edges[i], quartile_edges[i + 1]
        if i < 3:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else None
        quartiles.append({
            "quartile": i + 1,
            "confidence_range": [float(low), float(high)],
            "accuracy": acc,
            "n_tiles": n,
        })

    results = {
        "threshold_accuracy": threshold_results,
        "confidence_accuracy_correlation": corr,
        "quartile_accuracy": quartiles,
        "n_valid_tiles": int(valid_mask.sum()),
    }

    logger.info(
        "Modal map fraction correlation: r=%s",
        f"{corr:.4f}" if corr is not None else "N/A",
    )

    return results
