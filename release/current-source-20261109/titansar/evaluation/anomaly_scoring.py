"""Order Selk holdout tiles by relative uncertainty and model disagreement.

The reported Selk score combines entropy of the ensemble-mean probabilities
with model disagreement. Scaling within the holdout gives a relative ordering,
without estimating anomaly probabilities. Scientific priorities and mission
decisions require independent evidence. The optional VIMS component requires
quantitative measurements and is unused in that run. The available VIMS display
channels are unsuitable for this calculation.

1. Classification entropy: uncertainty of the ensemble-mean class probabilities.

2. VIMS spectral anomaly (optional, unused): distance of a quantitative ratio
   vector from its predicted class's reference distribution.

3. Model disagreement: fraction of ensemble members whose predictions differ
   from a modal class prediction.
"""

import logging
from typing import Optional

import numpy as np

from titansar.configs.defaults import TitanSARConfig

logger = logging.getLogger(__name__)


def reference_order(reference_ids, model_ids) -> np.ndarray:
    """Index array reordering rows indexed by ``model_ids`` into ``reference_ids``.

    The ensemble entropy/disagreement combine each model's per-tile outputs
    row-wise, so all models must share one tile order even if their feature caches
    were written in different orders. Raises ValueError if the two ID sets differ.
    When the orders already match, the returned index is the identity.
    """
    model_ids = [str(t) for t in model_ids]
    reference_ids = [str(t) for t in reference_ids]
    if len(set(model_ids)) != len(model_ids) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Selk tile-id lists contain duplicates")
    if set(model_ids) != set(reference_ids):
        raise ValueError("Selk tile-id sets differ across model caches")
    pos = {tid: i for i, tid in enumerate(model_ids)}
    return np.array([pos[tid] for tid in reference_ids], dtype=int)


def classification_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of class probability distributions.

    H = -Σ p_i log p_i

    Higher entropy means the probe is less certain about the tile; it does not
    by itself indicate novel or interesting terrain.

    Args:
        probabilities: (N, num_classes) softmax outputs

    Returns:
        (N,) entropy values
    """
    # Clip to avoid log(0)
    p = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def vims_spectral_anomaly(
    vims_band_ratios: np.ndarray,
    predicted_labels: np.ndarray,
    class_means: Optional[np.ndarray] = None,
    class_covs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute classwise distances for quantitative VIMS band-ratio vectors.

    Each three-component vector is compared with its predicted class mean
    using Mahalanobis distance. If either statistics array is absent, both
    are estimated from the input rows grouped by predicted class. Estimated
    covariances receive 1e-6 diagonal regularization; classes with fewer than
    two rows use zero means and identity covariances. If covariance inversion
    raises LinAlgError, including for a detected singular covariance, the
    score is the Euclidean distance from the class mean.

    The reported Selk analysis excludes this optional component.

    Args:
        vims_band_ratios: (N, 3) quantitative VIMS band-ratio vectors
        predicted_labels: (N,) predicted terrain class indices
        class_means: Optional (num_classes, 3) reference means
        class_covs: Optional (num_classes, 3, 3) reference covariances

    Returns:
        (N,) Mahalanobis distances, with Euclidean distances where covariance
        inversion fails
    """
    num_classes = int(predicted_labels.max()) + 1

    # Compute class statistics if not provided
    if class_means is None or class_covs is None:
        class_means = np.zeros((num_classes, 3))
        class_covs = np.zeros((num_classes, 3, 3))
        for c in range(num_classes):
            mask = predicted_labels == c
            if mask.sum() < 2:
                class_covs[c] = np.eye(3)
                continue
            class_data = vims_band_ratios[mask]
            class_means[c] = class_data.mean(axis=0)
            class_covs[c] = np.cov(class_data.T) + 1e-6 * np.eye(3)

    scores = np.zeros(len(vims_band_ratios))
    for i in range(len(vims_band_ratios)):
        c = predicted_labels[i]
        diff = vims_band_ratios[i] - class_means[c]
        try:
            cov_inv = np.linalg.inv(class_covs[c])
            scores[i] = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            scores[i] = np.linalg.norm(diff)

    return scores


def model_disagreement(
    predictions_per_model: list[np.ndarray],
) -> np.ndarray:
    """Compute the fraction of ensemble members differing from a modal prediction.

    Args:
        predictions_per_model: list of (N,) arrays, one per ensemble member
            (a model-fold probe in the reported Selk analysis)

    Returns:
        (N,) disagreement scores in [0, 1]
    """
    num_models = len(predictions_per_model)
    if num_models <= 1:
        return np.zeros(len(predictions_per_model[0]))

    preds = np.stack(predictions_per_model, axis=0)  # (M, N)
    n_tiles = preds.shape[1]

    scores = np.zeros(n_tiles)
    for i in range(n_tiles):
        tile_preds = preds[:, i]
        # Mode = most common prediction
        values, counts = np.unique(tile_preds, return_counts=True)
        max_agreement = counts.max()
        scores[i] = 1.0 - max_agreement / num_models

    return scores


def composite_priority_score(
    entropy_scores: np.ndarray,
    vims_scores: Optional[np.ndarray] = None,
    disagreement_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the composite relative uncertainty/disagreement score.

    Each component is min-max normalized to [0, 1] within the input set and
    averaged. Higher score means the ensemble is more uncertain or less
    consistent on that tile. The score provides a relative ordering within the
    input set; it does not establish a sampling priority.

    Args:
        entropy_scores: (N,) classification entropy
        vims_scores: (N,) VIMS spectral anomaly (optional)
        disagreement_scores: (N,) model disagreement (optional)

    Returns:
        (N,) composite relative scores in [0, 1]
    """

    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    components = [normalize(entropy_scores)]
    if vims_scores is not None:
        components.append(normalize(vims_scores))
    if disagreement_scores is not None:
        components.append(normalize(disagreement_scores))

    return np.mean(components, axis=0)


def generate_priority_map(
    tile_ids: list[str],
    tile_lats: np.ndarray,
    tile_lons: np.ndarray,
    priority_scores: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Rank the top-K tiles by composite score within the analyzed region.

    Returns a list of dicts describing each high-ranked tile. Ranks are
    relative to the input set only.
    """
    # Stable sort so tied scores rank deterministically by input order.
    top_indices = np.argsort(-priority_scores, kind="stable")[:top_k]

    priority_sites = []
    for rank, idx in enumerate(top_indices):
        site = {
            "rank": rank + 1,
            "tile_id": tile_ids[idx],
            "lat": float(tile_lats[idx]),
            "lon": float(tile_lons[idx]),
            "priority_score": float(priority_scores[idx]),
            "predicted_class": class_names[predicted_labels[idx]],
            "predicted_class_idx": int(predicted_labels[idx]),
        }
        priority_sites.append(site)

    return priority_sites
