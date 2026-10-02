"""Domain gap metrics and evaluation for TitanSAR experiments.

Implements:
- Maximum Mean Discrepancy (MMD): single, repeated, permutation-tested, and
  group-resampled (jackknife/bootstrap) estimators
- Proxy A-distance (Ben-David et al. 2010)
- Per-class centroid distances
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_mmd(
    source_features: np.ndarray,
    target_features: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    seed: int = 0,
    max_samples: int = 2000,
    estimator: str = "biased",
    equal_sample_size: bool = True,
) -> float:
    """Compute a seeded MMD estimate between two finite feature arrays.

    RBF bandwidth estimation occurs after random subsampling, avoiding dependence
    on catalog row order. Equal sample sizes are used by default so comparisons
    are not driven by unequal diagonal bias. ``estimator`` may be ``biased``
    (V-statistic) or ``unbiased`` (U-statistic).
    """
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD inputs must be two-dimensional")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("MMD inputs must have the same feature dimension")
    if len(source_features) == 0 or len(target_features) == 0:
        raise ValueError("MMD inputs must be non-empty")
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise ValueError("MMD inputs contain NaN or infinite values")
    if estimator not in {"biased", "unbiased"}:
        raise ValueError("estimator must be 'biased' or 'unbiased'")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    rng = np.random.default_rng(seed)
    if equal_sample_size:
        n = min(len(source_features), len(target_features), max_samples)
        source_features = source_features[rng.choice(len(source_features), n, replace=False)]
        target_features = target_features[rng.choice(len(target_features), n, replace=False)]
    else:
        if len(source_features) > max_samples:
            source_features = source_features[
                rng.choice(len(source_features), max_samples, replace=False)
            ]
        if len(target_features) > max_samples:
            target_features = target_features[
                rng.choice(len(target_features), max_samples, replace=False)
            ]

    if kernel == "linear":
        mean_diff = source_features.mean(axis=0) - target_features.mean(axis=0)
        mmd_sq = float(np.dot(mean_diff, mean_diff))
        if estimator == "unbiased":
            if min(len(source_features), len(target_features)) < 2:
                raise ValueError("unbiased MMD requires at least two samples per domain")
            mmd_sq -= float(source_features.var(axis=0, ddof=1).sum() / len(source_features))
            mmd_sq -= float(target_features.var(axis=0, ddof=1).sum() / len(target_features))
        return mmd_sq
    if kernel != "rbf":
        raise ValueError("kernel must be 'rbf' or 'linear'")

    if gamma is None:
        from scipy.spatial.distance import pdist

        bandwidth_sample = np.concatenate([source_features, target_features], axis=0)
        if len(bandwidth_sample) > 2000:
            bandwidth_sample = bandwidth_sample[
                rng.choice(len(bandwidth_sample), 2000, replace=False)
            ]
        squared_distances = pdist(bandwidth_sample, metric="sqeuclidean")
        positive = squared_distances[squared_distances > 0]
        gamma = 1.0 / float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")

    def rbf_kernel(x, y):
        # Direct differences avoid cancellation for tightly clustered features
        # with a large common offset, matching pdist in bandwidth estimation.
        squared_distances = cdist(x, y, metric="sqeuclidean")
        return np.exp(-gamma * squared_distances)

    k_ss = rbf_kernel(source_features, source_features)
    k_tt = rbf_kernel(target_features, target_features)
    k_st = rbf_kernel(source_features, target_features)
    if estimator == "biased":
        mmd_sq = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
        return float(max(mmd_sq, 0.0))

    n_s, n_t = len(source_features), len(target_features)
    if n_s < 2 or n_t < 2:
        raise ValueError("unbiased MMD requires at least two samples per domain")
    mmd_sq = (
        (k_ss.sum() - np.trace(k_ss)) / (n_s * (n_s - 1))
        + (k_tt.sum() - np.trace(k_tt)) / (n_t * (n_t - 1))
        - 2.0 * k_st.mean()
    )
    return float(mmd_sq)


def compute_mmd_repeated(
    source_features: np.ndarray,
    target_features: np.ndarray,
    repeats: int = 100,
    base_seed: int = 0,
    **kwargs,
) -> dict:
    """Return all repeated-subsampling MMD estimates and summary quantiles.

    The ``ci95`` entry holds the empirical 2.5/97.5 percent quantiles of the
    subsampling distribution. It describes subsampling variability, not a
    parametric confidence interval for the population MMD.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    values = np.array([
        compute_mmd(source_features, target_features, seed=base_seed + i, **kwargs)
        for i in range(repeats)
    ])
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "repeats": repeats,
        "base_seed": base_seed,
    }


def compute_mmd_permutation_test(
    source_features: np.ndarray,
    target_features: np.ndarray,
    permutations: int = 1000,
    seed: int = 0,
    max_samples: int = 500,
    gamma: Optional[float] = None,
) -> dict:
    """Test equal distributions using a fixed-bandwidth biased MMD permutation test."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD inputs must be two-dimensional")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("MMD inputs must have the same feature dimension")
    if min(len(source_features), len(target_features), max_samples) < 1:
        raise ValueError("MMD inputs and max_samples must be non-empty/positive")
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise ValueError("MMD inputs contain NaN or infinite values")
    rng = np.random.default_rng(seed)
    n = min(len(source_features), len(target_features), max_samples)
    source = np.asarray(source_features)[rng.choice(len(source_features), n, replace=False)]
    target = np.asarray(target_features)[rng.choice(len(target_features), n, replace=False)]
    pooled = np.concatenate([source, target])
    if gamma is None:
        from scipy.spatial.distance import pdist
        positive = pdist(pooled, metric="sqeuclidean")
        positive = positive[positive > 0]
        gamma = 1.0 / float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    distances = cdist(pooled, pooled, metric="sqeuclidean")
    kernel = np.exp(-gamma * distances)

    def statistic(first):
        selected = np.zeros(2 * n, dtype=bool)
        selected[first] = True
        return float(
            kernel[np.ix_(selected, selected)].mean()
            + kernel[np.ix_(~selected, ~selected)].mean()
            - 2.0 * kernel[np.ix_(selected, ~selected)].mean()
        )

    observed = statistic(np.arange(n))
    null_values = np.array([
        statistic(rng.permutation(2 * n)[:n]) for _ in range(permutations)
    ])
    p_value = (1.0 + float(np.sum(null_values >= observed))) / (permutations + 1.0)
    return {
        "observed": observed,
        "p_value": p_value,
        "permutations": permutations,
        "seed": seed,
        "sample_size_per_domain": n,
        "gamma": float(gamma),
    }


def compute_group_resampling_mmd(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    repeats: int = 1000,
    seed: int = 0,
    mode: str = "target_bootstrap",
    **kwargs,
) -> dict:
    """Estimate MMD sensitivity to target spatial bootstrap or source jackknife.

    ``ci95`` holds the empirical 2.5/97.5 percent quantiles over the resampled
    (or jackknifed) values; it is a resampling spread, not a parametric
    confidence interval.
    """
    source_groups = np.asarray(source_groups)
    target_groups = np.asarray(target_groups)
    if len(source_groups) != len(source_features) or len(target_groups) != len(target_features):
        raise ValueError("Group arrays must align with feature arrays")
    rng = np.random.default_rng(seed)
    values = []
    if mode == "source_jackknife":
        groups = np.unique(source_groups)
        for group in groups:
            values.append(compute_mmd(
                source_features[source_groups != group], target_features,
                seed=seed, **kwargs,
            ))
    elif mode == "target_bootstrap":
        groups = np.unique(target_groups)
        group_indices = {group: np.flatnonzero(target_groups == group) for group in groups}
        for repeat in range(repeats):
            sampled = rng.choice(groups, len(groups), replace=True)
            indices = np.concatenate([group_indices[group] for group in sampled])
            values.append(compute_mmd(
                source_features, target_features[indices], seed=seed + repeat, **kwargs,
            ))
    else:
        raise ValueError("mode must be 'source_jackknife' or 'target_bootstrap'")
    values = np.asarray(values)
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "mode": mode,
        "groups": int(len(np.unique(source_groups if mode == "source_jackknife" else target_groups))),
        "seed": seed,
    }


def compute_proxy_a_distance(
    source_features: np.ndarray,
    target_features: np.ndarray,
    max_samples: int = 5000,
    seed: int = 0,
) -> float:
    """Compute proxy A-distance (Ben-David et al. 2010).

    Trains a linear classifier to distinguish source from target features.
    A-distance = 2 * (1 - 2 * error), where error is the classifier's
    generalization error. Higher A-distance means larger domain gap.

    Caveat: with high-dimensional (e.g. 768-d) features and a few thousand
    points, a linear classifier is almost always separable, so this proxy
    saturates near 2 regardless of true overlap (a random-feature control gives
    ~2 as well). Interpret with that ceiling in mind, or reduce dimensionality
    (e.g. PCA) before computing it. Subsampling and shuffling are seeded.

    Returns value in [0, 2].
    """
    # Equal domain priors prevent a majority-domain classifier from appearing
    # to detect a shift solely because one catalog contains more tiles.
    rng = np.random.default_rng(seed)
    sample_size = min(len(source_features), len(target_features), max_samples)
    if sample_size < 2:
        raise ValueError("Proxy A-distance requires at least two samples per domain")
    if len(source_features) > sample_size:
        idx = rng.choice(len(source_features), sample_size, replace=False)
        source_features = source_features[idx]
    if len(target_features) > sample_size:
        idx = rng.choice(len(target_features), sample_size, replace=False)
        target_features = target_features[idx]

    # Binary domain labels: 0 = source, 1 = target
    X = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])

    # Stratified seeded split, then fit preprocessing on the training partition only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SGD linear classifier
    clf = SGDClassifier(loss="hinge", max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    error = 1.0 - balanced_accuracy_score(y_test, clf.predict(X_test))

    # A-distance
    a_distance = 2.0 * (1.0 - 2.0 * error)
    return float(np.clip(a_distance, 0.0, 2.0))


def compute_centroid_distances(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int = 6,
) -> dict:
    """Compute per-class centroid distances between domains.

    For each class, computes the L2 distance between the mean feature
    vector in the source domain and the mean feature vector in the target
    domain. Small distances suggest domain-invariant representations for
    that class; large distances indicate a failure mode.
    """
    distances = {}
    for c in range(num_classes):
        s_mask = source_labels == c
        t_mask = target_labels == c

        if s_mask.sum() == 0 or t_mask.sum() == 0:
            distances[c] = float("nan")
            continue

        s_centroid = source_features[s_mask].mean(axis=0)
        t_centroid = target_features[t_mask].mean(axis=0)
        distances[c] = float(np.linalg.norm(s_centroid - t_centroid))

    # Mean distance across classes (excluding NaN)
    valid = [d for d in distances.values() if not np.isnan(d)]
    distances["mean"] = float(np.mean(valid)) if valid else float("nan")

    return distances
