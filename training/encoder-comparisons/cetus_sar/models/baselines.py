"""Classical image features and simple classification baselines."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_predictions(labels, predictions, num_classes, probabilities=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(num_classes),
        zero_division=0,
    )
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1[support > 0])) if (support > 0).any() else 0.0,
        # Mean F1 over classes present in the evaluation labels, matching the
        # balanced-accuracy convention (absent classes are excluded, not zeroed).
        "kappa": float(cohen_kappa_score(labels, predictions)),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(num_classes)
        ).tolist(),
        "predictions": np.asarray(predictions).tolist(),
    }
    if probabilities is not None:
        result["probabilities"] = np.asarray(probabilities).tolist()
    return result


def majority_baseline(train_labels, test_labels, num_classes):
    majority = int(np.bincount(train_labels, minlength=num_classes).argmax())
    predictions = np.full(len(test_labels), majority)
    probabilities = np.zeros((len(test_labels), num_classes), dtype=float)
    probabilities[:, majority] = 1.0
    result = evaluate_predictions(
        test_labels, predictions, num_classes, probabilities
    )
    result["majority_class"] = majority
    return result


def class_prior_baseline(train_labels, test_labels, num_classes, seed):
    counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    predictions = rng.choice(num_classes, size=len(test_labels), p=probabilities)
    probability_rows = np.repeat(probabilities[None, :], len(test_labels), axis=0)
    result = evaluate_predictions(
        test_labels, predictions, num_classes, probability_rows
    )
    result["class_prior"] = probabilities.tolist()
    return result


def _normalized_uint8(image):
    finite = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.quantile(finite, [0.01, 0.99])
    scaled = np.clip((finite - low) / max(high - low, 1e-8), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8), scaled.astype(np.float32)


def intensity_features(image):
    values = np.asarray(image, dtype=np.float32).ravel()
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    # Standardized moments are undefined at zero variance. Encode a constant
    # tile with zero moment features; its mean and zero spread remain explicit.
    constant = values.min() == values.max()
    skew = 0.0 if constant else stats.skew(values)
    kurtosis = 0.0 if constant else stats.kurtosis(values)
    return np.array([
        values.mean(), values.std(), skew, kurtosis,
        values.min(), values.max(), *quantiles,
    ], dtype=np.float32)


def histogram_features(image, bins=32):
    _, scaled = _normalized_uint8(image)
    histogram, _ = np.histogram(scaled, bins=bins, range=(0.0, 1.0), density=True)
    return histogram.astype(np.float32)


def texture_features(image):
    from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
    from skimage.filters import gabor
    from skimage.transform import resize

    quantized, scaled = _normalized_uint8(image)
    quantized = (quantized // 16).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 4],
        angles=[0.0, np.pi / 2.0],
        levels=16,
        symmetric=True,
        normed=True,
    )
    glcm_values = np.concatenate([
        graycoprops(glcm, property_name).ravel()
        for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    ])

    # LBP intentionally runs on the 16-level quantized image, matching the
    # canonical published runs; switching to the full-range image would
    # change texture-baseline results.
    lbp = local_binary_pattern(quantized, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10), density=True)

    gabor_values = []
    for frequency in (0.10, 0.25):
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0):
            real, imaginary = gabor(scaled, frequency=frequency, theta=theta)
            magnitude = np.hypot(real, imaginary)
            gabor_values.extend([magnitude.mean(), magnitude.std()])

    reduced = resize(scaled, (64, 64), anti_aliasing=True)
    hog_values = hog(
        reduced,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    import pywt
    coefficients = pywt.wavedec2(reduced, "haar", level=2)
    wavelet_values = []
    for coefficient in [coefficients[0], *sum((list(level) for level in coefficients[1:]), [])]:
        wavelet_values.extend([np.mean(np.abs(coefficient)), np.std(coefficient)])

    return np.concatenate([
        glcm_values,
        lbp_hist,
        np.asarray(gabor_values),
        hog_values,
        np.asarray(wavelet_values),
    ]).astype(np.float32)


def classical_features(image):
    return np.concatenate([
        intensity_features(image),
        histogram_features(image),
        texture_features(image),
    ])


def fit_classical_probe(train_features, train_labels, seed):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    return classifier.fit(train_features, train_labels)
