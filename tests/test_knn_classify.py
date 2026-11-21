"""Cosine kNN requires a nonempty reference set and a positive integer k."""

import numpy as np
import pytest

from titansar.models.probing import knn_classify


@pytest.mark.parametrize("k", [
    0, -1, np.int64(0), True, False, np.bool_(True), 1.5, "2", None,
])
def test_invalid_neighborhood_rejects(k):
    with pytest.raises(ValueError, match="positive integer"):
        knn_classify(
            np.eye(2), np.array([1, 1]), np.array([[1.0, 0.0]]),
            np.array([1]), k=k, num_classes=2,
        )


@pytest.mark.parametrize("feature_count,label_count", [(0, 0), (0, 1), (1, 0)])
def test_empty_training_rejects_before_capping_k(feature_count, label_count):
    with pytest.raises(ValueError, match="nonempty training"):
        knn_classify(
            np.ones((feature_count, 2)), np.ones(label_count, dtype=int),
            np.array([[1.0, 0.0]]), np.array([1]), k=20, num_classes=2,
        )


@pytest.mark.parametrize("k", [1, np.int64(1)])
def test_nearest_cosine_neighbors_and_recall(k):
    result = knn_classify(
        np.array([[10.0, 0.0], [1.0, 1.0], [0.0, 3.0]]),
        np.array([0, 0, 1]), np.eye(2), np.array([0, 1]), k=k, num_classes=2,
    )
    np.testing.assert_array_equal(result["predictions"], [0, 1])
    assert result["per_class_accuracy"] == [1.0, 1.0]
    assert result["macro_accuracy"] == 1.0
    assert result["k"] == 1


def test_large_k_uses_all_available_training_rows():
    result = knn_classify(
        np.eye(2), np.array([1, 1]), np.array([[1.0, 0.0]]),
        np.array([1]), k=20, num_classes=2,
    )
    assert result["k"] == 2
    np.testing.assert_array_equal(result["predictions"], [1])


def test_equal_vote_counts_retain_lowest_class_tie_break():
    result = knn_classify(
        np.eye(2), np.array([1, 0]), np.array([[2.0, 1.0]]),
        np.array([0]), k=2, num_classes=2,
    )
    np.testing.assert_array_equal(result["predictions"], [0])


def test_error_metrics_and_absent_class_retain_public_contract():
    result = knn_classify(
        np.eye(2), np.array([0, 1]),
        np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 1.0]]),
        np.array([0, 0, 1]), k=1, num_classes=3,
    )
    assert set(result) == {
        "macro_accuracy", "per_class_accuracy", "kappa", "k", "predictions",
    }
    np.testing.assert_array_equal(result["predictions"], [0, 1, 1])
    assert result["per_class_accuracy"][:2] == [0.5, 1.0]
    assert np.isnan(result["per_class_accuracy"][2])
    assert result["macro_accuracy"] == 0.75
    # Counts [[1, 1], [0, 1]] give observed agreement 2/3 and expected 4/9.
    assert result["kappa"] == pytest.approx(0.4)
    assert result["k"] == 1
