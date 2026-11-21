"""Synthetic checks for cosine kNN and its input contracts."""

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

@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("label_count", [1, 3])
def test_row_mismatch_rejects_even_when_missing_label_is_not_selected(split, label_count):
    arrays = {
        "train_features": np.eye(2),
        "train_labels": np.array([0, 1]),
        "test_features": np.array([[1., 0.], [1., 0.]]),
        "test_labels": np.array([0, 0]),
    }
    arrays[f"{split}_labels"] = np.zeros(label_count, dtype=int)
    with pytest.raises(ValueError, match="aligned"):
        knn_classify(**arrays, k=1, num_classes=2)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("features", [np.array(1.), np.array([1., 0.]), np.ones((2, 1, 2))])
def test_features_require_two_dimensions(split, features):
    arrays = {
        "train_features": np.eye(2), "train_labels": np.array([0, 1]),
        "test_features": np.eye(2), "test_labels": np.array([0, 1]),
    }
    arrays[f"{split}_features"] = features
    with pytest.raises(ValueError, match="two-dimensional"):
        knn_classify(**arrays, k=1, num_classes=2)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("labels", [np.array(0), np.array([[0], [1]])])
def test_labels_require_one_dimension(split, labels):
    arrays = {
        "train_features": np.eye(2), "train_labels": np.array([0, 1]),
        "test_features": np.eye(2), "test_labels": np.array([0, 1]),
    }
    arrays[f"{split}_labels"] = labels
    with pytest.raises(ValueError, match="one-dimensional"):
        knn_classify(**arrays, k=1, num_classes=2)


def test_feature_widths_must_match():
    with pytest.raises(ValueError, match="feature width"):
        knn_classify(
            np.eye(2), np.array([0, 1]), np.ones((1, 3)),
            np.array([0]), k=1, num_classes=2,
        )


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_reject_before_ranking(split, value):
    arrays = {
        "train_features": np.eye(2), "train_labels": np.array([0, 1]),
        "test_features": np.eye(2), "test_labels": np.array([0, 1]),
    }
    arrays[f"{split}_features"][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        knn_classify(**arrays, k=1, num_classes=2)


def test_empty_test_set_retains_zero_metrics_and_absent_classes():
    result = knn_classify(
        np.eye(2), np.array([0, 1]), np.empty((0, 2)),
        np.empty(0, dtype=int), k=20, num_classes=2,
    )
    assert result["predictions"].shape == (0,)
    assert result["predictions"].dtype == np.dtype(float)
    assert result["macro_accuracy"] == 0.0
    assert result["kappa"] == 0.0
    assert np.isnan(result["per_class_accuracy"]).all()
    assert result["k"] == 2


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_zero_vectors_remain_valid(dtype):
    with np.errstate(divide="raise", invalid="raise"):
        result = knn_classify(
            np.zeros((2, 2), dtype=dtype), np.array([1, 1]),
            np.zeros((1, 2), dtype=dtype), np.array([1]), k=2, num_classes=2,
        )
    np.testing.assert_array_equal(result["predictions"], [1])
    assert result["macro_accuracy"] == 1.0



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


def test_float16_zero_reference_has_zero_similarity():
    with np.errstate(divide="raise", invalid="raise"):
        result = knn_classify(
            np.array([[0., 0.], [-1., 0.]], dtype=np.float16),
            np.array([1, 0]), np.array([[1., 0.]], dtype=np.float16),
            np.array([1]), k=1, num_classes=2,
        )
    np.testing.assert_array_equal(result["predictions"], [1])
    assert result["macro_accuracy"] == 1.0


@pytest.mark.parametrize("k,expected", [(1, 1), (2, 0), (20, 0)])
def test_float16_zero_query_preserves_neighbor_and_vote_ties(k, expected):
    with np.errstate(divide="raise", invalid="raise"):
        result = knn_classify(
            np.array([[1., 0.], [-1., 0.]], dtype=np.float16),
            np.array([1, 0]), np.zeros((1, 2), dtype=np.float16),
            np.array([expected]), k=k, num_classes=2,
        )
    np.testing.assert_array_equal(result["predictions"], [expected])
    assert result["k"] == min(k, 2)


def test_float16_zero_reference_with_empty_queries_retains_empty_result():
    with np.errstate(divide="raise", invalid="raise"):
        result = knn_classify(
            np.zeros((2, 2), dtype=np.float16), np.array([1, 0]),
            np.empty((0, 2), dtype=np.float16), np.empty(0, dtype=int),
            k=20, num_classes=2,
        )
    assert result["predictions"].shape == (0,)
    assert result["predictions"].dtype == np.dtype(float)
    assert result["macro_accuracy"] == result["kappa"] == 0.0
    assert np.isnan(result["per_class_accuracy"]).all()
    assert result["k"] == 2


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("dtype,magnitude", [
    (np.float16, 1e-4), (np.float16, 300.),
    (np.float32, 1e-30), (np.float32, 1e20),
    (np.float64, 1e-200), (np.float64, 1e200),
])
def test_finite_rows_beyond_original_norm_range_are_supported(split, dtype, magnitude):
    arrays = {
        "train_features": np.array([[1., 1.], [1., 0.]], dtype=dtype),
        "train_labels": np.array([0, 1]),
        "test_features": np.array([[1., 0.]], dtype=dtype),
        "test_labels": np.array([1]),
    }
    features = arrays[f"{split}_features"]
    features[-1] = [magnitude, 0.]
    assert np.isfinite(features).all() and np.any(features[-1] != 0)
    with np.errstate(over="ignore", under="ignore"):
        raw = np.linalg.norm(features[-1])
    assert raw == 0 or not np.isfinite(raw)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = knn_classify(**arrays, k=1, num_classes=2)
    np.testing.assert_array_equal(result["predictions"], [1])


# Decimal computes the oracle directly from squared norms and dot products,
# independently of the implementation's maximum-component scaling.
def _cosine_oracle_predictions(reference, labels, queries, k):
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        rows = [[Decimal.from_float(float(x)) for x in row] for row in reference]
        norms = [sum(x*x for x in row).sqrt() for row in rows]
        predictions = []
        for query in queries:
            q = [Decimal.from_float(float(x)) for x in query]
            qnorm = sum(x*x for x in q).sqrt()
            scores = [float(sum(x*y for x, y in zip(row, q)) / (norm*qnorm))
                      if norm and qnorm else 0.0 for row, norm in zip(rows, norms)]
            indices = np.argsort(-np.array(scores))[:min(k, len(labels))]
            predictions.append(np.argmax(np.bincount(labels[indices])))
    return np.array(predictions)


@pytest.mark.parametrize("dtype,small", [
    (np.float16, 0.0003), (np.float32, 1e-10), (np.float64, 1e-12),
])
def test_positive_finite_norm_collinear_neighbor_matches_cosine_oracle(dtype, small):
    reference = np.array([[1., .4], [small, 0.]], dtype=dtype)
    queries = np.array([[1., 0.]], dtype=dtype)
    labels = np.array([0, 1])
    expected = _cosine_oracle_predictions(reference, labels, queries, 1)
    np.testing.assert_array_equal(expected, [1])
    result = knn_classify(reference, labels, queries, expected, k=1, num_classes=2)
    np.testing.assert_array_equal(result["predictions"], expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("side", ["reference", "query"])
@pytest.mark.parametrize("edge", ["subnormal", "maximum"])
def test_finite_extreme_axis_vectors_match_oracle(dtype, side, edge):
    magnitude = np.nextafter(dtype(0), dtype(1)) if edge == "subnormal" else np.finfo(dtype).max
    reference = np.array([[1., 1.], [1., 0.], [-1., 0.]], dtype=dtype)
    queries = np.array([[1., 0.], [-1., 0.]], dtype=dtype)
    labels = np.array([0, 1, 2])
    if side == "reference":
        reference[1:, 0] = [magnitude, -magnitude]
    else:
        queries[:, 0] = [magnitude, -magnitude]
    expected = _cosine_oracle_predictions(reference, labels, queries, 1)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = knn_classify(reference, labels, queries, expected, k=1, num_classes=3)
    np.testing.assert_array_equal(result["predictions"], expected)


@pytest.mark.parametrize("dtype,exponent", [(np.float16, 12), (np.float32, 100), (np.float64, 900)])
@pytest.mark.parametrize("k", [1, 3, 20])
def test_independent_row_rescaling_preserves_cosine_votes(dtype, exponent, k):
    reference = np.array([[1., .5], [-.5, 1.], [-1., -.5], [.5, -1.]], dtype=dtype)
    queries = np.array([[1., .25], [-.25, 1.], [-1., -.25], [.25, -1.]], dtype=dtype)
    labels = np.array([0, 1, 2, 3])
    expected = _cosine_oracle_predictions(reference, labels, queries, k)
    # Powers of two preserve these represented component ratios exactly.
    factors = np.ldexp(np.ones(4, dtype=dtype), [-exponent, exponent, exponent, -exponent])
    scaled_reference = reference * factors[:, None]
    scaled_queries = queries * factors[::-1, None]
    for ref, query in [(reference, queries), (scaled_reference, queries),
                       (reference, scaled_queries), (scaled_reference, scaled_queries)]:
        np.testing.assert_array_equal(_cosine_oracle_predictions(ref, labels, query, k), expected)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            result = knn_classify(ref, labels, query, expected, k=k, num_classes=4)
        np.testing.assert_array_equal(result["predictions"], expected)
        assert result["k"] == min(k, len(labels))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_distinct_near_cosine_scores_use_float64_working_precision(dtype):
    reference = np.array([[1., 2e-4], [1., 1e-4]], dtype=dtype)
    query = np.array([[1., 0.]], dtype=dtype)
    labels = np.array([0, 1])
    expected = _cosine_oracle_predictions(reference, labels, query, 1)
    np.testing.assert_array_equal(expected, [1])
    result = knn_classify(reference, labels, query, expected, k=1, num_classes=2)
    np.testing.assert_array_equal(result["predictions"], expected)


def test_readonly_noncontiguous_arrays_are_not_mutated():
    reference = np.array([[9., 0., 8., 1.], [9., 1., 8., 0.], [9., -1., 8., 0.]])[:, 3::-2]
    queries = np.array([[9., 0., 8., 1.], [9., 1., 8., 0.]])[:, 3::-2]
    labels = np.array([0, 1, 2])
    expected = _cosine_oracle_predictions(reference, labels, queries, 1)
    before = [a.copy() for a in (reference, queries)]
    for array in (reference, queries):
        assert not array.flags.c_contiguous
        array.flags.writeable = False
    result = knn_classify(reference, labels, queries, expected, k=1, num_classes=3)
    np.testing.assert_array_equal(result["predictions"], expected)
    for actual, original in zip((reference, queries), before):
        np.testing.assert_array_equal(actual, original)
        assert not actual.flags.writeable


@pytest.mark.parametrize("k", [1, 2, 20])
def test_batch_boundary_and_equal_similarity_vote_policy(k):
    reference = np.array([[1., 0.], [2., 0.], [0., 1.], [0., 0.]])
    labels = np.array([1, 0, 2, 2])
    queries = np.tile(np.array([[1., 0.], [0., 0.], [0., 1.]]), (172, 1))
    expected = _cosine_oracle_predictions(reference, labels, queries, k)
    result = knn_classify(reference, labels, queries, expected, k=k, num_classes=3)
    np.testing.assert_array_equal(result["predictions"], expected)
    assert len(result["predictions"]) == 516
    assert result["k"] == min(k, len(labels))


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
def test_integer_features_convert_before_magnitude_operations(dtype):
    info = np.iinfo(dtype)
    reference = np.array([[info.max, 0], [info.min, info.max]], dtype=dtype)
    labels = np.array([0, 1])
    queries = reference[::-1]
    expected = _cosine_oracle_predictions(reference, labels, queries, 1)
    result = knn_classify(reference, labels, queries, expected, k=1, num_classes=2)
    np.testing.assert_array_equal(result["predictions"], expected)


@pytest.mark.parametrize("side", ["reference", "query"])
@pytest.mark.parametrize("dtype", [np.bool_, np.complex64, np.complex128, object, "U3"])
def test_unsupported_feature_dtypes_reject_explicitly(side, dtype):
    reference, query = np.eye(2), np.eye(2)
    if side == "reference":
        reference = reference.astype(dtype)
    else:
        query = query.astype(dtype)
    with pytest.raises(ValueError, match="integer or float16/32/64"):
        knn_classify(reference, np.array([0, 1]), query, np.array([0, 1]), k=1, num_classes=2)


@pytest.mark.parametrize("side", ["reference", "query"])
def test_wider_float_features_reject_without_silent_downcast(side):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform has no wider NumPy float")
    reference, query = np.eye(2), np.eye(2)
    if side == "reference":
        reference = reference.astype(np.longdouble)
    else:
        query = query.astype(np.longdouble)
    with pytest.raises(ValueError, match="integer or float16/32/64"):
        knn_classify(reference, np.array([0, 1]), query, np.array([0, 1]), k=1, num_classes=2)


def test_zero_width_vectors_retain_zero_similarity_policy():
    result = knn_classify(np.empty((2, 0)), np.array([1, 0]),
                          np.empty((1, 0)), np.array([0]), k=2, num_classes=2)
    np.testing.assert_array_equal(result["predictions"], [0])
