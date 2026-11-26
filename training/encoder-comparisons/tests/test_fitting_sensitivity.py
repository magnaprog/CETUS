"""Check the quantities used to interpret the finite classifier fits."""
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, _logistic
from scipy.optimize._optimize import MemoizeJac, _prepare_scalar_function

from scripts.run_fitting_sensitivity import balanced_weights, class_metrics, loss_gradient, write_json
from scripts import run_fitting_sensitivity as runner


class FittingSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(620)
        self.y = np.repeat(np.arange(6), [1, 2, 3, 4, 2, 3])
        self.x = self.rng.normal(size=(len(self.y), 4)) * [3, .2, 1, 1] + [2, -4, 8, 0]
        self.x[:, -1] = 3
        self.w = self.rng.normal(scale=.1, size=(6, 4))
        self.b = self.rng.normal(scale=.1, size=6)

    def test_balanced_weight_totals(self):
        q = balanced_weights(self.y)
        assert_allclose(q.sum(), len(self.y), rtol=0, atol=1e-12)
        assert_allclose([q[self.y == c].sum() for c in range(6)], len(self.y) / 6)

    def test_loss_and_gradient_match_pinned_library(self):
        q = balanced_weights(self.y)
        loss = LinearModelLoss(HalfMultinomialLoss(n_classes=6), fit_intercept=True)
        for penalty in ("none", "l2"):
            with self.subTest(penalty=penalty):
                actual = loss_gradient(self.x, self.y, self.w, self.b, penalty)
                expected, gradient = loss.loss_gradient(
                    np.column_stack((self.w, self.b)), self.x, self.y.astype(float),
                    sample_weight=q, l2_reg_strength=0 if penalty == "none" else 1 / q.sum(), n_threads=1)
                assert_allclose(actual["objective"], expected, rtol=0, atol=1e-12)
                assert_allclose(actual["gradient_coef"], gradient[:, :-1], rtol=0, atol=1e-12)
                assert_allclose(actual["gradient_intercept"], gradient[:, -1], rtol=0, atol=1e-12)

    def test_directional_derivative(self):
        dw, db = self.rng.normal(size=self.w.shape), self.rng.normal(size=self.b.shape)
        eps = 1e-5
        for penalty in ("none", "l2"):
            with self.subTest(penalty=penalty):
                actual = loss_gradient(self.x, self.y, self.w, self.b, penalty)
                plus = loss_gradient(self.x, self.y, self.w + eps * dw, self.b + eps * db, penalty)
                minus = loss_gradient(self.x, self.y, self.w - eps * dw, self.b - eps * db, penalty)
                numeric = (plus["objective"] - minus["objective"]) / (2 * eps)
                analytic = np.sum(actual["gradient_coef"] * dw) + np.dot(actual["gradient_intercept"], db)
                assert_allclose(numeric, analytic, rtol=1e-6, atol=1e-8)

    def test_intercept_is_unpenalized(self):
        a = loss_gradient(self.x, self.y, self.w, self.b, "none")
        b = loss_gradient(self.x, self.y, self.w, self.b, "l2")
        assert_allclose(b["gradient_coef"] - a["gradient_coef"], self.w / len(self.y), atol=1e-12)
        assert_array_equal(a["gradient_intercept"], b["gradient_intercept"])
        assert_allclose(b["penalty"], np.square(self.w).sum() / (2 * len(self.y)), atol=1e-12)

    def test_native_diagnostic_matches_actual_scipy_forwarding(self):
        class Captured(Exception):
            pass

        for dtype in (np.float32, np.float64):
            for standard in (False, True):
                for penalty in ("none", "l2"):
                    with self.subTest(dtype=dtype, standard=standard, penalty=penalty):
                        x = self.x.astype(dtype)
                        if standard:
                            x = StandardScaler().fit_transform(x)
                        captured = {}

                        def intercept_optimizer(func, theta0, args, **kwargs):
                            captured.update(func=func, theta0=theta0, args=args)
                            raise Captured()

                        classifier = LogisticRegression(C=np.inf if penalty == "none" else 1.,
                                                        l1_ratio=0, solver="lbfgs", class_weight="balanced")
                        with patch.object(_logistic.optimize, "minimize", side_effect=intercept_optimizer):
                            with self.assertRaises(Captured):
                                classifier.fit(x, self.y)
                        memo = MemoizeJac(captured["func"])
                        scalar = _prepare_scalar_function(memo, captured["theta0"], jac=memo.derivative,
                                                          args=captured["args"])
                        w, b = self.w.astype(dtype), self.b.astype(dtype)
                        theta = np.column_stack((w, b)).astype(np.float64).ravel(order="F")
                        expected, gradient = scalar.fun_and_grad(theta)
                        actual = runner.solver_oracle_at_saved_state(x, self.y, w, b, penalty)
                        self.assertEqual(expected, actual["objective"])
                        assert_array_equal(gradient.reshape((6, 5), order="F"),
                                           np.column_stack((actual["gradient_coef"], actual["gradient_intercept"])))
                        self.assertEqual(actual["gradient_dtype"], str(np.dtype(dtype)))

    def test_float32_weights_and_prediction_replay(self):
        x, w, b = (v.astype(np.float32) for v in (self.x, self.w, self.b))
        actual = runner.solver_oracle_at_saved_state(x, self.y, w, b, "l2")
        self.assertEqual(actual["weight_sum"], 14.999999046325684)
        self.assertEqual(actual["regularization_strength"], 1 / actual["weight_sum"])
        classifier = LogisticRegression()
        classifier.coef_, classifier.intercept_ = w, b
        classifier.classes_, classifier.n_features_in_ = np.arange(6), 4
        pred, prob = runner.native_prediction_replay(x, w, b)
        assert_array_equal(pred, classifier.predict(x))
        assert_array_equal(prob, classifier.predict_proba(x))
        self.assertTrue(runner.valid_probability_rows(prob))
        mathematical = loss_gradient(x, self.y, w, b, "l2")
        self.assertFalse(np.allclose(prob, mathematical["probabilities"], rtol=0, atol=1e-12))
        self.assertFalse(np.allclose(prob.sum(axis=1), 1, rtol=0, atol=1e-12))

    def test_probability_validity_rejects_bad_rows(self):
        for row in ([.2, .2], [np.nan, 1], [-.1, 1.1]):
            with self.subTest(row=row):
                self.assertFalse(runner.valid_probability_rows(np.array([row], dtype=np.float32)))

    def test_float32_gradient_residual_is_not_float64_residual(self):
        x = np.full((len(self.y), 4), 10000, dtype=np.float32)
        w, b = np.zeros((6, 4), dtype=np.float32), np.zeros(6, dtype=np.float32)
        oracle = runner.solver_oracle_at_saved_state(x, self.y, w, b, "none")
        mathematical = loss_gradient(x, self.y, w, b, "none")
        reference = StandardScaler().fit(x)

        def residual(values):
            g, gb = (values[k].astype(float) for k in ("gradient_coef", "gradient_intercept"))
            return max(np.abs((g - gb[:, None] * reference.mean_) / reference.scale_).max(), np.abs(gb).max())

        self.assertGreater(residual(oracle), 1e-4)
        self.assertLess(residual(mathematical), 1e-10)

    def test_cross_dtype_argmax_can_differ(self):
        x = np.array([[1e8, 1]], dtype=np.float32).repeat(6, axis=0)
        w = np.zeros((6, 2), dtype=np.float32)
        w[:, 0], w[1, 1] = 1, 1
        b = np.full(6, -1e8, dtype=np.float32)
        pred, prob = runner.native_prediction_replay(x, w, b)
        mathematical = loss_gradient(x, np.arange(6), w, b, "none")
        assert_array_equal(pred, np.zeros(6, dtype=int))
        assert_array_equal(mathematical["logits"].argmax(axis=1), np.ones(6, dtype=int))
        self.assertTrue(runner.valid_probability_rows(prob))

    def test_affine_logits_and_penalty_maps(self):
        scaler = StandardScaler().fit(self.x)
        self.assertEqual(scaler.scale_[-1], 1)
        z = scaler.transform(self.x)
        v = self.w * scaler.scale_
        a = self.b + self.w @ scaler.mean_
        assert_allclose(z @ v.T + a, self.x @ self.w.T + self.b, rtol=0, atol=1e-10)
        assert_allclose(np.square(v).sum(), np.sum(np.square(self.w) * np.square(scaler.scale_)), atol=1e-10)
        assert_allclose(np.square(v / scaler.scale_).sum(), np.square(self.w).sum(), atol=1e-10)

    def test_gradient_coordinate_transform(self):
        scaler = StandardScaler().fit(self.x)
        raw = loss_gradient(self.x, self.y, self.w, self.b, "none")
        standard = loss_gradient(scaler.transform(self.x), self.y,
                                 self.w * scaler.scale_, self.b + self.w @ scaler.mean_, "none")
        expected = (raw["gradient_coef"] - raw["gradient_intercept"][:, None] * scaler.mean_) / scaler.scale_
        assert_allclose(standard["gradient_coef"], expected, rtol=0, atol=1e-10)
        assert_allclose(standard["gradient_intercept"], raw["gradient_intercept"], rtol=0, atol=1e-10)

    def test_batch_weighting_is_not_global_weighting(self):
        q = balanced_weights(self.y)
        losses = np.arange(len(self.y), dtype=float)
        singleton_means = q * losses / q
        assert_array_equal(singleton_means, losses)
        self.assertNotAlmostEqual(float(singleton_means.mean()), float(q @ losses / q.sum()))
        class_weights = 1 / np.bincount(self.y).astype(float)
        class_weights /= class_weights.mean()
        a = class_weights[self.y]
        assert_allclose(a @ losses / a.sum(), q @ losses / q.sum(), atol=1e-12)

    def test_class_scores_include_empty_prediction_classes(self):
        predicted = np.array([0] * len(self.y))
        actual = class_metrics(self.y, predicted)
        expected = precision_recall_fscore_support(self.y, predicted, labels=np.arange(6), zero_division=0)
        for name, values in zip(("precision", "recall", "f1", "support"), expected):
            assert_allclose(actual[name], values, rtol=0, atol=1e-12)

    def test_result_writer_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            write_json(path, {"attempt": 1})
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {"attempt": 2})
            self.assertEqual(path.read_bytes(), first)

    def test_nonfinite_json_leaves_no_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_publication_failure_preserves_staging_and_prior_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            with patch.object(runner.os, "link", side_effect=OSError("injected publication failure")):
                with self.assertRaises(OSError):
                    write_json(path, {"loss": 2})
            self.assertFalse(path.exists())
            staged = list(Path(folder).glob("*.partial"))
            self.assertEqual(len(staged), 1)
            self.assertEqual(json.loads(staged[0].read_text()), {"loss": 2})

    def test_runtime_binding_rejects_substituted_used_class(self):
        expected = runner.runtime_identity()
        self.assertEqual(runner.verify_runtime(expected), expected)

        class AlternateClassifier:
            def fit(self):
                raise AssertionError("must not reach fit")

        with patch.object(runner, "LogisticRegression", AlternateClassifier):
            with self.assertRaisesRegex(ValueError, "Imported runtime implementation differs"):
                runner.verify_runtime(expected)

    def test_failed_fit_has_terminal_record_and_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = {"output_root": str(Path(folder) / "runs"), "feature_root": str(Path(folder) / "inputs"),
                    "producer_commit": "synthetic", "python_version": "synthetic", "package_versions": {},
                    "runtime_identity": {}, "source_sha256": {}}
            cell = {"id": "dinov2-fold-0-raw-none", "model": "dinov2", "fold": 0,
                    "standardize": False, "regularization": "none"}
            with patch.object(runner, "load_inputs", return_value=({}, {})), \
                    patch.object(runner, "fit_arrays", side_effect=RuntimeError("injected fit failure")):
                with self.assertRaisesRegex(RuntimeError, "injected fit failure"):
                    runner.fit_cell(plan, cell, "synthetic-plan")
                output = Path(plan["output_root"]) / cell["id"]
                before = {p.name: p.read_bytes() for p in output.iterdir()}
                failure = json.loads(before["failed.json"])
                self.assertEqual(failure["status"], "failed")
                self.assertEqual(failure["plan_sha256"], "synthetic-plan")
                self.assertEqual(failure["error_type"], "RuntimeError")
                self.assertNotIn("result.json", before)
                with self.assertRaises(FileExistsError):
                    runner.fit_cell(plan, cell, "synthetic-plan")
                self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
