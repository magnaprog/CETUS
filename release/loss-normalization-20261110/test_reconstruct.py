"""Invented small populations; no training or scientific output access."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reconstruct as r


def hand_metrics(perfect):
    matrix = [[int(i == j) for j in range(6)] for i in range(6)]
    if not perfect:
        matrix[5] = [1, 0, 0, 0, 0, 0]
    return {
        "confusion_matrix": matrix,
        "support_per_class": [1] * 6,
        "predicted_per_class": [1] * 6 if perfect else [2, 1, 1, 1, 1, 0],
        "true_positive": [1] * 6 if perfect else [1, 1, 1, 1, 1, 0],
        "false_positive": [0] * 6 if perfect else [1, 0, 0, 0, 0, 0],
        "false_negative": [0] * 6 if perfect else [0, 0, 0, 0, 0, 1],
        "precision_per_class": [1.0] * 6 if perfect else [0.5, 1.0, 1.0, 1.0, 1.0, 0.0],
        "recall_per_class": [1.0] * 6 if perfect else [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "f1_per_class": [1.0] * 6 if perfect else [2 / 3, 1.0, 1.0, 1.0, 1.0, 0.0],
        "macro_f1": 1.0 if perfect else 7 / 9,
    }


def fixture():
    catalog = {"class_names": r.CLASSES, "tiles": [
        {"tile_id": f"r{f}c{c}", "label": c} for f in range(5) for c in range(6)]}
    splits = [{"policy": {"test_fold": f}, "assignments": {
        t["tile_id"]: "test" if int(t["tile_id"][1]) == f else
        "val" if int(t["tile_id"][1]) == (f + 1) % 5 else "train"
        for t in catalog["tiles"]}} for f in range(5)]
    rows_a, rows_b, answers = [], [], []
    qualified = r.COUNTS + r.SCORES + ["macro_f1"]
    for f in range(5):
        for s in range(3):
            perfect = s != 1
            ma, mb = hand_metrics(False), hand_metrics(perfect)
            histories = []
            for e in range(1, 101):
                histories.append({"history": {
                    "epoch": e, "examples": 18, "optimizer_steps": 1,
                    "finite_gradients": True, "lr": 0.0001,
                    "val_confusion_matrix": hand_metrics(True)["confusion_matrix"],
                    "val_macro_f1": 1.0, "loss_numerator": 3.0, "loss_denominator": 2.0,
                    "train_loss": 1.5, "val_loss_numerator": 2.0,
                    "val_loss_denominator": 4.0, "val_loss": 0.5,
                    "training_order_sha256": "1" * 64}, "source_epoch_sha256": "2" * 64})
            tests = []
            for m, is_perfect in ((ma, False), (mb, perfect)):
                preds = list(range(6)) if is_perfect else [0, 1, 2, 3, 4, 0]
                tests.append({"tile_ids": [f"r{f}c{c}" for c in range(6)],
                              "labels": list(range(6)), "predictions": preds,
                              "logits": [[1.0 if c == p else 0.0 for c in range(6)] for p in preds],
                              "logits_dtype": "float32", "class_order": list(range(6)),
                              **{k: m[k] for k in qualified}})
            validation = {k: hand_metrics(True)[k] for k in qualified}
            a = {"cell": ["final", f, "pretrained", s], "result_sha256": "3" * 64,
                 "selected_epoch": 1, "selected_score": 1.0,
                 "validation_history": copy.deepcopy(histories), "validation": validation,
                 "test": tests[0]}
            b = {"fold": f, "seed": s, "result_sha256": "4" * 64,
                 "control_result_sha256": "3" * 64, "specification_sha256": "5" * 64,
                 "selected_epoch": 1, "selected_score": 1.0, "selected_state_sha256": "6" * 64,
                 "validation_history": histories, "validation": validation, "test": tests[1]}
            rows_a.append(a)
            rows_b.append(b)
            answers.append({"fold": f, "seed": s, "craters_index": 5,
                            "selected_epochs": {"A": 1, "B": 1}, "A": ma, "B": mb,
                            "craters_f1_delta": 1.0 if perfect else 0.0,
                            "macro_f1_delta": 1 - 7 / 9 if perfect else 0.0,
                            "identities": {"A_result_sha256": "3" * 64,
                                           "B_result_sha256": "4" * 64,
                                           "B_specification_sha256": "5" * 64}})
    delta = {"craters_f1_delta": 2 / 3, "macro_f1_delta": 2 * (1 - 7 / 9) / 3}
    expected = {"schema_version": r.SCHEMA, "producer": r.B_PRODUCER,
                "independent_report_sha256": "7" * 64,
                "comparison": {"cells": answers,
                               "regions": [{"fold": f, "region": f + 1, **delta} for f in range(5)],
                               "equal_region_delta": delta}}
    data = {"schema_version": r.SCHEMA, "purpose": "synthetic_cpu_test", "class_order": r.CLASSES,
            "a_archive_sha256": r.A_ZIP, "producer": r.B_PRODUCER,
            "control_producer": r.A_PRODUCER, "model": "resnet18", "initialization": "pretrained",
            "endpoint": "validation_selected", "cells": rows_b}
    return data, {"schema_version": "model-followup-numeric-v1", "cells": rows_a}, catalog, splits, expected


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()
        self.data, self.baseline, self.catalog, self.splits, self.expected = self.inputs
        self.row = self.data["cells"][0]

    def run_fixture(self):
        return r.reconstruct(*self.inputs, real=False)

    def rejects(self, text):
        with self.assertRaisesRegex(ValueError, text):
            self.run_fixture()

    def test_hand_counted_complete_study(self):
        result = self.run_fixture()
        self.assertEqual(len(result["cells"]), 15)
        self.assertEqual(result["equal_region_delta"]["craters_f1_delta"], 2 / 3)

    def test_missing_cell(self):
        self.data["cells"].pop()
        self.rejects("Missing fold/seed")

    def test_duplicate_cell(self):
        self.data["cells"][-1] = self.data["cells"][0]
        self.rejects("duplicate fold/seed")

    def test_boolean_fold(self):
        self.row["fold"] = False
        self.rejects("fold/seed")

    def test_changed_control_binding(self):
        self.row["control_result_sha256"] = "8" * 64
        self.rejects("A result binding")

    def test_reordered_ids(self):
        self.row["test"]["tile_ids"].reverse()
        self.rejects("Ordered test IDs")

    def test_boolean_prediction(self):
        self.row["test"]["predictions"][0] = False
        self.rejects("Invalid class index")

    def test_float_label(self):
        self.row["test"]["labels"][0] = 0.0
        self.rejects("Invalid class index")

    def test_native_argmax_first_tie(self):
        self.row["test"]["logits"][0] = [1.0] * 6
        self.run_fixture()
        self.row["test"]["logits"][1] = [1.0] * 6
        self.rejects("argmax")

    def test_float64_only_logit(self):
        self.row["test"]["logits"][0][0] = 1.000000000000001
        self.rejects("float32")

    def test_nonfinite_logit(self):
        self.row["test"]["logits"][0][0] = float("nan")
        self.rejects("finite")

    def test_wrong_confusion_count(self):
        self.row["test"]["confusion_matrix"][0][0] = 2
        self.rejects("exact value differs")

    def test_incomplete_history(self):
        self.row["validation_history"].pop()
        self.rejects("Incomplete validation history")

    def test_exact_earliest_maximum(self):
        self.row["selected_epoch"] = 2
        self.rejects("Earliest unrounded")

    def test_near_tie_is_not_rounded(self):
        self.row["validation_history"][0]["history"]["val_macro_f1"] = math.nextafter(1.0, 0.0)
        self.rejects("Earliest unrounded")
        self.row["selected_epoch"] = 2
        self.expected["comparison"]["cells"][0]["selected_epochs"]["B"] = 2
        self.run_fixture()

    def test_learning_rate_sequence(self):
        self.row["validation_history"][10]["history"]["lr"] = 0.001
        self.rejects("learning rate")

    def test_loss_ratio(self):
        self.row["validation_history"][99]["history"]["train_loss"] = 1.500000000000001
        self.rejects("diagnostic ratio")

    def test_independent_answer_mismatch(self):
        self.expected["comparison"]["regions"][4]["craters_f1_delta"] = 0.7
        self.rejects("Independent targets")

    def test_independent_integer_fields_remain_exact(self):
        original = copy.deepcopy(self.expected)
        for field in ("selected_epoch", "count", "fold", "seed"):
            with self.subTest(field=field):
                self.expected.clear()
                self.expected.update(copy.deepcopy(original))
                row = self.expected["comparison"]["cells"][0]
                if field == "selected_epoch":
                    row["selected_epochs"]["B"] = math.nextafter(1.0, 2.0)
                elif field == "count":
                    row["B"]["true_positive"][0] = 1.0
                else:
                    row[field] = math.ulp(1.0)
                self.rejects("Independent targets")

    def test_unknown_record_field(self):
        self.row["machine_path"] = "/private"
        self.rejects("B record fields")

    def test_baseline_missing_cell(self):
        self.baseline["cells"].pop()
        self.rejects("Missing A cell")

    def test_synthetic_never_real(self):
        with self.assertRaisesRegex(ValueError, "Wrong data purpose"):
            r.reconstruct(*self.inputs)

    def test_duplicate_catalog_id(self):
        self.catalog["tiles"][-1]["tile_id"] = self.catalog["tiles"][0]["tile_id"]
        self.rejects("catalog IDs")


class FileTests(unittest.TestCase):
    def test_json_refuses_duplicate_and_nonfinite(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                r.decode(raw)

    def test_output_no_overwrite_and_no_empty_nonfinite_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                r.publish(path, {"bad": float("nan")})
            self.assertEqual(list(path.parent.iterdir()), [])
            r.publish(path, {"valid": True})
            with self.assertRaisesRegex(ValueError, "exists"):
                r.publish(path, {})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})

    def test_failed_link_preserves_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with patch.object(r.os, "link", side_effect=OSError("test publication failure")):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    r.publish(path, {"valid": True})
            self.assertFalse(path.exists())
            self.assertEqual(json.loads(path.with_name("result.json.partial").read_text()), {"valid": True})

    def test_package_inventory_hash_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = root / "data.json"
            member.write_bytes(b"{}")
            manifest = {"schema_version": r.SCHEMA, "files": {
                "data.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "SHA256SUMS").write_text("".join(
                r.sha(root / name) + "  " + name + "\n" for name in ("data.json", "manifest.json")))
            r.package(root)
            member.write_bytes(b"[]")
            with self.assertRaisesRegex(ValueError, "identity"):
                r.package(root)
            member.write_bytes(b"{}")
            (root / "extra").write_text("extra")
            with self.assertRaisesRegex(ValueError, "members"):
                r.package(root)
            (root / "extra").unlink()
            (root / "extra").symlink_to(member)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                r.package(root)


if __name__ == "__main__":
    unittest.main()
