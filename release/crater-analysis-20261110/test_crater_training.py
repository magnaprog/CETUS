"""Finite synthetic checks; these tests never open the scientific inputs."""

from copy import deepcopy
from fractions import Fraction
import io
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import analyze_crater_training as analysis


def fixture():
    tiles, test_folds = [], []
    for fold, craters in enumerate([8, 1, 14, 61, 48]):
        for label in list(range(5)) + [5] * craters:
            tiles.append({"tile_id": f"image_{len(tiles):04d}", "label": label})
            test_folds.append(fold)
    ids = np.asarray([t["tile_id"] for t in tiles])
    labels = np.asarray([t["label"] for t in tiles])
    roles = np.asarray([[2 if f == tf else 1 if (f+1) % 5 == tf else 0
                         for f in range(5)] for tf in test_folds])
    counts = np.zeros((len(ids), 4, 6), dtype=np.int64)
    for i, label in enumerate(labels):
        counts[i, :, label] = 4
    census = {"tile_ids": ids, "modal_labels": labels, "roles": roles,
              "counts": counts, "unknown": np.zeros((len(ids), 4), dtype=np.int64)}
    # Catalog order deliberately differs from NPZ order.
    catalog = {"class_names": analysis.CLASSES.copy(), "tiles": list(reversed(tiles))}
    splits, fold_counts, cells = [], [], []
    for fold in range(5):
        assignments = {t: analysis.ROLES[int(r)] for t, r in zip(ids, roles[:, fold])}
        splits.append({"policy": {"test_fold": fold}, "assignments": assignments})
        fold_counts.append({name: {"total": int((roles[:, fold] == role).sum()),
                                  "class_counts": np.bincount(
                                      labels[roles[:, fold] == role], minlength=6).tolist()}
                            for role, name in enumerate(analysis.ROLES[:3])})
        test_tiles = [t for t in catalog["tiles"] if assignments[t["tile_id"]] == "test"]
        test = {"tile_ids": [t["tile_id"] for t in test_tiles],
                "labels": [t["label"] for t in test_tiles],
                "predictions": [t["label"] for t in test_tiles], "class_order": list(range(6))}
        for kind, arm, seed in [("dino_c1", "pretrained", 42)] + [
                ("final", arm, seed) for arm in ("scratch", "pretrained") for seed in range(3)]:
            cells.append({"cell": [kind, fold, arm, seed], "identity": {"initialization": arm},
                          "inputs_sha256": str(fold) * 64, "test": deepcopy(test), "seed": seed,
                          "recipe": {"batch_size": 32}, "training_class_weights": [1.] * 6})
    records = {"class_order": analysis.CLASSES.copy(), "cells": cells, "fold_counts": fold_counts}
    metadata = {
        "schema_version": 1,
        "original_results_sha256": "bb38930b99034148b53f4376bac4e465fe7d5d1f6a1a36a60279b13e498682d6",
        "map_cells_npz_sha256": analysis.INPUT_SHA256["map_census"],
        "array_conventions": {"class_order": analysis.CLASSES.copy(),
                              "mode_order": analysis.MODES.copy(), "role_order": analysis.ROLES.copy()},
    }
    return records, catalog, splits, census, metadata


class CraterTrainingTests(unittest.TestCase):
    def test_exact_small_hypergeometric_cases_with_partial_batches(self):
        for n in range(1, 9):
            for k in range(n + 1):
                for b in range(1, n + 2):
                    observed = [sum(bool(set(chosen).intersection(range(j, min(j+b, n))))
                                    for j in range(0, n, b))
                                for chosen in combinations(range(n), k)]
                    expectation = Fraction(sum(observed), len(observed))
                    batches = (n + b - 1) // b
                    bound = Fraction(max(observed), batches)
                    result = analysis.batch_availability(n, k, b)
                    self.assertEqual(result["expected_positive_batches_uniform_shuffle"], float(expectation))
                    self.assertEqual(result["expected_positive_batch_fraction_uniform_shuffle"],
                                     float(expectation / batches))
                    self.assertEqual(result["bound_fraction"], [bound.numerator, bound.denominator])
                    self.assertEqual(result["max_positive_batches"], max(observed))
                    self.assertEqual(result["last_batch_size"], (n - 1) % b + 1)

    def test_invalid_batch_counts_and_labels(self):
        for args in [(0, 0, 1), (3, 4, 2), (3, -1, 2), (3, 1, 0), (True, 1, 1), (3, 1., 2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                analysis.batch_availability(*args)
        for labels in [[True], [1.], ["1"], [6], [-1], []]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                analysis.integer_labels(labels, 1)

    def test_all_three_fixed_hashes_reject_changed_bytes_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad"
            path.write_bytes(b"not a zip, NPZ or JSON")
            for name in analysis.INPUT_SHA256:
                bindings = {}
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA256"):
                    analysis.read_bound(path, name, bindings)
                self.assertEqual(bindings, {})
            output = Path(directory) / "output"
            with patch.object(analysis.json, "loads") as decode, self.assertRaisesRegex(ValueError, "SHA256"):
                analysis.main(["--initialization-zip", str(path), "--map-census", str(path),
                               "--census-metadata", str(path), "--output", str(output)])
            decode.assert_not_called()
            self.assertFalse(output.exists())

    def test_selected_grid_is_complete_unique_and_ordered(self):
        records, *_ = fixture()
        records["cells"].append({"cell": ["dino_c1", 0, "scratch", 0]})
        records["cells"].append({"cell": ["tuning", 0, "scratch", .001]})
        cells = analysis.select_cells(records)
        self.assertEqual(len(cells), 35)
        self.assertEqual(cells[0]["cell"], ["dino_c1", 0, "pretrained", 42])
        self.assertEqual(cells[-1]["cell"], ["final", 4, "scratch", 2])
        for defect in ["missing", "extra_duplicate", "replacement_duplicate", "fold", "seed", "bool", "classes"]:
            bad = deepcopy(records)
            if defect == "missing":
                bad["cells"].pop(0)
            elif defect == "extra_duplicate":
                bad["cells"].append(deepcopy(bad["cells"][0]))
            elif defect == "replacement_duplicate":
                bad["cells"][1] = deepcopy(bad["cells"][0])
            elif defect == "fold":
                bad["cells"][0]["cell"][1] = 5
            elif defect == "seed":
                bad["cells"][0]["cell"][3] = 43
            elif defect == "bool":
                bad["cells"][0]["cell"][1] = False
            else:
                bad["class_order"].reverse()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.select_cells(bad)

    def test_census_axes_shapes_types_and_unique_ids(self):
        *_, census, metadata = fixture()
        analysis.validate_census(metadata, census)
        for defect in ["mode", "class", "role", "duplicate", "shape", "dtype", "negative", "modal", "repeat_test"]:
            c, m = deepcopy(census), deepcopy(metadata)
            if defect in ("mode", "class", "role"):
                m["array_conventions"][defect + "_order"].reverse()
            elif defect == "duplicate":
                c["tile_ids"][1] = c["tile_ids"][0]
            elif defect == "shape":
                c["counts"] = c["counts"][:, :3]
            elif defect == "dtype":
                c["counts"] = c["counts"].astype(float)
            elif defect == "negative":
                c["unknown"][0, 0] = -1
            elif defect == "modal":
                c["modal_labels"][0] = 6
            else:
                c["roles"][0, :] = 2
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.validate_census(m, c)

    def test_public_metadata_schema_and_lineage_admission(self):
        *_, census, metadata = fixture()
        analysis.validate_census(metadata, census)
        for defect in ["version", "boolean_version", "npz_lineage", "original_lineage", "private_schema"]:
            bad = deepcopy(metadata)
            if defect == "version":
                bad["schema_version"] = 2
            elif defect == "boolean_version":
                bad["schema_version"] = True
            elif defect == "npz_lineage":
                bad["map_cells_npz_sha256"] = "0" * 64
            elif defect == "original_lineage":
                bad["original_results_sha256"] = "0" * 64
            else:
                bad["arrays"] = bad.pop("array_conventions")
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.validate_census(bad, census)

    def test_catalog_and_every_fold_are_bound_to_census(self):
        records, catalog, splits, census, _ = fixture()
        ordered = analysis.fold_inputs(records, catalog, splits, census)
        for fold in range(5):
            self.assertEqual(ordered[fold]["test"]["tile_ids"], records["cells"][fold * 7]["test"]["tile_ids"])
        for defect in ["count", "total", "catalog_label", "catalog_id", "catalog_class", "split_id", "split_fold", "split_role"]:
            r, c, s, n = deepcopy((records, catalog, splits, census))
            if defect == "count":
                r["fold_counts"][0]["train"]["class_counts"][0] += 1
            elif defect == "total":
                r["fold_counts"][0]["train"]["total"] += 1
            elif defect == "catalog_label":
                c["tiles"][0]["label"] = 0
            elif defect == "catalog_id":
                c["tiles"][0]["tile_id"] = "unknown"
            elif defect == "catalog_class":
                c["class_names"].reverse()
            elif defect == "split_id":
                del s[0]["assignments"][c["tiles"][0]["tile_id"]]
            elif defect == "split_fold":
                s[0]["policy"]["test_fold"] = 1
            else:
                s[0]["assignments"][c["tiles"][0]["tile_id"]] = "selk_holdout"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.fold_inputs(r, c, s, n)

    def test_test_join_uses_ids_and_rejects_inconsistent_order_labels_or_roles(self):
        records, catalog, splits, census, _ = fixture()
        ordered = analysis.fold_inputs(records, catalog, splits, census)[0]["test"]
        test = records["cells"][0]["test"]
        indices, truth, prediction = analysis.join_test(test, ordered, census, 0)
        self.assertEqual(census["tile_ids"][indices].tolist(), ordered["tile_ids"])
        self.assertEqual(truth.tolist(), prediction.tolist())
        for defect in ["order", "duplicate", "unknown", "label", "modal", "role", "prediction", "class_order"]:
            t, o, c = deepcopy((test, ordered, census))
            if defect == "order":
                t["tile_ids"].reverse()
            elif defect == "duplicate":
                t["tile_ids"][0] = o["tile_ids"][0] = t["tile_ids"][1]
            elif defect == "unknown":
                t["tile_ids"][0] = o["tile_ids"][0] = "unknown"
            elif defect == "label":
                t["labels"][0] = 0
            elif defect == "modal":
                t["labels"][0] = o["labels"][0] = 0
            elif defect == "role":
                c["roles"][indices[0], 0] = 0
            elif defect == "prediction":
                t["predictions"][0] = True
            else:
                t["class_order"].reverse()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.join_test(t, o, c, 0)

    def test_map_rows_keep_unknown_empty_domains_and_endpoint_errors_separate(self):
        *_, census, _ = fixture()
        census["counts"][:4] = 0
        census["counts"][0, 2, [0, 5]] = [2, 2]
        census["counts"][1, 2, 5] = 4
        census["counts"][3, 2, 5] = 3
        census["unknown"][:4, 2] = [1, 0, 4, 1]
        rows, summary = analysis.map_rows("model", 0, 7, "all_four", range(4), [0, 5, 0, 0], census)
        self.assertEqual([r["craters_fraction"] for r in rows], [.5, 1., None, 1.])
        self.assertEqual([r["predicted_class_fraction"] for r in rows], [.5, 1., None, 0.])
        self.assertEqual([r["all_known_cells_craters"] for r in rows], [False, True, False, True])
        self.assertEqual([r["unknown_cells"] for r in rows], [1, 0, 4, 1])
        self.assertEqual(summary, {"model": "model", "fold": 0, "seed": 7, "mode": "all_four",
                                  "craters_test_images": 4, "correct": 1, "undefined_map_fraction": 1,
                                  "errors": 3, "errors_with_predicted_class_present": 1,
                                  "all_known_cells_craters": 2, "correct_all_known_cells_craters": 1})

    def test_synthetic_all_35_cells_preserve_per_seed_and_per_mode_rows(self):
        records, catalog, splits, census, _ = fixture()
        result = analysis.analyze(records, catalog, splits, census)
        self.assertEqual([len(result[k]) for k in ("training_batch_availability", "map_prediction_cohorts", "image_rows")],
                         [30, 70, 1848])
        self.assertEqual(len({(r["model"], r["fold"], r["seed"], r["mode"], r["tile_id"])
                              for r in result["image_rows"]}), 1848)
        self.assertTrue(all(r["correct"] for r in result["image_rows"]))
        self.assertTrue(all(r["errors"] == 0 for r in result["map_prediction_cohorts"]))
        # Fold 0 has 138 training images with 123 Craters; batch sizes are 32,32,32,32,10.
        crater = result["training_batch_availability"][5]
        train = records["fold_counts"][0]["train"]
        self.assertEqual(crater["images"], train["total"])
        self.assertEqual(crater["class_images"], train["class_counts"][5])

    def test_cnn_binding_batch_and_weights_must_agree(self):
        records, catalog, splits, census, _ = fixture()
        for defect in ["binding", "batch", "seed", "weights", "bad_weight", "initialization"]:
            r = deepcopy(records)
            cell = r["cells"][2]  # Same fold as other selected cells.
            if defect == "binding":
                cell["inputs_sha256"] = "x" * 64
            elif defect == "batch":
                cell["recipe"]["batch_size"] = 64
            elif defect == "seed":
                cell["seed"] = 9
            elif defect == "weights":
                cell["training_class_weights"][0] = 2.
            elif defect == "bad_weight":
                cell["training_class_weights"][0] = float("nan")
            else:
                cell["identity"]["initialization"] = "other"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                analysis.analyze(r, catalog, splits, census)

    def test_cli_synthetic_wiring_and_portable_input_receipt(self):
        records, catalog, splits, census, _ = fixture()
        bindings = {k: {"sha256": v, "bytes": 123} for k, v in analysis.INPUT_SHA256.items()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.object(analysis, "load_inputs", return_value=(records, catalog, splits, census, bindings)), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout:
                analysis.main(["--initialization-zip", "inputs.zip", "--map-census", "map.npz",
                               "--census-metadata", "metadata.json", "--output", str(output)])
            result = json.loads((output / "results.json").read_bytes())
            self.assertEqual(result["inputs"], bindings)
            self.assertEqual(json.loads(stdout.getvalue())["image_rows"], 1848)
            self.assertEqual(len(result["source_sha256"]), 64)
            self.assertNotIn(str(output), json.dumps(result))

    def test_terminal_serialization_failure_cannot_publish_output(self):
        records, catalog, splits, census, _ = fixture()
        original_dumps = json.dumps

        def fail_terminal(value, *args, **kwargs):
            if "cells" in value:
                raise TypeError("terminal serialization failed")
            return original_dumps(value, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with patch.object(analysis, "load_inputs", return_value=(records, catalog, splits, census, {})), \
                    patch.object(analysis.json, "dumps", side_effect=fail_terminal), \
                    self.assertRaisesRegex(TypeError, "terminal serialization failed"):
                analysis.main(["--initialization-zip", "inputs.zip", "--map-census", "map.npz",
                               "--census-metadata", "metadata.json", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_atomic_new_output_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fresh"
            analysis.write_output(output, {"status": "complete"})
            saved = (output / "results.json").read_bytes()
            self.assertEqual(json.loads(saved), {"status": "complete"})
            self.assertEqual([p.name for p in output.iterdir()], ["results.json"])
            existing_dir, existing_file, dangling = root / "empty", root / "file", root / "link"
            existing_dir.mkdir()
            existing_file.write_text("keep")
            dangling.symlink_to(root / "absent")
            for existing in [output, existing_dir, existing_file, dangling]:
                with self.subTest(existing=existing.name), self.assertRaises(FileExistsError):
                    analysis.write_output(existing, {"changed": True})
            self.assertEqual((output / "results.json").read_bytes(), saved)
            self.assertEqual(existing_file.read_text(), "keep")
            with self.assertRaises(ValueError):
                analysis.write_output(root / "nan", {"value": float("nan")})
            self.assertFalse((root / "nan").exists())
            interrupted = root / "interrupted"
            with patch.object(Path, "rename", side_effect=OSError("interrupted")), self.assertRaises(OSError):
                analysis.write_output(interrupted, {"status": "complete"})
            self.assertFalse((interrupted / "results.json").exists())
            self.assertTrue((interrupted / "results.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
