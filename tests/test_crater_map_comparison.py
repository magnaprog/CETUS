"""Guard and saved-input checks for the standalone primary-LR Craters adapter.

Saved-input tests use the two files included in the public release. To check
copies elsewhere, set both CETUS_CRATER_RESULTS and CETUS_PRIMARY_PREDICTIONS.
The tests verify the file hashes before exercising the join.
"""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import summarize_crater_map_comparison as adapter


class InputAndOutputTests(unittest.TestCase):
    def test_bad_hashes_fail_before_either_json_is_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad"
            bad.write_bytes(b"not JSON or gzip")
            with patch.object(adapter.json, "loads") as decode:
                with self.assertRaisesRegex(ValueError, "SHA256.*crater_results"):
                    adapter.main(["--crater-results", str(bad), "--predictions", str(bad),
                                  "--output-dir", str(root / "output")])
            decode.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_output_never_overwrites_directory_file_or_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            marker = existing / "keep"
            marker.write_text("unchanged")
            file = root / "file"
            file.write_text("unchanged")
            dangling = root / "dangling"
            dangling.symlink_to(root / "absent")
            for output in (existing, file, dangling):
                with self.subTest(path=output.name), self.assertRaises(FileExistsError):
                    adapter.write_outputs(output, {"rows": []})
                with self.subTest(cli_path=output.name), self.assertRaisesRegex(ValueError, "must be new"):
                    adapter.main(["--crater-results", "unread", "--predictions", "unread",
                                  "--output-dir", str(output)])
            self.assertEqual(marker.read_text(), "unchanged")
            self.assertEqual(file.read_text(), "unchanged")
            self.assertTrue(dangling.is_symlink())
            with self.assertRaises(ValueError):
                adapter.write_outputs(root / "invalid", {"rows": [], "invalid": float("nan")})
            self.assertFalse((root / "invalid").exists())

    def test_second_file_failure_rolls_back_first_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            original_open = Path.open

            def fail_second(path, *args, **kwargs):
                if path.suffix == ".tex":
                    raise OSError("second file failed")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", fail_second):
                with self.assertRaisesRegex(OSError, "second file failed"):
                    adapter.write_outputs(output, {"rows": []})
            self.assertFalse(output.exists())
            adapter.write_outputs(output, {"rows": []})
            self.assertTrue((output / "crater-map-primary-lr.tex").is_file())

    def test_rollback_keeps_file_added_by_another_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            original_open = Path.open

            def populate_second(path, *args, **kwargs):
                if path.suffix == ".tex":
                    with original_open(path, "x") as other:
                        other.write("keep")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", populate_second):
                with self.assertRaises(FileExistsError):
                    adapter.write_outputs(output, {"rows": []})
            self.assertFalse((output / "results.json").exists())
            self.assertEqual((output / "crater-map-primary-lr.tex").read_text(), "keep")


class SavedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [os.environ.get(name) for name in ("CETUS_CRATER_RESULTS", "CETUS_PRIMARY_PREDICTIONS")]
        if not any(paths):
            experiments = Path(__file__).resolve().parents[1] / "experiments"
            paths = [experiments / "crater-training/results.json",
                     experiments / "regional-predictions/results/predictions.json.gz"]
        elif not all(paths):
            raise ValueError("Both saved-input fixture paths are required")
        cls.paths = [Path(p) for p in paths]
        cls.crater, cls.predictions, cls.bindings = adapter.load_inputs(*cls.paths)

    def test_exact_saved_counts_all_eight_families(self):
        result = adapter.summarize(self.crater, self.predictions)
        # Independently admitted counts: R4, R5, /39, TP/132, FP/23114, all predictions.
        expected = {
            "dinov2": (0, 0, 0, 2, 167, 169),
            "dofa": (1, 0, 1, 5, 278, 283),
            "croma": (2, 2, 4, 7, 477, 484),
            "random_init": (3, 1, 4, 22, 1517, 1539),
            "raw_intensity_statistics": (7, 6, 13, 48, 1958, 2006),
            "intensity_histogram": (5, 8, 13, 28, 1818, 1846),
            "texture_glcm_lbp_gabor_hog_wavelet": (0, 1, 1, 2, 303, 305),
            "combined_classical": (0, 2, 2, 5, 127, 132),
        }
        self.assertEqual([r["family"] for r in result["rows"]], list(expected))
        for row in result["rows"]:
            p = row["pooled"]
            self.assertEqual((row["region_4"]["subset_correct"], row["region_5"]["subset_correct"],
                              p["subset_correct"], p["true_positives"], p["false_positives"],
                              p["craters_predictions"]), expected[row["family"]])
        self.assertEqual(result["population"], {"test_tiles": 23246, "modal_craters": 132,
                                                "non_craters": 23114, "subset": 39})
        self.assertEqual([r["subset"] for r in result["regions"]], [0, 0, 0, 23, 16])

    def test_map_row_order_is_not_used_as_a_join_key(self):
        crater = deepcopy(self.crater)
        crater["image_rows"].reverse()
        crater["map_prediction_cohorts"].reverse()
        self.assertEqual(adapter.summarize(crater, self.predictions),
                         adapter.summarize(self.crater, self.predictions))

    def test_schema_order_ids_labels_and_complete_family_arrays(self):
        def replace_crater_label(fold):
            i = fold["true_labels"].index(5)
            j = fold["true_labels"].index(0)
            fold["true_labels"][i], fold["true_labels"][j] = 0, 5

        defects = {
            "version": lambda p: p.update(schema_version="2"),
            "classes": lambda p: p["class_names"].reverse(),
            "fold_missing": lambda p: p["folds"].pop(),
            "fold_order": lambda p: p["folds"].reverse(),
            "fold_bool": lambda p: p["folds"][0].update(fold=False),
            "id_missing": lambda p: p["folds"][0]["test_ids"].pop(),
            "id_order": lambda p: p["folds"][0]["test_ids"].reverse(),
            "id_duplicate": lambda p: p["folds"][0]["test_ids"].__setitem__(0, p["folds"][0]["test_ids"][1]),
            "id_type": lambda p: p["folds"][0]["test_ids"].__setitem__(0, 5),
            "truth_domain": lambda p: p["folds"][0]["true_labels"].__setitem__(0, 6),
            "truth_bool": lambda p: p["folds"][0]["true_labels"].__setitem__(0, True),
            "truth_join": lambda p: replace_crater_label(p["folds"][0]),
            "family_missing": lambda p: p["folds"][0]["common"].pop("dofa"),
            "family_extra": lambda p: p["folds"][0]["classical"].update(extra=[]),
            "array_short": lambda p: p["folds"][0]["common"]["croma"].pop(),
            "array_bool": lambda p: p["folds"][0]["common"]["dinov2"].__setitem__(0, True),
            "array_float": lambda p: p["folds"][0]["common"]["random_init"].__setitem__(0, 0.0),
        }
        for name, mutate in defects.items():
            with self.subTest(defect=name):
                predictions = deepcopy(self.predictions)
                mutate(predictions)
                with self.assertRaises(ValueError):
                    adapter.summarize(self.crater, predictions)
        predictions = deepcopy(self.predictions)
        predictions["folds"][1]["test_ids"][0] = predictions["folds"][0]["test_ids"][0]
        predictions["folds"][1]["test_ids"].sort()
        with self.assertRaisesRegex(ValueError, "overlap"):
            adapter.summarize(self.crater, predictions)

    def test_map_schema_denominators_flags_and_complete_cohorts(self):
        defects = {
            "classes": lambda c: c["class_order"].reverse(),
            "source": lambda c: c.update(source_sha256="0" * 64),
            "row_missing": lambda c: c["image_rows"].pop(),
            "row_duplicate": lambda c: c["image_rows"].__setitem__(1, deepcopy(c["image_rows"][0])),
            "unknown_id": lambda c: c["image_rows"][0].update(tile_id="titan_unknown"),
            "seed": lambda c: c["image_rows"][0].update(seed=0),
            "mode": lambda c: c["image_rows"][0].update(mode="weighted"),
            "negative": lambda c: c["image_rows"][0]["class_counts"].__setitem__(0, -1),
            "float": lambda c: c["image_rows"][0]["class_counts"].__setitem__(0, 1152.0),
            "known": lambda c: c["image_rows"][0].update(known_cells=4095),
            "unknown": lambda c: c["image_rows"][0].update(unknown_cells=-1),
            "fraction": lambda c: c["image_rows"][0].update(craters_fraction=1.0),
            "flag": lambda c: c["image_rows"][0].update(all_known_cells_craters=True),
            "correct": lambda c: c["image_rows"][0].update(correct=True),
            "cohort_missing": lambda c: c["map_prediction_cohorts"].pop(),
            "cohort_duplicate": lambda c: c["map_prediction_cohorts"].__setitem__(1, deepcopy(c["map_prediction_cohorts"][0])),
            "cohort_count": lambda c: c["map_prediction_cohorts"][0].update(craters_test_images=9),
            "cohort_flag_count": lambda c: c["map_prediction_cohorts"][0].update(all_known_cells_craters=1),
            "cohort_bool": lambda c: c["map_prediction_cohorts"][0].update(undefined_map_fraction=False),
        }
        for name, mutate in defects.items():
            with self.subTest(defect=name):
                crater = deepcopy(self.crater)
                mutate(crater)
                with self.assertRaises(ValueError):
                    adapter.summarize(crater, self.predictions)

    def test_both_modes_and_subset_denominator_are_derived_from_counts(self):
        eligible = next(r for r in self.crater["image_rows"] if r["all_known_cells_craters"])
        tid = eligible["tile_id"]
        for modes, message in [({"all_four"}, "different subset IDs"),
                               ({"baseline", "all_four"}, "Single-map-class denominator")]:
            crater = deepcopy(self.crater)
            # Change one tile in every seed/model cohort consistently, including
            # reported fractions/flags and aggregate summaries. Raw counts must
            # still cause the endpoint/mode check to fail.
            for row in crater["image_rows"]:
                if row["tile_id"] != tid or row["mode"] not in modes:
                    continue
                row["class_counts"][0] = 1
                row["class_counts"][5] -= 1
                row["all_known_cells_craters"] = False
                row["craters_fraction"] = row["class_counts"][5] / row["known_cells"]
                row["predicted_class_fraction"] = row["class_counts"][row["prediction"]] / row["known_cells"]
                cohort = next(c for c in crater["map_prediction_cohorts"]
                              if all(c[k] == row[k] for k in ("model", "fold", "seed", "mode")))
                cohort["all_known_cells_craters"] -= 1
                cohort["correct_all_known_cells_craters"] -= int(row["prediction"] == 5)
                cohort["errors_with_predicted_class_present"] += int(row["prediction"] == 0)
            with self.subTest(modes=modes), self.assertRaisesRegex(ValueError, message):
                adapter.summarize(crater, self.predictions)

    def test_empty_known_cells_cannot_qualify_and_repeated_maps_must_agree(self):
        crater = deepcopy(self.crater)
        row = next(r for r in crater["image_rows"] if r["all_known_cells_craters"])
        row.update(class_counts=[0] * 6, known_cells=0, unknown_cells=4096,
                   craters_fraction=None, predicted_class_fraction=None)
        with self.assertRaisesRegex(ValueError, "eligibility"):
            adapter.summarize(crater, self.predictions)

        crater = deepcopy(self.crater)
        row = next(r for r in crater["image_rows"]
                   if r["model"] == "resnet18_pretrained" and r["seed"] == 0)
        row["class_counts"][0] += 1
        row["class_counts"][5] -= 1
        row["craters_fraction"] = row["class_counts"][5] / row["known_cells"]
        row["predicted_class_fraction"] = row["class_counts"][row["prediction"]] / row["known_cells"]
        with self.assertRaisesRegex(ValueError, "differ across model/seed"):
            adapter.summarize(crater, self.predictions)

        crater = deepcopy(self.crater)
        row = next(r for r in crater["image_rows"] if r["mode"] == "all_four")
        row["prediction"] = (row["prediction"] + 1) % 6
        row["correct"] = row["prediction"] == 5
        row["predicted_class_fraction"] = row["class_counts"][row["prediction"]] / row["known_cells"]
        with self.assertRaisesRegex(ValueError, "prediction differs between map modes"):
            adapter.summarize(crater, self.predictions)

    def test_primary_hash_rejection_and_authenticated_bytes_not_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "predictions.gz"
            bad.write_bytes(b"bad gzip")
            with patch.object(adapter.json, "loads") as decode:
                with self.assertRaisesRegex(ValueError, "SHA256.*predictions"):
                    adapter.load_inputs(self.paths[0], bad)
            decode.assert_not_called()
        original = Path.read_bytes
        reads = []

        def read_once(path):
            self.assertNotIn(path, reads)
            reads.append(path)
            return original(path)

        with patch.object(Path, "read_bytes", read_once):
            crater, predictions, bindings = adapter.load_inputs(*self.paths)
        self.assertEqual(reads, self.paths)
        self.assertEqual((crater, predictions, bindings), (self.crater, self.predictions, self.bindings))

    def test_cli_deterministic_json_and_eight_row_tex(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / name for name in ("first", "second")]
            for output in outputs:
                with patch("sys.stdout", new_callable=io.StringIO):
                    adapter.main(["--crater-results", str(self.paths[0]), "--predictions", str(self.paths[1]),
                                  "--output-dir", str(output)])
            for name in ("results.json", "crater-map-primary-lr.tex"):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())
            result = json.loads((outputs[0] / "results.json").read_text())
            self.assertEqual(result["inputs"], self.bindings)
            tex = (outputs[0] / "crater-map-primary-lr.tex").read_text()
            for heading in (r"\shortstack{Region 4\\ /23}", r"\shortstack{Region 5\\ /16}",
                            r"\shortstack{True positives\\ /132}", r"\shortstack{False positives\\ /23,114}"):
                self.assertIn(heading, tex)
            self.assertEqual(sum(" & " in line for line in tex.splitlines()), 9)
            self.assertIn("DINOv2 (primary extraction) & 0 & 0 & 0 & 2 & 167", tex)
            self.assertNotIn(str(directory), json.dumps(result))
            with self.assertRaisesRegex(ValueError, "must be new"):
                adapter.main(["--crater-results", str(self.paths[0]), "--predictions", str(self.paths[1]),
                              "--output-dir", str(outputs[0])])


if __name__ == "__main__":
    unittest.main()
