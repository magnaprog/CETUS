"""Synthetic package/comparison checks; never open the scientific input ZIP."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import reconstruct as reader


class ReconstructionTests(unittest.TestCase):
    def result(self):
        return {"metrics": [{"ap": 0.5}] * 360, "craters_curve_index": [{}] * 60,
                "summaries": {"regional": [{}] * 140, "contrasts": [{}] * 70,
                              "geographic": [{}] * 42}, "execution": {"synthetic": True}}

    def arrays(self):
        return {f"cell_{i:02d}_{key}": np.array([0, 1], dtype="int64" if key in ("tp", "fp") else "float64")
                for i in range(60) for key in ("thresholds", "tp", "fp", "precision", "recall")}

    def test_exact_results_and_only_execution_metadata_may_differ(self):
        a, b = self.result(), self.result()
        b["execution"] = {"different_runtime_receipt": True}
        self.assertEqual(reader.compare_results(a, b, self.arrays(), self.arrays())["status"],
                         "reconstruction_verified")
        b["metrics"][0] = {"ap": 0.6}
        with self.assertRaisesRegex(ValueError, "Scientific result differs"):
            reader.compare_results(a, b, self.arrays(), self.arrays())

    def test_aggregation_tamper_is_rejected(self):
        a, b = self.result(), self.result()
        b["summaries"]["contrasts"][0] = {"difference": 0.1}
        with self.assertRaisesRegex(ValueError, "Scientific result differs"):
            reader.compare_results(a, b, self.arrays(), self.arrays())

    def test_missing_metric_is_rejected(self):
        a, b = self.result(), self.result()
        b["metrics"].pop()
        with self.assertRaisesRegex(ValueError, "Result grid differs"):
            reader.compare_results(a, b, self.arrays(), self.arrays())

    def test_curve_value_dtype_shape_and_inventory_are_checked(self):
        a = self.arrays()
        for fault in ("value", "dtype", "shape", "missing"):
            with self.subTest(fault=fault):
                b = {k: v.copy() for k, v in a.items()}
                key = "cell_59_precision"
                if fault == "value":
                    b[key][0] = 0.01
                elif fault == "dtype":
                    b[key] = b[key].astype("float32")
                elif fault == "shape":
                    b[key] = b[key].reshape(1, 2)
                else:
                    b.pop(key)
                with self.assertRaisesRegex(ValueError, "Curve array"):
                    reader.compare_results(self.result(), self.result(), a, b)

    def test_file_inventory_hash_and_checksum_list_are_checked(self):
        for fault in (None, "tamper", "missing", "extra", "symlink", "checksums"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = b"inert fixture"
                bound = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                (root / "payload.txt").write_bytes(payload)
                (root / "manifest.json").write_text(json.dumps(
                    {"schema": "class-ranking-package-v1", "files": {"payload.txt": bound}}))
                sums = bound["sha256"] + "  payload.txt\n" + reader.sha(root / "manifest.json") + "  manifest.json\n"
                (root / "SHA256SUMS").write_text(sums)
                if fault == "tamper":
                    (root / "payload.txt").write_bytes(b"changed data")
                elif fault == "missing":
                    (root / "payload.txt").unlink()
                elif fault == "extra":
                    (root / "extra.txt").write_text("extra")
                elif fault == "symlink":
                    (root / "payload.txt").unlink()
                    (root / "payload.txt").symlink_to(root / "manifest.json")
                elif fault == "checksums":
                    (root / "SHA256SUMS").write_text(sums.replace(bound["sha256"], "0" * 64))
                if fault is None:
                    reader.check_package(root)
                else:
                    with self.assertRaises(ValueError):
                        reader.check_package(root)


if __name__ == "__main__":
    unittest.main()
