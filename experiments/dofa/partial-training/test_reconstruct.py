"""Tiny invented records and an independent Fraction oracle; no scientific data."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import reconstruct as app

WARNING = "synthetic-only-not-scientific-acceptance"
CLASS_ORDER = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
FILES = {"README.md", "LICENSE", "NOTICE.md", "reconstruct.py", "test_reconstruct.py",
         "records.json.gz", "expected.json", "provenance.json"}


def raw_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def bound(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tree(flat):
    return {"macro_f1": flat["macro_f1"], "balanced_accuracy": flat["balanced_accuracy"],
            "per_class": {k: [flat[f"{k}{c}"] for c in range(6)] for k in ("precision", "recall", "f1")}}


def rational_cell(labels, predictions):
    """Count pairs independently, then use exact rational ratios."""
    counts = Counter(zip(labels, predictions))
    truth, predicted = Counter(labels), Counter(predictions)
    flat = {}
    for c in range(6):
        tp = counts[c, c]
        flat[f"precision{c}"] = Q(tp, predicted[c]) if predicted[c] else Q(0)
        flat[f"recall{c}"] = Q(tp, truth[c]) if truth[c] else Q(0)
        flat[f"f1{c}"] = Q(2*tp, truth[c]+predicted[c]) if truth[c]+predicted[c] else Q(0)
    flat["macro_f1"] = sum(flat[f"f1{c}"] for c in range(6))/6
    flat["balanced_accuracy"] = sum(flat[f"recall{c}"] for c in range(6))/6
    return flat, {"confusion_counts": [[counts[r, c] for c in range(6)] for r in range(6)],
                  "support": [truth[c] for c in range(6)], "prediction_counts": [predicted[c] for c in range(6)]}


def independent_expected(records):
    """No reader arithmetic/canonicalizer: exact rational reduction of flat keys."""
    rational, cells, fixed = [], [], []
    for fold in records["folds"]:
        digest = hashlib.sha256(raw_json({"tile_ids": fold["tile_ids"], "labels": fold["labels"]})).hexdigest()
        fixed.append({"fold": fold["fold"], "region": fold["region"], "rows": len(fold["labels"]),
                      "ordered_truth_sha256": digest})
    for record in records["cells"]:
        scores, counts = rational_cell(records["folds"][record["fold"]]["labels"], record["predictions"])
        rational.append(scores)
        cells.append({k: record[k] for k in ("cell", "fold", "seed", "band")}
                     | counts | {"scores": tree({k: float(v) for k, v in scores.items()})})
    keys = list(rational[0])
    pairs, paired_values, fold_rows, fold_values, paired_folds = [], [], [], {}, []
    for fold in range(5):
        for seed in range(5):
            i = 10*fold+2*seed
            values = {k: rational[i+1][k]-rational[i][k] for k in keys}
            paired_values.append(values)
            pairs.append({"fold": fold, "seed": seed, "cells": [i, i+1],
                          "difference": tree({k: float(v) for k, v in values.items()})})
        for arm, band in enumerate([13.78, 3.75]):
            ids = [10*fold+2*s+arm for s in range(5)]
            values = {k: sum(rational[i][k] for i in ids)/5 for k in keys}
            fold_values[fold, arm] = values
            fold_rows.append({"fold": fold, "band": band, "cells": ids, "seeds": [0,1,2,3,4],
                              "scores": tree({k: float(v) for k, v in values.items()})})
        values = {k: sum(paired_values[5*fold+s][k] for s in range(5))/5 for k in keys}
        fold_values[fold, 2] = values
        paired_folds.append({"fold": fold, "seeds": [0,1,2,3,4],
                             "difference": tree({k: float(v) for k, v in values.items()})})
    overall = {}
    for arm, name in enumerate(["13.78", "3.75", "difference_3.75_minus_13.78"]):
        statistics = {}
        for key in keys:
            values = [fold_values[f, arm][key] for f in range(5)]
            mean = sum(values)/5
            variance = sum((value-mean)**2 for value in values)/4
            statistics[key] = {"mean": float(mean), "sample_std": math.sqrt(float(variance)),
                               "fold_values": list(map(float, values))}
        overall[name] = tree(statistics)
    return {"schema_version": "dofa-partial-expected-v1", **header(), "fixed_tests": fixed,
            "comparison": {"cell_rows": cells, "paired_seed_rows": pairs, "fold_band_rows": fold_rows,
                           "paired_fold_rows": paired_folds, "overall": overall}}


def header():
    return {"class_order": CLASS_ORDER[:], "band_order": [13.78,3.75], "fold_order": [0,1,2,3,4],
            "seed_order": [0,1,2,3,4], "units": "fractions", "fixture_purpose": WARNING}


def fixture(equal_regions=False):
    folds, cells = [], []
    for f in range(5):
        g = 0 if equal_regions else f
        labels = [c for c, n in enumerate([4+g,6+2*g,3+g%2,2,1+g,5]) for _ in range(n)]
        folds.append({"fold": f, "region": f+1, "tile_ids": [f"invented-\u03b1-{f}-{i}" for i in range(len(labels))], "labels": labels})
        for s in range(5):
            for arm, band in enumerate([13.78,3.75]):
                pred = [c if (j+s+2*c)%7 < 3+g%3+arm else (c+1+(j+s)%3)%6 for j, c in enumerate(labels)]
                cells.append({"cell": 10*f+2*s+arm, "fold": f, "seed": s, "band": band, "predictions": pred})
    records = {"schema_version": "dofa-partial-numeric-v1", **header(), "folds": folds, "cells": cells}
    fake = lambda text: bound(("invented binding only: " + text).encode())
    source = {"schema_version": "dofa-partial-provenance-v1", "fixture_purpose": WARNING,
              "producer_commit": "3a91cc7f19ab6c3f91b9015f5e9e09b97364aaf0",
              "contract": {"bytes": 43796, "sha256": "7eecb0d70e467c222a501b96150c502ab606ec318a207569729426f354ee5d39"},
              "checker_report": fake("checker"), "aggregate": fake("aggregate"), "exporter": fake("exporter"),
              "fixed_test_sources": [{"fold": f, "catalog": fake("catalog"), "split": fake(f"split {f}")} for f in range(5)],
              "cell_sources": [{"cell": i, "result": fake(f"result {i}"), "replay": fake(f"replay {i}"),
                                "history": fake(f"history {i}"), "selected_epoch": 1+i%50} for i in range(50)],
              "field_locators": {"tile_ids": "/test/tile_ids", "labels": "/test/labels", "predictions": "/test/predictions",
                                 "selected_epoch": "/cell_rows/{cell}/selection_amp/selected_epoch",
                                 "cell_metrics": "/cells/{cell}/recomputed_test",
                                 "summaries": ["/paired_seed_rows", "/fold_band_rows", "/paired_fold_rows", "/overall"]}}
    return records, independent_expected(records), source


def seal(root):
    """Test packaging only; does not calculate scientific expectations."""
    manifest = {"schema_version": "dofa-partial-package-v1", "fixture_purpose": WARNING,
                "files": {name: bound((root/name).read_bytes()) for name in sorted(FILES)}}
    (root/"manifest.json").write_bytes(raw_json(manifest))
    sums = {name: value["sha256"] for name, value in manifest["files"].items()}
    sums["manifest.json"] = bound((root/"manifest.json").read_bytes())["sha256"]
    (root/"SHA256SUMS").write_text("".join(f"{sums[name]}  {name}\n" for name in sorted(sums)), encoding="utf-8")


def make_package(root, data=None):
    root.mkdir()
    for name in ("README.md", "LICENSE", "NOTICE.md", "reconstruct.py", "test_reconstruct.py"):
        shutil.copyfile(Path(__file__).parent/name, root/name)
    records, expected, source = data or fixture()
    (root/"records.json.gz").write_bytes(gzip.compress(raw_json(records), mtime=0))
    (root/"expected.json").write_bytes(raw_json(expected))
    (root/"provenance.json").write_bytes(raw_json(source))
    seal(root)
    return root


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        keep = os.environ.get("DOFA_COMPANION_TEST_ROOT")
        if keep:
            self.temp = Path(tempfile.mkdtemp(prefix=self._testMethodName+"-", dir=keep))
        else:
            temp = tempfile.TemporaryDirectory(prefix="dofa-partial-")
            self.addCleanup(temp.cleanup)
            self.temp = Path(temp.name)
        self.records, self.expected, self.source = fixture()

    def reconstruct(self, records=None, expected=None, source=None):
        return app.reconstruct(records or self.records, expected or self.expected, source or self.source, True)

    def test_literal_asymmetric_counts_zero_denominators(self):
        got = app.metrics([0,0,0,1,1,2], [0,1,1,0,1,0])
        self.assertEqual(got["confusion_counts"], [[1,2,0,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0]]+[[0]*6 for _ in range(3)])
        self.assertEqual(got["support"], [3,2,1,0,0,0])
        self.assertEqual(got["prediction_counts"], [3,3,0,0,0,0])
        self.assertEqual(got["scores"]["per_class"]["precision"], [1/3,1/3,0,0,0,0])
        self.assertEqual(got["scores"]["per_class"]["recall"], [1/3,1/2,0,0,0,0])
        self.assertAlmostEqual(got["scores"]["macro_f1"], 11/90, places=15)
        self.assertAlmostEqual(got["scores"]["balanced_accuracy"], 5/36, places=15)

    def test_full_rational_oracle_all_components_and_populations(self):
        got = self.reconstruct()
        for key, n in (("cell_rows",50),("paired_seed_rows",25),("fold_band_rows",10),("paired_fold_rows",5),("overall",3)):
            self.assertEqual(len(got[key]), n)
        # Assert against the independent oracle without using app.compare.
        def visit(a, e):
            if isinstance(e, dict):
                self.assertEqual(set(a), set(e))
                for k in e: visit(a[k], e[k])
            elif isinstance(e, list):
                self.assertEqual(len(a), len(e))
                for x, y in zip(a,e): visit(x,y)
            elif type(e) is float:
                self.assertLessEqual(abs(a-e), 1e-12)
            else:
                self.assertEqual((type(a),a), (type(e),e))
        visit(got, self.expected["comparison"])
        o = got["overall"]["13.78"]
        p,r,f = (o["per_class"][k][0]["mean"] for k in ("precision","recall","f1"))
        self.assertGreater(abs(f-2*p*r/(p+r)), 1e-5)
        populations = [len(row["labels"]) for row in self.records["folds"]]
        values = o["macro_f1"]["fold_values"]
        weighted = sum(n*v for n,v in zip(populations,values))/sum(populations)
        self.assertGreater(abs(weighted-o["macro_f1"]["mean"]), 1e-5)

    def test_paired_sample_sd_not_marginal_sd_or_sem(self):
        got = self.reconstruct()["overall"]
        sd = got["difference_3.75_minus_13.78"]["macro_f1"]["sample_std"]
        marginal = math.hypot(got["13.78"]["macro_f1"]["sample_std"], got["3.75"]["macro_f1"]["sample_std"])
        self.assertGreater(sd, 0)
        self.assertGreater(abs(sd-marginal), 1e-4)
        self.assertGreater(abs(sd-sd/math.sqrt(5)), 1e-4)

    def test_seed_variation_with_identical_regional_means(self):
        data = fixture(equal_regions=True)
        got = app.reconstruct(*data, synthetic=True)
        values = [r["difference"]["macro_f1"] for r in got["paired_seed_rows"]]
        self.assertGreater(max(values)-min(values), 0.01)
        self.assertEqual(got["overall"]["difference_3.75_minus_13.78"]["macro_f1"]["sample_std"], 0.0)
        self.assertEqual(got["overall"]["13.78"]["macro_f1"]["sample_std"], 0.0)

    def test_pc01_ordered_ids_and_labels_binding(self):
        literal = '{"labels":[0,1],"tile_ids":["\u03b1","b"]}'.encode("utf-8")
        self.assertEqual(app.truth_digest(["\u03b1","b"],[0,1]), hashlib.sha256(literal).hexdigest())
        for key in ("tile_ids", "labels"):
            bad = deepcopy(self.records)
            if key == "tile_ids":
                ids = bad["folds"][0][key]; ids[0], ids[1] = ids[1], ids[0]
                # Labels/predictions unchanged, hence every numerical metric is unchanged.
            else:
                bad["folds"][0][key][0] = 5
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Ordered truth digest"):
                self.reconstruct(records=bad)

    def test_identity_shape_order_units_and_scalar_rejections(self):
        mutations = [lambda r: r["cells"].pop(), lambda r: r["cells"].__setitem__(1,deepcopy(r["cells"][0])),
                     lambda r: r["cells"].reverse(), lambda r: r["folds"].reverse(),
                     lambda r: r.__setitem__("units","percent"), lambda r: r["class_order"].reverse(),
                     lambda r: r["band_order"].reverse(), lambda r: r["seed_order"].reverse(),
                     lambda r: r["cells"][0].__setitem__("seed",True),
                     lambda r: r["cells"][0].__setitem__("cell",0.0),
                     lambda r: r["cells"][0].__setitem__("band",3.75),
                     lambda r: r["cells"][0]["predictions"].pop(),
                     lambda r: r["folds"][0]["tile_ids"].__setitem__(1,r["folds"][0]["tile_ids"][0])]
        for value in (True, 1.0, -1, 6, "1"):
            mutations.append(lambda r,v=value: r["cells"][0]["predictions"].__setitem__(0,v))
            mutations.append(lambda r,v=value: r["folds"][0]["labels"].__setitem__(0,v))
        for i, change in enumerate(mutations):
            bad = deepcopy(self.records); change(bad)
            with self.subTest(mutation=i), self.assertRaises(ValueError): self.reconstruct(records=bad)

    def test_expected_types_structure_and_metric_only_tolerance(self):
        for value in (True, float("nan"), float("inf"), "0.2"):
            bad = deepcopy(self.expected); bad["comparison"]["cell_rows"][0]["scores"]["macro_f1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.reconstruct(expected=bad)
        for value in (True, 4.0, 4+1e-13):
            bad = deepcopy(self.expected); bad["comparison"]["cell_rows"][0]["support"][0] = value
            with self.subTest(count=value), self.assertRaises(ValueError): self.reconstruct(expected=bad)
        for key, value in (("extra",0), ("macro_f1",None)):
            bad = deepcopy(self.expected)
            scores = bad["comparison"]["cell_rows"][0]["scores"]
            if value is None: del scores[key]
            else: scores[key] = value
            with self.assertRaises(ValueError): self.reconstruct(expected=bad)
        for delta, accepted in ((0.5e-12, True),(2e-12, False)):
            bad = deepcopy(self.expected); bad["comparison"]["cell_rows"][0]["scores"]["macro_f1"] += delta
            if accepted: self.reconstruct(expected=bad)
            else:
                with self.assertRaisesRegex(ValueError,"Expected metric"): self.reconstruct(expected=bad)

    def test_strict_json_and_provenance(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): app.decode(raw)
        for change in (lambda p: p.__setitem__("producer_commit","old-loss-source"),
                       lambda p: p["cell_sources"][0].__setitem__("selected_epoch",True),
                       lambda p: p["cell_sources"].pop(), lambda p: p.__setitem__("private_path","/host/a"),
                       lambda p: p["checker_report"].__setitem__("bytes",True),
                       lambda p: p["field_locators"].__setitem__("labels","/wrong")):
            bad = deepcopy(self.source); change(bad)
            with self.assertRaises(ValueError): self.reconstruct(source=bad)

    def test_exact_inventory_checksums_and_symlink_refusal(self):
        for mode in ("extra","missing","tamper","symlink","directory","checksum_missing","manifest_escape"):
            with self.subTest(mode=mode):
                root = make_package(self.temp/mode)
                if mode == "extra": (root/"extra").write_text("x")
                if mode == "missing": (root/"LICENSE").unlink()
                if mode == "tamper": (root/"README.md").write_text("altered")
                if mode in ("symlink","directory"):
                    (root/"LICENSE").unlink()
                    if mode == "directory": (root/"LICENSE").mkdir()
                    else: (root/"LICENSE").symlink_to(Path(__file__).parent/"LICENSE")
                if mode == "checksum_missing": (root/"SHA256SUMS").write_bytes(b"")
                if mode == "manifest_escape":
                    manifest = json.loads((root/"manifest.json").read_bytes())
                    manifest["files"]["../README.md"] = manifest["files"].pop("README.md")
                    (root/"manifest.json").write_bytes(raw_json(manifest))
                with self.assertRaises(ValueError): app.package(root, True)

    def test_truncated_gzip_and_bounded_compressed_plain_decoded_reads(self):
        root = make_package(self.temp/"package")
        archive = root/"records.json.gz"
        original = archive.read_bytes()
        archive.write_bytes(original[:-6]); seal(root)
        with self.assertRaises((ValueError,EOFError,OSError)): app.package(root,True)
        archive.write_bytes(gzip.compress(b" "*(512*1024+1),mtime=0)); seal(root)
        with patch.object(app,"FILE_LIMIT",512*1024), self.assertRaisesRegex(ValueError,"Decoded records"):
            app.package(root,True)
        archive.write_bytes(original); seal(root)
        with patch.object(app,"FILE_LIMIT",8), self.assertRaisesRegex(ValueError,"Declared size"):
            app.package(root,True)
        with patch.object(app,"META_LIMIT",8), self.assertRaisesRegex(ValueError,"operational cap"):
            app.package(root,True)
        with self.assertRaisesRegex(ValueError,"operational cap"): app.read_bounded(archive,8)

    def test_all_container_synthetic_markers_agree(self):
        for i in range(3):
            data = deepcopy((self.records,self.expected,self.source)); del data[i]["fixture_purpose"]
            with self.subTest(container=i), self.assertRaises(ValueError): app.reconstruct(*data,synthetic=True)
        with self.assertRaises(ValueError): app.reconstruct(self.records,self.expected,self.source,False)
        root = make_package(self.temp/"package")
        manifest = json.loads((root/"manifest.json").read_bytes()); manifest["fixture_purpose"] = "other"
        (root/"manifest.json").write_bytes(raw_json(manifest))
        with self.assertRaises(ValueError): app.package(root,True)

    def test_write_body_flush_and_close_failures_preserve_primary(self):
        class Stream:
            def __init__(self, body, close): self.body, self.closing = body, close
            def write(self,data):
                if self.body == "write": raise RuntimeError("primary write")
                return len(data)
            def flush(self):
                if self.body == "flush": raise RuntimeError("primary flush")
            def fileno(self): return 123
            def close(self):
                if self.closing: raise OSError("close failure")
        for body, close in (("write",True),("flush",True),(None,True)):
            stream = Stream(body,close)
            with patch.object(Path,"open",return_value=stream), patch.object(app.os,"fsync"):
                with self.assertRaisesRegex(RuntimeError if body else OSError, "primary "+body if body else "close failure") as caught:
                    app.write_new(self.temp/"file",b"data")
                if body: self.assertIn("Close also failed", caught.exception.__notes__[0])

    def test_partial_publication_and_secondary_diagnostic_failure(self):
        root = make_package(self.temp/"package"); output = self.temp/"out.json"
        primary = OSError("primary publication failure")
        original = app.write_new
        def write(path, raw):
            if str(path).endswith(".failure.json"): raise RuntimeError("secondary receipt failure")
            original(path,raw)
        with patch.object(app.os,"link",side_effect=primary), patch.object(app,"write_new",side_effect=write):
            with self.assertRaises(OSError) as caught: app.run(root,output,True)
        self.assertIs(caught.exception,primary)
        self.assertFalse(output.exists())
        self.assertTrue(Path(str(output)+".partial").is_file())
        self.assertIn("Failure receipt could not be written",primary.__notes__[0])
        with self.assertRaisesRegex(ValueError,"Fresh output"): app.run(root,output,True)
        with self.assertRaisesRegex(ValueError,"outside package"): app.run(root,root/"out.json",True)

    def test_real_cli_success_refusal_and_tampered_truth(self):
        root = make_package(self.temp/"package")
        output = self.temp/"synthetic.json"
        def invoke(out, flag=True):
            return subprocess.run([sys.executable,"-B",str(root/"reconstruct.py"),"--output",str(out)]+(["--synthetic"] if flag else []),
                                  capture_output=True,text=True,timeout=30)
        result = invoke(output)
        self.assertEqual(result.returncode,0,result.stderr)
        console = json.loads(result.stdout)
        report = json.loads(output.read_bytes())
        self.assertEqual(console["fixture_purpose"],WARNING)
        self.assertEqual(report["status"],"synthetic_numeric_reconstruction_pass")
        self.assertFalse(report["scope"]["execution_authenticated"])
        before = output.read_bytes()
        self.assertNotEqual(invoke(output).returncode,0)
        self.assertEqual(output.read_bytes(),before)
        scientific = self.temp/"scientific.json"
        self.assertNotEqual(invoke(scientific,False).returncode,0)
        self.assertFalse(scientific.exists())
        self.assertEqual(json.loads(Path(str(scientific)+".failure.json").read_bytes())["status"],"failed")
        records = deepcopy(self.records)
        ids = records["folds"][0]["tile_ids"]; ids[0],ids[1] = ids[1],ids[0]
        (root/"records.json.gz").write_bytes(gzip.compress(raw_json(records),mtime=0)); seal(root)
        failed = invoke(self.temp/"bad-order.json")
        self.assertNotEqual(failed.returncode,0)
        self.assertIn("Ordered truth digest differs",failed.stderr)


if __name__ == "__main__":
    unittest.main()
