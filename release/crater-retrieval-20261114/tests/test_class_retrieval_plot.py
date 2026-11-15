"""Synthetic six-class display checks; no saved scientific probabilities."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import plot_class_retrieval as p
from tests.test_class_ranking_plot import report  # noqa: F401 -- shared sentinel report


@pytest.fixture
def inputs(tmp_path, report):
    entries, arrays = [], {}
    for row in report["metrics"]:
        kind, fold, arm, seed = row["cell"]
        c, n, k = row["class_index"], row["n"], row["positives"]
        prefix = f"{kind}__fold_{fold}__{arm}__seed_{seed}__class_{c}"
        counts, tp = np.array([1, 2, n]), np.array([0, 1, k])
        names = {name: prefix + "_" + name for name in p.base.FIELDS}
        values = {"tp": tp, "fp": counts - tp,
                  "thresholds": np.array([.9 - c * .01, .5, 0.]),
                  "precision": tp / counts, "recall": tp / k}
        arrays.update({names[key]: value for key, value in values.items()})
        row.update(distinct_scores=3, tied_groups=1, largest_tie=n - 2)
        if k == 1:
            row["single_positive"] = {"rank_first": 2, "rank_last": 2,
                                      "tie_size": 1, "tile_id": "SYNTHETIC"}
        entries.append({"prefix": prefix, "cell": row["cell"], "class_index": c,
                        "class": row["class"], "region": fold + 1, "score_groups": 3,
                        "arrays": names})
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps(report))
    export = {"schema_version": "class-retrieval-export-v1", "status": "computed_pending_publication",
              "accepted_results": {"sha256": p.base.digest(accepted)},
              "class_order": list(p.CLASSES), "fold_counts": report["fold_counts"],
              "metrics": copy.deepcopy(report["metrics"]), "curve_index": entries,
              "array_inventory": {key: {"dtype": value.dtype.str, "shape": list(value.shape),
                                        "bytes": value.nbytes} for key, value in arrays.items()}}
    return accepted, export, arrays


def arguments(tmp_path, inputs):
    accepted, export, arrays = inputs
    exported, curves = tmp_path / "export.json", tmp_path / "curves.npz"
    exported.write_text(json.dumps(export, allow_nan=False))
    np.savez(curves, **arrays)
    return ["--report", str(accepted), "--report-sha256", p.base.digest(accepted),
            "--export", str(exported), "--export-sha256", p.base.digest(exported),
            "--curves", str(curves), "--curves-sha256", p.base.digest(curves)]


def load(args):
    accepted = p.base.read_report(Path(args[1]), args[3])
    return p.read_curves(Path(args[5]), args[7], Path(args[9]), args[11], accepted, args[3])


def test_complete_six_class_grid_order_and_saved_endpoints(tmp_path, inputs):
    curves = load(arguments(tmp_path, inputs))
    assert len(curves) == 360
    assert {c["class"] for c in curves} == set(p.CLASSES)
    for curve in curves:
        prefix = curve["prefix"]
        np.testing.assert_array_equal(curve["count"], inputs[2][prefix + "_tp"] + inputs[2][prefix + "_fp"])
        np.testing.assert_array_equal(curve["recall_percent"], 100 * inputs[2][prefix + "_recall"])
    inputs[1]["metrics"].reverse()
    inputs[1]["curve_index"].reverse()
    reordered = load(arguments(tmp_path, inputs))
    assert [c["prefix"] for c in reordered] == [c["prefix"] for c in curves]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_class", "class_name",
                                     "class_remap", "array_remap", "metric", "bound_report", "inventory"])
def test_complete_identity_checks_before_output(tmp_path, inputs, mutation):
    export = inputs[1]
    first = export["curve_index"][0]
    if mutation == "missing": export["curve_index"].pop()
    elif mutation == "duplicate": export["curve_index"][-1] = copy.deepcopy(first)
    elif mutation == "bool_class": first["class_index"] = False
    elif mutation == "class_name": first["class"] = "craters"
    elif mutation == "class_remap": first["class_index"] = 5
    elif mutation == "array_remap": first["arrays"]["tp"] = first["arrays"]["fp"]
    elif mutation == "metric": export["metrics"][0]["ap"] += .1
    elif mutation == "bound_report": export["accepted_results"]["sha256"] = "0" * 64
    else: export["array_inventory"].pop(first["arrays"]["tp"])
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        p.main([*arguments(tmp_path, inputs), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_count", "bad_dtype"])
def test_array_validation_for_a_non_craters_class(tmp_path, inputs, mutation):
    key = next(k for k in inputs[2] if "__class_2_tp" in k)
    if mutation == "missing": inputs[2].pop(key)
    elif mutation == "bad_count": inputs[2][key][-1] -= 1
    elif mutation == "bad_dtype": inputs[2][key] = inputs[2][key].astype(np.float64)
    args = arguments(tmp_path, inputs)
    if mutation == "duplicate":
        with zipfile.ZipFile(args[9], "a") as z:
            data = z.read(key + ".npy")
            with pytest.warns(UserWarning, match="Duplicate name"):
                z.writestr(key + ".npy", data)
        args[11] = p.base.digest(Path(args[9]))
    with pytest.raises(ValueError):
        load(args)


@pytest.mark.parametrize("which", ["export", "curves"])
def test_hash_precedes_decoding(tmp_path, inputs, monkeypatch, which):
    args = arguments(tmp_path, inputs)
    index = 5 if which == "export" else 9
    Path(args[index]).write_bytes(b"invalid bytes")
    if which == "export": monkeypatch.setattr(p, "decode", lambda *a: pytest.fail("decoded bad hash"))
    else: monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("decoded bad hash"))
    with pytest.raises(ValueError, match="SHA256"):
        load(args)


@pytest.mark.parametrize("variant", ["cvpr", "full"])
def test_dense_class_artists_keep_every_endpoint_and_correct_label(tmp_path, inputs, variant):
    curves = [c for c in load(arguments(tmp_path, inputs)) if c["class_index"] == 2]
    fig, record = p.base.draw_figure(curves, variant, class_name="Hummocky", dense=True)
    assert "Recall of Hummocky map labels (%)" in {t.get_text() for t in fig.texts}
    assert len(record["artists"]) == 60
    for curve, actual in zip(curves, record["artists"]):
        assert actual["all_groups"]["point_count"] == len(curve["count"])
        assert actual["all_groups"]["x_sha256"] == p.base.coordinate_digest(curve["count"])
        assert actual["all_groups"]["y_sha256"] == p.base.coordinate_digest(curve["recall_percent"])
        assert actual["all_groups"]["rasterized"] is True
        assert actual["tp_increases"]["point_count"] == 0
        kind, _, arm, seed = curve["cell"]
        marker = next(m for a, s, m, _ in p.base.KEYS[kind] if (a, s) == (arm, seed))
        assert actual["all_groups"]["marker"] == marker
    assert not record["qa"]["clipped_text"]


def test_complete_cli_saves_all_classes_and_never_rescores(tmp_path, inputs, monkeypatch):
    # No scoring implementation is present in this standalone companion.
    out = tmp_path / "rendered"
    p.main([*arguments(tmp_path, inputs), "--output", str(out)])
    complete = json.loads((out / "complete.json").read_text())
    assert complete["curves_per_variant"] == 360 and complete["sheets_per_variant"] == 6
    assert len(complete["outputs"]) == 25
    for name, binding in complete["outputs"].items():
        assert p.base.digest(out / name) == binding["sha256"]
        assert (out / name).stat().st_size == binding["bytes"] > 0
    for name in p.CLASSES:
        for variant in ("cvpr", "full"):
            data = (out / f"{name}-retrieval-{variant}.pdf").read_bytes()
            assert data.startswith(b"%PDF") and str(tmp_path).encode() not in data
    with pytest.raises(ValueError, match="fresh"):
        p.main([*arguments(tmp_path, inputs), "--output", str(out)])


def test_partial_save_has_no_completion(tmp_path, inputs, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("synthetic save failure")
    monkeypatch.setattr(p.base.roman, "save_figure", fail)
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="synthetic save failure"):
        p.main([*arguments(tmp_path, inputs), "--output", str(out)])
    assert out.is_dir() and not (out / "complete.json").exists()


def test_public_imports_are_closed_without_analysis_or_gpu_modules():
    import subprocess
    import sys

    code = '''
import importlib.util
from pathlib import Path
import sys
from scripts import plot_class_retrieval, plot_dataset_geography
root = Path.cwd().resolve()
for name in ("scripts.analyze_class_ranking", "scripts.capture_gpu_environment",
             "scripts.prepare_dataset_geography", "scripts.export_class_retrieval"):
    assert importlib.util.find_spec(name) is None, name
for name, module in tuple(sys.modules.items()):
    if name.startswith("scripts."):
        assert Path(module.__file__).resolve().is_relative_to(root), name
    assert name.split(".")[0] not in {"torch", "torchvision", "timm", "sklearn", "rasterio", "geopandas"}, name
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=p.base.ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["write_completion", "write_new_atomic", "write_bytes_new"])
def test_extracted_writers_never_overwrite(tmp_path, name):
    from scripts import display_io

    target = tmp_path / "existing"
    target.write_bytes(b"original")
    value = b"replacement" if name == "write_bytes_new" else {"replacement": True}
    with pytest.raises(FileExistsError):
        getattr(display_io, name)(target, value)
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("name", ["write_completion", "write_new_atomic", "write_bytes_new"])
def test_extracted_writer_link_failure_publishes_nothing(tmp_path, monkeypatch, name):
    from scripts import display_io

    def fail(*args):
        raise OSError("synthetic link failure")
    monkeypatch.setattr(display_io.os, "link", fail)
    target = tmp_path / "complete.json"
    value = b"complete" if name == "write_bytes_new" else {"status": "complete"}
    with pytest.raises(OSError, match="synthetic link failure"):
        getattr(display_io, name)(target, value)
    assert not target.exists() and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["write_completion", "write_new_atomic"])
def test_json_serialization_failure_has_no_completion(tmp_path, name):
    from scripts import display_io

    target = tmp_path / "complete.json"
    with pytest.raises(ValueError):
        getattr(display_io, name)(target, {"invalid": float("nan")})
    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_terminal_link_survives_temporary_cleanup_failure(tmp_path, monkeypatch):
    from scripts import display_io

    def fail(*args):
        raise OSError("synthetic cleanup failure")
    monkeypatch.setattr(display_io.os, "unlink", fail)
    target = tmp_path / "complete.json"
    display_io.write_completion(target, {"status": "complete"})
    assert json.loads(target.read_text()) == {"status": "complete"}
