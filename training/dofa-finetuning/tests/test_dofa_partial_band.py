"""Synthetic contracts and inert subprocesses only: no NumPy/Torch/model imports."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_dofa_partial_band.py"
spec = importlib.util.spec_from_file_location("partial_band_under_test", SOURCE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def binding(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "bytes": size}


def plan_fixture(root):
    return {
        "schema": "dofa-partial-band-v1", "status": "reviewed_for_execution", "policy": copy.deepcopy(P.POLICY),
        "consumer": {"path": str(root / "consumer"), "commit": "b" * 40, "script": binding(root / "consumer/runner.py")},
        "helper": binding(root / "helper.py", P.HELPER_SHA),
        "historical_source": {"path": str(root / "historical"), "commit": P.HISTORICAL, "files": {}},
        "protocol": binding(root / "inputs/protocol.json", P.PROTOCOL_SHA),
        "catalog": binding(root / "inputs/catalog.json", P.CATALOG_SHA),
        "weights": binding(root / "weights.pth", P.H.WEIGHTS_SHA, 447669164),
        "upstream": {"path": str(root / "upstream"), "commit": P.H.UPSTREAM_COMMIT,
                     "files": {n: binding(root / "upstream" / n, h) for n, h in P.H.UPSTREAM_FILES.items()}},
        "head_state_sha256": list(P.HEADS),
        "initial_feature_state": {"sha256": "d" * 64, "tensor_count": 170, "excludes": ["head.weight", "head.bias"]},
        "python": binding(root / "python"), "python_entrypoint": str(root / "venv/bin/python"),
        "tiles": binding(root / "tiles.json"), "output_root": str(root / "results"),
        "fit_timeout_seconds": 5400, "replay_timeout_seconds": 600,
        "devices": [{"uuid": "GPU-synthetic-" + str(i), "lock": str(root / (str(i) + ".lock"))} for i in range(3)],
        "folds": [{"fold": f, "counts": list(P.COUNTS[f]), "split": binding(root / f"split-{f}.json"),
                   "reference": binding(root / f"analysis-{f}/probing_results.json"),
                   "analysis_manifest": binding(root / f"analysis-{f}/run_manifest.json"),
                   "feature_manifest": binding(root / f"features-{f}/run_manifest.json"),
                   "features_dir": str(root / f"features-{f}/features")} for f in range(5)],
        "cells": [{"cell": i, "fold": i // 10, "seed": (i // 2) % 5, "band": P.BANDS[i % 2],
                   "device": (i // 2) % 3, "output": str(root / f"results/cell-{i:02d}"),
                   "replay_output": str(root / f"results/replay-{i:02d}")} for i in range(50)],
    }


def test_exact_complete_grid_and_historical_reference(tmp_path):
    c = plan_fixture(tmp_path)
    P.validate_shape(c)
    for cell in c["cells"]:
        args = P.historical_args(c, cell)
        assert args.batch_size == 8 and args.grad_accum_steps == 4 and args.epochs == 50
        assert args.num_workers == 2 and args.models == ["dofa"] and args.seeds == [cell["seed"]]
        assert args.repaired_protocol == c["protocol"]["path"]
    assert P.historical_args(c, c["cells"][0]).__dict__ | {"output_dir": "same"} == P.historical_args(c, c["cells"][1]).__dict__ | {"output_dir": "same"}


@pytest.mark.parametrize("change", [
    lambda c: c.update(status="draft"),
    lambda c: c["policy"].update(epochs=1),
    lambda c: c["policy"].update(amp_test=True),
    lambda c: c["cells"].pop(),
    lambda c: c["cells"][1].update(band=13.78),
    lambda c: c["cells"][1].update(device=2),
    lambda c: c["cells"][0].update(seed=True),
    lambda c: c["cells"][0].update(output=c["cells"][1]["output"]),
    lambda c: c["cells"][0].update(output=str(Path(c["output_root"]).parent / "escape")),
    lambda c: c["protocol"].update(sha256="0" * 64),
    lambda c: c["weights"].update(sha256="0" * 64),
    lambda c: c["upstream"].update(path="/unbound/source"),
    lambda c: c["head_state_sha256"].__setitem__(0, "0" * 64),
    lambda c: c["historical_source"].update(commit="0" * 40),
    lambda c: c.update(fit_timeout_seconds=999999),
    lambda c: c["folds"][0].update(counts=[1, 2, 3]),
])
def test_plan_rejects_confounded_or_incomplete_cells(tmp_path, change):
    c = plan_fixture(tmp_path)
    change(c)
    with pytest.raises(ValueError):
        P.validate_shape(c)


def test_import_and_help_do_not_import_scientific_packages(tmp_path):
    code = """
import sys, importlib.abc, runpy
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','timm','titansar','sklearn'}:
            raise RuntimeError('scientific import: ' + fullname)
sys.meta_path.insert(0, Reject())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(SOURCE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "check-plan" in result.stdout


def test_cli_bad_contract_has_no_output_effect(tmp_path):
    c = plan_fixture(tmp_path)
    c["status"] = "draft"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(c))
    result = subprocess.run([sys.executable, "-B", str(SOURCE), "fit", "--contract", str(path),
                             "--contract-sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "--cell", "0"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "reviewed executable" in result.stderr
    assert not Path(c["output_root"]).exists()


def test_direct_band_route_all_roles_and_no_typeerror_retry():
    ledger = P.BandLedger(3.75)
    calls = []

    def forward(model, images, wave_list):
        calls.append(wave_list)
        return "features"

    for stage in ("train", "val", "test", "replay_test"):
        model = SimpleNamespace(training=stage == "train")
        assert P.forward_with_band(forward, model, SimpleNamespace(shape=(8, 1, 224, 224)), ledger,
                                   stage, 3.75, stage in ("train", "val")) == "features"
    assert calls == [[3.75]] * 4
    assert ledger.summary()["incomplete_call"] is None

    def broken(*args, **kwargs):
        calls.append(kwargs["wave_list"])
        raise TypeError("real forward failure")

    with pytest.raises(TypeError, match="real forward failure"):
        P.forward_with_band(broken, SimpleNamespace(training=False), SimpleNamespace(shape=(2,)), ledger,
                            "test", 3.75, False)
    assert len(calls) == 5
    assert ledger.summary()["incomplete_call"]["rows"] == 2


@pytest.mark.parametrize("stage,rows,band,amp,training", [
    ("train", 8, 13.78, True, True), ("val", 8, 3.75, False, False),
    ("test", 8, 3.75, True, False), ("replay_test", 8, 3.75, False, True),
    ("train", 32, 3.75, True, True), ("unknown", 8, 3.75, False, False),
])
def test_band_ledger_rejects_drift(stage, rows, band, amp, training):
    with pytest.raises(ValueError):
        P.BandLedger(3.75).begin(stage, rows, band, amp, training)


def completed_ledger(fold=0):
    ledger = P.BandLedger(13.78)
    for stage, n, repeats in (("train", P.COUNTS[fold][0], 50), ("val", P.COUNTS[fold][1], 50), ("test", P.COUNTS[fold][2], 1)):
        ledger.stages[stage] = {"calls": repeats * ((n + 7) // 8), "rows": repeats * n}
    return ledger


def test_ledger_requires_all_50_epochs_and_retained_partial_batches():
    ledger = completed_ledger()
    ledger.verify(0)
    ledger.stages["train"]["rows"] -= 3
    with pytest.raises(ValueError, match="coverage"):
        ledger.verify(0)
    ledger = completed_ledger()
    ledger.stages["val"]["calls"] -= (P.COUNTS[0][1] + 7) // 8
    with pytest.raises(ValueError, match="coverage"):
        ledger.verify(0)


def runtime_fixture():
    return {**{k: {"value": 1} for k in ("settings", "platform", "python", "python_entrypoint", "python_binary", "nvidia_smi", "environment", "autocast_dtype")},
            "packages": {"torch": {"version": "pinned"}}, "imports": {"titansar": "source"},
            "loaded_files": {k: {"path": "digest"} for k in ("module_paths", "module_files", "mapped_native_files")},
            "feature_state": "initial", "rng": "initial"}


def test_training_state_can_change_but_runtime_cannot():
    before = runtime_fixture()
    after = copy.deepcopy(before)
    after.update(feature_state="trained", rng="advanced")
    after["loaded_files"]["module_files"]["late_import"] = "new"
    P.compare_runtime(before, after)
    after["settings"]["value"] = 2
    with pytest.raises(ValueError, match="settings"):
        P.compare_runtime(before, after)


@pytest.mark.parametrize("field", ["imports", "loaded_files", "packages"])
def test_runtime_origin_and_library_drift_rejected(field):
    before = runtime_fixture()
    after = copy.deepcopy(before)
    if field == "imports":
        after[field]["titansar"] = "shadow"
    elif field == "packages":
        after[field]["torch"]["version"] = "changed"
    else:
        after[field]["mapped_native_files"]["path"] = "changed"
    with pytest.raises(ValueError):
        P.compare_runtime(before, after)


def test_preloaded_foreign_historical_script_rejected(tmp_path, monkeypatch):
    c = plan_fixture(tmp_path)
    monkeypatch.setattr(P.H, "imported_bindings", lambda c: {})
    monkeypatch.setitem(sys.modules, "scripts.foreign", SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    with pytest.raises(ValueError, match="shadowed"):
        P.check_origins(c)


def write_json(path, payload):
    path.write_text(json.dumps(payload))
    return P.H.binding(path)


def test_remaining_grid_requires_both_pilots_and_replays(tmp_path, monkeypatch):
    c = plan_fixture(tmp_path)
    cb = binding(tmp_path / "contract.json")
    checked = []

    def complete(c, cb, i, replay=False):
        checked.append((i, replay))
        return binding(tmp_path / (f"{i}-" + ("replay" if replay else "fit")))

    monkeypatch.setattr(P, "complete_output", complete)
    checker = tmp_path / "checker.py"
    checker.write_text("# inert independent-checker fixture")
    report = tmp_path / "checked.json"
    report.write_text("{}")
    receipt = {"schema": "dofa-partial-pilot-acceptance-v1", "status": "accepted", "contract": cb,
               "cells": [0, 1], "selection_by_test_score": False, "checker": P.H.binding(checker),
               "report": P.H.binding(report), "outputs": [{"cell": i, "fit": complete(c, cb, i),
                                                           "replay": complete(c, cb, i, True)} for i in (0, 1)]}
    with pytest.raises(ValueError, match="independent pilot"):
        P.admitted_dependencies(c, cb, c["cells"][2], "fit", None)
    acceptance = write_json(tmp_path / "acceptance.json", receipt)
    checked.clear()
    P.admitted_dependencies(c, cb, c["cells"][2], "fit", acceptance)
    assert checked == [(0, False), (0, True), (1, False), (1, True)]
    receipt["selection_by_test_score"] = True
    acceptance = write_json(tmp_path / "acceptance.json", receipt)
    with pytest.raises(ValueError, match="acceptance contract"):
        P.admitted_dependencies(c, cb, c["cells"][2], "fit", acceptance)


def test_pair_and_replay_prerequisites(tmp_path, monkeypatch):
    c = plan_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(P, "complete_output", lambda c, cb, i: calls.append(i))
    P.admitted_dependencies(c, {}, c["cells"][0], "fit", None)
    assert calls == []
    P.admitted_dependencies(c, {}, c["cells"][1], "fit", None)
    P.admitted_dependencies(c, {}, c["cells"][1], "replay", None)
    assert calls == [0, 1]


def test_terminal_failure_cannot_be_used_as_complete(tmp_path):
    c = plan_fixture(tmp_path)
    output = Path(c["cells"][0]["output"])
    output.mkdir(parents=True)
    cb = binding(tmp_path / "contract.json")
    terminal = {"status": "failed", "contract": cb, "cell": 0, "mode": "fit", "files": {}, "process_exit": 0}
    for name in ("supervisor", "worker"):
        write_json(output / (name + "-terminal.json"), terminal)
    with pytest.raises(ValueError, match="incomplete"):
        P.complete_output(c, cb, 0)


def test_failure_receipt_and_existing_output_preservation(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    def fail(record):
        record["marker"] = "started"
        raise RuntimeError("synthetic failure")
    assert P.H.attempt(output, "worker", fail) == 1
    terminal = P.H.read(output / "worker-terminal.json")
    assert terminal["status"] == "failed" and terminal["error"]["message"] == "synthetic failure"
    with pytest.raises(FileExistsError):
        P.H.fresh_output(output, [])
    with pytest.raises(FileExistsError):
        P.H.publish(output / "worker-terminal.json", {})


def test_supervised_inert_failure_and_timeout(tmp_path):
    lock = tmp_path / "device.lock"
    lock.touch()
    for name, code, timeout in (("failure", "raise SystemExit(7)", 3), ("timeout", "import time; time.sleep(10)", 0.05)):
        output = tmp_path / name
        output.mkdir()
        record = {}
        with pytest.raises((ValueError, TimeoutError)):
            P.H.supervise([sys.executable, "-B", "-c", code], {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                          output, lock, timeout, record)
        assert record["process_exit"] is not None
        assert (output / "launch.json").is_file()
    assert P.H.read(tmp_path / "failure/child-started.json")["pid"] > 0


def test_raw_tile_changes_fail_without_replaying_data(tmp_path):
    path = tmp_path / "tile.npy"
    path.write_bytes(b"synthetic")
    st = path.stat()
    stats = {str(path): [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]}
    P.check_tile_stats(stats)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata changed"):
        P.check_tile_stats(stats)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_result_cannot_be_serialized_as_success(bad):
    with pytest.raises(ValueError, match="Nonfinite"):
        P.native_json({"loss": [bad]})


def test_valid_scalar_types_preserved():
    assert P.native_json({"a": True, "b": [1, 2.0, None]}) == {"a": True, "b": [1, 2.0, None]}


def test_fit_calls_original_trainer_with_fixed_budget_and_supplied_head(tmp_path):
    """Exercise the consumer boundary with inert functions, not a neural model."""
    c = plan_fixture(tmp_path)
    cell = c["cells"][1]
    events = []
    selected = SimpleNamespace(state_dict=lambda: {})
    wrapper = SimpleNamespace(phase=None, model=selected)
    probe = SimpleNamespace(state_dict=lambda: {})
    supplied_head = object()
    history = {k: [1] * 50 for k in ("train_loss", "val_balanced_accuracy", "optimizer_steps", "amp_skipped_steps", "loss_scale")}
    class Train:
        def __len__(self):
            return P.COUNTS[0][0]
        def get_class_weights(self):
            return [1, 2, 3, 4, 5, 6]
    test_loader = SimpleNamespace(dataset="test-membership")
    loaders = (Train(), "train-loader", "validation-loader", test_loader)

    def train(**kwargs):
        events.append("train")
        assert kwargs == {"model": wrapper, "train_loader": "train-loader", "val_loader": "validation-loader", "num_classes": 6,
                          "unfreeze_blocks": 2, "lr": 1e-4, "epochs": 50, "class_weights": [1, 2, 3, 4, 5, 6],
                          "device": "cuda:0", "use_amp": True, "grad_accum_steps": 4, "return_history": True, "init_probe": supplied_head}
        assert wrapper.phase == "fit"
        return wrapper, probe, 0.5, history

    def evaluate(model, head, loader, classes, device):
        events.append("test")
        assert model is wrapper and head is probe and model.phase == "test"
        return {"probabilities": [[0.25, 0.75]], "predictions": [1]}

    def save(path, model, head, seed, depth, best, provenance):
        events.append("save_selected")
        assert model is wrapper and head is probe and seed == 0 and depth == 2 and best == 0.5
        assert provenance["numeric_band_input"] == [3.75] and provenance["selected_epoch"] == 7
        path.write_bytes(b"inert selected checkpoint")

    runner = SimpleNamespace(train_finetuned_model=train, evaluate_finetuned=evaluate, save_checkpoint=save,
        repaired_history_metadata=lambda h, n, v: {"selected_epoch": 7}, _check_prediction=lambda r, d: {"f1": 0.2}, torch=object())
    ledger = P.BandLedger(3.75)
    ledger.stages = completed_ledger().stages
    record = {}
    P.fit(runner, c, cell, {"loaders": loaders}, wrapper, supplied_head, {"numeric_band_input": [3.75]}, ledger, tmp_path, record)
    assert events == ["train", "test", "save_selected"]
    result = P.H.read(tmp_path / "result.json")
    assert result["cell"]["band"] == 3.75 and result["provenance"]["selected_epoch"] == 7
    assert len(P.H.read(tmp_path / "history.json")["history"]["train_loss"]) == 50


@pytest.mark.parametrize("wrong_band,wrong_prediction,wrong_selection", [(False, False, False), (True, False, False),
                                                                 (False, True, False), (False, False, True)])
def test_replay_restores_selected_state_and_rejects_band_or_prediction_drift(tmp_path, wrong_band, wrong_prediction, wrong_selection):
    c = plan_fixture(tmp_path)
    cell = c["cells"][1]
    fit_root = Path(cell["output"])
    fit_root.mkdir(parents=True)
    output = tmp_path / "replay"
    output.mkdir()
    test = {"predictions": [1], "probabilities": [[0.25, 0.75]]}
    provenance = {"numeric_band_input": [13.78 if wrong_band else 3.75], "selected_epoch": 9 if wrong_selection else 7}
    write_json(fit_root / "result.json", {"provenance": provenance, "cell": cell, "test": test, "best_validation_macro_recall": 0.5})
    write_json(fit_root / "history.json", {"selection": provenance, "history": {"synthetic": [1] * 50}})
    (fit_root / "selected.pt").write_bytes(b"inert checkpoint")
    restored = []
    class InertState:
        def load_state_dict(self, state, strict):
            assert strict is True
            restored.append(state)
        def to(self, device):
            assert device == "cuda:0"
            return self
        def eval(self):
            return self
        def parameters(self):
            return []
    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"provenance": provenance, "seed": 0, "unfreeze_blocks": 2, "best_val_accuracy": 0.5, "model_state_dict": "selected-encoder",
                "probe_state_dict": "selected-head"}
    wrapper = SimpleNamespace(model=InertState(), phase=None)
    def evaluate(model, probe, loader, classes, device):
        assert model.phase == "replay_test" and restored == ["selected-encoder", "selected-head"]
        return {**test, "predictions": [0]} if wrong_prediction else test
    runner = SimpleNamespace(torch=SimpleNamespace(load=load), evaluate_finetuned=evaluate, _check_prediction=lambda r, d: None,
                             repaired_history_metadata=lambda h, n, v: {"selected_epoch": 7})
    ledger = P.BandLedger(3.75)
    n = P.COUNTS[0][2]
    ledger.stages["replay_test"] = {"calls": (n + 7) // 8, "rows": n}
    args = (runner, c, cell, {"loaders": (range(P.COUNTS[0][0]), None, None, SimpleNamespace(dataset="test"))}, wrapper, InertState(), {}, ledger, output, {})
    if wrong_band or wrong_prediction or wrong_selection:
        with pytest.raises(ValueError, match="band differs" if wrong_band else "epoch/history" if wrong_selection else "replay differs"):
            P.replay(*args)
        assert not (output / "replay.json").exists()
    else:
        P.replay(*args)
        assert P.H.read(output / "replay.json")["test_equal"] is True


@pytest.mark.parametrize("nonfinite", [False, True])
@pytest.mark.parametrize("diagnostic_write_fails", [False, True])
def test_rejected_returned_history_survives_without_masking_original_error(tmp_path, monkeypatch, nonfinite, diagnostic_write_fails):
    """IR-01: returned evidence survives rejection; accepted outputs stay absent."""
    history = {k: [1] * 50 for k in ("train_loss", "val_balanced_accuracy", "optimizer_steps", "amp_skipped_steps", "loss_scale")}
    history["train_loss" if nonfinite else "optimizer_steps"][0] = float("nan") if nonfinite else 0
    original_error = ValueError("nonfinite returned history" if nonfinite else "epoch had no optimizer updates")
    class InertTrain:
        def __len__(self):
            return P.COUNTS[0][0]
        def get_class_weights(self):
            return [1] * 6
    def reject(actual, count, best):
        assert actual is history and count == P.COUNTS[0][0] and best == 0.5
        raise original_error
    runner = SimpleNamespace(train_finetuned_model=lambda **kwargs: (object(), object(), 0.5, history),
                             repaired_history_metadata=reject)
    if diagnostic_write_fails:
        publish = P.H.publish_bytes
        def fail_diagnostic(path, raw):
            if Path(path).name == "returned-history-unvalidated.txt":
                raise OSError("synthetic diagnostic disk failure")
            return publish(path, raw)
        monkeypatch.setattr(P.H, "publish_bytes", fail_diagnostic)
    c = plan_fixture(tmp_path)
    record = {}
    with pytest.raises(ValueError) as caught:
        P.fit(runner, c, c["cells"][0], {"loaders": (InertTrain(), object(), object(), object())},
              SimpleNamespace(phase=None), object(), {}, None, tmp_path, record)
    assert caught.value is original_error
    diagnostic = tmp_path / "returned-history-unvalidated.txt"
    if diagnostic_write_fails:
        assert not diagnostic.exists()
        assert record["returned_history_diagnostic_error"] == {
            "type": "OSError", "message": "synthetic diagnostic disk failure"}
    else:
        raw = diagnostic.read_text()
        assert raw.startswith("UNVALIDATED returned history; not accepted scientific JSON.\n")
        assert repr(history) in raw
        if nonfinite:
            assert "nan" in raw
        assert "returned_history_diagnostic_error" not in record
    assert not any((tmp_path / n).exists() for n in ("history.json", "result.json", "selected.pt"))


def real_completed_pilot_fixture(root):
    """Tiny inert files with real worker/supervisor hashes; no completion stubs."""
    c = plan_fixture(root)
    cb = write_json(root / "contract.json", c)
    for cell_id in (0, 1):
        for replay in (False, True):
            cell = c["cells"][cell_id]
            output = Path(cell["replay_output"] if replay else cell["output"])
            output.mkdir(parents=True)
            names = {"initial.json", "runtime-final.json", "band-calls.json", "supervisor-started.json", "worker-started.json"}
            names |= {"replay.json"} if replay else {"result.json", "history.json", "selected.pt"}
            for name in names:
                if name.endswith(".pt"):
                    (output / name).write_bytes(b"inert checkpoint, never deserialized")
                else:
                    write_json(output / name, {"synthetic": True, "cell": cell_id, "file": name})
            for name in ("stdout.log", "stderr.log"):
                (output / name).write_text("")
            worker = {"status": "complete", "contract": cb, "cell": cell_id,
                      "mode": "replay" if replay else "fit",
                      "files": {p.name: P.H.binding(p) for p in output.iterdir()}}
            write_json(output / "worker-terminal.json", worker)
            supervisor = {**worker, "process_exit": 0,
                          "files": {p.name: P.H.binding(p) for p in output.iterdir()}}
            write_json(output / "supervisor-terminal.json", supervisor)
    checker = root / "independent-checker.py"
    checker.write_text("# inert source binding; not an executed reviewer\n")
    report = write_json(root / "independent-report.json", {"synthetic": True, "cells": [0, 1]})
    acceptance = {"schema": "dofa-partial-pilot-acceptance-v1", "status": "accepted", "contract": cb,
                  "cells": [0, 1], "selection_by_test_score": False, "checker": P.H.binding(checker), "report": report,
                  "outputs": [{"cell": i, "fit": P.H.binding(Path(c["cells"][i]["output"]) / "result.json"),
                               "replay": P.H.binding(Path(c["cells"][i]["replay_output"]) / "replay.json")} for i in (0, 1)]}
    return c, cb, write_json(root / "pilot-acceptance.json", acceptance)


def test_real_completed_output_chain_admits_both_pilots(tmp_path):
    c, cb, acceptance = real_completed_pilot_fixture(tmp_path)
    for cell in (0, 1):
        for replay in (False, True):
            result = P.complete_output(c, cb, cell, replay)
            assert Path(result["path"]).name == ("replay.json" if replay else "result.json")
    assert P.pilot_gate(c, cb, acceptance) == acceptance
    P.admitted_dependencies(c, cb, c["cells"][2], "fit", acceptance)


@pytest.mark.parametrize("fault,cell,replay,expected_error", [
    ("worker_failed", 1, True, "Required job incomplete"),
    ("supervisor_failed", 1, False, "Required job incomplete"),
    ("nonzero_exit", 0, True, "Required process failed"),
    ("missing_required", 0, False, "Required retained output inventory incomplete"),
    ("artifact_tamper", 1, False, "Bound file changed"),
])
def test_real_pilot_chain_rejects_targeted_failures(tmp_path, fault, cell, replay, expected_error):
    """IR-02: keep unrelated bindings valid so each intended guard is reached."""
    c, cb, acceptance = real_completed_pilot_fixture(tmp_path)
    output = Path(c["cells"][cell]["replay_output"] if replay else c["cells"][cell]["output"])
    worker = P.H.read(output / "worker-terminal.json")
    supervisor = P.H.read(output / "supervisor-terminal.json")
    if fault == "worker_failed":
        worker["status"] = "failed"
        write_json(output / "worker-terminal.json", worker)
        supervisor["files"]["worker-terminal.json"] = P.H.binding(output / "worker-terminal.json")
    elif fault == "supervisor_failed":
        supervisor["status"] = "failed"
    elif fault == "nonzero_exit":
        supervisor["process_exit"] = 9
    elif fault == "missing_required":
        (output / "history.json").unlink()
        worker["files"].pop("history.json")
        supervisor["files"].pop("history.json")
        write_json(output / "worker-terminal.json", worker)
        supervisor["files"]["worker-terminal.json"] = P.H.binding(output / "worker-terminal.json")
    elif fault == "artifact_tamper":
        (output / "selected.pt").write_bytes(b"changed checkpoint bytes")
    write_json(output / "supervisor-terminal.json", supervisor)
    with pytest.raises(ValueError, match=expected_error):
        P.complete_output(c, cb, cell, replay)
    with pytest.raises(ValueError, match=expected_error):
        P.pilot_gate(c, cb, acceptance)
