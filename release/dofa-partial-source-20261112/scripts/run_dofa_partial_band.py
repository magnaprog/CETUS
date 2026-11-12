"""Prospective paired DOFA training consumer; the parent owns launch and acceptance.

Importing this file imports only the standard library and one authenticated,
unchanged stdlib helper. Scientific imports occur solely in supervised workers.
The historical reference seal stays at 13.78; both current arms start afresh.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = "998d877d4af302f29b2146755da198b27f934f2319091c6193e86c3e5fccc8c6"
HISTORICAL = "ed2eb3ecd4722c22c0aff8b5370ac9a4b23e4376"
PROTOCOL_SHA = "dd280f9b830acd8fba84a0603504c12ee39941b40fb89912a3e264002c8f44cb"
CATALOG_SHA = "2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec"
BANDS = (13.78, 3.75)
COUNTS = ((14603, 4342, 3660), (12542, 5420, 4497), (12348, 4429, 5721),
          (13350, 4466, 4748), (14644, 3453, 4620))
HEADS = (
    "4185ad1a4d653714eb71bc4114cffb8d98e41ccc96239f4a2df4dc3cdd36041e",
    "192a620ada65d8fce89dc8c2aba1a2f1e3a5b33d26f817a8742cd78cfdfc82bb",
    "34b77217c198d20b97357dc0dacbcdf08a61f79f9ccd7f9d120c5e0d11ba50cc",
    "9dd53afab7e11584e0624279f38aabd33e9908ced92789403d1cef557a77be0d",
    "18a96e823cc044a524c3f40ce31e4b773303039e4107baf6a7dff83984e961eb",
)
POLICY = {
    "scope": "both_current_runtime_bands", "folds": list(range(5)),
    "seeds": list(range(5)), "bands": list(BANDS), "cells": 50,
    "epochs": 50, "microbatch": 8, "accumulation": 4, "workers": 2,
    "unfreeze_blocks": 2, "backbone_lr": 0.0001, "head_lr": 0.001,
    "weight_decay": 0.01, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
    "T_max": 50, "amp_training_and_validation": True, "amp_test": False,
    "selection": "earliest highest validation macro recall among updated epochs",
    "loss": "historical effective-batch weighted mean CE",
    "pilots": [0, 1], "pilot_epochs": 50, "pilot_selection_by_score": False,
    "checkpoint_retention": "selected_only_server_side", "resume": False,
}


def load_helper():
    path = ROOT / "scripts/run_dofa_band_sensitivity.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("Unchanged stdlib helper bytes differ")
    name = "_dofa_partial_band_io"
    if name in sys.modules:
        raise ValueError("Helper is already imported")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


H = load_helper()
require = H.require


def digest(value):
    return hashlib.sha256(H.encode(value)).hexdigest()


def file_record(record):
    require(set(record) == {"path", "sha256", "bytes"}, "Invalid file binding fields")
    require(isinstance(record["path"], str) and Path(record["path"]).is_absolute(), "Absolute input path required")
    require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64
            and all(x in "0123456789abcdef" for x in record["sha256"]), "Invalid digest")
    require(type(record["bytes"]) is int and record["bytes"] >= 0, "Invalid byte count")


def validate_shape(c):
    """Pure structural admission; neither creates a plan nor authorizes a run."""
    require(c["schema"] == "dofa-partial-band-v1" and c["status"] == "reviewed_for_execution",
            "A separately reviewed executable contract is required")
    require(c["policy"] == POLICY, "Scientific policy differs")
    require(c["historical_source"]["commit"] == HISTORICAL, "Historical producer differs")
    require(c["protocol"]["sha256"] == PROTOCOL_SHA and c["catalog"]["sha256"] == CATALOG_SHA,
            "Historical seal or catalog differs")
    require(c["weights"]["sha256"] == H.WEIGHTS_SHA and c["weights"]["bytes"] == 447669164,
            "Pretrained checkpoint differs")
    require(c["upstream"]["commit"] == H.UPSTREAM_COMMIT, "Upstream revision differs")
    require(set(c["upstream"]["files"]) == set(H.UPSTREAM_FILES), "Upstream file inventory differs")
    for name, expected in H.UPSTREAM_FILES.items():
        require(c["upstream"]["files"][name]["sha256"] == expected, "Upstream file differs")
        require(c["upstream"]["files"][name]["path"] == str(Path(c["upstream"]["path"]) / name),
                "Upstream import directory differs from authenticated files")
    require(c["head_state_sha256"] == list(HEADS), "Initial head pins differ")
    feature = c["initial_feature_state"]
    require(set(feature) == {"sha256", "tensor_count", "excludes"}
            and feature["tensor_count"] == 170 and feature["excludes"] == ["head.weight", "head.bias"]
            and len(feature["sha256"]) == 64, "Expected initial feature state required")
    require(type(c["fit_timeout_seconds"]) is int and 2400 <= c["fit_timeout_seconds"] <= 7200,
            "Fit deadline must be 2400–7200 seconds")
    require(type(c["replay_timeout_seconds"]) is int and 60 <= c["replay_timeout_seconds"] <= 1800,
            "Replay deadline must be 60–1800 seconds")
    require(len(c["folds"]) == 5 and len(c["cells"]) == 50, "Complete five-fold, 50-cell inventory required")
    require(len(c["devices"]) == 3 and len({d["uuid"] for d in c["devices"]}) == 3, "Three distinct physical GPUs required")
    for device in c["devices"]:
        require(device["uuid"].startswith("GPU-") and Path(device["lock"]).is_absolute(), "Invalid GPU/lock")
    outputs = []
    for f, fold in enumerate(c["folds"]):
        require(type(fold["fold"]) is int and fold["fold"] == f, "Fold order differs")
        require(fold["counts"] == list(COUNTS[f]), "Role counts differ")
        for key in ("split", "reference", "analysis_manifest", "feature_manifest"):
            file_record(fold[key])
        require(Path(fold["reference"]["path"]).name == "probing_results.json", "Reference filename differs")
        require(Path(fold["features_dir"]).is_absolute() and Path(fold["features_dir"]).name == "features",
                "Historical feature directory differs")
    for i, cell in enumerate(c["cells"]):
        require(type(cell["cell"]) is int and cell["cell"] == i, "Cell ordering differs")
        require(type(cell["fold"]) is int and type(cell["seed"]) is int
                and (cell["fold"], cell["seed"], cell["band"]) == (i // 10, (i // 2) % 5, BANDS[i % 2]),
                "Cell fold/seed/band differs")
        require(type(cell["device"]) is int and cell["device"] in range(3), "Invalid device assignment")
        if i % 2:
            require(cell["device"] == c["cells"][i - 1]["device"], "A band pair must use the same GPU")
        for key in ("output", "replay_output"):
            path = Path(cell[key])
            require(path.is_absolute() and path.parent == Path(c["output_root"]), "Outputs must be direct children of output_root")
            outputs.append(str(path))
    require(len(set(outputs)) == 100, "Fit/replay outputs must be unique")
    for record in (c["protocol"], c["catalog"], c["weights"], c["tiles"], c["python"],
                   c["consumer"]["script"], c["helper"], *c["upstream"]["files"].values()):
        file_record(record)
    require(c["helper"]["sha256"] == HELPER_SHA, "Helper pin differs")


def source_inventory(root, commit):
    identity = H.clean_source(root, commit)
    names = H.git(root, "ls-tree", "-r", "--name-only", commit, "titansar", "scripts").splitlines()
    files = {}
    for name in names:
        if name.endswith(".py"):
            raw = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + name])
            files[name] = H.binding(Path(root) / name, hashlib.sha256(raw).hexdigest())
    require(files, "Historical source inventory empty")
    return {**identity, "files": files}


def validate_contract(c):
    """Authenticate source and small reference records before scientific imports."""
    validate_shape(c)
    require(c["consumer"] == {**H.clean_source(ROOT), "script": H.binding(__file__)}, "Consumer is not the frozen clean source")
    require(c["helper"] == H.binding(H.__file__, HELPER_SHA), "Helper origin differs")
    require(c["historical_source"] == source_inventory(c["historical_source"]["path"], HISTORICAL),
            "Historical source inventory changed")
    for item in (c["protocol"], c["catalog"], c["tiles"], c["python"], *c["upstream"]["files"].values()):
        H.check_binding(item)
    require(str(Path(sys.executable).absolute()) == c["python_entrypoint"] and H.binding(sys.executable) == c["python"],
            "Declared virtual-environment entrypoint required")
    spec = H.read(c["protocol"]["path"])["specification"]
    for f, fold in enumerate(c["folds"]):
        expected = spec["folds"][f]
        reference = next(x for x in spec["frozen_jobs"] if x["model"] == "dofa" and x["fold"] == f)
        require(fold["split"]["sha256"] == expected["split_sha256"]
                and fold["reference"]["sha256"] == reference["result_sha256"]
                and fold["analysis_manifest"]["sha256"] == reference["analysis_manifest_sha256"]
                and fold["feature_manifest"]["sha256"] == reference["extraction_manifest_sha256"], "Fold/reference pins differ")
        require(Path(fold["analysis_manifest"]["path"]) == Path(fold["reference"]["path"]).parent / "run_manifest.json"
                and Path(fold["feature_manifest"]["path"]) == Path(fold["features_dir"]).parent / "run_manifest.json",
                "Reference path layout differs")
        for key in ("split", "reference", "analysis_manifest", "feature_manifest"):
            H.check_binding(fold[key])
    for name, expected in spec["scientific_source_sha256"].items():
        require(c["historical_source"]["files"][name]["sha256"] == expected, "Original scientific seal changed")
    return spec


def protected_paths(c, contract_path):
    return [ROOT, contract_path, c["historical_source"]["path"], c["upstream"]["path"],
            Path(c["catalog"]["path"]).parent, c["weights"]["path"], c["protocol"]["path"], c["tiles"]["path"],
            *[Path(f["reference"]["path"]).parent for f in c["folds"]],
            *[Path(f["features_dir"]).parent for f in c["folds"]]]


def check_origins(c):
    """Check actual historical scripts as well as titansar and pinned upstream."""
    result = H.imported_bindings(c)
    root = Path(c["historical_source"]["path"])
    for name, module in list(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts."):
            path = getattr(module, "__file__", None)
            if path is None:
                require(list(map(str, module.__path__)) == [str(root / "scripts")], "Historical scripts namespace shadowed")
            else:
                path = Path(path).resolve()
                require(path.is_relative_to(root), "Historical script import shadowed: " + name)
                rel = str(path.relative_to(root))
                require(rel in c["historical_source"]["files"]
                        and H.binding(path) == c["historical_source"]["files"][rel], "Historical script bytes differ")
                result[name] = H.binding(path)
    return result


def import_runner(c, output, record):
    require(not any(n == "titansar" or n.startswith("titansar.") or n == "scripts" or n.startswith("scripts.")
                    or n in ("dofa_v1", "wave_dynamic_layer") for n in sys.modules), "Fresh scientific worker required")
    sys.addaudithook(H.deny_network)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(output / "unused-pycache")
    sys.path.insert(0, c["historical_source"]["path"])
    runner = importlib.import_module("scripts.run_finetuning")
    record["imports"] = check_origins(c)
    return runner


def historical_args(c, cell):
    fold = c["folds"][cell["fold"]]
    return SimpleNamespace(models=["dofa"], seeds=[cell["seed"]], unfreeze_blocks=[2], epochs=50,
        batch_size=8, grad_accum_steps=4, lr=1e-4, num_workers=2, allow_nondeterministic=False,
        disable_amp=False, device="cuda:0", require_real_weights=True, titan_catalog=c["catalog"]["path"],
        titan_split_manifest=fold["split"]["path"], repaired_protocol=c["protocol"]["path"],
        frozen_reference=fold["reference"]["path"], frozen_features_dir=fold["features_dir"], output_dir=cell["output"])


def tile_inventory(c, cell):
    """Expected hashes cover every used raw image; actual reads occur only on admission."""
    inventory = H.read(c["tiles"]["path"])
    catalog = H.read(c["catalog"]["path"])
    require(inventory["schema"] == "dofa-partial-tiles-v1" and inventory["catalog_sha256"] == CATALOG_SHA,
            "Raw tile inventory catalog differs")
    entries = {e["tile_id"]: e for e in catalog["tiles"]}
    require(len(entries) == len(catalog["tiles"]) and set(inventory["files"]) == set(entries), "Raw tile coverage differs")
    assignments = H.read(c["folds"][cell["fold"]]["split"]["path"])["assignments"]
    require(set(assignments) == set(entries), "Split coverage differs")
    used = {}
    for tile_id, entry in entries.items():
        require(Path(tile_id).name == tile_id and tile_id not in ("", ".", ".."), "Unsafe tile ID")
        bound = inventory["files"][tile_id]
        file_record(bound)
        require(bound["path"] == str(Path(c["catalog"]["path"]).parent / "tiles" / (tile_id + ".npy"))
                and bound["sha256"] == entry.get("tile_sha256", entry.get("sha256")), "Raw tile binding differs")
        if assignments[tile_id] in ("train", "val", "test"):
            H.check_binding(bound)
            st = Path(bound["path"]).stat()
            used[bound["path"]] = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    return used


def check_tile_stats(before):
    for path, expected in before.items():
        st = Path(path).stat()
        require([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == expected,
                "Raw input metadata changed during work: " + path)


class BandLedger:
    """Bounded call evidence: counts plus an ordered digest, not millions of rows."""
    def __init__(self, band):
        require(type(band) in (int, float) and band in BANDS, "Unexpected numeric band")
        self.band = band
        self.stages = {}
        self.pending = None
        self.chain = hashlib.sha256()

    def begin(self, stage, rows, band, autocast, training):
        require(self.pending is None and band == self.band, "Band drift or incomplete previous call")
        require(stage in ("train", "val", "test", "replay_test"), "Unknown forward role")
        require(type(rows) is int and 1 <= rows <= 8, "Unexpected microbatch")
        require(autocast == (stage in ("train", "val")) and training == (stage == "train"), "Forward mode/precision differs")
        self.pending = {"stage": stage, "rows": rows, "numeric_band_input": [band],
                        "autocast": autocast, "training": training}

    def complete(self):
        require(self.pending is not None, "No started band call")
        call = self.pending
        self.chain.update(H.encode(call))
        item = self.stages.setdefault(call["stage"], {"calls": 0, "rows": 0, "first": call})
        item["calls"] += 1
        item["rows"] += call["rows"]
        item["last"] = call
        self.pending = None

    def summary(self):
        return {"numeric_band_input": [self.band], "stages": self.stages,
                "completed_call_sha256": self.chain.hexdigest(), "incomplete_call": self.pending}

    def verify(self, fold, replay=False):
        require(self.pending is None, "Incomplete forward call")
        ntrain, nval, ntest = COUNTS[fold]
        counts = {"replay_test": (ntest, 1)} if replay else {"train": (ntrain, 50), "val": (nval, 50), "test": (ntest, 1)}
        require(set(self.stages) == set(counts), "Forward roles differ")
        for stage, (n, repeats) in counts.items():
            actual = self.stages[stage]
            require(actual["calls"] == repeats * ((n + 7) // 8) and actual["rows"] == repeats * n,
                    "Incomplete epoch/role call coverage: " + stage)


def forward_with_band(forward, model, images, ledger, stage, band, autocast):
    ledger.begin(stage, int(images.shape[0]), band, autocast, bool(model.training))
    result = forward(model, images, wave_list=[band])
    ledger.complete()
    return result


def load_model(c, cell, runner, ledger, record):
    """Local subclass; no historical class/function or instance is patched."""
    torch = runner.torch
    for bound in c["upstream"]["files"].values():
        H.check_binding(bound)
    H.check_binding(c["weights"])
    sys.path.insert(0, c["upstream"]["path"])
    record["rng_before_upstream"] = H.rng_state(torch)
    upstream = importlib.import_module("dofa_v1")
    record["rng_after_upstream"] = H.rng_state(torch)
    require(torch.initial_seed() == 1234, "Upstream import seed side effect differs")
    check_origins(c)
    forward = upstream.OFAViT.forward_features
    require(H.binding(inspect.getsourcefile(forward)) == c["upstream"]["files"]["dofa_v1.py"], "Forward origin differs")
    model = upstream.vit_base_patch16()
    state = torch.load(c["weights"]["path"], map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    require(set(incompatible.missing_keys) == {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}
            and set(incompatible.unexpected_keys) == {"mask_token", "norm.weight", "norm.bias", "projector.weight", "projector.bias"},
            "Accepted checkpoint key coverage differs")
    require(model.global_pool is True and model.fc_norm.eps == 1e-6
            and bool(torch.all(model.fc_norm.weight == 1)) and bool(torch.all(model.fc_norm.bias == 0)), "Pooling/normalization differs")
    require(H.feature_state(model) == c["initial_feature_state"], "Initial pretrained feature state differs")
    wrappers = importlib.import_module("titansar.models.foundation_models")

    class CurrentBandDOFA(wrappers.DOFAWrapper):
        phase = "fit"

        def _forward(self, images):
            stage = ("train" if self.model.training else "val") if self.phase == "fit" else self.phase
            require(self.wavelength == cell["band"], "Wrapper band changed")
            return forward_with_band(forward, self.model, images, ledger, stage, self.wavelength,
                                     torch.is_autocast_enabled("cuda"))

    wrapper = CurrentBandDOFA(device="cuda:0", wavelength=cell["band"])
    wrapper.model = model.to("cuda:0").eval()
    wrapper.weights_source = "pretrained"
    wrapper.weights_sha256, wrapper.model_revision = H.WEIGHTS_SHA, H.UPSTREAM_COMMIT
    record["strict_adapter_source"] = H.binding(inspect.getsourcefile(wrapper._forward))
    require(record["strict_adapter_source"] == c["consumer"]["script"], "Adapter origin differs")
    return wrapper


def runtime(c, cell, runner, wrapper):
    require(sys.version_info[:3] == (3, 12, 3), "Historical Python version differs")
    context = {**c, "launch": {"gpu_uuid": c["devices"][cell["device"]]["uuid"]},
               "historical_runtime": {"platform": "Linux-6.8.0-139-generic"}}
    result = H.runtime(runner.torch, wrapper, context)
    result["imports"] = check_origins(c)
    result["autocast_dtype"] = str(runner.torch.get_autocast_dtype("cuda"))
    require(result["autocast_dtype"] == "torch.float16", "Historical autocast dtype differs")
    require(result["settings"]["deterministic"] and result["settings"]["cudnn_deterministic"]
            and not result["settings"]["cudnn_benchmark"], "Historical deterministic settings differ")
    return result


def compare_runtime(before, after):
    """Training changes feature tensors and RNG state, not this execution contract."""
    for key in ("settings", "platform", "python", "python_entrypoint", "python_binary", "nvidia_smi", "environment", "autocast_dtype"):
        require(before[key] == after[key], "Current runtime differs: " + key)
    for name, package in before["packages"].items():
        require(all(after["packages"][name].get(k) == v for k, v in package.items()), "Package changed: " + name)
    for kind in ("module_paths", "module_files", "mapped_native_files"):
        for name, bound in before["loaded_files"][kind].items():
            require(after["loaded_files"][kind].get(name) == bound, "Loaded dependency changed: " + name)
    for name, bound in before["imports"].items():
        require(after["imports"].get(name) == bound, "Scientific import changed: " + name)


def native_json(value):
    """Strict finite JSON; unlike the historical serializer, never turn NaN into null."""
    if isinstance(value, dict):
        return {str(k): native_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native_json(x) for x in value]
    if hasattr(value, "tolist"):
        return native_json(value.tolist())
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite scientific result")
    require(value is None or type(value) in (str, bool, int, float), "Unsupported JSON value")
    return value


def tensor_state(module):
    """Full encoder identity includes the unused original head as well."""
    hashed = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        hashed.update(H.encode({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}))
        hashed.update(array.tobytes())
    return hashed.hexdigest()


def complete_output(c, cb, index, replay=False):
    cell = c["cells"][index]
    output = Path(cell["replay_output"] if replay else cell["output"])
    supervisor = H.read(output / "supervisor-terminal.json")
    worker = H.read(output / "worker-terminal.json")
    required = {"initial.json", "runtime-final.json", "band-calls.json"}
    required |= {"replay.json"} if replay else {"selected.pt", "history.json", "result.json"}
    require(required <= set(worker["files"]) and "worker-terminal.json" in supervisor["files"],
            "Required retained output inventory incomplete")
    for terminal in (supervisor, worker):
        require(terminal["status"] == "complete" and terminal["contract"] == cb
                and terminal["cell"] == index and terminal["mode"] == ("replay" if replay else "fit"), "Required job incomplete or mismatched")
        for name, bound in terminal["files"].items():
            require(Path(bound["path"]) == output / name and Path(name).name == name,
                    "Terminal file points outside the declared attempt")
            H.check_binding(bound)
    require(supervisor["process_exit"] == 0, "Required process failed")
    name = "replay.json" if replay else "result.json"
    return H.binding(output / name)


def pilot_gate(c, cb, acceptance):
    require(acceptance is not None, "Remaining 48 cells require independent pilot acceptance")
    H.check_binding(acceptance)
    receipt = H.read(acceptance["path"])
    require(receipt["schema"] == "dofa-partial-pilot-acceptance-v1" and receipt["status"] == "accepted"
            and receipt["contract"] == cb and receipt["cells"] == [0, 1]
            and receipt["selection_by_test_score"] is False, "Pilot acceptance contract differs")
    for key in ("checker", "report"):
        H.check_binding(receipt[key])
    expected = [{"cell": i, "fit": complete_output(c, cb, i), "replay": complete_output(c, cb, i, True)} for i in (0, 1)]
    require(receipt["outputs"] == expected, "Pilot acceptance output bindings differ")
    return acceptance


def fit(runner, c, cell, data, wrapper, head, provenance, ledger, output, record):
    train, train_loader, val_loader, test_loader = data["loaders"]
    started = time.monotonic()
    wrapper.phase = "fit"
    model, probe, best_val, history = runner.train_finetuned_model(
        model=wrapper, train_loader=train_loader, val_loader=val_loader, num_classes=6,
        unfreeze_blocks=2, lr=1e-4, epochs=50, class_weights=train.get_class_weights(),
        device="cuda:0", use_amp=True, grad_accum_steps=4, return_history=True, init_probe=head)
    try:
        provenance.update(runner.repaired_history_metadata(history, len(train), best_val))
    except BaseException:
        # Preserve returned evidence before the original rejection escapes.
        # Text permits NaN/Inf diagnostics without weakening accepted JSON.
        try:
            diagnostic = "UNVALIDATED returned history; not accepted scientific JSON.\n"
            diagnostic += repr({"best_validation_macro_recall": best_val, "history": history}) + "\n"
            H.publish_bytes(output / "returned-history-unvalidated.txt", diagnostic.encode())
        except BaseException as diagnostic_error:
            record["returned_history_diagnostic_error"] = {
                "type": type(diagnostic_error).__name__, "message": str(diagnostic_error)}
        raise
    H.publish(output / "history.json", native_json({"history": history, "selection": provenance}))
    wrapper.phase = "test"
    test = runner.evaluate_finetuned(model, probe, test_loader, 6, "cuda:0")
    metrics = runner._check_prediction(test, test_loader.dataset)
    ledger.verify(cell["fold"])
    for module in (model.model, probe):
        require(all(bool(runner.torch.isfinite(t).all()) for t in module.state_dict().values()), "Nonfinite selected state")
    runner.save_checkpoint(output / "selected.pt", model, probe, cell["seed"], 2, best_val,
                           provenance=native_json(provenance))
    result = {"schema": "dofa-partial-result-v1", "cell": cell, "provenance": provenance,
              "test": test, "metrics": metrics, "best_validation_macro_recall": best_val,
              "time_seconds": time.monotonic() - started, "band_calls": ledger.summary()}
    H.publish(output / "result.json", native_json(result))
    record["result"] = H.binding(output / "result.json")


def replay(runner, c, cell, data, wrapper, head, provenance, ledger, output, record):
    fit_root = Path(cell["output"])
    expected = H.read(fit_root / "result.json")
    state = runner.torch.load(fit_root / "selected.pt", map_location="cpu", weights_only=True)
    require(state["provenance"] == expected["provenance"] and state["seed"] == cell["seed"]
            and state["unfreeze_blocks"] == 2 and expected["cell"] == cell, "Selected checkpoint provenance differs")
    require(state["provenance"]["numeric_band_input"] == [cell["band"]], "Selected checkpoint band differs")
    require(state["best_val_accuracy"] == expected["best_validation_macro_recall"], "Checkpoint selection score differs")
    saved_history = H.read(fit_root / "history.json")
    require(saved_history["selection"] == state["provenance"], "History/checkpoint selection metadata differs")
    selection = runner.repaired_history_metadata(saved_history["history"], len(data["loaders"][0]), state["best_val_accuracy"])
    require(all(state["provenance"].get(k) == v for k, v in selection.items()), "Selected epoch/history budget differs")
    wrapper.model.load_state_dict(state["model_state_dict"], strict=True)
    head.load_state_dict(state["probe_state_dict"], strict=True)
    head = head.to("cuda:0").eval()
    for parameter in wrapper.model.parameters():
        parameter.requires_grad = False
    wrapper.model.eval()
    wrapper.phase = "replay_test"
    test_loader = data["loaders"][3]
    test = runner.evaluate_finetuned(wrapper, head, test_loader, 6, "cuda:0")
    runner._check_prediction(test, test_loader.dataset)
    ledger.verify(cell["fold"], replay=True)
    require(H.encode(native_json(test)) == H.encode(expected["test"]), "Selected-state test replay differs; no tolerance or retry")
    H.publish(output / "replay.json", {"schema": "dofa-partial-replay-v1", "cell": cell,
        "result": H.binding(fit_root / "result.json"), "checkpoint": H.binding(fit_root / "selected.pt"),
        "test_equal": True, "band_calls": ledger.summary(), "independent_acceptance": False})


def worker(c, cb, cell, output, record, mode):
    runner = import_runner(c, output, record)
    raw_stats = tile_inventory(c, cell)
    record["raw_input_inventory"] = c["tiles"]
    record["used_raw_input_count"] = len(raw_stats)
    # This is the original gate, including its entire 13.78 reference/source seal.
    data = runner.prepare_repaired_run(historical_args(c, cell))
    require(not runner.torch.cuda.is_initialized(), "Historical preflight initialized CUDA")
    runner.seed_everything(cell["seed"], deterministic=True)
    ledger = BandLedger(cell["band"])
    try:
        wrapper = load_model(c, cell, runner, ledger, record)
        head, trainable = runner.repaired_training_metadata(wrapper, cell["seed"])
        require(trainable["initial_head_state_sha256"] == HEADS[cell["seed"]]
                and trainable["unfrozen_blocks"] == ["blocks.10", "blocks.11"]
                and trainable["trainable_backbone_parameters"] == 14175744
                and trainable["trainable_head_parameters"] == 4614, "Initial head/trainable selection differs")
        initial = {"feature_state": H.feature_state(wrapper.model), "full_encoder_state_sha256": tensor_state(wrapper.model), "head": trainable,
                   "runtime": runtime(c, cell, runner, wrapper), "reference_provenance": data["provenance"]}
        require(initial["feature_state"] == c["initial_feature_state"], "Initial state changed on device transfer")
        H.publish(output / "initial.json", initial)
        comparison = None
        if mode == "replay":
            comparison = H.read(Path(cell["output"]) / "initial.json")
        elif cell["cell"] % 2:
            comparison = H.read(Path(c["cells"][cell["cell"] - 1]["output"]) / "initial.json")
        if comparison is not None:
            require(initial["feature_state"] == comparison["feature_state"]
                    and initial["full_encoder_state_sha256"] == comparison["full_encoder_state_sha256"]
                    and initial["head"] == comparison["head"]
                    and initial["reference_provenance"] == comparison["reference_provenance"], "Paired initial state/reference differs")
            compare_runtime(comparison["runtime"], initial["runtime"])
        # Historical metadata stays nested; it is not relabeled as 3.75 provenance.
        provenance = {"contract": cb, "numeric_band_input": [cell["band"]], "band_semantics": "numeric identifier, no physical conversion",
                      "historical_reference": data["provenance"], "current_recipe": POLICY, "initial": trainable,
                      "initial_feature_state": initial["feature_state"], "cell": cell["cell"],
                      "training_class_weights": native_json(data["loaders"][0].get_class_weights())}
        (fit if mode == "fit" else replay)(runner, c, cell, data, wrapper, head, provenance, ledger, output, record)
        final = runtime(c, cell, runner, wrapper)
        compare_runtime(initial["runtime"], final)
        if mode == "fit" and cell["cell"] % 2:
            control_final = H.read(Path(c["cells"][cell["cell"] - 1]["output"]) / "runtime-final.json")
            # Check both directions so libraries loaded only during fitting are
            # covered, including additions absent from either initial snapshot.
            compare_runtime(control_final, final)
            compare_runtime(final, control_final)
        H.publish(output / "runtime-final.json", final)
        record["imports_final"] = check_origins(c)
        check_tile_stats(raw_stats)
        validate_contract(c)
        H.check_binding(c["weights"])
    finally:
        H.publish(output / "band-calls.json", ledger.summary())


def admitted_dependencies(c, cb, cell, mode, acceptance):
    if cell["cell"] >= 2:
        pilot_gate(c, cb, acceptance)
    if mode == "replay":
        complete_output(c, cb, cell["cell"])
    elif cell["cell"] % 2:
        complete_output(c, cb, cell["cell"] - 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-plan", "fit", "replay", "_fit-worker", "_replay-worker"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--cell", type=int, choices=range(50))
    parser.add_argument("--pilot-acceptance", type=Path)
    parser.add_argument("--pilot-acceptance-sha256")
    args = parser.parse_args(argv)
    cb = H.binding(args.contract, args.contract_sha256)
    c = H.read(args.contract)
    validate_shape(c)
    if args.mode == "check-plan":
        validate_contract(c)
        print(json.dumps({"status": "source_and_reference_contract_checked", "scientific_preflight": False}))
        return 0
    require(args.cell is not None, "A single declared cell is required")
    require(bool(args.pilot_acceptance) == bool(args.pilot_acceptance_sha256), "Both pilot acceptance arguments required")
    acceptance = H.binding(args.pilot_acceptance, args.pilot_acceptance_sha256) if args.pilot_acceptance else None
    cell = c["cells"][args.cell]
    worker_mode = args.mode.startswith("_")
    mode = "replay" if "replay" in args.mode else "fit"
    path = Path(cell["replay_output"] if mode == "replay" else cell["output"])
    if worker_mode:
        started = H.read(path / "supervisor-started.json")
        require(os.getppid() == started["pid"], "Worker requires its live foreground supervisor")
    else:
        path = H.fresh_output(path, protected_paths(c, args.contract))

    def action(record):
        record.update(contract=cb, consumer=c["consumer"], historical_source=c["historical_source"],
                      cell=args.cell, mode=mode, pilot_acceptance=acceptance)
        validate_contract(c)
        admitted_dependencies(c, cb, cell, mode, acceptance)
        if worker_mode:
            worker(c, cb, cell, path, record, mode)
        else:
            device = c["devices"][cell["device"]]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env.update(H.THREAD_ENV)
            env.update(CUDA_VISIBLE_DEVICES=device["uuid"], PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            command = [c["python_entrypoint"], "-B", str(Path(__file__).resolve()), "_" + mode + "-worker",
                       "--contract", str(args.contract.resolve()), "--contract-sha256", args.contract_sha256,
                       "--cell", str(args.cell)]
            if acceptance:
                command += ["--pilot-acceptance", acceptance["path"], "--pilot-acceptance-sha256", acceptance["sha256"]]
            H.supervise(command, env, path, device["lock"], c[mode + "_timeout_seconds"], record)
            terminal = H.read(path / "worker-terminal.json")
            require(terminal["status"] == "complete", "Worker did not finish its contract")
    return H.attempt(path, "worker" if worker_mode else "supervisor", action)


if __name__ == "__main__":
    raise SystemExit(main())
