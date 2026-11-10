"""Synthetic runner admission/refusal tests; no pretrained model or GPU execution."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import loss_comparison as comparison
from scripts import model_followup_support as support
from scripts import run_model_followup as runner
from tests.test_model_followup import TinyImages, cpu_environment, epoch_row, fixture_spec, seal_runtime
from titansar.models.followup import RESNET_WEIGHTS, state_digest
from titansar.reproducibility import artifact_hashes


@pytest.fixture
def control_bundle(tmp_path, monkeypatch):
    """A small explicitly synthetic accepted-control shape, with real file hashes."""
    spec = fixture_spec(tmp_path)
    spec["recipe"]["lr"] = 1e-4
    checkpoint = tmp_path / "weights" / "fixture.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic checkpoint; model construction is replaced")
    spec["model"].update(initialization="pretrained", weights=RESNET_WEIGHTS,
                         checkpoint=support.binding(checkpoint))
    monkeypatch.setattr(runner, "RESNET18_SHA256", spec["model"]["checkpoint"]["sha256"])
    for name in ("_lazy_init", "set_device", "get_device_properties", "device_count"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: pytest.fail("CPU fixture queried CUDA"))

    def builder(model_spec, device):
        assert device == "cpu"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_spec["head_seed"])
            model = TinyImages()
        return model, {"model": "resnet18", "initialization": "pretrained",
                       "synthetic_fixture": True, "initial_state_sha256": state_digest(model.state_dict())}

    monkeypatch.setattr(runner, "build_resnet18", builder)
    inputs = support.validate_inputs(spec)
    original_source = {"commit": comparison.ORIGINAL_COMMIT, "files": dict(comparison.ORIGINAL_FILES)}
    seal_runtime(spec)
    prior = copy.deepcopy(spec)
    prior["source"] = original_source
    root = tmp_path / "control"
    root.mkdir()
    prior["output_dir"] = str(root)
    prior["tuning_results"] = [{"path": str(tmp_path / f"unavailable-tuning-{i}.json"), "sha256": "a" * 64}
                               for i in range(3)]
    model, identity = builder(spec["model"], "cpu")
    torch.save(model.state_dict(), root / "initial_state.pt")
    for name in ("selected.pt", "final.pt", "started.json", "preparation.json", "epoch-001.json", "epoch-002.json"):
        (root / name).write_bytes(b"synthetic inventoried artifact; not deserialized")
    history = [dict(epoch_row(epoch, np.eye(6, dtype=int), steps=2), examples=6, finite_gradients=True)
               for epoch in (1, 2)]
    control = {"status": "complete", "kind": "resnet18", "purpose": "synthetic_cpu_test", "fold": 0,
               "specification": prior, "inputs": inputs, "identity": identity,
               "training_class_weights": inputs["training_class_weights"], "history": history,
               "selected_epoch": 1, "test": {**inputs["ordered"]["test"], "class_order": list(range(6))},
               "selection": {"lr": 1e-4, "rule": "maximum validation macro_f1, then lower LR",
                             "candidates": [{"lr": lr, "score": score, "result": binding}
                                            for lr, score, binding in zip(support.CANDIDATE_LRS, (.9, .7, .6),
                                                                         prior["tuning_results"])]}}
    spec["loss_comparison"] = {"schema_version": "population-weight-v1",
                               "original_source": copy.deepcopy(original_source), "control": {}}

    def publish():
        control["specification_sha256"] = support.digest(control["specification"])
        control["outputs"] = artifact_hashes(root, exclude=["result.json"])
        (root / "result.json").write_text(json.dumps(control))
        spec["loss_comparison"]["control"] = support.binding(root / "result.json")

    publish()
    return spec, inputs, control, root, publish


def test_preparation_reuses_receipt_without_model_or_tuning(control_bundle, monkeypatch):
    spec, inputs, control, root, _ = control_bundle
    before = artifact_hashes(root)
    monkeypatch.setattr(runner, "build_resnet18", lambda *a: pytest.fail("preparation built model"))
    monkeypatch.setattr(runner, "_selected_lr", lambda *a: pytest.fail("comparison reopened tuning"))
    prepared = runner.prepare(spec)
    assert prepared["ready"] and not prepared["model_checks_executed"]
    assert prepared["selection"] == control["selection"]
    normalizer = prepared["loss_comparison"]["normalizer"]
    assert normalizer["native_weights"] == inputs["training_class_weights"]
    assert normalizer["class_counts"] == [1] * 6 and normalizer["m32"] == 1.0
    assert "scripts/loss_comparison.py" in prepared["source_files"]
    assert artifact_hashes(root) == before
    assert not Path(spec["output_dir"]).exists()


@pytest.mark.parametrize("fault", ["schema", "extra", "source_extra", "source_missing", "scratch", "phase", "lr", "tuning"])
def test_comparison_schema_refusals(control_bundle, fault):
    spec, *_ = control_bundle
    field = spec["loss_comparison"]
    if fault == "schema": field["schema_version"] = "unknown"
    elif fault == "extra": field["unrequested"] = True
    elif fault == "source_extra": field["original_source"]["files"]["extra.py"] = "a" * 64
    elif fault == "source_missing": field["original_source"]["files"].pop("scripts/run_model_followup.py")
    elif fault == "scratch": spec["model"]["initialization"] = "scratch"
    elif fault == "phase": spec["phase"] = "tuning"
    elif fault == "lr": spec["recipe"]["lr"] = .001
    else: spec["tuning_results"] = []
    with pytest.raises(ValueError): runner.prepare(spec)
    assert not Path(spec["output_dir"]).exists()


@pytest.mark.parametrize("fault", ["commit", "file_hash", "cpu"])
def test_scientific_comparison_requires_exact_original_source_and_gpu(control_bundle, fault):
    spec, *_ = control_bundle
    spec.update(purpose="titan_followup", device="cuda:0")
    if fault == "commit": spec["loss_comparison"]["original_source"]["commit"] = "0" * 40
    elif fault == "file_hash": spec["loss_comparison"]["original_source"]["files"]["scripts/run_model_followup.py"] = "0" * 64
    else: spec["device"] = "cpu"
    # Pure host-side gate: no attempt to construct the synthetic fixture as Titan.
    with pytest.raises(ValueError): comparison.validate_spec(spec)


@pytest.mark.parametrize("fault", ["fold", "seed", "loader_seed", "model", "runtime", "recipe", "device",
                                   "execution", "source", "source_hash", "top_fold", "spec_digest", "inputs",
                                   "weights", "history", "updates", "examples", "selected_epoch", "test",
                                   "selection", "candidate", "nested_comparison", "missing_artifact"])
def test_resealed_control_metadata_must_match(control_bundle, fault):
    spec, inputs, control, root, publish = control_bundle
    prior = control["specification"]
    if fault in ("fold", "seed", "loader_seed"): prior[fault] += 1
    elif fault == "model": prior["model"]["head_seed"] += 1
    elif fault == "runtime": prior["runtime"]["threads"] = 17
    elif fault == "recipe": prior["recipe"]["lr"] = .001
    elif fault == "device": prior["device"] = "cuda:0"
    elif fault == "execution": prior["execution"] = {"gpu_uuid": "other"}
    elif fault == "source": prior["source"]["commit"] = "0" * 40
    elif fault == "source_hash": prior["source"]["files"]["scripts/run_model_followup.py"] = "0" * 64
    elif fault == "top_fold": control["fold"] = 1
    elif fault == "inputs": control["inputs"] = {**inputs, "normalization_sha256": "0" * 64}
    elif fault == "weights": control["training_class_weights"] = [2.] * 6
    elif fault == "history": control["history"].pop()
    elif fault == "updates": control["history"][0]["optimizer_steps"] = 0
    elif fault == "examples": control["history"][0]["examples"] = 5
    elif fault == "selected_epoch": control["selected_epoch"] = 2
    elif fault == "test": control["test"]["tile_ids"] = list(reversed(control["test"]["tile_ids"]))
    elif fault == "selection": control["selection"]["lr"] = .001
    elif fault == "candidate": control["selection"]["candidates"][0]["result"] = {"path": "other", "sha256": "0" * 64}
    elif fault == "nested_comparison": prior["loss_comparison"] = copy.deepcopy(spec["loss_comparison"])
    elif fault == "missing_artifact": (root / "final.pt").unlink()
    publish()
    if fault == "spec_digest":
        control["specification_sha256"] = "0" * 64
        (root / "result.json").write_text(json.dumps(control))
        spec["loss_comparison"]["control"] = support.binding(root / "result.json")
    with pytest.raises(ValueError): comparison.prepare(spec, inputs, runner._completed)


@pytest.mark.parametrize("fault", ["modified", "missing", "extra"])
def test_control_full_inventory_is_authenticated(control_bundle, fault):
    spec, inputs, _, root, _ = control_bundle
    if fault == "modified": (root / "epoch-002.json").write_bytes(b"changed")
    elif fault == "missing": (root / "epoch-002.json").unlink()
    else: (root / "unlisted.json").write_text("{}")
    with pytest.raises(ValueError, match="Producer inventory differs"):
        comparison.prepare(spec, inputs, runner._completed)


@pytest.mark.parametrize("fault", [None, "uuid", "device", "receipt"])
def test_original_gpu_record_binding_without_cuda_execution(control_bundle, tmp_path, fault):
    # Metadata-only exercise of the scientific receipt gate, not admission of
    # these synthetic inputs/runtime through the full scientific runner.
    spec, inputs, control, root, publish = control_bundle
    receipt = tmp_path / "cuda" / "receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("{}")
    spec.update(purpose="titan_followup", device="cuda:0",
                execution={"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
                           "cuda_dependencies": support.binding(receipt)})
    spec["recipe"].update(epochs=100, batch_size=32)
    prior = control["specification"]
    for key in ("purpose", "device", "execution", "recipe"):
        prior[key] = copy.deepcopy(spec[key])
    control["purpose"] = spec["purpose"]
    control["history"] = [dict(epoch_row(epoch, np.eye(6, dtype=int)), examples=6, finite_gradients=True)
                          for epoch in range(1, 101)]
    for epoch in range(3, 101): (root / f"epoch-{epoch:03d}.json").write_text("{}")
    (root / "gpu-started.json").write_text("{}")
    gpu = {"uuid": spec["execution"]["gpu_uuid"], "device": "cuda:0",
           "cuda_dependencies": copy.deepcopy(spec["execution"]["cuda_dependencies"])}
    control["gpu_execution"] = gpu
    if fault == "uuid": gpu["uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    elif fault == "device": gpu["device"] = "cuda:1"
    elif fault == "receipt": gpu["cuda_dependencies"]["sha256"] = "0" * 64
    publish()
    if fault:
        with pytest.raises(ValueError, match="Original control GPU/receipt differs"):
            comparison.prepare(spec, inputs, runner._completed)
    else:
        assert comparison.prepare(spec, inputs, runner._completed)["selection"] == control["selection"]


@pytest.mark.parametrize("location", ["same", "ancestor", "descendant", "symlink", "sibling"])
def test_control_output_directory_is_protected(control_bundle, tmp_path, monkeypatch, location):
    spec, _, _, root, _ = control_bundle
    output = {"same": root, "ancestor": root.parent, "descendant": root / "new",
              "sibling": root.parent / "new"}.get(location)
    if location == "symlink":
        alias = tmp_path / "control-alias"
        alias.symlink_to(root, target_is_directory=True)
        output = alias / "new"
    spec["output_dir"] = str(output)
    before = artifact_hashes(root)
    if location == "sibling": assert support.checked_output(spec) == output
    else:
        monkeypatch.setattr(runner, "prepare", lambda *a: pytest.fail("prepared before output protection"))
        with pytest.raises(ValueError, match="Output overlaps"):
            runner.train_resnet(spec)
    assert artifact_hashes(root) == before


@pytest.mark.parametrize("fault", ["identity", "built_tensor", "saved_tensor", "artifact_bytes"])
def test_initial_state_refused_before_loader_or_optimizer(control_bundle, monkeypatch, fault):
    spec, _, _, root, publish = control_bundle
    if fault == "saved_tensor":
        saved = torch.load(root / "initial_state.pt", weights_only=True)
        saved["fc.weight"][0, 0] += 1
        torch.save(saved, root / "initial_state.pt")
        publish()  # File is authentically bound; its tensor content is still wrong.
    prepared = runner.prepare(spec)
    build = runner.build_resnet18

    def changed(*args):
        model, identity = build(*args)
        if fault == "identity": identity["initial_state_sha256"] = "0" * 64
        elif fault == "built_tensor":
            with torch.no_grad(): model.fc.weight[0, 0] += 1
        elif fault == "artifact_bytes": (root / "initial_state.pt").write_bytes(b"changed after preparation")
        return model, identity

    monkeypatch.setattr(runner, "build_resnet18", changed)
    monkeypatch.setattr(support, "make_loaders", lambda *a, **k: pytest.fail("loader before initial-state gate"))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer before initial-state gate"))
    with pytest.raises(ValueError): runner._train(spec, prepared, Path(spec["output_dir"]))
    assert not Path(spec["output_dir"]).exists()


def test_initial_state_refusal_preserves_attempt_failure_record(control_bundle, monkeypatch):
    spec, _, _, root, _ = control_bundle
    before = artifact_hashes(root)
    build = runner.build_resnet18

    def changed(*args):
        model, identity = build(*args)
        identity["initial_state_sha256"] = "0" * 64
        return model, identity

    monkeypatch.setattr(runner, "build_resnet18", changed)
    monkeypatch.setattr(support, "make_loaders", lambda *a, **k: pytest.fail("trained after initial refusal"))
    with pytest.raises(ValueError, match="Built identity differs"):
        runner.train_resnet(spec)
    output = Path(spec["output_dir"])
    assert json.loads((output / "failed.json").read_text())["error_type"] == "ValueError"
    assert not (output / "result.json").exists() and not (output / "initial_state.pt").exists()
    assert artifact_hashes(root) == before


@pytest.mark.parametrize("population", [False, True])
def test_tiny_runner_keeps_native_calls_and_separates_batch_records(control_bundle, monkeypatch, population):
    spec, _, _, root, _ = control_bundle
    if not population: del spec["loss_comparison"]
    before = artifact_hashes(root)
    calls = []
    train = runner.train_epoch

    def observe(*args, **kwargs):
        calls.append(set(kwargs))
        return train(*args, **kwargs)

    monkeypatch.setattr(runner, "train_epoch", observe)
    result = runner.train_resnet(spec)
    assert result["status"] == "complete" and len(result["history"]) == 2
    assert "test" in result
    output = Path(spec["output_dir"])
    assert artifact_hashes(root) == before
    runner._completed(support.binding(output / "result.json"), "resnet18")
    assert calls == [{"class_weights", "device"} | ({"population_mean_weight", "record_batches"} if population else set())] * 2
    assert ("normalizer" in result) is population
    for row in result["history"]:
        assert "batch_records" not in row
        assert ("training_order_sha256" in row) is population
        path = output / f"batches-{row['epoch']:03d}.json"
        assert path.exists() is population
        if population:
            batches = json.loads(path.read_text())
            assert batches["epoch"] == row["epoch"]
            assert [b["examples"] for b in batches["batch_records"]] == [4, 2]
            assert len(row["training_order_sha256"]) == 64
            assert all(b["optimization_denominator"] == b["examples"] * result["normalizer"]["m32"]
                       for b in batches["batch_records"])
