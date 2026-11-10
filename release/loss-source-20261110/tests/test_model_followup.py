"""Synthetic contract tests; no Titan model experiment or pretrained claim."""
import copy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision
from sklearn.preprocessing import StandardScaler

from scripts import model_followup_support as support
from scripts import run_model_followup as runner
from titansar.models import followup
from titansar.models.image_training import evaluate_images, train_epoch
from titansar.reproducibility import artifact_hashes, git_state, sha256_file


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch):
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("XFORMERS_DISABLED", "1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
    torch.set_num_threads(threads)


def source_lock():
    return {"commit": git_state(support.ROOT)["commit"],
            "files": {p: sha256_file(support.ROOT / p) for p in support.SOURCE_FILES}}


def resnet_spec(initialization="scratch", seed=0):
    module = importlib.import_module("torchvision.models.resnet")
    return {"initialization": initialization, "backbone_seed": seed, "head_seed": 20000 + seed,
            "torchvision_version": torchvision.__version__, "resnet_source_sha256": sha256_file(Path(module.__file__))}


def fixture_spec(tmp_path):
    inputs = tmp_path / "inputs"
    tiles = inputs / "tiles"
    tiles.mkdir(parents=True)
    entries = []
    for role in support.ROLES:
        for label in range(6):
            tid = f"{role}_{label}"
            path = tiles / f"{tid}.npy"
            np.save(path, np.linspace(1 + label, 200 + label, 128 * 128, dtype=np.float32).reshape(128, 128))
            entries.append({"tile_id": tid, "label": label, "split": role, "center_lat": 0., "center_lon": 0.,
                            "tile_sha256": sha256_file(path)})
    catalog = inputs / "catalog.json"
    catalog.write_text(json.dumps({"tiles": entries, "intensity_space": "hisar_log_dn",
                                   "benchmark_track": "synthetic_fixture", "split_policy": "external_manifest"}))
    split = inputs / "split.json"
    split.write_text(json.dumps({"catalog_sha256": sha256_file(catalog),
                                "policy": {"test_fold": 0},
                                "assignments": {e["tile_id"]: e["split"] for e in entries}}))
    spec = {"schema_version": "model-followup-v1", "kind": "resnet18", "purpose": "synthetic_cpu_test",
            "phase": "final", "fold": 0, "seed": 0, "loader_seed": 10000, "device": "cpu",
            "output_dir": str(tmp_path / "output"), "model": resnet_spec(),
            "inputs": {"catalog": support.binding(catalog), "split": support.binding(split),
                       "benchmark_track": "synthetic_fixture",
                       "counts": {role: {"total": 6, "class_counts": [1] * 6} for role in support.ROLES}},
            "recipe": {**support.RECIPE, "epochs": 2, "batch_size": 4, "lr": .001}}
    spec["inputs"]["normalization_sha256"] = support.validate_inputs(spec)["normalization_sha256"]
    return spec


def seal_runtime(spec):
    spec["source"] = source_lock()
    spec["runtime"] = support.runtime_identity()
    return spec


class TinyImages(torch.nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.fc = torch.nn.Linear(3, classes)

    def forward(self, x):
        return self.fc(x.mean(dim=(-2, -1)))


def test_transform_matches_independent_bilinear_and_channel_math():
    values = torch.arange(128, dtype=torch.float32) / 127
    image = values[None, None, None, :].expand(1, 1, 128, 128).contiguous()
    actual = followup.prepare_rgb224(image)
    # Analytic align_corners=False interpolation of a linear horizontal ramp.
    locations = ((torch.arange(224, dtype=torch.float64) + .5) * 128 / 224 - .5).clamp(0, 127) / 127
    for channel, (mean, std) in enumerate(zip(followup.TRANSFORM["mean"], followup.TRANSFORM["std"])):
        # The float64 analytic ramp avoids copying Torch's interpolation algorithm;
        # allow eight float32 rounding units through interpolation/standardization.
        torch.testing.assert_close(actual[0, channel].double(), ((locations - mean) / std).expand(224, 224),
                                   atol=8 * torch.finfo(torch.float32).eps / std, rtol=0)
    inherited = followup.DINOv2Wrapper("cpu").prepare_input(image)
    assert torch.equal(actual, inherited)
    with pytest.raises(ValueError, match="normalized"):
        followup.prepare_rgb224(torch.full_like(image, float("nan")))


def test_normalization_and_independent_loader_order(tmp_path):
    spec = fixture_spec(tmp_path)
    before = support.validate_inputs(spec)
    first = support.make_loaders(spec, roles=("train", "val"), shuffle_train=True, seed=10000)
    other = support.make_loaders(spec, roles=("train",), shuffle_train=True, seed=10000)
    def order(loader):
        return [i for batch in loader for i in batch["tile_id"]]
    assert order(first["train"]) == order(other["train"])
    order(first["val"])
    assert order(first["train"]) == order(other["train"])
    catalog = Path(spec["inputs"]["catalog"]["path"])
    content = json.loads(catalog.read_text())
    for row in content["tiles"]:
        if row["split"] != "train":
            path = catalog.parent / "tiles" / f"{row['tile_id']}.npy"
            np.save(path, np.full((128, 128), 1e6, dtype=np.float32))
            row["tile_sha256"] = sha256_file(path)
    catalog.write_text(json.dumps(content))
    spec["inputs"]["catalog"] = support.binding(catalog)
    split = Path(spec["inputs"]["split"]["path"])
    content = json.loads(split.read_text()); content["catalog_sha256"] = sha256_file(catalog)
    split.write_text(json.dumps(content)); spec["inputs"]["split"] = support.binding(split)
    after = support.validate_inputs(spec)
    assert before["normalization"] == after["normalization"]
    assert before["training_class_weights"] == after["training_class_weights"]


@pytest.mark.parametrize("fault", ["tile_bytes", "duplicate", "missing_class", "counts", "normalization", "role"])
def test_input_mutations_fail_before_model(tmp_path, monkeypatch, fault):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(runner, "build_resnet18", lambda *a: pytest.fail("constructed model"))
    if fault == "tile_bytes":
        np.save(tmp_path / "inputs/tiles/test_5.npy", np.ones((128, 128), dtype=np.float32))
    elif fault in ("duplicate", "missing_class"):
        path = Path(spec["inputs"]["catalog"]["path"])
        content = json.loads(path.read_text())
        if fault == "duplicate": content["tiles"].append(content["tiles"][0])
        else: content["tiles"][0]["label"] = 1
        path.write_text(json.dumps(content)); spec["inputs"]["catalog"] = support.binding(path)
        split = Path(spec["inputs"]["split"]["path"])
        content = json.loads(split.read_text()); content["catalog_sha256"] = sha256_file(path)
        split.write_text(json.dumps(content)); spec["inputs"]["split"] = support.binding(split)
    elif fault == "counts": spec["inputs"]["counts"]["test"]["total"] = 7
    elif fault == "normalization": spec["inputs"]["normalization_sha256"] = "0" * 64
    else:
        path = Path(spec["inputs"]["split"]["path"])
        content = json.loads(path.read_text()); content["assignments"]["test_0"] = "unknown"
        path.write_text(json.dumps(content)); spec["inputs"]["split"] = support.binding(path)
    with pytest.raises(ValueError): support.validate_inputs(spec)


def test_training_step_matches_independent_weighted_gradient():
    torch.manual_seed(5)
    model = TinyImages(classes=2)
    reference = copy.deepcopy(model)
    images = torch.stack([torch.full((1, 128, 128), v) for v in (.1, .3, .7)])
    labels, weights = torch.tensor([0, 1, 1]), torch.tensor([1., 3.])
    x = followup.prepare_rgb224(images).mean((-2, -1))
    logits = x @ reference.fc.weight.T + reference.fc.bias
    p = logits.softmax(1)
    residual = p - torch.nn.functional.one_hot(labels, 2)
    residual *= weights[labels, None] / weights[labels].sum()
    expected_weight = reference.fc.weight.detach() - .01 * residual.T @ x
    expected_bias = reference.fc.bias.detach() - .01 * residual.sum(0)
    result = train_epoch(model, [{"image": images, "label": labels}], torch.optim.SGD(model.parameters(), lr=.01),
                         class_weights=weights)
    torch.testing.assert_close(model.fc.weight, expected_weight)
    torch.testing.assert_close(model.fc.bias, expected_bias)
    assert result["examples"] == 3 and result["optimizer_steps"] == 1 and result["loss_denominator"] == 7


def test_validation_uses_logits_and_global_weighted_loss():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([0., 1e-8, -30., -30., -30., -30.]).expand(len(x), -1)
    batch = {"image": torch.zeros(6, 1, 128, 128), "label": torch.arange(6), "tile_id": list("abcdef")}
    batches = [{k: v[:4] for k, v in batch.items()}, {k: v[4:] for k, v in batch.items()}]
    weights = torch.arange(1, 7, dtype=torch.float32)
    result = evaluate_images(Fixed(), batches, class_weights=weights)
    assert result["predictions"] == [1] * 6
    assert np.asarray(result["probabilities"]).argmax(1).tolist() == [0] * 6
    native = [float(v) for v in Fixed()(batch["image"])[0]]
    maximum = max(native)
    log_normalizer = maximum + math.log(sum(math.exp(v - maximum) for v in native))
    per_example = [log_normalizer - native[label] for label in range(6)]
    loss = sum((i + 1) * value for i, value in enumerate(per_example)) / 21
    # Independent scalar float64 oracle. Eight FP32 rounding units cover the
    # log/weighted sums/division; an unweighted average of batch means fails.
    bound = 8 * np.finfo(np.float32).eps * max(per_example)
    assert abs(result["loss"] - loss) <= bound
    assert result["loss_denominator"] == 21
    with pytest.raises(ValueError, match="truth class"):
        evaluate_images(Fixed(), [batches[0]])


def epoch_row(epoch, matrix, steps=1):
    matrix = np.asarray(matrix, dtype=int)
    score = float(np.mean(2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1))))
    return {"epoch": epoch, "val_confusion_matrix": matrix.tolist(), "val_macro_f1": score,
            "optimizer_steps": steps, "train_loss": 1., "val_loss": 1.}


def test_selection_is_f1_earliest_unrounded_and_requires_updates():
    first = np.eye(6, dtype=int); first[0, 0] = 100; first[5, 5] = 0; first[5, 0] = 1
    second = np.eye(6, dtype=int); second[0, 0] = 50; second[0, 1] = 50
    assert np.mean(first.diagonal() / first.sum(1)) < np.mean(second.diagonal() / second.sum(1))
    history = [epoch_row(1, first), epoch_row(2, second), epoch_row(3, first), epoch_row(4, np.eye(6), steps=0)]
    assert support.select_epoch(history) == 1
    history[0]["val_macro_f1"] = float("nan")
    with pytest.raises(ValueError): support.select_epoch(history)
    with pytest.raises(ValueError, match="updates"):
        support.select_epoch([epoch_row(1, np.eye(6), steps=0)])


def test_atomic_publication_and_failed_attempt(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    support.publish_json(path, {"first": True})
    with pytest.raises(FileExistsError): support.publish_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}
    with pytest.raises(ValueError): support.publish_json(tmp_path / "nan.json", {"x": float("nan")})
    assert not (tmp_path / "nan.json").exists()
    with monkeypatch.context() as patch:
        patch.setattr(support.os, "link", lambda *a: (_ for _ in ()).throw(OSError("injected link failure")))
        with pytest.raises(OSError): support.publish_json(tmp_path / "lost.json", {"x": 1})
    assert list(tmp_path.glob(".lost.json.*.partial"))
    spec = fixture_spec(tmp_path)
    # Bad counts fail in actual preparation, before any constructor.
    spec["inputs"]["counts"]["train"]["total"] += 1
    with pytest.raises(ValueError): runner.train_resnet(spec)
    output = Path(spec["output_dir"])
    assert json.loads((output / "failed.json").read_text())["status"] == "failed"
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError): runner.train_resnet(spec)


@pytest.mark.parametrize("location", ["descendant", "same", "ancestor", "symlink", "sibling"])
def test_final_output_preserves_tuning_bundles(tmp_path, monkeypatch, location):
    candidates = []
    for index in range(3):
        root = tmp_path / "tuning" / str(index)
        root.mkdir(parents=True)
        (root / "selected.pt").write_bytes(b"synthetic state; never loaded")
        (root / "result.json").write_text(json.dumps({
            "status": "complete", "kind": "resnet18", "outputs": artifact_hashes(root)}))
        candidates.append(support.binding(root / "result.json"))
    prior = Path(candidates[1]["path"]).parent
    output = {"descendant": prior / "final", "same": prior,
              "ancestor": prior.parent, "sibling": tmp_path / "final"}.get(location)
    if location == "symlink":
        alias = tmp_path / "tuning-alias"
        alias.symlink_to(prior, target_is_directory=True)
        output = alias / "final"
    spec = {"output_dir": str(output), "tuning_results": candidates, "device": "cpu",
            "kind": "resnet18", "purpose": "synthetic_cpu_test", "fold": 0}
    prepared = {"ready": True, "missing_prerequisites": [], "source_files": {},
                "runtime": {}, "inputs": {}, "selection": None, "cuda_dependencies": None}
    preparations, work = [], []

    def prepare(job):
        preparations.append(job)
        # Authenticate actual files, so a nested attempt would invalidate a bundle.
        for candidate in candidates:
            runner._completed(candidate, "resnet18")
        return prepared

    monkeypatch.setattr(runner, "prepare", prepare)
    original = artifact_hashes(tmp_path / "tuning")
    if location == "sibling":
        result = runner._attempt(spec, lambda *args: work.append(True) or {})
        assert result["status"] == "complete" and work == [True]
        assert len(preparations) == 2
    else:
        with pytest.raises(ValueError, match="Output overlaps source or inputs"):
            runner._attempt(spec, lambda *args: pytest.fail("work before output protection"))
        assert preparations == []
        if location in ("descendant", "symlink"):
            assert not output.exists()
    assert artifact_hashes(tmp_path / "tuning") == original
    for candidate in candidates:
        runner._completed(candidate, "resnet18")


def test_prepare_no_model_and_complete_tiny_pipeline_replay(tmp_path, monkeypatch):
    spec = seal_runtime(fixture_spec(tmp_path))
    with monkeypatch.context() as patch:
        patch.setattr(runner, "build_resnet18", lambda *a: pytest.fail("prepare constructed a model"))
        prepared = runner.prepare(spec)
    assert prepared["ready"] and not prepared["model_checks_executed"]
    assert not Path(spec["output_dir"]).exists()
    def tiny_builder(model_spec, device):
        torch.manual_seed(model_spec["head_seed"])
        return TinyImages(), {"model": "synthetic_tiny_fixture"}
    monkeypatch.setattr(runner, "build_resnet18", tiny_builder)
    result = runner.train_resnet(spec)
    assert result["budget_disposition"] == "complete finite training recipe; no convergence claim"
    assert result["status"] == "complete" and len(result["history"]) == 2
    assert result["inputs"]["counts"]["val"]["class_counts"] == [1] * 6
    assert all(row["optimizer_steps"] == 2 and row["examples"] == 6 for row in result["history"])
    state = torch.load(Path(spec["output_dir"]) / "selected.pt", weights_only=True)
    fresh = TinyImages(); fresh.load_state_dict(state["model_state_dict"])
    loader = support.make_loaders(spec, roles=("val",), seed=10000)["val"]
    assert evaluate_images(fresh, loader) == result["selected_validation"]
    assert "test" in result and result["test"]["tile_ids"] == [f"test_{i}" for i in range(6)]
    snapshot = runner._cpu_copy(fresh.state_dict())
    with torch.no_grad(): fresh.fc.weight.add_(1)
    assert not torch.equal(fresh.fc.weight, snapshot["fc.weight"])


def feature_bundle(tmp_path, spec):
    root = tmp_path / "feature-producer"; directory = root / "features"; directory.mkdir(parents=True)
    inputs = support.validate_inputs(spec)
    identity = {"model": "dinov2_random_matched", "initialization": "random", "synthetic_fixture": True}
    caches = {}
    for role in support.ROLES:
        labels = np.arange(6, dtype=np.int64); ids = np.array([f"{role}_{i}" for i in range(6)])
        features = np.zeros((6, 768), dtype=np.float32); features[:, :6] = np.eye(6)
        if role != "train": features += 10  # held-out shift must not enter scaler fit
        expected = {"identity": identity, "inputs": inputs, "fold": 0, "role": role,
                    "transform": followup.TRANSFORM, "purpose": "synthetic_cpu_test",
                    "tile_ids": ids.tolist(), "labels": labels.tolist()}
        hashes = {}
        for suffix, array, key in (("feats", features, "feature_sha256"), ("ids", ids, "tile_ids_sha256"),
                                   ("labels", labels, "labels_sha256")):
            path = directory / f"titan_{role}_{suffix}.npy"; np.save(path, array); hashes[key] = sha256_file(path)
        (directory / f"titan_{role}_feats.meta.json").write_text(json.dumps({
            "schema_version": "model-followup-v1", "model": identity["model"], "array": f"titan_{role}_feats",
            "weights_source": "random", "n_features": 6, "feature_dim": 768, "provenance": {**expected, **hashes}}))
        caches[role] = expected
    result = {"status": "complete", "kind": "dino_features", "purpose": "synthetic_cpu_test", "fold": 0,
              "inputs": inputs, "identity": identity, "caches": caches, "outputs": artifact_hashes(root)}
    path = root / "result.json"; path.write_text(json.dumps(result))
    return root, result, path


@pytest.mark.parametrize("fault", ["ids", "dtype", "nan", "seed", "weights_source"])
def test_cache_semantic_mutations_with_refreshed_hashes(tmp_path, fault):
    spec = fixture_spec(tmp_path); root, result, _ = feature_bundle(tmp_path, spec)
    path = root / "features/titan_train_feats.npy"
    meta_path = path.with_suffix(".meta.json"); meta = json.loads(meta_path.read_text())
    if fault == "ids":
        ids = path.with_name("titan_train_ids.npy"); np.save(ids, np.array(["train_0"] * 6))
        meta["provenance"]["tile_ids_sha256"] = sha256_file(ids)
    elif fault in ("dtype", "nan"):
        a = np.load(path)
        if fault == "dtype": a = a.astype(np.float64)
        else: a[0, 0] = np.nan
        np.save(path, a); meta["provenance"]["feature_sha256"] = sha256_file(path)
    elif fault == "seed": meta["provenance"]["identity"]["encoder_seed"] = 999
    else: meta["weights_source"] = "placeholder"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises((ValueError, RuntimeError)):
        support.load_followup_cache(path, result["caches"]["train"])


def test_c1_consumer_fits_only_training_and_replays_saved_state(tmp_path, monkeypatch):
    base = fixture_spec(tmp_path); root, producer, result_path = feature_bundle(tmp_path, base)
    spec = seal_runtime({"schema_version": "model-followup-v1", "kind": "dino_c1", "purpose": "synthetic_cpu_test",
                         "fold": 0, "device": "cpu", "output_dir": str(tmp_path / "readout"),
                         "features": support.binding(result_path), "feature_identity": producer["identity"]})
    result = runner.readout_c1(spec)
    assert result["classifier_parameters"]["C"] == 1 and result["classifier_parameters"]["max_iter"] == 2000
    state = np.load(tmp_path / "readout/estimator.npz")
    train = np.load(root / "features/titan_train_feats.npy")
    train64 = train.astype(np.float64)
    variance = train64.var(0)
    np.testing.assert_array_equal(state["scaler_mean"], train64.mean(0))
    np.testing.assert_array_equal(state["scaler_var"], variance)
    np.testing.assert_array_equal(state["scaler_scale"], np.where(variance == 0, 1, np.sqrt(variance)))
    assert state["scaler_samples"].item() == len(train)

    def no_refit(*args, **kwargs):
        pytest.fail("Saved scaler replay must not refit")

    monkeypatch.setattr(StandardScaler, "fit", no_refit)
    monkeypatch.setattr(StandardScaler, "partial_fit", no_refit)
    scaler = StandardScaler(**result["scaler_parameters"])
    scaler.mean_ = state["scaler_mean"]
    scaler.var_ = state["scaler_var"]
    scaler.scale_ = state["scaler_scale"]
    scaler.n_samples_seen_ = state["scaler_samples"]
    scaler.n_features_in_ = train.shape[1]
    # Replay the serialized scaler through its native transform. Manual float64
    # statistic arithmetic omits its float32 casts before subtraction/division.
    held_out = np.load(root / "features/titan_test_feats.npy")
    x = scaler.transform(held_out)
    assert x.dtype == held_out.dtype == np.float32
    logits = x @ state["coef"].T + state["intercept"]
    assert str(logits.dtype) == result["predictions"]["test"]["logits_dtype"]
    np.testing.assert_allclose(logits, result["predictions"]["test"]["logits"], rtol=0, atol=1e-12)
    assert logits.argmax(1).tolist() == result["predictions"]["test"]["predictions"]
    assert "warnings" in result and result["convergence_claim"] is False


def test_resnet_rejects_arbitrary_pretrained_state_before_construction(tmp_path, monkeypatch):
    spec = resnet_spec("pretrained"); spec["weights"] = followup.RESNET_WEIGHTS
    path = tmp_path / "arbitrary.pt"; torch.save({"weight": torch.ones(1)}, path)
    spec["checkpoint"] = support.binding(path)
    module = importlib.import_module("torchvision.models.resnet")
    monkeypatch.setattr(module, "resnet18", lambda *a, **k: pytest.fail("constructed before rejecting weights"))
    with pytest.raises(ValueError, match="pretrained release"):
        followup.build_resnet18(spec)


def test_failure_receipt_error_does_not_mask_original(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    spec["inputs"]["counts"]["train"]["total"] += 1
    original = support.publish_json
    def broken_failure_receipt(path, value):
        if Path(path).name == "failed.json":
            raise OSError("receipt volume unavailable")
        return original(path, value)
    monkeypatch.setattr(support, "publish_json", broken_failure_receipt)
    with pytest.raises(ValueError, match="Partition counts differ") as failure:
        runner.train_resnet(spec)
    assert any("receipt volume unavailable" in note for note in failure.value.__notes__)


def test_current_catalog_and_fold_cannot_be_redeclared(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    monkeypatch.setattr(runner, "build_resnet18", lambda *a: pytest.fail("constructed model"))
    spec["purpose"] = "titan_followup"
    with pytest.raises(ValueError, match="current Titan catalog"):
        support.validate_inputs(spec)
    spec["inputs"]["catalog"]["sha256"] = support.CURRENT_CATALOG_SHA256
    spec["inputs"]["split"]["sha256"] = support.CURRENT_SPLITS[0]
    spec["fold"] = 1
    with pytest.raises(ValueError, match="split for this fold"):
        support.validate_inputs(spec)
    # Also reject policy relabeling for a synthetic fixture after its hashes change.
    spec["purpose"] = "synthetic_cpu_test"
    spec["inputs"]["catalog"] = support.binding(tmp_path / "inputs/catalog.json")
    spec["inputs"]["split"] = support.binding(tmp_path / "inputs/split.json")
    with pytest.raises(ValueError, match="policy fold"):
        support.validate_inputs(spec)


def test_recorded_split_pins_match_checked_in_authority():
    for fold, expected in enumerate(support.CURRENT_SPLITS):
        path = support.ROOT / f"audit/gpu_execution/repaired_splits/titan_contiguous_fold_{fold}.json"
        assert sha256_file(path) == expected
        content = json.loads(path.read_text())
        assert content["catalog_sha256"] == support.CURRENT_CATALOG_SHA256
        assert content["policy"]["test_fold"] == fold and len(content["assignments"]) == 23380


def test_all_three_complete_candidates_and_lower_lr_tie(tmp_path, monkeypatch):
    base = fixture_spec(tmp_path)
    base["recipe"].update(epochs=1, batch_size=6)
    def tiny_builder(model_spec, device):
        model = TinyImages()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
        return model, {"model": "synthetic_zero_initialized_tiny_fixture"}
    monkeypatch.setattr(runner, "build_resnet18", tiny_builder)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: pytest.fail("CPU job queried GPU"))
    candidates, scores = [], []
    for index, lr in enumerate(support.CANDIDATE_LRS):
        spec = copy.deepcopy(base)
        spec.update(phase="tuning", seed=100, loader_seed=10100, output_dir=str(tmp_path / f"tuning-{index}"))
        spec["model"].update(backbone_seed=100, head_seed=20100)
        spec["recipe"]["lr"] = lr
        seal_runtime(spec)
        result = runner.train_resnet(spec)
        assert "test" not in result and "gpu_execution" not in result
        candidates.append(support.binding(Path(spec["output_dir"]) / "result.json"))
        scores.append(result["history"][0]["val_macro_f1"])
    assert scores[0] == scores[1] == scores[2]
    base["recipe"]["lr"] = support.CANDIDATE_LRS[0]
    base["tuning_results"] = candidates
    seal_runtime(base)
    inputs = support.validate_inputs(base)
    selected = runner._selected_lr(base, inputs)
    assert selected["lr"] == 1e-4
    base["tuning_results"] = list(reversed(candidates))
    assert runner._selected_lr(base, inputs)["lr"] == selected["lr"]
    base["tuning_results"] = candidates[:2]
    with pytest.raises(ValueError, match="three complete"):
        runner._selected_lr(base, inputs)
    base["tuning_results"] = [candidates[0], candidates[0], candidates[2]]
    with pytest.raises(ValueError, match="duplicate LR"):
        runner._selected_lr(base, inputs)
    base["tuning_results"] = candidates; base["fold"] = 1
    with pytest.raises(ValueError, match="source/task"):
        runner._selected_lr(base, inputs)


def test_one_job_extraction_with_actual_tiny_torch_encoder_then_c1(tmp_path, monkeypatch):
    spec = fixture_spec(tmp_path)
    torch.manual_seed(42)
    tiny = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 768))
    checkpoint = tmp_path / "inputs/tiny-initial.pt"; torch.save(tiny.state_dict(), checkpoint)
    identity = {"model": "dinov2_random_matched", "initialization": "random", "synthetic_fixture": True,
                "initial_state_sha256": followup.state_digest(tiny.state_dict())}
    wrapper = followup.DINOv2Wrapper("cpu"); wrapper.model = tiny
    # Routing uses a real tiny Torch encoder. Official source/architecture is
    # exercised separately by the opt-in integration test, not by this fixture.
    monkeypatch.setattr(runner, "load_matched_dino", lambda *a: (wrapper, identity))
    monkeypatch.setattr(runner, "verify_dino_source", lambda model: tmp_path / "inputs")
    spec.update(kind="dino_features", recipe={"batch_size": 4},
                model={"initialization": "random", "encoder_seed": 42, "checkpoint": support.binding(checkpoint),
                       "initial_state_sha256": identity["initial_state_sha256"]})
    seal_runtime(spec)
    result = runner.extract_dino(spec)
    assert set(result["caches"]) == set(support.ROLES)
    assert all(result["caches"][role]["tile_ids"] == [f"{role}_{i}" for i in range(6)] for role in support.ROLES)
    assert followup.state_digest(tiny.state_dict()) == identity["initial_state_sha256"]
    consumer = seal_runtime({"schema_version": "model-followup-v1", "kind": "dino_c1", "purpose": "synthetic_cpu_test",
                             "fold": 0, "device": "cpu", "output_dir": str(tmp_path / "one-job-readout"),
                             "features": support.binding(Path(spec["output_dir"]) / "result.json"),
                             "feature_identity": identity})
    fitted = runner.readout_c1(consumer)
    assert fitted["status"] == "complete" and set(fitted["predictions"]) == {"val", "test"}


@pytest.fixture
def cuda_precision_spec(tmp_path, monkeypatch, request):
    # Only host settings are exercised: every CUDA initialization/query is forbidden.
    def no_device(*args, **kwargs):
        pytest.fail("Precision validation touched a CUDA device")
    for name in ("_lazy_init", "set_device", "get_device_properties", "device_count"):
        monkeypatch.setattr(torch.cuda, name, no_device)
    monkeypatch.setattr(support, "validate_inputs", no_device)
    monkeypatch.setattr(runner, "build_resnet18", no_device)
    monkeypatch.setattr(runner, "load_matched_dino", no_device)
    matmul_precision = torch.backends.cuda.matmul.fp32_precision
    convolution_precision = torch.backends.cudnn.conv.fp32_precision

    def restore_precision():
        torch.backends.cuda.matmul.fp32_precision = matmul_precision
        torch.backends.cudnn.conv.fp32_precision = convolution_precision

    # These proxy setters update native state, not an instance-dictionary entry.
    # pytest 9.2's attribute undo would try to delete that nonexistent entry.
    request.addfinalizer(restore_precision)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    # Inter-op counts are metadata here; the fresh-process CLI test checks real setup.
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    expected_uuid = "GPU-00000000-0000-0000-0000-000000000001"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected_uuid)
    receipt = tmp_path / "environment/dependencies.json"
    receipt.parent.mkdir()
    receipt.write_text('{"scope": "synthetic receipt; no CUDA dependencies verified"}')
    dtype = torch.get_default_dtype()
    autocast = torch.is_autocast_enabled("cuda")
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield {"schema_version": "model-followup-v1", "kind": "dino_features",
           "purpose": "titan_followup", "fold": 0, "device": "cuda:0",
           "model": {"encoder_seed": 42}, "recipe": {"batch_size": 32},
           "execution": {"gpu_uuid": expected_uuid, "cuda_dependencies": support.binding(receipt)}}
    torch.set_default_dtype(dtype)
    torch.set_autocast_enabled("cuda", autocast)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("fault, message", [
    ("matmul", "CUDA matmul"), ("convolution", "cuDNN convolution"),
    ("autocast", "autocast"), ("benchmark", "deterministic cuDNN"),
    ("cudnn_deterministic", "deterministic cuDNN"), ("warn_only", "must raise"),
    ("workspace_missing", "CUBLAS_WORKSPACE_CONFIG"), ("workspace_other", "CUBLAS_WORKSPACE_CONFIG"),
    ("torch_override", "TF32 environment"), ("nvidia_override", "TF32 environment"),
    ("interop", "inter-op"), ("dtype", "float32 default"), ("device", "indexed CUDA"),
    ("uuid_missing", "canonical full GPU UUID"), ("visibility", "declared GPU UUID"),
    ("receipt_missing", "CUDA dependency receipt"),
])
def test_cuda_precision_rejected_before_inputs_or_model(cuda_precision_spec, monkeypatch, fault, message):
    if fault == "matmul": torch.backends.cuda.matmul.fp32_precision = "tf32"
    elif fault == "convolution": torch.backends.cudnn.conv.fp32_precision = "tf32"
    elif fault == "autocast": torch.set_autocast_enabled("cuda", True)
    elif fault == "benchmark": torch.backends.cudnn.benchmark = True
    elif fault == "cudnn_deterministic": torch.backends.cudnn.deterministic = False
    elif fault == "warn_only": torch.use_deterministic_algorithms(True, warn_only=True)
    elif fault == "workspace_missing": monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    elif fault == "workspace_other": monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    elif fault == "torch_override": monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    elif fault == "nvidia_override": monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    elif fault == "interop": monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 3)
    elif fault == "dtype": torch.set_default_dtype(torch.float64)
    elif fault == "device": cuda_precision_spec["device"] = "cuda:not-an-index"
    elif fault == "uuid_missing": cuda_precision_spec["execution"].pop("gpu_uuid")
    elif fault == "visibility": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif fault == "receipt_missing": cuda_precision_spec["execution"].pop("cuda_dependencies")
    with pytest.raises(ValueError, match=message):
        runner.prepare(cuda_precision_spec)


def test_cuda_precision_contract_needs_no_gpu_and_cpu_needs_no_workspace(cuda_precision_spec, monkeypatch):
    runner._validate_spec(cuda_precision_spec)
    cuda_precision_spec.update(device="cpu", purpose="synthetic_cpu_test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    runner._validate_spec(cuda_precision_spec)


@pytest.mark.parametrize("matmul, convolution", [("ieee", "tf32"), ("tf32", "ieee")])
def test_runtime_precision_records_actual_host_flags(cuda_precision_spec, matmul, convolution):
    torch.backends.cuda.matmul.fp32_precision = matmul
    torch.backends.cudnn.conv.fp32_precision = convolution
    identity = support.runtime_identity()
    assert identity["cuda_matmul_fp32_precision"] == matmul
    assert identity["cudnn_conv_fp32_precision"] == convolution
    assert identity["cuda_autocast_enabled"] is False
    assert identity["deterministic_warn_only"] is False
    assert identity["interop_threads"] == 1
    assert identity["environment"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert identity["environment"]["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] is None
    assert identity["environment"]["NVIDIA_TF32_OVERRIDE"] is None


def test_cli_precision_configuration_in_fresh_cpu_process(tmp_path):
    code = """
import sys
import torch
from scripts import run_model_followup as runner
try:
    runner.main(["--spec", sys.argv[1]])
except FileNotFoundError:
    pass
else:
    raise AssertionError("Missing specification unexpectedly loaded")
assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert torch.backends.cuda.matmul.fp32_precision == "ieee"
assert torch.backends.cudnn.conv.fp32_precision == "ieee"
assert not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path / "missing.json")],
                   cwd=support.ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("fault, message", [("uuid", "Actual GPU UUID"),
                                          ("count", "one visible GPU"),
                                          ("initialized", "fresh process")])
def test_cuda_lane_rejected_before_work_with_fake_properties(cuda_precision_spec, tmp_path, monkeypatch, fault, message):
    spec = cuda_precision_spec
    spec["output_dir"] = str(tmp_path / "attempt")
    prepared = {"ready": True, "missing_prerequisites": [],
                "cuda_dependencies": spec["execution"]["cuda_dependencies"]}
    monkeypatch.setattr(runner, "prepare", lambda spec: prepared)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: fault == "initialized")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if fault == "count" else 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        uuid="00000000-0000-0000-0000-000000000002", name="CPU fixture", total_memory=0))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: pytest.fail("reset before lane check"))
    with pytest.raises(ValueError, match=message):
        runner._attempt(spec, lambda *a: pytest.fail("model work before lane check"))
    failure = json.loads((tmp_path / "attempt/failed.json").read_text())
    assert failure["status"] == "failed" and message in failure["error"]


def test_cuda_dependency_receipt_change_rejected_without_library_scan(cuda_precision_spec, monkeypatch):
    spec = cuda_precision_spec
    # Reach only the small receipt check. Input/model/device calls remain forbidden.
    monkeypatch.setattr(support, "checked_output", lambda spec: None)
    monkeypatch.setattr(runner, "git_state", lambda root: {"clean": True, "commit": "fixture"})
    monkeypatch.setattr(support, "verify_source_imports", lambda files: None)
    spec["source"] = {"commit": "fixture", "files": {name: sha256_file(support.ROOT / name)
                                                   for name in support.SOURCE_FILES}}
    spec["runtime"] = support.runtime_identity()
    receipt = spec["execution"]["cuda_dependencies"]
    Path(receipt["path"]).write_text('{"scope": "changed receipt"}')
    with pytest.raises(ValueError, match="File hash differs"):
        runner.prepare(spec)


def test_cuda_lane_receipt_emitted_with_fake_properties(cuda_precision_spec, tmp_path, monkeypatch):
    spec = cuda_precision_spec
    spec["output_dir"] = str(tmp_path / "attempt")
    prepared = {"ready": True, "missing_prerequisites": [], "source_files": {},
                "runtime": {}, "inputs": {}, "selection": None,
                "cuda_dependencies": spec["execution"]["cuda_dependencies"]}
    monkeypatch.setattr(runner, "prepare", lambda spec: prepared)
    events = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("select", device)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        uuid=spec["execution"]["gpu_uuid"].removeprefix("GPU-"), name="CPU fixture", total_memory=0))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append(("reset", device)))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 0)
    monkeypatch.setattr(runner, "_gpu_finished", lambda record: record)
    result = runner._attempt(spec, lambda *a: events.append(("work", spec["device"])) or {})
    assert events == [("select", "cuda:0"), ("reset", "cuda:0"), ("work", "cuda:0")]
    assert result["gpu_execution"]["uuid"] == spec["execution"]["gpu_uuid"]
    assert result["gpu_execution"]["cuda_dependencies"] == spec["execution"]["cuda_dependencies"]


def test_cuda_dependency_receipt_must_match_tuning_candidates(monkeypatch):
    spec = {"purpose": "titan_followup", "fold": 0, "source": {}, "runtime": {},
            "model": {"initialization": "scratch"}, "tuning_results": [{}, {}, {}],
            "execution": {"cuda_dependencies": {"path": "receipt", "sha256": "a" * 64}}}
    prior = {**spec, "phase": "tuning", "seed": 100,
             "execution": {"cuda_dependencies": {"path": "receipt", "sha256": "b" * 64}}}
    monkeypatch.setattr(runner, "_completed", lambda *a: (None, {"specification": prior, "inputs": {}}))
    with pytest.raises(ValueError, match="Candidate CUDA dependency receipt differs"):
        runner._selected_lr(spec, {})
