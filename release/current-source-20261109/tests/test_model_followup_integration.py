"""Opt-in actual models on CPU synthetic tensors, with authenticated local assets.

Set TITANSAR_RUN_MODEL_INTEGRATION=1, TITANSAR_RESNET18_CHECKPOINT and
TITANSAR_DINO_SOURCE_SPEC (a local source specification JSON, no downloads).
"""
import gc
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from scripts.model_followup_support import binding
from titansar.models import followup
from titansar.models.image_training import evaluate_images, train_epoch
from titansar.reproducibility import sha256_file

pytestmark = pytest.mark.skipif(os.environ.get("TITANSAR_RUN_MODEL_INTEGRATION") != "1",
                                reason="Actual model integration requires explicit local asset opt-in")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setenv("XFORMERS_DISABLED", "1")
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    yield
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(deterministic)


def model_spec():
    module = importlib.import_module("torchvision.models.resnet")
    return {"initialization": "scratch", "backbone_seed": 0, "head_seed": 20000,
            "torchvision_version": torchvision.__version__, "resnet_source_sha256": sha256_file(Path(module.__file__))}


def test_actual_resnet_one_step_all_stages_bn_and_checkpoint_replay(tmp_path):
    spec = model_spec()
    before_rng = torch.get_rng_state().clone()
    model, identity = followup.build_resnet18(spec)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert model.fc.in_features == 512 and model.fc.out_features == 6
    names = ["conv1.weight", "layer1.0.conv1.weight", "layer2.0.conv1.weight", "layer3.0.conv1.weight",
             "layer4.0.conv1.weight", "fc.weight"]
    before = {name: model.state_dict()[name].clone() for name in names}
    bn = model.bn1.running_mean.clone()
    images = torch.rand(6, 1, 128, 128, generator=torch.Generator().manual_seed(8))
    batch = {"image": images, "label": torch.arange(6), "tile_id": list("abcdef")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    result = train_epoch(model, [batch], optimizer, class_weights=torch.ones(6))
    assert result["optimizer_steps"] == 1
    for name in names:
        assert not torch.equal(model.state_dict()[name], before[name]), name
    assert not torch.equal(model.bn1.running_mean, bn)
    bn = model.bn1.running_mean.clone()
    predicted = evaluate_images(model, [batch])
    assert torch.equal(model.bn1.running_mean, bn)
    path = tmp_path / "trained.pt"; torch.save(model.state_dict(), path)
    fresh, _ = followup.build_resnet18(spec); fresh.load_state_dict(torch.load(path, weights_only=True))
    replay = evaluate_images(fresh, [batch])
    assert predicted == replay
    assert followup.state_digest(fresh.state_dict()) != identity["initial_state_sha256"]


def test_actual_resnet_imagenet_identity_and_independent_paired_head():
    path = os.environ.get("TITANSAR_RESNET18_CHECKPOINT")
    if not path: pytest.skip("Missing local resnet18-f37072fd.pth")
    assert sha256_file(Path(path)) == followup.RESNET18_SHA256
    spec = model_spec()
    scratch, random_identity = followup.build_resnet18(spec)
    head = {k: v.clone() for k, v in scratch.fc.state_dict().items()}
    del scratch; gc.collect()
    torch.rand(100)  # different constructor/caller consumption cannot alter paired head
    spec.update(initialization="pretrained", checkpoint=binding(path), weights=followup.RESNET_WEIGHTS)
    pretrained, identity = followup.build_resnet18(spec)
    original = torch.load(path, map_location="cpu", weights_only=True)
    for name, value in pretrained.state_dict().items():
        if not name.startswith("fc."):
            if name in original:
                assert torch.equal(value, original[name]), name
            else:
                assert name.endswith(".num_batches_tracked") and int(value) == 0
                assert name in identity["zero_initialized_legacy_batch_counters"]
    assert len(identity["zero_initialized_legacy_batch_counters"]) == 20
    for name, value in pretrained.fc.state_dict().items():
        assert torch.equal(value, head[name])
    assert random_identity["head_state_sha256"] == identity["head_state_sha256"]
    assert random_identity["initial_state_sha256"] != identity["initial_state_sha256"]


def test_official_dino_seed_states_and_one_pretrained_forward(tmp_path):
    path = os.environ.get("TITANSAR_DINO_SOURCE_SPEC")
    if not path: pytest.skip("Missing authenticated official DINO source and checkpoint specification")
    original_spec = json.loads(Path(path).read_text())
    followup.verify_dino_source(original_spec)
    assert sha256_file(Path(original_spec["checkpoint"]["path"])) == followup.DINOV2_SHA256
    spec = {**original_spec, "initialization": "random", "encoder_seed": 42}
    spec.pop("checkpoint"); spec.pop("initial_state_sha256", None)
    before_rng = torch.get_rng_state().clone()
    first, identity = followup.load_matched_dino(spec)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert tuple(first.model.pos_embed.shape) == (1, 1370, 768)
    initial = tmp_path / "random-initial.pt"; torch.save(first.model.state_dict(), initial)
    del first; gc.collect()
    repeat, same = followup.load_matched_dino({**spec, "checkpoint": binding(initial),
                                             "initial_state_sha256": identity["initial_state_sha256"]})
    assert same["initial_state_sha256"] == identity["initial_state_sha256"]
    del repeat; gc.collect()
    other, changed = followup.load_matched_dino({**spec, "encoder_seed": 43})
    assert changed["initial_state_sha256"] != identity["initial_state_sha256"]
    del other; gc.collect()
    pretrained, ready = followup.load_matched_dino({**original_spec, "initialization": "pretrained", "encoder_seed": 42})
    captured = []
    hook = pretrained.model.norm.register_forward_hook(lambda module, args, output: captured.append(output[:, 0].detach().clone()))
    try:
        features = pretrained.extract_features(torch.zeros(1, 1, 128, 128))
    finally:
        hook.remove()
    assert features.shape == (1, 768) and bool(torch.isfinite(features).all()) and not features.requires_grad
    assert torch.equal(features, captured[0])
    assert followup.state_digest(pretrained.model.state_dict()) == ready["initial_state_sha256"]
    assert np.isfinite(features.numpy()).all()
