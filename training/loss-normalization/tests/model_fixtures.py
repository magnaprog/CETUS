"""Small CPU model, data and runtime fixtures for the loss-normalization tests."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from scripts import model_followup_support as support
from cetus_sar.reproducibility import git_state, sha256_file


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


def epoch_row(epoch, matrix, steps=1):
    matrix = np.asarray(matrix, dtype=int)
    score = float(np.mean(2 * matrix.diagonal() / (matrix.sum(0) + matrix.sum(1))))
    return {"epoch": epoch, "val_confusion_matrix": matrix.tolist(), "val_macro_f1": score,
            "optimizer_steps": steps, "train_loss": 1., "val_loss": 1.}
