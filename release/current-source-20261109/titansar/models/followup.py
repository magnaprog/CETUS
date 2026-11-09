"""Offline model contracts for the separately identified CVPR follow-up.

No constructor in this module downloads code or weights. Callers supply local
source/checkpoint bindings; a missing or changed binding is an error.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

from titansar.models.foundation_models import DINOV2_REVISION, DINOV2_SHA256, DINOv2Wrapper
from titansar.reproducibility import sha256_file

TRANSFORM = {
    "name": "cetus_full_tile_rgb224_v1", "input_shape": [1, 128, 128],
    "dtype": "float32", "resize": [224, 224], "mode": "bilinear",
    "align_corners": False, "antialias": False,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}
DINO_CONSTRUCTOR = {
    "img_size": 518, "patch_size": 14, "init_values": 1.0,
    "ffn_layer": "mlp", "block_chunks": 0, "num_register_tokens": 0,
    "interpolate_antialias": False, "interpolate_offset": 0.1,
}
RESNET_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
# All 157 Python files restored from the authenticated retained upstream cache.
# This accepts both a source checkout and Torch Hub's archive without a .git dir.
DINO_SOURCE_SHA256 = "929c0b25f4287511f065c8696667b39d2b9499c68f284bcab30b3e27099a488f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_file(binding):
    path = Path(binding["path"]).resolve(strict=True)
    expected = binding["sha256"]
    require(isinstance(expected, str) and len(expected) == 64, "Full SHA256 required")
    require(sha256_file(path) == expected, f"File hash differs: {path}")
    return path


def state_digest(state_dict):
    """Hash sorted names, dtype, shape and contiguous CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_rgb224(images):
    require(images.dtype == torch.float32 and images.ndim == 4
            and tuple(images.shape[1:]) == (1, 128, 128), "Expected float32 Bx1x128x128")
    require(bool(torch.isfinite(images).all()) and bool(((images >= 0) & (images <= 1)).all()),
            "Expected finite normalized SAR input")
    # An unloaded wrapper provides the existing, source-bound transform only.
    return DINOv2Wrapper(device=str(images.device)).prepare_input(images)


def verify_dino_source(spec):
    root = Path(spec["source_dir"]).resolve(strict=True)
    require(spec["source_revision"] == DINOV2_REVISION, "Unsupported DINO revision")
    files = spec["source_files"]
    actual = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}
    sealed = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(actual == files and sealed == DINO_SOURCE_SHA256, "DINO source inventory differs")
    for name, module in list(sys.modules.items()):
        if name == "dinov2" or name.startswith("dinov2."):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), f"DINO import shadowed: {name}")
    return root


def verify_resnet_source(spec):
    import torchvision
    module = importlib.import_module("torchvision.models.resnet")
    require(torchvision.__version__ == spec["torchvision_version"], "torchvision version differs")
    require(sha256_file(Path(module.__file__)) == spec["resnet_source_sha256"], "ResNet source differs")
    require(module.ResNet18_Weights.IMAGENET1K_V1.url == RESNET18_URL, "Official weight URL differs")
    return module


def _checkpoint(spec, expected=None):
    binding = spec["checkpoint"]
    if expected is not None:
        require(binding["sha256"] == expected, "Checkpoint is not the declared pretrained release")
    path = check_file(binding)
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values()),
            "Expected a plain complete model state dictionary")
    return state


def load_matched_dino(spec, device="cpu"):
    root = verify_dino_source(spec)
    require(spec["initialization"] in ("random", "pretrained"), "Unknown DINO initialization")
    seed = spec["encoder_seed"]
    require(type(seed) is int and seed in range(42, 47), "DINO seed must be 42..46")
    require(os.environ.get("XFORMERS_DISABLED") == "1", "Set XFORMERS_DISABLED=1 before DINO import")
    initial = _checkpoint(spec, DINOV2_SHA256 if spec["initialization"] == "pretrained" else None) if spec.get("checkpoint") else None
    require(spec["initialization"] != "pretrained" or initial is not None, "Missing DINO pretrained checkpoint")
    # Import only the official backbone entrypoint; Hub's unrelated task imports
    # are unnecessary. Never call a remote torch.hub loader.
    sys.path.insert(0, str(root))
    try:
        entrypoint = importlib.import_module("dinov2.hub.backbones").dinov2_vitb14
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = entrypoint(pretrained=False, **DINO_CONSTRUCTOR)
        if spec["initialization"] == "pretrained":
            model.load_state_dict(initial, strict=True)
        elif initial is not None:
            require(state_digest(initial) == state_digest(model.state_dict()),
                    "Random checkpoint differs from the seeded official initialization")
            model.load_state_dict(initial, strict=True)
        require(tuple(model.pos_embed.shape) == (1, 1370, 768) and model.embed_dim == 768,
                "DINO architecture differs")
        imported = {}
        for name, module in list(sys.modules.items()):
            if name == "dinov2" or name.startswith("dinov2."):
                path = Path(module.__file__).resolve()
                require(path.is_relative_to(root), f"DINO import shadowed: {name}")
                relative = str(path.relative_to(root))
                require(spec["source_files"].get(relative) == sha256_file(path), "Imported DINO source differs")
                imported[name] = relative
        attention = importlib.import_module("dinov2.layers.attention")
        require(attention.XFORMERS_AVAILABLE is False, "Unexpected DINO attention backend")
    finally:
        sys.path.remove(str(root))
    digest = state_digest(model.state_dict())
    if spec.get("initial_state_sha256"):
        require(digest == spec["initial_state_sha256"], "Initial DINO state differs")
    wrapper = DINOv2Wrapper(device=device)
    wrapper.model = model.to(device).eval().requires_grad_(False)
    wrapper.name = f"dinov2_{spec['initialization']}_matched"
    wrapper.weights_source = spec["initialization"]
    wrapper.model_revision = DINOV2_REVISION
    wrapper.weights_sha256 = digest
    identity = {"model": wrapper.name, "initialization": spec["initialization"],
                "encoder_seed": seed, "revision": DINOV2_REVISION,
                "source_python_inventory_sha256": DINO_SOURCE_SHA256,
                "constructor": DINO_CONSTRUCTOR, "transform": TRANSFORM,
                "initial_state_sha256": digest, "checkpoint": spec.get("checkpoint"),
                "imported_source": imported, "dtype": "float32", "backend": "torch_sdpa_no_xformers"}
    return wrapper, identity


def build_resnet18(spec, device="cpu"):
    module = verify_resnet_source(spec)
    require(spec["initialization"] in ("scratch", "pretrained"), "Unknown ResNet initialization")
    for field in ("backbone_seed", "head_seed"):
        require(type(spec[field]) is int and spec[field] >= 0, f"Invalid {field}")
    pretrained = None
    if spec["initialization"] == "pretrained":
        require(spec["weights"] == RESNET_WEIGHTS, "Explicit ImageNet weight enum required")
        pretrained = _checkpoint(spec, RESNET18_SHA256)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec["backbone_seed"])
        model = module.resnet18(weights=None)
    compatibility = []
    if spec["initialization"] == "pretrained":
        compatibility = sorted(set(model.state_dict()) - set(pretrained))
        require(all(name.endswith(".num_batches_tracked") for name in compatibility),
                "Pretrained checkpoint is missing learned parameters or running moments")
        model.load_state_dict(pretrained, strict=True)
        # The authenticated legacy torchvision release predates these counters.
        # PyTorch's BatchNorm loader supplies zero even with strict=True.
        require(all(int(model.state_dict()[name]) == 0 for name in compatibility),
                "Legacy BatchNorm counter initialization differs")
    else:
        require(not spec.get("checkpoint") and not spec.get("weights"), "Scratch ResNet cannot consume pretrained weights")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec["head_seed"])
        model.fc = torch.nn.Linear(512, 6)
    identity = {"model": "resnet18", "initialization": spec["initialization"],
                "backbone_seed": spec["backbone_seed"], "head_seed": spec["head_seed"],
                "head_state_sha256": state_digest(model.fc.state_dict()),
                "initial_state_sha256": state_digest(model.state_dict()),
                "torchvision_version": spec["torchvision_version"],
                "resnet_source_sha256": spec["resnet_source_sha256"],
                "checkpoint": spec.get("checkpoint"), "weights": spec.get("weights"),
                "zero_initialized_legacy_batch_counters": compatibility,
                "transform": TRANSFORM, "num_classes": 6, "dtype": "float32"}
    return model.to(device).requires_grad_(True), identity
