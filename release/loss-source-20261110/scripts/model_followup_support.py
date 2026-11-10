"""Small binding/IO helpers for one follow-up job; no scheduling or grid runner."""
from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_finetuning import seed_worker
from titansar.data.dataset import TitanSARDataset
from titansar.models.followup import (
    DINO_CONSTRUCTOR, DINO_SOURCE_SHA256, DINOV2_REVISION, DINOV2_SHA256,
    TRANSFORM, check_file, require,
)
from titansar.models.probing import assert_cache_real_weights, load_feature_metadata
from titansar.reproducibility import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_model_followup.py", "scripts/model_followup_support.py",
    "titansar/models/followup.py", "titansar/models/image_training.py",
    "scripts/run_finetuning.py", "titansar/models/foundation_models.py",
    "titansar/models/probing.py", "titansar/models/baselines.py",
    "titansar/data/dataset.py", "titansar/configs/defaults.py", "titansar/reproducibility.py",
    "scripts/loss_comparison.py",
)
ROLES = ("train", "val", "test")
CLASS_ORDER = ["plains", "dunes", "hummocky", "labyrinths", "lakes", "craters"]
CURRENT_CATALOG_SHA256 = "2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec"
CURRENT_SPLITS = (
    "1079b1c63588ee9af674184b7dd6937ef01d3a354c8773e8ee9caad4ab8cc388",
    "f9bb93592f5bb9e111118f796c4cdc4e2478cb1ef18f1d61d7044525d0a820d2",
    "7862dd95df2eed01d0252a7a37068ee780509d98dd9e0105e1930ad0ffa75610",
    "8317384eb1945fc2af628e0051b8181637525d0412c2f2c1b38b7b2967fee9b6",
    "25e66ac42fd246d824fb957547e30d2e9f453dfd039130a4adf0b5cd0d796d09",
)
ANONYMOUS_INPUT_CONTRACT = "cvpr2027-anonymous-v1"
ANONYMOUS_CATALOG_SHA256 = '8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967'
ANONYMOUS_SPLITS = (
    'b69e2fa3551dde125edf31a5601c52ccae755cb6169ed57d145b16f6983f85ab',
    '58ad14b783c0d6ffd2851cb622f1d0d4fd1438978aeb146e32e5dcb8d55a8adc',
    '2eee98044e7a636a6726800328b0361991432e9ea272eaee7873b3faa2ee28d2',
    '2dbd6117de1c299655116af827ee95998839c67da443bf2a03729b7e4e2fab75',
    '1df555f0f95c16e3fb44243ce7ecca41225e4e588353148f25725c68f1cb4271',
)
RECIPE = {"optimizer": "AdamW", "epochs": 100, "batch_size": 32,
          "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
          "scheduler": "CosineAnnealingLR", "dtype": "float32", "amp": False,
          "selection": "earliest maximum unrounded validation macro_f1 with optimizer updates"}
CANDIDATE_LRS = [1e-4, 1e-3, 1e-2]


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def verify_source_imports(files):
    for name, expected in files.items():
        module = importlib.import_module(name[:-3].replace("/", "."))
        actual = Path(module.__file__).resolve()
        require(actual == (ROOT / name).resolve() and sha256_file(actual) == expected,
                f"Imported project source differs: {name}")


def publish_artifact(path, writer):
    """Durable no-replace publication, adapted from run_fitting_sensitivity.

    Retain staging on failure. Completion of a multi-file job is marked only by
    its final result, not by the existence of any individual file.
    """
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        staging = Path(stream.name)
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(staging, path)
    staging.unlink()
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    publish_artifact(path, lambda stream: stream.write(data))


def configure_runtime():
    """Once per fresh producer process, before runtime sealing or model work."""
    require(not torch.cuda.is_initialized(), "Configure runtime before CUDA initialization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    # Use only the per-operation precision API (Torch 2.11/2.12), not legacy setters.
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_identity(modules=None, used=None):
    """Bind actual imported files/callables, using the existing fitting pattern."""
    import torchvision
    import torchvision.models.resnet as resnet
    modules = modules or {n: importlib.import_module(n) for n in (
        "numpy", "numpy._core._multiarray_umath", "torch", "torch._C",
        "torch.backends", "torch.backends.cuda", "torch.backends.cudnn", "torchvision",
        "torchvision.models.resnet", "sklearn", "sklearn.metrics._classification",
        "sklearn.linear_model._logistic", "sklearn.preprocessing._data", "scipy.optimize._lbfgsb")}
    used = used or {"resnet18": resnet.resnet18, "resnet_forward": resnet.ResNet.forward,
                    "adamw_step": torch.optim.AdamW.step,
                    "cosine_step": torch.optim.lr_scheduler.CosineAnnealingLR.step,
                    "batchnorm_load": torch.nn.modules.batchnorm._NormBase._load_from_state_dict,
                    "interpolate": torch.nn.functional.interpolate,
                    "cross_entropy": torch.nn.functional.cross_entropy}
    libraries = Path(torch.__file__).parent / "lib"
    return {"python": sys.version, "executable": binding(sys.executable),
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                         "numpy": np.__version__, "sklearn": importlib.import_module("sklearn").__version__},
            "modules": {name: binding(module.__file__) for name, module in modules.items()},
            "used": {name: {**binding(inspect.getfile(inspect.unwrap(obj))),
                            "module": obj.__module__, "qualname": obj.__qualname__}
                     for name, obj in used.items()},
            "torch_native": {p.name: binding(p) for p in sorted(libraries.glob("libtorch_cpu.*"))},
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
            "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "environment": {n: os.environ.get(n) for n in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XFORMERS_DISABLED",
                             "CUBLAS_WORKSPACE_CONFIG", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                             "NVIDIA_TF32_OVERRIDE")}}


def require_current_inputs(inputs, fold):
    require(inputs["catalog"]["sha256"] == CURRENT_CATALOG_SHA256, "Not the accepted current Titan catalog")
    require(inputs["split"]["sha256"] == CURRENT_SPLITS[fold], "Not the accepted split for this fold")


def require_followup_inputs(inputs, fold):
    """Select an exact input contract; the historical default remains unchanged."""
    contract = inputs.get("binding_contract")
    if contract is None:
        return require_current_inputs(inputs, fold)
    require(contract == ANONYMOUS_INPUT_CONTRACT, "Unknown follow-up input contract")
    require(type(fold) is int and fold in range(5), "Invalid anonymous fold")
    require(inputs["catalog"]["sha256"] == ANONYMOUS_CATALOG_SHA256,
            "Not the accepted anonymous Titan catalog")
    require(inputs["split"]["sha256"] == ANONYMOUS_SPLITS[fold],
            "Not the accepted anonymous split for this fold")


def validate_inputs(spec):
    inputs = spec["inputs"]
    if spec["purpose"] == "titan_followup":
        require_followup_inputs(inputs, spec["fold"])
    catalog_path, split_path = (check_file(inputs[name]) for name in ("catalog", "split"))
    catalog, split = (json.loads(p.read_text()) for p in (catalog_path, split_path))
    require(catalog["intensity_space"] == "hisar_log_dn"
            and catalog["benchmark_track"] == inputs["benchmark_track"], "Catalog track/intensity differs")
    require(spec["purpose"] != "synthetic_cpu_test" or catalog["benchmark_track"] == "synthetic_fixture",
            "CPU fixture jobs require a synthetic catalog")
    entries = catalog["tiles"]
    if spec["purpose"] == "titan_followup":
        require(len(entries) == 23380 and catalog["class_names"] == CLASS_ORDER
                and catalog["num_classes"] == 6 and catalog["labels_available"] is True,
                "Current catalog class/count contract differs")
    require(type(split["policy"]["test_fold"]) is int and split["policy"]["test_fold"] == spec["fold"],
            "Split policy fold differs from job")
    ids = [e["tile_id"] for e in entries]
    require(all(isinstance(i, str) and i and Path(i).name == i for i in ids)
            and len(ids) == len(set(ids)), "Invalid or duplicate catalog IDs")
    require(split["catalog_sha256"] == inputs["catalog"]["sha256"]
            and set(split["assignments"]) == set(ids), "Split binding or coverage differs")
    require(set(split["assignments"].values()) <= set(ROLES) | {"buffer_excluded", "selk_holdout"},
            "Unknown split role")
    tiles = (catalog_path.parent / "tiles").resolve(strict=True)
    tile_hashes = {}
    counts, ordered = {}, {}
    for role in ROLES:
        rows = [e for e in entries if split["assignments"][e["tile_id"]] == role]
        labels = [e["label"] for e in rows]
        require(all(type(v) is int and 0 <= v < 6 for v in labels)
                and set(labels) == set(range(6)), f"Every {role} truth class must occur")
        ordered[role] = {"tile_ids": [e["tile_id"] for e in rows], "labels": labels}
        counts[role] = {"total": len(rows), "class_counts": np.bincount(labels, minlength=6).tolist()}
        for row in rows:
            path = (tiles / f"{row['tile_id']}.npy").resolve(strict=True)
            require(path.is_relative_to(tiles), "Tile escapes input root")
            expected = row.get("tile_sha256", row.get("sha256"))
            require(expected is not None and sha256_file(path) == expected, f"Tile hash differs: {row['tile_id']}")
            image = np.load(path, allow_pickle=False, mmap_mode="r")
            require(image.shape == (128, 128) and image.dtype == np.float32, "Raw tile shape/dtype differs")
            tile_hashes[row["tile_id"]] = expected
    require(counts == inputs["counts"], "Partition counts differ")
    datasets = {role: TitanSARDataset(catalog_path, split=role, split_manifest_path=split_path) for role in ROLES}
    normalization = datasets["train"].normalization_provenance
    sampling = normalization["sampling"]
    require(normalization["enabled"] and not normalization["fallback_used"]
            and sampling["sample_count"] > 0 and not sampling["missing_tile_ids"] and not sampling["empty_tile_ids"],
            "Incomplete training normalization")
    require(all(ds.normalization_provenance == normalization for ds in datasets.values()),
            "Partition normalization differs")
    if inputs.get("normalization_sha256"):
        require(digest(normalization) == inputs["normalization_sha256"], "Normalization binding differs")
    return {"catalog": inputs["catalog"], "split": inputs["split"], "benchmark_track": inputs["benchmark_track"],
            **({"binding_contract": inputs["binding_contract"]} if "binding_contract" in inputs else {}),
            "catalog_count": len(entries),
            "counts": counts, "ordered": ordered, "class_order": CLASS_ORDER,
            "tile_inventory_sha256": digest(tile_hashes), "normalization": normalization,
            "normalization_sha256": digest(normalization),
            "training_class_weights": datasets["train"].get_class_weights().tolist()}


def make_loaders(spec, *, roles=ROLES, shuffle_train=False, seed):
    inputs = spec["inputs"]
    result = {}
    for role in roles:
        require(role in ROLES, "Unknown requested role")
        dataset = TitanSARDataset(inputs["catalog"]["path"], split=role,
                                 split_manifest_path=inputs["split"]["path"])
        generator = torch.Generator().manual_seed(seed + ROLES.index(role))
        result[role] = DataLoader(dataset, batch_size=spec["recipe"]["batch_size"],
                                  shuffle=shuffle_train and role == "train", drop_last=False,
                                  num_workers=0, worker_init_fn=seed_worker, generator=generator)
    return result


def select_epoch(history):
    require(bool(history), "Empty epoch history")
    eligible = []
    for index, row in enumerate(history, 1):
        require(row["epoch"] == index, "Epoch order differs")
        matrix = np.asarray(row["val_confusion_matrix"])
        require(matrix.shape == (6, 6) and matrix.dtype.kind in "iu" and (matrix >= 0).all()
                and (matrix.sum(axis=1) > 0).all(), "Invalid six-class validation confusion")
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        score = float(np.mean(2 * matrix.diagonal() / denominator))
        require(np.isfinite(row["val_macro_f1"]) and np.isclose(score, row["val_macro_f1"], rtol=0, atol=1e-15)
                and np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"]), "Invalid epoch score/loss")
        require(type(row["optimizer_steps"]) is int and row["optimizer_steps"] >= 0, "Invalid update count")
        if row["optimizer_steps"]:
            eligible.append(row)
    require(bool(eligible), "No epoch with optimizer updates")
    return max(eligible, key=lambda r: (r["val_macro_f1"], -r["epoch"]))["epoch"]


def load_followup_cache(path, expected):
    path = Path(path)
    stem = path.stem
    identity = expected["identity"]
    require(identity["model"] in ("dinov2_random_matched", "dinov2_pretrained_matched"), "Unknown follow-up model")
    require(identity["initialization"] == ("random" if identity["model"] == "dinov2_random_matched" else "pretrained"),
            "Model initialization differs")
    if expected["purpose"] == "titan_followup":
        require(identity["revision"] == DINOV2_REVISION and identity["source_python_inventory_sha256"] == DINO_SOURCE_SHA256
                and identity["constructor"] == DINO_CONSTRUCTOR and identity["transform"] == TRANSFORM
                and type(identity["encoder_seed"]) is int and identity["encoder_seed"] in range(42, 47)
                and len(identity["initial_state_sha256"]) == 64 and identity["dtype"] == "float32"
                and identity["backend"] == "torch_sdpa_no_xformers", "Incomplete official DINO identity")
        if identity["initialization"] == "pretrained":
            require(identity["checkpoint"]["sha256"] == DINOV2_SHA256, "Pretrained cache weight release differs")
    assert_cache_real_weights(path.parent, stem, identity["model"], strict=False)
    metadata = load_feature_metadata(path.parent, stem)
    require(metadata["schema_version"] == "model-followup-v1", "Cache schema differs")
    require(metadata["weights_source"] == identity["initialization"], "Unexpected weights source")
    provenance = metadata["provenance"]
    for key in ("identity", "inputs", "fold", "role", "transform", "purpose"):
        require(provenance[key] == expected[key], f"Cache {key} differs")
    require(provenance["transform"] == TRANSFORM, "Unsupported transform")
    arrays = {}
    for suffix, key in (("feats", "feature_sha256"), ("ids", "tile_ids_sha256"), ("labels", "labels_sha256")):
        member = path.with_name(stem.replace("_feats", "_" + suffix) + ".npy")
        require(provenance.get(key) == sha256_file(member), f"Cache companion differs: {suffix}")
        arrays[suffix] = np.load(member, allow_pickle=False)
    ids, labels, features = arrays["ids"], arrays["labels"], arrays["feats"]
    require(ids.ndim == labels.ndim == 1 and ids.dtype.kind == "U" and labels.dtype.kind in "iu",
            "Invalid companion dtype/shape")
    require(ids.tolist() == expected["tile_ids"] and len(ids) == len(set(ids.tolist()))
            and labels.tolist() == expected["labels"], "Cache row identity differs")
    require(features.dtype == np.float32 and features.shape == (len(ids), 768)
            and np.isfinite(features).all() and set(labels.tolist()) == set(range(6)), "Invalid feature matrix/classes")
    return arrays


def checked_output(spec):
    output = Path(spec["output_dir"]).resolve()
    protected = [ROOT]
    for value in spec.get("inputs", {}).values():
        if isinstance(value, dict) and "path" in value:
            protected.append(Path(value["path"]).resolve().parent)
    if spec.get("features"):
        protected.append(Path(spec["features"]["path"]).resolve().parent)
    protected.extend(Path(candidate["path"]).resolve().parent
                     for candidate in spec.get("tuning_results", []))
    if "loss_comparison" in spec:
        from scripts.loss_comparison import validate_spec
        validate_spec(spec)
        protected.append(Path(spec["loss_comparison"]["control"]["path"]).resolve().parent)
    if spec.get("execution", {}).get("cuda_dependencies"):
        protected.append(Path(spec["execution"]["cuda_dependencies"]["path"]).resolve().parent)
    model = spec.get("model", {})
    if model.get("source_dir"):
        protected.append(Path(model["source_dir"]).resolve())
    if model.get("checkpoint"):
        protected.append(Path(model["checkpoint"]["path"]).resolve().parent)
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            "Output overlaps source or inputs")
    return output
