"""Admission of one population-weight comparison against its immutable control."""
from __future__ import annotations

import math

import torch

from scripts import model_followup_support as support
from titansar.models.followup import check_file, require, state_digest
from titansar.models.image_training import population_weight_mean


ORIGINAL_COMMIT = "1f3fc8fe0e831777e3188d70caf2c5450b656333"
# Exact eleven-file source lock in the accepted original ImageNet controls.
ORIGINAL_FILES = {
    "scripts/run_model_followup.py": "bfa30e12e25ac41ce5755b512fd0e0647ab1f1689a6e7857421b2a2fd4bdc270",
    "scripts/model_followup_support.py": "1d381693e0abb38c299569cd70630acb49e9b048daae630f7ab4781f4209118f",
    "titansar/models/followup.py": "8f6f986efcfd5b5d0f2f231ee50ad592824757a2b32937452cfc064f4aaa7d66",
    "titansar/models/image_training.py": "3474ddf206feea2178da43d973dbfff84e22b7356c140cbc90a663e8d51ad917",
    "scripts/run_finetuning.py": "cf8a388e54f9686addb84da2256e7bcc3c8d904cf6e13e6057ce34700632c49c",
    "titansar/models/foundation_models.py": "ae9bdfde05ff6ac955e4fa017762337d5ef4e6d40d421ffc89d31b50db077b14",
    "titansar/models/probing.py": "75d270bf00e763530a9c686b60ce7ecef53e46d25caf5fec692a18199157e946",
    "titansar/models/baselines.py": "156f0ddb0e9419cbdeef3e7cb9b11c17989248caf6457b7fd14870c705fb18c1",
    "titansar/data/dataset.py": "68150c666fa3f18b0726a63944ae260a9668005fc804e034d642f2e37ad58fa0",
    "titansar/configs/defaults.py": "8352fd3b4051afbfaa1ce01a25a9e5aab8fae9086fe783c020acd1ad5133a709",
    "titansar/reproducibility.py": "b263b955c7d7186d5e95d95831efcd60e860185a22454cd8981457d5117f2a05",
}


def validate_spec(spec):
    comparison = spec["loss_comparison"]
    require(isinstance(comparison, dict)
            and set(comparison) == {"schema_version", "control", "original_source"}
            and comparison["schema_version"] == "population-weight-v1", "Invalid loss comparison schema")
    require(isinstance(comparison["control"], dict)
            and set(comparison["control"]) == {"path", "sha256"}, "Invalid original control binding")
    original = comparison["original_source"]
    require(isinstance(original, dict) and set(original) == {"commit", "files"}
            and isinstance(original["files"], dict) and set(original["files"]) == set(ORIGINAL_FILES),
            "Original source requires exactly eleven files")
    require(spec["kind"] == "resnet18" and spec["phase"] == "final"
            and spec["model"]["initialization"] == "pretrained"
            and spec["recipe"]["lr"] == 1e-4, "Loss comparison requires final pretrained ResNet18 at LR 1e-4")
    require("tuning_results" not in spec, "Loss comparison must not supply tuning_results")
    if spec["purpose"] == "titan_followup":
        require(original == {"commit": ORIGINAL_COMMIT, "files": ORIGINAL_FILES},
                "Original scientific source lock differs")
        require(spec["device"] == "cuda:0", "Scientific loss comparison requires its original GPU")


def prepare(spec, inputs, completed):
    """Authenticate the complete control; reuse its selected-LR receipt, never tune."""
    validate_spec(spec)
    comparison = spec["loss_comparison"]
    root, control = completed(comparison["control"], "resnet18")
    prior = control["specification"]
    require(prior["source"] == comparison["original_source"], "Original control source differs")
    require("loss_comparison" not in prior, "Control must use the original native loss")
    # These are the only permitted differences between the old and new job.
    omit = {"source", "output_dir", "tuning_results", "loss_comparison"}
    require({k: v for k, v in prior.items() if k not in omit}
            == {k: v for k, v in spec.items() if k not in omit}, "Original control task/runtime/recipe differs")
    require(control["specification_sha256"] == support.digest(prior)
            and all(control[k] == spec[k] for k in ("kind", "purpose", "fold"))
            and control["inputs"] == inputs, "Original control specification/inputs differ")
    require(control["training_class_weights"] == inputs["training_class_weights"],
            "Original control training weights differ")
    epochs = prior["recipe"]["epochs"]
    required = {"selected.pt", "final.pt", "initial_state.pt", "started.json", "preparation.json"}
    required.update(f"epoch-{epoch:03d}.json" for epoch in range(1, epochs + 1))
    if spec["purpose"] == "titan_followup":
        required.add("gpu-started.json")
        gpu = control["gpu_execution"]
        require(gpu["device"] == spec["device"] and gpu["uuid"] == spec["execution"]["gpu_uuid"]
                and gpu["cuda_dependencies"] == spec["execution"]["cuda_dependencies"],
                "Original control GPU/receipt differs")
    require(required <= set(control["outputs"]), "Original control artifacts are incomplete")
    history = control["history"]
    count = inputs["counts"]["train"]["total"]
    steps = math.ceil(count / prior["recipe"]["batch_size"])
    require(len(history) == epochs and all(row["examples"] == count and row["optimizer_steps"] == steps
                                         and row["finite_gradients"] is True for row in history),
            "Original control training budget is incomplete")
    require(support.select_epoch(history) == control["selected_epoch"], "Original control selected epoch differs")
    test = control["test"]
    require(test["tile_ids"] == inputs["ordered"]["test"]["tile_ids"]
            and test["labels"] == inputs["ordered"]["test"]["labels"]
            and test["class_order"] == list(range(6)), "Original control test membership differs")
    selection = control["selection"]
    candidates = selection["candidates"]
    require(selection["rule"] == "maximum validation macro_f1, then lower LR"
            and len(candidates) == 3 and sorted(row["lr"] for row in candidates) == support.CANDIDATE_LRS
            and all(math.isfinite(row["score"]) and 0 <= row["score"] <= 1 for row in candidates)
            and [row["result"] for row in candidates] == prior["tuning_results"],
            "Original selected-LR receipt differs")
    require(selection["lr"] == spec["recipe"]["lr"]
            == max(candidates, key=lambda row: (row["score"], -row["lr"]))["lr"],
            "Original selected learning rate differs")
    weights = torch.tensor(inputs["training_class_weights"], dtype=torch.float32)
    require(weights.tolist() == inputs["training_class_weights"], "Original weights are not float32 values")
    normalizer = population_weight_mean(weights, inputs["counts"]["train"]["class_counts"])
    return {"control": comparison["control"], "identity": control["identity"],
            "initial_state": {"path": str(root / "initial_state.pt"),
                              "sha256": control["outputs"]["initial_state.pt"]},
            "selection": selection, "normalizer": normalizer}


def verify_initial_state(model, identity, comparison):
    """Refuse before an optimizer exists unless control and built tensors agree."""
    check_file(comparison["control"])
    path = check_file(comparison["initial_state"])
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(saved, dict) and saved and all(isinstance(t, torch.Tensor) for t in saved.values()),
            "Original initial state must contain tensors")
    require(identity == comparison["identity"], "Built identity differs from original control")
    require(state_digest(saved) == identity["initial_state_sha256"] == state_digest(model.state_dict()),
            "Initial tensor state differs from original control")
    check_file(comparison["initial_state"])
