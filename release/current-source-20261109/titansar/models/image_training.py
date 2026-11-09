"""Ordinary full-image training and evaluation, without checkpoint selection."""
from __future__ import annotations

import numpy as np
import torch

from titansar.models.baselines import evaluate_predictions
from titansar.models.followup import prepare_rgb224, require


def train_epoch(model, train_loader, optimizer, *, class_weights, device="cpu"):
    require(len(train_loader) > 0, "Empty training loader")
    parameters = list(model.parameters())
    supplied = [p for group in optimizer.param_groups for p in group["params"]]
    require(all(p.requires_grad for p in parameters)
            and len(supplied) == len(parameters) == len({id(p) for p in supplied})
            and {id(p) for p in supplied} == {id(p) for p in parameters},
            "Full training requires each model parameter exactly once")
    weights = class_weights.to(device=device, dtype=torch.float32)
    require(weights.ndim == 1 and bool(torch.isfinite(weights).all()) and bool((weights > 0).all()),
            "Invalid training class weights")
    model.train()
    numerator = denominator = 0.0
    examples = steps = 0
    for batch in train_loader:
        labels = batch["label"].to(device)
        images = prepare_rgb224(batch["image"].to(device))
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_sum = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")
        weight_sum = weights[labels].sum()
        require(bool(torch.isfinite(loss_sum)), "Nonfinite training loss")
        (loss_sum / weight_sum).backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
                "Missing or nonfinite full-model gradient")
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "Nonfinite updated parameter")
        numerator += float(loss_sum.detach())
        denominator += float(weight_sum)
        examples += len(labels)
        steps += 1
    return {"train_loss": numerator / denominator, "loss_numerator": numerator,
            "loss_denominator": denominator, "examples": examples,
            "optimizer_steps": steps, "finite_gradients": True}


@torch.no_grad()
def evaluate_images(model, loader, *, num_classes=6, device="cpu", class_weights=None):
    require(len(loader) > 0, "Empty evaluation loader")
    model.eval()
    logits, labels, ids = [], [], []
    weights = (torch.ones(num_classes) if class_weights is None else class_weights).to(device)
    require(weights.shape == (num_classes,) and bool(torch.isfinite(weights).all())
            and bool((weights > 0).all()), "Invalid evaluation class weights")
    loss_sum = weight_sum = 0.0
    for batch in loader:
        output = model(prepare_rgb224(batch["image"].to(device)))
        require(output.shape == (len(batch["label"]), num_classes)
                and bool(torch.isfinite(output).all()), "Invalid evaluation logits")
        target = batch["label"].to(device)
        loss_sum += float(torch.nn.functional.cross_entropy(output, target, weight=weights, reduction="sum"))
        weight_sum += float(weights[target].sum())
        logits.append(output.cpu())
        labels.extend(batch["label"].tolist())
        ids.extend(batch["tile_id"])
    require(len(ids) == len(set(ids)), "Duplicate evaluation tile ID")
    values = torch.cat(logits)
    probabilities = values.softmax(dim=1).numpy()
    truth = np.asarray(labels, dtype=np.int64)
    require(np.array_equal(np.unique(truth), np.arange(num_classes)), "Every evaluation truth class must occur")
    prediction = values.argmax(dim=1).numpy()
    result = evaluate_predictions(truth, prediction, num_classes, probabilities)
    matrix = np.asarray(result["confusion_matrix"])
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            and np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0), "Invalid native probabilities")
    result.update(tile_ids=ids, labels=truth.tolist(), logits=values.tolist(),
                  class_order=list(range(num_classes)), loss=loss_sum / weight_sum,
                  loss_numerator=loss_sum, loss_denominator=weight_sum,
                  logits_dtype=str(values.numpy().dtype), probabilities_dtype=str(probabilities.dtype),
                  probability_argmax_disagreements=int(np.sum(prediction != probabilities.argmax(1))),
                  true_positive=matrix.diagonal().tolist(),
                  false_positive=(matrix.sum(0) - matrix.diagonal()).tolist(),
                  false_negative=(matrix.sum(1) - matrix.diagonal()).tolist())
    return result
