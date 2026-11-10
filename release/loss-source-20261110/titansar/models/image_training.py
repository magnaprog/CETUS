"""Ordinary full-image training and evaluation, without checkpoint selection."""
from __future__ import annotations

import hashlib
import json
import math
import struct

import numpy as np
import torch

from titansar.models.baselines import evaluate_predictions
from titansar.models.followup import prepare_rgb224, require


def population_weight_mean(class_weights, class_counts):
    """Population mean of the retained float32 training weights."""
    require(class_weights.dtype == torch.float32 and class_weights.ndim == 1
            and bool(torch.isfinite(class_weights).all()) and bool((class_weights > 0).all()),
            "Population normalization requires positive float32 weights")
    require(isinstance(class_counts, (list, tuple)) and len(class_counts) == len(class_weights)
            and all(type(n) is int and n > 0 for n in class_counts), "Invalid training class counts")
    weights = class_weights.detach().cpu().tolist()
    mean64 = math.fsum(float(n) * w for n, w in zip(class_counts, weights)) / sum(class_counts)
    mean32 = float(torch.tensor(mean64, dtype=torch.float32))
    require(math.isfinite(mean32) and mean32 > 0, "Invalid population mean weight")
    bits = lambda x: struct.unpack("!I", struct.pack("!f", x))[0]
    return {"class_counts": list(class_counts), "native_weights": weights,
            "weight_bits": [bits(w) for w in weights], "m64": mean64,
            "m32": mean32, "m32_bits": bits(mean32)}


def train_epoch(model, train_loader, optimizer, *, class_weights, device="cpu",
                population_mean_weight=None, record_batches=False):
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
    require(type(record_batches) is bool, "Invalid batch recording option")
    population_mean = None
    if population_mean_weight is not None:
        require(type(population_mean_weight) is float and math.isfinite(population_mean_weight)
                and population_mean_weight > 0, "Invalid population mean weight")
        population_mean = torch.tensor(population_mean_weight, dtype=torch.float32, device=device)
        require(float(population_mean) == population_mean_weight, "Population mean must already be float32")
    records = []
    order = hashlib.sha256() if record_batches else None
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
        loss_denominator = weight_sum
        if population_mean is not None:
            loss_denominator = torch.tensor(len(labels), dtype=torch.float32, device=device) * population_mean
            require(bool(torch.isfinite(loss_denominator)) and float(loss_denominator) > 0,
                    "Invalid population loss denominator")
        loss = loss_sum / loss_denominator
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
                "Missing or nonfinite full-model gradient")
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters), "Nonfinite updated parameter")
        numerator += float(loss_sum.detach())
        denominator += float(weight_sum)
        examples += len(labels)
        steps += 1
        if record_batches:
            ids = list(batch["tile_id"])
            require(len(ids) == len(labels) and all(isinstance(i, str) for i in ids), "Invalid batch IDs")
            labels_cpu = labels.detach().cpu()
            data = (json.dumps([ids, labels_cpu.tolist()], separators=(",", ":"),
                               ensure_ascii=True) + "\n").encode()
            order.update(data)
            records.append({"step": steps, "examples": len(labels),
                            "order_sha256": hashlib.sha256(data).hexdigest(),
                            "class_counts": torch.bincount(labels_cpu, minlength=len(weights)).tolist(),
                            "loss_numerator": float(loss_sum.detach()),
                            "native_denominator": float(weight_sum),
                            "optimization_denominator": float(loss_denominator),
                            "optimization_loss": float(loss.detach())})
    result = {"train_loss": numerator / denominator, "loss_numerator": numerator,
            "loss_denominator": denominator, "examples": examples,
            "optimizer_steps": steps, "finite_gradients": True}
    if record_batches:
        result.update(batch_records=records, training_order_sha256=order.hexdigest())
    return result


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
