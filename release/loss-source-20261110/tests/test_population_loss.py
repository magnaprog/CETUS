"""Arithmetic and sequence checks for population loss normalization."""
import copy
import hashlib
import json
import math
import struct

import numpy as np
import pytest
import torch

from titansar.models.followup import prepare_rgb224
from titansar.models.image_training import population_weight_mean, train_epoch


class MeanLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, images):
        return self.fc(images.mean((-2, -1)))


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def samples():
    return [{"image": torch.full((1, 128, 128), v), "label": y, "tile_id": str(i)}
            for i, (v, y) in enumerate(zip((.1, .3, .5, .7, .9), (0, 0, 1, 0, 1)))]


def test_population_mean_uses_counts_and_retains_weight_bits():
    weights = torch.tensor([.2, 1.8], dtype=torch.float32)
    before = weights.clone()
    result = population_weight_mean(weights, [9, 1])
    expected64 = math.fsum((9 * float(weights[0]), float(weights[1]))) / 10
    assert result["m64"] == expected64
    assert result["m32"] == float(np.float32(expected64))
    assert result["weight_bits"] == [struct.unpack("!I", struct.pack("!f", float(w)))[0] for w in weights]
    assert result["m32_bits"] == struct.unpack("!I", struct.pack("!f", np.float32(expected64)))[0]
    assert result["m32"] != float(weights.mean())
    assert torch.equal(weights, before)


@pytest.mark.parametrize("counts", ([9, 0], [9, -1], [9, True], [9., 1], [9], [9, 1, 1]))
def test_population_rejects_invalid_counts(counts):
    with pytest.raises(ValueError, match="counts"):
        population_weight_mean(torch.tensor([.2, 1.8]), counts)


@pytest.mark.parametrize("weights", (torch.tensor([1., 2.], dtype=torch.float64),
                                   torch.tensor([1., float("nan")]), torch.tensor([0., 1.])))
def test_population_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        population_weight_mean(weights, [9, 1])


def test_two_sgd_steps_match_independent_gradient_with_partial_batch():
    torch.manual_seed(712)
    model = MeanLinear()
    reference = copy.deepcopy(model)
    weights = torch.tensor([.2, 1.8])
    mean = population_weight_mean(weights, [9, 1])["m32"]
    loader = torch.utils.data.DataLoader(samples(), batch_size=3, shuffle=False)
    # Explicit softmax derivative; neither cross_entropy nor autograd supplies
    # this expected update. The second batch has two images, not three.
    expected_records = []
    for batch in loader:
        x = prepare_rgb224(batch["image"]).mean((-2, -1))
        labels = batch["label"]
        with torch.no_grad():
            logits = x @ reference.fc.weight.T + reference.fc.bias
            residual = logits.softmax(1) - torch.nn.functional.one_hot(labels, 2)
            divisor = np.float32(np.float32(len(labels)) * np.float32(mean))
            residual *= weights[labels, None] / float(divisor)
            reference.fc.weight -= .01 * residual.T @ x
            reference.fc.bias -= .01 * residual.sum(0)
        expected_records.append((len(labels), float(divisor), float(weights[labels].sum())))
    stats = train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=.01),
                        class_weights=weights, population_mean_weight=mean, record_batches=True)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    assert [(r["examples"], r["optimization_denominator"], r["native_denominator"])
            for r in stats["batch_records"]] == expected_records
    assert stats["loss_denominator"] == sum(r[2] for r in expected_records)
    assert stats["train_loss"] == stats["loss_numerator"] / stats["loss_denominator"]
    assert stats["examples"] == 5 and stats["optimizer_steps"] == 2


def test_logging_preserves_native_updates_rng_and_loader_sequence():
    torch.manual_seed(34)
    first = MeanLinear()
    second = copy.deepcopy(first)
    weights = torch.tensor([.2, 1.8])
    loaders = [torch.utils.data.DataLoader(samples(), batch_size=3, shuffle=True,
                generator=torch.Generator().manual_seed(10000)) for _ in range(3)]
    expected_bytes = []
    for batch in loaders[2]:
        expected_bytes.append((json.dumps([list(batch["tile_id"]), batch["label"].tolist()],
                               separators=(",", ":"), ensure_ascii=True) + "\n").encode())
    state = torch.get_rng_state().clone()
    native = train_epoch(first, loaders[0], torch.optim.SGD(first.parameters(), lr=.01), class_weights=weights)
    rng_native = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    logged = train_epoch(second, loaders[1], torch.optim.SGD(second.parameters(), lr=.01),
                         class_weights=weights, record_batches=True)
    assert torch.equal(torch.get_rng_state(), rng_native)
    assert torch.equal(loaders[0].generator.get_state(), loaders[1].generator.get_state())
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    assert {key: logged[key] for key in native} == native
    assert logged["training_order_sha256"] == hashlib.sha256(b"".join(expected_bytes)).hexdigest()
    assert [r["order_sha256"] for r in logged["batch_records"]] == [hashlib.sha256(b).hexdigest() for b in expected_bytes]
    expected_counts = [[json.loads(b)[1].count(c) for c in range(2)] for b in expected_bytes]
    assert [r["class_counts"] for r in logged["batch_records"]] == expected_counts


@pytest.mark.parametrize("mean", (0., -1., float("nan"), float("inf"), .1, True))
def test_invalid_mean_refused_before_step(mean):
    model = MeanLinear()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="Population mean|population mean"):
        train_epoch(model, torch.utils.data.DataLoader(samples(), batch_size=3),
                    torch.optim.SGD(model.parameters(), lr=.01), class_weights=torch.tensor([.2, 1.8]),
                    population_mean_weight=mean)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
