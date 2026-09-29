"""The random control must retain its architecture and identify its weights."""

import sys
from types import SimpleNamespace

import pytest
import torch

from titansar.models.foundation_models import RandomInitWrapper


def test_random_control_is_repeatable_and_preserves_cpu_rng(monkeypatch):
    def create_model(name, **kwargs):
        assert name == 'vit_base_patch14_dinov2.lvd142m'
        assert kwargs == {'pretrained': False, 'in_chans': 1, 'img_size': 224}
        model = torch.nn.Linear(3, 2)
        model.embed_dim = 2
        return model

    monkeypatch.setitem(sys.modules, 'timm', SimpleNamespace(
        create_model=create_model, __version__='fixture'))
    before = torch.get_rng_state().clone()
    first, second, other = [RandomInitWrapper('cpu', seed) for seed in [42, 42, 43]]
    for model in [first, second, other]:
        model.load(allow_placeholder=False)
    assert torch.equal(before, torch.get_rng_state())
    assert first.weights_sha256 == second.weights_sha256
    assert first.weights_sha256 != other.weights_sha256
    assert first.model_revision.endswith('vit_base_patch14_dinov2.lvd142m:seed=42')


def test_random_control_fails_if_requested_architecture_cannot_load(monkeypatch):
    calls = []

    def create_model(name, **kwargs):
        calls.append(name)
        raise RuntimeError('architecture unavailable')

    monkeypatch.setitem(sys.modules, 'timm', SimpleNamespace(
        create_model=create_model, __version__='fixture'))
    with pytest.raises(RuntimeError, match='architecture unavailable'):
        RandomInitWrapper('cpu').load(allow_placeholder=False)
    assert calls == ['vit_base_patch14_dinov2.lvd142m']
