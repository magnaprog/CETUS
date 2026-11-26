"""Test feature extraction and cached classifiers on synthetic radar tiles."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_probing as runner
from cetus_sar.configs.defaults import CETUSConfig
from cetus_sar.models import foundation_models


def test_real_cpu_probe_and_cache_on_synthetic_tiles(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    entries = []
    for role, repeats in [("train", 2), ("test", 1)]:
        for label in range(6):
            for repeat in range(repeats):
                tile_id = f"{role}_{label}_{repeat}"
                entries.append(dict(tile_id=tile_id, label=label,
                                    center_lat=0., center_lon=0.))
                np.save(tiles / f"{tile_id}.npy",
                        np.arange(1, 65, dtype=np.float32).reshape(8, 8) + label)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(tiles=entries, intensity_space="hisar_log_dn",
                                      benchmark_track="synthetic-classifier-workflow")))
    split = tmp_path / "split.json"
    split.write_text(json.dumps(dict(
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        assignments={e["tile_id"]: e["tile_id"].split("_")[0] for e in entries})))
    encoder = SimpleNamespace(model=torch.nn.Identity(), weights_source="random_init",
        weights_sha256="a" * 64, model_revision="synthetic-smoke-only",
        extract_features=lambda images: images.flatten(1)[:, :8])
    monkeypatch.setattr(runner, "get_model", lambda *a, **kw: encoder)
    monkeypatch.setattr(runner, "CETUSConfig", lambda: CETUSConfig(
        probe_epochs=2, probe_batch_size=6, knn_k=3))
    kwargs = dict(model_name="random_init", device="cpu", titan_catalog=str(catalog),
        titan_split_manifest=str(split), earth_catalog=None, earth_split_manifest=None,
        output_dir=str(tmp_path / "extraction"), batch_size=4, seeds=[0, 1],
        num_workers=0, within_titan_only=True, allow_placeholder=False)
    extracted = runner.run_single_model(**kwargs, features_only=True)
    monkeypatch.setattr(runner, "get_model", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("Cached analysis loaded an encoder")))
    result = runner.run_single_model(**kwargs,
        cached_features_dir=str(tmp_path / "extraction/features"))
    assert result["source_metadata"] == extracted["source_metadata"]
    assert result["source_metadata"]["normalization_provenance"]["fallback_used"] is False
    heads = result["linear_probe"]["titan_to_titan"]["seed_results"]
    assert [h["seed"] for h in heads] == [0, 1]
    for head in heads:
        assert head["true_labels"] == list(range(6))
        assert head["tile_ids"] == [f"test_{i}_0" for i in range(6)]
        probabilities = np.asarray(head["probabilities"])
        assert probabilities.shape == (6, 6) and np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(1), 1., atol=1e-6)
        np.testing.assert_array_equal(probabilities.argmax(1), head["predictions"])
        counts = np.zeros((6, 6), dtype=int)
        np.add.at(counts, (head["true_labels"], head["predictions"]), 1)
        np.testing.assert_array_equal(counts, head["confusion_matrix"])
    assert result["knn"]["titan_to_titan"]["tile_ids"] == heads[0]["tile_ids"]
    assert "domain_gap" not in result


@pytest.mark.parametrize(("current", "legacy", "strict"), [
    (None, None, False),
    ("1", None, True),
    (None, "tRuE", True),
    ("off", "yes", True),
    ("YES", "0", True),
    ("0", "false", False),
])
def test_real_weight_switches_reach_model_loader(monkeypatch, current, legacy, strict):
    for name, value in [
        ("CETUS_REQUIRE_REAL_WEIGHTS", current),
        ("TITANSAR_REQUIRE_REAL_WEIGHTS", legacy),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(foundation_models.CROMAWrapper, "load",
                        lambda self, allow_placeholder: calls.append(allow_placeholder))
    foundation_models.get_model("croma", device="cpu")
    foundation_models.get_model("croma", device="cpu", allow_placeholder=False)
    assert calls == [not strict, False]
