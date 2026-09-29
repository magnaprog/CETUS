"""Normalization records must describe the training pixels actually used."""

import hashlib
import json

import numpy as np

from titansar.data.dataset import TitanSARDataset, _sample_train_pixels


def test_sampling_metadata_preserves_values_and_exposes_omissions(tmp_path):
    entries = [{"tile_id": name} for name in ("valid", "empty", "missing")]
    np.save(tmp_path / 'valid.npy', np.array([0, 1, 2, 3, np.nan, np.inf]))
    np.save(tmp_path / 'empty.npy', np.zeros(4))
    original = _sample_train_pixels(entries, tmp_path, px_per_tile=2)
    samples, record = _sample_train_pixels(entries, tmp_path, px_per_tile=2, return_metadata=True)
    np.testing.assert_array_equal(original, samples)
    expected = np.array([1., 2., 3.])[np.random.default_rng(0).choice(3, 2, replace=False)]
    np.testing.assert_array_equal(samples, expected)
    assert record['selected_tile_ids'] == ['valid', 'empty', 'missing']
    assert record['used_tile_ids'] == ['valid']
    assert record['empty_tile_ids'] == ['empty']
    assert record['missing_tile_ids'] == ['missing']
    assert record['sample_count'] == 2
    assert record['sample_float64_le_sha256'] == hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest()


def test_heldout_pixels_cannot_change_recorded_training_normalization(tmp_path):
    tiles = tmp_path / 'tiles'
    tiles.mkdir()
    entries = [{'tile_id': role, 'split': role, 'label': 0} for role in ('train', 'val', 'test')]
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'tiles': entries, 'intensity_space': 'hisar_log_dn',
                               'benchmark_track': 'test-track'}))
    values = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    np.save(tiles / 'train.npy', values)
    for role in ('val', 'test'):
        np.save(tiles / f'{role}.npy', values * 1000)
    datasets = [TitanSARDataset(path, split=role) for role in ('train', 'val', 'test')]
    record = datasets[0].normalization_provenance
    assert all(d.normalization_provenance == record for d in datasets)
    assert record['sampling']['selected_tile_ids'] == ['train']
    assert record['lower'] == 1.99
    assert record['upper'] == 99.01
    assert record['fallback_used'] is False
    assert datasets[0].benchmark_track == 'test-track'
    np.save(tiles / 'test.npy', np.zeros((10, 10), dtype=np.float32))
    assert TitanSARDataset(path, split='test').normalization_provenance == record
    np.save(tiles / 'train.npy', values + 1)
    updated = TitanSARDataset(path).normalization_provenance
    assert updated['sampling']['sample_float64_le_sha256'] != record['sampling']['sample_float64_le_sha256']
    assert updated['lower'] == record['lower'] + 1


def test_empty_and_constant_samples_record_fallback(tmp_path):
    (tmp_path / 'tiles').mkdir()
    path = tmp_path / 'catalog.json'
    entry = {'tile_id': 'constant', 'split': 'train', 'label': 0}
    path.write_text(json.dumps({'tiles': [entry], 'intensity_space': 'hisar_log_dn'}))
    for value in (0., 5.):
        np.save(tmp_path / 'tiles/constant.npy', np.full((8, 8), value, dtype=np.float32))
        record = TitanSARDataset(path).normalization_provenance
        assert record['fallback_used'] is True
        assert (record['lower'], record['upper']) == (0., 255.)
        assert record['sampling']['sample_count'] == (0 if value == 0 else 64)
