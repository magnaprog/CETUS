"""Feature reuse must reject a different fold, mislabeled rows and duplicate IDs."""

import numpy as np
import pytest

from titansar.models.probing import load_bound_feature_cache, save_feature_metadata
from titansar.reproducibility import sha256_file


@pytest.mark.parametrize('fault', [None, 'fold', 'labels', 'duplicate', 'coverage', 'missing_binding'])
def test_cache_is_bound_to_catalog_labels_and_exact_split(tmp_path, fault):
    ids = np.array(['a', 'b'] if fault != 'duplicate' else ['a', 'a'])
    labels = np.array([0, 1] if fault != 'labels' else [1, 0])
    stem = 'titan_train_feats'
    provenance = {'titan_split_manifest_sha256': 'fold0'}
    for suffix, array, key in [
        ('feats', np.ones((2, 3)), 'feature_sha256'),
        ('ids', ids, 'tile_ids_sha256'),
        ('labels', labels, 'labels_sha256'),
    ]:
        path = tmp_path / f'titan_train_{suffix}.npy'
        np.save(path, array)
        provenance[key] = sha256_file(path)
    if fault == 'missing_binding':
        del provenance['labels_sha256']
    save_feature_metadata(tmp_path, stem, 'dinov2', 'pretrained', 2, 3, provenance)
    expected = {'titan_split_manifest_sha256': 'fold1' if fault == 'fold' else 'fold0'}
    expected_labels = {'a': 0, 'c' if fault == 'coverage' else 'b': 1}
    if fault:
        with pytest.raises(RuntimeError):
            load_bound_feature_cache(tmp_path, stem, 'dinov2', expected, expected_labels)
    else:
        features, actual_labels, actual_ids, _ = load_bound_feature_cache(
            tmp_path, stem, 'dinov2', expected, expected_labels)
        assert features.shape == (2, 3)
        assert actual_labels.tolist() == [0, 1]
        assert actual_ids == ['a', 'b']
