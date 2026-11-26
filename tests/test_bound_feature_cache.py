"""Feature reuse must reject a different fold, mislabeled rows and duplicate IDs."""

import numpy as np
import pytest

from cetus_sar.models.probing import load_bound_feature_cache, save_feature_metadata
from cetus_sar.reproducibility import sha256_file


def write_cache(directory, model, weights, features, labels, ids, fault=None):
    stem = 'titan_train_feats'
    provenance = {'titan_split_manifest_sha256': 'fold0'}
    if model == 'dofa':
        provenance['band_identifier_ghz'] = 13.78
    for suffix, array, key in [
        ('feats', features, 'feature_sha256'),
        ('ids', ids, 'tile_ids_sha256'),
        ('labels', labels, 'labels_sha256'),
    ]:
        path = directory / f'titan_train_{suffix}.npy'
        np.save(path, array)
        provenance[key] = sha256_file(path)
    if fault == 'missing_binding':
        del provenance['labels_sha256']
    save_feature_metadata(directory, stem, model, weights,
                          n_features=features.shape[0], feature_dim=features.shape[1],
                          provenance=provenance)
    return stem


@pytest.mark.parametrize('weights_source,strict', [
    ('pretrained', None), ('pretrained', False),
    ('random_placeholder', False), ('random_placeholder', True),
])
@pytest.mark.parametrize('fault', [None, 'fold', 'labels', 'duplicate', 'coverage', 'missing_binding'])
def test_cache_is_bound_to_catalog_labels_and_exact_split(tmp_path, fault, weights_source, strict):
    ids = np.array(['a', 'b'] if fault != 'duplicate' else ['a', 'a'])
    labels = np.array([0, 1] if fault != 'labels' else [1, 0])
    stem = write_cache(tmp_path, 'dinov2', weights_source, np.ones((2, 3)), labels, ids, fault)
    policy = {} if strict is None else {'strict': strict}
    expected = {'titan_split_manifest_sha256': 'fold1' if fault == 'fold' else 'fold0'}
    expected_labels = {'a': 0, 'c' if fault == 'coverage' else 'b': 1}
    if fault or (strict is True and weights_source == 'random_placeholder'):
        with pytest.raises(RuntimeError):
            load_bound_feature_cache(tmp_path, stem, 'dinov2', expected, expected_labels, **policy)
    else:
        features, actual_labels, actual_ids, _ = load_bound_feature_cache(
            tmp_path, stem, 'dinov2', expected, expected_labels, **policy)
        assert features.shape == (2, 3)
        assert actual_labels.tolist() == [0, 1]
        assert actual_ids == ['a', 'b']


@pytest.mark.parametrize('model,weights', [
    ('dinov2', 'pretrained'), ('dofa', 'pretrained'), ('croma', 'pretrained'),
    ('random_init', 'random_init'),
])
def test_encoder_cache_requires_its_own_weight_source(tmp_path, model, weights):
    labels = np.array([0, 1])
    args = (tmp_path, 'titan_train_feats', model, {}, {'a': 0, 'b': 1})
    for source in [weights, 'random_init' if weights == 'pretrained' else 'pretrained', 'placeholder']:
        write_cache(tmp_path, model, source, np.eye(2), labels, ['a', 'b'])
        if source == weights:
            load_bound_feature_cache(*args)
        else:
            with pytest.raises(RuntimeError, match="weights_source"):
                load_bound_feature_cache(*args)
        load_bound_feature_cache(*args, strict=False)


@pytest.mark.parametrize('dtype', [np.float32, np.bool_])
@pytest.mark.parametrize('strict', [False, True])
def test_cache_rejects_noninteger_labels_even_when_values_match(tmp_path, dtype, strict):
    stem = write_cache(tmp_path, 'dinov2', 'pretrained', np.eye(2),
                       np.array([0, 1], dtype=dtype), ['a', 'b'])
    with pytest.raises(RuntimeError, match="integer"):
        load_bound_feature_cache(tmp_path, stem, 'dinov2', {}, {'a': 0, 'b': 1}, strict=strict)
