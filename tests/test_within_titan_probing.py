"""Titan-only extraction and cached analysis must never depend on Earth data."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_probing as script


@pytest.mark.parametrize('earth_flags', [[], ['--earth_catalog', 'earth.json'],
                                        ['--earth_split_manifest', 'earth-split.json']])
def test_default_requires_both_earth_inputs_before_output(tmp_path, monkeypatch, earth_flags):
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['run_probing', '--titan_catalog', 'titan.json',
                        '--titan_split_manifest', 'split.json', '--output_dir', str(output), *earth_flags])
    monkeypatch.setattr(script, 'start_run_manifest', lambda *a, **k: pytest.fail('Created output before CLI validation'))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2 and not output.exists()


def test_new_cli_requires_only_titan_inputs(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_probing', '--within_titan_only',
                        '--titan_catalog', 'titan.json', '--titan_split_manifest', 'split.json'])
    args = script.parse_args()
    assert args.within_titan_only and args.earth_catalog is None and args.earth_split_manifest is None


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'benchmark_track': 'corrected-titan-geographic-fixture'}))
    split_path = tmp_path / 'split.json'
    split_path.write_text('{}')
    normalization = {'enabled': True, 'lower': 3., 'upper': 200.,
                     'training_ids_sha256': 'fixture', 'sampling': {
                         'seed': 0, 'missing_tile_ids': [], 'empty_tile_ids': [], 'sample_count': 100}}
    calls = {'extract': [], 'train': []}

    class Dataset:
        def __init__(self, catalog, split, **kwargs):
            self.split = split
            self.benchmark_track = json.loads(Path(catalog).read_text())['benchmark_track']
            self.normalization_provenance = normalization
            labels = list(range(6)) if split == 'train' else ([0, 2, 5] if split == 'test' else [4])
            self.entries = [{'tile_id': f'{split}_{i}', 'label': label} for i, label in enumerate(labels)]

        def __len__(self):
            return len(self.entries)

        def get_class_weights(self):
            return torch.ones(6)

    model = SimpleNamespace(weights_source='pretrained', model_revision='fixture-revision',
                            weights_sha256='a' * 64, wavelength=13.78)

    def extract(wrapper, dataset, device):
        assert wrapper.wavelength == 13.78
        calls['extract'].append(dataset.split)
        labels = np.array([r['label'] for r in dataset.entries])
        return np.eye(6, dtype=np.float32)[labels], labels, [r['tile_id'] for r in dataset.entries]

    def train(features, labels, *, seed, num_classes, class_weights, config, device):
        calls['train'].append((seed, labels.tolist(), class_weights.tolist()))
        assert num_classes == 6 and labels.tolist() == list(range(6))
        return torch.nn.Identity()

    monkeypatch.setattr(script, 'TitanSARDataset', Dataset)
    monkeypatch.setattr(script, 'EarthAnalogDataset', lambda *a, **k: pytest.fail('Constructed Earth dataset'))
    monkeypatch.setattr(script, 'DataLoader', lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: model)
    monkeypatch.setattr(script, 'extract_features', extract)
    monkeypatch.setattr(script, 'train_linear_probe', train)
    monkeypatch.setattr(script, 'compute_silhouette_score', lambda *a, **k: .25)
    for name in ('compute_mmd_repeated', 'compute_mmd_permutation_test', 'compute_group_resampling_mmd',
                 'compute_proxy_a_distance', 'compute_centroid_distances'):
        monkeypatch.setattr(script, name, lambda *a, **k: pytest.fail('Computed cross-domain metric'))
    kwargs = dict(model_name='dofa', device='cpu', titan_catalog=str(catalog), earth_catalog=None,
                  titan_split_manifest=str(split_path), earth_split_manifest=None,
                  output_dir=str(tmp_path / 'output'), batch_size=2, seeds=[0, 1],
                  num_workers=0, within_titan_only=True, allow_placeholder=False)
    return kwargs, calls, normalization


def test_fresh_mode_keeps_titan_predictions_and_omits_earth(pipeline):
    kwargs, calls, normalization = pipeline
    result = script.run_single_model(**kwargs)
    assert calls['extract'] == ['train', 'test', 'selk_holdout']
    assert len(calls['train']) == 2
    assert result['evaluation_scope'] == 'within_titan_only'
    assert set(result['linear_probe']) == set(result['knn']) == {'titan_to_titan'}
    assert 'domain_gap' not in result
    assert result['source_metadata']['normalization_provenance'] == normalization
    assert result['source_metadata']['benchmark_track'] == 'corrected-titan-geographic-fixture'
    assert result['source_metadata']['earth_catalog_sha256'] is None
    assert result['source_metadata']['earth_split_manifest_sha256'] is None
    for prediction in [*result['linear_probe']['titan_to_titan']['seed_results'], result['knn']['titan_to_titan']]:
        assert prediction['tile_ids'] == ['test_0', 'test_1', 'test_2']
        assert prediction['true_labels'] == [0, 2, 5]
        assert len(prediction['predictions']) == 3
        assert not any(k.startswith('source_supported') for k in prediction)
    assert result['linear_probe']['titan_to_titan']['macro_accuracy_mean'] == 1.


def test_features_only_and_cache_reuse_preserve_row_identity(pipeline, monkeypatch):
    kwargs, calls, normalization = pipeline
    result = script.run_single_model(**kwargs, features_only=True)
    assert result['feature_extraction_only'] is True and not calls['train']
    assert result['evaluation_scope'] == 'within_titan_only'
    directory = Path(kwargs['output_dir']) / 'features' / 'dofa'
    assert not list(directory.glob('earth*'))
    for meta_path in directory.glob('*.meta.json'):
        meta = json.loads(meta_path.read_text())['provenance']
        assert meta['evaluation_scope'] == 'within_titan_only'
        assert meta['normalization_provenance'] == normalization
        assert meta['earth_catalog_sha256'] is None and meta['earth_split_manifest_sha256'] is None
        assert meta['benchmark_track'] == 'corrected-titan-geographic-fixture'
        assert meta['model_revision'] == 'fixture-revision' and meta['weights_sha256'] == 'a' * 64
        assert meta['band_identifier_ghz'] == 13.78
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: pytest.fail('Loaded encoder for cached analysis'))
    monkeypatch.setattr(script, 'extract_features', lambda *a, **k: pytest.fail('Extracted cached features'))
    cached = script.run_single_model(**{**kwargs, 'cached_features_dir': str(directory.parent),
                                       'output_dir': str(directory.parent.parent / 'analysis')})
    assert cached['linear_probe']['titan_to_titan']['macro_accuracy_mean'] == 1.
    assert cached['linear_probe']['titan_to_titan']['seed_results'][0]['tile_ids'] == ['test_0', 'test_1', 'test_2']
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('changed', ['normalization_provenance', 'benchmark_track', 'evaluation_scope',
                                   'earth_catalog_sha256', 'weights_sha256', 'titan_split_manifest_sha256'])
def test_cache_metadata_mismatch_rejected(pipeline, changed):
    kwargs, _, _ = pipeline
    script.run_single_model(**kwargs, features_only=True)
    directory = Path(kwargs['output_dir']) / 'features' / 'dofa'
    path = directory / 'titan_test_feats.meta.json'
    meta = json.loads(path.read_text())
    meta['provenance'][changed] = 'wrong'
    path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        script.run_single_model(**kwargs, cached_features_dir=str(directory.parent))


def test_default_worker_requires_earth_before_datasets(pipeline, monkeypatch):
    kwargs, _, _ = pipeline
    monkeypatch.setattr(script, 'TitanSARDataset', lambda *a, **k: pytest.fail('Loaded data before validation'))
    with pytest.raises(ValueError, match='Earth'):
        script.run_single_model(**{**kwargs, 'within_titan_only': False})


@pytest.mark.parametrize('problem', ['missing_tile_ids', 'empty_tile_ids', 'sample_count'])
def test_missing_normalization_samples_rejected_before_extraction(pipeline, problem):
    kwargs, calls, normalization = pipeline
    normalization['sampling'][problem] = 0 if problem == 'sample_count' else ['train_0']
    with pytest.raises(ValueError, match='normalization requires'):
        script.run_single_model(**kwargs, features_only=True)
    assert calls['extract'] == []


@pytest.mark.parametrize('change', ['unchanged', 'changed_pixels', 'missing_tile', 'empty_tile'])
def test_real_dataset_normalization_is_recomputed_for_cached_analysis(pipeline, monkeypatch, change):
    import hashlib
    from titansar.data.dataset import TitanSARDataset

    kwargs, _, _ = pipeline
    catalog, split = Path(kwargs['titan_catalog']), Path(kwargs['titan_split_manifest'])
    tiles = catalog.parent / 'tiles'
    tiles.mkdir()
    entries = [{'tile_id': f'{role}_{i}', 'label': label, 'split': role}
               for role, labels in [('train', range(6)), ('test', [0, 2, 5])]
               for i, label in enumerate(labels)]
    raw = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
    for entry in entries:
        np.save(tiles / f"{entry['tile_id']}.npy", raw)
    catalog.write_text(json.dumps(dict(tiles=entries, intensity_space='hisar_log_dn',
                                       benchmark_track='corrected-titan-geographic-fixture')))
    split.write_text(json.dumps(dict(catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
                                    assignments={e['tile_id']: e['split'] for e in entries})))
    monkeypatch.setattr(script, 'TitanSARDataset', TitanSARDataset)
    extracted = script.run_single_model(**kwargs, features_only=True)
    normalization = extracted['source_metadata']['normalization_provenance']
    assert normalization['sampling']['selected_tile_ids'] == [f'train_{i}' for i in range(6)]
    assert [normalization['lower'], normalization['upper']] == pytest.approx([1.99, 99.01], abs=1e-12)
    changed_path = tiles / 'train_0.npy'
    if change == 'changed_pixels':
        np.save(changed_path, raw + 10)
    elif change == 'missing_tile':
        changed_path.unlink()
    elif change == 'empty_tile':
        np.save(changed_path, np.zeros_like(raw))
    cache = str(Path(kwargs['output_dir']) / 'features')
    if change == 'unchanged':
        result = script.run_single_model(**kwargs, cached_features_dir=cache)
        assert result['source_metadata'] == extracted['source_metadata']
    elif change == 'changed_pixels':
        with pytest.raises(RuntimeError, match='normalization_provenance'):
            script.run_single_model(**kwargs, cached_features_dir=cache)
    else:
        with pytest.raises(ValueError, match='normalization requires'):
            script.run_single_model(**kwargs, cached_features_dir=cache)


@pytest.mark.parametrize('empty_split', ['train', 'test'])
def test_empty_titan_partition_rejected(pipeline, monkeypatch, empty_split):
    kwargs, calls, _ = pipeline
    make_dataset = script.TitanSARDataset

    def dataset(*args, **kwargs):
        value = make_dataset(*args, **kwargs)
        if value.split == empty_split:
            value.entries = []
        return value

    monkeypatch.setattr(script, 'TitanSARDataset', dataset)
    with pytest.raises(ValueError, match='must be nonempty'):
        script.run_single_model(**kwargs)
    assert calls['extract'] == []


def test_main_passes_mode_and_omits_absent_earth_manifest_inputs(pipeline, monkeypatch):
    kwargs, _, _ = pipeline
    output = Path(kwargs['output_dir'])
    args = SimpleNamespace(**{k: v for k, v in kwargs.items() if k not in ('model_name', 'num_workers', 'allow_placeholder')},
                           models=['dofa'], features_only=False, cached_features_dir=None,
                           parallel=False, require_real_weights=True)
    recorded = {}

    def start(path, command, inputs):
        assert 'earth_catalog' not in inputs and 'earth_split_manifest' not in inputs
        recorded.update(inputs)
        path.mkdir()
        return path / 'run_manifest.json'

    monkeypatch.setattr(script, 'parse_args', lambda: args)
    monkeypatch.setattr(script, 'start_run_manifest', start)
    monkeypatch.setattr(script, 'complete_run_manifest', lambda path: None)
    script.main()
    result = json.loads((output / 'probing_results.json').read_text())['dofa']
    assert result['evaluation_scope'] == 'within_titan_only'
    assert result['source_metadata']['earth_catalog_sha256'] is None
    assert result['source_metadata']['normalization_provenance']['enabled'] is True
    assert recorded['titan_catalog'] == kwargs['titan_catalog']
