"""Repaired fine tuning must bind the frozen reference before loading a model."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_finetuning as script
from titansar.reproducibility import artifact_hashes, sha256_file


def test_repaired_options_are_all_required_before_outputs(monkeypatch, tmp_path):
    import sys
    monkeypatch.setattr(sys, 'argv', ['run_finetuning', '--titan_catalog', 'catalog',
        '--titan_split_manifest', 'split', '--output_dir', str(tmp_path / 'out'),
        '--repaired_protocol', 'protocol'])
    monkeypatch.setattr(script, 'start_run_manifest', lambda *a, **k: pytest.fail('Created output'))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2


def test_repaired_cli_options_exist(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['run_finetuning', '--titan_catalog', 'catalog',
        '--titan_split_manifest', 'split', '--output_dir', 'out',
        '--repaired_protocol', 'protocol', '--frozen_reference', 'reference',
        '--frozen_features_dir', 'features', '--models', 'dofa', '--seeds', '2'])
    args = script.parse_args()
    assert args.repaired_protocol == 'protocol' and args.frozen_reference == 'reference'


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


@pytest.fixture
def reference(tmp_path, monkeypatch):
    catalog_path, split_path = tmp_path / 'catalog.json', tmp_path / 'split.json'
    tiles = tmp_path / 'tiles'
    tiles.mkdir()
    entries = []
    for role in ('train', 'val', 'test'):
        for label in range(6):
            tile_id = f'{role}_{label}'
            path = tiles / f'{tile_id}.npy'
            np.save(path, np.arange(1, 16385, dtype=np.float32).reshape(128, 128) + label)
            entries.append(dict(tile_id=tile_id, label=label, split=role, center_lat=0., center_lon=0.,
                                tile_sha256=sha256_file(path)))
    write_json(catalog_path, dict(tiles=entries, intensity_space='hisar_log_dn', benchmark_track='fixture-track'))
    write_json(split_path, dict(catalog_sha256=sha256_file(catalog_path), assignments={e['tile_id']: e['split'] for e in entries}))
    dataset = script.TitanSARDataset(str(catalog_path), split='train', split_manifest_path=str(split_path))
    normalization = dataset.normalization_provenance
    identity = dict(model_revision='fixture-revision', weights_sha256='a' * 64)
    source = dict(titan_catalog_sha256=sha256_file(catalog_path), earth_catalog_sha256=None,
                  titan_split_manifest_sha256=sha256_file(split_path), earth_split_manifest_sha256=None,
                  evaluation_scope='within_titan_only', benchmark_track='fixture-track',
                  normalization_provenance=normalization, **identity)
    feature_root = tmp_path / 'producer'
    features_dir = feature_root / 'features'
    model_dir = features_dir / 'dofa'
    model_dir.mkdir(parents=True)
    for role in ('train', 'test'):
        stem = f'titan_{role}'
        ids = np.array([e['tile_id'] for e in entries if e['split'] == role])
        np.save(model_dir / f'{stem}_feats.npy', np.ones((6, 768), dtype=np.float32))
        np.save(model_dir / f'{stem}_ids.npy', ids)
        np.save(model_dir / f'{stem}_labels.npy', np.arange(6))
        provenance = {**source, 'band_identifier_ghz': 13.78}
        for suffix, key in [('feats', 'feature_sha256'), ('ids', 'tile_ids_sha256'), ('labels', 'labels_sha256')]:
            provenance[key] = sha256_file(model_dir / f'{stem}_{suffix}.npy')
        write_json(model_dir / f'{stem}_feats.meta.json', dict(model='dofa', array=f'{stem}_feats',
                   weights_source='pretrained', n_features=6, feature_dim=768, provenance=provenance))
    analysis_root = tmp_path / 'analysis'
    analysis_root.mkdir()
    head = dict(tile_ids=[f'test_{i}' for i in range(6)], true_labels=list(range(6)),
                predictions=list(range(6)), probabilities=np.eye(6).tolist(), confusion_matrix=np.eye(6, dtype=int).tolist(),
                confusion_matrix_normalized=np.eye(6).tolist(), macro_accuracy=1., per_class_accuracy=[1.] * 6, kappa=1.)
    payload = dict(dofa=dict(weights_source='pretrained', source_metadata=source, evaluation_scope='within_titan_only',
                            linear_probe={'titan_to_titan': {'seed_results': [{**head, 'seed': s} for s in range(5)]}}))
    result_path = analysis_root / 'probing_results.json'
    write_json(result_path, payload)
    counts = {r: dict(total=6, class_counts=[1] * 6) for r in ('train', 'val', 'test')}
    spec = dict(recipe=copy.deepcopy(script.REPAIRED_RECIPE), scientific_source_sha256={},
                catalog_sha256=source['titan_catalog_sha256'], benchmark_track='fixture-track',
                frozen_source_commit='b' * 40, encoders={'dofa': identity},
                folds=[dict(fold=0, split_sha256=source['titan_split_manifest_sha256'], counts=counts,
                            normalization_sha256=script._json_digest(normalization))], frozen_jobs=[])
    protocol_path = tmp_path / 'protocol.json'

    def seal():
        inputs = {k: dict(path=str(p), sha256=sha256_file(p)) for k, p in
                  [('titan_catalog', catalog_path), ('titan_split_manifest', split_path)]}
        git = dict(commit=spec['frozen_source_commit'], clean=True, status=[])
        producer = dict(status='complete', git=git, inputs=inputs,
                        outputs=artifact_hashes(feature_root, exclude=['run_manifest.json']))
        write_json(feature_root / 'run_manifest.json', producer)
        consumer = dict(status='complete', git=git, inputs={**inputs, 'cached_features': dict(
            path=str(features_dir), files={k.removeprefix('features/'): v for k,v in producer['outputs'].items()})},
            outputs=artifact_hashes(analysis_root, exclude=['run_manifest.json']))
        write_json(analysis_root / 'run_manifest.json', consumer)
        spec['frozen_jobs'] = [dict(model='dofa', fold=0, result_sha256=sha256_file(result_path),
            analysis_manifest_sha256=sha256_file(analysis_root / 'run_manifest.json'),
            extraction_manifest_sha256=sha256_file(feature_root / 'run_manifest.json'))]
        write_json(protocol_path, dict(specification=spec, specification_sha256=script._json_digest(spec)))
    seal()
    args = SimpleNamespace(titan_catalog=str(catalog_path), titan_split_manifest=str(split_path),
        output_dir=str(tmp_path / 'out'), repaired_protocol=str(protocol_path), frozen_reference=str(result_path),
        frozen_features_dir=str(features_dir), models=['dofa'], seeds=[2], unfreeze_blocks=[2], batch_size=8,
        grad_accum_steps=4, epochs=50, lr=1e-4, num_workers=0, device='cuda:0',
        allow_nondeterministic=False, disable_amp=False, require_real_weights=False)
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: pytest.fail('Loaded GPU before validation'))
    return SimpleNamespace(args=args, spec=spec, payload=payload, seal=seal, model_dir=model_dir,
                           result_path=result_path, normalization=normalization, head=head)


def test_preflight_is_read_only_and_selects_matching_seed(reference, tmp_path):
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    context = script.prepare_repaired_run(reference.args)
    assert context['frozen_head']['seed'] == 2
    assert context['provenance']['normalization_provenance'] == reference.normalization
    assert context['provenance']['frozen_reference']['validated_head_seeds'] == list(range(5))
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not Path(reference.args.output_dir).exists()


@pytest.mark.parametrize('problem', ['hash', 'ids', 'labels', 'duplicate_seed', 'probabilities', 'metadata', 'cache_ids',
                                     'missing_tile', 'empty_tile', 'changed_tile', 'normalization', 'track', 'budget', 'source'])
def test_bad_reference_fails_before_model_load(reference, problem):
    r = reference
    if problem == 'hash':
        r.result_path.write_text('{}')
    elif problem in ('ids', 'labels', 'duplicate_seed', 'probabilities', 'metadata'):
        head = r.payload['dofa']['linear_probe']['titan_to_titan']['seed_results'][0]
        if problem == 'ids': head['tile_ids'] = list(reversed(head['tile_ids']))
        if problem == 'labels': head['true_labels'] = [0] * 6
        if problem == 'duplicate_seed': head['seed'] = 1
        if problem == 'probabilities': head['probabilities'] = [[0.] * 6] * 6
        if problem == 'metadata': r.payload['dofa']['source_metadata']['weights_sha256'] = 'wrong'
        write_json(r.result_path, r.payload)
        r.seal()
    elif problem == 'cache_ids':
        p = r.model_dir / 'titan_test_ids.npy'
        np.save(p, np.array([f'test_{i}' for i in reversed(range(6))]))
        meta_path = r.model_dir / 'titan_test_feats.meta.json'
        meta = json.loads(meta_path.read_text())
        meta['provenance']['tile_ids_sha256'] = sha256_file(p)
        write_json(meta_path, meta)
        r.seal()
    elif problem in ('missing_tile', 'empty_tile', 'changed_tile'):
        p = Path(r.args.titan_catalog).parent / 'tiles' / 'train_0.npy'
        if problem == 'missing_tile': p.unlink()
        else: np.save(p, np.full((128,128), 0. if problem == 'empty_tile' else 3., dtype=np.float32))
    elif problem == 'normalization':
        r.spec['folds'][0]['normalization_sha256'] = 'wrong'
        r.seal()
    elif problem == 'track':
        r.spec['benchmark_track'] = 'wrong'
        r.seal()
    elif problem == 'budget': r.args.epochs = 49
    elif problem == 'source':
        r.spec['scientific_source_sha256']['scripts/run_finetuning.py'] = 'wrong'
        r.seal()
    with pytest.raises((ValueError, RuntimeError)):
        script.run_model(r.args, 'dofa')
    assert not Path(r.args.output_dir).exists()


def tiny_model():
    backbone = torch.nn.Module()
    backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    return SimpleNamespace(model=backbone, embedding_dim=768, get_transformer_blocks=lambda: list(backbone.blocks),
                           weights_source='pretrained', model_revision='fixture-revision', weights_sha256='a' * 64,
                           wavelength=13.78)


def good_history():
    return dict(train_loss=[1.] * 50, val_balanced_accuracy=[.5] * 2 + [1.] * 48,
                optimizer_steps=[1] * 50, amp_skipped_steps=[0] * 50, loss_scale=[1024.] * 50)


def test_repaired_run_reuses_frozen_head_and_binds_checkpoint(reference, monkeypatch):
    model = tiny_model()
    calls = []
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: (calls.append(k) or model))
    monkeypatch.setattr(script, 'extract_features', lambda *a, **k: pytest.fail('Repeated frozen extraction'))
    monkeypatch.setattr(script, 'train_linear_probe', lambda *a, **k: pytest.fail('Repeated frozen fitting'))
    def train(**kwargs):
        assert kwargs['unfreeze_blocks'] == 2 and kwargs['epochs'] == 50 and kwargs['grad_accum_steps'] == 4
        assert kwargs['lr'] == 1e-4 and kwargs['use_amp'] is True
        return kwargs['model'], kwargs['init_probe'], 1., good_history()
    monkeypatch.setattr(script, 'train_finetuned_model', train)
    monkeypatch.setattr(script, 'evaluate_finetuned', lambda *a, **k: {**reference.head, 'labels': list(range(6))})
    result = script.run_model(reference.args, 'dofa')
    assert calls == [dict(device='cuda:0', allow_placeholder=False)]
    assert result['configurations']['frozen']['runs'][0]['seed'] == 2
    run = result['configurations']['ft_2block']['runs'][0]
    provenance = run['provenance']
    assert provenance['selected_epoch'] == 3 and provenance['optimizer_steps_total'] == 50
    assert provenance['unfrozen_blocks'] == ['blocks.1', 'blocks.2']
    assert provenance['trainable_backbone_parameters'] == 40
    assert provenance['trainable_head_parameters'] == 4614
    checkpoint = torch.load(Path(reference.args.output_dir) / 'checkpoints/dofa/ft_2block_seed_2.pt', weights_only=True)
    assert checkpoint['provenance'] == provenance
    assert run['metrics']['macro']['f1'] == 1.
    roundtrip = json.loads(json.dumps(script.sanitize_json({'dofa': result}), allow_nan=False))['dofa']
    saved = roundtrip['configurations']['ft_2block']['runs'][0]['provenance']
    assert saved == checkpoint['provenance']
    for p in (roundtrip['provenance'], saved):
        norm = p['normalization_provenance']
        assert norm['enabled'] is True and norm['fallback_used'] is False
        assert script._json_digest(norm) == reference.spec['folds'][0]['normalization_sha256']
        assert p['recipe']['amp'] is True and p['recipe']['deterministic'] is True
    assert roundtrip['deterministic'] is True


def test_head_seed_is_independent_of_model_loading_rng():
    first, a = script.repaired_training_metadata(tiny_model(), 2)
    torch.rand(100)
    second, b = script.repaired_training_metadata(tiny_model(), 2)
    assert a['initial_head_state_sha256'] == b['initial_head_state_sha256']
    assert torch.equal(first.fc.weight, second.fc.weight)


@pytest.mark.parametrize('problem', ['nonfinite', 'missing_epoch', 'zero_updates', 'budget', 'selection'])
def test_invalid_training_history_cannot_be_saved_as_complete(problem):
    history = good_history()
    best = 1.
    if problem == 'nonfinite': history['train_loss'][0] = float('nan')
    if problem == 'missing_epoch': history['optimizer_steps'].pop()
    if problem == 'zero_updates': history['optimizer_steps'][0] = 0
    if problem == 'budget': history['optimizer_steps'][0] = 2
    if problem == 'selection': best = .5
    with pytest.raises(ValueError): script.repaired_history_metadata(history, 6, best)


@pytest.mark.parametrize('change', ['weights_sha256', 'model_revision', 'wavelength'])
def test_loaded_encoder_must_match_reference(reference, monkeypatch, change):
    model = tiny_model()
    setattr(model, change, 5.405 if change == 'wavelength' else 'wrong')
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: model)
    monkeypatch.setattr(script, 'train_finetuned_model', lambda **kw: pytest.fail('Trained mismatched encoder'))
    with pytest.raises(ValueError, match='differs'):
        script.run_model(reference.args, 'dofa')


def test_legacy_still_extracts_and_fits_its_own_reference(reference, monkeypatch):
    args = reference.args
    args.repaired_protocol = args.frozen_reference = args.frozen_features_dir = None
    args.unfreeze_blocks = []
    calls = []
    monkeypatch.setattr(script, 'get_model', lambda *a, **k: tiny_model())
    def extract(model, loader, device):
        calls.append(loader.dataset.split)
        return np.eye(6, dtype=np.float32), np.arange(6), [e['tile_id'] for e in loader.dataset.entries]
    monkeypatch.setattr(script, 'extract_features', extract)
    monkeypatch.setattr(script, 'train_linear_probe', lambda *a, **k: calls.append(k['seed']))
    monkeypatch.setattr(script, 'evaluate_linear_probe', lambda *a, **k: reference.head)
    result = script.run_model(args, 'dofa')
    assert calls == ['train', 'test', 2]
    assert set(result['configurations']) == {'frozen'}
    assert 'provenance' not in result


def test_json_preserves_python_and_numpy_booleans():
    record = {'enabled': True, 'fallback_used': False, 'numpy_true': np.bool_(True),
              'numpy_false': np.bool_(False), 'array': np.array([True, False]), 'count': np.int64(3)}
    saved = json.loads(json.dumps(script.sanitize_json(record)))
    assert saved['enabled'] is saved['numpy_true'] is True
    assert saved['fallback_used'] is saved['numpy_false'] is False
    assert saved['array'][0] is True and saved['array'][1] is False
    assert type(saved['count']) is int


def test_preflight_failure_precedes_main_output_creation(reference, monkeypatch):
    reference.result_path.write_text('{}')
    monkeypatch.setattr(script, 'parse_args', lambda: reference.args)
    monkeypatch.setattr(script, 'start_run_manifest', lambda *a, **k: pytest.fail('Created output for invalid reference'))
    with pytest.raises(ValueError, match='hash differs'):
        script.main()
