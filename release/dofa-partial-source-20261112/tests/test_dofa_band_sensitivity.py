"""Synthetic admission/lifecycle tests. Never import Torch or load a model."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_dofa_band_sensitivity as pilot


def catalog_fixture():
    # Six class anchors occupy exactly the five historical-shaped batches.
    labels = {0: 0, 2641: 1, 173: 2, 2: 3, 69: 4, 2936: 5}
    train = [dict(tile_id=f'train_{i:06d}', label=labels.get(i, 0)) for i in range(3008)]
    val = [dict(tile_id=f'val_{i:06d}', label=i) for i in range(6)]
    catalog = {'tiles': train + val}
    split = {'assignments': {e['tile_id']: r for r, entries in [('train', train), ('val', val)] for e in entries}}
    return catalog, split


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Authenticate small synthetic files via the same preparation logic, no NPY/model reads."""
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    catalog, split = catalog_fixture()
    roles, chosen = pilot.selection(catalog, split)
    train_ids = [e['tile_id'] for e in roles['train']]
    sample = [train_ids[0]] + [i for i in train_ids if i not in chosen['ids']['train']][:199]
    norm = {'enabled': True, 'lower': 1., 'upper': 200., 'sampling': {
        'selected_tile_ids': sample, 'used_tile_ids': sample, 'missing_tile_ids': [],
        'empty_tile_ids': [], 'sample_count': 102400}}
    tiles = inputs / 'tiles'
    tiles.mkdir()
    needed = set(sample) | set(chosen['ids']['train']) | set(chosen['ids']['val'])
    for e in catalog['tiles']:
        e['tile_sha256'] = __import__('hashlib').sha256(e['tile_id'].encode()).hexdigest()
        if e['tile_id'] in needed:
            (tiles / (e['tile_id'] + '.npy')).write_bytes(e['tile_id'].encode())
    def write(name, value):
        path = inputs / name
        path.write_bytes(pilot.encode(value))
        return path
    catalog_path = write('catalog.json', catalog)
    split['catalog_sha256'] = pilot.sha(catalog_path)
    split_path = write('split.json', split)
    upstream = inputs / 'upstream'
    upstream.mkdir()
    pins = {}
    for name in pilot.UPSTREAM_FILES:
        path = upstream / name
        path.write_text('# Synthetic inert source; never imported.\n')
        pins[name] = pilot.sha(path)
    monkeypatch.setattr(pilot, 'UPSTREAM_FILES', pins)
    weights = inputs / 'weights.pth'
    weights.write_bytes(b'not a checkpoint; must never be loaded')
    monkeypatch.setattr(pilot, 'WEIGHTS_SHA', pilot.sha(weights))
    cache = inputs / 'cache'
    features = cache / 'features/dofa'
    features.mkdir(parents=True)
    for suffix in ('feats.npy', 'labels.npy', 'ids.npy'):
        (features / ('titan_train_' + suffix)).write_bytes(suffix.encode())
    meta = {'provenance': {'normalization_provenance': norm, 'band_identifier_ghz': 13.78,
        'weights_sha256': pilot.WEIGHTS_SHA, 'model_revision': pilot.UPSTREAM_COMMIT,
        'titan_catalog_sha256': pilot.sha(catalog_path), 'titan_split_manifest_sha256': pilot.sha(split_path)}}
    (features / 'titan_train_feats.meta.json').write_bytes(pilot.encode(meta))
    manifest = {'git': {'commit': pilot.HISTORICAL_COMMIT},
                'environment': {'platform': 'Linux-historical-kernel'},
                'outputs': {str(p.relative_to(cache)): pilot.sha(p) for p in features.iterdir()}}
    (cache / 'run_manifest.json').write_bytes(pilot.encode(manifest))
    protocol = write('protocol.json', {'specification': {
        'catalog': {'sha256': pilot.sha(catalog_path)},
        'caches': [{'model': 'dofa', 'fold': 0, 'manifest_sha256': pilot.sha(cache / 'run_manifest.json')}],
        'folds': [{'split': {'sha256': pilot.sha(split_path)}, 'normalization_provenance': norm}]}})
    monkeypatch.setattr(pilot, 'PROTOCOL_SHA', pilot.sha(protocol))
    historical = inputs / 'historical'
    historical.mkdir()
    source = historical / 'inert.py'
    source.write_text('# never imported\n')
    monkeypatch.setattr(pilot, 'clean_source', lambda root, commit=None: {'path': str(Path(root).resolve()), 'commit': commit or 'a' * 40})
    monkeypatch.setattr(pilot, 'historical_source', lambda root: {
        'path': str(Path(root).resolve()), 'commit': pilot.HISTORICAL_COMMIT,
        'files': {'inert.py': pilot.binding(source)}})
    args = SimpleNamespace(protocol=protocol, catalog=catalog_path, split=split_path, cache=cache,
        historical_source=historical, upstream=upstream, weights=weights, python=Path(sys.executable),
        gpu_uuid='GPU-00000000-0000-0000-0000-000000000001', gpu_lock=tmp_path / 'gpu.lock', timeout=10)
    return args, pilot.prepare(args)


def test_preparation_is_draft_and_accepts_explicit_kernel_difference(prepared):
    args, contract = prepared
    assert contract['status'] == 'draft_not_for_execution'
    assert contract['historical_runtime']['platform'] != contract['preparation_platform']
    assert len(contract['images']) == 365
    assert contract['selection']['training_batch_indices'] == [0, 2, 5, 82, 91]
    with pytest.raises(ValueError, match='not reviewed'):
        pilot.validate_contract(contract)
    contract['status'] = 'reviewed_for_pilot'
    pilot.validate_contract(contract)


@pytest.mark.parametrize('mutation', ['rows', 'norm', 'policy', 'source', 'image', 'weights', 'metadata'])
def test_contract_and_byte_mutations_are_rejected(prepared, mutation):
    args, contract = prepared
    contract['status'] = 'reviewed_for_pilot'
    if mutation == 'rows':
        contract['selection']['rows']['train'][0] = 1
    elif mutation == 'norm':
        contract['normalization']['lower'] = 2.
    elif mutation == 'policy':
        contract['policy'] = {**contract['policy'], 'comparison': 'allclose'}
    elif mutation == 'source':
        Path(contract['historical_source']['files']['inert.py']['path']).write_text('# changed\n')
    else:
        path = (next(iter(contract['images'].values()))['path'] if mutation == 'image'
                else contract['weights']['path'] if mutation == 'weights'
                else contract['cache']['feats.meta.json']['path'])
        Path(path).write_bytes(b'changed')
    with pytest.raises(ValueError):
        pilot.validate_contract(contract)


def test_missing_normalization_input_fails_before_model_work(prepared):
    args, contract = prepared
    tid = contract['normalization']['sampling']['selected_tile_ids'][1]
    assert tid not in contract['selection']['ids']['train']
    Path(contract['images'][tid]['path']).unlink()
    with pytest.raises(FileNotFoundError):
        pilot.prepare(args)


def test_selection_uses_authoritative_roles_and_original_batch_context():
    catalog, split = catalog_fixture()
    for e in catalog['tiles']:
        e['split'] = 'test'  # Stored catalog assignments must not win.
    roles, chosen = pilot.selection(catalog, split)
    assert chosen['targets']['train'] == [0, 2641, 173, 2, 69, 2936]
    assert chosen['rows']['train'] == [i for b in [0, 2, 5, 82, 91] for i in range(b * 32, b * 32 + 32)]
    assert len(chosen['ids']['train']) == 160 and len(chosen['ids']['val']) == 6
    assert not set(chosen['ids']['train']) & set(chosen['ids']['val'])
    catalog['tiles'][-1]['label'] = 0
    with pytest.raises(ValueError, match='Missing pilot class'):
        pilot.selection(catalog, split)


def test_full_dataset_normalization_required_before_subset():
    expected = {'enabled': True, 'sampling': {'selected_tile_ids': ['train1', 'train2']}}
    ds = SimpleNamespace(transform=None, include_vims=False, normalize=True,
                         entries=[{'tile_id': 'train1', 'label': 0}], normalization_provenance=copy.deepcopy(expected))
    pilot.check_dataset(ds, 'train', ds.entries, expected)
    ds.normalization_provenance['sampling']['selected_tile_ids'] = ['train1']
    with pytest.raises(ValueError, match='Normalization differs'):
        pilot.check_dataset(ds, 'train', ds.entries, expected)


def test_historical_rng_path_requires_full_membership_not_just_sample_ids(tmp_path):
    # Compile only this pure NumPy helper from the pinned Git blob. No package,
    # Torch import, model or real scientific image is involved.
    text = subprocess.check_output(['git', '-C', str(pilot.ROOT), 'show',
        pilot.HISTORICAL_COMMIT + ':titansar/data/dataset.py'], text=True)
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_sample_train_pixels')
    namespace = {'np': np, 'Path': Path, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-synthetic-normalization-test>', 'exec'), namespace)
    sample_pixels = namespace['_sample_train_pixels']
    entries = [{'tile_id': f'tile{i:03d}'} for i in range(201)]
    for i, entry in enumerate(entries):
        np.save(tmp_path / (entry['tile_id'] + '.npy'), np.arange(1, 1025, dtype=np.float32) + i * 1024)
    _, full = sample_pixels(entries, tmp_path, return_metadata=True)
    selected = [{'tile_id': tid} for tid in full['selected_tile_ids']]
    _, shortcut = sample_pixels(selected, tmp_path, return_metadata=True)
    assert full['selected_tile_ids'] == shortcut['selected_tile_ids'] == full['used_tile_ids']
    assert full['sample_count'] == shortcut['sample_count'] == 102400
    assert full['sample_float64_le_sha256'] != shortcut['sample_float64_le_sha256']


@pytest.mark.parametrize('change', ['one_bit', 'signed_zero'])
def test_bitwise_gate_rejects_differences_hidden_by_tolerance(change):
    a = np.ones((160, 768), np.float32)
    b = a.copy()
    if change == 'one_bit':
        b.view(np.uint32)[159, 767] += 1
    else:
        a[159, 767], b[159, 767] = 0., -0.
    assert np.allclose(a, b)
    result = pilot.bitwise_reference(b, a)
    assert result == {'passed': False, 'comparison': 'bitwise_float32', 'rows_checked': 160, 'mismatched_rows': [159]}


def arm_fixture():
    _, chosen = pilot.selection(*catalog_fixture())
    arrays = {r: (np.ones((len(chosen['ids'][r]), 768), np.float32),
                  np.array(chosen['labels'][r]), chosen['ids'][r]) for r in ('train', 'val')}
    return chosen, arrays


def test_both_arms_use_same_order_and_baseline_gate(tmp_path):
    chosen, arrays = arm_fixture()
    calls = []
    def extract(band, role):
        calls.append((band, role))
        return arrays[role]
    record = {'device_uuid': 'GPU-fixture'}
    pilot.run_arms(extract, lambda: {'settings': {'uuid': 'one'}}, arrays['train'][0], chosen, tmp_path, record)
    assert calls == [(13.78, 'train'), (13.78, 'val'), (3.75, 'train'), (3.75, 'val')]
    assert record['historical_training_comparison']['passed']
    assert set(record['arms']) == {'13.78', '3.75'}
    assert len(list(tmp_path.glob('*.npy'))) == 12


@pytest.mark.parametrize('failure', ['baseline', 'nonfinite', 'ids', 'labels'])
def test_failed_arm_is_saved_and_stops_before_sensitivity(tmp_path, failure):
    chosen, arrays = arm_fixture()
    reference = arrays['train'][0].copy()
    x, y, ids = arrays['train']
    if failure == 'baseline':
        x.view(np.uint32)[100, 0] += 1
    elif failure == 'nonfinite':
        x[0, 0] = np.nan
    elif failure == 'ids':
        ids = list(reversed(ids))
    else:
        y = y.astype(float)
    calls, record = [], {'device_uuid': 'GPU-fixture'}
    def extract(band, role):
        calls.append((band, role))
        return x, y, ids
    with pytest.raises(ValueError):
        pilot.run_arms(extract, lambda: {}, reference, chosen, tmp_path, record)
    assert calls == [(13.78, 'train')]
    assert (tmp_path / '13.78-train-features.npy').is_file()
    assert record['arms']['3.75']['status'] == 'not_reached'
    assert record['arms']['13.78']['roles']['train']['status'] == 'failed'
    assert record['arms']['13.78']['roles']['val']['status'] == 'not_reached'


def test_runtime_drift_is_not_hidden_by_a_successful_feature_gate(tmp_path):
    chosen, arrays = arm_fixture()
    snapshots = iter([{'settings': {'tf32': False}}, {'settings': {'tf32': True}}])
    with pytest.raises(ValueError, match='Runtime settings changed'):
        pilot.run_arms(lambda band, role: arrays[role], lambda: next(snapshots), arrays['train'][0], chosen,
                       tmp_path, {'device_uuid': 'GPU-fixture'})


def test_changed_loaded_library_rejected_but_late_loads_recorded():
    a = {'settings': {}, 'packages': {}, 'environment': {},
         'platform': 'current', 'python': '3.12.3', 'python_entrypoint': '/venv/bin/python',
         'python_binary': 'hash', 'nvidia_smi': ['GPU-current, driver-current'], 'feature_state': 'feature-state', 'loaded_files': {
        'module_paths': {'numpy': '/numpy.py'}, 'module_files': {'/numpy.py': 'hash'},
        'mapped_native_files': {'/lib.so': 'hash'}}}
    b = copy.deepcopy(a)
    b['loaded_files']['mapped_native_files']['/late.so'] = 'new'
    pilot.unchanged_loaded_runtime(a, b)
    b['loaded_files']['mapped_native_files']['/lib.so'] = 'changed'
    with pytest.raises(ValueError, match='Loaded dependency changed'):
        pilot.unchanged_loaded_runtime(a, b)


def test_actual_process_device_cannot_inherit_another_gpu_admission():
    rows = '12, GPU-first\n34, GPU-second\n'
    assert pilot.checked_gpu_uuid(rows, 34, 'GPU-second') == 'GPU-second'
    with pytest.raises(ValueError, match='UUID differs'):
        pilot.checked_gpu_uuid(rows, 34, 'GPU-first')
    with pytest.raises(ValueError, match='exactly one'):
        pilot.checked_gpu_uuid(rows + '34, GPU-first\n', 34, 'GPU-second')
    with pytest.raises(ValueError, match='exactly one'):
        pilot.checked_gpu_uuid(rows, 99, 'GPU-first')


def test_shadowed_import_is_rejected_without_loading_a_module(tmp_path, monkeypatch):
    wrong = tmp_path / 'wrong.py'
    wrong.write_text('# inert\n')
    monkeypatch.setitem(sys.modules, 'titansar', SimpleNamespace(__file__=str(wrong)))
    with pytest.raises(ValueError, match='shadowed'):
        pilot.imported_bindings({'historical_source': {'path': str(tmp_path / 'correct'), 'files': {}}, 'upstream': {}})


@pytest.mark.parametrize('name', ['titansar', 'dofa_v1', 'wave_dynamic_layer'])
def test_preloaded_bare_or_historical_module_denied_before_import(name, monkeypatch):
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__='/wrong/source.py'))
    with pytest.raises(ValueError, match='fresh worker'):
        pilot.load_historical({}, {'import_trace': []})


def test_bad_checkpoint_hash_rejected_before_factory_or_load(prepared):
    args, contract = prepared
    args.weights.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Bound file changed'):
        pilot.load_offline_model(contract, None, None, {})


def test_direct_feature_call_propagates_typeerror_without_unconditioned_retry():
    calls, trace = [], []
    def forward(model, images, **kwargs):
        calls.append(kwargs)
        raise TypeError('error inside pinned forward')
    with pytest.raises(TypeError, match='inside pinned forward'):
        pilot.strict_feature_call(forward, object(), SimpleNamespace(shape=(32, 1, 224, 224)), 3.75, 'train', trace)
    assert calls == [{'wave_list': [3.75]}]
    assert trace == [{'numeric_band_input': [3.75], 'role': 'train', 'batch_rows': 32, 'completed': False}]


def test_actual_forward_argument_is_recorded_for_each_arm():
    seen, trace = [], []
    def forward(model, images, wave_list):
        seen.append(wave_list)
        return 'inert-result'
    for band in (13.78, 3.75):
        assert pilot.strict_feature_call(forward, object(), SimpleNamespace(shape=(6, 1, 224, 224)),
                                         band, 'val', trace) == 'inert-result'
    assert seen == [[13.78], [3.75]]
    assert all(x['completed'] and x['batch_rows'] == 6 for x in trace)


def test_feature_state_excludes_unused_head_but_binds_final_normalization():
    class ArrayTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)
        def detach(self):
            return self
        def cpu(self):
            return self
        def contiguous(self):
            return self
        def numpy(self):
            return self.array
    # Inert array mapping with the minimal tensor interface; no model exists.
    states = {'fc_norm.weight': ArrayTensor([1, 1]), 'fc_norm.bias': ArrayTensor([0, 0]),
              'head.weight': ArrayTensor([.1, .2]), 'pos_embed': ArrayTensor([.3, .4])}
    state_holder = SimpleNamespace(state_dict=lambda: states)
    original = pilot.feature_state(state_holder)
    states['head.weight'] = ArrayTensor([2, 3])
    assert pilot.feature_state(state_holder) == original
    states['fc_norm.weight'] = ArrayTensor([1, 2])
    assert pilot.feature_state(state_holder) != original


def test_atomic_publication_rejects_nonfinite_and_existing_output(tmp_path):
    path = tmp_path / 'receipt.json'
    pilot.publish(path, {'status': 'first'})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.publish(path, {'status': 'second'})
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        pilot.publish(tmp_path / 'invalid.json', {'value': float('nan')})
    assert not (tmp_path / 'invalid.json').exists()
    assert not list(tmp_path.glob('.pending-*'))


def test_failure_terminal_contains_partial_outputs(tmp_path):
    def action(record):
        pilot.publish(tmp_path / 'partial.json', {'stage': 'before error'})
        raise RuntimeError('synthetic failure')
    assert pilot.attempt(tmp_path, 'pilot', action) == 1
    terminal = pilot.read(tmp_path / 'pilot-terminal.json')
    assert terminal['status'] == 'failed' and terminal['scientific_acceptance'] is False
    assert 'partial.json' in terminal['files']
    assert terminal['error']['message'] == 'synthetic failure'


@pytest.mark.parametrize('kind', ['exit', 'timeout', 'cancel'])
def test_supervisor_records_real_failure_and_reaps_worker(tmp_path, kind):
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    body = {'exit': 'import sys; sys.exit(7)',
            'timeout': 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)',
            'cancel': 'import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)'}[kind]
    def action(record):
        pilot.supervise([sys.executable, '-B', '-c', body], os.environ.copy(), tmp_path, lock,
                        0.2 if kind == 'timeout' else 5, record)
    assert pilot.attempt(tmp_path, 'pilot', action) == 1
    terminal = pilot.read(tmp_path / 'pilot-terminal.json')
    assert terminal['process_exit'] == 7 if kind == 'exit' else terminal['process_exit'] < 0
    if kind == 'cancel':
        assert terminal['cancelled_signals'] == [signal.SIGTERM]
    with pytest.raises(ProcessLookupError):
        os.kill(terminal['child_pid'], 0)


def test_busy_gpu_lock_prevents_worker_spawn(tmp_path):
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    record = {}
    with lock.open('r+') as held:
        pilot.fcntl.flock(held, pilot.fcntl.LOCK_EX | pilot.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            pilot.supervise([sys.executable, '-c', 'raise AssertionError("must not run")'],
                            os.environ.copy(), tmp_path, lock, 1, record)
    assert record['process_exit'] is None
    assert not (tmp_path / 'child-started.json').exists()


def test_outer_launch_receipt_precedes_child_execution(tmp_path):
    lock = tmp_path / 'gpu.lock'
    lock.touch()
    source = 'from pathlib import Path; import sys; assert Path(sys.argv[1]).is_file()'
    record = {'contract': {'sha256': 'declared'}, 'consumer': {'commit': 'consumer'}}
    assert pilot.supervise([sys.executable, '-B', '-c', source, str(tmp_path / 'launch.json')],
                           os.environ.copy(), tmp_path, lock, 5, record) == 0
    launch = pilot.read(tmp_path / 'launch.json')
    assert launch['contract']['sha256'] == 'declared' and launch['consumer']['commit'] == 'consumer'


def test_cli_draft_refusal_has_terminal_record_without_scientific_import(tmp_path):
    contract = tmp_path / 'contract.json'
    pilot.publish(contract, {'schema': 'dofa-band-pilot-v1', 'status': 'draft_not_for_execution',
        'policy': pilot.POLICY, 'historical_source': {'path': str(tmp_path / 'source')},
        'upstream': {'path': str(tmp_path / 'upstream')}, 'catalog': {'path': str(tmp_path / 'data/catalog.json')},
        'cache_manifest': {'path': str(tmp_path / 'cache/run_manifest.json')}})
    output = tmp_path / 'attempt'
    run = subprocess.run([sys.executable, '-B', str(Path(pilot.__file__)), 'pilot', '--contract', str(contract),
                          '--contract-sha256', pilot.sha(contract), '--output', str(output)], capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    terminal = pilot.read(output / 'pilot-terminal.json')
    assert 'not reviewed' in terminal['error']['message']
    assert not (output / 'child-started.json').exists()


def test_module_import_and_network_guard_in_fresh_process():
    source = """
import json,sys,socket
from scripts import run_dofa_band_sensitivity as p
assert not any(n in sys.modules for n in ['torch','timm','sklearn','numpy','titansar'])
sys.addaudithook(p.deny_network)
try:
    socket.getaddrinfo('example.com',443)
except RuntimeError as exc:
    print(str(exc))
else:
    raise AssertionError('Network was allowed')
"""
    run = subprocess.run([sys.executable, '-B', '-c', source], cwd=pilot.ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'forbids network' in run.stdout


def test_outputs_cannot_be_created_in_input_or_repository(tmp_path):
    protected = tmp_path / 'input'
    protected.mkdir()
    with pytest.raises(ValueError, match='overlaps'):
        pilot.fresh_output(protected / 'attempt', [protected])
    with pytest.raises(ValueError, match='Git worktrees'):
        pilot.fresh_output(pilot.ROOT / 'attempt-that-must-not-exist', [])
