"""Prepare or supervise one reviewed DOFA band pilot. No fitting/expansion CLI.

Imports are deliberately stdlib-only until the admitted scientific worker.
Historical source, this consumer, and observed current runtime are distinct.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RELATIVE_SOURCE = 'scripts/run_dofa_band_sensitivity.py'
HISTORICAL_COMMIT = '6295af2bc35c8d8a2e47fc982b196809245f4550'
PROTOCOL_SHA = '48515a7583e99c0cfd9fb179600cdfeda7dab484f424b4bcbdb912570632f9ca'
UPSTREAM_COMMIT = '73ff5c0721da322689ae890bed5b4efd78935f47'
WEIGHTS_SHA = '4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1'
UPSTREAM_FILES = {
    'hubconf.py': '8a7d625cf3d5babc3b321de95305726eeff09896491c19bbecbdd6a4a2fb954a',
    'dofa_v1.py': 'ed99d82a4339f2c71446720094f07b5e21920e427729d369beb1bd0e5f2a2b37',
    'wave_dynamic_layer.py': '828e72df0c935a5403e5bb90349814355fd659a6219b63380f2d2d61baae4bf9',
}
PACKAGES = {'numpy': '2.4.4', 'torch': '2.12.1+cu126',
            'torchvision': '0.27.1+cu126', 'timm': '1.0.27',
            'huggingface-hub': '1.19.0', 'scipy': '1.17.1',
            'scikit-learn': '1.9.0', 'rasterio': '1.5.0',
            'scikit-image': '0.26.0', 'matplotlib': '3.11.0'}
THREAD_ENV = {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
              'OPENBLAS_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
POLICY = {
    'scope': 'fold0_pilot_only', 'bands': [13.78, 3.75], 'batch_size': 32,
    'num_workers': 4, 'train_rows': 160, 'validation_rows': 6,
    'feature_dim': 768, 'dtype': 'float32', 'comparison': 'bitwise',
    'runtime_scope': 'declared_current_runtime', 'historical_environment_identical': False,
    'kernel_equality_required': False, 'historical_package_versions': PACKAGES,
    'thread_environment': THREAD_ENV, 'historical_validation_reference': False,
    'fits': 0, 'full_study_implemented': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def publish(path, value):
    """Atomic, finite, durable JSON with no replacement of prior evidence."""
    publish_bytes(path, encode(value))


def publish_bytes(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)  # EEXIST preserves every previous attempt.
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def binding(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    require(expected is None or digest == expected, 'Input hash differs: ' + str(path))
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def check_binding(record):
    require(binding(record['path']) == record, 'Bound file changed: ' + record['path'])


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args],
                                   text=True).strip()


def clean_source(root, commit=None):
    root = Path(root).resolve(strict=True)
    head = git(root, 'rev-parse', 'HEAD')
    require(commit is None or head == commit, 'Source commit differs')
    require(not git(root, 'status', '--porcelain=v1'), 'Source checkout is not clean')
    return {'path': str(root), 'commit': head}


def historical_source(root):
    source = clean_source(root, HISTORICAL_COMMIT)
    names = git(root, 'ls-tree', '-r', '--name-only', HISTORICAL_COMMIT, 'titansar').splitlines()
    files = {}
    for name in names:
        if name.endswith('.py'):
            raw = subprocess.check_output(['git', '-C', str(root), 'show', HISTORICAL_COMMIT + ':' + name])
            files[name] = binding(Path(root) / name, hashlib.sha256(raw).hexdigest())
    require(files, 'Empty historical source inventory')
    return {**source, 'files': files}


def selection(catalog, split):
    entries = catalog['tiles']
    ids = [e['tile_id'] for e in entries]
    require(len(set(ids)) == len(ids) and set(split['assignments']) == set(ids), 'Catalog/split coverage differs')
    require(all(isinstance(i, str) and Path(i).name == i and i not in ('', '.', '..') for i in ids), 'Unsafe tile ID')
    require(all(type(e['label']) is int and 0 <= e['label'] < 6 for e in entries), 'Invalid class label')
    roles = {r: [e for e in entries if split['assignments'][e['tile_id']] == r] for r in ('train', 'val')}
    targets, rows = {}, {}
    for role, es in roles.items():
        require(set(e['label'] for e in es) == set(range(6)), 'Missing pilot class')
        targets[role] = [min((i for i,e in enumerate(es) if e['label'] == c),
                             key=lambda i: es[i]['tile_id']) for c in range(6)]
    blocks = sorted({i // 32 for i in targets['train']})
    rows['train'] = [i for b in blocks for i in range(b * 32, min(b * 32 + 32, len(roles['train'])))]
    rows['val'] = sorted(targets['val'])
    require(len(rows['train']) == 160 and len(blocks) == 5, 'Original five full training batches required')
    return roles, {'targets': targets, 'rows': rows, 'training_batch_indices': blocks,
                   'ids': {r: [roles[r][i]['tile_id'] for i in rows[r]] for r in rows},
                   'labels': {r: [roles[r][i]['label'] for i in rows[r]] for r in rows}}


def prepare(args):
    """Hash/catalog preparation only. No imports from scientific packages."""
    consumer = {**clean_source(ROOT), 'script': binding(__file__)}
    protocol = binding(args.protocol, PROTOCOL_SHA)
    spec = read(args.protocol)['specification']
    cache_spec = next(c for c in spec['caches'] if c['model'] == 'dofa' and c['fold'] == 0)
    fold = spec['folds'][0]
    catalog_b = binding(args.catalog, spec['catalog']['sha256'])
    split_b = binding(args.split, fold['split']['sha256'])
    catalog, split = read(args.catalog), read(args.split)
    require(split['catalog_sha256'] == catalog_b['sha256'], 'Split catalog binding differs')
    roles, chosen = selection(catalog, split)
    manifest_b = binding(Path(args.cache) / 'run_manifest.json', cache_spec['manifest_sha256'])
    manifest = read(manifest_b['path'])
    require(manifest['git']['commit'] == HISTORICAL_COMMIT, 'Extraction producer differs')
    files = {}
    for suffix in ('feats.npy', 'ids.npy', 'labels.npy', 'feats.meta.json'):
        rel = 'features/dofa/titan_train_' + suffix
        files[suffix] = binding(Path(args.cache) / rel, manifest['outputs'][rel])
    metadata = read(files['feats.meta.json']['path'])
    p = metadata['provenance']
    require(p['normalization_provenance'] == fold['normalization_provenance'] and
            p['band_identifier_ghz'] == 13.78 and p['weights_sha256'] == WEIGHTS_SHA and
            p['model_revision'] == UPSTREAM_COMMIT and
            p['titan_catalog_sha256'] == catalog_b['sha256'] and
            p['titan_split_manifest_sha256'] == split_b['sha256'], 'Original cache provenance differs')
    norm = fold['normalization_provenance']
    sample = norm['sampling']
    require(len(sample['selected_tile_ids']) == len(sample['used_tile_ids']) == 200 and
            sample['selected_tile_ids'] == sample['used_tile_ids'] and
            sample['missing_tile_ids'] == sample['empty_tile_ids'] == [] and
            sample['sample_count'] == 102400, 'Incomplete normalization provenance')
    require(set(sample['used_tile_ids']) <= {e['tile_id'] for e in roles['train']}, 'Nontraining normalization tile')
    needed = set(sample['used_tile_ids']) | set(chosen['ids']['train']) | set(chosen['ids']['val'])
    lookup = {e['tile_id']: e for e in catalog['tiles']}
    images = {i: binding(Path(args.catalog).resolve().parent / 'tiles' / (i + '.npy'),
                         lookup[i]['tile_sha256']) for i in sorted(needed)}
    require(len(images) == 365, 'Unexpected pilot dependency count')
    require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu_uuid),
            'Require one complete GPU UUID')
    require(1 <= args.timeout <= 1800, 'Pilot timeout must be 1..1800 seconds')
    return {'schema': 'dofa-band-pilot-v1', 'status': 'draft_not_for_execution',
            'policy': POLICY, 'consumer': consumer, 'historical_source': historical_source(args.historical_source),
            'upstream': {'commit': UPSTREAM_COMMIT, 'path': str(Path(args.upstream).resolve()),
                         'files': {n: binding(Path(args.upstream) / n, h) for n,h in UPSTREAM_FILES.items()}},
            'weights': binding(args.weights, WEIGHTS_SHA), 'protocol': protocol,
            'catalog': catalog_b, 'split': split_b, 'cache_manifest': manifest_b, 'cache': files,
            'normalization': norm, 'selection': chosen, 'images': images,
            'historical_runtime': manifest['environment'],
            'launch': {'python': binding(args.python), 'python_entrypoint': str(Path(args.python).absolute()),
                       'gpu_uuid': args.gpu_uuid,
                       'gpu_lock': str(Path(args.gpu_lock).resolve()), 'timeout_seconds': args.timeout},
            'preparation_platform': platform.platform(), 'prepared_utc': now()}


def validate_contract(contract, reviewed=True):
    require(contract['schema'] == 'dofa-band-pilot-v1' and encode(contract['policy']) == encode(POLICY),
            'Pilot scope/policy differs')
    require(not reviewed or contract['status'] == 'reviewed_for_pilot', 'Contract is not reviewed for pilot')
    require(contract['consumer'] == {**clean_source(ROOT), 'script': binding(__file__)}, 'Consumer identity differs')
    require(contract['historical_source'] == historical_source(contract['historical_source']['path']),
            'Historical source differs')
    require(contract['protocol']['sha256'] == PROTOCOL_SHA and contract['weights']['sha256'] == WEIGHTS_SHA,
            'Historical reference identity differs')
    require(contract['upstream']['commit'] == UPSTREAM_COMMIT and
            {n:b['sha256'] for n,b in contract['upstream']['files'].items()} == UPSTREAM_FILES,
            'Upstream identity differs')
    # Reconstruct the entire data contract; seals alone must not admit edited membership.
    args = argparse.Namespace()
    for key in ('protocol', 'catalog', 'split', 'weights'):
        setattr(args, key, contract[key]['path'])
    args.cache = str(Path(contract['cache_manifest']['path']).parent)
    args.historical_source = contract['historical_source']['path']
    args.upstream = contract['upstream']['path']
    args.python = contract['launch']['python_entrypoint']
    args.gpu_uuid = contract['launch']['gpu_uuid']
    args.gpu_lock = contract['launch']['gpu_lock']
    args.timeout = contract['launch']['timeout_seconds']
    reconstructed = prepare(args)
    for key in reconstructed.keys() - {'status', 'prepared_utc', 'preparation_platform'}:
        require(encode(contract[key]) == encode(reconstructed[key]), 'Contract field differs: ' + key)


def imported_bindings(contract):
    """Bind actual modules; reject source shadowing, including preloaded modules."""
    source = contract['historical_source']
    historical_root = Path(source['path'])
    upstream = contract['upstream']
    result = {}
    for name, module in list(sys.modules.items()):
        if name == 'titansar' or name.startswith('titansar.'):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(historical_root), 'Historical import shadowed: ' + name)
            rel = str(path.relative_to(historical_root))
            require(rel in source['files'] and binding(path) == source['files'][rel], 'Imported historical file differs: ' + name)
            result[name] = binding(path)
        elif name in ('dofa_v1', 'wave_dynamic_layer'):
            require(binding(module.__file__) == upstream['files'][name + '.py'], 'Upstream import shadowed: ' + name)
            result[name] = binding(module.__file__)
    return result


def deny_network(event, args):
    if event == 'socket.getaddrinfo' or (event in ('socket.connect', 'socket.sendto') and
                                        args[0].family in (socket.AF_INET, socket.AF_INET6)):
        raise RuntimeError('DOFA pilot forbids network access')


def load_historical(contract, record):
    require(not any(n == 'titansar' or n.startswith('titansar.') or
                    n in ('dofa_v1', 'wave_dynamic_layer') for n in sys.modules),
            'Use a fresh worker without preloaded scientific source')
    sys.path.insert(0, contract['historical_source']['path'])
    modules = []
    for name in ('titansar.data.dataset', 'titansar.models.probing', 'titansar.models.foundation_models'):
        record['import_trace'].append({'event': 'explicit_import_request', 'module': name})
        modules.append(importlib.import_module(name))
        record['import_trace'].append({'event': 'explicit_import_complete', 'module': name})
    imported_bindings(contract)
    return modules


def rng_state(torch):
    return {'cpu_initial_seed': torch.initial_seed(),
            'cpu_state_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'cuda_initialized': torch.cuda.is_initialized()}


def strict_feature_call(forward, model, images, band, role, trace):
    """Direct pinned call: a TypeError is an error, never a retry without the band."""
    require(band in (13.78, 3.75), 'Unexpected numeric band input')
    call = {'numeric_band_input': [band], 'role': role, 'batch_rows': int(images.shape[0]), 'completed': False}
    trace.append(call)
    result = forward(model, images, wave_list=[band])
    call['completed'] = True
    return result


def feature_state(model):
    """Bind registered feature parameters/buffers, excluding the unused head."""
    digest, count = hashlib.sha256(), 0
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('head.'):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(encode({'name': name, 'dtype': str(array.dtype), 'shape': list(array.shape)}))
        digest.update(array.tobytes())
        count += 1
    return {'sha256': digest.hexdigest(), 'tensor_count': count, 'excludes': ['head.weight', 'head.bias']}


def load_offline_model(contract, wrappers, torch, record):
    # Authenticate again immediately before any upstream import or state load.
    for item in contract['upstream']['files'].values():
        check_binding(item)
    check_binding(contract['weights'])
    sys.path.insert(0, contract['upstream']['path'])
    record['rng_before_upstream_import'] = rng_state(torch)
    record['import_trace'].append({'event': 'explicit_import_request', 'module': 'dofa_v1'})
    upstream = importlib.import_module('dofa_v1')
    record['import_trace'].append({'event': 'explicit_import_complete', 'module': 'dofa_v1'})
    record['rng_after_upstream_import'] = rng_state(torch)
    require(torch.initial_seed() == 1234, 'Pinned wave module import seed side effect differs')
    factory = upstream.vit_base_patch16
    forward = upstream.OFAViT.forward_features
    imported_bindings(contract)
    record['strict_forward_source'] = binding(inspect.getsourcefile(forward))
    require(record['strict_forward_source'] == contract['upstream']['files']['dofa_v1.py'],
            'Actual upstream feature function source differs')
    model = factory()  # Exactly the pinned hubconf factory's architecture defaults.
    state = torch.load(contract['weights']['path'], map_location='cpu', weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    require(set(incompatible.missing_keys) == {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'} and
            set(incompatible.unexpected_keys) == {'mask_token', 'norm.weight', 'norm.bias', 'projector.weight', 'projector.bias'},
            'Checkpoint key coverage differs from accepted inspection')
    require(model.global_pool is True and model.fc_norm.eps == 1e-6 and
            bool(torch.all(model.fc_norm.weight == 1)) and bool(torch.all(model.fc_norm.bias == 0)),
            'DOFA pooling/final normalization differs')
    record['rng_after_model_construction'] = rng_state(torch)
    record['feature_state_loaded'] = feature_state(model)
    record['band_calls'] = []
    class PilotDOFA(wrappers.DOFAWrapper):
        # Inherit the historical CPU preparation and no-grad extraction, not
        # its generic TypeError recovery branch. No historical class is changed.
        def _forward(self, images):
            return strict_feature_call(forward, self.model, images, self.wavelength,
                                       self.active_role, record['band_calls'])
    wrapper = PilotDOFA(device='cuda:0', wavelength=13.78)
    record['strict_adapter_source'] = binding(inspect.getsourcefile(wrapper._forward))
    require(record['strict_adapter_source'] == contract['consumer']['script'], 'Actual strict adapter source differs')
    wrapper.model = model.to('cuda:0').eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrapper.weights_source = 'pretrained'
    wrapper.weights_sha256, wrapper.model_revision = WEIGHTS_SHA, UPSTREAM_COMMIT
    return wrapper


def check_dataset(dataset, role, entries, normalization):
    require(dataset.transform is None and dataset.include_vims is False and dataset.normalize is True,
            'Dataset transforms differ')
    require([(e['tile_id'], e['label']) for e in dataset.entries] ==
            [(e['tile_id'], e['label']) for e in entries], 'Dataset membership differs: ' + role)
    require(encode(dataset.normalization_provenance) == encode(normalization), 'Normalization differs: ' + role)


def check_features(features, labels, ids, expected_ids, expected_labels):
    import numpy as np
    require(features.dtype == np.dtype('float32') and features.shape == (len(expected_ids), 768),
            'Feature shape/dtype differs')
    require(bool(np.isfinite(features).all()), 'Nonfinite features')
    require(list(ids) == expected_ids and labels.ndim == 1 and labels.dtype.kind in 'iu' and
            labels.tolist() == expected_labels, 'Output IDs/labels differ')


def bitwise_reference(actual, reference):
    import numpy as np
    require(actual.dtype == reference.dtype == np.dtype('float32') and actual.shape == reference.shape,
            'Reference shape/dtype differs')
    require(bool(np.isfinite(actual).all()) and bool(np.isfinite(reference).all()), 'Nonfinite reference/output')
    different = np.any(actual.view(np.uint32) != reference.view(np.uint32), axis=1)
    rows = np.flatnonzero(different).tolist()
    return {'passed': not rows, 'comparison': 'bitwise_float32', 'rows_checked': len(actual),
            'mismatched_rows': rows}


def save_arrays(output, stem, features, labels, ids):
    import io
    import numpy as np
    records = {}
    for suffix, value in [('features', features), ('labels', labels), ('ids', np.asarray(ids))]:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        path = output / (stem + '-' + suffix + '.npy')
        publish_bytes(path, stream.getvalue())
        records[suffix] = binding(path)
    return records


def run_arms(extract, snapshot, reference, chosen, output, record):
    """Small injectable boundary: tests use arrays, never models/checkpoints."""
    record['arms'] = {
        str(band): {'numeric_band_input': [band], 'status': 'not_reached',
                    'semantics': 'accepted numeric input' if band == 13.78 else 'one-channel v1 radar placeholder sensitivity',
                    'device_uuid': record['device_uuid'], 'attempt': str(output.resolve()),
                    'roles': {r: {'role': r, 'status': 'not_reached',
                                  'source_role_rows': chosen['rows'][r], 'batch_size': 32,
                                  'historical_batch_indices': chosen['training_batch_indices'] if r == 'train' else None,
                                  'new_batch_positions': list(range(6)) if r == 'val' else None}
                              for r in ('train', 'val')}} for band in (13.78, 3.75)}
    state = None
    for band in (13.78, 3.75):
        arm = record['arms'][str(band)]
        arm['status'] = 'running'
        for role in ('train', 'val'):
            item = arm['roles'][role]
            item['status'] = 'running'
            try:
                features, labels, ids = extract(band, role)
                item['arrays'] = save_arrays(output, str(band) + '-' + role, features, labels, ids)
                check_features(features, labels, ids, chosen['ids'][role], chosen['labels'][role])
                if band == 13.78 and role == 'train':
                    record['historical_training_comparison'] = bitwise_reference(features, reference)
                    publish(output / 'historical-training-comparison.json', record['historical_training_comparison'])
                    require(record['historical_training_comparison']['passed'], 'Historical training features differ')
                item.update(status='complete', structural_check_passed=True)
            except BaseException:
                item['status'] = arm['status'] = 'failed'
                raise
        arm['runtime'] = snapshot()
        if state is None:
            state = arm['runtime']['settings']
        else:
            require(encode(state) == encode(arm['runtime']['settings']), 'Runtime settings changed between arms')
        arm['status'] = 'complete'
        publish(output / ('arm-' + str(band) + '.json'), arm)


def package_records():
    records = {}
    for name, expected in PACKAGES.items():
        dist = importlib.metadata.distribution(name)
        require(dist.version == expected, 'Historical package version differs: ' + name)
        metadata = [p for p in dist.files or [] if str(p).endswith('.dist-info/METADATA')]
        require(len(metadata) == 1, 'Missing or ambiguous package METADATA: ' + name)
        records[name] = {'version': dist.version, 'metadata': binding(dist.locate_file(metadata[0]))}
        module_name = {'scikit-learn': 'sklearn', 'scikit-image': 'skimage'}.get(name, name.replace('-', '_'))
        module = sys.modules.get(module_name)
        if module is not None:
            file = Path(module.__file__).resolve()
            require(file.is_relative_to(Path(dist.locate_file('')).resolve()), 'Imported package path differs: ' + name)
            require(str(module.__version__) == expected, 'Imported package version differs: ' + name)
            records[name]['imported_module'] = binding(file)
    return records


def loaded_files():
    """Locations and hashes, never huge source text or raw /proc maps in JSON."""
    python = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            python[name] = str(Path(path).resolve())
    files = {p: binding(p) for p in sorted(set(python.values()))}
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            name = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[5])
            if '.so' in Path(name).name:
                require(not name.endswith(' (deleted)') and Path(name).is_file(),
                        'Unavailable mapped native dependency: ' + name)
                mapped.add(str(Path(name).resolve()))
    return {'module_paths': python, 'module_files': files,
            'mapped_native_files': {p: binding(p) for p in sorted(mapped)}}


def checked_gpu_uuid(processes, pid, expected):
    uuids = {row.split(',')[1].strip() for row in processes.splitlines()
             if row.split(',')[0].strip() == str(pid)}
    require(len(uuids) == 1, 'Cannot bind this process to exactly one physical GPU')
    uuid = uuids.pop()
    require(uuid == expected, 'Loaded GPU UUID differs')
    return uuid


def runtime(torch, wrapper, contract):
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
    props = torch.cuda.get_device_properties(0)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader,nounits'], text=True, timeout=15)
    uuid = checked_gpu_uuid(processes, os.getpid(), contract['launch']['gpu_uuid'])
    require(torch.get_num_threads() == 4 and torch.get_default_dtype() == torch.float32 and
            not torch.is_autocast_enabled('cuda') and not torch.is_autocast_enabled('cpu'),
            'Current FP32/thread/autocast contract differs')
    require(str(torch.__version__) == PACKAGES['torch'] and torch.version.cuda == '12.6' and
            torch.backends.cudnn.version() == 91002, 'Loaded Torch/CUDA/cuDNN differs')
    attention = {n: {'type': type(m).__module__ + '.' + type(m).__qualname__,
                     'fused_attn': bool(m.fused_attn)} for n,m in wrapper.model.named_modules() if hasattr(m, 'fused_attn')}
    settings = {'gpu_uuid': uuid, 'gpu_name': props.name, 'capability': [props.major, props.minor],
                'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'deterministic': torch.are_deterministic_algorithms_enabled(),
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                'float32_matmul_precision': torch.get_float32_matmul_precision(),
                'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
                'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
                'math_sdp': torch.backends.cuda.math_sdp_enabled(),
                'mha_fastpath': torch.backends.mha.get_fastpath_enabled(), 'attention': attention,
                'cpu_initial_seed': torch.initial_seed(), 'cuda_initial_seed': torch.cuda.initial_seed(),
                'threadpools': importlib.import_module('threadpoolctl').threadpool_info()}
    return {'settings': settings, 'platform': platform.platform(), 'python': sys.version,
            'rng': rng_state(torch), 'feature_state': feature_state(wrapper.model),
            'python_entrypoint': sys.executable, 'python_binary': binding(sys.executable),
            'packages': package_records(), 'historical_environment_identical': False,
            'historical_platform': contract['historical_runtime']['platform'],
            'kernel_differs': platform.platform() != contract['historical_runtime']['platform'],
            'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                                 '--format=csv,noheader'], text=True, timeout=15).splitlines(),
            'environment': {k: os.environ.get(k) for k in (*THREAD_ENV, 'CUDA_VISIBLE_DEVICES',
                'PYTHONPATH', 'PYTHONHASHSEED', 'PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE',
                'TIMM_FUSED_ATTN', 'CUDA_MODULE_LOADING', 'CUDNN_LOGLEVEL_DBG', 'NUMEXPR_NUM_THREADS',
                'OMP_DYNAMIC', 'MKL_DYNAMIC', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_DEVICE_ORDER')},
            'imports': imported_bindings(contract),
            'loaded_files': loaded_files()}


def unchanged_loaded_runtime(before, after):
    require(encode(before['settings']) == encode(after['settings']) and
            encode(before['packages']) == encode(after['packages']) and
            before['environment'] == after['environment'], 'Runtime declarations changed')
    for key in ('platform', 'python', 'python_entrypoint', 'python_binary', 'nvidia_smi'):
        require(before[key] == after[key], 'Current runtime changed: ' + key)
    require(before['feature_state'] == after['feature_state'], 'Feature-relevant state changed')
    for kind in ('module_paths', 'module_files', 'mapped_native_files'):
        for key, value in before['loaded_files'][kind].items():
            require(after['loaded_files'][kind].get(key) == value, 'Loaded dependency changed: ' + key)


def worker(contract, output, record):
    record['stages'].update(imports='running', normalization='not_reached', model_load='not_reached',
                            paired_extraction='not_reached', postcheck='not_reached')
    sys.addaudithook(deny_network)
    record['import_trace'] = []
    def import_event(event, args):
        if event == 'import' and (args[0].startswith('titansar') or args[0] in ('dofa_v1', 'wave_dynamic_layer')):
            record['import_trace'].append({'event': 'audit_import_request', 'module': args[0]})
    sys.addaudithook(import_event)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(output / 'unused-pycache')
    import numpy as np
    require(platform.python_version() == '3.12.3', 'Historical Python version differs')
    require(np.__version__ == PACKAGES['numpy'], 'Loaded NumPy version differs')
    package_records()
    dataset_module, probing, wrappers = load_historical(contract, record)
    functions = {'dataset_init': dataset_module.TitanSARDataset.__init__,
                 'dataset_getitem': dataset_module.TitanSARDataset.__getitem__,
                 'sample_train_pixels': dataset_module._sample_train_pixels,
                 'linear_norm_range': dataset_module.compute_linear_norm_range,
                 'prepare_input': wrappers.DOFAWrapper.prepare_input,
                 'wrapper_extract': wrappers.FoundationModelWrapper.extract_features,
                 'extract_features': probing.extract_features}
    record['scientific_callable_files'] = {}
    for name, function in functions.items():
        item = binding(inspect.getsourcefile(inspect.unwrap(function)))
        require(item in contract['historical_source']['files'].values(), 'Callable source differs: ' + name)
        record['scientific_callable_files'][name] = item
    torch = importlib.import_module('torch')
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization before input admission')
    record['stages']['imports'] = 'complete'
    roles, chosen = selection(read(contract['catalog']['path']), read(contract['split']['path']))
    record['device_uuid'] = contract['launch']['gpu_uuid']
    record['stages']['normalization'] = 'running'
    datasets, loaders = {}, {}
    for role in ('train', 'val'):
        ds = dataset_module.TitanSARDataset(contract['catalog']['path'], split=role,
                                           split_manifest_path=contract['split']['path'])
        check_dataset(ds, role, roles[role], contract['normalization'])
        datasets[role] = ds
        loaders[role] = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, chosen['rows'][role]),
            batch_size=32, shuffle=False, num_workers=4, drop_last=False, multiprocessing_context='fork')
    record['normalization_replayed'] = True
    record['stages']['normalization'] = 'complete'
    for item in contract['images'].values():
        raw = np.load(item['path'], allow_pickle=False, mmap_mode='r')
        require(raw.shape == (128, 128) and raw.dtype == np.float32, 'Image header differs')
    cache = contract['cache']
    ids, labels, features = [np.load(cache[s]['path'], allow_pickle=False, mmap_mode='r')
                             for s in ('ids.npy', 'labels.npy', 'feats.npy')]
    require(ids.tolist() == [e['tile_id'] for e in roles['train']] and
            labels.tolist() == [e['label'] for e in roles['train']] and
            features.shape == (len(ids), 768) and features.dtype == np.float32, 'Full training cache identity differs')
    reference = features[chosen['rows']['train']]
    require(torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == 4,
            'Unexpected model construction dtype/thread setting')
    record['stages']['model_load'] = 'running'
    model = load_offline_model(contract, wrappers, torch, record)
    record['stages']['model_load'] = 'complete'
    record['runtime_before'] = runtime(torch, model, contract)
    require(record['feature_state_loaded'] == record['runtime_before']['feature_state'], 'Feature state changed on device transfer')
    publish(output / 'runtime-before.json', record['runtime_before'])
    def extract(band, role):
        model.wavelength = band
        model.active_role = role
        return probing.extract_features(model, loaders[role], 'cuda:0')
    record['stages']['paired_extraction'] = 'running'
    run_arms(extract, lambda: runtime(torch, model, contract), reference, chosen, output, record)
    record['stages']['paired_extraction'] = 'complete'
    record['stages']['postcheck'] = 'running'
    unchanged_loaded_runtime(record['runtime_before'], record['arms']['13.78']['runtime'])
    unchanged_loaded_runtime(record['arms']['13.78']['runtime'], record['arms']['3.75']['runtime'])
    # All admitted bytes must survive both arms; existing files are never updated.
    validate_contract(contract)
    record['imports_after'] = imported_bindings(contract)
    record['stages']['postcheck'] = 'complete'
    record['scientific_acceptance'] = False
    record['scope'] = 'Pilot compatibility only; no full-fold/device-generalization or performance claim'


def supervise(command, env, output, lock_path, timeout, record):
    """Finite foreground process-group lifecycle adapted from retained replay supervisor."""
    child, cancelled = None, []
    started = time.monotonic()
    def stop(signum, frame):
        cancelled.append(signum)
    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with Path(lock_path).open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (output / 'stdout.log').open('xb') as stdout, (output / 'stderr.log').open('xb') as stderr:
                publish(output / 'launch.json', {'command': command, 'supervisor_pid': os.getpid(),
                    'contract': record.get('contract'), 'consumer': record.get('consumer'),
                    'historical_source': record.get('historical_source'), 'gpu_uuid': env.get('CUDA_VISIBLE_DEVICES'),
                    'environment': {k: env.get(k) for k in (*THREAD_ENV, 'PYTHONDONTWRITEBYTECODE')}, 'utc': now()})
                require(not cancelled, 'Cancelled before worker spawn')
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                    stderr=stderr, start_new_session=True, pass_fds=(lock.fileno(),))
                record['child_pid'] = child.pid
                publish(output / 'child-started.json', {'pid': child.pid, 'command': command, 'utc': now()})
                while child.poll() is None:
                    if cancelled:
                        raise InterruptedError('Supervisor cancelled by signal ' + str(cancelled[0]))
                    if time.monotonic() - started > timeout:
                        raise TimeoutError('Pilot deadline exceeded')
                    try:
                        child.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                require(not cancelled, 'Cancelled while worker exited')
                require(child.returncode == 0, 'Worker exit ' + str(child.returncode))
                return child.returncode
    finally:
        if child is not None:
            # Kill/reap the group even if its leader exited while DataLoader children survived.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        record.update(process_exit=None if child is None else child.returncode,
                      cancelled_signals=cancelled, elapsed_seconds=time.monotonic() - started)


def fresh_output(path, protected):
    output = Path(path).resolve()
    require(not any((p / '.git').exists() for p in (output, *output.parents)),
            'Outputs must be outside Git worktrees')
    for p in protected:
        root = Path(p).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'Output overlaps protected input/source')
    output.mkdir()  # Parent must exist; never reuse an attempt.
    return output


def attempt(output, name, action):
    record = {'started_utc': now(), 'pid': os.getpid(), 'status': 'started', 'scientific_acceptance': False}
    publish(output / (name + '-started.json'), record)
    code = 0
    try:
        action(record)
        record['status'] = 'complete'
    except BaseException as exc:
        code = 1
        record['status'] = 'failed'
        for stage, status in record.get('stages', {}).items():
            if status == 'running':
                record['stages'][stage] = 'failed'
        for arm in record.get('arms', {}).values():
            if arm['status'] == 'running':
                arm['status'] = 'failed'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        publish_bytes(output / (name + '-traceback.txt'), traceback.format_exc().encode())
    finally:
        record['finished_utc'] = now()
        record['files'] = {p.name: binding(p) for p in output.iterdir() if p.is_file() and not p.name.startswith('.pending-')}
        publish(output / (name + '-terminal.json'), record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prep = commands.add_parser('prepare', help='Write an unreviewed contract; no model/GPU imports')
    for name in ('historical-source', 'catalog', 'split', 'cache', 'protocol', 'upstream', 'weights', 'python', 'gpu-lock'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--gpu-uuid', required=True)
    prep.add_argument('--timeout', type=int, default=600)
    prep.add_argument('--output', type=Path, required=True)
    for mode in ('pilot', '_worker'):
        p = commands.add_parser(mode, help='Supervised pilot' if mode == 'pilot' else argparse.SUPPRESS)
        p.add_argument('--contract', type=Path, required=True)
        p.add_argument('--contract-sha256', required=True)
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if args.mode == 'prepare':
        output = fresh_output(args.output, [ROOT, args.historical_source, args.catalog.parent,
                              args.cache, args.upstream, args.weights, args.protocol])
        def action(record):
            contract = prepare(args)
            publish(output / 'contract.json', contract)
            record['contract'] = binding(output / 'contract.json')
        return attempt(output, 'prepare', action)
    if args.mode == '_worker':
        output = args.output.resolve()
        started = read(output / 'pilot-started.json')
        require(os.getppid() == started['pid'], 'Worker requires its live foreground supervisor')
    else:
        # Inspect only paths before creating the attempt, so even an admission
        # failure cannot write into a protected data/source directory.
        protected = [ROOT, args.contract]
        try:
            candidate = read(args.contract)
            protected += [candidate['historical_source']['path'], candidate['upstream']['path'],
                          Path(candidate['catalog']['path']).parent, Path(candidate['cache_manifest']['path']).parent]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # The attempted read/hash failure is recorded inside the new attempt.
        output = fresh_output(args.output, protected)
    def action(record):
        record['stages'] = {'admission': 'running', 'scientific_worker': 'not_reached'}
        cb = binding(args.contract, args.contract_sha256)
        contract = read(args.contract)
        record['contract'] = cb
        record['consumer'] = contract.get('consumer')
        record['historical_source'] = {k: contract.get('historical_source', {}).get(k) for k in ('path', 'commit')}
        for p in [contract['historical_source']['path'], contract['upstream']['path'],
                  str(Path(contract['catalog']['path']).parent), str(Path(contract['cache_manifest']['path']).parent)]:
            require(not output.is_relative_to(Path(p).resolve()), 'Attempt is inside a protected input')
        validate_contract(contract)
        require(binding(sys.executable) == contract['launch']['python'] and
                str(Path(sys.executable).absolute()) == contract['launch']['python_entrypoint'],
                'Use declared interpreter entrypoint')
        record['stages']['admission'] = 'complete'
        if args.mode == '_worker':
            record['stages']['scientific_worker'] = 'running'
            worker(contract, output, record)
            record['stages']['scientific_worker'] = 'complete'
            return
        env = os.environ.copy()
        env.update(THREAD_ENV)
        env.update(CUDA_VISIBLE_DEVICES=contract['launch']['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        env.pop('PYTHONPATH', None)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '_worker', '--contract', cb['path'],
                   '--contract-sha256', cb['sha256'], '--output', str(output)]
        record['command'] = command
        record['stages']['scientific_worker'] = 'running'
        supervise(command, env, output, contract['launch']['gpu_lock'],
                  contract['launch']['timeout_seconds'], record)
        terminal = read(output / 'worker-terminal.json')
        require(terminal['status'] == 'complete' and terminal['contract'] == cb and
                terminal['historical_training_comparison'] == {
                    'passed': True, 'comparison': 'bitwise_float32', 'rows_checked': 160, 'mismatched_rows': []} and
                set(terminal['arms']) == {'13.78', '3.75'}, 'Worker receipt incomplete')
        for item in terminal['files'].values():
            check_binding(item)
        record['worker_receipt'] = binding(output / 'worker-terminal.json')
        record['stages']['scientific_worker'] = 'complete'
    return attempt(output, 'worker' if args.mode == '_worker' else 'pilot', action)


if __name__ == '__main__':
    raise SystemExit(main())
