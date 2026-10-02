"""Audit all repaired Titan jobs before publishing a descriptive aggregate.

Requires NumPy, Git, summarize_review_frozen.py, and the producer's raw Titan
normalization tiles. All paths are explicit. No training code is imported.
Incomplete jobs return status 2 without writing an output file. Reports must
be new and outside the scientific source and result trees.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import summarize_review_frozen as common

SOURCE_COMMIT = '6295af2bc35c8d8a2e47fc982b196809245f4550'
PROTOCOL_PATH = 'audit/gpu_execution/repaired_splits/protocol.json'
PROTOCOL_SHA256 = 'ff554e9ee599eebc11753b0a798b288e1daeffbbf35da13a25a982d1e06c01e2'
RANDOM_STATE_SHA256 = '8d2f91054a2413fbd1ca78406586de5149801cda2777e0ffea82d91fd45a29e7'
MODELS = common.MODELS
METRICS = common.METRICS
CLASSES = list(range(6))
STEMS = {'titan_train': 'train', 'titan_test': 'test', 'selk_holdout': 'selk_holdout'}
sha256, read_json = common.sha256, common.read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all()
            and np.allclose(a, b, atol=1e-12, rtol=0), message)


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def partition_ids_hash(ids):
    """The split builder uses sorted IDs with JSON's default separators."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


@lru_cache(maxsize=None)
def source_blob(repository, path):
    return subprocess.check_output(['git', '-C', str(repository), 'show', f'{SOURCE_COMMIT}:{path}'])


def jobs(root):
    return [(model, fold, kind, root / f'repaired-{kind}' / model / f'fold-{fold}')
            for model in MODELS for fold in range(5) for kind in ('features', 'analysis')]


class IncompleteJobs(ValueError):
    def __init__(self, pending):
        self.pending = pending
        super().__init__(f'{len(pending)} jobs remain incomplete')


def require_complete_jobs(root):
    pending = []
    for model, fold, kind, path in jobs(root):
        manifest = path / 'run_manifest.json'
        status = read_json(manifest).get('status') if manifest.is_file() else 'missing'
        if status != 'complete':
            pending.append(dict(model=model, fold=fold, kind=kind, status=status))
    if pending:
        raise IncompleteJobs(pending)


def reconstruct_normalization(train_entries, tiles_dir, tile_size=128):
    """Replay the declared PCG64 sampler directly from catalog-bound NPY pixels."""
    ids = [e['tile_id'] for e in train_entries]
    require(bool(ids) and len(ids) == len(set(ids)), 'Training IDs must be nonempty and unique')
    require(all(isinstance(t, str) and Path(t).name == t and t not in ('', '.', '..') for t in ids), 'Invalid tile ID')
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ids), 200, replace=False) if len(ids) > 200 else range(len(ids))
    selected = [train_entries[i] for i in indices]
    chunks = []
    for entry in selected:
        path = tiles_dir / f"{entry['tile_id']}.npy"
        expected = entry.get('tile_sha256', entry.get('sha256'))
        require(path.is_file() and expected and sha256(path) == expected, f'Normalization tile missing or hash differs: {path}')
        image = np.load(path, allow_pickle=False)
        require(image.shape == (tile_size, tile_size) and image.dtype.kind == 'f', 'Unexpected normalization tile shape or dtype')
        values = image.astype(np.float64).ravel()
        values = values[np.isfinite(values) & (values > 0)]
        require(len(values) > 0, f'No usable normalization pixels: {path}')
        if len(values) > 512:
            values = values[rng.choice(len(values), 512, replace=False)]
        chunks.append(values)
    sample = np.concatenate(chunks)
    lo, hi = np.percentile(sample, [1, 99], method='linear')
    fallback = bool(hi - lo < 1e-6)
    selected_ids = [e['tile_id'] for e in selected]
    return dict(enabled=True, intensity_space='hisar_log_dn', training_ids_sha256=ids_hash(ids),
                training_ids_digest_format='compact JSON array in catalog order, UTF-8',
                method='linear display DN, clip to unit interval', percentiles=[1., 99.],
                lower=0. if fallback else float(lo), upper=255. if fallback else float(hi),
                denominator_epsilon=1e-8, fallback_used=fallback,
                sampling=dict(seed=0, tile_limit=200, pixels_per_tile_limit=512,
                    selection='without replacement, catalog training order, finite positive pixels',
                    selected_tile_ids=selected_ids, used_tile_ids=selected_ids, missing_tile_ids=[], empty_tile_ids=[],
                    sample_count=int(sample.size), sample_float64_le_sha256=hashlib.sha256(sample.astype('<f8').tobytes()).hexdigest()))


def check_normalization_record(record, expected):
    require(set(record) == set(expected), 'Normalization schema differs')
    for key in expected:
        if key in ('lower', 'upper'):
            close(record[key], expected[key], f'Normalization {key} differs')
        else:
            require(record[key] == expected[key], f'Normalization {key} differs')
    require(record['enabled'] is True and record['fallback_used'] is expected['fallback_used'], 'Invalid normalization flags')


def kappa_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    observed = np.trace(counts) / n
    expected = np.dot(counts.sum(0), counts.sum(1)) / n ** 2
    return float((observed - expected) / (1 - expected)) if expected != 1 else float(observed == 1)


def check_evaluation(record, ids, labels, knn=False):
    labels = np.asarray(labels)
    truth = np.asarray(record['true_labels'])
    require(record['tile_ids'] == ids and len(ids) == len(set(ids)), 'Prediction tile IDs or order differ')
    require(truth.dtype.kind in 'iu' and np.array_equal(truth, labels)
            and np.unique(labels).tolist() == CLASSES, 'True labels differ or six-class test support is incomplete')
    require(not any(k.startswith('source_supported') for k in record), 'Earth class filtering enters Titan-only evaluation')
    if knn:
        pred = np.asarray(record['predictions'])
        require(pred.dtype.kind in 'iu' and pred.shape == labels.shape and np.isin(pred, CLASSES).all(), 'Invalid kNN predictions')
        counts = np.bincount(6 * labels + pred, minlength=36).reshape(6, 6)
        checked = {**common.metrics_from_confusion(counts, CLASSES), 'confusion_counts': counts.tolist()}
        close(record['macro_accuracy'], checked['macro']['recall'], 'kNN recall differs')
        close(record['per_class_accuracy'], checked['per_class']['recall'], 'kNN per-class recall differs')
        require(type(record['k']) is int and record['k'] == 20, 'kNN neighbor count differs from protocol defaults')
    else:
        checked = common.checked_predictions(record, labels, CLASSES)
    kappa = kappa_from_counts(checked['confusion_counts'])
    close(record['kappa'], kappa, 'Saved kappa differs from predictions')
    return {**checked, 'kappa': kappa}


def aggregate_folds(folds):
    require(len(folds) == 5 and [f['fold'] for f in folds] == CLASSES[:5], 'Expected five ordered folds')
    reduced = {'linear_probe': [], 'knn': []}
    for fold in folds:
        heads = common.ordered_seeds(fold['probe_heads'])
        reduced['linear_probe'].append(dict(
            macro={m: float(np.mean([h['macro'][m] for h in heads])) for m in METRICS},
            per_class={m: np.mean([h['per_class'][m] for h in heads], axis=0).tolist() for m in METRICS}))
        reduced['knn'].append(fold['knn'])
    return {task: dict(macro={m: common.describe_folds([f['macro'][m] for f in values]) for m in METRICS},
                       per_class={m: [common.describe_folds([f['per_class'][m][k] for f in values]) for k in CLASSES] for m in METRICS})
            for task, values in reduced.items()}


def load_protocol(repository):
    raw = source_blob(repository, PROTOCOL_PATH)
    require(hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA256, 'Protocol file differs from launch identity')
    protocol = common.parse_json(raw)
    spec_raw = json.dumps(protocol['specification'], sort_keys=True, separators=(',', ':')).encode()
    require(hashlib.sha256(spec_raw).hexdigest() == protocol['specification_sha256'], 'Protocol specification hash differs')
    require([s['fold'] for s in protocol['splits']] == list(range(5)), 'Protocol fold identities differ')
    splits = []
    for fold, item in enumerate(protocol['splits']):
        require(item['path'] == f'titan_contiguous_fold_{fold}.json', 'Unexpected split filename')
        raw = source_blob(repository, str(Path(PROTOCOL_PATH).parent / item['path']))
        require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Split file hash differs')
        split = common.parse_json(raw)
        require(split['catalog_sha256'] == protocol['catalog_sha256']
                and split['policy']['test_fold'] == fold
                and split['policy']['protocol_specification_sha256'] == protocol['specification_sha256'], 'Split/protocol binding differs')
        splits.append(split)
    return protocol, splits


def check_source_metadata(metadata, expected, state):
    expected = {**expected, 'model_revision': state[0], 'weights_sha256': state[1]}
    require(set(metadata) == set(expected), 'Result source metadata schema differs')
    check_normalization_record(metadata['normalization_provenance'], expected['normalization_provenance'])
    for key in set(expected) - {'normalization_provenance'}:
        require(metadata[key] == expected[key], f'Result source metadata differs: {key}')


def check_commands(manifest, path, kind, model, feature_root):
    command = manifest['command']
    for flag, value in (('--models', [model]), ('--seeds', ['0', '1', '2', '3', '4']),
                        ('--batch_size', ['32']), ('--output_dir', [str(path)])):
        require(common.option(command, flag) == value, f'Producer command differs: {flag}')
    require(command.count('--within_titan_only') == 1 and command.count('--require_real_weights') == 1
            and '--earth_catalog' not in command and '--earth_split_manifest' not in command, 'Job is not strictly Titan only')
    for key in ('titan_catalog', 'titan_split_manifest'):
        require(common.option(command, f'--{key}') == [manifest['inputs'][key]['path']], f'Command/input path differs: {key}')
    if kind == 'features':
        require(command.count('--features_only') == 1 and '--cached_features_dir' not in command
                and common.option(command, '--device')[0].startswith('cuda'), 'Unexpected extraction mode')
    else:
        require('--features_only' not in command and common.option(command, '--device') == ['cpu']
                and common.option(command, '--cached_features_dir') == [str(feature_root / 'features')], 'Unexpected analysis mode')


def check_partition(catalog, split):
    ids = [e['tile_id'] for e in catalog['tiles']]
    require(len(ids) == len(set(ids)) and set(ids) == set(split['assignments']), 'Catalog/split ID coverage differs')
    require(all(type(e['label']) is int and e['label'] in CLASSES for e in catalog['tiles']), 'Invalid catalog class labels')
    roles = {role: [e for e in catalog['tiles'] if split['assignments'][e['tile_id']] == role]
             for role in ('train', 'val', 'test', 'selk_holdout', 'buffer_excluded')}
    require(sum(map(len, roles.values())) == len(ids), 'Unexpected partition role')
    summary = split['policy']['role_summary_before_exclusion_role_merge']
    for role in ('train', 'val', 'test', 'selk_holdout'):
        rows = roles[role]
        require(len(rows) == summary[role]['total']
                and np.bincount([r['label'] for r in rows], minlength=6).tolist() == summary[role]['class_counts']
                and partition_ids_hash([r['tile_id'] for r in rows]) == summary[role]['sorted_ids_sha256'], f'Partition summary differs: {role}')
    return roles


def summarize(root, repository):
    require_complete_jobs(root)
    protocol, splits = load_protocol(repository)
    paths = ('scripts/run_probing.py', 'titansar/data/dataset.py', 'titansar/models/probing.py',
             'titansar/models/foundation_models.py', 'titansar/models/croma_model.py',
             'titansar/configs/defaults.py', 'titansar/reproducibility.py')
    source_hashes = {p: hashlib.sha256(source_blob(repository, p)).hexdigest() for p in paths}
    tree = ast.parse(source_blob(repository, 'titansar/models/foundation_models.py'))
    pinned = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    report = dict(schema_version='1.0.0', source_commit=SOURCE_COMMIT, source_file_sha256=source_hashes,
                  auditor_sha256=sha256(Path(__file__)), helper_sha256=sha256(Path(common.__file__)),
                  results_root=str(root), protocol_file_sha256=PROTOCOL_SHA256,
                  protocol_specification_sha256=protocol['specification_sha256'], catalog_sha256=protocol['catalog_sha256'],
                  evaluation_scope='within_titan_only', class_names=list(common.CLASS_NAMES), classifier_outputs=6,
                  aggregation='mean five head seeds within fold, then equal-weight mean and sample SD across five folds; kNN has one evaluation per fold',
                  uncertainty='descriptive fold variation; training folds overlap; no confidence interval or independent-crater claim',
                  comparison_scope='new catalog population and geographic partitions; differences from v1 cannot isolate a buffer or repair effect',
                  dofa_identifier_ghz=13.78, random_encoder_seed=42, random_state_reference_sha256=RANDOM_STATE_SHA256,
                  metric_atol=1e-12, probability_sum_atol=1e-6, zero_division=0,
                  limitations=['No model inference, probe training, or neighbor search is repeated by this audit.',
                               'Encoder state identities are checked against checkpoint pins and the prior verified seed-42 digest; encoders are not recreated.',
                               'Raw image hashes are checked for normalization samples, not every pixel input to the encoders.',
                               'Catalog label truth and spatial independence are not established by artifact consistency.',
                               'One Crater test tile in fold 1 makes that class recall an especially fragile observation.'],
                  folds=[], jobs=[], models={}, validation=dict(complete_manifests=0, output_hashes=0,
                      feature_arrays=0, probe_heads=0, knn_vectors=0, prediction_rows=0, normalization_pixel_replays=0))
    stats = report['validation']
    fold_inputs, seen_test = {}, set()
    for model in MODELS:
        evaluated, states = [], set()
        for fold in range(5):
            print(f'Validating {model}/fold-{fold}', file=sys.stderr, flush=True)
            feature_root = root / 'repaired-features' / model / f'fold-{fold}'
            analysis_root = root / 'repaired-analysis' / model / f'fold-{fold}'
            manifests = {}
            for kind, path in (('features', feature_root), ('analysis', analysis_root)):
                manifest = common.verify_manifest(path, SOURCE_COMMIT)
                manifests[kind] = manifest
                expected_outputs = {'probing_results.json'}
                if kind == 'features':
                    expected_outputs.update(f'features/{model}/{stem}_{suffix}' for stem in STEMS
                                            for suffix in ('feats.npy', 'ids.npy', 'labels.npy', 'feats.meta.json'))
                require(set(manifest['outputs']) == expected_outputs, 'Unexpected job output inventory')
                require(set(manifest['inputs']) == {'titan_catalog', 'titan_split_manifest'} |
                        ({'cached_features'} if kind == 'analysis' else set()), 'Unexpected job input inventory')
                check_commands(manifest, path, kind, model, feature_root)
                for key, expected in (('titan_catalog', protocol['catalog_sha256']),
                                      ('titan_split_manifest', protocol['splits'][fold]['sha256'])):
                    item = manifest['inputs'][key]
                    require(item['sha256'] == expected and sha256(Path(item['path'])) == expected, f'Input hash differs: {key}')
                stats['complete_manifests'] += 1
                stats['output_hashes'] += len(manifest['outputs'])
            feature, analysis = manifests['features'], manifests['analysis']
            require(all(feature['inputs'][k] == analysis['inputs'][k] for k in ('titan_catalog', 'titan_split_manifest')), 'Producer/consumer input bindings differ')
            cache = analysis['inputs']['cached_features']
            require(Path(cache['path']).resolve() == (feature_root / 'features').resolve(), 'Analysis consumes another cache')
            expected_cache = {k.removeprefix('features/'): v for k, v in feature['outputs'].items() if k.startswith('features/')}
            require(cache['files'] == expected_cache, 'Producer/consumer cache hashes differ')
            common.verify_files(Path(cache['path']), expected_cache)
            catalog_path = Path(feature['inputs']['titan_catalog']['path'])
            if fold not in fold_inputs:
                catalog = read_json(catalog_path)
                require(catalog['class_names'] == list(common.CLASS_NAMES) and catalog['num_classes'] == 6
                        and catalog['intensity_space'] == 'hisar_log_dn' and catalog['tile_size'] == 128
                        and bool(catalog['benchmark_track']), 'Unexpected repaired catalog semantics')
                roles = check_partition(catalog, splits[fold])
                normalization = reconstruct_normalization(roles['train'], catalog_path.parent / 'tiles')
                ids = [e['tile_id'] for e in roles['test']]
                require(not seen_test.intersection(ids), 'Test IDs overlap across folds')
                seen_test.update(ids)
                fold_inputs[fold] = (catalog, roles, normalization)
                report['folds'].append(dict(fold=fold, split_sha256=protocol['splits'][fold]['sha256'],
                    test_ids=ids, normalization_provenance=normalization,
                    counts={r: dict(total=len(rows), class_counts=np.bincount([e['label'] for e in rows], minlength=6).tolist()) for r, rows in roles.items()},
                    recorded_geometry={k: splits[fold]['policy'][k] for k in ('name', 'buffer_policy', 'radius_m', 'footprint_m', 'gap_m', 'minimum_center_m', 'sector_boundaries_east_deg')}))
                stats['normalization_pixel_replays'] += 1
            catalog, roles, normalization = fold_inputs[fold]
            bindings = dict(titan_catalog_sha256=protocol['catalog_sha256'], earth_catalog_sha256=None,
                            titan_split_manifest_sha256=protocol['splits'][fold]['sha256'], earth_split_manifest_sha256=None,
                            evaluation_scope='within_titan_only', benchmark_track=catalog['benchmark_track'],
                            normalization_provenance=normalization)
            input_hashes = {k.removesuffix('_sha256'): v for k, v in bindings.items() if k.endswith('_sha256')}
            arrays = {}
            for stem, role in STEMS.items():
                directory = feature_root / 'features' / model
                ids, labels, state = common.check_cache(directory, model, stem, roles[role], input_hashes, pinned)
                meta = read_json(directory / f'{stem}_feats.meta.json')
                provenance = meta['provenance']
                require(provenance['evaluation_scope'] == 'within_titan_only'
                        and provenance['benchmark_track'] == catalog['benchmark_track'], 'Cache scope or track differs')
                check_normalization_record(provenance['normalization_provenance'], normalization)
                if model == 'random_init':
                    require(state[1] == RANDOM_STATE_SHA256, 'RandomInit state differs from the verified seed-42 reference')
                states.add(state)
                arrays[stem] = (ids, labels)
                stats['feature_arrays'] += 1
            require(len(states) == 1, 'Encoder revision or weights change across folds/arrays')
            state = next(iter(states))
            extraction = read_json(feature_root / 'probing_results.json')
            payload = read_json(analysis_root / 'probing_results.json')
            require(set(extraction) == set(payload) == {model}, 'Result model inventory differs')
            extraction, payload = extraction[model], payload[model]
            expected_weight_source = 'random_init' if model == 'random_init' else 'pretrained'
            for block in (extraction, payload):
                require(block['model'] == model and block['evaluation_scope'] == 'within_titan_only'
                        and block['weights_source'] == expected_weight_source, 'Result model, scope, or weight source differs')
                check_source_metadata(block['source_metadata'], bindings, state)
            require(extraction['feature_extraction_only'] is True and 'linear_probe' not in extraction, 'Unexpected extraction result')
            require(payload['device'] == 'cpu' and set(payload['linear_probe']) == set(payload['knn']) == {'titan_to_titan'}
                    and 'domain_gap' not in payload, 'Cross-domain output enters Titan-only report')
            ids, labels = arrays['titan_test']
            block = payload['linear_probe']['titan_to_titan']
            require(block['training_class_indices'] == np.unique(arrays['titan_train'][1]).tolist() == CLASSES,
                    'Probe training class support differs')
            heads = common.ordered_seeds(block['seed_results'])
            checked = [dict(seed=h['seed'], **check_evaluation(h, ids, labels)) for h in heads]
            for key, values in (('macro_accuracy', [h['macro']['recall'] for h in checked]), ('kappa', [h['kappa'] for h in checked])):
                close(block[f'{key}_mean'], np.mean(values), f'Seed mean differs: {key}')
                close(block[f'{key}_std'], np.std(values, ddof=1), f'Seed SD differs: {key}')
            close(block['per_class_accuracy'], np.mean([h['per_class']['recall'] for h in checked], axis=0), 'Seed per-class recall differs')
            knn = check_evaluation(payload['knn']['titan_to_titan'], ids, labels, knn=True)
            evaluated.append(dict(fold=fold, support=knn['support'], probe_heads=checked, knn=knn))
            report['jobs'].append(dict(model=model, fold=fold,
                extraction_manifest_sha256=sha256(feature_root / 'run_manifest.json'),
                analysis_manifest_sha256=sha256(analysis_root / 'run_manifest.json'),
                result_sha256=analysis['outputs']['probing_results.json']))
            stats['probe_heads'] += 5
            stats['knn_vectors'] += 1
            stats['prediction_rows'] += 6 * len(ids)
        report['models'][model] = dict(model_revision=state[0], weights_sha256=state[1], folds=evaluated, **aggregate_folds(evaluated))
    require(len(seen_test) == protocol['test_union_count']
            and partition_ids_hash(list(seen_test)) == protocol['test_union_sorted_ids_sha256'], 'Test union differs from sealed protocol')
    report['benchmark_track'] = catalog['benchmark_track']
    stats['distinct_test_tiles'] = len(seen_test)
    stats['normalization_sample_values'] = sum(f['normalization_provenance']['sampling']['sample_count'] for f in report['folds'])
    stats['status'] = 'pass'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results-root', 'source-repo', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) for p in (args.results_root, args.source_repo)):
        parser.error('Output must be new and outside scientific source and result trees')
    try:
        report = summarize(args.results_root.resolve(), args.source_repo.resolve())
    except IncompleteJobs as error:
        print(json.dumps(dict(status='incomplete', pending_jobs=error.pending, report_written=False), sort_keys=True))
        return 2
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(report['validation'], sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
