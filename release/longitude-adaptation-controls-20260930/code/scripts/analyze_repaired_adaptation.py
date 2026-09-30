"""Compare accepted adaptation and frozen recipes using small audit reports.

Requires independently accepted SHA256 values for the complete FT and replay
audits. Other inputs have fixed release pins. Uses NumPy only; never loads a
checkpoint, imports a producer, fits a classifier or runs Git. Example:
python -m scripts.analyze_repaired_adaptation --help
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[_name] = '1'
    sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts import plot_repaired_common_classifier as joint
from scripts import summarize_repaired_titan as native

common = native.common
require, close = joint.require, joint.close
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dinov2', 'dofa', 'croma')
PAIRS = {(m, f) for m in common.MODELS for f in range(5)}
JOBS = {(m, f, s) for m in MODELS for f in range(5) for s in range(5)}
PINS = dict(frozen_audit='db352133220ce64b71f68e79269f1f93407d1906ad6545c06797eadd24a092b2',
            common_audit=joint.PINS['common'], common_protocol=joint.PINS['common_protocol'],
            ft_protocol='dd280f9b830acd8fba84a0603504c12ee39941b40fb89912a3e264002c8f44cb',
            replay_protocol='b55152cd387cc868eec9a39f4bfc621b8c9fdb169f307008e5a4ca4af0a9e1ca')
FT_COMMIT = 'ed2eb3ecd4722c22c0aff8b5370ac9a4b23e4376'
REPLAY_COMMIT = '97d4c5cc50a63b0bc9f2b6c09f48ed8d88fad94e'
FT_PLAN_SHA = '9c6de3f07ad3ba063115dc27c478104ee766942ae03f9a24ca180b5b00423b15'
FOLLOWUP_PLAN_SHA = 'baaa83961f896523ad2edf711ccc166edb9bf14249a2e68f85321c2bc6b92d11'
REPLAY_PLAN_SHA = '94510a4141dcbf42b17438ce6eda01ffb458f3f2c40ec986e0fdc0dc388b1d60'
REPLAY_CHECKER_SHA = '94458b9c0f7327a1981c66ab818b504d37332afd56ce61f17b3cf2c49c88a9b5'
HISTORY = ('selected_epoch', 'epoch_indexing', 'optimizer_attempts_per_epoch',
           'optimizer_steps_total', 'amp_skipped_steps_total', 'training_example_presentations')
GROUPS = ('finetuned', 'frozen_sgd', 'common_lr', 'ft_minus_sgd', 'ft_minus_lr')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exact(actual, expected, message):
    require(digest(actual) == digest(expected), message)


def index(rows, fields, expected):
    keys = [tuple(row[k] for k in fields) for row in rows]
    require(all(type(row[k]) is int for row in rows for k in fields if k in ('fold', 'seed')),
            'Fold and seed identities must be integers')
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Incomplete or duplicate identity coverage')
    return dict(zip(keys, rows))


def accepted_inputs(data):
    """Check every acceptance and identity before reconstructing outcome metrics."""
    ft, replay, frozen, lr = (data[k] for k in ('ft_audit', 'replay_audit', 'frozen_audit', 'common_audit'))
    specs = {}
    for name in ('ft_protocol', 'replay_protocol', 'common_protocol'):
        protocol = data[name]
        exact(digest(protocol['specification']), protocol['specification_sha256'], 'Protocol specification differs')
        specs[name] = protocol['specification']
    fs, rs, cs = (specs[k] for k in ('ft_protocol', 'replay_protocol', 'common_protocol'))
    for spec in (fs, cs):
        exact([f['fold'] for f in spec['folds']], list(range(5)), 'Protocol fold order differs')
    exact([ft['status'], ft['validated_jobs'], ft['source_commit'], ft['protocol_sha256'],
           ft['specification_sha256'], ft['frozen_audit_sha256']],
          ['pass', 75, FT_COMMIT, PINS['ft_protocol'], data['ft_protocol']['specification_sha256'], PINS['frozen_audit']],
          'FT acceptance differs')
    exact([replay['status'], replay['verified_replays'], replay['replay_execution_commit'],
           replay['replay_protocol_sha256'], replay['parent_plan_sha256'], replay['plan_sha256']],
          ['pass', 75, REPLAY_COMMIT, PINS['replay_protocol'], FT_PLAN_SHA, REPLAY_PLAN_SHA], 'Replay acceptance differs')
    exact({k: replay['checker_source'][k] for k in ('commit', 'clean', 'status')},
          dict(commit=REPLAY_COMMIT, clean=True, status=[]), 'Replay checker source differs')
    exact(replay['checker_sha256'], REPLAY_CHECKER_SHA, 'Replay checker hash differs')
    exact(ft['auditor_sha256'], rs['replay_source_sha256']['scripts/summarize_repaired_finetuning.py'], 'FT auditor differs')
    exact(ft['helper_sha256'], fs['scientific_source_sha256']['scripts/summarize_review_frozen.py'], 'FT helper differs')
    for name, value in rs['replay_source_sha256'].items():
        exact(replay['checker_helpers_sha256'][name], value, 'Replay helper differs')
    exact([rs['producer_commit'], rs['training_protocol_sha256'], rs['verified_frozen_audit_sha256']],
          [FT_COMMIT, PINS['ft_protocol'], PINS['frozen_audit']], 'Replay protocol parent differs')
    exact(rs['producer_plan_sha256'], FT_PLAN_SHA, 'Replay producer plan differs')
    exact(rs['inference']['cpu_intraop_threads'], 4, 'Replay CPU thread count differs')
    exact(
        rs['execution']['thread_environment'],
        dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'),
        'Replay CPU thread environment differs',
    )
    exact([lr['status'], lr['producer_commit'], lr['protocol_sha256'], lr['plan_sha256'], lr['native_audit_sha256']],
          ['complete', joint.COMMITS['common'], PINS['common_protocol'], joint.PLANS['common'], PINS['frozen_audit']],
          'Common LR acceptance differs')
    exact(lr['validation'], dict(complete_jobs=20, estimator_replays=20, output_hashes=60), 'Common LR coverage differs')
    exact(lr['auditor_sha256'], cs['scientific_source_sha256']['scripts/summarize_repaired_common_classifier.py'], 'LR checker differs')
    require(frozen['validation']['status'] == 'pass' and frozen['validation']['probe_heads'] == 100
            and frozen['source_commit'] == fs['frozen_source_commit'] == native.SOURCE_COMMIT, 'Frozen acceptance differs')
    for record in (ft, frozen, cs):
        exact(record['class_names'], fs['class_names'], 'Class order differs')
    exact(fs['class_names'], list(common.CLASS_NAMES), 'Expected six map classes')
    exact(fs['catalog_sha256'], cs['catalog']['sha256'], 'FT/LR catalog differs')
    exact(frozen['catalog_sha256'], fs['catalog_sha256'], 'Frozen catalog differs')
    exact([fs['verified_frozen_audit_sha256'], cs['native_audit']['sha256']], [PINS['frozen_audit']] * 2, 'Frozen parent differs')
    ftr = index(ft['records'], ('model', 'fold', 'seed'), JOBS)
    rr = index(replay['jobs'], ('model', 'fold', 'seed'), JOBS)
    lrr = index(lr['folds'], ('model', 'fold'), PAIRS)
    caches = index(cs['caches'], ('model', 'fold'), PAIRS)
    njobs = index(frozen['jobs'], ('model', 'fold'), PAIRS)
    refs = index(fs['frozen_jobs'], ('model', 'fold'), {(m, f) for m in MODELS for f in range(5)})
    ff, cf, nf = [index(s['folds'], ('fold',), {(f,) for f in range(5)}) for s in (fs, cs, frozen)]
    for fold in range(5):
        f, c, n = (rows[fold,] for rows in (ff, cf, nf))
        exact([c['split']['sha256'], n['split_sha256']], [f['split_sha256']] * 2, 'Split differs')
        exact([c['normalization_sha256'], digest(n['normalization_provenance']), digest(c['normalization_provenance'])],
              [f['normalization_sha256']] * 3, 'Normalization differs')
        for role in ('train', 'test'):
            exact([c['counts'][role], n['counts'][role]], [f['counts'][role]] * 2, 'Role support differs')
        exact(n['counts']['val'], f['counts']['val'], 'Validation support differs')
        exact(native.ids_hash(n['test_ids']), c['test_ids_sha256'], 'Test ID order differs')
        exact(c['train_ids_sha256'], n['normalization_provenance']['training_ids_sha256'], 'Training IDs differ')
    for key, ref in refs.items():
        exact(ref, njobs[key], 'Frozen reference manifests differ')
        exact(caches[key]['manifest_sha256'], ref['extraction_manifest_sha256'], 'Common cache manifest differs')
        exact(caches[key]['encoder_identity'], fs['encoders'][key[0]], 'Encoder identity differs')
    for key, row in ftr.items():
        model, fold, seed = key
        p, f, r = row['provenance'], ff[fold,], rr[key]
        require(set(row['outputs']) == {'finetuning_results.json', f'checkpoints/{model}/ft_2block_seed_{seed}.pt'},
                'FT output inventory differs')
        hashes = [row['manifest_sha256'], r['producer_manifest_sha256'], r['replay_manifest_sha256'],
                  r['replay_sha256'], *row['outputs'].values()]
        require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in hashes), 'Invalid artifact digest')
        expected = dict(model_revision=fs['encoders'][model]['model_revision'], weights_sha256=fs['encoders'][model]['weights_sha256'],
            titan_catalog_sha256=fs['catalog_sha256'], titan_split_manifest_sha256=f['split_sha256'],
            fold=fold, seed=seed, recipe=fs['recipe'], class_counts=f['counts'],
            protocol_sha256=PINS['ft_protocol'], protocol_specification_sha256=data['ft_protocol']['specification_sha256'],
            normalization_provenance=nf[fold,]['normalization_provenance'],
            frozen_reference={**refs[model, fold], 'source_commit': native.SOURCE_COMMIT,
                              'matched_head_seed': seed, 'validated_head_seeds': list(range(5))},
            dofa_band_identifier_ghz=13.78 if model == 'dofa' else None, evaluation_scope='within_titan_only')
        exact({k: p[k] for k in expected}, expected, 'FT provenance differs')
        exact(r['producer_manifest_sha256'], row['manifest_sha256'], 'Replay producer manifest differs')
        exact(r['selected_epoch'], row['selected_epoch'], 'Replay selected epoch differs')
        comparisons = r['comparisons']
        require(set(comparisons) == {'validation', 'test'} and all(c['passed'] is True for c in comparisons.values()),
                'Replay comparison failed')
        v, t = comparisons['validation'], comparisons['test']
        require(v['historical_vector_available'] is False and 0 <= v['macro_recall_abs_error'] <= 1e-12
                and t['confusion_exact'] is True and t['prediction_mismatches'] == 0
                and t['mismatched_tile_ids'] == [] and t['probability_components_over_atol'] == 0
                and 0 <= t['probability_max_abs_error'] <= 1e-6, 'Replay tolerances differ')
    return fs, cs, ftr, rr, lrr


def training_record(row, fold):
    history = {k: row[k] for k in HISTORY}
    exact(history, {k: row['provenance'][k] for k in HISTORY}, 'Update provenance differs')
    require(all(type(history[k]) is int for k in HISTORY if k != 'epoch_indexing'), 'Invalid training counters')
    attempts = fold['optimizer_attempts_per_epoch']
    require(attempts == (fold['counts']['train']['total'] + 31) // 32
            and history['optimizer_attempts_per_epoch'] == attempts and history['epoch_indexing'] == 'one based'
            and 1 <= history['selected_epoch'] <= 50 and history['optimizer_steps_total'] >= 50
            and history['amp_skipped_steps_total'] >= 0
            and history['optimizer_steps_total'] + history['amp_skipped_steps_total'] == 50 * attempts
            and history['training_example_presentations'] == 50 * fold['counts']['train']['total'], 'Training budget differs')
    return {**history, 'optimizer_attempts_total': 50 * attempts}


def fit_record(row, spec):
    require(row['estimator_replay']['verified'] is True, 'LR estimator replay missing')
    fit = row['fit']
    for name in ('classifier', 'scaler'):
        exact(fit[name + '_parameters'], spec['resolved_parameters'][name], 'LR recipe differs')
    require(len(fit['n_iter']) == 1 and type(fit['n_iter'][0]) is int and 0 <= fit['n_iter'][0] <= 2000,
            'Invalid LR iteration count')
    exact(fit['iteration_limit_reached'], fit['n_iter'][0] == 2000, 'LR iteration cap differs')
    exact(fit['convergence_warning'], any(w['category'] == 'ConvergenceWarning' for w in fit['fit_warnings']), 'LR warning flag differs')
    return fit


def score_summary(rows):
    return {level: {metric: dict(values=[r[level][metric] for r in rows],
                   mean=np.mean([r[level][metric] for r in rows], axis=0).tolist(),
                   sample_std=np.std([r[level][metric] for r in rows], axis=0, ddof=1).tolist())
                   for metric in common.METRICS} for level in ('macro', 'per_class')}


def difference(left, right):
    return {level: {metric: (100 * (np.asarray(left[level][metric]) - right[level][metric])).tolist()
                    for metric in common.METRICS} for level in ('macro', 'per_class')}


def analyze(data):
    fs, cs, ftr, rr, lrr = accepted_inputs(data)
    frozen = data['frozen_audit']
    result = dict(status='complete', schema_version='1.0.0', class_names=fs['class_names'], models={},
        validation=dict(ft_jobs=75, checkpoint_replays=75, common_lr_jobs=20),
        definitions=dict(metrics='six-class macro and class precision, recall and F1; undefined precision is zero',
            units='Scores are fractions; FT minus reference differences are percentage points.',
            aggregation='Five seed metrics within each fold; equal mean and sample SD across five fold means.',
            ft_minus_sgd='Paired matching-seed differences, averaged within fold.',
            ft_minus_lr='FT seed mean minus the single common LR fit within fold.',
            training_counters='Successful updates, AMP skips and example presentations cover the full fifty epochs.',
            selection='Earliest highest validation macro recall among updated epochs; all fifty epochs run.'),
        limitations=['Expert-map agreement; independent geological accuracy remains unmeasured.',
            'Overlapping training populations; seed and fold SD are descriptive.',
            'Recipe differences include optimization, regularization, batching, precision, budget and selection.',
            'These contrasts do not isolate unfreezing or pretraining. RandomInit has no adaptation arm.',
            'Selected-checkpoint validation vectors were replayed; historical validation vectors were unavailable.',
            'Artifact and checkpoint acceptance is inherited from the supplied independently accepted report hashes.'])
    frozen_heads = {}
    for model in MODELS:
        folds = index(frozen['models'][model]['folds'], ('fold',), {(f,) for f in range(5)})
        for fold in range(5):
            for seed, row in enumerate(common.ordered_seeds(folds[fold,]['probe_heads'])):
                frozen_heads[model, fold, seed] = row
    # Check every common fit, including the five RandomInit references outside this contrast.
    for key, row in lrr.items():
        exact(row['counts'], cs['folds'][key[1]]['counts'], 'LR membership differs')
        fit_record(row, cs)
        joint.reconstruct(row, cs['folds'][key[1]]['counts']['test']['class_counts'])
    for model in MODELS:
        folds = []
        for fold in range(5):
            f = fs['folds'][fold]
            support = f['counts']['test']['class_counts']
            lr = joint.reconstruct(lrr[model, fold], support)
            seeds = []
            for seed in range(5):
                row, replay = ftr[model, fold, seed], rr[model, fold, seed]
                ft, sgd = [joint.reconstruct(row[k], support) for k in ('finetuned', 'frozen')]
                exact(sgd['confusion_counts'], joint.reconstruct(frozen_heads[model, fold, seed], support)['confusion_counts'],
                      'SGD seed reference differs')
                exact(ft['confusion_counts'], joint.reconstruct(replay['test_metrics'], support)['confusion_counts'],
                      'Replay test metrics differ')
                val = replay['validation_metrics']
                vm = joint.reconstruct({**val, 'kappa': native.kappa_from_counts(val['confusion_counts'])},
                                       f['counts']['val']['class_counts'])
                close(vm['macro']['recall'], replay['comparisons']['validation']['expected_macro_recall'], 'Validation recall differs')
                seeds.append(dict(seed=seed, finetuned=ft, frozen_sgd=sgd, ft_minus_sgd=difference(ft, sgd),
                    training=training_record(row, f), validation=vm, manifest_sha256=row['manifest_sha256'],
                    outputs=row['outputs'],
                    replay_manifest_sha256=replay['replay_manifest_sha256'], replay_sha256=replay['replay_sha256']))
            summaries = {name: score_summary([r[name] for r in seeds]) for name in ('finetuned', 'frozen_sgd', 'ft_minus_sgd')}
            values = {name: {level: {m: summaries[name][level][m]['mean'] for m in common.METRICS}
                            for level in ('macro', 'per_class')} for name in summaries}
            values['common_lr'] = {level: lr[level] for level in ('macro', 'per_class')}
            values['ft_minus_lr'] = difference(values['finetuned'], lr)
            folds.append(dict(fold=fold, counts=f['counts'], seeds=seeds, seed_summaries=summaries,
                common_lr=dict(metrics=lr, fit=fit_record(lrr[model, fold], cs), manifest_sha256=lrr[model, fold]['manifest_sha256']),
                values=values))
        aggregate = {name: {level: {m: joint.describe([f['values'][name][level][m] for f in folds])
                      for m in common.METRICS} for level in ('macro', 'per_class')} for name in GROUPS}
        result['models'][model] = dict(folds=folds, summary=aggregate)
    return result


def publish_report(output, encoded):
    """Publish closed, complete bytes exclusively; clean only our private file."""
    temporary = None
    receipt = dict(report_written=False)
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                prefix='.' + output.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            if stream.write(encoded) != len(encoded):
                raise OSError('Short report write')
        os.link(temporary, output)
        receipt['report_written'] = True
    except OSError as error:
        receipt['error'] = str(error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError as error:
                receipt.update(cleanup_error=str(error), temporary_remaining=str(temporary),
                               publication_complete=receipt['report_written'])
                if not receipt['report_written']:
                    # A private artifact remains; do not claim a clean no-output failure.
                    del receipt['report_written']
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ft-audit', 'replay-audit', 'frozen-audit', 'common-audit', 'ft-protocol', 'replay-protocol', 'common-protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('ft-audit-sha256', 'replay-audit-sha256'):
        parser.add_argument('--' + name, required=True, help='Digest from independently accepted complete audit')
    parser.add_argument('--output', type=Path, required=True, help='New JSON outside source and input directories')
    args = parser.parse_args(argv)
    try:
        paths = {name: getattr(args, name) for name in (*PINS, 'ft_audit', 'replay_audit')}
        expected = {**PINS, 'ft_audit': args.ft_audit_sha256, 'replay_audit': args.replay_audit_sha256}
        source = {str(Path(m.__file__).resolve().relative_to(ROOT)): common.sha256(m.__file__)
                  for m in (sys.modules[__name__], joint, native, common)}
        require(all(re.fullmatch('[0-9a-f]{64}', h) for h in expected.values()), 'Expected accepted SHA256 values')
        output = args.output.resolve()
        require(not args.output.exists() and not any(p.is_symlink() for p in (args.output, *args.output.parents)), 'Output must be new')
        require(not any(output.is_relative_to(p) for p in (ROOT, *(p.resolve().parent for p in paths.values()))),
                'Output overlaps source or inputs')
        data = {}
        for name, path in paths.items():
            require(common.sha256(path) == expected[name], 'Input SHA256 differs: ' + name)
            data[name] = common.read_json(path)
        result = analyze(data)
        result['inputs'] = {name: dict(path=str(path.resolve()), sha256=expected[name]) for name, path in paths.items()}
        result['analysis_source_sha256'] = source
        result['analysis_runtime'] = dict(python=sys.version, numpy=np.__version__)
        for name, path in paths.items():
            require(common.sha256(path) == expected[name], 'Input changed during analysis: ' + name)
        for name, value in source.items():
            require(common.sha256(ROOT / name) == value, 'Analysis source changed: ' + name)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps(dict(status='invalid', error=str(error), report_written=False, output_exists=args.output.exists())))
        return 1
    publication = publish_report(output, encoded)
    complete = publication.get('report_written') is True and not any(k in publication for k in ('error', 'cleanup_error'))
    diagnostic = dict(status='complete' if complete else 'publication_failed', output=str(output),
                      output_exists=output.exists(), **publication)
    if complete:
        diagnostic.update(ft_jobs=75, checkpoint_replays=75)
    try:
        print(json.dumps(diagnostic), flush=True)
    except OSError as error:
        diagnostic.update(status='diagnostic_failed', diagnostic_error=str(error))
        try:
            print(json.dumps(diagnostic), file=sys.stderr, flush=True)
        except OSError:
            pass  # Both diagnostic streams failed; retain the report and fail the command.
        return 1
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
