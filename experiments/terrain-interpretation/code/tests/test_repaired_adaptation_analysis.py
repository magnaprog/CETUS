"""Synthetic acceptance and arithmetic; no scientific outcomes or model runs."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import analyze_repaired_adaptation as analysis


def metric(support, shift):
    counts = np.zeros((6, 6), dtype=int)
    for c, n in enumerate(support):
        missed = min(n, (shift + c) % 5)
        counts[c, c] = n - missed
        counts[c, (c + 1) % 6] = missed
    return dict(analysis.common.metrics_from_confusion(counts, list(range(6))),
                confusion_counts=counts.tolist(), kappa=analysis.native.kappa_from_counts(counts))


@pytest.fixture
def reports():
    tag = lambda value: analysis.digest(value)
    classes = list(analysis.common.CLASS_NAMES)
    encoders = {m: dict(model_revision='revision-' + m, weights_sha256=tag(m)) for m in analysis.MODELS}
    helpers = {'scripts/summarize_repaired_finetuning.py': tag('ft checker'),
               'scripts/summarize_review_frozen.py': tag('metric helper'),
               'scripts/replay_repaired_checkpoints.py': tag('replay runner')}
    fs = dict(class_names=classes, catalog_sha256=tag('catalog'), frozen_source_commit=analysis.native.SOURCE_COMMIT,
              verified_frozen_audit_sha256=analysis.PINS['frozen_audit'], encoders=encoders,
              folds=[], frozen_jobs=[], recipe=dict(epochs=50, unfreeze_blocks=2),
              scientific_source_sha256={'scripts/summarize_review_frozen.py': helpers['scripts/summarize_review_frozen.py']})
    cs = dict(class_names=classes, catalog=dict(sha256=tag('catalog')), folds=[], caches=[],
              native_audit=dict(sha256=analysis.PINS['frozen_audit']),
              resolved_parameters=dict(classifier={'C': 1., 'max_iter': 2000}, scaler={'with_mean': True}),
              scientific_source_sha256={'scripts/summarize_repaired_common_classifier.py': tag('lr checker')})
    rs = dict(producer_commit=analysis.FT_COMMIT, training_protocol_sha256=analysis.PINS['ft_protocol'],
              verified_frozen_audit_sha256=analysis.PINS['frozen_audit'],
              replay_source_sha256=helpers, producer_plan_sha256=analysis.FT_PLAN_SHA,
              inference=dict(cpu_intraop_threads=4),
              execution=dict(thread_environment=dict(
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')))
    frozen = dict(source_commit=analysis.native.SOURCE_COMMIT, validation=dict(status='pass', probe_heads=100),
                  catalog_sha256=tag('catalog'), class_names=classes, jobs=[], folds=[], models={})
    lr = dict(status='complete', producer_commit=analysis.joint.COMMITS['common'],
              protocol_sha256=analysis.PINS['common_protocol'], plan_sha256=analysis.joint.PLANS['common'],
              native_audit_sha256=analysis.PINS['frozen_audit'],
              validation=dict(complete_jobs=20, estimator_replays=20, output_hashes=60),
              auditor_sha256=cs['scientific_source_sha256']['scripts/summarize_repaired_common_classifier.py'], folds=[])
    for f in range(5):
        support = [6 + 9 * f, 5 + f, 4 + f, 3 + f, 2 + f, 1 if f == 1 else 3]
        counts = {r: dict(class_counts=v, total=sum(v)) for r, v in
                  [('train', [100 + f] * 6), ('test', support), ('val', [5 + f] * 5 + [1])]}
        ids = [f'fold-{f}-test-{i}' for i in range(sum(support))]
        norm = dict(enabled=True, fallback_used=False, training_ids_sha256=tag(['train', f]), lower=1., upper=200.)
        ff = dict(fold=f, counts=counts, split_sha256=tag(['split', f]), normalization_sha256=tag(norm),
                  optimizer_attempts_per_epoch=(counts['train']['total'] + 31) // 32)
        fs['folds'].append(ff)
        cs['folds'].append(dict(fold=f, counts={r: counts[r] for r in ('train', 'test')}, split=dict(sha256=ff['split_sha256']),
                              normalization_sha256=tag(norm), normalization_provenance=norm,
                              train_ids_sha256=norm['training_ids_sha256'], test_ids_sha256=analysis.native.ids_hash(ids)))
        frozen['folds'].append(dict(fold=f, counts=counts, split_sha256=ff['split_sha256'], test_ids=ids, normalization_provenance=norm))
    for m in analysis.common.MODELS:
        frozen['models'][m] = dict(folds=[])
        for f in range(5):
            ref = dict(model=m, fold=f, extraction_manifest_sha256=tag([m, f, 'features']),
                       analysis_manifest_sha256=tag([m, f, 'analysis']), result_sha256=tag([m, f, 'result']))
            frozen['jobs'].append(ref)
            if m in analysis.MODELS:
                fs['frozen_jobs'].append(copy.deepcopy(ref))
            cs['caches'].append(dict(model=m, fold=f, manifest_sha256=ref['extraction_manifest_sha256'],
                                     encoder_identity=encoders.get(m, {})))
            support = fs['folds'][f]['counts']['test']['class_counts']
            frozen['models'][m]['folds'].append(dict(fold=f, probe_heads=[dict(seed=s, **metric(support, s + f + 1)) for s in range(5)]))
            fit = dict(n_iter=[2000 if f == 4 else 12], iteration_limit_reached=f == 4, convergence_warning=f == 4,
                       fit_warnings=[dict(category='ConvergenceWarning', message='iteration cap')] if f == 4 else [],
                       classifier_parameters=cs['resolved_parameters']['classifier'], scaler_parameters=cs['resolved_parameters']['scaler'])
            lr['folds'].append(dict(model=m, fold=f, counts=cs['folds'][f]['counts'], fit=fit,
                                    estimator_replay=dict(verified=True), manifest_sha256=tag([m, f, 'lr']), **metric(support, 3)))
    fp = dict(specification=fs, specification_sha256=tag(fs))
    cp = dict(specification=cs, specification_sha256=tag(cs))
    rp = dict(specification=rs, specification_sha256=tag(rs))
    ft = dict(status='pass', validated_jobs=75, source_commit=analysis.FT_COMMIT,
              protocol_sha256=analysis.PINS['ft_protocol'], specification_sha256=fp['specification_sha256'],
              frozen_audit_sha256=analysis.PINS['frozen_audit'], auditor_sha256=helpers['scripts/summarize_repaired_finetuning.py'],
              helper_sha256=helpers['scripts/summarize_review_frozen.py'], class_names=classes, records=[])
    replay = dict(status='pass', verified_replays=75, replay_execution_commit=analysis.REPLAY_COMMIT,
                  replay_protocol_sha256=analysis.PINS['replay_protocol'], parent_plan_sha256=analysis.FT_PLAN_SHA,
                  plan_sha256=analysis.REPLAY_PLAN_SHA, checker_source=dict(commit=analysis.REPLAY_COMMIT, clean=True, status=[]),
                  checker_sha256=analysis.REPLAY_CHECKER_SHA, checker_helpers_sha256=helpers, jobs=[])
    for m, f, s in sorted(analysis.JOBS):
        fold = fs['folds'][f]
        attempts = fold['optimizer_attempts_per_epoch']
        history = dict(selected_epoch=1 + f + s, epoch_indexing='one based', optimizer_attempts_per_epoch=attempts,
                       optimizer_steps_total=50 * attempts - s, amp_skipped_steps_total=s,
                       training_example_presentations=50 * fold['counts']['train']['total'])
        ref = next(r for r in fs['frozen_jobs'] if (r['model'], r['fold']) == (m, f))
        provenance = dict(**encoders[m], titan_catalog_sha256=fs['catalog_sha256'], titan_split_manifest_sha256=fold['split_sha256'],
            fold=f, seed=s, recipe=fs['recipe'], class_counts=fold['counts'], protocol_sha256=analysis.PINS['ft_protocol'],
            protocol_specification_sha256=fp['specification_sha256'], normalization_provenance=frozen['folds'][f]['normalization_provenance'],
            frozen_reference={**ref, 'source_commit': analysis.native.SOURCE_COMMIT, 'matched_head_seed': s, 'validated_head_seeds': list(range(5))},
            dofa_band_identifier_ghz=13.78 if m == 'dofa' else None, evaluation_scope='within_titan_only', **history)
        metrics = metric(fold['counts']['test']['class_counts'], 2 * s // 3 + f)
        ft['records'].append(dict(model=m, fold=f, seed=s, provenance=provenance, **history,
            manifest_sha256=tag([m, f, s, 'ft']), finetuned=metrics,
            outputs={'finetuning_results.json': tag([m, f, s, 'result']),
                     f'checkpoints/{m}/ft_2block_seed_{s}.pt': tag([m, f, s, 'checkpoint'])},
            frozen=copy.deepcopy(frozen['models'][m]['folds'][f]['probe_heads'][s])))
        val = metric(fold['counts']['val']['class_counts'], 2)
        val.pop('kappa')  # The replay checker stores checked_predictions for validation.
        replay['jobs'].append(dict(model=m, fold=f, seed=s, producer_manifest_sha256=tag([m, f, s, 'ft']),
            replay_manifest_sha256=tag([m, f, s, 'replay']), replay_sha256=tag([m, f, s, 'json']),
            selected_epoch=history['selected_epoch'], test_metrics=copy.deepcopy(metrics), validation_metrics=val,
            comparisons=dict(validation=dict(passed=True, historical_vector_available=False, macro_recall_abs_error=0.,
                                              expected_macro_recall=val['macro']['recall']),
                             test=dict(passed=True, confusion_exact=True, prediction_mismatches=0, mismatched_tile_ids=[],
                                       probability_components_over_atol=0, probability_max_abs_error=0.))))
    return dict(ft_audit=ft, replay_audit=replay, common_audit=lr, frozen_audit=frozen,
                ft_protocol=fp, common_protocol=cp, replay_protocol=rp)


def test_full_comparison_matches_independent_confusion_and_nested_arithmetic(reports):
    before = analysis.digest(reports)
    result = analysis.analyze(reports)
    assert analysis.digest(reports) == before
    model = result['models']['dofa']
    fold_values, paired_values, lr_values = [], [], []
    for f in range(5):
        rows = [r for r in reports['ft_audit']['records'] if r['model'] == 'dofa' and r['fold'] == f]
        # Direct Python arithmetic, independent of both imported metric helpers.
        def f1(record):
            cm = record['confusion_counts']
            return sum(2 * cm[c][c] / (sum(cm[c]) + sum(r[c] for r in cm)) for c in range(6)) / 6
        a, s = [f1(r['finetuned']) for r in rows], [f1(r['frozen']) for r in rows]
        l = next(r for r in reports['common_audit']['folds'] if (r['model'], r['fold']) == ('dofa', f))
        fold_values.append(sum(a) / 5)
        paired_values.append(sum(100 * (x - y) for x, y in zip(a, s)) / 5)
        lr_values.append(100 * (sum(a) / 5 - f1(l)))
        assert len(model['folds'][f]['seeds']) == 5
        np.testing.assert_allclose(model['folds'][f]['seed_summaries']['finetuned']['macro']['f1']['sample_std'], np.std(a, ddof=1))
        differences = [100 * (x - y) for x, y in zip(a, s)]
        np.testing.assert_allclose([r['ft_minus_sgd']['macro']['f1'] for r in model['folds'][f]['seeds']], differences, atol=1e-12)
        assert model['folds'][f]['seed_summaries']['ft_minus_sgd']['macro']['f1']['sample_std'] == pytest.approx(np.std(differences, ddof=1))
    for name, expected in [('finetuned', fold_values), ('ft_minus_sgd', paired_values), ('ft_minus_lr', lr_values)]:
        actual = model['summary'][name]['macro']['f1']
        np.testing.assert_allclose(actual['fold_values'], expected, atol=1e-12)
        assert actual['mean'] == pytest.approx(sum(expected) / 5)
        assert actual['sample_std'] == pytest.approx(np.std(expected, ddof=1))
    weights = [sum(f['counts']['test']['class_counts']) for f in model['folds']]
    assert not np.isclose(np.average(fold_values, weights=weights), np.mean(fold_values))
    assert 'common_lr' not in model['folds'][0]['seed_summaries']
    assert model['folds'][4]['common_lr']['fit']['convergence_warning'] is True
    assert model['folds'][1]['counts']['test']['class_counts'][-1] == 1
    assert sum(len(v['folds']) for v in result['models'].values()) == 15


@pytest.mark.parametrize('name,field', [('ft_audit', 'records'), ('replay_audit', 'jobs'), ('common_audit', 'folds')])
@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'extra', 'bool'])
def test_complete_identity_gate_precedes_any_metrics(reports, monkeypatch, name, field, defect):
    rows = reports[name][field]
    if defect == 'missing': rows.pop()
    if defect == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    if defect == 'extra': rows.append(copy.deepcopy(rows[0]))
    if defect == 'bool': rows[0]['fold'] = False
    monkeypatch.setattr(analysis.joint, 'reconstruct', lambda *a: pytest.fail('Metrics read before gate'))
    with pytest.raises(ValueError): analysis.analyze(reports)


@pytest.mark.parametrize('defect', ['ft_status', 'replay_status', 'ft_source', 'replay_source', 'checker', 'plan',
    'specification', 'cache', 'split', 'normalization', 'common_source', 'class_order', 'producer_manifest',
    'selected_epoch', 'comparison', 'empty_comparisons', 'lr_replay', 'sgd_seed', 'ft_metric', 'replay_confusion',
    'test_support', 'step_budget', 'bool_steps', 'provenance', 'warning', 'cap', 'classifier', 'checkpoint_inventory', 'digest'])
def test_corrupt_or_inconsistent_accepted_records_rejected(reports, defect):
    ft, replay, lr = reports['ft_audit'], reports['replay_audit'], reports['common_audit']
    row, rr = ft['records'][0], replay['jobs'][0]
    fs, cs = reports['ft_protocol']['specification'], reports['common_protocol']['specification']
    if defect == 'ft_status': ft['status'] = 'incomplete'
    if defect == 'replay_status': replay['status'] = 'incomplete'
    if defect == 'ft_source': ft['source_commit'] = '0' * 40
    if defect == 'replay_source': replay['checker_source']['clean'] = False
    if defect == 'checker': replay['checker_sha256'] = '0' * 64
    if defect == 'plan': replay['parent_plan_sha256'] = '0' * 64
    if defect == 'specification': fs['recipe']['epochs'] = 49
    if defect == 'cache': cs['caches'][0]['manifest_sha256'] = '0' * 64
    if defect == 'split': cs['folds'][0]['split']['sha256'] = '0' * 64
    if defect == 'normalization': cs['folds'][0]['normalization_sha256'] = '0' * 64
    if defect == 'common_source': lr['producer_commit'] = '0' * 40
    if defect == 'class_order': ft['class_names'] = list(reversed(ft['class_names']))
    if defect == 'producer_manifest': rr['producer_manifest_sha256'] = '0' * 64
    if defect == 'selected_epoch': rr['selected_epoch'] = 49
    if defect == 'comparison': rr['comparisons']['test']['passed'] = False
    if defect == 'empty_comparisons': rr['comparisons'] = {}
    if defect == 'lr_replay': lr['folds'][0]['estimator_replay']['verified'] = False
    if defect == 'sgd_seed': row['frozen'] = metric(row['frozen']['support'], 4)
    if defect == 'ft_metric': row['finetuned']['macro']['f1'] += .1
    if defect == 'replay_confusion': rr['test_metrics'] = metric(rr['test_metrics']['support'], 4)
    if defect == 'test_support': row['finetuned']['support'][0] += 1
    if defect in ('step_budget', 'bool_steps'):
        row['optimizer_steps_total'] = True if defect == 'bool_steps' else 0
        row['provenance']['optimizer_steps_total'] = row['optimizer_steps_total']
    if defect == 'provenance': row['provenance']['weights_sha256'] = '0' * 64
    if defect == 'checkpoint_inventory': row['outputs'].pop(next(k for k in row['outputs'] if k.startswith('checkpoints/')))
    if defect == 'digest': rr['replay_sha256'] = 'unchecked'
    if defect == 'warning': lr['folds'][0]['fit']['convergence_warning'] = True
    if defect == 'cap': lr['folds'][0]['fit']['iteration_limit_reached'] = True
    if defect == 'classifier':
        lr['folds'][0]['fit']['classifier_parameters'] = dict(C=2, max_iter=2000)
    # Keep protocol hashes internally consistent for semantic binding defects.
    if defect in ('cache', 'split', 'normalization'):
        reports['common_protocol']['specification_sha256'] = analysis.digest(cs)
    with pytest.raises(ValueError): analysis.analyze(reports)


def test_training_totals_and_zero_prediction_precision(reports):
    result = analysis.analyze(reports)
    seed = result['models']['dinov2']['folds'][2]['seeds'][4]
    h = seed['training']
    assert h['amp_skipped_steps_total'] == 4
    assert h['optimizer_attempts_total'] == h['optimizer_steps_total'] + 4
    assert h['selected_epoch'] == 7
    cm = np.zeros((6, 6), dtype=int)
    cm[:, 0] = [10, 2, 2, 2, 2, 1]
    record = dict(analysis.common.metrics_from_confusion(cm, list(range(6))), confusion_counts=cm.tolist(),
                  kappa=analysis.native.kappa_from_counts(cm))
    checked = analysis.joint.reconstruct(record, cm.sum(1).tolist())
    assert checked['per_class']['precision'] == [10 / 19, 0, 0, 0, 0, 0]
    assert checked['macro']['f1'] == pytest.approx((20 / 29) / 6)
    assert checked['false_positive'] == [9, 0, 0, 0, 0, 0]


@pytest.fixture
def command_files(reports, tmp_path, monkeypatch):
    directory = tmp_path / 'inputs'
    directory.mkdir()
    paths = {}
    for name, value in reports.items():
        path = directory / (name + '.json')
        path.write_text(json.dumps(value))
        paths[name] = path
    # CLI hash verification and protocol cross-binding use the same production
    # constants. Tests replace the five fixed hashes with the fixture bytes.
    # The in-memory records keep production parent pins; restore them only in
    # the analyze call so semantic tests above remain independent of this seam.
    real_pins = dict(analysis.PINS)
    monkeypatch.setattr(analysis, 'PINS', {k: analysis.common.sha256(paths[k]) for k in real_pins})
    original = analysis.analyze
    def analyze_fixture(data):
        fixture_pins = analysis.PINS
        analysis.PINS = real_pins
        try: return original(data)
        finally: analysis.PINS = fixture_pins
    monkeypatch.setattr(analysis, 'analyze', analyze_fixture)
    args = []
    for name, path in paths.items(): args.extend(['--' + name.replace('_', '-'), str(path)])
    for name in ('ft_audit', 'replay_audit'):
        args.extend(['--' + name.replace('_', '-') + '-sha256', analysis.common.sha256(paths[name])])
    return paths, args, tmp_path / 'result.json'


def test_cli_deterministic_fresh_output_and_input_preservation(command_files):
    paths, args, output = command_files
    before = {k: analysis.common.sha256(p) for k, p in paths.items()}
    assert analysis.main([*args, '--output', str(output)]) == 0
    repeat = output.with_name('repeat.json')
    assert analysis.main([*args, '--output', str(repeat)]) == 0
    assert output.read_bytes() == repeat.read_bytes()
    assert before == {k: analysis.common.sha256(p) for k, p in paths.items()}
    assert analysis.main([*args, '--output', str(output)]) == 1
    assert output.read_bytes() == repeat.read_bytes()
    saved = json.loads(output.read_text())
    assert saved['validation']['ft_jobs'] == saved['validation']['checkpoint_replays'] == 75
    assert len(saved['analysis_source_sha256']) == 4
    assert saved['analysis_runtime'] == dict(python=sys.version, numpy=np.__version__)


@pytest.mark.parametrize('defect', ['bytes', 'pending', 'output_overlap', 'source_changed', 'input_changed'])
def test_cli_failure_never_writes_output(command_files, monkeypatch, defect):
    paths, args, output = command_files
    if defect == 'bytes': paths['ft_audit'].write_text('{}')
    if defect == 'pending':
        value = json.loads(paths['ft_audit'].read_text()); value['records'].pop()
        paths['ft_audit'].write_text(json.dumps(value))
        args[args.index('--ft-audit-sha256') + 1] = analysis.common.sha256(paths['ft_audit'])
    if defect == 'output_overlap': output = paths['ft_audit'].parent / 'new.json'
    if defect in ('source_changed', 'input_changed'):
        original = analysis.analyze
        def changed(data):
            result = original(data)
            if defect == 'input_changed': paths['ft_audit'].write_text('{}')
            else:
                sha = analysis.common.sha256
                monkeypatch.setattr(analysis.common, 'sha256', lambda p: '0' * 64 if Path(p) == Path(analysis.__file__) else sha(p))
            return result
        monkeypatch.setattr(analysis, 'analyze', changed)
    assert analysis.main([*args, '--output', str(output)]) == 1
    assert not output.exists()


def test_import_uses_no_training_or_gpu_packages():
    code = "import sys; from scripts import analyze_repaired_adaptation; assert not ({'torch', 'sklearn', 'joblib', 'timm'} & set(sys.modules))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=analysis.ROOT, check=True)


@pytest.mark.parametrize('failure', ['prefix_error', 'short_write', 'close_error'])
def test_publication_stream_failure_cleans_only_private_file(command_files, monkeypatch, capsys, failure):
    paths, args, output = command_files
    before = {k: analysis.common.sha256(p) for k, p in paths.items()}
    unrelated = output.with_name('.result.json.another-writer.tmp')
    unrelated.write_bytes(b'preserve another writer')
    factory = analysis.tempfile.NamedTemporaryFile
    created = []

    class FailingStream:
        def __init__(self, stream):
            self.stream, self.name = stream, stream.name
            created.append(Path(self.name))

        def __enter__(self):
            return self

        def write(self, encoded):
            assert not output.exists()
            if failure == 'close_error':
                return self.stream.write(encoded)
            self.stream.write(encoded[:113])
            self.stream.flush()
            if failure == 'prefix_error':
                raise OSError('disk full after prefix')
            return 113

        def __exit__(self, *error):
            self.stream.__exit__(*error)
            assert not output.exists()
            if failure == 'close_error':
                raise OSError('close failed')

    monkeypatch.setattr(analysis.tempfile, 'NamedTemporaryFile', lambda *a, **kw: FailingStream(factory(*a, **kw)))
    assert analysis.main([*args, '--output', str(output)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'publication_failed' and receipt['report_written'] is False
    assert not receipt['output_exists'] and not output.exists()
    assert len(created) == 1 and not created[0].exists()
    assert unrelated.read_bytes() == b'preserve another writer'
    assert before == {k: analysis.common.sha256(p) for k, p in paths.items()}


def test_publication_race_preserves_existing_destination(command_files, monkeypatch, capsys):
    _, args, output = command_files
    link = analysis.os.link
    created = []

    def competing_destination(source, destination):
        created.append(source)
        staged = json.loads(source.read_text())
        assert staged['validation']['ft_jobs'] == 75
        assert staged['analysis_runtime'] == dict(python=sys.version, numpy=np.__version__)
        destination.write_bytes(b'existing external output')
        link(source, destination)

    monkeypatch.setattr(analysis.os, 'link', competing_destination)
    assert analysis.main([*args, '--output', str(output)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'publication_failed' and receipt['report_written'] is False
    assert receipt['output_exists'] is True
    assert output.read_bytes() == b'existing external output'
    assert len(created) == 1 and not created[0].exists()


def test_existing_destination_is_never_overwritten_by_publisher(tmp_path):
    output = tmp_path / 'existing.json'
    output.write_bytes(b'original')
    receipt = analysis.publish_report(output, '{"replacement": true}\n')
    assert receipt['report_written'] is False and 'error' in receipt
    assert output.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [output]


def test_exclusive_link_failure_cleans_stage_without_creating_output(tmp_path, monkeypatch):
    output = tmp_path / 'report.json'
    def fail(*args):
        raise OSError('link unavailable')
    monkeypatch.setattr(analysis.os, 'link', fail)
    receipt = analysis.publish_report(output, '{"complete": true}\n')
    assert receipt == dict(report_written=False, error='link unavailable')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('published', [False, True])
def test_cleanup_failure_records_ownership_and_publication_state(tmp_path, monkeypatch, published):
    output = tmp_path / 'report.json'
    unlink = Path.unlink
    def fail_unlink(path, *args, **kwargs):
        raise OSError('cleanup denied')
    def fail_link(*args):
        raise OSError('link denied')
    monkeypatch.setattr(Path, 'unlink', fail_unlink)
    if not published:
        monkeypatch.setattr(analysis.os, 'link', fail_link)
    encoded = '{"complete": true}\n'
    receipt = analysis.publish_report(output, encoded)
    temporary = Path(receipt['temporary_remaining'])
    assert receipt['cleanup_error'] == 'cleanup denied'
    assert receipt['publication_complete'] is published
    assert receipt.get('report_written') is (True if published else None)
    assert temporary.parent == tmp_path and temporary != output
    assert temporary.read_text() == encoded
    assert output.exists() is published
    if published:
        assert output.read_text() == encoded
    unlink(temporary)  # Test cleanup uses the saved original operation.


@pytest.mark.parametrize('both_streams_fail', [False, True])
def test_postpublication_diagnostic_failure_preserves_report(command_files, monkeypatch, capsys, both_streams_fail):
    import builtins
    _, args, output = command_files
    calls = []
    def failed_diagnostic(message, *args, **kwargs):
        calls.append(json.loads(message))
        if both_streams_fail or kwargs.get('file') is not sys.stderr:
            raise OSError('diagnostic stream unavailable')
        return builtins.print(message, *args, **kwargs)
    monkeypatch.setattr(analysis, 'print', failed_diagnostic, raising=False)
    assert analysis.main([*args, '--output', str(output)]) == 1
    captured = capsys.readouterr()
    saved = json.loads(output.read_text())
    assert saved['status'] == 'complete' and saved['validation']['ft_jobs'] == 75
    assert len(calls) == 2 and all(c['report_written'] is True for c in calls)
    assert calls[-1]['status'] == 'diagnostic_failed'
    if not both_streams_fail:
        assert json.loads(captured.err)['report_written'] is True
    assert not list(output.parent.glob('.' + output.name + '.*.tmp'))


def test_recovered_replay_uses_its_own_plan_and_keeps_original_control_plan():
    assert analysis.REPLAY_COMMIT == '97d4c5cc50a63b0bc9f2b6c09f48ed8d88fad94e'
    assert analysis.REPLAY_CHECKER_SHA == (
        '94458b9c0f7327a1981c66ab818b504d37332afd56ce61f17b3cf2c49c88a9b5'
    )
    assert analysis.PINS['replay_protocol'] == (
        'b55152cd387cc868eec9a39f4bfc621b8c9fdb169f307008e5a4ca4af0a9e1ca'
    )
    assert analysis.REPLAY_PLAN_SHA == (
        '94510a4141dcbf42b17438ce6eda01ffb458f3f2c40ec986e0fdc0dc388b1d60'
    )
    assert analysis.FOLLOWUP_PLAN_SHA == (
        'baaa83961f896523ad2edf711ccc166edb9bf14249a2e68f85321c2bc6b92d11'
    )
    assert len({analysis.FT_PLAN_SHA, analysis.FOLLOWUP_PLAN_SHA,
                analysis.REPLAY_PLAN_SHA}) == 3


@pytest.mark.parametrize('defect', [
    'old_protocol', 'old_execution', 'old_checker_commit', 'old_checker_hash',
    'old_followup_as_replay', 'ft_plan_as_replay', 'replay_plan_as_parent',
    'count_only_duplicate', 'wrong_producer_plan', 'one_thread',
    'boolean_thread', 'thread_environment', 'missing_thread_environment',
])
def test_recovery_contract_fails_before_metrics(reports, monkeypatch, defect):
    replay = reports['replay_audit']
    spec = reports['replay_protocol']['specification']
    if defect == 'old_protocol':
        replay['replay_protocol_sha256'] = (
            '053c1b2272405a13a1e45e341944d3560dcd543c78c3f88492591d832339791a'
        )
    elif defect == 'old_execution':
        replay['replay_execution_commit'] = (
            '54281f05cfd67453f9b699810021b1e2da20c3ec'
        )
    elif defect == 'old_checker_commit':
        replay['checker_source']['commit'] = (
            '54281f05cfd67453f9b699810021b1e2da20c3ec'
        )
    elif defect == 'old_checker_hash':
        replay['checker_sha256'] = (
            'a71b1d89705dd349527ea97003b3bc27814d9dada315f90ac708ac30ffc55dc5'
        )
    elif defect == 'old_followup_as_replay':
        replay['plan_sha256'] = analysis.FOLLOWUP_PLAN_SHA
    elif defect == 'ft_plan_as_replay':
        replay['plan_sha256'] = analysis.FT_PLAN_SHA
    elif defect == 'replay_plan_as_parent':
        replay['parent_plan_sha256'] = analysis.REPLAY_PLAN_SHA
    elif defect == 'count_only_duplicate':
        replay['jobs'][-1] = copy.deepcopy(replay['jobs'][0])
    elif defect == 'wrong_producer_plan':
        spec['producer_plan_sha256'] = analysis.REPLAY_PLAN_SHA
    elif defect == 'one_thread':
        spec['inference']['cpu_intraop_threads'] = 1
    elif defect == 'boolean_thread':
        spec['inference']['cpu_intraop_threads'] = True
    elif defect == 'thread_environment':
        spec['execution']['thread_environment']['MKL_NUM_THREADS'] = '1'
    else:
        del spec['execution']['thread_environment']
    # A count of 75 cannot establish the identity of either same-sized cohort.
    assert reports['ft_audit']['validated_jobs'] == replay['verified_replays'] == 75
    assert len(reports['ft_audit']['records']) == len(replay['jobs']) == 75
    reports['replay_protocol']['specification_sha256'] = analysis.digest(spec)
    monkeypatch.setattr(
        analysis.joint, 'reconstruct',
        lambda *args: pytest.fail('Outcome metrics reached before recovery gate'),
    )
    with pytest.raises((ValueError, KeyError)):
        analysis.analyze(reports)
