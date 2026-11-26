"""Synthetic display contracts and artifacts; never read scientific outcomes."""

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import plot_repaired_adaptation as plots
from tests.test_repaired_adaptation_analysis import reports


@pytest.fixture
def accepted_analysis(reports):
    report = plots.analysis.analyze(reports)
    report['inputs'] = {k: dict(path='synthetic/' + k, sha256=plots.analysis.digest(k))
                        for k in (*plots.analysis.PINS, 'ft_audit', 'replay_audit')}
    report['analysis_source_sha256'] = {str(Path(m.__file__).resolve().relative_to(plots.ROOT)): plots.common.sha256(m.__file__)
                                      for m in (plots.analysis, plots.joint, plots.common, plots.analysis.native)}
    report['analysis_runtime'] = dict(python=sys.version, numpy=np.__version__)
    return report


def test_absolute_scores_convert_once_and_support_does_not_count_seeds(accepted_analysis):
    values = plots.display_values(accepted_analysis)
    assert [r['model'] for r in values['macro_rows']] == list(plots.NAMES)
    assert len(values['class_rows']) == 18 and len(values['training_rows']) == 75
    for model, name in zip(plots.MODELS, plots.NAMES):
        original = accepted_analysis['models'][model]
        macro = next(r for r in values['macro_rows'] if r['model'] == name)
        for field, metric in [('ft_f1', 'f1'), ('ft_recall', 'recall')]:
            assert macro[field + '_mean'] == 100 * original['summary']['finetuned']['macro'][metric]['mean']
            assert macro[field + '_sample_std'] == 100 * original['summary']['finetuned']['macro'][metric]['sample_std']
        rows = [r for r in values['class_rows'] if r['model'] == name]
        assert [r['terrain_class'] for r in rows] == list(plots.CLASSES)
        for c, row in enumerate(rows):
            support = [f['counts']['test']['class_counts'][c] for f in original['folds']]
            assert row['test_support'] == sum(support)
            assert [row[f'support_fold_{f}'] for f in range(5)] == support
            for prefix, group in [('ft', 'finetuned'), ('lr', 'common_lr')]:
                for metric in plots.METRICS:
                    assert row[f'{prefix}_{metric}_mean'] == 100 * original['summary'][group]['per_class'][metric]['mean'][c]
        training = next(t for t in values['training_summary'] if t['model'] == name)
        assert training['jobs'] == 25 and training['selected_epoch_min'] == 1 and training['selected_epoch_max'] == 9
        assert training['optimizer_attempts_total'] == training['optimizer_steps_total'] + training['amp_skipped_steps_total']
        assert training['amp_skipped_steps_total'] == 50


def test_signed_differences_remain_pp_and_use_sample_fold_sd(accepted_analysis):
    from statistics import mean, stdev
    points = [-2., 4., -6., 8., -10.]
    for model in plots.MODELS:
        block = accepted_analysis['models'][model]
        for ref, sign in [('ft_minus_sgd', 1), ('ft_minus_lr', -1)]:
            for metric in ('f1', 'recall'):
                data = [sign * x for x in points]
                block['summary'][ref]['macro'][metric] = dict(fold_values=data, mean=mean(data), sample_std=stdev(data))
                for f, value in zip(block['folds'], data):
                    f['values'][ref]['macro'][metric] = value
    display = plots.display_values(accepted_analysis)
    for model in plots.MODELS:
        for ref, sign in [('ft_minus_sgd', 1), ('ft_minus_lr', -1)]:
            for metric in ('f1', 'recall'):
                row = display['models'][model]['summary'][ref]['macro'][metric]
                assert row['fold_values'] == [sign * x for x in points]
                assert row['mean'] == pytest.approx(sign * mean(points))
                assert row['sample_std'] == pytest.approx(stdev(points))
                assert row['sample_std'] != pytest.approx(np.std(points, ddof=0))


def test_output_orders_are_fixed_after_input_reordering(accepted_analysis):
    expected = plots.display_values(accepted_analysis)
    reordered = copy.deepcopy(accepted_analysis)
    reordered['models'] = dict(reversed(list(reordered['models'].items())))
    for model in reordered['models'].values():
        model['folds'].reverse()
        for fold in model['folds']:
            fold['seeds'].reverse()
    assert plots.display_values(reordered) == expected


def test_tex_rounding_hides_only_negative_zero_and_preserves_exact_data(accepted_analysis, tmp_path):
    display = plots.display_values(accepted_analysis)
    display['macro_rows'][0]['ft_minus_lr_f1_mean'] = -0.004
    display['macro_rows'][1]['ft_minus_lr_f1_mean'] = -0.006
    before = json.dumps(display, sort_keys=True)
    plots.table_files(display, tmp_path)
    text = (tmp_path / 'repaired_adaptation_macro_20260930.tex').read_text()
    assert '-0.00' not in text and '$0.00 \\pm' in text and '$-0.01 \\pm' in text
    assert plots.tex_number(-0.) == '0.00'
    assert plots.tex_number(-12.345) == '-12.35'
    assert json.dumps(display, sort_keys=True) == before
    with (tmp_path / 'repaired_adaptation_macro_20260930.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]['ft_minus_lr_f1_mean']) == -0.004
    assert float(rows[1]['ft_minus_lr_f1_mean']) == -0.006


@pytest.mark.parametrize('defect', ['status', 'schema', 'ft_count', 'replay_count', 'model', 'class', 'fold', 'seed',
    'bool_fold', 'support', 'mean', 'population_sd', 'fold_value', 'source', 'runtime', 'nonfinite'])
def test_minimum_display_contract_rejects_incomplete_or_inconsistent_analysis(accepted_analysis, defect):
    r = accepted_analysis
    block = r['models']['dinov2']
    if defect == 'status': r['status'] = 'incomplete'
    if defect == 'schema': r['schema_version'] = '2.0.0'
    if defect == 'ft_count': r['validation']['ft_jobs'] = 74
    if defect == 'replay_count': r['validation']['checkpoint_replays'] = 74
    if defect == 'model': r['models'].pop('croma')
    if defect == 'class': r['class_names'].reverse()
    if defect == 'fold': block['folds'].pop()
    if defect == 'seed': block['folds'][0]['seeds'].pop()
    if defect == 'bool_fold': block['folds'][0]['fold'] = False
    if defect == 'support': block['folds'][0]['counts']['test']['total'] += 1
    if defect == 'mean': block['summary']['finetuned']['macro']['f1']['mean'] += .1
    if defect == 'population_sd':
        row = block['summary']['finetuned']['macro']['f1']
        row['sample_std'] = float(np.std(row['fold_values'], ddof=0))
    if defect == 'fold_value': block['folds'][0]['values']['ft_minus_lr']['macro']['f1'] += 1
    if defect == 'source': r['analysis_source_sha256']['scripts/analyze_repaired_adaptation.py'] = '0' * 64
    if defect == 'runtime': r.pop('analysis_runtime')
    if defect == 'nonfinite': block['summary']['finetuned']['macro']['f1']['fold_values'][0] = float('nan')
    with pytest.raises((ValueError, KeyError)):
        plots.display_values(r)


def command(report, tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    path = inputs / 'synthetic_accepted_analysis.json'
    path.write_text(json.dumps(report))
    args = ['--analysis', str(path), '--analysis-sha256', plots.common.sha256(path)]
    return path, args


def test_rendered_package_is_deterministic_bound_and_uses_one_table_pipeline(accepted_analysis, tmp_path):
    import matplotlib
    path, args = command(accepted_analysis, tmp_path)
    first, second = tmp_path / 'first', tmp_path / 'second'
    before = path.read_bytes()
    assert plots.main([*args, '--output-dir', str(first)]) == 0
    assert plots.main([*args, '--output-dir', str(second)]) == 0
    names = {p.name for p in first.iterdir()}
    assert len(names) == 9 and names == {p.name for p in second.iterdir()}
    assert all((first / n).read_bytes() == (second / n).read_bytes() for n in names)
    metadata = json.loads((first / 'repaired_adaptation_display_20260930.json').read_text())
    assert metadata['input']['sha256'] == plots.common.sha256(path)
    assert metadata['analysis_provenance']['source_sha256'] == accepted_analysis['analysis_source_sha256']
    assert metadata['analysis_provenance']['runtime'] == accepted_analysis['analysis_runtime']
    assert metadata['display_runtime'] == dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__)
    assert len(metadata['display_source_sha256']) == 4 and len(metadata['artifacts']) == 7
    sums = (first / 'SHA256SUMS').read_text().splitlines()
    assert len(sums) == 8
    for line in sums:
        digest, name = line.split('  ')
        assert plots.common.sha256(first / name) == digest
    for kind, count in [('macro', 3), ('classes', 18), ('training', 75)]:
        with (first / f'repaired_adaptation_{kind}_20260930.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
    macro_tex = (first / 'repaired_adaptation_macro_20260930.tex').read_text()
    for row in metadata['display']['macro_rows']:
        for field in ('ft_f1', 'ft_recall', 'ft_minus_lr_f1', 'ft_minus_sgd_f1'):
            assert f"${row[field + '_mean']:.2f} \\pm {row[field + '_sample_std']:.2f}$" in macro_tex
    class_tex = (first / 'repaired_adaptation_classes_20260930.tex').read_text()
    with (first / 'repaired_adaptation_classes_20260930.csv').open(newline='') as stream:
        class_csv = list(csv.DictReader(stream))
    for row, csv_row in zip(metadata['display']['class_rows'], class_csv):
        assert csv_row == {k: str(v) for k, v in row.items()}
        for prefix in ('ft', 'lr'):
            for metric in plots.METRICS:
                mean, sd = (row[f'{prefix}_{metric}_{k}'] for k in ('mean', 'sample_std'))
                assert rf'\shortstack{{${mean:.2f}$\\$\pm {sd:.2f}$}}' in class_tex
    assert plots.main([*args, '--output-dir', str(first)]) == 1
    assert all((first / n).read_bytes() == (second / n).read_bytes() for n in names)
    assert path.read_bytes() == before
    pdf = (first / 'repaired_adaptation_recipe_20260930.pdf').read_bytes()
    assert b'/CreationDate' not in pdf and b'/ModDate' not in pdf


@pytest.mark.parametrize('defect', ['digest', 'existing', 'input_overlap', 'render_failure', 'input_changed'])
def test_cli_refusal_or_failure_preserves_owned_boundary(accepted_analysis, tmp_path, monkeypatch, defect, capsys):
    path, args = command(accepted_analysis, tmp_path)
    output = tmp_path / 'display'
    if defect == 'digest': args[-1] = '0' * 64
    if defect == 'existing':
        output.mkdir(); (output / 'sentinel').write_bytes(b'keep')
    if defect == 'input_overlap': output = path.parent / 'display'
    if defect in ('render_failure', 'input_changed'):
        def fake_figure(*args):
            if defect == 'render_failure': raise OSError('injected figure error')
            path.write_text('{}')
            return 'fixture'
        monkeypatch.setattr(plots, 'figure', fake_figure)
    assert plots.main([*args, '--output-dir', str(output)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'invalid'
    assert not (output / 'SHA256SUMS').exists()
    if defect == 'existing': assert (output / 'sentinel').read_bytes() == b'keep'
    elif defect in ('render_failure', 'input_changed'):
        assert receipt['incomplete_output_dir'] == str(output) and output.exists()
    else: assert not output.exists()


@pytest.mark.parametrize('target', ['SHA256SUMS', 'repaired_adaptation_display_20260930.json'])
@pytest.mark.parametrize('failure', ['prefix_error', 'short_write', 'close_error'])
def test_publication_failure_leaves_no_complete_metadata(accepted_analysis, tmp_path, monkeypatch, capsys, target, failure):
    path, args = command(accepted_analysis, tmp_path)
    before = path.read_bytes()
    output = tmp_path / 'display'
    metadata = output / 'repaired_adaptation_display_20260930.json'
    factory = plots.analysis.tempfile.NamedTemporaryFile
    created, attempted = [], []

    def fixture_figure(display, directory):
        for extension in ('pdf', 'png'):
            (directory / ('repaired_adaptation_recipe_20260930.' + extension)).write_bytes(b'synthetic figure')
        return 'fixture'

    class FailingStream:
        def __init__(self, stream):
            self.stream, self.name = stream, stream.name
            created.append(Path(self.name))

        def __enter__(self):
            return self

        def write(self, encoded):
            if failure == 'close_error':
                return self.stream.write(encoded)
            self.stream.write(encoded[:113])
            self.stream.flush()
            if failure == 'prefix_error':
                raise OSError('disk full after prefix')
            return 113

        def __exit__(self, *error):
            self.stream.__exit__(*error)
            if failure == 'close_error':
                raise OSError('close failed')

    def staged(*args, **kwargs):
        attempted.append(kwargs['prefix'])
        stream = factory(*args, **kwargs)
        return FailingStream(stream) if kwargs['prefix'] == '.' + target + '.' else stream

    monkeypatch.setattr(plots, 'figure', fixture_figure)
    monkeypatch.setattr(plots.analysis.tempfile, 'NamedTemporaryFile', staged)
    assert plots.main([*args, '--output-dir', str(output)]) == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'invalid' and receipt['incomplete_output_dir'] == str(output)
    assert len(created) == 1 and not created[0].exists()
    assert not list(output.glob('.*.tmp')) and path.read_bytes() == before
    assert not metadata.exists()
    if target == 'SHA256SUMS':
        assert attempted == ['.SHA256SUMS.']
        assert not (output / 'SHA256SUMS').exists() and len(list(output.iterdir())) == 7
    else:
        assert attempted == ['.SHA256SUMS.', '.' + metadata.name + '.']
        sums = (output / 'SHA256SUMS').read_text().splitlines()
        assert len(sums) == 8 and len(list(output.iterdir())) == 8
        assert sum(line.endswith('  ' + metadata.name) for line in sums) == 1
        for line in sums:
            digest, name = line.split('  ')
            if name != metadata.name:
                assert plots.common.sha256(output / name) == digest


def test_figure_retains_every_signed_fold_point_mean_and_zero(accepted_analysis, tmp_path, monkeypatch):
    display = plots.display_values(accepted_analysis)
    observed = []
    def inspect(fig, stem):
        for ax, metric in zip(fig.axes, ('f1', 'recall')):
            expected = []
            for model in plots.MODELS:
                for reference in plots.REFERENCES:
                    data = display['models'][model]['summary'][reference]['macro'][metric]
                    expected.extend([*data['fold_values'], data['mean']])
            actual = [float(c.get_offsets()[0, 0]) for c in ax.collections]
            np.testing.assert_array_equal(actual, expected)
            assert len(actual) == 36  # Thirty fold points and six black means.
            assert len(ax.lines) == 1 and list(ax.lines[0].get_xdata()) == [0, 0]
            assert ax.get_xlim()[0] < min(0, min(actual)) and ax.get_xlim()[1] > max(0, max(actual))
            assert not ax.containers  # No confidence/error bars.
            for collection in ax.collections[5::6]:
                np.testing.assert_array_equal(collection.get_facecolors()[0, :3], [0, 0, 0])
            observed.append(actual)
    monkeypatch.setattr(plots.joint, 'save_figure', inspect)
    plots.figure(display, tmp_path)
    assert len(observed) == 2


def test_recovery_analyzer_digest_matches_source_and_rejects_old_analysis(
    accepted_analysis,
):
    assert plots.ANALYZER_SHA == plots.common.sha256(plots.analysis.__file__)
    assert plots.display_values(accepted_analysis)['training_summary'][0]['jobs'] == 25
    accepted_analysis['analysis_source_sha256'][
        'scripts/analyze_repaired_adaptation.py'
    ] = 'ffc13f08426e5a03607cddbd0826c6853ba8061dc36c1ecaa98ed82ab9c3de56'
    with pytest.raises(ValueError, match='Unaccepted analysis source'):
        plots.display_values(accepted_analysis)
