"""Small synthetic checks of seed-level class F1 and signed fold contrasts."""

import copy
import csv
import json
from statistics import mean, stdev

import numpy as np
import pytest

from scripts import plot_repaired_class_changes as changes
from tests.test_repaired_adaptation_analysis import reports
from tests.test_repaired_adaptation_plots import accepted_analysis


def independent_f1(counts, c):
    denominator = sum(counts[c]) + sum(row[c] for row in counts)
    return 2 * counts[c][c] / denominator if denominator else 0.


def test_all_contrasts_and_sample_sd_match_independent_arithmetic(accepted_analysis):
    result = changes.class_changes(accepted_analysis)
    assert result['class_names'] == list(changes.plots.CLASSES)
    for model in changes.plots.MODELS:
        record = result['models'][model]
        folds = accepted_analysis['models'][model]['folds']
        expected = [[100 * (mean(independent_f1(s['finetuned']['confusion_counts'], c) for s in f['seeds'])
                            - independent_f1(f['common_lr']['metrics']['confusion_counts'], c))
                     for c in range(6)] for f in folds]
        np.testing.assert_allclose(record['difference']['fold_values'], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(record['difference']['mean'], [mean(v[c] for v in expected) for c in range(6)], atol=1e-12)
        np.testing.assert_allclose(record['difference']['sample_std'], [stdev(v[c] for v in expected) for c in range(6)], atol=1e-12)
        np.testing.assert_allclose(record['macro_difference']['fold_values'], [mean(v) for v in expected], atol=1e-12)
        assert np.any(np.asarray(expected) > 0) and np.any(np.asarray(expected) < 0)
    shuffled = copy.deepcopy(accepted_analysis)
    for block in shuffled['models'].values():
        block['folds'].reverse()
        for fold in block['folds']:
            fold['seeds'].reverse()
    assert changes.class_changes(shuffled) == result


def test_f1_is_computed_before_averaging_seed_counts(accepted_analysis):
    fold = accepted_analysis['models']['dinov2']['folds'][0]
    seed_counts = [s['finetuned']['confusion_counts'] for s in fold['seeds']]
    expected = [100 * mean(independent_f1(cm, c) for cm in seed_counts) for c in range(6)]
    pooled = np.sum(seed_counts, axis=0).tolist()
    wrong = [100 * independent_f1(pooled, c) for c in range(6)]
    assert max(abs(a - b) for a, b in zip(expected, wrong)) > .1
    actual = changes.class_changes(accepted_analysis)['models']['dinov2']['finetuned_f1_percent'][0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('target', ['ft', 'lr'])
@pytest.mark.parametrize('defect', ['negative', 'fraction', 'bool', 'shape', 'support', 'same_support_wrong_counts'])
def test_incorrect_confusion_counts_rejected(accepted_analysis, target, defect):
    fold = accepted_analysis['models']['dinov2']['folds'][0]
    record = fold['seeds'][0]['finetuned'] if target == 'ft' else fold['common_lr']['metrics']
    counts = record['confusion_counts']
    if defect == 'negative': counts[0][0] = -1
    if defect == 'fraction': counts[0][0] = .5
    if defect == 'bool': record['confusion_counts'] = [[False] * 6 for _ in range(6)]
    if defect == 'shape': counts.pop()
    if defect == 'support': counts[0][0] += 1
    if defect == 'same_support_wrong_counts': counts[0][0] -= 1; counts[0][1] += 1
    with pytest.raises(ValueError, match='confusion|Confusion|support'):
        changes.class_changes(accepted_analysis)


def test_macro_contrast_must_match_reconstructed_classes(accepted_analysis):
    block = accepted_analysis['models']['dinov2']
    # Keep the existing display validator internally consistent while corrupting
    # the registered macro contrast relative to the underlying count matrices.
    for fold in block['folds']:
        fold['values']['ft_minus_lr']['macro']['f1'] += 1
    summary = block['summary']['ft_minus_lr']['macro']['f1']
    summary['fold_values'] = [v + 1 for v in summary['fold_values']]
    summary['mean'] += 1
    with pytest.raises(ValueError, match='Class-average contrast'):
        changes.class_changes(accepted_analysis)


def test_cli_artifacts_bind_input_source_runtime_and_shared_plot_scale(accepted_analysis, tmp_path, monkeypatch):
    source = tmp_path / 'input'
    source.mkdir()
    path = source / 'analysis.json'
    path.write_text(json.dumps(accepted_analysis))
    digest = changes.plots.common.sha256(path)
    output = tmp_path / 'output'
    saved = changes.plots.joint.save_figure

    def inspect_figure(fig, stem):
        assert len(fig.axes) == 3
        assert all(ax.get_xlim() == fig.axes[0].get_xlim() for ax in fig.axes)
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == list(changes.plots.CLASSES)
        assert all(len(ax.collections) == 36 for ax in fig.axes)  # 30 fold points + six means
        saved(fig, stem)

    monkeypatch.setattr(changes.plots.joint, 'save_figure', inspect_figure)
    args = ['--analysis', str(path), '--analysis-sha256', digest, '--output-dir', str(output)]
    assert changes.main(args) == 0
    result = json.loads((output / 'class_changes.json').read_text())
    assert result['status'] == 'complete' and result['input']['sha256'] == digest
    assert set(result['artifacts']) == {'class_changes.csv', 'class_changes.png', 'class_changes.pdf'}
    assert len(result['source_sha256']) == 6
    for name, record in result['artifacts'].items():
        assert changes.plots.common.sha256(output / name) == record['sha256']
    with (output / 'class_changes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18
    for row in rows:
        assert row['input_sha256'] == digest and json.loads(row['source_sha256']) == result['source_sha256']
        assert {k: row[k] for k in result['runtime']} == result['runtime']
        c = result['class_names'].index(row['terrain_class'])
        expected = result['models'][row['model']]['difference']
        assert float(row['difference_mean_pp']) == expected['mean'][c]
        assert float(row['difference_sample_sd_pp']) == expected['sample_std'][c]
        assert [float(row[f'difference_fold_{f}_pp']) for f in range(5)] == [v[c] for v in expected['fold_values']]
    assert changes.main(args) == 1  # Never overwrite an existing artifact directory.
    assert changes.plots.common.sha256(path) == digest


def test_cli_rejects_wrong_input_hash_before_creating_output(accepted_analysis, tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text(json.dumps(accepted_analysis))
    output = tmp_path / 'output'
    assert changes.main(['--analysis', str(path), '--analysis-sha256', '0' * 64,
                         '--output-dir', str(output)]) == 1
    assert not output.exists()
