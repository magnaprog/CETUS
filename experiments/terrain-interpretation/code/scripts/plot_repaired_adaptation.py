"""Render accepted adaptation analysis into Figure A, tables and metadata.

Supply --analysis, its independently accepted --analysis-sha256, and a fresh
--output-dir outside source and inputs. Requires NumPy and Matplotlib. No
scientific gate, model, training, upstream artifact read or Git call is run.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[_name] = '1'
    sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts import analyze_repaired_adaptation as analysis

joint, common = analysis.joint, analysis.common
require = joint.require
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dinov2', 'dofa', 'croma')
NAMES = ('DINOv2', 'DOFA', 'CROMA')
CLASSES = ('Plains', 'Dunes', 'Hummocky', 'Labyrinths', 'Lakes', 'Craters')
METRICS = ('precision', 'recall', 'f1')
REFERENCES = ('ft_minus_sgd', 'ft_minus_lr')
ANALYZER_SHA = '8b7b842e4829eb54b45bc0b741891bfbde603391ed52602a6060b4992548187b'
STEM = 'repaired_adaptation_'
DATE = '_20260930'
CAPTION = ('Five seeds are averaged within each fold; common LR has one reference fit per fold. '
           'Checkpoint selection used validation macro recall. Fold points and sample fold SD describe variation '
           'across geographic partitions. Contrasts compare complete recipes and do not isolate unfreezing or '
           'pretraining. Targets are expert map labels. There is no RandomInit adaptation arm.')


def ordered(rows, field):
    require(len(rows) == 5 and all(type(r[field]) is int for r in rows)
            and sorted(r[field] for r in rows) == list(range(5)), 'Expected five unique ' + field + ' identities')
    return sorted(rows, key=lambda r: r[field])


def display_summary(summary, expected, factor, shape):
    values = np.asarray(summary['fold_values'], dtype=float)
    require(values.shape == (5, *shape) and np.isfinite(values).all(), 'Invalid display fold values')
    require((values >= (0 if factor == 100 else -100)).all()
            and (values <= (1 if factor == 100 else 100)).all(), 'Display units outside contract')
    joint.close(values, expected, 'Fold values differ from analysis rows')
    measured = joint.describe(values)
    for key in ('mean', 'sample_std'):
        joint.close(summary[key], measured[key], 'Display ' + key + ' differs from five folds')
    return {k: (factor * np.asarray(summary[k])).tolist() for k in ('fold_values', 'mean', 'sample_std')}


def display_values(report):
    require(report['status'] == 'complete' and report['schema_version'] == '1.0.0', 'Expected complete analysis schema')
    require(report['validation'] == dict(ft_jobs=75, checkpoint_replays=75, common_lr_jobs=20), 'Incomplete analysis cohort')
    require(set(report['models']) == set(MODELS) and report['class_names'] == [c.lower() for c in CLASSES], 'Model or class order differs')
    require(report['analysis_source_sha256']['scripts/analyze_repaired_adaptation.py'] == ANALYZER_SHA,
            'Unaccepted analysis source')
    require(set(report['inputs']) == {*analysis.PINS, 'ft_audit', 'replay_audit'}, 'Analysis input inventory differs')
    hashes = [*report['analysis_source_sha256'].values(), *(r['sha256'] for r in report['inputs'].values())]
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in hashes), 'Invalid analysis provenance hash')
    require(set(report['analysis_runtime']) == {'python', 'numpy'}
            and all(isinstance(v, str) and v for v in report['analysis_runtime'].values()), 'Analysis runtime missing')
    display = dict(models={}, macro_rows=[], class_rows=[], training_rows=[], training_summary=[], caption=CAPTION,
                   score_units='percent', difference_units='percentage points', sd='sample fold SD, ddof=1')
    shared_counts = None
    for model, name in zip(MODELS, NAMES):
        block = report['models'][model]
        folds = ordered(block['folds'], 'fold')
        counts = [f['counts'] for f in folds]
        for f in counts:
            for role in ('train', 'val', 'test'):
                c = f[role]
                require(len(c['class_counts']) == 6 and all(type(n) is int and n > 0 for n in c['class_counts'])
                        and type(c['total']) is int and sum(c['class_counts']) == c['total'], 'Invalid role support')
        if shared_counts is None:
            shared_counts = counts
        require(counts == shared_counts, 'Model populations differ')
        summaries = {}
        for group in ('finetuned', 'common_lr', *REFERENCES):
            summaries[group] = {}
            for level in (('macro',) if group in REFERENCES else ('macro', 'per_class')):
                summaries[group][level] = {}
                for metric in (('f1', 'recall') if group in REFERENCES else METRICS):
                    summaries[group][level][metric] = display_summary(block['summary'][group][level][metric],
                        [f['values'][group][level][metric] for f in folds], 1 if group in REFERENCES else 100,
                        () if level == 'macro' else (6,))
        display['models'][model] = dict(name=name, summary=summaries)
        macro = dict(model=name)
        for label, group, metric in [('ft_f1', 'finetuned', 'f1'), ('ft_recall', 'finetuned', 'recall'),
                                      ('ft_minus_lr_f1', 'ft_minus_lr', 'f1'), ('ft_minus_sgd_f1', 'ft_minus_sgd', 'f1')]:
            macro.update({label + '_' + k: summaries[group]['macro'][metric][k] for k in ('mean', 'sample_std')})
        display['macro_rows'].append(macro)
        for c, class_name in enumerate(CLASSES):
            support = [f['test']['class_counts'][c] for f in counts]
            row = dict(model=name, terrain_class=class_name, test_support=sum(support))
            row.update({f'support_fold_{f}': n for f, n in enumerate(support)})
            for prefix, group in (('ft', 'finetuned'), ('lr', 'common_lr')):
                for metric in METRICS:
                    row.update({prefix + '_' + metric + '_' + k: summaries[group]['per_class'][metric][k][c]
                                for k in ('mean', 'sample_std')})
            display['class_rows'].append(row)
        training = []
        for fold in folds:
            for seed in ordered(fold['seeds'], 'seed'):
                t = seed['training']
                require(t['epoch_indexing'] == 'one based' and all(type(t[k]) is int and t[k] >= 0
                        for k in (*analysis.HISTORY, 'optimizer_attempts_total') if k != 'epoch_indexing')
                        and 1 <= t['selected_epoch'] <= 50, 'Invalid training display metadata')
                row = dict(model=name, fold=fold['fold'], seed=seed['seed'],
                           **{k: t[k] for k in (*analysis.HISTORY, 'optimizer_attempts_total')})
                training.append(row)
        display['training_rows'].extend(training)
        display['training_summary'].append(dict(model=name, jobs=len(training),
            selected_epoch_min=min(t['selected_epoch'] for t in training),
            selected_epoch_max=max(t['selected_epoch'] for t in training),
            **{k: sum(t[k] for t in training) for k in ('optimizer_attempts_total', 'optimizer_steps_total',
                                                      'amp_skipped_steps_total', 'training_example_presentations')}))
    display['support_by_fold'] = shared_counts
    display['training_scope'] = 'Counters sum all fifty epochs for all 25 jobs per encoder; selected-epoch-only counters are unavailable.'
    return display


def tex_number(value):
    formatted = f'{value:.2f}'
    return '0.00' if formatted == '-0.00' else formatted


def table_files(display, directory):
    for name in ('macro', 'classes', 'training'):
        rows = display[{'macro': 'macro_rows', 'classes': 'class_rows', 'training': 'training_rows'}[name]]
        with (directory / (STEM + name + DATE + '.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def cell(row, key, stacked=False):
        mean, sd = (tex_number(row[key + '_' + k]) for k in ('mean', 'sample_std'))
        if stacked:
            return rf'\shortstack{{${mean}$\\$\pm {sd}$}}'
        return f'${mean} \\pm {sd}$'
    macro = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Encoder & FT F1, \% & FT recall, \% & FT minus LR F1, pp & FT minus SGD F1, pp \\', r'\midrule']
    for row in display['macro_rows']:
        macro.append(' & '.join([row['model'], *(cell(row, k) for k in ('ft_f1', 'ft_recall', 'ft_minus_lr_f1', 'ft_minus_sgd_f1'))]) + r' \\')
    classes = [r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
        r'Encoder & Class & Support & FT P, \% & FT R, \% & FT F1, \% & LR P, \% & LR R, \% & LR F1, \% \\', r'\midrule']
    for row in display['class_rows']:
        classes.append(' & '.join([row['model'], row['terrain_class'], str(row['test_support']),
            *(cell(row, p + '_' + m, stacked=True) for p in ('ft', 'lr') for m in METRICS)]) + r' \\')
    for name, lines in (('macro', macro), ('classes', classes)):
        lines.extend([r'\bottomrule', r'\end{tabular}'])
        with (directory / (STEM + name + DATE + '.tex')).open('x', encoding='utf-8') as stream:
            stream.write('% Equal-fold mean and sample fold SD. Support counts test tiles once.\n' + '\n'.join(lines) + '\n')


def figure(display, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style = {'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11.5, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220}
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8C6D31']
    markers = ['o', 's', '^', 'v', 'D']
    rows = [(m, ref) for m in MODELS for ref in REFERENCES]
    points = [v for metric in ('f1', 'recall') for m, ref in rows
              for v in display['models'][m]['summary'][ref]['macro'][metric]['fold_values']]
    low, high = min(0, min(points)), max(0, max(points))
    padding = max(1., (high - low) * .1)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(9.1, 6.2), sharey=True)
        fig.subplots_adjust(left=.245, right=.97, bottom=.29, top=.85, wspace=.19)
        for ax, metric, title in zip(axes, ('f1', 'recall'), ('A  Macro F1', 'B  Macro recall')):
            for row, (model, ref) in enumerate(rows):
                summary = display['models'][model]['summary'][ref]['macro'][metric]
                for fold, value in enumerate(summary['fold_values']):
                    ax.scatter(value, row + (fold - 2) * .09, color=colors[fold], marker=markers[fold], s=33, zorder=3)
                ax.scatter(summary['mean'], row, color='black', marker='|', s=240, linewidths=2.4, zorder=4)
            ax.axvline(0, color='.45', linewidth=.9)
            ax.set(xlim=(low - padding, high + padding), ylim=(5.6, -.6),
                   xlabel='FT minus frozen reference, pp', yticks=range(6))
            ax.set_title(title, loc='left')
            ax.grid(axis='x', color='.9', linewidth=.7)
            ax.set_axisbelow(True)
        axes[0].set_yticklabels([display['models'][m]['name'] + '\n' + ('FT minus SGD' if ref == 'ft_minus_sgd' else 'FT minus common LR')
                                 for m, ref in rows])
        handles = [Line2D([], [], color=colors[f], marker=markers[f], linestyle='', label=f'Fold {f}', markersize=5) for f in range(5)]
        handles.append(Line2D([], [], color='black', marker='|', linestyle='', label='Mean', markersize=10, markeredgewidth=2))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.57, .155), ncol=6,
                   frameon=False, fontsize=10.5, columnspacing=.8, handletextpad=.3)
        fig.suptitle('Fixed adaptation recipe versus frozen references', fontsize=12, y=.95)
        fig.text(.025, .12, 'Five seeds are averaged within each fold; common LR has one fit per fold.\n'
                 'Checkpoint selection uses validation macro recall. Fold spread is descriptive.\n'
                 'Contrasts compare complete recipes.', fontsize=10.5, va='top')
        try:
            joint.save_figure(fig, directory / (STEM + 'recipe' + DATE))
        finally:
            plt.close(fig)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--analysis-sha256', required=True, help='Digest from independent acceptance of the complete analysis')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    created = False
    try:
        require(re.fullmatch('[0-9a-f]{64}', args.analysis_sha256), 'Expected an accepted analysis SHA256')
        require(common.sha256(args.analysis) == args.analysis_sha256, 'Analysis SHA256 differs')
        report = common.read_json(args.analysis)
        display = display_values(report)
        output = args.output_dir.resolve()
        require(not args.output_dir.exists() and not any(p.is_symlink() for p in (args.output_dir, *args.output_dir.parents)), 'Output directory must be new')
        require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (ROOT, args.analysis.resolve().parent)), 'Output overlaps source or inputs')
        sources = {str(Path(m.__file__).resolve().relative_to(ROOT)): common.sha256(m.__file__)
                   for m in (sys.modules[__name__], joint, common, analysis)}
        output.mkdir()
        created = True
        table_files(display, output)
        matplotlib_version = figure(display, output)
        require(common.sha256(args.analysis) == args.analysis_sha256, 'Analysis changed during rendering')
        require(all(common.sha256(ROOT / p) == h for p, h in sources.items()), 'Display source changed during rendering')
        metadata = dict(status='complete', schema_version='1.0.0', display=display,
            input=dict(path=str(args.analysis.resolve()), sha256=args.analysis_sha256),
            analysis_provenance=dict(input_sha256={k: r['sha256'] for k, r in report['inputs'].items()},
                source_sha256=report['analysis_source_sha256'], runtime=report['analysis_runtime']),
            display_source_sha256=sources, display_runtime=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib_version),
            artifacts={p.name: dict(bytes=p.stat().st_size, sha256=common.sha256(p)) for p in sorted(output.iterdir())})
        metadata_name = STEM + 'display' + DATE + '.json'
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n'
        digests = {name: record['sha256'] for name, record in metadata['artifacts'].items()}
        digests[metadata_name] = hashlib.sha256(metadata_text.encode('utf-8')).hexdigest()
        sums = ''.join(digests[name] + '  ' + name + '\n' for name in sorted(digests))
        receipt = analysis.publish_report(output / 'SHA256SUMS', sums)
        require(receipt == dict(report_written=True), 'Checksum publication failed: ' + json.dumps(receipt))
        # The complete metadata is the last file published in the package.
        receipt = analysis.publish_report(output / metadata_name, metadata_text)
        require(receipt == dict(report_written=True), 'Metadata publication failed: ' + json.dumps(receipt))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(status='invalid', error=str(error), incomplete_output_dir=str(args.output_dir) if created else None)))
        return 1
    print(json.dumps(dict(status='complete', output_dir=str(output), macro_rows=3, class_rows=18, training_rows=75)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
