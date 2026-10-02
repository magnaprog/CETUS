"""Reconstruct class F1 contrasts from accepted adaptation confusion counts.

Run with --analysis, its accepted --analysis-sha256, and a new --output-dir
outside source and inputs. Uses CPU metadata only; no training or inference.
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
from scripts import plot_repaired_adaptation as plots

ROOT = Path(__file__).resolve().parents[1]
AGGREGATION = ('Compute class F1 separately for each FT seed; average five seeds within each fold; '
               'subtract the same-fold common LR class F1; retain five fold contrasts; '
               'report their equal-weight mean and sample SD (ddof=1).')


def class_f1(record, support):
    """Reconstruct ratios independently of saved metrics and metric helpers."""
    counts = np.asarray(record['confusion_counts'])
    plots.require(counts.shape == (6, 6) and counts.dtype.kind in 'iu' and (counts >= 0).all(),
                  'Invalid confusion counts')
    plots.require(np.array_equal(counts.sum(axis=1), support), 'Confusion support differs')
    counts = counts.astype(np.float64)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    values = np.divide(2 * counts.diagonal(), denominator, out=np.zeros(6), where=denominator > 0)
    plots.joint.close(values, record['per_class']['f1'], 'Class F1 differs from confusion counts')
    plots.joint.close(values.mean(), record['macro']['f1'], 'Macro F1 differs from confusion counts')
    return values


def class_changes(report):
    validated = plots.display_values(report)
    result = dict(class_names=list(plots.CLASSES), fold_ids=list(range(5)), seed_ids=list(range(5)),
                  score_units='percent', difference_units='percentage points', aggregation=AGGREGATION,
                  interpretation='Complete recipe contrasts in expert-map agreement; fold spread is descriptive.',
                  models={})
    for model, name in zip(plots.MODELS, plots.NAMES):
        ft, lr = [], []
        for fold in plots.ordered(report['models'][model]['folds'], 'fold'):
            support = fold['counts']['test']['class_counts']
            ft.append(np.mean([class_f1(seed['finetuned'], support)
                               for seed in plots.ordered(fold['seeds'], 'seed')], axis=0))
            lr.append(class_f1(fold['common_lr']['metrics'], support))
        ft, lr = 100 * np.asarray(ft), 100 * np.asarray(lr)
        summary = validated['models'][model]['summary']
        for values, group in ((ft, 'finetuned'), (lr, 'common_lr')):
            plots.joint.close(values, summary[group]['per_class']['f1']['fold_values'],
                              'Reconstructed class F1 differs from registered fold values')
        contrasts = ft - lr
        macro = plots.joint.describe(contrasts.mean(axis=1))
        registered = summary['ft_minus_lr']['macro']['f1']
        for key in ('fold_values', 'mean', 'sample_std'):
            plots.joint.close(macro[key], registered[key], 'Class-average contrast differs from registered macro ' + key)
        result['models'][model] = dict(name=name, finetuned_f1_percent=ft.tolist(),
            common_lr_f1_percent=lr.tolist(), difference=plots.joint.describe(contrasts),
            macro_difference=macro)
    return result


def figure(changes, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    style = {'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11.5,
             'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220}
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8C6D31']
    markers = ['o', 's', '^', 'v', 'D']
    points = np.asarray([changes['models'][m]['difference']['fold_values'] for m in plots.MODELS])
    low, high = min(0., points.min()), max(0., points.max())
    padding = max(1., (high - low) * .08)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.7), sharex=True, sharey=True)
        fig.subplots_adjust(left=.105, right=.985, bottom=.32, top=.82, wspace=.14)
        for ax, model, panel in zip(axes, plots.MODELS, 'ABC'):
            record = changes['models'][model]
            values = np.asarray(record['difference']['fold_values'])
            for c in range(6):
                for fold in range(5):
                    ax.scatter(values[fold, c], c + (fold - 2) * .09, color=colors[fold],
                               marker=markers[fold], s=28, zorder=3)
                ax.scatter(record['difference']['mean'][c], c, color='black', marker='|',
                           s=180, linewidths=2.2, zorder=4)
            ax.axvline(0, color='.45', linewidth=.9)
            ax.set(xlim=(low - padding, high + padding), ylim=(5.6, -.6), yticks=range(6))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_title(panel + '  ' + record['name'], loc='left')
            ax.grid(axis='x', color='.9', linewidth=.7)
            ax.set_axisbelow(True)
        axes[0].set_yticklabels(changes['class_names'])
        handles = [Line2D([], [], color=colors[f], marker=markers[f], linestyle='',
                          label=f'Fold {f}', markersize=5) for f in range(5)]
        handles.append(Line2D([], [], color='black', marker='|', linestyle='', label='Mean',
                              markersize=10, markeredgewidth=2))
        fig.suptitle('Class F1: adaptation minus common LR', fontsize=12, y=.96)
        fig.supxlabel('F1 difference (percentage points)', fontsize=10.5, y=.21)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .105), ncol=6,
                   frameon=False, columnspacing=1., handletextpad=.3)
        fig.text(.105, .085, 'FT: five seed metrics averaged per fold. LR: one fit per fold.\n'
                 'Equal fold weights; fold variation is descriptive. Targets are map classes.', fontsize=9.5, va='top')
        try:
            plots.joint.save_figure(fig, directory / 'class_changes')
        finally:
            plt.close(fig)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--analysis-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.analysis.read_bytes()
        plots.require(re.fullmatch('[0-9a-f]{64}', args.analysis_sha256)
                      and hashlib.sha256(raw).hexdigest() == args.analysis_sha256, 'Analysis SHA256 differs')
        changes = class_changes(json.loads(raw))
        output = args.output_dir.resolve()
        plots.require(not args.output_dir.exists()
                      and not any(p.is_symlink() for p in (args.output_dir, *args.output_dir.parents)),
                      'Output directory must be new')
        plots.require(not any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (ROOT, args.analysis.resolve().parent)), 'Output overlaps source or inputs')
        sources = {str(Path(m.__file__).resolve().relative_to(ROOT)): plots.common.sha256(m.__file__)
                   for m in (sys.modules[__name__], plots, plots.analysis, plots.joint, plots.common, plots.analysis.native)}
        output.mkdir()
        version = figure(changes, output)
        runtime = dict(python=sys.version.split()[0], numpy=np.__version__, matplotlib=version)
        provenance = dict(input_sha256=args.analysis_sha256, source_sha256=json.dumps(sources, sort_keys=True),
                          **runtime, aggregation=AGGREGATION)
        with (output / 'class_changes.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = None
            for model in plots.MODELS:
                record = changes['models'][model]
                diff = record['difference']
                for c, terrain in enumerate(changes['class_names']):
                    row = dict(model=model, terrain_class=terrain,
                        ft_mean_f1_percent=float(np.mean(record['finetuned_f1_percent'], axis=0)[c]),
                        lr_mean_f1_percent=float(np.mean(record['common_lr_f1_percent'], axis=0)[c]),
                        difference_mean_pp=diff['mean'][c], difference_sample_sd_pp=diff['sample_std'][c],
                        **{f'difference_fold_{f}_pp': diff['fold_values'][f][c] for f in range(5)}, **provenance)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
        plots.require(plots.common.sha256(args.analysis) == args.analysis_sha256, 'Analysis changed during rendering')
        plots.require(all(plots.common.sha256(ROOT / p) == h for p, h in sources.items()), 'Source changed during rendering')
        metadata = dict(status='complete', schema_version='1.0.0', **changes,
            input=dict(path=str(args.analysis.resolve()), sha256=args.analysis_sha256),
            source_sha256=sources, runtime=runtime,
            artifacts={p.name: dict(bytes=p.stat().st_size, sha256=plots.common.sha256(p)) for p in sorted(output.iterdir())})
        receipt = plots.analysis.publish_report(output / 'class_changes.json',
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n')
        plots.require(receipt == dict(report_written=True), 'Metadata publication failed')
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(status='invalid', error=str(error))))
        return 1
    print(json.dumps(dict(status='complete', output_dir=str(output), models=3, classes=6, fold_contrasts=90)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
