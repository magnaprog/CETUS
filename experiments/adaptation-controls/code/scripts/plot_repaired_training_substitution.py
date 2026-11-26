"""Display independently accepted training-substitution audits; never fit models.

Run with ``python -m scripts.plot_repaired_training_substitution --help``.
Audit SHA256 arguments must come from independent acceptance, not this exporter.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np

from scripts import plot_repaired_common_classifier as shared

ROOT = Path(__file__).resolve().parents[1]
MODELS, METRICS = shared.MODELS, shared.METRICS
PINS = dict(
    buffer_protocol='cbe9e909fc586970d05b4ddeb6fe2c68c021744ff75e8aee4142e67bb73cc450',
    buffer_specification='ffa953a973bceea6a923facec253e0757f4f676e838789b65e552c1412a0ab0b',
    common_protocol='48515a7583e99c0cfd9fb179600cdfeda7dab484f424b4bcbdb912570632f9ca',
    common_specification='b9d645fcb93f847bf1bf3edf0a048fe169b678a51436a61c283a3f2d0a0e1318',
    common_audit='e0c007157b51ea83ab8cdfb7b661232456ecc1a6181d231dbd03104ac5473379',
    common_plan='b9219491ac71e198be93c1071cbf9c15c6eca72813c13c8b4862d35c7f637df3',
    native_audit='db352133220ce64b71f68e79269f1f93407d1906ad6545c06797eadd24a092b2',
    catalog='2c77251230c47596f33d13eafd91ff7d0c927c8f8275451d814fad875df893ec')
require, close = shared.require, shared.close
CAPTION = (
    'Training substitution sensitivity on the same repaired Titan test folds. '
    'Each point is replacement minus reference macro F1 for one fold, in percentage points; '
    'black marks and printed values are equal-weight means of five folds. '
    'Left: common LR averages five replacement draws per fold, with one deterministic fit per draw '
    'and one original reference fit. Right: original SGD averages five head seeds within each draw '
    'and then five draws; its reference averages five original head seeds. '
    'Each draw reinstates boundary training tiles and removes the same class counts from eligible '
    'interior training tiles. Training class counts, test membership, and image normalization remain fixed. '
    'LR refits feature standardization on each replacement training set. '
    'All six classes enter the macro metrics; undefined class scores are zero. '
    'All warned or iteration-capped replacement LR fits are retained. Training sets overlap across folds and draws. '
    'These descriptive differences combine geographic-composition and fitting effects; '
    'they do not estimate a causal leakage effect. Classical feature families are outside this control.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def hex_hash(value, length=64):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None,
            'Invalid hash or commit')


def ordered_five(rows, field):
    require(len(rows) == 5 and all(type(r[field]) is int for r in rows)
            and [r[field] for r in rows] == list(range(5)), 'Five ordered '+field+' values required')


def grid(rows):
    require(len(rows) == 20 and all(type(r['fold']) is int for r in rows), 'Twenty model/fold jobs required')
    result = {(r['model'], r['fold']): r for r in rows}
    require(len(result) == 20 and set(result) == {(m, f) for m in MODELS for f in range(5)},
            'Model/fold membership differs')
    for row in rows:
        hex_hash(row['manifest_sha256'])
    return result


def check_design(buffer_protocol, common_protocol):
    for label, protocol in [('buffer', buffer_protocol), ('common', common_protocol)]:
        require(digest(protocol['specification']) == protocol['specification_sha256'] == PINS[label+'_specification'],
                'Protocol specification differs')
    b, c = buffer_protocol['specification'], common_protocol['specification']
    require(b['models'] == list(MODELS) and b['probe_seeds'] == b['draw_seeds'] == list(range(5)), 'Sealed design differs')
    require(b['catalog_sha256'] == c['catalog']['sha256'] == PINS['catalog']
            and b['frozen_audit_sha256'] == c['native_audit']['sha256'] == PINS['native_audit'], 'Population differs')
    require(b['class_order'] == c['class_names'] == [v.lower() for v in shared.CLASSES], 'Class order differs')
    ordered_five(b['folds'], 'fold'); ordered_five(c['folds'], 'fold')
    support = []
    for bf, cf in zip(b['folds'], c['folds']):
        ordered_five(bf['draws'], 'draw_seed')
        require(bf['split_sha256'] == cf['split']['sha256'] and bf['class_counts'] == cf['counts']['train']['class_counts'],
                'Split or training class counts differ')
        require(bf['normalization_provenance'] == cf['normalization_provenance']
                and digest(bf['normalization_provenance']) == cf['normalization_sha256'], 'Image normalization differs')
        require(digest(bf['retained_train_ids']) == cf['train_ids_sha256']
                and digest(bf['test_ids']) == cf['test_ids_sha256'], 'Training or test membership differs')
        support.append(dict(fold=bf['fold'], train_class_counts=bf['class_counts'],
                            test_class_counts=cf['counts']['test']['class_counts'],
                            boundary_class_counts=bf['boundary_class_counts'],
                            train_total=cf['counts']['train']['total'], test_total=cf['counts']['test']['total'],
                            boundary_total=len(bf['boundary_ids']), validation_total=len(bf['validation_ids']),
                            split_sha256=bf['split_sha256'], test_ids_sha256=cf['test_ids_sha256'],
                            normalization_sha256=cf['normalization_sha256'],
                            replacement_training_ids_sha256=[d['training_ids_sha256'] for d in bf['draws']]))
    return b, c, support


def fit_diagnostics(draw, parameters):
    fit = draw['fit']
    require(fit['classifier_parameters'] == parameters['classifier'] and fit['scaler_parameters'] == parameters['scaler'],
            'Classifier or scaler recipe differs')
    iterations = fit['n_iter']
    require(len(iterations) == 1 and type(iterations[0]) is int and 0 <= iterations[0] <= 2000, 'Invalid iteration count')
    warnings = fit['fit_warnings']
    require(isinstance(warnings, list) and all(set(w) == {'category', 'message'}
            and all(isinstance(v, str) for v in w.values()) for w in warnings), 'Invalid fit warnings')
    for key, expected in [('iteration_limit_reached', iterations[0] >= 2000),
                          ('convergence_warning', any(w['category'] == 'ConvergenceWarning' for w in warnings))]:
        require(type(fit[key]) is bool and fit[key] == expected, 'Fit diagnostic flag differs')
    hex_hash(fit['estimator_sha256'])
    replay = draw['estimator_replay']
    error = replay['probability_max_abs_difference']
    require(replay['verified'] is True and type(error) in (float, int) and 0 <= error <= 1e-12, 'Unaccepted estimator replay')
    return dict(fit=fit, estimator_replay=replay)


def summarize_metric(reference, replacement, saved):
    ref, repl = np.asarray(reference, dtype=float), np.asarray(replacement, dtype=float)
    require(ref.shape == repl.shape == (5,) and np.isfinite([ref, repl]).all()
            and (ref >= 0).all() and (ref <= 1).all() and (repl >= 0).all() and (repl <= 1).all(),
            'Metrics must be five finite fractions')
    result = {}
    for name, values in [('reference', ref), ('replacement', repl), ('difference', repl-ref)]:
        measured = shared.describe(values)
        for key in ('fold_values', 'mean', 'sample_std'):
            close(measured[key], saved[name][key], 'Accepted aggregate differs: '+name+'/'+key)
        # Inputs are fractions. Absolute rates use percent; differences and SD use pp.
        result[name] = shared.describe(100*values)
    return result


def analyze(sgd, lr, buffer_protocol, common_protocol, sgd_sha256):
    b, c, support = check_design(buffer_protocol, common_protocol)
    require(sgd['status'] == lr['status'] == 'complete', 'Both independently accepted reports must be complete')
    require(sgd['protocol_sha256'] == PINS['buffer_protocol']
            and sgd['specification_sha256'] == PINS['buffer_specification'], 'SGD protocol differs')
    require(lr['schema_version'] == '1.0.0', 'LR schema differs')
    require(lr['validation'] == dict(complete_jobs=20, estimator_replays=100)
            and all(type(v) is int for v in lr['validation'].values()), 'LR audit count differs')
    # The caller's accepted report hash binds these future execution identities.
    # Fixed parent design pins above remain independent of the pending producer seal.
    for key in ('protocol_sha256', 'specification_sha256', 'plan_sha256', 'auditor_sha256'):
        hex_hash(lr[key])
    hex_hash(lr['producer_commit'], 40)
    expected_parents = {k: PINS[k] for k in ('common_protocol', 'common_audit', 'common_plan', 'buffer_protocol')}
    expected_parents['buffer_audit'] = sgd_sha256
    require(set(lr['parents']) == set(expected_parents) | {'finetuning_audit'}, 'LR parent inventory differs')
    for key, expected in expected_parents.items():
        require(lr['parents'][key]['sha256'] == expected, 'LR parent binding differs: '+key)
    hex_hash(lr['parents']['finetuning_audit']['sha256'])
    grid(sgd['input_fit_manifests'])
    jobs = grid(lr['folds'])
    require(set(sgd['models']) == set(lr['models']) == set(MODELS), 'Model summary membership differs')
    result = dict(schema_version='1.0.0', status='complete', model_order=list(MODELS), class_order=list(shared.CLASSES),
                  units=dict(reference='percent', replacement='percent', difference='percentage points', sample_std='percentage points'),
                  caption=CAPTION, support_design=support, recipes={'lr': {}, 'sgd': {}}, lr_fit_diagnostics=[],
                  resolved_lr_parameters=c['resolved_parameters'], sgd_recipe=b['recipe'],
                  validation=dict(sgd_manifest_jobs=20, sgd_draw_means=100,
                                  sgd_heads_inherited_from_accepted_audit=500, lr_jobs=20, lr_estimator_replays=100),
                  acceptance_scope='Caller-supplied report hashes attest independent acceptance. This display checks summary bindings and arithmetic; it does not replay heads, load estimators, or re-audit raw arrays.',
                  source_report_bindings=dict(sgd_protocol_sha256=sgd['protocol_sha256'],
                      lr={k: lr[k] for k in ('producer_commit', 'protocol_sha256', 'specification_sha256', 'plan_sha256', 'auditor_sha256')},
                      lr_parents={k: v['sha256'] for k, v in lr['parents'].items()},
                      sgd_fit_manifests=sgd['input_fit_manifests'],
                      lr_fit_manifests=[{k: r[k] for k in ('model', 'fold', 'manifest_sha256')} for r in lr['folds']]))
    for model in MODELS:
        sf = sgd['models'][model]['folds']; ordered_five(sf, 'fold')
        refs, replacements = [], []
        for f in range(5):
            row = jobs[model, f]; ordered_five(row['draws'], 'draw_seed')
            test_support = support[f]['test_class_counts']
            ref = shared.reconstruct(row['reference'], test_support)['macro']; refs.append(ref)
            draws = []
            for draw, planned in zip(row['draws'], b['folds'][f]['draws']):
                require(draw['training_ids_sha256'] == planned['training_ids_sha256']
                        and draw['train_class_counts'] == support[f]['train_class_counts'], 'Draw membership differs')
                measured = shared.reconstruct(draw['metrics'], test_support)
                for m in METRICS:
                    close(draw['difference'][m], measured['macro'][m]-ref[m], 'Draw difference differs')
                close(draw['difference']['kappa'], measured['kappa']-row['reference']['kappa'], 'Kappa difference differs')
                draws.append(measured['macro'])
                result['lr_fit_diagnostics'].append(dict(model=model, fold=f, draw_seed=draw['draw_seed'],
                                                       **fit_diagnostics(draw, c['resolved_parameters'])))
            replacements.append({m: float(np.mean([d[m] for d in draws])) for m in METRICS})
            sd = sf[f]['draw_means']
            require(len(sd) == 5 and all(set(d) == set(METRICS) for d in sd), 'Five SGD draw means required')
            require(all(type(d[m]) in (int, float) and 0 <= d[m] <= 1 for d in sd for m in METRICS),
                    'SGD draw means must be finite fractions')
            for m in METRICS:
                close(sf[f]['replacement'][m], np.mean([d[m] for d in sd]), 'SGD draw aggregation differs')
        for recipe, reference, replacement in [
                ('lr', refs, replacements), ('sgd', [r['reference'] for r in sf], [r['replacement'] for r in sf])]:
            metrics = {}
            for m in METRICS:
                saved = (lr['models'][model][m] if recipe == 'lr' else
                         {k: sgd['models'][model][k][m] for k in ('reference', 'replacement', 'difference')})
                metrics[m] = summarize_metric([r[m] for r in reference], [r[m] for r in replacement], saved)
            result['recipes'][recipe][model] = metrics
    diagnostics = result['lr_fit_diagnostics']
    result['lr_diagnostic_counts'] = dict(fits=len(diagnostics),
        fits_with_warnings=sum(bool(d['fit']['fit_warnings']) for d in diagnostics),
        convergence_warnings=sum(d['fit']['convergence_warning'] for d in diagnostics),
        iteration_caps=sum(d['fit']['iteration_limit_reached'] for d in diagnostics))
    return result


def mean_label(value):
    text = f'{value:+.2f}'
    return '0.00' if float(text) == 0 else text


def make_figure(report, synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220})
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8C6D31']
    markers = ['o', 's', '^', 'v', 'D']
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.125, right=.91, bottom=.31, top=.77, wspace=.22)
    extent = max(1., max(abs(v) for recipe in report['recipes'].values() for model in recipe.values()
                        for v in model['f1']['difference']['fold_values'])) * 1.18
    for ax, recipe, title in zip(axes, ('lr', 'sgd'), ('Common LR', 'Original SGD')):
        ax.axvline(0, color='.5', linewidth=.8)
        for row, model in enumerate(MODELS):
            data = report['recipes'][recipe][model]['f1']['difference']
            for fold, value in enumerate(data['fold_values']):
                ax.scatter(value, row+(fold-2)*.085, color=colors[fold], marker=markers[fold], s=30, zorder=3)
            ax.scatter(data['mean'], row, color='black', marker='|', s=220, linewidths=2.3, zorder=4)
            ax.text(1.025, row, mean_label(data['mean']), transform=ax.get_yaxis_transform(), va='center', fontsize=9)
        ax.text(1.025, 1.05, 'Mean', transform=ax.transAxes, fontsize=9)
        ax.set(yticks=range(4), yticklabels=shared.NAMES[:4], ylim=(3.5, -.5), xlim=(-extent, extent), title=title)
        ax.grid(axis='x', color='.9'); ax.set_axisbelow(True)
    fig.suptitle('SYNTHETIC TEST FIXTURE' if synthetic else 'Training substitution sensitivity',
                 fontsize=11, y=.965)
    fig.supxlabel('Replacement minus reference macro F1, percentage points', y=.19, fontsize=10.5)
    handles = [Line2D([], [], color=colors[f], marker=markers[f], linestyle='', label=f'Fold {f}') for f in range(5)]
    handles.append(Line2D([], [], color='black', marker='|', markersize=12, linestyle='', label='Mean'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .092), ncol=6, frameon=False, fontsize=9)
    counts = report['lr_diagnostic_counts']
    fig.text(.52, .038, 'LR: five draws. SGD: five heads per draw, then five draws. Equal fold weights.\n'
             f"LR replacement fits with warnings: {counts['fits_with_warnings']}/100; at iteration cap: {counts['iteration_caps']}/100. All retained.",
             ha='center', va='center', fontsize=8.5)
    return fig


def write_tables(report, directory):
    summary, folds = [], []
    for recipe in ('lr', 'sgd'):
        for model in MODELS:
            for metric in METRICS:
                for condition in ('reference', 'replacement', 'difference'):
                    data = report['recipes'][recipe][model][metric][condition]
                    row = dict(recipe=recipe, model=model, metric=metric, condition=condition, units=report['units'][condition])
                    summary.append(dict(**row, mean=data['mean'], sample_sd_pp=data['sample_std']))
                    folds.extend(dict(**row, fold=f, value=value) for f, value in enumerate(data['fold_values']))
    fits = [{k: r[k] for k in ('model', 'fold', 'draw_seed')} | r['fit']
            | {'estimator_replay': r['estimator_replay']} for r in report['lr_fit_diagnostics']]
    for name, rows in [('summary', summary), ('folds', folds), ('lr_fits', fits),
                       ('support', report['support_design'])]:
        with (directory/(name+'.csv')).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})


def guard_output(path, inputs):
    path = path.absolute()
    require(not path.exists() and not path.is_symlink(), 'Output directory must be new')
    require(path.parent.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)), 'Output parent must exist without symlinks')
    path = path.resolve()
    require(all(not path.is_relative_to(p.resolve()) for p in [ROOT, *(p.parent for p in inputs)]),
            'Output must be outside source and input directories')
    require(not any((p/'.git').exists() for p in path.parents), 'Output must be outside source checkouts')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sgd-audit', 'lr-audit', 'buffer-protocol', 'common-protocol', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('sgd-audit-sha256', 'lr-audit-sha256'):
        parser.add_argument('--'+name, required=True, help='SHA256 from independent acceptance')
    args = parser.parse_args(argv)
    paths = {k: getattr(args, k) for k in ('sgd_audit', 'lr_audit', 'buffer_protocol', 'common_protocol')}
    hashes = dict(sgd_audit=args.sgd_audit_sha256, lr_audit=args.lr_audit_sha256,
                  buffer_protocol=PINS['buffer_protocol'], common_protocol=PINS['common_protocol'])
    for value in hashes.values():
        hex_hash(value)
    output = guard_output(args.output_dir, paths.values())
    sources = {Path(__file__).resolve(): shared.sha(__file__), Path(shared.__file__).resolve(): shared.sha(shared.__file__)}
    inputs = {k: shared.read_bound(path, hashes[k]) for k, path in paths.items()}
    report = analyze(inputs['sgd_audit'], inputs['lr_audit'], inputs['buffer_protocol'], inputs['common_protocol'], hashes['sgd_audit'])
    report['inputs'] = {k: dict(name=p.name, sha256=hashes[k]) for k, p in paths.items()}
    report['display_source_sha256'] = {str(p.relative_to(ROOT)): v for p, v in sources.items()}
    output.mkdir()  # Exclusive; existing directories are never reused.
    write_tables(report, output)
    (output/'caption.txt').write_text(CAPTION+'\n')
    fig = make_figure(report)
    shared.save_figure(fig, output/'repaired_training_substitution_20260930')
    import matplotlib
    import matplotlib.pyplot as plt
    plt.close(fig)
    report['runtime'] = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__)
    report['artifacts'] = {p.name: dict(bytes=p.stat().st_size, sha256=shared.sha(p)) for p in sorted(output.iterdir())}
    require(all(shared.sha(p) == hashes[k] for k, p in paths.items())
            and all(shared.sha(p) == v for p, v in sources.items()), 'Input or display source changed during export')
    # Written last: its presence marks a completed export. Failures leave a new, incomplete directory.
    with (output/'summary.json').open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    main()
