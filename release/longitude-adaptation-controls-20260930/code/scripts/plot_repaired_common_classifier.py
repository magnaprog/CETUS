"""Compare accepted repaired Titan feature families under one classifier recipe.

Reads small, hash-pinned audit records only. No fitting, cache loading or GPU use.
Run as ``python -m scripts.plot_repaired_common_classifier --help``.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

MODELS = ('dinov2', 'dofa', 'croma', 'random_init')
CLASSICAL = ('raw_intensity_statistics', 'intensity_histogram',
             'texture_glcm_lbp_gabor_hog_wavelet', 'combined_classical')
FAMILIES = MODELS + CLASSICAL
NAMES = ('DINOv2', 'DOFA', 'CROMA', 'Random Init', 'Intensity statistics',
         'Intensity histogram', 'Texture', 'Combined classical')
DIMENSIONS = (768, 768, 768, 768, 13, 32, 1820, 1865)
CLASSES = ('Plains', 'Dunes', 'Hummocky', 'Labyrinths', 'Lakes', 'Craters')
METRICS = ('precision', 'recall', 'f1')
PINS = dict(common='e0c007157b51ea83ab8cdfb7b661232456ecc1a6181d231dbd03104ac5473379',
            baseline='1c9dc259986b742f80afd92ff857c2798f32ecb41178b2a8fe12cff4de041cd2',
            recipe_analysis='8b32f77f7bacf90164d4ce930237ddee7958f7d845dba56be5b38047f7beead8',
            common_protocol='48515a7583e99c0cfd9fb179600cdfeda7dab484f424b4bcbdb912570632f9ca',
            baseline_protocol='a96d9ffc113ddec8ac637dc29726e8cc241a1dff1b69bc4d9cfe5b7d320a0961')
COMMITS = dict(common='3dd19a2e8c55d07e293f77ce139ded53112a7ef3',
               baseline='54281f05cfd67453f9b699810021b1e2da20c3ec')
PLANS = dict(common='b9219491ac71e198be93c1071cbf9c15c6eca72813c13c8b4862d35c7f637df3',
             baseline='78a229a6741267f769c142b66fb19046ba723f72abeaf938a4509ca51eb9dbc5')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path, digest):
    require(sha(path) == digest, 'Input SHA256 differs: '+str(path))
    return json.loads(Path(path).read_bytes())


def close(left, right, message):
    require(np.shape(left) == np.shape(right) and np.allclose(left, right, atol=1e-12, rtol=0), message)


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.shape[0] == 5 and np.isfinite(a).all(), 'Expected five finite fold values')
    return dict(fold_values=a.tolist(), mean=a.mean(axis=0).tolist(), sample_std=a.std(axis=0, ddof=1).tolist())


def reconstruct(record, support):
    cm = np.asarray(record['confusion_counts'])
    require(cm.shape == (6, 6) and cm.dtype.kind in 'iu' and (cm >= 0).all(), 'Invalid confusion counts')
    require(np.array_equal(cm.sum(axis=1), support) and min(support) > 0, 'Class support differs')
    tp, predicted = cm.diagonal(), cm.sum(axis=0)
    p = np.divide(tp, predicted, out=np.zeros(6), where=predicted > 0)
    values = dict(precision=p, recall=tp / np.asarray(support), f1=2*tp / (np.asarray(support)+predicted))
    for metric, values_ in values.items():
        close(values_, record['per_class'][metric], 'Per-class metric differs')
        close(values_.mean(), record['macro'][metric], 'Macro metric differs')
    close(predicted, record['prediction_counts'], 'Prediction counts differ')
    close(support, record['support'], 'Recorded support differs')
    n = sum(support)
    pe = np.dot(support, predicted) / n**2
    close((tp.sum()/n-pe)/(1-pe), record['kappa'], 'Kappa differs')
    return dict(confusion_counts=cm.tolist(), support=list(support), prediction_counts=predicted.tolist(),
                true_positive=tp.tolist(), false_positive=(predicted-tp).tolist(),
                false_negative=(np.asarray(support)-tp).tolist(),
                macro={m:float(v.mean()) for m,v in values.items()},
                per_class={m:v.tolist() for m,v in values.items()}, kappa=record['kappa'])


def analyze(common, baseline, previous, common_protocol, baseline_protocol):
    cp, bp = common_protocol['specification'], baseline_protocol['specification']
    for key, report in [('common', common), ('baseline', baseline)]:
        require(report['status'] == 'complete' and report['producer_commit'] == COMMITS[key]
                and report['protocol_sha256'] == PINS[key+'_protocol'] and report['plan_sha256'] == PLANS[key],
                'Unaccepted report or producer identity')
    require(common['validation'] == dict(complete_jobs=20, estimator_replays=20, output_hashes=60), 'Common audit incomplete')
    require(baseline['validation'] == dict(complete_folds=5, estimator_replays=20, output_hashes=60,
            classical_fits=20, majority_records=5, class_prior_records=25), 'Classical audit incomplete')
    require(common['baseline_protocol_sha256'] == cp['baseline_protocol']['sha256'] == PINS['baseline_protocol'], 'Baseline protocol binding differs')
    require(common['native_audit_sha256'] == cp['native_audit']['sha256'] == bp['frozen_reference_audit_sha256']
            == previous['inputs']['native_audit_sha256'], 'Native audit binding differs')
    require(previous['inputs']['common_audit_sha256'] == PINS['common'], 'Prior recipe comparison differs')
    require(cp['catalog']['sha256'] == bp['catalog_sha256'] == baseline['catalog_sha256'] == previous['catalog_sha256'], 'Catalog differs')
    require(cp['class_names'] == bp['class_names'] == baseline['class_names'] == [c.lower() for c in CLASSES], 'Class order differs')
    require(bp['recipe']['feature_columns'] == dict(zip(CLASSICAL, ([0,13],[13,45],[45,1865],[0,1865]))),
            'Classical feature columns differ from displayed dimensions')
    for folds in (cp['folds'], bp['folds'], baseline['folds']):
        require([f['fold'] for f in folds] == list(range(5)), 'Fold coverage or order differs')
    for f, (c, b, audited) in enumerate(zip(cp['folds'], bp['folds'], baseline['folds'])):
        for name in ('counts', 'normalization_sha256', 'train_ids_sha256', 'test_ids_sha256'):
            require(c[name] == b[name], 'Matched protocol field differs: '+name)
        require(c['split']['sha256'] == b['split_sha256'], 'Split differs')
        require(audited['counts'] == b['counts'] and audited['normalization_sha256'] == b['normalization_sha256'], 'Audited fold identity differs')
        require(c['counts']['test']['class_counts'] == previous['support_by_fold'][f], 'Prior support differs')
        require(set(audited['classical']) == set(CLASSICAL), 'Classical feature family coverage differs')
    jobs = {(r['model'], r['fold']): r for r in common['folds']}
    require(len(common['folds']) == len(jobs) == 20 and set(jobs) == {(m,f) for m in MODELS for f in range(5)}, 'Common feature family coverage differs')
    joint = dict(family_order=list(FAMILIES), class_names=list(CLASSES), families={}, contrasts={},
                 support_by_fold=previous['support_by_fold'], test_support=previous['test_support'],
                 fold_identities=[{k:v for k,v in f.items() if k != 'normalization_provenance'} for f in cp['folds']],
                 feature_columns=bp['recipe']['feature_columns'], resolved_parameters=cp['resolved_parameters'],
                 aggregation='One fit per family and fold. All six classes enter each macro metric. Equal-fold mean and sample SD; zero undefined precision. Paired contrasts are descriptive, with overlapping training sets.',
                 validation=dict(classical_fits=20, encoder_fits=20, reconstructed_matrices=0, estimator_replays_in_source_audits=40))
    for family, name, dimension in zip(FAMILIES, NAMES, DIMENSIONS):
        folds=[]
        for fold in range(5):
            record=jobs[family,fold] if family in MODELS else baseline['folds'][fold]['classical'][family]
            if family in MODELS:
                require(record['counts'] == cp['folds'][fold]['counts'], 'Encoder membership counts differ')
            require(record['estimator_replay']['verified'] is True, 'Unreplayed estimator')
            fit=record['fit']
            require(fit['classifier_parameters'] == cp['resolved_parameters']['classifier']
                    and fit['scaler_parameters'] == cp['resolved_parameters']['scaler'], 'Classifier or scaler recipe differs')
            require(len(fit['n_iter']) == 1 and type(fit['n_iter'][0]) is int and 0 < fit['n_iter'][0] <= 2000, 'Invalid iteration count')
            require(fit['iteration_limit_reached'] == (fit['n_iter'][0] >= 2000), 'Iteration cap flag differs')
            require(fit['convergence_warning'] == any(w['category'] == 'ConvergenceWarning' for w in fit['fit_warnings']), 'Warning flag differs')
            metrics=reconstruct(record, previous['support_by_fold'][fold])
            folds.append(dict(fold=fold, fit=fit, **metrics))
            joint['validation']['reconstructed_matrices'] += 1
        data=dict(name=name, dimensions=dimension, folds=folds,
                  macro={m:describe([f['macro'][m] for f in folds]) for m in METRICS},
                  per_class={m:describe([f['per_class'][m] for f in folds]) for m in METRICS},
                  kappa=describe([f['kappa'] for f in folds]),
                  warnings=[dict(fold=f['fold'], messages=f['fit']['fit_warnings']) for f in folds if f['fit']['fit_warnings']],
                  capped_folds=[f['fold'] for f in folds if f['fit']['iteration_limit_reached']])
        for m in METRICS:
            expected=(common if family in MODELS else baseline)['methods'][family][m]
            for key in ('mean','sample_std','fold_values'):
                close(data['macro'][m][key], expected[key], 'Accepted aggregate differs')
        for field in ('confusion_counts','true_positive','false_positive','false_negative','prediction_counts'):
            data['pooled_'+field]=np.sum([f[field] for f in folds],axis=0).tolist()
        joint['families'][family]=data
    for first, second in itertools.combinations(FAMILIES, 2):
        a,b=(joint['families'][k] for k in (first,second))
        contrast=dict(first=first, second=second, definition='first minus second, percentage points', macro={}, per_class={})
        for m in METRICS:
            delta=100*(np.array(a['macro'][m]['fold_values'])-b['macro'][m]['fold_values'])
            contrast['macro'][m]=dict(**describe(delta),positive_folds=int(np.sum(delta>1e-12)),
                                      negative_folds=int(np.sum(delta < -1e-12)),tie_folds=int(np.sum(np.abs(delta)<=1e-12)))
            contrast['per_class'][m]=describe(100*(np.array(a['per_class'][m]['fold_values'])-b['per_class'][m]['fold_values']))
        joint['contrasts'][first+'__minus__'+second]=contrast
    joint['controls']={}
    for name in ('majority','class_prior_random'):
        values=[]
        for f in baseline['folds']:
            records=[f['controls'][name]] if name=='majority' else f['controls'][name]
            require(len(records)==(1 if name=='majority' else 5), 'Control coverage differs')
            if name=='class_prior_random':
                require([r['seed'] for r in records]==list(range(5)), 'Control seed coverage differs')
            checked=[reconstruct(r,f['counts']['test']['class_counts']) for r in records]
            values.append({m:float(np.mean([r['macro'][m] for r in checked])) for m in METRICS})
        joint['controls'][name]={m:describe([v[m] for v in values]) for m in METRICS}
        for m in METRICS:
            for k in ('mean','sample_std','fold_values'):
                close(joint['controls'][name][m][k],baseline['methods'][name][m][k], 'Control aggregate differs')
    joint['validation']['reconstructed_control_matrices']=30
    result=json.loads(json.dumps(previous))
    result['joint_fixed_classifier']=joint
    result['joint_classical_comparison']=dict(status='accepted',classical_results_read=True)
    result['inputs'].update(baseline_audit_sha256=PINS['baseline'],baseline_protocol_sha256=PINS['baseline_protocol'],
                            prior_recipe_analysis_sha256=PINS['recipe_analysis'],joint_analysis_source_sha256=sha(__file__))
    return result


def tables(report, directory):
    joint=report['joint_fixed_classifier']
    fields=['feature_family','dimensions','macro_precision_percent','macro_recall_percent','macro_f1_percent',
            'macro_f1_sample_sd_pp','iterations_by_fold','warning_folds','capped_folds']
    lines=['| Feature family | Dimensions | Macro P, % | Macro R, % | Macro F1, % | Fold SD, pp | Iterations, folds 0 to 4 | Warnings / caps |',
           '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    with (directory/'repaired_common_classifier_summary.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key in FAMILIES:
            d=joint['families'][key];rates=[100*d['macro'][m]['mean'] for m in METRICS]
            iterations=[f['fit']['n_iter'][0] for f in d['folds']]
            writer.writerow(dict(zip(fields,[d['name'],d['dimensions'],*rates,100*d['macro']['f1']['sample_std'],
                                              json.dumps(iterations),len(d['warnings']),len(d['capped_folds'])])))
            lines.append('| '+d['name']+f" | {d['dimensions']} | "+' | '.join(f'{v:.2f}' for v in rates)
                         +f" | {100*d['macro']['f1']['sample_std']:.2f} | "+', '.join(map(str,iterations))
                         +f" | {len(d['warnings'])} / {len(d['capped_folds'])} |")
    (directory/'repaired_common_classifier_summary.md').write_text('\n'.join(lines)+'\n')
    with (directory/'repaired_common_classifier_classes.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['feature_family','class','unique_test_support','precision_percent','recall_percent','f1_percent','f1_fold_sd_pp','pooled_tp','pooled_fp','pooled_fn'])
        for key in FAMILIES:
            d=joint['families'][key]
            for k,name in enumerate(CLASSES):
                writer.writerow([d['name'],name,joint['test_support'][k],*[100*d['per_class'][m]['mean'][k] for m in METRICS],
                                 100*d['per_class']['f1']['sample_std'][k],d['pooled_true_positive'][k],d['pooled_false_positive'][k],d['pooled_false_negative'][k]])


def figures(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    joint=report['joint_fixed_classifier'];families=joint['families']
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8C6D31'];markers=['o','s','^','v','D']
    fig=plt.figure(figsize=(9.1,6.7))
    grid=fig.add_gridspec(1,2,left=.245,right=.945,bottom=.28,top=.77,width_ratios=[1,1.23],wspace=.12)
    left,right=fig.add_subplot(grid[0]),fig.add_subplot(grid[1])
    y=np.arange(8)
    for row,key in enumerate(FAMILIES):
        d=families[key]
        for fold,v in enumerate(d['macro']['f1']['fold_values']):
            left.scatter(100*v,row+(fold-2)*.09,c=colors[fold],marker=markers[fold],s=29,zorder=3)
        left.scatter(100*d['macro']['f1']['mean'],row,c='black',marker='|',s=240,linewidths=2.4,zorder=4)
    left.set(yticks=y,yticklabels=[f'{name}\n[{dim:,}]' for name,dim in zip(NAMES,DIMENSIONS)],xlim=(0,50),ylim=(7.6,-.6),xlabel='Macro F1, %')
    left.grid(axis='x',color='.9',linewidth=.7);left.set_axisbelow(True)
    fig.text(.245,.955,'A  Same classifier recipe',fontsize=11)
    matrix=np.array([families[k]['per_class']['f1']['mean'] for k in FAMILIES])*100
    right.imshow(matrix,cmap='Blues',vmin=0,vmax=80,aspect='auto');right.set_ylim(7.6,-.6)
    right.set(yticks=[],xticks=np.arange(6),xticklabels=[f'{name}\n{n:,}' for name,n in zip(CLASSES,joint['test_support'])])
    right.xaxis.tick_top();right.tick_params(axis='x',length=0,pad=5,labelsize=10.5)
    plt.setp(right.get_xticklabels(),rotation=40,ha='left',rotation_mode='anchor')
    fig.text(.58,.955,'B  Class F1, %',fontsize=11)
    for row in range(8):
        for col in range(6):
            right.text(col,row,f'{matrix[row,col]:.1f}',ha='center',va='center',fontsize=10.5,color='white' if matrix[row,col]>45 else '#17232f')
    handles=[Line2D([],[],color=colors[f],marker=markers[f],linestyle='',label=f'Fold {f}',markersize=5) for f in range(5)]
    handles.append(Line2D([],[],color='black',marker='|',linestyle='',label='Mean',markersize=10,markeredgewidth=2))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.145),ncol=6,frameon=False,fontsize=10,columnspacing=1.1,handletextpad=.35)
    warnings=sum(len(d['warnings']) for d in families.values());caps=sum(len(d['capped_folds']) for d in families.values())
    fig.text(.02,.095,'Brackets give feature dimensions. Class headers give unique test support.\n'
             f'40 fits; {warnings} fits with warnings; {caps} at the 2,000-iteration cap. All five folds retained.',fontsize=10.5)
    fig.text(.02,.035,'Map agreement on geographic test folds. Fold spread is descriptive; training partitions overlap.',fontsize=10.5)
    save_figure(fig,directory/'repaired_common_classifier_eight_families_20260930')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.3))
    fig.subplots_adjust(left=.2,right=.97,bottom=.29,top=.84)
    for row,key in enumerate(MODELS):
        values=report['models'][key]['delta_pp']['macro']['f1']['fold_values']
        for fold,v in enumerate(values):
            ax.scatter(v,row+(fold-2)*.085,c=colors[fold],marker=markers[fold],s=32,zorder=3)
        ax.scatter(np.mean(values),row,c='black',marker='|',s=240,linewidths=2.4,zorder=4)
    ax.axvline(0,color='.5',linewidth=.8)
    ax.set(yticks=range(4),yticklabels=NAMES[:4],ylim=(3.5,-.5),xlim=(-4,26),xlabel='Common classifier minus original SGD macro F1, pp')
    ax.grid(axis='x',color='.9');ax.set_axisbelow(True)
    ax.set_title('Sensitivity to the complete classifier recipe',loc='left',fontsize=11)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.015),ncol=6,frameon=False,fontsize=10,columnspacing=.7,handletextpad=.3)
    save_figure(fig,directory/'repaired_classifier_recipe_sensitivity_20260930');plt.close(fig)
    return dict(matplotlib=matplotlib.__version__,numpy=np.__version__)


def save_figure(fig, stem):
    for suffix in ('.pdf','.png'):
        options=dict(metadata={'CreationDate':None,'ModDate':None}) if suffix=='.pdf' else {}
        fig.savefig(stem.with_suffix(suffix),**options)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in (*PINS,'output_dir'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    inputs={k:read_bound(getattr(args,k),v) for k,v in PINS.items()}
    report=analyze(inputs['common'],inputs['baseline'],inputs['recipe_analysis'],inputs['common_protocol'],inputs['baseline_protocol'])
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(),'Output directory must be new')
    args.output_dir.mkdir(parents=True)
    tables(report,args.output_dir)
    versions=figures(report,args.output_dir)
    report['joint_fixed_classifier']['rendering_versions']=versions
    report['joint_fixed_classifier']['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(args.output_dir.iterdir())}
    with (args.output_dir/'COMMON_CLASSIFIER_RESULTS_2026-09-29.json').open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


if __name__=='__main__':
    main()
