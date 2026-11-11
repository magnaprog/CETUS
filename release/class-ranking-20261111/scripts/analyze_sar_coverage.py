"""Compare saved Titan predictions by SAR coverage, without training or inference."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

import numpy as np

CLASSES = ('plains', 'dunes', 'hummocky', 'labyrinths', 'lakes', 'craters')
ENCODERS = ('dinov2', 'dofa', 'croma', 'random_init')
CLASSICAL = ('raw_intensity_statistics', 'intensity_histogram',
             'texture_glcm_lbp_gabor_hog_wavelet', 'combined_classical')
GROUPS = ('all', 'complete', 'partial')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scores(truth, predicted):
    """Keep absent classes visible; six-class macro scores require all six."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(truth.ndim == predicted.ndim == 1 and truth.shape == predicted.shape,
            'Prediction shape differs')
    require(truth.dtype.kind in 'iu' and predicted.dtype.kind in 'iu'
            and np.isin(truth, range(6)).all() and np.isin(predicted, range(6)).all(),
            'Integer class indices 0 through 5 required')
    cm = np.bincount(6 * truth + predicted, minlength=36).reshape(6, 6)
    support, predicted_counts, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    result = {'n': len(truth), 'confusion': cm.tolist(), 'support': support.tolist(),
              'predicted_counts': predicted_counts.tolist(),
              'absent_classes': np.flatnonzero(support == 0).tolist()}
    for name, top, bottom in [('precision', tp, predicted_counts),
                              ('recall', tp, support), ('f1', 2 * tp, support + predicted_counts)]:
        values = np.divide(top, bottom, out=np.zeros(6), where=bottom != 0)
        result[name] = [float(v) if bottom[c] else None for c, v in enumerate(values)]
        result['macro_' + name] = float(values.mean()) if (support > 0).all() else None
    return result


def unique(rows, fields, expected, name):
    keyed = {tuple(row[k] for k in fields): row for row in rows}
    require(len(keyed) == len(rows) and set(keyed) == set(expected), name + ' coverage differs')
    return keyed


def analyze(catalog, splits, common, training):
    require(tuple(catalog['class_names']) == CLASSES, 'Class order differs')
    entries = catalog['tiles']
    by_id = {t['tile_id']: t for t in entries}
    require(len(by_id) == len(entries) and entries, 'Duplicate or empty catalog')
    for t in entries:
        require(type(t['label']) is int and t['label'] in range(6), 'Invalid catalog label')
        v = t['valid_fraction']
        require(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1,
                'Invalid SAR fraction')
    lr = unique(common['folds'], ('fold',), [(f,) for f in range(5)], 'Linear fits')
    ft = unique(training['test'], ('model', 'fold', 'seed'),
                [(m, f, s) for m in ENCODERS[:3] for f in range(5) for s in range(5)],
                'Partial training')
    require(len(splits) == 5, 'Five splits required')
    rows, populations, seen = [], [], set()
    for fold, split in enumerate(splits):
        require(split['policy']['test_fold'] == fold, 'Fold identity differs')
        require(set(split['assignments']) == set(by_id), 'Split catalog differs')
        ids = [t['tile_id'] for t in entries if split['assignments'][t['tile_id']] == 'test']
        require(ids and not seen.intersection(ids), 'Missing or repeated test tiles')
        seen.update(ids)
        truth = np.asarray([by_id[t]['label'] for t in ids], dtype=np.int64)
        fraction = np.asarray([by_id[t]['valid_fraction'] for t in ids])
        masks = {'all': np.ones(len(ids), dtype=bool), 'complete': fraction == 1,
                 'partial': fraction < 1}
        populations.append({'fold': fold, 'tile_ids': {g: [t for t, yes in zip(ids, mask) if yes]
                                                     for g, mask in masks.items()}})
        record = lr[(fold,)]
        require(record['test_ids'] == ids and record['true_labels'] == truth.tolist(),
                'Linear prediction identities differ')
        require(set(record['common']) == set(ENCODERS)
                and set(record['classical']) == set(CLASSICAL), 'Feature families differ')
        jobs = [('linear', m, None, p) for m, p in
                (list(record['common'].items()) + list(record['classical'].items()))]
        for model in ENCODERS[:3]:
            for seed in range(5):
                item = ft[(model, fold, seed)]
                # The accepted package stores these vectors in the catalog test order;
                # its pinned bytes bind that order to the original export verification.
                require(item['family'] == 'adaptation', 'Partial-training family differs')
                jobs.append(('partial_training', model, seed, item['predictions']))
        for family, model, seed, predictions in jobs:
            predictions = np.asarray(predictions)
            # Check the whole vector before indexing any subset.
            all_scores = scores(truth, predictions)
            result = {'all': all_scores}
            result.update({g: scores(truth[mask], predictions[mask])
                           for g, mask in masks.items() if g != 'all'})
            rows.append({'family': family, 'model': model, 'fold': fold,
                         'seed': seed, 'scores': result})
    summaries = []
    for family, model in sorted({(r['family'], r['model']) for r in rows}):
        selected = [r for r in rows if (r['family'], r['model']) == (family, model)]
        for group in GROUPS:
            means = []
            for fold in range(5):
                vals = [r['scores'][group]['macro_f1'] for r in selected if r['fold'] == fold]
                means.append(float(np.mean(vals)) if all(v is not None for v in vals) else None)
            complete = all(v is not None for v in means)
            summaries.append({'family': family, 'model': model, 'group': group,
                              'fold_mean_macro_f1': means,
                              'mean_macro_f1': float(np.mean(means)) if complete else None,
                              'sample_sd': float(np.std(means, ddof=1)) if complete else None})
    return {'class_names': list(CLASSES), 'test_images': len(seen),
            'populations': populations, 'results': rows, 'summaries': summaries,
            'interpretation': 'Coverage subsets change class and geographic composition. '
                              'Differences do not isolate effects of missing pixels. '
                              'Undefined ratios are null; aggregate metrics retain the original '
                              'zero-division convention. Six-class macro F1 requires '
                              'all six classes in each region. Seeds are averaged before regions. '
                              'Region SD is not a population confidence interval.'}


def load_package(path, expected_sha256):
    require(digest(path) == expected_sha256, 'Scientific supplement hash differs')
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        def read(name):
            data = bundle.read(name)
            return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)
        catalog = read('data/catalog.json')
        require(len(catalog['tiles']) == 23380, 'Unexpected Titan catalog size')
        splits = [read(f'data/splits/titan_contiguous_fold_{f}.json') for f in range(5)]
        cat_hash = hashlib.sha256(bundle.read('data/catalog.json')).hexdigest()
        require(all(s['catalog_sha256'] == cat_hash for s in splits), 'Split catalog hash differs')
        return (catalog, splits, read('results/common_predictions.json.gz'),
                read('results/partial_training_predictions.json.gz'))


def write_result(path, result):
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    # A failed write cannot leave a partial result; linking refuses replacement.
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--sha256', required=True, help='Expected accepted package digest')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(*load_package(args.package, args.sha256))
    require(result['test_images'] == 23246 and len(result['results']) == 115,
            'Unexpected evaluation population')
    result['execution'] = {'package_sha256': args.sha256, 'script_sha256': digest(__file__),
                           'python': platform.python_version(), 'numpy': np.__version__,
                           'argv': sys.argv, 'model_execution': False}
    write_result(args.output, result)
    print(json.dumps({'test_images': result['test_images'], 'prediction_vectors': len(result['results'])}))


if __name__ == '__main__':
    main()
