#!/usr/bin/env python3
"""Reconstruct saved map counts and compare them with the census totals.

The count functions reuse the separate census checker; see NOTICE.md.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

N = 23380
CLASSES = ('plains', 'dunes', 'hummocky', 'labyrinths', 'lakes', 'craters')
MODES = ('baseline', 'weighted', 'all_four', 'any_valid')
ROLES = ('train', 'val', 'test', 'buffer_excluded', 'selk_holdout')
HASHES = {
    'counts.json': '302f12413c0a92cf0b8152383bf30765adac57d05719ff2609852efdab037126',
    'map_cells.npz': 'd6511e2ffb01cd6fc5f5e78b68f4d9ae7fe7ecf6176fb67aea4f0e756359ab96',
}
CONVENTIONS = {
    'file': 'map_cells.npz', 'class_order': list(CLASSES), 'mode_order': list(MODES),
    'role_order': list(ROLES), 'sar_mask_shape': [128, 128],
    'sar_mask_flat_order': 'C', 'sar_mask_bitorder': 'little',
    'weighted_units': 'integer SAR-valid samples per nominal map cell; range 0 to 4',
    'other_modes_units': 'map cells',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def schema(n):
    return {
        'tile_ids': ((n,), '<U12'), 'map_grids': ((n, 64, 64), 'uint8'),
        'sar_valid_packed': ((n, 2048), 'uint8'), 'roles': ((n, 5), 'uint8'),
        'modal_labels': ((n,), 'uint8'), 'counts': ((n, 4, 6), '<i8'),
        'unknown': ((n, 4), '<i8'), 'weight_histograms': ((n, 5), '<i8'),
    }


def bound_inputs(directory):
    """Hash both files before parsing either; decode these same in-memory bytes."""
    payloads = {}
    for name, expected in HASHES.items():
        raw = (directory / name).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == expected, f'{name}: SHA256 mismatch: {actual} != {expected}')
        payloads[name] = raw
    return payloads


def load_arrays(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(len(names) == 8 and set(names) == {k + '.npy' for k in schema(N)},
                'NPZ: missing, extra or duplicate array members')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in schema(N)}


def validate_arrays(arrays, n):
    require(set(arrays) == set(schema(n)), 'array keys differ from schema')
    for key, (shape, dtype) in schema(n).items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == np.dtype(dtype),
                f'{key}: expected shape {shape}, dtype {dtype}; '
                f'got {value.shape}, {value.dtype}')
    ids = arrays['tile_ids'].tolist()
    require(len(set(ids)) == n and all(re.fullmatch(r'titan_[0-9]{6}', x) for x in ids),
            'tile_ids: expected complete unique titan_XXXXXX IDs')
    require(np.all(arrays['roles'] < 5), 'roles: codes must be 0..4')
    require(np.all(arrays['modal_labels'] < 6), 'modal_labels: codes must be 0..5')
    for key in ('counts', 'unknown', 'weight_histograms'):
        require(np.all(arrays[key] >= 0), f'{key}: negative saved values')


def same_array(name, actual, expected):
    require(actual.shape == expected.shape, f'{name}: shape mismatch')
    unequal = actual != expected
    if np.any(unequal):
        index = tuple(np.unravel_index(int(np.argmax(unequal)), unequal.shape))
        raise ValueError(f'{name}{index}: reconstructed {actual[index]}, saved {expected[index]}')


def same_json(actual, expected, path='counts'):
    """Exact recursive equality, including JSON types; report the first difference."""
    require(type(actual) is type(expected), f'{path}: JSON type mismatch')
    if isinstance(actual, dict):
        require(actual.keys() == expected.keys(), f'{path}: JSON keys mismatch')
        for key in actual:
            same_json(actual[key], expected[key], f'{path}/{key}')
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f'{path}: JSON length mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            same_json(a, b, f'{path}/{i}')
    else:
        require(actual == expected, f'{path}: reconstructed {actual!r}, expected {expected!r}')


# The following two functions are copied verbatim from the attributed checker.
def independent_counts(grids, packed):
    """Axis order: tile, mode, class. Slice additions avoid reshape grouping."""
    valid = np.unpackbits(packed, axis=1, count=128 * 128, bitorder='little')
    valid = valid.reshape((-1, 128, 128), order='C')
    weights = (valid[:, 0::2, 0::2] + valid[:, 0::2, 1::2] +
               valid[:, 1::2, 0::2] + valid[:, 1::2, 1::2])
    # Category 6 is unknown, never a seventh terrain class.
    categories = np.stack([grids == c for c in (*range(6), 255)], axis=1)
    measured = np.empty((len(grids), 4, 7), dtype=np.int64)
    for m, eligibility in enumerate((np.ones_like(weights), weights,
                                     weights == 4, weights > 0)):
        measured[:, m, :] = np.sum(categories * eligibility[:, None, :, :],
                                  axis=(2, 3), dtype=np.int64)
    hist = np.stack([(weights == k).sum(axis=(1, 2), dtype=np.int64)
                     for k in range(5)], axis=1)
    return valid, weights, measured[:, :, :6], measured[:, :, 6], hist


def summary(counts, unknown, labels, selected):
    selected = np.asarray(selected, dtype=bool)
    subset = counts[selected]
    modal = labels[selected]
    result = {'images': int(selected.sum()),
              'modal_class_images': [int(np.sum(modal == c)) for c in range(6)], 'modes': {}}
    for m, mode in enumerate(MODES):
        cells = subset[:, m, :]
        result['modes'][mode] = {
            'class_counts': cells.sum(axis=0, dtype=np.int64).tolist(),
            'known_total': int(cells.sum(dtype=np.int64)),
            'unknown_total': int(unknown[selected, m].sum(dtype=np.int64)),
            'images_containing_class': (cells > 0).sum(axis=0).tolist(),
            # Rows are catalog modal classes; columns are dense map classes.
            'class_counts_by_modal_class': [cells[modal == c].sum(axis=0).tolist()
                                           for c in range(6)]}
    return result


def reconstruct(arrays, n=N):
    """Stored counts/unknown/histograms are comparison targets, never inputs to sums."""
    validate_arrays(arrays, n)
    counts = np.empty((n, 4, 6), dtype=np.int64)
    unknown = np.empty((n, 4), dtype=np.int64)
    labels = np.empty(n, dtype=np.uint8)
    for first in range(0, n, 128):
        last = min(first + 128, n)
        grids = arrays['map_grids'][first:last]
        require(np.all((grids < 6) | (grids == 255)),
                f'map_grids[{first}:{last}]: codes must be 0..5 or unknown 255')
        _, weights, c, u, h = independent_counts(grids, arrays['sar_valid_packed'][first:last])
        counts[first:last], unknown[first:last] = c, u
        for name, measured in (('counts', c), ('unknown', u), ('weight_histograms', h)):
            same_array(f'{name}[{first}:{last}]', measured, arrays[name][first:last])
        denominators = np.column_stack((np.full(last-first, 4096),
            weights.sum(axis=(1, 2)), (weights == 4).sum(axis=(1, 2)),
            (weights > 0).sum(axis=(1, 2))))
        same_array('known plus unknown denominators', c.sum(axis=2) + u, denominators)
        same_array('histogram cell totals', h.sum(axis=1), np.full(last-first, 4096))
        require(np.all(c[:, 0].sum(axis=1) > 0), 'modal label has no known map cells')
        labels[first:last] = np.argmax(c[:, 0], axis=1)
        same_array(f'modal_labels[{first}:{last}]', labels[first:last],
                   arrays['modal_labels'][first:last])
    groups = {
        'catalog': summary(counts, unknown, labels, np.ones(n, dtype=bool)),
        'folds': [{'fold': f, 'roles': {
            role: summary(counts, unknown, labels, arrays['roles'][:, f] == r)
            for r, role in enumerate(ROLES)}} for f in range(5)],
    }
    return groups, {
        'images': n, 'map_cells': n * 4096, 'saved_mask_bits': n * 16384,
        'role_assignments': n * 5, 'groups': 26, 'mode_summaries': 104,
        'unknown_map_images': int(np.count_nonzero(unknown[:, 0])),
        'maximum_unknown_map_cells_per_image': int(unknown[:, 0].max()),
    }


def check(directory):
    raw = bound_inputs(directory)
    expected = json.loads(raw['counts.json'])
    same_json(CONVENTIONS, expected['array_conventions'], 'array_conventions')
    arrays = load_arrays(raw['map_cells.npz'])
    measured, scope = reconstruct(arrays, n=N)
    same_json(measured, expected['counts'])
    return {'status': 'passed', 'input_sha256': dict(HASHES), 'checked': scope,
            'reconstructed_counts': measured}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='directory containing the two exact accepted files')
    parser.add_argument('--output', type=Path,
                        help='new JSON report file; preserves failures, refuses overwrite')
    args = parser.parse_args(argv)
    output = None
    try:
        if args.output:
            output = args.output.open('x', encoding='utf-8')
    except OSError as error:
        print(f'Cannot create report: {error}', file=sys.stderr)
        return 1
    code = 0
    try:
        report = check(args.data_dir)
    except (Exception, KeyboardInterrupt) as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
    report.update(python=sys.version.split()[0], numpy=np.__version__,
                  scope='Accepted saved arrays only; no original SAR, raster, catalog or split rejoin.')
    try:
        if output:
            with output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write('\n')
            print(json.dumps({k: v for k, v in report.items() if k != 'reconstructed_counts'}))
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except OSError as error:
        print(f'Cannot write report: {error}', file=sys.stderr)
        return 1
    return code


if __name__ == '__main__':
    sys.exit(main())
