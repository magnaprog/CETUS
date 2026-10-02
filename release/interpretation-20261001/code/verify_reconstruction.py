"""Reconstruct numerical results from released evidence; keep outputs external.

This packaging entry point calls the exported scientific functions directly, so
an unpacked source export needs neither a development checkout nor Git metadata.
"""

import argparse
import csv
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True

from scripts import analyze_repaired_interpretation as interpretation
from scripts import plot_repaired_class_changes as classes
from scripts import plot_repaired_examples as examples

PACKAGE = Path(__file__).resolve().parents[1]
INPUT = 'longitude-adaptation-controls-20260930/results/adaptation.json'


def compare(actual, expected, path='value'):
    """Exact structure/counts; absolute tolerances only for floating quantities."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Keys differ: ' + path)
        for key, value in expected.items():
            compare(actual[key], value, path + '/' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Length differs: ' + path)
        for i, value in enumerate(expected):
            compare(actual[i], value, path + '/' + str(i))
    elif isinstance(expected, float):
        tolerance = 1e-6 if path.endswith('/nearest_reinstated_center_m') else 1e-12
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isfinite(expected) or abs(actual - expected) > tolerance):
            raise ValueError('Number differs: ' + path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('Value differs: ' + path)


def compare_csv(path, rows):
    with path.open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise ValueError('CSV row count differs: ' + path.name)
    for i, (record, expected) in enumerate(zip(saved, rows)):
        if set(record) != set(expected):
            raise ValueError('CSV columns differ: ' + path.name)
        for key, value in expected.items():
            text = record[key]
            actual = (float(text) if type(value) is float else int(text) if type(value) is int
                      else text == 'True' if type(value) is bool and text in ('True', 'False')
                      else None if value is None and text == '' else text)
            compare(actual, value, f'{path.name}/{i}/{key}')


def check_files(manifest):
    for group in ('files', 'artifacts'):
        for name, row in manifest[group].items():
            if examples.sha256(PACKAGE / name) != row['sha256']:
                raise ValueError('Export SHA256 differs: ' + name)


def runtime_versions():
    return dict(python=platform.python_version(), numpy=version('numpy'), matplotlib=version('matplotlib'))


def exact_runtime(manifest):
    compare(runtime_versions(), manifest['figure_runtime'], 'figure_runtime')
    for line in (PACKAGE / 'code/requirements.txt').read_text().splitlines():
        package, pinned = line.split()[0].split('==')
        if version(package) != pinned:
            raise ValueError('Exact figures require the pinned runtime: ' + package)


def verify(release, output, tiles=None, exact_figures=False):
    release, output = release.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (PACKAGE, release)):
        raise ValueError('Output must be new and disjoint from source and release inputs')
    if tiles is not None:
        tiles = tiles.resolve()
        if output.is_relative_to(tiles) or tiles.is_relative_to(output):
            raise ValueError('Output overlaps supplied arrays')
    manifest = interpretation.base.read_json(PACKAGE / 'source_export.json')
    check_files(manifest)
    if exact_figures:
        if tiles is None:
            raise ValueError('Exact figure comparison also requires --tiles-dir')
        exact_runtime(manifest)
    data = {name: interpretation.load_bound(release / name, digest)
            for name, digest in interpretation.PINS.items()}
    computed, tables = interpretation.analyze(data)
    compare(computed, interpretation.base.read_json(PACKAGE / 'interpretation/analysis.json'), 'analysis')
    for name, rows in tables.items():
        compare_csv(PACKAGE / 'interpretation' / (name + '.csv'), rows)

    expected_classes = interpretation.base.read_json(PACKAGE / 'class-changes/class_changes.json')
    compare(expected_classes['input'], dict(path=INPUT, sha256=interpretation.PINS[INPUT]), 'class_input')
    class_values = classes.class_changes(data[INPUT])
    compare(class_values, {key: expected_classes[key] for key in class_values}, 'class_changes')
    catalog = release / 'longitude-review-20260929/catalogs/titan_catalog.json'
    protocol = release / 'longitude-review-20260929/splits/protocol.json'
    selected = examples.selection_manifest(catalog, protocol)
    compare(selected, interpretation.base.read_json(PACKAGE / 'examples/selected.json'), 'selection')
    example_manifest = interpretation.base.read_json(PACKAGE / 'examples/repaired_examples_manifest.json')
    compare(selected, example_manifest['selection'], 'render_selection')
    if (examples.sha256(PACKAGE / 'examples/selected.json') != example_manifest['selection_sha256']
            or examples.sha256(Path(examples.__file__)) != example_manifest['renderer_sha256']):
        raise ValueError('Example selection or renderer binding differs')

    output.mkdir(parents=True)
    derived = output / 'interpretation'
    derived.mkdir()
    interpretation.write_outputs(derived, computed, tables)
    for path in derived.glob('*.tex'):
        if path.read_bytes() != (PACKAGE / 'interpretation' / path.name).read_bytes():
            raise ValueError('TeX display differs: ' + path.name)
    # The original CLI also verifies source hashes and emits the class CSV.
    if classes.main(['--analysis', str(release / INPUT), '--analysis-sha256', interpretation.PINS[INPUT],
                     '--output-dir', str(output / 'class-changes')]) != 0:
        raise ValueError('Class figure reconstruction failed')
    rebuilt = interpretation.base.read_json(output / 'class-changes/class_changes.json')
    compare(rebuilt['source_sha256'], expected_classes['source_sha256'], 'class_sources')
    with (output / 'class-changes/class_changes.csv').open(newline='') as stream:
        rebuilt_rows = list(csv.DictReader(stream))
    with (PACKAGE / 'class-changes/class_changes.csv').open(newline='') as stream:
        saved_rows = list(csv.DictReader(stream))
    if len(rebuilt_rows) != len(saved_rows):
        raise ValueError('Class CSV row count differs')
    numeric = ('ft_mean_f1_percent', 'lr_mean_f1_percent', 'difference_mean_pp', 'difference_sample_sd_pp',
               *(f'difference_fold_{f}_pp' for f in range(5)))
    for a, b in zip(rebuilt_rows, saved_rows):
        # Runtime is recorded honestly, but is not a scientific comparison field.
        for row in (a, b):
            for key in runtime_versions():
                row.pop(key)
            for key in numeric:
                row[key] = float(row[key])
        compare(a, b, 'class_csv')

    examples.write_json(output / 'selected.json', selected)
    if tiles is not None:
        if examples.main(['render', '--catalog', str(catalog), '--protocol', str(protocol),
                          '--selection', str(output / 'selected.json'), '--tiles-dir', str(tiles),
                          '--output-dir', str(output / 'examples')]) != 0:
            raise ValueError('Example rendering failed')
        rendered = interpretation.base.read_json(output / 'examples/repaired_examples_manifest.json')
        for key in ('schema_version', 'selection', 'selection_sha256', 'renderer_sha256', 'display', 'tiles'):
            compare(rendered[key], example_manifest[key], 'example_manifest/' + key)
    if exact_figures:
        for folder, stem in [('class-changes', 'class_changes'), ('examples', 'repaired_examples')]:
            for suffix in ('.pdf', '.png'):
                name = folder + '/' + stem + suffix
                if examples.sha256(output / name) != manifest['artifacts'][name]['sha256']:
                    raise ValueError('Figure bytes differ in the pinned runtime: ' + name)

    check_files(manifest)
    for name, digest in interpretation.PINS.items():
        if examples.sha256(release / name) != digest:
            raise ValueError('Input changed during reconstruction: ' + name)
    result = dict(status='pass', producer_commit=manifest['producer_commit'], runtime=runtime_versions(),
                  interpretation_csv_rows={name: len(rows) for name, rows in tables.items()},
                  class_fold_contrasts=90, examples_selected=len(selected['selected']),
                  examples_rerendered=tiles is not None, exact_figure_bytes=exact_figures,
                  numerical_absolute_tolerance=1e-12, distance_absolute_tolerance_m=1e-6,
                  units='Interpretation scores are fractions; class-change scores are percent and differences points.',
                  integer_counts_and_memberships='exact', source_and_input_hashes_rechecked=True)
    examples.write_json(output / 'verification.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiles-dir', type=Path, help='Optional separately supplied selected hash-matching arrays')
    parser.add_argument('--exact-figures', action='store_true', help='Require supplied arrays and the pinned runtime; compare both PDF/PNG pairs')
    args = parser.parse_args(argv)
    try:
        result = verify(args.release_root, args.output, args.tiles_dir, args.exact_figures)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps(dict(status='invalid', error=str(error))))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
