"""Select and render Titan examples from the pinned public catalog/test union.

Run ``select`` before acquiring just the selected .npy files, then ``render``
with those supplied files. No network, inference, or map-overlay reconstruction.
Outputs must be outside this repository. Selection does not inspect image pixels.
"""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CATALOG_SHA256 = '8152226f876e0cb5fad105e0b00a37ef5b62d156d7f6eb3a85b77c549a402967'
PROTOCOL_SHA256 = '1ce5fd9e8b135a172ce3e671b12a9a06e160e089debb9cac22bb14213f784060'
TEST_UNION_SHA256 = 'd331c732e89b48bce96e2d23992cb9af7d36b2d0f75b5f7c6bfb69cdc0b144d7'
CLASSES = ['plains', 'dunes', 'hummocky', 'labyrinths', 'lakes', 'craters']
USGS_URL = 'https://astrogeology.usgs.gov/search/map/titan_cassini_sar_hisar_global_mosaic_351m'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_population(catalog_path, protocol_path):
    require(sha256(catalog_path) == CATALOG_SHA256, 'Public catalog SHA256 differs')
    require(sha256(protocol_path) == PROTOCOL_SHA256, 'Public protocol SHA256 differs')
    catalog = json.loads(Path(catalog_path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    require(catalog['class_names'] == CLASSES and catalog['domain'] == 'titan'
            and catalog['intensity_space'] == 'hisar_log_dn'
            and catalog['tile_size'] == 128 and catalog['footprint_size_m'] == 45000,
            'Unexpected image product or class definition')
    rows = {r['tile_id']: r for r in catalog['tiles']}
    require(len(rows) == len(catalog['tiles']) == 23380, 'Catalog population differs')
    require(protocol['catalog_sha256'] == CATALOG_SHA256, 'Protocol catalog differs')
    require([s['fold'] for s in protocol['splits']] == list(range(5)), 'Expected five folds')
    test_folds, split_hashes = {}, {}
    for entry in protocol['splits']:
        name = f"titan_contiguous_fold_{entry['fold']}.json"
        require(entry['path'] == name, 'Unexpected split path')
        path = Path(protocol_path).parent / name
        require(sha256(path) == entry['sha256'], 'Split SHA256 differs')
        split_hashes[name] = entry['sha256']
        split = json.loads(path.read_text())
        require(split['catalog_sha256'] == CATALOG_SHA256
                and set(split['assignments']) == set(rows), 'Split catalog or IDs differ')
        for tid, role in split['assignments'].items():
            if role == 'test':
                require(tid not in test_folds, 'Test folds overlap')
                test_folds[tid] = entry['fold']
    union_hash = hashlib.sha256(json.dumps(sorted(test_folds)).encode()).hexdigest()
    require(len(test_folds) == protocol['test_union_count'] == 23246
            and union_hash == protocol['test_union_sorted_ids_sha256'] == TEST_UNION_SHA256,
            'Exact test union differs')
    return rows, test_folds, split_hashes


def median_pick(candidates):
    """Use exact decimal distances so rounding cannot defeat the tile-ID tie rule."""
    values = sorted(Decimal(str(r['label_confidence'])) for r in candidates)
    require(bool(values), 'Empty selection stratum')
    n = len(values)
    median = (values[(n - 1) // 2] + values[n // 2]) / 2
    row = min(candidates, key=lambda r: (abs(Decimal(str(r['label_confidence'])) - median), r['tile_id']))
    return row, float(median)


def select_examples(rows, test_folds):
    require(set(test_folds) <= set(rows), 'Test ID absent from catalog')
    candidates = [rows[tid] for tid in sorted(test_folds)]
    for row in candidates:
        require(re.fullmatch(r'titan_\d{6}', row['tile_id']) is not None, 'Invalid tile ID')
        require(type(row['label']) is int and row['label'] in range(6), 'Invalid map label')
        require(0 <= row['label_confidence'] <= 1 and .8 <= row['valid_fraction'] <= 1,
                'Invalid map or SAR fraction')
        require(re.fullmatch(r'[0-9a-f]{64}', row['tile_sha256']) is not None, 'Invalid tile digest')
    selected = []

    def add(pool, purpose):
        row, median = median_pick(pool)
        selected.append(dict(tile_id=row['tile_id'], label=row['label'],
            class_name=CLASSES[row['label']], modal_map_fraction=row['label_confidence'],
            valid_sar_fraction=row['valid_fraction'], tile_sha256=row['tile_sha256'],
            center_lat=row['center_lat'], center_lon=row['center_lon'],
            test_fold=test_folds[row['tile_id']], selection_purpose=purpose,
            candidate_count=len(pool), candidate_median_modal_fraction=median,
            mixed_below_half=row['label_confidence'] < .5))

    for label in range(6):
        add([r for r in candidates if r['label'] == label], 'class_median')
    if not any(r['mixed_below_half'] for r in selected):
        add([r for r in candidates if r['label_confidence'] < .5], 'mixed_median')
    return selected


def selection_manifest(catalog_path, protocol_path):
    rows, test_folds, split_hashes = load_population(catalog_path, protocol_path)
    return dict(schema_version=1, catalog_sha256=CATALOG_SHA256, protocol_sha256=PROTOCOL_SHA256,
        splits_sha256=split_hashes, test_union_count=len(test_folds),
        test_union_sorted_ids_sha256=TEST_UNION_SHA256, class_names=CLASSES,
        source_product='USGS Cassini SAR HiSAR Global Mosaic 351m', source_url=USGS_URL,
        footprint_width_km=45, tile_shape=[128, 128], intensity_space='hisar_log_dn',
        selection_rule=('Within each map class in the exact test union, choose the tile nearest '
                        'the median recorded modal map fraction; break ties by tile ID. '
                        'Use decimal arithmetic on the recorded fractions. If none has modal '
                        'fraction <0.5, add the median-nearest tile from that subset, with the same tie rule.'),
        fraction_support=('Only the modal fraction is available. It is the fraction of valid '
                          '64x64 map samples assigned to the modal class, without the SAR validity mask. '
                          'Other class fractions and a spatial map overlay are unavailable here.'),
        interpretation='Metadata-selected image examples, not typical geology or independent label validation.',
        selected=select_examples(rows, test_folds))


def stretch(array):
    """Historical display recipe: finite positive DN, per-tile 2/98 percentiles."""
    valid = np.isfinite(array) & (array > 0)
    require(valid.any(), 'Tile has no valid SAR pixels')
    lo, hi = np.percentile(array[valid].astype(np.float64), [2, 98])
    # Match the historical constant-panel fallback, while retaining the mask.
    values = np.zeros(array.shape, dtype=np.float64)
    if hi - lo >= 1e-6:
        values[valid] = np.clip((array[valid].astype(np.float64) - lo) / (hi - lo), 0, 1)
    return np.ma.array(values, mask=~valid), dict(percentile_2_dn=float(lo),
        percentile_98_dn=float(hi), constant_stretch=bool(hi - lo < 1e-6),
        valid_pixel_count=int(valid.sum()), invalid_pixel_count=int((~valid).sum()))


def load_tiles(manifest, directory):
    panels, records = [], []
    for row in manifest['selected']:
        path = Path(directory) / (row['tile_id'] + '.npy')
        require(sha256(path) == row['tile_sha256'], f"Tile SHA256 differs: {row['tile_id']}")
        array = np.load(path, allow_pickle=False)
        require(array.shape == (128, 128) and array.dtype == np.float32, 'Unexpected tile shape or dtype')
        panel, stats = stretch(array)
        require(round(stats['valid_pixel_count'] / array.size, 6) == row['valid_sar_fraction'],
                f"SAR fraction differs: {row['tile_id']}")
        panels.append(panel)
        records.append(dict(tile_id=row['tile_id'], local_sha256=row['tile_sha256'], **stats))
    return panels, records


def render(manifest, panels, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Match the repaired-paper plots without importing analysis/training modules.
    style = {'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    columns = 4 if len(panels) == 7 else 3
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, columns, figsize=(6.5, 5.9))
        fig.subplots_adjust(left=.025, right=.975, bottom=.29, top=.875, wspace=.16, hspace=.9)
        cmap = plt.get_cmap('gray').copy()
        cmap.set_bad('#CC79A7')
        for ax, row, panel in zip(axes.flat, manifest['selected'], panels):
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            title = row['class_name'].capitalize()
            if row['mixed_below_half']:
                title += '\nMixed map footprint'
            ax.set_title(title, fontsize=8)
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(f"{row['tile_id']}\nMap class {100 * row['modal_map_fraction']:.2f}%"
                          f"\nValid SAR {100 * row['valid_sar_fraction']:.2f}%\n45 km width", fontsize=7.5)
        for ax in list(axes.flat)[len(panels):]:
            ax.axis('off')
            ax.text(.03, .95, 'Mixed map footprint:\nmodal class covers <50%\nof valid map samples.\n\n'
                    'Only the modal fraction\nis available in the catalog.',
                    transform=ax.transAxes, va='top', fontsize=7.5, linespacing=1.4)
            ax.legend(handles=[Patch(facecolor='#CC79A7', label='Invalid SAR pixels\nwhere present')],
                      loc='lower left', bbox_to_anchor=(0, -.4), frameon=False, fontsize=7.5,
                      handlelength=1.2, borderaxespad=0)
        if len(panels) == 6:
            fig.legend(handles=[Patch(facecolor='#CC79A7', label='Invalid SAR pixels, where present')],
                       loc='lower left', bbox_to_anchor=(.025, .14), frameon=False, fontsize=7.5)
        fig.suptitle('Titan SAR display inputs and modal map labels', y=.965, fontsize=10)
        fig.text(.025, .13, 'Map: modal class fraction of valid map samples. SAR: valid pixel fraction from a separate mask.\n'
                 'USGS Cassini SAR/HiSAR display DN. Independent 2nd to 98th percentile stretch per tile.\n'
                 'Selected by map-fraction medians from the current test union. Panel brightness is not comparable.\n'
                 'The examples illustrate catalog labels and image appearance. Their geological representativeness is untested.', fontsize=7.5, va='top')
        try:
            fig.savefig(Path(directory) / 'repaired_examples.pdf', dpi=220,
                        metadata={'CreationDate': None, 'ModDate': None})
            fig.savefig(Path(directory) / 'repaired_examples.png', dpi=180)
        finally:
            plt.close(fig)
    return matplotlib.__version__


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['select', 'render'])
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True, help='New output for select; verified input for render')
    parser.add_argument('--tiles-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = selection_manifest(args.catalog, args.protocol)
        destination = args.selection if args.action == 'select' else args.output_dir
        require(destination is not None and not destination.exists() and not destination.is_symlink(),
                'Output must be new')
        require(not destination.resolve().is_relative_to(ROOT), 'Output must be outside the repository')
        if args.action == 'select':
            write_json(args.selection, manifest)
        else:
            require(args.tiles_dir is not None, 'Supply --tiles-dir')
            require(json.loads(args.selection.read_text()) == manifest, 'Selection differs from deterministic catalog selection')
            panels, records = load_tiles(manifest, args.tiles_dir)
            args.output_dir.mkdir()
            version = render(manifest, panels, args.output_dir)
            write_json(args.output_dir / 'repaired_examples_manifest.json', dict(
                schema_version=1, selection=manifest, selection_sha256=sha256(args.selection),
                renderer_sha256=sha256(Path(__file__)),
                runtime=dict(python=sys.version.split()[0], numpy=np.__version__, matplotlib=version),
                display=dict(percentiles=[2, 98], mask='nonfinite or nonpositive DN',
                             invalid_color='#CC79A7', additional_log_transform=False,
                             interpolation='nearest', independent_panel_scaling=True),
                tiles=records, outputs_sha256={name: sha256(args.output_dir / name)
                    for name in ['repaired_examples.pdf', 'repaired_examples.png']}))
        print(json.dumps({'action': args.action, 'selected_ids': [r['tile_id'] for r in manifest['selected']]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    raise SystemExit(main())
