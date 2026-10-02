"""Scientific selection, pixel integrity and rendering guards for Titan examples."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_repaired_examples as examples


def catalog_rows():
    rows = {}
    for label in range(6):
        for i, fraction in enumerate([.6, .8, 1.]):
            tid = f'titan_{label * 10 + i:06d}'
            rows[tid] = dict(tile_id=tid, label=label, label_confidence=fraction,
                valid_fraction=1., tile_sha256='a' * 64, center_lat=0., center_lon=label * 45.)
    for i, fraction in enumerate([.3, .4, .45]):
        tid = f'titan_{100 + i:06d}'
        rows[tid] = dict(rows['titan_000000'], tile_id=tid, label_confidence=fraction)
    return rows


def test_selection_uses_exact_test_union_and_is_order_independent():
    rows = catalog_rows()
    folds = {tid: i % 5 for i, tid in enumerate(rows)}
    # A non-test entry must not change a class median or become an example.
    rows['titan_999999'] = dict(rows['titan_000011'], tile_id='titan_999999', label_confidence=.2)
    result = examples.select_examples(rows, folds)
    reordered = examples.select_examples(dict(reversed(list(rows.items()))), dict(reversed(list(folds.items()))))
    assert result == reordered
    assert [r['label'] for r in result[:6]] == list(range(6))
    assert len(result) == 7 and result[-1]['tile_id'] == 'titan_000101'
    assert result[-1]['modal_map_fraction'] == .4
    assert result[-1]['candidate_count'] == 3
    assert all(r['tile_id'] in folds and r['test_fold'] == folds[r['tile_id']] for r in result)


def test_decimal_median_ties_use_tile_id_not_binary_roundoff():
    rows = [dict(tile_id='titan_000001', label_confidence=.8008),
            dict(tile_id='titan_000000', label_confidence=.802)]
    row, median = examples.median_pick(rows)
    assert median == .8014 and row['tile_id'] == 'titan_000000'


def test_mixed_class_example_does_not_add_duplicate_seventh_panel():
    rows = catalog_rows()
    for row in rows.values():
        if row['label'] == 5:
            row['label_confidence'] = .4
    result = examples.select_examples(rows, {tid: 0 for tid in rows})
    assert len(result) == 6 and result[5]['mixed_below_half']
    assert len({r['tile_id'] for r in result}) == 6


def test_absent_class_or_absent_mixed_stratum_is_explicit():
    rows = catalog_rows()
    with pytest.raises(ValueError, match='Empty selection stratum'):
        examples.select_examples(rows, {tid: 0 for tid, r in rows.items() if r['label'] != 2})
    with pytest.raises(ValueError, match='Empty selection stratum'):
        examples.select_examples(rows, {tid: 0 for tid, r in rows.items() if r['label_confidence'] >= .5})


def test_stretch_masks_invalid_dn_and_uses_valid_pixel_percentiles():
    array = np.array([[0., -1., np.nan], [10., 20., 100.]], dtype=np.float32)
    panel, record = examples.stretch(array)
    assert panel.mask.tolist() == [[True, True, True], [False, False, False]]
    assert record['valid_pixel_count'] == record['invalid_pixel_count'] == 3
    assert record['percentile_2_dn'] == pytest.approx(10.4)
    assert record['percentile_98_dn'] == pytest.approx(96.8)
    np.testing.assert_allclose(panel.compressed(), [0., (20 - 10.4) / (96.8 - 10.4), 1.])
    constant, stats = examples.stretch(np.array([[0, 7], [7, 7]], dtype=np.float32))
    assert stats['constant_stretch'] and constant.mask[0, 0]
    np.testing.assert_array_equal(constant.compressed(), 0)


def test_load_tiles_checks_hash_shape_and_mask_fraction(tmp_path):
    path = tmp_path / 'titan_000001.npy'
    array = np.ones((128, 128), dtype=np.float32)
    array[0] = 0
    np.save(path, array)
    row = dict(tile_id='titan_000001', tile_sha256=examples.sha256(path), valid_sar_fraction=.992188)
    manifest = dict(selected=[row])
    panels, records = examples.load_tiles(manifest, tmp_path)
    assert panels[0].mask[0].all() and records[0]['invalid_pixel_count'] == 128
    row['valid_sar_fraction'] = 1.
    with pytest.raises(ValueError, match='SAR fraction differs'):
        examples.load_tiles(manifest, tmp_path)
    row['valid_sar_fraction'] = .992188
    np.save(path, array * 2)
    with pytest.raises(ValueError, match='SHA256 differs'):
        examples.load_tiles(manifest, tmp_path)
    np.save(path, np.ones((64, 64), dtype=np.float32))
    row['tile_sha256'] = examples.sha256(path)
    with pytest.raises(ValueError, match='shape or dtype'):
        examples.load_tiles(manifest, tmp_path)


def test_wrong_public_catalog_rejected_before_reading_protocol(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='catalog SHA256'):
        examples.load_population(path, tmp_path / 'absent.json')


@pytest.mark.parametrize('panel_count', [6, 7])
def test_render_cli_revalidates_selection_and_binds_output_hashes(tmp_path, monkeypatch, panel_count):
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig

    def inspect_and_save(fig, *args, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_boxes = [t.get_window_extent(renderer) for t in fig.texts if t is not fig._suptitle]
        footer_boxes += [legend.get_window_extent(renderer) for legend in fig.legends]
        assert fig.get_figwidth() == 6.5
        for ax in fig.axes:
            if ax.get_xlabel():
                box = ax.xaxis.label.get_window_extent(renderer)
                assert not any(box.overlaps(footer) for footer in footer_boxes)
                assert '45 km width' in ax.get_xlabel()
        return original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect_and_save)
    rows = catalog_rows()
    if panel_count == 6:
        for row in rows.values():
            if row['label'] == 5:
                row['label_confidence'] = .4
    manifest = dict(selected=examples.select_examples(rows, {tid: 0 for tid in rows}))
    assert len(manifest['selected']) == panel_count
    tile_dir = tmp_path / 'tiles'
    tile_dir.mkdir()
    for row in manifest['selected']:
        path = tile_dir / (row['tile_id'] + '.npy')
        np.save(path, np.arange(1, 16385, dtype=np.float32).reshape(128, 128))
        row['tile_sha256'] = examples.sha256(path)
    monkeypatch.setattr(examples, 'selection_manifest', lambda *args: manifest)
    selection = tmp_path / 'selected.json'
    examples.write_json(selection, manifest)
    output = tmp_path / 'figure'
    args = ['render', '--catalog', 'unused', '--protocol', 'unused', '--selection', str(selection),
            '--tiles-dir', str(tile_dir), '--output-dir', str(output)]
    assert examples.main(args) == 0
    record = json.loads((output / 'repaired_examples_manifest.json').read_text())
    assert record['selection'] == manifest
    assert record['selection_sha256'] == examples.sha256(selection)
    assert record['renderer_sha256'] == examples.sha256(Path(examples.__file__))
    for name, digest in record['outputs_sha256'].items():
        assert examples.sha256(output / name) == digest
    assert examples.main(args) == 1  # Existing output is never overwritten.
    altered = json.loads(selection.read_text())
    altered['selected'][0]['label'] = 5
    selection.write_text(json.dumps(altered))
    args[-1] = str(tmp_path / 'altered')
    assert examples.main(args) == 1 and not Path(args[-1]).exists()
