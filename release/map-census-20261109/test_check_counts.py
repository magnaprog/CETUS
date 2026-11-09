"""Synthetic-only regression tests; never open the accepted census payloads."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import check_counts as checker


def fixture():
    """Scalar cell oracle, deliberately different from the vectorized recount."""
    n = 10
    grids = np.empty((n, 64, 64), dtype=np.uint8)
    valid = np.zeros((n, 128, 128), dtype=np.uint8)
    counts = np.zeros((n, 4, 6), dtype=np.int64)
    unknown = np.zeros((n, 4), dtype=np.int64)
    hist = np.zeros((n, 5), dtype=np.int64)
    for i in range(n):
        for y in range(64):
            for x in range(64):
                code = i % 6 if y else (x + i) % 7
                code = 255 if code == 6 else code
                grids[i, y, x] = code
                weight = (x + 2*y + i) % 5
                # Mixed little-bit patterns distinguish C order and 2x2 grouping.
                for dy, dx in ((0, 1), (1, 0), (1, 1), (0, 0))[:weight]:
                    valid[i, 2*y+dy, 2*x+dx] = 1
                hist[i, weight] += 1
                for mode, amount in enumerate((1, weight, int(weight == 4), int(weight > 0))):
                    if code == 255:
                        unknown[i, mode] += amount
                    else:
                        counts[i, mode, code] += amount
    return {'tile_ids': np.array([f'titan_{3*i:06d}' for i in range(n)], dtype='<U12'),
            'map_grids': grids,
            'sar_valid_packed': np.packbits(valid.reshape(n, -1), axis=1, bitorder='little'),
            'roles': np.array([[(i+f) % 5 for f in range(5)] for i in range(n)], dtype=np.uint8),
            'modal_labels': np.array([i % 6 for i in range(n)], dtype=np.uint8),
            'counts': counts, 'unknown': unknown, 'weight_histograms': hist}


def oracle_summary(arrays, rows):
    result = {'images': len(rows), 'modal_class_images': [0]*6, 'modes': {}}
    for i in rows:
        result['modal_class_images'][int(arrays['modal_labels'][i])] += 1
    for m, mode in enumerate(checker.MODES):
        cells, present, matrix, unknown = [0]*6, [0]*6, [[0]*6 for _ in range(6)], 0
        for i in rows:
            label = int(arrays['modal_labels'][i])
            unknown += int(arrays['unknown'][i, m])
            for c in range(6):
                value = int(arrays['counts'][i, m, c])
                cells[c] += value
                present[c] += int(value > 0)
                matrix[label][c] += value
        result['modes'][mode] = {'class_counts': cells, 'known_total': sum(cells),
            'unknown_total': unknown, 'images_containing_class': present,
            'class_counts_by_modal_class': matrix}
    return result


class RecountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = fixture()
        cls.n = len(cls.base['tile_ids'])

    def test_scalar_oracle_all_groups_modes_and_units(self):
        groups, scope = checker.reconstruct(self.base, self.n)
        self.assertEqual(scope['groups'], 26)
        expected = {'catalog': oracle_summary(self.base, list(range(self.n))), 'folds': []}
        for f in range(5):
            expected['folds'].append({'fold': f, 'roles': {
                role: oracle_summary(self.base, [i for i in range(self.n)
                                                if self.base['roles'][i, f] == r])
                for r, role in enumerate(checker.ROLES)}})
        checker.same_json(groups, expected)
        modes = groups['catalog']['modes']
        weighted = modes['weighted']
        self.assertEqual(weighted['known_total'] + weighted['unknown_total'],
                         int((self.base['weight_histograms'] * np.arange(5)).sum()))
        self.assertNotEqual(weighted['class_counts'], modes['baseline']['class_counts'])
        matrix = np.asarray(modes['all_four']['class_counts_by_modal_class'])
        self.assertFalse(np.array_equal(matrix, matrix.T))

    def test_schema_shapes_types_keys_and_ids(self):
        mutations = {
            'missing': lambda a: a.pop('unknown'),
            'extra': lambda a: a.update(extra=np.zeros(1)),
            'dtype': lambda a: a.update(counts=a['counts'].astype(np.float64)),
            'shape': lambda a: a.update(roles=a['roles'][:, :4]),
            'incomplete': lambda a: a.update(tile_ids=a['tile_ids'][:-1]),
            'duplicate': lambda a: a['tile_ids'].__setitem__(0, a['tile_ids'][1]),
            'malformed': lambda a: a['tile_ids'].__setitem__(0, 'titan_bad'),
            'role_code': lambda a: a['roles'].__setitem__((0, 0), 5),
            'modal_code': lambda a: a['modal_labels'].__setitem__(0, 255),
            'negative': lambda a: a['unknown'].__setitem__((0, 0), -1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                arrays = copy.deepcopy(self.base)
                mutate(arrays)
                with self.assertRaises(ValueError):
                    checker.reconstruct(arrays, self.n)

    def test_corrupted_saved_arrays_rejected(self):
        for name, index in [('counts', (0, 1, 0)), ('unknown', (0, 2)),
                            ('weight_histograms', (0, 4)), ('modal_labels', (0,)),
                            ('sar_valid_packed', (0, 0)), ('map_grids', (0, 0, 0))]:
            with self.subTest(array=name):
                arrays = copy.deepcopy(self.base)
                arrays[name][index] ^= 1
                with self.assertRaises(ValueError):
                    checker.reconstruct(arrays, self.n)

    def test_unknown_not_terrain_or_missing_sar(self):
        _, _, c, u, hist = checker.independent_counts(
            self.base['map_grids'], self.base['sar_valid_packed'])
        np.testing.assert_array_equal(c, self.base['counts'])
        np.testing.assert_array_equal(u, self.base['unknown'])
        np.testing.assert_array_equal(hist, self.base['weight_histograms'])
        self.assertTrue(np.all(u > 0))
        arrays = copy.deepcopy(self.base)
        arrays['map_grids'][0, 0, 0] = 6
        with self.assertRaisesRegex(ValueError, 'codes must'):
            checker.reconstruct(arrays, self.n)

    def test_wrong_bitorder_and_flattening(self):
        for order in ('big', 'transpose'):
            with self.subTest(order=order):
                arrays = copy.deepcopy(self.base)
                mask = np.unpackbits(arrays['sar_valid_packed'], axis=1, bitorder='little')
                if order == 'transpose':
                    mask = mask.reshape(-1, 128, 128).transpose(0, 2, 1).reshape(self.n, -1)
                arrays['sar_valid_packed'] = np.packbits(mask, axis=1,
                                                       bitorder='big' if order == 'big' else 'little')
                with self.assertRaises(ValueError):
                    checker.reconstruct(arrays, self.n)

    def test_modal_tie_uses_lowest_code_and_excludes_unknown(self):
        grids = np.full((1, 64, 64), 255, dtype=np.uint8)
        grids[0, 0, :4] = [2, 1, 2, 1]
        packed = np.full((1, 2048), 255, dtype=np.uint8)
        _, _, c, u, h = checker.independent_counts(grids, packed)
        arrays = {k: v[:1].copy() for k, v in self.base.items()}
        arrays.update(map_grids=grids, sar_valid_packed=packed, counts=c, unknown=u,
                      weight_histograms=h, modal_labels=np.array([1], dtype=np.uint8))
        groups, _ = checker.reconstruct(arrays, 1)
        self.assertEqual(groups['catalog']['modal_class_images'], [0, 1, 0, 0, 0, 0])
        arrays['map_grids'].fill(255)
        _, _, c, u, h = checker.independent_counts(arrays['map_grids'], packed)
        arrays.update(counts=c, unknown=u, weight_histograms=h)
        with self.assertRaisesRegex(ValueError, 'no known map cells'):
            checker.reconstruct(arrays, 1)

    def test_batch_boundary_and_row_order(self):
        arrays = {k: np.concatenate([v]*13) for k, v in self.base.items()}
        arrays['tile_ids'] = np.array([f'titan_{i:06d}' for i in range(130)], dtype='<U12')
        expected, _ = checker.reconstruct(arrays, 130)
        reversed_arrays = {k: v[::-1] for k, v in arrays.items()}
        measured, _ = checker.reconstruct(reversed_arrays, 130)
        checker.same_json(measured, expected)
        arrays['counts'][128, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, r'counts\[128:130\]'):
            checker.reconstruct(arrays, 130)

    def test_every_group_and_mode_is_compared(self):
        groups, _ = checker.reconstruct(self.base, self.n)
        locations = [('catalog',)] + [('folds', f, 'roles', r)
                                      for f in range(5) for r in checker.ROLES]
        for loc in locations:
            for mode in checker.MODES:
                with self.subTest(group=loc, mode=mode):
                    changed = copy.deepcopy(groups)
                    group = changed
                    for key in loc:
                        group = group[key]
                    group['modes'][mode]['class_counts_by_modal_class'][1][4] += 1
                    with self.assertRaises(ValueError):
                        checker.same_json(groups, changed)

    def test_json_types_keys_lengths_and_other_counts(self):
        for actual, expected in [(1, True), (1, 1.0), ({'a': 1}, {'a': 1, 'b': 2}),
                                 ([1, 2], [1]), (0, 1)]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                checker.same_json(actual, expected)
        groups, _ = checker.reconstruct(self.base, self.n)
        for field in ('class_counts', 'images_containing_class', 'known_total', 'unknown_total'):
            changed = copy.deepcopy(groups)
            mode = changed['catalog']['modes']['weighted']
            if isinstance(mode[field], list):
                mode[field][0] += 1
            else:
                mode[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.same_json(groups, changed)


class InputAndCliTests(unittest.TestCase):
    def test_synthetic_cli_end_to_end_and_role_count_failure(self):
        arrays = fixture()
        n = len(arrays['tile_ids'])
        expected = {'catalog': oracle_summary(arrays, list(range(n))), 'folds': [
            {'fold': f, 'roles': {role: oracle_summary(arrays, [i for i in range(n)
                                  if arrays['roles'][i, f] == r])
                                  for r, role in enumerate(checker.ROLES)}} for f in range(5)]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'counts.json').write_text(json.dumps(
                {'counts': expected, 'array_conventions': checker.CONVENTIONS}))
            for altered in (False, True):
                if altered:
                    arrays['roles'][0, 0] = 1
                np.savez_compressed(directory / 'map_cells.npz', **arrays)
                hashes = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                          for name in checker.HASHES}
                output = directory / f'report-{altered}.json'
                with patch.object(checker, 'N', n), patch.object(checker, 'HASHES', hashes), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = checker.main(['--data-dir', str(directory), '--output', str(output)])
                report = json.loads(output.read_text())
                if altered:
                    self.assertNotEqual(code, 0)
                    self.assertEqual(report['status'], 'failed')
                    self.assertIn('counts/folds/0/roles/train/images', report['error'])
                else:
                    self.assertEqual(code, 0)
                    checker.same_json(report['reconstructed_counts'], expected)

    def test_each_hash_checked_before_array_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            payloads = {'counts.json': b'{}', 'map_cells.npz': b'fixture'}
            hashes = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
            for corrupt in payloads:
                for name, raw in payloads.items():
                    (directory / name).write_bytes(raw + (b' ' if name == corrupt else b''))
                with self.subTest(corrupt=corrupt), patch.object(checker, 'HASHES', hashes), \
                        patch.object(checker, 'load_arrays') as loader:
                    with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                        checker.check(directory)
                    loader.assert_not_called()

    def test_hash_and_load_use_same_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw = {'counts.json': b'{}', 'map_cells.npz': b'example'}
            for name, value in raw.items():
                (directory / name).write_bytes(value)
            hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
            with patch.object(checker, 'HASHES', hashes):
                bound = checker.bound_inputs(directory)
            (directory / 'counts.json').write_bytes(b'changed')
            self.assertEqual(bound, raw)

    def test_archive_members_and_object_arrays(self):
        arrays = fixture()
        for kind in ('missing', 'extra', 'duplicate', 'object'):
            changed = dict(arrays)
            if kind == 'missing':
                changed.pop('counts')
            if kind == 'extra':
                changed['extra'] = np.zeros(1)
            if kind == 'object':
                changed['tile_ids'] = np.array(['object']*10, dtype=object)
            raw = io.BytesIO()
            np.savez_compressed(raw, **changed)
            if kind == 'duplicate':
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    with zipfile.ZipFile(raw, 'a') as archive:
                        archive.writestr('counts.npy', b'bad')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                checker.load_arrays(raw.getvalue())

    def test_cli_failures_and_interrupt_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, error in enumerate((ValueError('mismatch'), OSError('missing input'),
                                            KeyboardInterrupt())):
                output = Path(temp) / f'failure-{index}.json'
                with patch.object(checker, 'check', side_effect=error), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = checker.main(['--output', str(output)])
                self.assertNotEqual(code, 0)
                report = json.loads(output.read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error_type'], type(error).__name__)

    def test_cli_refuses_overwrite_and_stdout_default(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'existing.json'
            output.write_text('keep')
            with patch.object(checker, 'check') as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main(['--output', str(output)]), 1)
                run.assert_not_called()
            self.assertEqual(output.read_text(), 'keep')
            stdout = io.StringIO()
            with patch.object(checker, 'check', return_value={'status': 'passed'}), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(checker.main([]), 0)
            self.assertEqual(json.loads(stdout.getvalue())['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
