"""Packaging checks: scientific comparisons must not reduce to file hashes."""

import math

import pytest

import verify_reconstruction as verify


def test_numeric_tolerances_preserve_exact_counts_and_membership():
    verify.compare({'score': .5 + 1e-13, 'count': 4, 'exposed': True},
                   {'score': .5, 'count': 4, 'exposed': True})
    for actual in ({'score': .5001, 'count': 4, 'exposed': True},
                   {'score': .5, 'count': 5, 'exposed': True},
                   {'score': .5, 'count': 4., 'exposed': True},
                   {'score': .5, 'count': 4, 'exposed': False},
                   {'score': math.nan, 'count': 4, 'exposed': True}):
        with pytest.raises(ValueError):
            verify.compare(actual, {'score': .5, 'count': 4, 'exposed': True})


def test_distance_tolerance_is_separate_from_score_tolerance():
    verify.compare(1000. + 1e-8, 1000., 'row/nearest_reinstated_center_m')
    with pytest.raises(ValueError):
        verify.compare(.5 + 1e-8, .5, 'score')
    with pytest.raises(ValueError):
        verify.compare(1000.01, 1000., 'row/nearest_reinstated_center_m')


def test_csv_comparison_checks_values_supports_and_all_rows(tmp_path):
    path = tmp_path / 'values.csv'
    path.write_text('score,n,exposed,empty\n0.5,4,True,\n')
    expected = [dict(score=.5, n=4, exposed=True, empty=None)]
    verify.compare_csv(path, expected)
    for text in ['score,n,exposed,empty\n0.6,4,True,\n',
                 'score,n,exposed,empty\n0.5,4,False,\n',
                 'score,n,exposed,empty\n0.5,4,True,\n0.5,4,True,\n']:
        path.write_text(text)
        with pytest.raises(ValueError):
            verify.compare_csv(path, expected)


def test_runtime_pin_is_required_only_by_exact_figure_mode(monkeypatch):
    monkeypatch.setattr(verify, 'runtime_versions', lambda: dict(python='different', numpy='2', matplotlib='3'))
    with pytest.raises(ValueError, match='figure_runtime'):
        verify.exact_runtime(dict(figure_runtime=dict(python='3.12.3', numpy='2', matplotlib='3')))


def test_reconstruction_refuses_existing_or_overlapping_output(tmp_path):
    with pytest.raises(ValueError, match='new and disjoint'):
        verify.verify(tmp_path / 'release', tmp_path)
    with pytest.raises(ValueError, match='new and disjoint'):
        verify.verify(tmp_path / 'release', tmp_path / 'release' / 'new')
