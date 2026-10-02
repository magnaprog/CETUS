"""Scientific regression checks on synthetic counts/coordinates; no model runs."""

import copy
import gzip
import hashlib
import json
import math

import numpy as np
import pytest

from scripts import analyze_repaired_interpretation as analysis


def independent_f1(cm):
    return sum(2 * cm[k][k] / (sum(cm[k]) + sum(row[k] for row in cm)) for k in range(6)) / 6


def test_pool_folds_within_seed_before_metrics_never_pool_seeds():
    small = np.eye(6, dtype=int)
    large = np.diag([9, 9, 1, 1, 1, 1])
    miss_small, miss_large = small.copy(), large.copy()
    miss_small[0, :2] = [0, 1]
    miss_large[0, :2] = [0, 9]
    ft = np.array([[small, miss_large], [miss_small, large]])
    lr = np.array([small, large])
    result = analysis.aggregate(ft, lr)
    manual = [independent_f1((a + b).tolist()) for a, b in ft]
    pooled = result["pooled_test_secondary"]["adaptation"]["macro"]["f1"]
    assert pooled == pytest.approx(sum(manual) / 2, abs=1e-14)
    assert pooled != pytest.approx(independent_f1(ft.sum(axis=(0, 1)).tolist()))
    equal = sum(independent_f1(c.tolist()) for seed in ft for c in seed) / 4
    assert result["equal_fold_primary"]["adaptation"]["macro"]["f1"] == pytest.approx(equal)
    assert pooled != pytest.approx(equal)
    assert result["pooled_test_secondary"]["lr"]["confusion_counts"] == (small + large).tolist()
    assert result["ft_minus_lr"]["pooled_test_secondary"]["macro"]["f1"] == pytest.approx(pooled - 1)


def test_exposure_uses_any_reinstated_center_strict_threshold_and_spherical_wrap():
    radius = 1000
    # Check physical geometry first, then equality with the computed binary64
    # distance (an analytic pi/2 differs by a last bit under haversine).
    coordinates = [[0, 0], [0, 90], [0, 91], [0, 179]]
    distance = analysis.nearest_distances(coordinates, [[0, 0], [0, -179]], radius)
    assert distance == pytest.approx([0, math.pi * radius / 2, math.pi * radius / 2, math.pi * radius / 90])
    _, mask = analysis.exposure(coordinates, [[0, 0], [0, -179]], radius, distance[1])
    assert mask.tolist() == [True, False, False, True]
    _, reversed_mask = analysis.exposure(coordinates, [[0, -179], [0, 0]], radius, distance[1])
    assert reversed_mask.tolist() == mask.tolist()
    # A slightly larger exact threshold admits equality; no km rounding first.
    _, larger = analysis.exposure([[0, 90]], [[0, 0]], radius, np.nextafter(distance[1], np.inf))
    assert larger.tolist() == [True]


def test_single_crater_leverage_matches_a_direct_counterfactual_confusion():
    cm = np.diag([40, 20, 10, 5, 3, 1])
    cm[0, 0], cm[0, 1] = 30, 10
    before = analysis.single_crater_leverage(cm)
    cm[5, 5], cm[5, 2] = 0, 1
    after = analysis.single_crater_leverage(cm)
    assert before["if_incorrect"] == pytest.approx(after["validation_macro_recall"])
    assert after["if_correct"] == pytest.approx(before["validation_macro_recall"])
    assert before["validation_macro_recall"]-after["validation_macro_recall"] == pytest.approx(1/6)
    cm[5, 2] = 2
    with pytest.raises(ValueError, match="Single-Craters"):
        analysis.single_crater_leverage(cm)


def test_error_summary_keeps_counts_partition_support_and_averages_draws():
    labels = [0, 1, 2, 3, 4, 5]
    baseline = [1, 1, 2, 0, 4, 5]
    draws = [[0, 1, 2, 0, 4, 5], [1, 0, 2, 3, 4, 5]]
    mask = np.array([True, True, False, False, False, False])
    rows = analysis.error_rows(labels, baseline, draws, mask)
    assert [r["substitution_errors"] for r in rows] == [0, 2, 1.0]
    assert rows[-1]["baseline_errors"] == 1
    assert rows[-1]["difference"] == 0
    assert rows[-1]["substitution_error_rate"] == .5
    assert [rows[-1][c] for c in analysis.CLASSES] == [1, 1, 0, 0, 0, 0]
    remaining = analysis.error_rows(labels, baseline, draws, ~mask)[-1]
    full = analysis.error_rows(labels, baseline, draws, np.ones(6, dtype=bool))[-1]
    assert full["substitution_errors"] == rows[-1]["substitution_errors"] + remaining["substitution_errors"]
    empty = analysis.error_rows(labels, baseline, draws, np.zeros(6, dtype=bool))[-1]
    assert empty["n"] == 0 and empty["difference"] is None and empty["baseline_error_rate"] is None


@pytest.fixture
def memberships():
    tiles = [dict(tile_id=f"t{f}{c}", label=c) for f in range(5) for c in range(6)]
    tiles.append(dict(tile_id="boundary", label=0))
    catalog = dict(class_names=list(analysis.CLASSES), tiles=tiles)
    membership, auxiliary, splits = dict(folds=[]), dict(class_names=list(analysis.CLASSES), folds=[]), []
    for f in range(5):
        roles = {t["tile_id"]: ("buffer_excluded" if t["tile_id"] == "boundary" else
                 "test" if t["tile_id"][1] == str(f) else
                 "val" if t["tile_id"][1] == str((f+1) % 5) else "train") for t in tiles}
        selected = lambda role: [t["tile_id"] for t in tiles if roles[t["tile_id"]] == role]
        train, val, test = [selected(role) for role in ("train", "val", "test")]
        splits.append(dict(assignments=roles, policy=dict(test_fold=f, exclusion_reasons={"boundary": "train_buffer_excluded"})))
        new = (set(train) - {train[0]}) | {"boundary"}
        ordered = [t["tile_id"] for t in tiles if t["tile_id"] in new]
        draws = [dict(draw_seed=s, removed_ids=[train[0]], training_ids=ordered,
                      training_ids_sha256=analysis.base.digest(ordered)) for s in range(5)]
        membership["folds"].append(dict(fold=f, retained_train_ids=train, validation_ids=val,
            test_ids=test, test_labels=list(range(6)), validation_labels=list(range(6)), boundary_ids=["boundary"],
            eligible_removal_ids=train, draws=draws))
        auxiliary["folds"].append(dict(fold=f, test_ids=test, true_labels=list(range(6)),
                                       common={m: list(range(6)) for m in analysis.MODELS}))
    return catalog, splits, membership, auxiliary


def test_complete_membership_join_accepts_synthetic_population(memberships):
    entries, members, auxiliary = analysis.check_membership(*memberships)
    assert len(entries) == 31 and len(members) == len(auxiliary) == 5


@pytest.mark.parametrize("mutation", ["order", "missing_fold", "duplicate_fold", "extra_tile", "wrong_truth", "wrong_boundary", "duplicate_removed"])
def test_membership_rejects_incomplete_or_misaligned_joins(memberships, mutation):
    catalog, splits, mem, aux = copy.deepcopy(memberships)
    if mutation == "order":
        aux["folds"][0]["test_ids"].reverse()
    elif mutation == "missing_fold":
        mem["folds"].pop()
    elif mutation == "duplicate_fold":
        aux["folds"].append(copy.deepcopy(aux["folds"][0]))
    elif mutation == "extra_tile":
        splits[0]["assignments"]["unknown"] = "train"
    elif mutation == "wrong_truth":
        mem["folds"][0]["test_labels"][0] = 1
    elif mutation == "wrong_boundary":
        mem["folds"][0]["boundary_ids"] = mem["folds"][0]["validation_ids"][:1]
    else:
        mem["folds"][0]["draws"][0]["removed_ids"] *= 2
    with pytest.raises(ValueError):
        analysis.check_membership(catalog, splits, mem, aux)


def test_hash_binding_precedes_parsing_and_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "input.json.gz"
    path.write_bytes(gzip.compress(b'{"value":1}'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert analysis.load_bound(path, digest) == {"value": 1}
    path.write_bytes(gzip.compress(b'{"value":2}'))
    with pytest.raises(ValueError, match="hash differs"):
        analysis.load_bound(path, digest)
    path.write_bytes(gzip.compress(b'{"value":1,"value":2}'))
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        analysis.load_bound(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("predictions", [[0, 1], [0, 1, 2, 3, 4, 6], [0., 1., 2., 3., 4., 5.], [True]*6])
def test_prediction_vectors_reject_length_or_class_errors(predictions):
    with pytest.raises(ValueError, match="Invalid labels"):
        analysis.confusion(list(range(6)), predictions)


def test_prediction_cohort_rejects_missing_and_duplicate_seeds():
    expected = [("croma", 0, s) for s in range(5)]
    rows = [dict(model=m, fold=f, seed=s) for m, f, s in expected]
    assert len(analysis.index(rows, ("model", "fold", "seed"), expected)) == 5
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            analysis.index(bad, ("model", "fold", "seed"), expected)
