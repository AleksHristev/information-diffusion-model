"""Checks of purged, embargoed cross-validation folds."""

import numpy as np
import pytest

from information_diffusion.predictive import purged_cv


def _random_schedule(rng, n, gap=86400.0 * 10):
    start = np.cumsum(rng.exponential(gap, n))
    return start, start + 5400.0


def test_every_event_is_tested_once_and_folds_are_contiguous():
    rng = np.random.default_rng(0)
    start, end = _random_schedule(rng, 33)
    order, folds = purged_cv.event_folds(start, end, 5, embargo_seconds=3600.0)
    tested = np.concatenate([fold.test for fold in folds])
    assert sorted(tested) == list(range(33))
    assert all(np.all(np.diff(fold.test) == 1) for fold in folds)
    assert purged_cv.count_violations(start, end, order, folds, 3600.0) == {"not_tested_once": 0, "train_test_shared": 0, "train_test_overlap": 0, "embargo_violations": 0}


def test_overlapping_events_are_purged_from_training():
    start = np.array([0.0, 100.0, 1000.0, 2000.0])
    end = np.array([150.0, 200.0, 1100.0, 2100.0])
    order, folds = purged_cv.event_folds(start, end, 2, embargo_seconds=0.0)
    assert 2 in folds[0].train and 1 not in folds[0].train
    violations = purged_cv.count_violations(start, end, order, folds, 0.0)
    assert violations["train_test_overlap"] == 0


def test_embargo_removes_events_starting_soon_after_a_test_window():
    start = np.array([0.0, 200.0, 1000.0])
    end = np.array([100.0, 300.0, 1100.0])
    order, folds = purged_cv.event_folds(start, end, 3, embargo_seconds=150.0)
    assert 1 not in folds[0].train and 2 in folds[0].train
    assert purged_cv.count_violations(start, end, order, folds, 150.0)["embargo_violations"] == 0


def test_checker_flags_a_planted_violation():
    start = np.array([0.0, 100.0, 1000.0])
    end = np.array([150.0, 200.0, 1100.0])
    order, folds = purged_cv.event_folds(start, end, 3, embargo_seconds=0.0)
    broken = purged_cv.EventFold(train=np.array([1, 2]), test=np.array([0]))
    violations = purged_cv.count_violations(start, end, order, [broken], 0.0)
    assert violations["train_test_overlap"] == 1


def test_many_seeds_hold_the_fold_properties():
    for seed in range(200):
        rng = np.random.default_rng(seed)
        start, end = _random_schedule(rng, 33, gap=4000.0)
        order, folds = purged_cv.event_folds(start, end, 5, embargo_seconds=7200.0)
        violations = purged_cv.count_violations(start, end, order, folds, 7200.0)
        assert sum(violations.values()) == 0


def test_fold_rows_map_events_to_row_indices():
    start = np.array([0.0, 10000.0, 20000.0])
    end = start + 100.0
    order, folds = purged_cv.event_folds(start, end, 3)
    event_ids = np.array([0, 0, 1, 1, 2, 2])
    train, test = purged_cv.fold_rows(event_ids, order, folds[0])
    assert set(test) == {0, 1} and set(train) == {2, 3, 4, 5}


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        purged_cv.event_folds(np.zeros(3), np.zeros(4), 2)
    with pytest.raises(ValueError):
        purged_cv.event_folds(np.zeros(3), np.zeros(3), 5)
