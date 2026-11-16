"""Purged and embargoed cross-validation folds grouped by event."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventFold:
    """One fold: train and test hold positions in the sorted event order."""

    train: np.ndarray
    test: np.ndarray


def event_folds(event_start: np.ndarray, event_end: np.ndarray, n_splits: int, *, embargo_seconds: float = 0.0) -> tuple[np.ndarray, list[EventFold]]:
    """Split events sorted by start time into contiguous test blocks with purged and embargoed train sets.

    For a test event [a, b] and embargo h, an event [c, d] stays in the train set only if d < a or c > b + h.
    Arguments: event_start and event_end are the event windows in seconds; n_splits is the number of folds;
    embargo_seconds is the embargo after each test event.
    Returns: the sorted event order and the list of EventFold.
    Reference: Lopez de Prado (2018), Advances in Financial Machine Learning, Chapter 7.
    """
    event_start = np.asarray(event_start, dtype=float)
    event_end = np.asarray(event_end, dtype=float)
    if event_start.size != event_end.size:
        raise ValueError("event_start and event_end must align")
    if not 2 <= n_splits <= event_start.size:
        raise ValueError("need 2 <= n_splits <= number of events")
    if embargo_seconds < 0 or np.any(event_end < event_start):
        raise ValueError("embargo must be non-negative and event ends must not precede starts")
    order = np.argsort(event_start, kind="stable")
    start, end = event_start[order], event_end[order]
    position = np.arange(order.size)
    folds = []
    for block in np.array_split(position, n_splits):
        keep = np.ones(order.size, dtype=bool)
        keep[block] = False
        for test_event in block:
            before = end < start[test_event]
            after = start > end[test_event] + embargo_seconds
            keep &= before | after
        folds.append(EventFold(train=np.flatnonzero(keep), test=block))
    return order, folds


def fold_rows(event_ids: np.ndarray, order: np.ndarray, fold: EventFold, unique_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Map a fold to row indices.

    Arguments: event_ids is the event identifier of each row; order is the sorted event order; fold is the EventFold;
    unique_ids maps positions in the unsorted event list to identifiers (0, 1, ... by default).
    Returns: the row indices of the train and test sets.
    """
    event_ids = np.asarray(event_ids)
    identifiers = np.arange(order.size) if unique_ids is None else np.asarray(unique_ids)
    return np.flatnonzero(np.isin(event_ids, identifiers[order[fold.train]])), np.flatnonzero(np.isin(event_ids, identifiers[order[fold.test]]))


def count_violations(event_start: np.ndarray, event_end: np.ndarray, order: np.ndarray, folds: list[EventFold], embargo_seconds: float) -> dict[str, int]:
    """Count violations of the fold properties.

    Arguments: event_start and event_end are the event windows; order and folds are from event_folds; embargo_seconds is
    the embargo.
    Returns: counts of events not tested exactly once, positions shared by train and test, overlapping train-test pairs
    and embargo violations; all zero for valid folds.
    """
    start = np.asarray(event_start, dtype=float)[order]
    end = np.asarray(event_end, dtype=float)[order]
    n_events = order.size
    tested = np.zeros(n_events, dtype=int)
    overlap = 0
    embargo = 0
    shared = 0
    for fold in folds:
        tested[fold.test] += 1
        shared += np.intersect1d(fold.train, fold.test).size
        for test_event in fold.test:
            for train_event in fold.train:
                if start[train_event] <= end[test_event] and end[train_event] >= start[test_event]:
                    overlap += 1
                elif start[train_event] > end[test_event] and start[train_event] <= end[test_event] + embargo_seconds:
                    embargo += 1
    return {"not_tested_once": int((tested != 1).sum()), "train_test_shared": int(shared), "train_test_overlap": overlap, "embargo_violations": embargo}
