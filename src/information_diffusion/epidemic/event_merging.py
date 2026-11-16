"""Merging of prints into one marked event per time cell (one millisecond by default)."""

from __future__ import annotations

import numpy as np

MERGE_RESOLUTION_SECONDS = 0.001


_FLOOR_TOLERANCE = 1e-7


def merge_events(times, marks=None, resolution: float = MERGE_RESOLUTION_SECONDS):
    """Merge prints in the same time cell into one event: cell a = floor(t / d) gives time a * d and the summed mark.

    Arguments: times are print times in seconds; marks the print marks (1 each when None); resolution is the cell width d
    in seconds (default 0.001).
    Returns: (sorted event times, merged marks).
    """
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    times = times[order]
    marks = np.ones_like(times) if marks is None else np.asarray(marks, dtype=float)[order]
    if times.size == 0:
        return times, marks
    cells = np.floor(times / resolution + _FLOOR_TOLERANCE)
    is_first = np.concatenate(([True], cells[1:] != cells[:-1]))
    merged_marks = np.bincount(np.cumsum(is_first) - 1, weights=marks)
    return cells[is_first] * resolution, merged_marks
