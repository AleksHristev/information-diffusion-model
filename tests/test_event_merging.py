"""Merging of event times on a coarse clock, and the ceiling on the branching ratio."""

from __future__ import annotations

import numpy as np
import pytest

from information_diffusion.epidemic.event_merging import merge_events
from information_diffusion.epidemic.hawkes_exponential import ETA_CEILING


def test_coarsening_groups_events_into_bins_and_keeps_bin_starts():
    times = np.array([0.0, 0.05, 0.11, 0.99, 1.0, 1.99, 3.0])
    assert merge_events(times, resolution=1.0)[0].tolist() == [0.0, 1.0, 3.0]
    assert merge_events(times, resolution=0.1)[0].tolist() == pytest.approx([0.0, 0.1, 0.9, 1.0, 1.9, 3.0], abs=1e-6)


def test_eta_ceiling_is_the_module_constant():
    assert 0.0 < ETA_CEILING < 1.0
