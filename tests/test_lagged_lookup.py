"""The integer-nanosecond lagged leader lookup at the lag boundary and at a negative lag."""

import numpy as np

from information_diffusion.influence import lagged_lookup as ll


def exact_reference(times_ns, values, lag_ns, steps, step_ns=5_000_000_000):
    out = []
    for k in range(steps):
        values_at = []
        for q in (k * step_ns - lag_ns, k * step_ns - lag_ns - step_ns):
            seen = [v for t, v in zip(times_ns, values) if t <= q]
            values_at.append(seen[-1] if seen else np.nan)
        out.append(values_at[0] - values_at[1])
    return np.array(out)


def test_a_trade_exactly_at_the_lagged_time_is_kept():
    steps, lag = 6, 0.433
    times_ns = np.array([0, 10_000_000_000 - 433_000_000, 20_000_000_000 - 433_000_000], dtype=np.int64)
    values = np.array([0.0, 1.0, 3.0])
    got = ll.leader_return_ns((times_ns, values), lag, steps)
    np.testing.assert_array_equal(got, exact_reference(times_ns, values, 433_000_000, steps))
    assert got[2] == 1.0 and got[4] == 2.0


def test_a_negative_lag_is_floored_at_zero():
    times_ns = np.array([0, 4_000_000_000, 6_000_000_000], dtype=np.int64)
    values = np.array([0.0, 1.0, 5.0])
    np.testing.assert_array_equal(ll.leader_return_ns((times_ns, values), -2.0, 3), ll.leader_return_ns((times_ns, values), 0.0, 3))


