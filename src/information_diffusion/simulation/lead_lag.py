"""Lead-lag network simulator: asynchronously observed leader and follower prices on a variance clock, and VAR paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_SECONDS = 5400.0
RELEASE_SECONDS = 1800.0
OPEN_WINDOW_SECONDS = 1800.0
STEP_SECONDS = 5.0
STEPS_PER_EVENT = int(WINDOW_SECONDS / 5)

SPIKE_AMPLITUDE = 9.0
SPIKE_DECAY_SECONDS = 300.0
VOLATILITY_PER_ROOT_SECOND = 0.20 / np.sqrt(252) / np.sqrt(23400.0)
HALF_SPREAD_LOG = 0.005 / 100.0
MIDPOINT_NOISE_LOG = (0.005 / 100.0) / np.sqrt(12.0)

LAG_SECONDS = (0.01, 0.1, 0.5, 2.0)
COMMON_SHARE = 0.5
FOLLOWER_LINK_VARIANCE = 0.5
SECTOR_VARIANCE = 0.15
IDIOSYNCRATIC_VARIANCE = 0.35
LEADERS = ("SPY", "QQQ", "IWM")
LIQUID_FOLLOWERS = ("SMH", "SOXX", "XLE", "XLK", "XLF")
_PRE_WINDOW_SECONDS = 10.0
SECTORS = {
    "XLK": "technology", "SMH": "technology", "SOXX": "technology", "VGT": "technology", "IGV": "technology",
    "XLF": "financials", "KRE": "financials", "KBE": "financials",
    "XLE": "energy", "XOP": "energy", "OIH": "energy",
    "XLRE": "real_estate", "IYR": "real_estate", "VNQ": "real_estate",
    "XLY": "cyclical", "XRT": "cyclical", "XHB": "cyclical", "ITB": "cyclical",
    "XLP": "defensive", "XLU": "defensive", "XLV": "defensive", "IBB": "defensive", "XBI": "defensive",
    "XLI": "industrial", "XLB": "industrial", "XME": "industrial", "XLC": "industrial",
}


@dataclass(frozen=True)
class Window:
    """A simulation window on the simulator's clock, in seconds from the clock start (the release is at 1,800 seconds).

    Fields: name; start; seconds, the length; all_instruments, whether thin instruments are simulated as well.
    The predefined windows are WINDOW_A, 5,400 seconds from 0 containing the release, and WINDOW_B, 1,800 seconds from
    5,400.
    """

    name: str
    start: float
    seconds: float
    all_instruments: bool

    @property
    def end(self) -> float:
        """End of the window in seconds on the simulator's clock."""
        return self.start + self.seconds

    @property
    def steps(self) -> int:
        """Number of 5-second steps in the window."""
        return int(self.seconds / STEP_SECONDS)


WINDOW_A = Window("a", 0.0, WINDOW_SECONDS, False)
WINDOW_B = Window("b", WINDOW_SECONDS, OPEN_WINDOW_SECONDS, True)
WINDOWS = (WINDOW_A, WINDOW_B)


def cells(window: Window) -> tuple[tuple[float, bool], ...]:
    """Simulation cells (lag in seconds, thin follower) of a window: every lag in LAG_SECONDS, with thin followers only when the window has all instruments."""
    return tuple((lag, thin) for lag in LAG_SECONDS for thin in ((False, True) if window.all_instruments else (False,)))


def shape(t):
    """Activity and variance multiplier g(t) = 1 + 9 exp(-(t - 1800) / 300) for t >= 1,800 seconds, else 1."""
    t = np.asarray(t, dtype=float)
    return 1.0 + SPIKE_AMPLITUDE * np.exp(-np.maximum(t - RELEASE_SECONDS, 0.0) / SPIKE_DECAY_SECONDS) * (t >= RELEASE_SECONDS)


def cumulative_shape(t):
    """Variance clock G(t), the integral of g from 0, extended linearly for t < 0."""
    t = np.asarray(t, dtype=float)
    return t + SPIKE_AMPLITUDE * SPIKE_DECAY_SECONDS * (1.0 - np.exp(-np.maximum(t - RELEASE_SECONDS, 0.0) / SPIKE_DECAY_SECONDS)) * (t >= RELEASE_SECONDS)


_TIME_TABLE = np.arange(0.0, WINDOW_B.end + 0.25, 0.25)
_SHAPE_TABLE = cumulative_shape(_TIME_TABLE)


def observation_times(expected_count: float, rng: np.random.Generator, window: Window = WINDOW_A) -> np.ndarray:
    """Observation times on a window as an inhomogeneous Poisson process with intensity proportional to g, by inversion of G.

    Arguments: expected_count is the expected number of observations; rng the generator; window the Window.
    Returns: sorted times in seconds on the simulator's clock.
    """
    count = int(rng.poisson(expected_count))
    if count == 0:
        return np.empty(0)
    low, high = float(cumulative_shape(window.start)), float(cumulative_shape(window.end))
    return np.interp(low + np.sort(rng.random(count)) * (high - low), _SHAPE_TABLE, _TIME_TABLE)


def _brownian_at(sorted_times: np.ndarray, rng: np.random.Generator, window: Window = WINDOW_A) -> np.ndarray:
    clock = cumulative_shape(sorted_times)
    start = float(cumulative_shape(window.start - _PRE_WINDOW_SECONDS))
    steps = np.diff(clock, prepend=start)
    return np.cumsum(np.sqrt(np.maximum(steps, 0.0)) * rng.standard_normal(sorted_times.size))


def _observed(efficient: np.ndarray, quotes: bool, rng: np.random.Generator) -> np.ndarray:
    if quotes:
        return efficient + MIDPOINT_NOISE_LOG * rng.standard_normal(efficient.size)
    return efficient + HALF_SPREAD_LOG * rng.choice((-1.0, 1.0), size=efficient.size)


@dataclass(frozen=True)
class Network:
    """A simulated lead-lag network.

    Leader l has efficient log price X_l = sqrt(a) C + sqrt(1 - a) W_l with a common factor C; follower f has
    X_f(t) = sum_l lambda_lf X_l(t - theta_lf) + b S(t) + c W_f(t), with lambda_lf and theta_lf the loading and lag of
    each link, S a sector factor shared by followers of one sector, b = sqrt(0.15) and c = sqrt(0.35); all paths run on
    the variance clock G and are scaled to 20% annual volatility at g = 1 (252 days of 23,400 seconds).
    Fields: names, the tickers (every follower name must be a key of SECTORS); expected_observations, the expected
    number of observations per node in the window; thin, whether each node is observed on quote midpoints (with
    rounding noise) rather than trades (with bid-ask bounce); leaders, the leader indices; links, a DataFrame with
    columns leader, follower, lag_seconds and loading; common_share, a; window, the Window simulated.
    """

    names: tuple[str, ...]
    expected_observations: np.ndarray
    thin: np.ndarray
    leaders: tuple[int, ...]
    links: pd.DataFrame
    common_share: float = COMMON_SHARE
    window: Window = WINDOW_B

    def followers_of(self, leader: int) -> pd.DataFrame:
        """Links out of the given leader index."""
        return self.links[self.links.leader == leader]

    def leaders_of(self, follower: int) -> pd.DataFrame:
        """Links into the given follower index."""
        return self.links[self.links.follower == follower]




def link_lag(network: Network) -> dict[tuple[int, int], float]:
    """True lag in seconds of each (leader, follower) link of the network; a pair absent from the map is not linked."""
    return {(int(row.leader), int(row.follower)): float(row.lag_seconds) for row in network.links.itertuples()}


def simulate_event(network: Network, rng: np.random.Generator, followers=None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Simulate one event of the network on its window; efficient prices are drawn only at the times needed, so lags are exact.

    Arguments: network is the Network; rng the generator; followers optional follower indices to simulate (all by
    default); leaders are always simulated.
    Returns: a dict from node index to (seconds since the window opened, observed log prices).
    Reference: Roll (1984), Journal of Finance (bid-ask bounce).
    """
    window = network.window
    wanted = [node for node in range(len(network.names)) if node not in network.leaders] if followers is None else sorted(set(followers))
    active = sorted(set(network.leaders) | set(wanted))
    times = {node: observation_times(network.expected_observations[node], rng, window) for node in active}
    links = network.links[network.links.follower.isin(wanted)]
    queries = {leader: [times[leader]] for leader in network.leaders}
    slots: dict[int, list[tuple[int, float, int, int]]] = {leader: [] for leader in network.leaders}
    for link in links.itertuples():
        wanted_times = times[int(link.follower)] - link.lag_seconds
        stack = queries[int(link.leader)]
        begin = sum(array.size for array in stack)
        stack.append(wanted_times)
        slots[int(link.leader)].append((int(link.follower), float(link.loading), begin, begin + wanted_times.size))
    all_queries = np.concatenate([np.concatenate(queries[leader]) for leader in network.leaders])
    grid = np.sort(all_queries)
    common = _brownian_at(grid, rng, window)
    efficient_on_grid = {}
    for leader in network.leaders:
        stack = np.concatenate(queries[leader])
        order = np.argsort(stack, kind="stable")
        rank = np.empty(stack.size, dtype=np.int64)
        rank[order] = np.arange(stack.size)
        sorted_stack = stack[order]
        path = np.sqrt(network.common_share) * common[np.searchsorted(grid, sorted_stack)] + np.sqrt(1 - network.common_share) * _brownian_at(sorted_stack, rng, window)
        efficient_on_grid[leader] = path[rank]
    efficient = {node: np.zeros(times[node].size) for node in wanted}
    for leader in network.leaders:
        path = efficient_on_grid[leader]
        for follower, loading, begin, end in slots[leader]:
            efficient[follower] += loading * path[begin:end]
    sector_groups: dict[str, list[int]] = {}
    for node in wanted:
        sector_groups.setdefault(SECTORS[network.names[node]], []).append(node)
    for members in sector_groups.values():
        stack = np.concatenate([times[node] for node in members])
        order = np.argsort(stack, kind="stable")
        sorted_stack = stack[order]
        factor = np.empty(stack.size)
        factor[order] = _brownian_at(sorted_stack, rng, window)
        begin = 0
        for node in members:
            end = begin + times[node].size
            efficient[node] += np.sqrt(SECTOR_VARIANCE) * factor[begin:end]
            begin = end
    for node in wanted:
        if times[node].size:
            efficient[node] += np.sqrt(IDIOSYNCRATIC_VARIANCE) * _brownian_at(times[node], rng, window)
    observed = {}
    for leader in network.leaders:
        own = efficient_on_grid[leader][: times[leader].size]
        observed[leader] = (times[leader] - window.start, _observed(VOLATILITY_PER_ROOT_SECOND * own, False, rng))
    for node in wanted:
        observed[node] = (times[node] - window.start, _observed(VOLATILITY_PER_ROOT_SECOND * efficient[node], bool(network.thin[node]), rng))
    return observed


def simulate_correlated_pair(expected_x: float, expected_y: float, correlation: float, rng: np.random.Generator, *, quotes_x: bool = False, quotes_y: bool = False, window: Window = WINDOW_A):
    """Simulate two nodes whose efficient increments have a constant correlation at zero lag, with the network's sampling and noise.

    Arguments: expected_x and expected_y are the expected observation counts; correlation the increment correlation;
    rng the generator; quotes_x and quotes_y select quote-midpoint noise instead of bid-ask bounce; window the Window.
    Returns: (times_x, log prices x, times_y, log prices y), times in seconds since the window opened.
    """
    tx, ty = observation_times(expected_x, rng, window), observation_times(expected_y, rng, window)
    stack = np.concatenate([tx, ty])
    order = np.argsort(stack, kind="stable")
    sorted_stack = stack[order]
    common = np.empty(stack.size)
    common[order] = _brownian_at(sorted_stack, rng, window)
    x = np.sqrt(correlation) * common[: tx.size] + np.sqrt(1 - correlation) * _brownian_at(tx, rng, window)
    y = np.sqrt(correlation) * common[tx.size :] + np.sqrt(1 - correlation) * _brownian_at(ty, rng, window)
    return tx - window.start, _observed(VOLATILITY_PER_ROOT_SECOND * x, quotes_x, rng), ty - window.start, _observed(VOLATILITY_PER_ROOT_SECOND * y, quotes_y, rng)


def innovation_scale(step: np.ndarray, window: Window = WINDOW_A) -> np.ndarray:
    """Standard deviation multiplier sqrt((G(end) - G(start)) / 5) of each 5-second step index of the window."""
    start = window.start + step * STEP_SECONDS
    end = start + STEP_SECONDS
    return np.sqrt((cumulative_shape(end) - cumulative_shape(start)) / STEP_SECONDS)


def var_design(link_scale: float, own_lag: float = 0.1, innovation_common: float = 0.3, network: Network | None = None, strong_share: float = 0.5, strong_multiple: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """True VAR(1) coefficient matrix A and innovation covariance Sigma on the network's links.

    A has own_lag on the diagonal and, on each link (leader into follower), link_scale, or strong_multiple * link_scale
    for every round(1 / strong_share)-th link. Followers do not feed back into leaders, so the process is stable.
    Sigma = innovation_common 11' + (1 - innovation_common) I.
    Arguments: link_scale, own_lag, innovation_common, strong_share and strong_multiple as above; network is required.
    Returns: (A, Sigma).
    """
    if network is None:
        raise ValueError("network is required")
    size = len(network.names)
    coefficient = np.diag(np.full(size, own_lag))
    strong = np.arange(len(network.links)) % round(1 / strong_share) == 0
    for position, link in enumerate(network.links.itertuples()):
        coefficient[int(link.follower), int(link.leader)] = link_scale * (strong_multiple if strong[position] else 1.0)
    covariance = innovation_common * np.ones((size, size)) + (1 - innovation_common) * np.eye(size)
    return coefficient, covariance


def simulate_var_events(coefficient: np.ndarray, covariance: np.ndarray, rng: np.random.Generator, *, events: int, steps: int | None = None, burn_in: int = 50, window: Window = WINDOW_A) -> list[np.ndarray]:
    """Simulate VAR(1) paths y_t = A y_{t-1} + scale(t) eps_t, eps ~ N(0, Sigma), on the 5-second grid of the window.

    scale follows the variance clock; each event is an independent path started from a burn-in at scale one.
    Arguments: coefficient is A; covariance Sigma; rng the generator; events the number of paths; steps the number of
    steps (default the window's); burn_in the burn-in steps; window the Window.
    Returns: a list of (steps, nodes) arrays, one per event.
    """
    root = np.linalg.cholesky(covariance)
    size = coefficient.shape[0]
    steps = window.steps if steps is None else steps
    scale = innovation_scale(np.arange(steps), window)
    state = np.zeros((events, size))
    for _ in range(burn_in):
        state = state @ coefficient.T + rng.standard_normal((events, size)) @ root.T
    paths = np.empty((steps, events, size))
    for step in range(steps):
        state = state @ coefficient.T + scale[step] * (rng.standard_normal((events, size)) @ root.T)
        paths[step] = state
    return [paths[:, event] for event in range(events)]
