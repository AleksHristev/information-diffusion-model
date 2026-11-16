"""Reproducible, low-priority parallel Monte Carlo with one random stream per replicate."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import numpy as np

T = TypeVar("T")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
LOW_PRIORITY = 19
DEFAULT_WORKERS = 2







def _start_worker() -> None:
    os.nice(LOW_PRIORITY)






def run_replicates(task: Callable[[np.random.SeedSequence], T], replicates: int, seed: int, *, workers: int | None = None) -> list[T]:
    """Run a task once per replicate, each with its own SeedSequence stream, so results do not depend on the number of workers.

    With more than one worker the replicates run in spawned processes at nice 19 with one numerical thread each; the
    thread-count environment variables (OMP_NUM_THREADS and the like) of the calling process are set to 1 and stay so.
    Arguments: task is a picklable top-level callable of a SeedSequence; replicates the number of replicates; seed the
    root seed; workers the number of processes (default 2; 1 or fewer runs in this process).
    Returns: the task results in replicate order.
    """
    streams = np.random.SeedSequence(seed).spawn(replicates)
    workers = DEFAULT_WORKERS if workers is None else workers
    if workers <= 1:
        return [task(stream) for stream in streams]
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_start_worker) as pool:
        return list(pool.map(task, streams, chunksize=1))
