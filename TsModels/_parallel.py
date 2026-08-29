"""Private parallel execution helpers for automatic model selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import TypeVar, cast


T = TypeVar("T")
R = TypeVar("R")
ProgressCallback = Callable[[int, int], None]


def _validate_n_jobs(n_jobs: int) -> int:
    """Validate a requested automatic-selection worker count."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral):
        raise TypeError("n_jobs must be an integer")
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0; use 1 for serial execution")
    return n_jobs


def _resolve_n_jobs(n_jobs: int, n_tasks: int) -> int:
    """Resolve a worker request to a bounded positive process count."""
    n_jobs = _validate_n_jobs(n_jobs)
    if isinstance(n_tasks, bool) or not isinstance(n_tasks, Integral):
        raise TypeError("n_tasks must be an integer")
    n_tasks = int(n_tasks)
    if n_tasks < 0:
        raise ValueError("n_tasks must be non-negative")
    if n_tasks <= 1:
        return 1

    cpu_count = os_cpu_count()
    max_jobs = max(1, cpu_count - 1)
    if n_jobs == -1:
        requested = max_jobs
    elif n_jobs < -1:
        requested = max(1, min(cpu_count + n_jobs, max_jobs))
    else:
        requested = min(max(1, n_jobs), max_jobs)
    return min(requested, n_tasks)


def os_cpu_count() -> int:
    """Return a positive CPU count even on systems with incomplete metadata."""
    import os

    count = os.cpu_count()
    return count if count is not None and count > 0 else 1


def _map_candidates(
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    n_jobs: int,
    n_tasks: int,
    progress_callback: ProgressCallback | None = None,
) -> list[R]:
    """Evaluate candidate tasks in stable input order.

    The caller supplies ``n_tasks`` so large lazy candidate generators do not
    need to be materialized merely to resolve the worker count.  ``loky`` is
    deliberately fixed here: automatic model fitting is CPU-bound and this
    helper must not expose a second backend policy to each model family.
    """
    effective_jobs = _resolve_n_jobs(n_jobs, n_tasks)
    if effective_jobs == 1:
        results = []
        for completed, item in enumerate(items, start=1):
            results.append(worker(item))
            if progress_callback is not None:
                progress_callback(completed, n_tasks)
        if len(results) != n_tasks:
            raise ValueError("n_tasks does not match the candidate iterable")
        return results

    from joblib import Parallel, delayed

    def indexed_results():
        for index, item in enumerate(items):
            yield delayed(_run_indexed_candidate)(index, worker, item)

    generator = Parallel(
        n_jobs=effective_jobs,
        backend="loky",
        prefer="processes",
        batch_size="auto",
        pre_dispatch="2*n_jobs",
        return_as="generator_unordered",
    )(indexed_results())
    ordered: list[R | None] = [None] * n_tasks
    completed = 0
    for index, result in generator:
        ordered[index] = result
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, n_tasks)
    if completed != n_tasks:
        raise ValueError("n_tasks does not match the candidate iterable")
    return cast(list[R], ordered)


def _run_indexed_candidate(
    index: int,
    worker: Callable[[T], R],
    item: T,
) -> tuple[int, R]:
    """Run one candidate and return its input index for stable reordering."""
    return index, worker(item)


__all__ = ["_map_candidates", "_resolve_n_jobs", "_validate_n_jobs"]
