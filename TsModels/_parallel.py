"""Private parallel execution helpers for automatic model selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import TypeVar, cast


T = TypeVar("T")
R = TypeVar("R")
ProgressCallback = Callable[[int, int], None]

_SARIMAX_MIN_PARALLEL_TASKS = 8
_SARIMAX_MIN_PARALLEL_WORK = 1024
_SARIMAX_MAX_WORKERS = 4


@dataclass(frozen=True)
class _CandidateSchedule:
    """Resolved execution policy for one bounded candidate search."""

    mode: str
    worker_count: int
    candidate_count: int
    estimated_work: int
    reason: str

    @property
    def is_parallel(self) -> bool:
        """Whether this schedule uses more than one process."""
        return self.worker_count > 1

    def metadata(self) -> dict[str, int | float | str]:
        """Return stable, user-reportable execution facts."""
        return {
            "mode": self.mode,
            "worker_count": self.worker_count,
            "candidate_count": self.candidate_count,
            "estimated_work": self.estimated_work,
            "reason": self.reason,
        }


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


def _resolve_sarimax_candidate_schedule(
    n_jobs: int,
    n_tasks: int,
    *,
    nobs: int,
    n_exog: int,
    model_complexity: int,
) -> _CandidateSchedule:
    """Choose a bounded serial or process schedule for automatic SARIMAX.

    Candidate fits are independent, but short grids and very small models are
    dominated by process startup and IPC costs.  The policy therefore keeps
    those searches serial and caps CPU-bound fits at four processes.
    """
    n_jobs = _validate_n_jobs(n_jobs)
    for name, value in {
        "n_tasks": n_tasks,
        "nobs": nobs,
        "n_exog": n_exog,
        "model_complexity": model_complexity,
    }.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    estimated_work = int(n_tasks) * max(1, int(nobs))
    estimated_work *= max(1, int(n_exog) + 1) * max(1, int(model_complexity))
    if n_jobs == 1:
        return _CandidateSchedule(
            mode="serial",
            worker_count=1,
            candidate_count=int(n_tasks),
            estimated_work=estimated_work,
            reason="explicit_serial_request",
        )
    if n_tasks < _SARIMAX_MIN_PARALLEL_TASKS:
        return _CandidateSchedule(
            mode="serial",
            worker_count=1,
            candidate_count=int(n_tasks),
            estimated_work=estimated_work,
            reason="candidate_count_below_threshold",
        )
    if estimated_work < _SARIMAX_MIN_PARALLEL_WORK:
        return _CandidateSchedule(
            mode="serial",
            worker_count=1,
            candidate_count=int(n_tasks),
            estimated_work=estimated_work,
            reason="estimated_work_below_threshold",
        )

    worker_count = min(
        _resolve_n_jobs(n_jobs, n_tasks),
        _SARIMAX_MAX_WORKERS,
    )
    return _CandidateSchedule(
        mode="parallel" if worker_count > 1 else "serial",
        worker_count=worker_count,
        candidate_count=int(n_tasks),
        estimated_work=estimated_work,
        reason=(
            "bounded_process_parallelism"
            if worker_count > 1
            else "single_available_worker"
        ),
    )


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
    inner_max_num_threads: int | None = None,
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

    from joblib import Parallel, delayed, parallel_config

    if inner_max_num_threads is not None:
        if (
            isinstance(inner_max_num_threads, bool)
            or not isinstance(inner_max_num_threads, Integral)
            or inner_max_num_threads < 1
        ):
            raise ValueError("inner_max_num_threads must be a positive integer")

    def indexed_results():
        for index, item in enumerate(items):
            yield delayed(_run_indexed_candidate)(index, worker, item)

    parallel_kwargs = {
        "n_jobs": effective_jobs,
        "batch_size": 1,
        "pre_dispatch": "n_jobs",
        "return_as": "generator_unordered",
    }
    if inner_max_num_threads is None:
        generator = Parallel(
            backend="loky",
            prefer="processes",
            **parallel_kwargs,
        )(indexed_results())
    else:
        with parallel_config(
            backend="loky",
            inner_max_num_threads=inner_max_num_threads,
        ):
            generator = Parallel(prefer="processes", **parallel_kwargs)(
                indexed_results()
            )
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


__all__ = [
    "_map_candidates",
    "_resolve_n_jobs",
    "_resolve_sarimax_candidate_schedule",
    "_validate_n_jobs",
]
