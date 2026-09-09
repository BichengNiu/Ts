"""Candidate execution and ranking primitives for automatic model selection.

The model-family modules own candidate construction and fitting.  This module
owns the small, shared lifecycle around those candidates: capturing warnings
and failures, preserving input order, and selecting the minimum information
criterion with stable first-in-order tie handling.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Generic, Iterable, Mapping, Sequence, TypeVar

from Ts.TsModels._parallel import _map_candidates


Candidate = TypeVar("Candidate")
Result = TypeVar("Result")
Task = TypeVar("Task")


@dataclass(frozen=True)
class GridSearchResult(Generic[Candidate, Result]):
    """Successful candidates and diagnostics from one serial grid search."""

    best_result: Result
    best_order: Candidate
    candidate_results: list[Result]
    candidate_orders: list[Candidate]
    n_attempted: int
    search_messages: list[str]


@dataclass(frozen=True)
class CandidateSearchResult(Generic[Task, Candidate, Result]):
    """Stable candidate lifecycle output shared by automatic model families."""

    candidate_tasks: list[Task]
    candidate_keys: list[Candidate]
    candidate_results: list[Result]
    criterion_values: list[float]
    n_attempted: int
    search_messages: list[str]
    best_index: int

    @property
    def best_key(self):
        """Return the selected candidate key."""
        return self.candidate_keys[self.best_index]

    @property
    def best_result(self):
        """Return the selected candidate result."""
        return self.candidate_results[self.best_index]


@dataclass(frozen=True)
class CandidateSearchLifecycle(Generic[Task, Candidate, Result]):
    """Completed search state, including optional refit and metadata."""

    search: CandidateSearchResult[Task, Candidate, Result]
    selected_result: Result
    search_messages: list[str]
    search_metadata: dict[str, object]


@dataclass
class SelectionMetadata:
    """Model-family-neutral record of one completed candidate selection."""

    best_result: object
    best_order: object
    candidate_results: list
    candidate_orders: list
    criterion_values: list[float]
    selection_criterion: str
    search_method: str
    n_attempted: int | None = None
    best_seasonal_order: object = None
    candidate_seasonal_orders: list = field(default_factory=list)
    search_messages: list[str] = field(default_factory=list)
    search_metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        *,
        best_result,
        best_order,
        candidate_results,
        candidate_orders,
        criterion_values,
        selection_criterion,
        search_method,
        n_attempted=None,
        best_seasonal_order=None,
        candidate_seasonal_orders=None,
        search_messages=None,
        search_metadata=None,
    ):
        """Create a detached metadata snapshot from a search outcome."""
        return cls(
            best_result=best_result,
            best_order=best_order,
            candidate_results=list(candidate_results),
            candidate_orders=list(candidate_orders),
            criterion_values=list(criterion_values),
            selection_criterion=selection_criterion,
            search_method=search_method,
            n_attempted=n_attempted,
            best_seasonal_order=best_seasonal_order,
            candidate_seasonal_orders=(
                []
                if candidate_seasonal_orders is None
                else list(candidate_seasonal_orders)
            ),
            search_messages=[] if search_messages is None else list(search_messages),
            search_metadata=(
                {} if search_metadata is None else dict(search_metadata)
            ),
        )


def finalize_candidate_search(
    search: CandidateSearchResult[Task, Candidate, Result],
    *,
    refit_selected: Callable[[Task], tuple[Result | None, Iterable[str]]] | None = None,
    metadata_factory: Callable[[], Mapping[str, object]] | None = None,
    elapsed_seconds: float | None = None,
) -> CandidateSearchLifecycle[Task, Candidate, Result]:
    """Complete a candidate search with selected refit and metadata capture.

    Candidate execution, scoring, warning collection, selected-candidate
    refitting, and search metadata are intentionally one lifecycle.  A model
    family supplies only its worker-specific refit callback and scheduler
    metadata factory.
    """

    selected_result = search.best_result
    search_messages = list(search.search_messages)
    if refit_selected is not None:
        selected_result, messages = refit_selected(
            search.candidate_tasks[search.best_index]
        )
        search_messages.extend(messages)
        if selected_result is None:
            raise RuntimeError(
                "The selected candidate could not be refit after parallel selection"
            )

    search_metadata = dict(metadata_factory() if metadata_factory else {})
    if elapsed_seconds is not None:
        search_metadata["elapsed_seconds"] = elapsed_seconds
    return CandidateSearchLifecycle(
        search=search,
        selected_result=selected_result,
        search_messages=search_messages,
        search_metadata=search_metadata,
    )


def run_candidate_search(
    tasks: Iterable[Task],
    worker: Callable[[Task], object],
    *,
    n_jobs: int,
    n_tasks: int,
    candidate_key: Callable[[Task, object], Candidate],
    candidate_result: Callable[[object], Result | None],
    criterion_value: Callable[[Result], float],
    messages: Callable[[object], Iterable[str]] = lambda _outcome: (),
    progress_callback=None,
    inner_max_num_threads=None,
) -> CandidateSearchResult[Task, Candidate, Result]:
    """Execute, collect, score, and select one candidate schedule.

    Model-family adapters provide only the task worker and the small outcome
    accessors.  This module owns stable ordering, failure-envelope collection,
    criterion scoring, and first-in-order tie handling for serial or process
    execution.
    """

    task_list = list(tasks)
    outcomes = _map_candidates(
        task_list,
        worker,
        n_jobs=n_jobs,
        n_tasks=n_tasks,
        progress_callback=progress_callback,
        inner_max_num_threads=inner_max_num_threads,
    )
    candidate_tasks: list[Task] = []
    candidate_keys: list[Candidate] = []
    candidate_results: list[Result] = []
    search_messages: list[str] = []
    for task, outcome in zip(task_list, outcomes, strict=True):
        search_messages.extend(messages(outcome))
        result = candidate_result(outcome)
        if result is None:
            continue
        candidate_tasks.append(task)
        candidate_keys.append(candidate_key(task, outcome))
        candidate_results.append(result)

    criterion_values = [criterion_value(result) for result in candidate_results]
    _, best_index = select_best_candidate(candidate_keys, criterion_values)
    return CandidateSearchResult(
        candidate_tasks=candidate_tasks,
        candidate_keys=candidate_keys,
        candidate_results=candidate_results,
        criterion_values=criterion_values,
        n_attempted=len(outcomes),
        search_messages=search_messages,
        best_index=best_index,
    )


def select_best_candidate(
    candidates: Sequence[Candidate],
    criterion_values: Sequence[float],
) -> tuple[Candidate, int]:
    """Return the minimum criterion candidate and its stable input index.

    Information criteria are minimized.  ``min`` is intentionally keyed by
    ``(value, index)`` so equal values preserve the first candidate's order in
    serial and parallel searches alike.
    """

    if len(candidates) != len(criterion_values):
        raise ValueError("candidates and criterion_values must have equal length")
    if not candidates:
        raise RuntimeError("No model converged during grid search")
    finite_positions = [
        position
        for position, value in enumerate(criterion_values)
        if isfinite(float(value))
    ]
    if not finite_positions:
        raise RuntimeError("No model produced a finite selection criterion")
    index = min(
        finite_positions,
        key=lambda position: (float(criterion_values[position]), position),
    )
    return candidates[index], index


def rank_candidates(
    candidates: Sequence[Candidate],
    criterion_values: Sequence[float],
) -> list[tuple[Candidate, float]]:
    """Return candidates ordered by criterion, preserving ties stably."""

    if len(candidates) != len(criterion_values):
        raise ValueError("candidates and criterion_values must have equal length")
    return [
        (candidates[index], float(criterion_values[index]))
        for index in sorted(
            range(len(candidates)),
            key=lambda position: (
                not isfinite(float(criterion_values[position])),
                float(criterion_values[position])
                if isfinite(float(criterion_values[position]))
                else float("inf"),
                position,
            ),
        )
    ]


def run_grid_search(
    orders: Iterable[Candidate],
    build_model: Callable[[Candidate], object],
    criterion: str,
    criterion_value: Callable[[Result, str], float],
    *,
    fit_kwargs: dict | None = None,
) -> GridSearchResult[Candidate, Result]:
    """Run the shared serial fit loop used by automatic model families.

    Failed candidates are recorded in ``search_messages`` and do not abort the
    search.  Warnings from successful candidates are retained; warnings raised
    by failed candidates remain suppressed alongside their failure message,
    matching the historical AutoGARCH behavior.
    """

    candidate_results: list[Result] = []
    candidate_orders: list[Candidate] = []
    criterion_values: list[float] = []
    search_messages: list[str] = []
    n_attempted = 0

    for order in orders:
        n_attempted += 1
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                model = build_model(order)
                result = model.fit(**(fit_kwargs or {}))
                value = criterion_value(result, criterion)
            except Exception as error:
                search_messages.append(f"{order}: {type(error).__name__}: {error}")
                continue
        search_messages.extend(
            f"{order}: {warning.category.__name__}: {warning.message}"
            for warning in caught_warnings
        )
        candidate_results.append(result)
        candidate_orders.append(order)
        criterion_values.append(float(value))

    best_order, best_index = select_best_candidate(
        candidate_orders,
        criterion_values,
    )
    return GridSearchResult(
        best_result=candidate_results[best_index],
        best_order=best_order,
        candidate_results=candidate_results,
        candidate_orders=candidate_orders,
        n_attempted=n_attempted,
        search_messages=search_messages,
    )


__all__ = [
    "CandidateSearchLifecycle",
    "CandidateSearchResult",
    "GridSearchResult",
    "SelectionMetadata",
    "finalize_candidate_search",
    "rank_candidates",
    "run_candidate_search",
    "run_grid_search",
    "select_best_candidate",
]
