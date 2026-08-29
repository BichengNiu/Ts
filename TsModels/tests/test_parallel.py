"""Tests for the private automatic-selection parallel executor."""

import pytest

from Ts.TsModels._parallel import (
    _map_candidates,
    _resolve_n_jobs,
    _resolve_sarimax_candidate_schedule,
    _validate_n_jobs,
)


def _increment(value):
    """Pickle-safe worker used to verify stable parallel ordering."""
    return value + 1


def test_resolve_n_jobs_reserves_one_cpu_and_caps_tasks(monkeypatch):
    """Automatic workers use CPU-1 and never exceed candidate count."""
    monkeypatch.setattr("Ts.TsModels._parallel.os_cpu_count", lambda: 8)

    assert _resolve_n_jobs(-1, 100) == 7
    assert _resolve_n_jobs(-2, 100) == 6
    assert _resolve_n_jobs(100, 4) == 4
    assert _resolve_n_jobs(-1, 1) == 1


def test_sarimax_schedule_uses_bounded_parallelism_for_substantial_search(
    monkeypatch,
):
    """SARIMAX limits process workers while retaining CPU and task bounds."""
    monkeypatch.setattr("Ts.TsModels._parallel.os_cpu_count", lambda: 8)

    schedule = _resolve_sarimax_candidate_schedule(
        -1,
        32,
        nobs=120,
        n_exog=2,
        model_complexity=5,
    )

    assert schedule.mode == "parallel"
    assert schedule.worker_count == 4
    assert schedule.metadata()["candidate_count"] == 32


def test_sarimax_schedule_falls_back_for_small_or_light_searches():
    """Candidate startup and IPC costs do not dominate a small grid."""
    small = _resolve_sarimax_candidate_schedule(
        -1,
        4,
        nobs=10_000,
        n_exog=10,
        model_complexity=10,
    )
    light = _resolve_sarimax_candidate_schedule(
        -1,
        8,
        nobs=10,
        n_exog=0,
        model_complexity=1,
    )

    assert (small.mode, small.reason) == (
        "serial",
        "candidate_count_below_threshold",
    )
    assert (light.mode, light.reason) == (
        "serial",
        "estimated_work_below_threshold",
    )


def test_map_candidates_preserves_input_order_with_processes():
    """Parallel results retain candidate-generation order."""
    assert _map_candidates(
        range(4),
        _increment,
        n_jobs=2,
        n_tasks=4,
    ) == [1, 2, 3, 4]


def test_parallel_candidate_mapping_reports_parent_progress():
    """Parallel completion callbacks run in order of completion count."""
    progress = []

    result = _map_candidates(
        range(4),
        _increment,
        n_jobs=2,
        n_tasks=4,
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert result == [1, 2, 3, 4]
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


@pytest.mark.parametrize("value", [0, True, "2"])
def test_validate_n_jobs_rejects_invalid_values(value):
    """Zero, booleans, and non-integers are not valid worker requests."""
    with pytest.raises((TypeError, ValueError)):
        _validate_n_jobs(value)
