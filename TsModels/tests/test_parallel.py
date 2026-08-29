"""Tests for the private automatic-selection parallel executor."""

import pytest

from Ts.TsModels._parallel import (
    _map_candidates,
    _resolve_n_jobs,
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
