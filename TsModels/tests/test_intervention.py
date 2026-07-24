"""Tests for dated intervention specifications and design matrices."""

import numpy as np
import pandas as pd
import pytest


def test_event_spec_defaults_to_period_mapping():
    from Ts.TsModels._intervention import EventSpec

    event = EventSpec(name="policy", dates=["2025-03-15"], kind="pulse")

    assert event.date_rule == "period"
    assert event.window is None
    assert event.reference is None
    assert event.name == "policy"
    assert event.dates == (pd.Timestamp("2025-03-15"),)


@pytest.mark.parametrize("kind", ["other", "", None])
def test_event_spec_rejects_unknown_kind(kind):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="kind"):
        EventSpec(name="policy", dates=["2025-03-15"], kind=kind)


@pytest.mark.parametrize("name", ["", "   ", None])
def test_event_spec_rejects_empty_name(name):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="name"):
        EventSpec(name=name, dates=["2025-03-15"], kind="pulse")


def test_event_spec_rejects_empty_or_duplicate_dates():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="dates"):
        EventSpec(name="policy", dates=[], kind="pulse")
    with pytest.raises(ValueError, match="duplicate"):
        EventSpec(
            name="policy",
            dates=["2025-03-15", "2025-03-15"],
            kind="pulse",
        )


def test_event_spec_rejects_unknown_date_rule():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="date_rule"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="pulse",
            date_rule="nearest",
        )


def test_step_rejects_dynamic_window():
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="window.*pulse"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="step",
            window=(-2, 4),
            reference=-1,
        )


@pytest.mark.parametrize(
    ("window", "reference"),
    [
        ((-2, 4), None),
        (None, -1),
        ((4, -2), 0),
        ((-2.0, 4), 0),
        ((False, 4), 0),
        ((-2, 4), 5),
    ],
)
def test_event_spec_rejects_invalid_window_reference(window, reference):
    from Ts.TsModels._intervention import EventSpec

    with pytest.raises(ValueError, match="window|reference"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="pulse",
            window=window,
            reference=reference,
        )


def test_reference_zero_is_valid_and_event_is_immutable():
    from dataclasses import FrozenInstanceError

    from Ts.TsModels._intervention import EventSpec

    event = EventSpec(
        name="policy",
        dates=["2025-03-15"],
        kind="pulse",
        window=(-2, 2),
        reference=0,
    )

    assert event.reference == 0
    with pytest.raises(FrozenInstanceError):
        event.name = "changed"


def test_period_maps_within_month_to_monthly_observation():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=5, freq="MS")
    matrix, metadata = build_event_matrix(
        dates,
        [EventSpec("policy", ["2025-03-15"], "pulse")],
    )

    assert matrix["event__policy"].tolist() == [0, 0, 1, 0, 0]
    assert metadata["policy"].columns == ("event__policy",)
    assert metadata["policy"].mapped_positions == (2,)


@pytest.mark.parametrize(
    ("freq", "start", "event_date", "expected_position"),
    [
        ("ME", "2025-01-31", "2025-03-15", 2),
        ("D", "2025-01-01", "2025-01-03 12:00", 2),
        ("QS", "2025-01-01", "2025-05-10", 1),
    ],
)
def test_period_mapping_supports_common_regular_frequencies(
    freq,
    start,
    event_date,
    expected_position,
):
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range(start, periods=5, freq=freq)
    matrix, _ = build_event_matrix(
        dates,
        [EventSpec("policy", [event_date], "pulse")],
    )

    assert np.flatnonzero(matrix["event__policy"]).tolist() == [
        expected_position
    ]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("next", 2),
        ("previous", 1),
    ],
)
def test_directional_date_rules(rule, expected):
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=4, freq="MS")
    matrix, _ = build_event_matrix(
        dates,
        [
            EventSpec(
                "policy",
                ["2025-02-15"],
                "pulse",
                date_rule=rule,
            )
        ],
    )

    assert np.flatnonzero(matrix["event__policy"]).tolist() == [expected]


def test_exact_rejects_missing_date_inside_calendar():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.DatetimeIndex(["2025-01-01", "2025-01-03", "2025-01-04"])

    with pytest.raises(ValueError, match="exact.*2025-01-02"):
        build_event_matrix(
            dates,
            [
                EventSpec(
                    "policy",
                    ["2025-01-02"],
                    "pulse",
                    date_rule="exact",
                )
            ],
        )


def test_period_rejects_irregular_calendar():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.DatetimeIndex(
        ["2025-01-01", "2025-01-02", "2025-01-04"]
    )

    with pytest.raises(ValueError, match="frequency"):
        build_event_matrix(
            dates,
            [EventSpec("policy", ["2025-01-02"], "pulse")],
        )


def test_repeated_step_dates_create_cumulative_staircase():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    matrix, _ = build_event_matrix(
        dates,
        [
            EventSpec(
                "policy",
                ["2025-02-10", "2025-04-20"],
                "step",
            )
        ],
    )

    assert matrix["event__policy"].tolist() == [0, 1, 1, 2, 2, 2]


def test_event_window_excludes_reference_and_counts_overlap():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    event = EventSpec(
        "announcement",
        ["2025-01-03", "2025-01-04"],
        "pulse",
        window=(-1, 2),
        reference=-1,
        date_rule="exact",
    )
    matrix, metadata = build_event_matrix(dates, [event])

    assert "event__announcement__m1" not in matrix
    assert matrix.loc["2025-01-04", "event__announcement__p1"] == 1
    assert matrix.loc["2025-01-04", "event__announcement__p0"] == 1
    assert metadata["announcement"].relative_periods == (0, 1, 2)


def test_step_before_target_slice_remains_active():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    calendar = pd.date_range("2025-01-01", periods=6, freq="MS")
    target = calendar[3:]
    matrix, _ = build_event_matrix(
        target,
        [EventSpec("policy", ["2025-02-15"], "step")],
        calendar=calendar,
    )

    assert matrix["event__policy"].tolist() == [1, 1, 1]


def test_event_outside_calendar_contributes_zero_until_extension():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    history = pd.date_range("2025-01-01", periods=4, freq="MS")
    event = EventSpec("policy", ["2025-06-15"], "pulse")

    history_matrix, _ = build_event_matrix(history, [event])
    extended = pd.date_range("2025-01-01", periods=6, freq="MS")
    future_matrix, _ = build_event_matrix(
        extended[-2:],
        [event],
        calendar=extended,
    )

    assert not history_matrix.to_numpy().any()
    assert future_matrix["event__policy"].tolist() == [0, 1]


def test_build_event_matrix_rejects_name_collisions():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    dates = pd.date_range("2025-01-01", periods=4, freq="MS")
    event = EventSpec("policy", ["2025-02-01"], "pulse")

    with pytest.raises(ValueError, match="duplicate event name"):
        build_event_matrix(dates, [event, event])
    with pytest.raises(ValueError, match="collision"):
        build_event_matrix(
            dates,
            [event],
            reserved_names=["event__policy"],
        )


def test_build_event_matrix_requires_target_subset_and_matching_timezone():
    from Ts.TsModels._intervention import EventSpec, build_event_matrix

    calendar = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="D",
        tz="UTC",
    )
    event = EventSpec(
        "policy",
        [pd.Timestamp("2025-01-02", tz="Europe/London")],
        "pulse",
        date_rule="exact",
    )

    with pytest.raises(ValueError, match="timezone"):
        build_event_matrix(calendar, [event])
    with pytest.raises(ValueError, match="target dates"):
        build_event_matrix(
            pd.DatetimeIndex(["2025-01-05"], tz="UTC"),
            [],
            calendar=calendar,
        )
