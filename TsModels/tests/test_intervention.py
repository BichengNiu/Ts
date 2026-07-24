"""Tests for dated intervention specifications and design matrices."""

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
