"""Tests for explicit forecast-evaluation split schemes."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsMetrics._schemes import Holdout, RollingOrigin


def test_holdout_resolves_closed_positional_bounds_and_gap():
    split = Holdout(train=(0, 9), test=(12, 14)).split(20, None)[0]

    assert split.split == 0
    assert split.train_indices.tolist() == list(range(10))
    assert split.target_indices.tolist() == [12, 13, 14]
    assert split.train_start == 0
    assert split.train_end == 9
    assert split.forecast_start == 12
    assert split.forecast_end == 14
    assert split.gap == 2
    assert split.window == "holdout"


def test_holdout_preserves_calendar_labels():
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    split = Holdout(
        train=(dates[0], dates[9]),
        test=(dates[10], dates[12]),
    ).split(20, dates)[0]

    assert split.train_start == dates[0]
    assert split.train_end == dates[9]
    assert split.forecast_start == dates[10]
    assert split.forecast_end == dates[12]


@pytest.mark.parametrize(
    ("scheme", "error", "match"),
    [
        (Holdout(train=(0, 8), test=(9, 10)), ValueError, "at least 10"),
        (Holdout(train=(0, 9), test=(9, 10)), ValueError, "strictly later"),
        (Holdout(train=(5, 4), test=(10, 11)), ValueError, "start"),
        (Holdout(train=0, test=(10, 11)), TypeError, "pair"),
        (Holdout(train=(0, 9), test=(10, 20)), ValueError, "outside"),
    ],
)
def test_holdout_rejects_invalid_periods(scheme, error, match):
    with pytest.raises(error, match=match):
        scheme.split(20, None)


def test_holdout_rejects_absent_calendar_bound():
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")

    with pytest.raises(ValueError, match="does not exist"):
        Holdout(
            train=(dates[0], dates[9]),
            test=("2020-10-15", dates[12]),
        ).split(20, dates)


def test_rolling_origin_generates_only_complete_expanding_windows():
    splits = RollingOrigin(initial_window=10, horizon=3, step=4).split(20, None)

    assert [split.train_indices.tolist() for split in splits] == [
        list(range(10)),
        list(range(14)),
    ]
    assert [split.target_indices.tolist() for split in splits] == [
        [10, 11, 12],
        [14, 15, 16],
    ]
    assert [split.split for split in splits] == [0, 1]
    assert all(split.window == "expanding" for split in splits)


def test_rolling_origin_keeps_a_fixed_training_window():
    splits = RollingOrigin(
        initial_window=12,
        horizon=2,
        step=4,
        window="rolling",
        window_size=10,
    ).split(24, None)

    assert [split.train_indices.tolist() for split in splits] == [
        list(range(2, 12)),
        list(range(6, 16)),
        list(range(10, 20)),
    ]


def test_rolling_origin_gap_follows_the_initial_training_sample():
    first = RollingOrigin(
        initial_window=10,
        horizon=2,
        gap=2,
    ).split(20, None)[0]

    assert first.train_indices.tolist() == list(range(10))
    assert first.target_indices.tolist() == [12, 13]
    assert first.gap == 2


def test_rolling_origin_labels_splits_with_dates():
    dates = pd.date_range("2021-01-01", periods=18, freq="MS")
    split = RollingOrigin(initial_window=10, horizon=2).split(18, dates)[0]

    assert split.train_start == dates[0]
    assert split.train_end == dates[9]
    assert split.forecast_start == dates[10]
    assert split.forecast_end == dates[11]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"initial_window": True}, TypeError, "initial_window"),
        ({"initial_window": 9}, ValueError, "initial_window"),
        ({"initial_window": 10, "horizon": 0}, ValueError, "horizon"),
        ({"initial_window": 10, "step": False}, TypeError, "step"),
        ({"initial_window": 10, "gap": -1}, ValueError, "gap"),
        ({"initial_window": 10, "window": "random"}, ValueError, "window"),
        (
            {"initial_window": 10, "window": "expanding", "window_size": 10},
            ValueError,
            "window_size",
        ),
        (
            {"initial_window": 10, "window": "rolling", "window_size": 11},
            ValueError,
            "window_size",
        ),
    ],
)
def test_rolling_origin_rejects_invalid_configuration(kwargs, error, match):
    with pytest.raises(error, match=match):
        RollingOrigin(**kwargs).split(20, None)


def test_rolling_origin_rejects_a_sample_without_one_complete_forecast():
    with pytest.raises(ValueError, match="complete forecast"):
        RollingOrigin(initial_window=10, horizon=3, gap=2).split(14, None)


def test_split_indices_are_read_only():
    split = RollingOrigin(initial_window=10).split(12, None)[0]

    with pytest.raises(ValueError, match="read-only"):
        split.train_indices[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        split.target_indices[0] = 99


def test_split_rejects_a_mismatched_calendar_length():
    dates = pd.date_range("2021-01-01", periods=19, freq="MS")

    with pytest.raises(ValueError, match="one date per observation"):
        RollingOrigin(initial_window=10).split(20, dates)


def test_scheme_accepts_numpy_integer_arguments():
    splits = RollingOrigin(
        initial_window=np.int64(10),
        horizon=np.int64(2),
    ).split(np.int64(14), None)

    assert len(splits) == 3
