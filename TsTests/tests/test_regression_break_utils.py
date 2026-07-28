"""Tests for the shared regression-stability design builder."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsTests._regression_break_utils import _prepare_regression_break_design


def test_builds_array_design_with_constant_trend_and_two_regressors():
    y = np.arange(12, dtype=float) ** 2
    exog = np.column_stack([np.sin(np.arange(12)), np.cos(np.arange(12))])
    design = _prepare_regression_break_design(y, exog=exog, trend="ct")

    assert design.exog.shape == (12, 4)
    assert design.column_names == ("const", "trend", "x1", "x2")
    np.testing.assert_array_equal(design.exog[:, 0], 1.0)
    np.testing.assert_array_equal(design.exog[:, 1], np.arange(12))


def test_dataframe_columns_and_labels_are_preserved():
    frame = pd.DataFrame(
        {
            "year": np.arange(2000, 2012),
            "y": np.arange(12, dtype=float) ** 2,
            "policy": np.sin(np.arange(12)),
        }
    )
    design = _prepare_regression_break_design(
        frame,
        y_col="y",
        time_col="year",
        exog_cols=["policy"],
    )

    assert design.column_names == ("const", "policy")
    np.testing.assert_array_equal(design.endog, frame["y"])
    np.testing.assert_array_equal(design.time_index, frame["year"])


def test_series_exog_name_is_used():
    y = np.arange(10, dtype=float) ** 2
    exog = pd.Series(np.sin(np.arange(10)), name="cycle")
    design = _prepare_regression_break_design(y, exog=exog)
    assert design.column_names == ("const", "cycle")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"trend": "invalid"}, "trend"),
        ({"trend": "n"}, "requires"),
        ({"exog": np.ones(9)}, "length"),
        ({"exog": np.ones((10, 1, 1))}, "1-D or 2-D"),
    ],
)
def test_rejects_invalid_shapes_and_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _prepare_regression_break_design(np.arange(10, dtype=float), **kwargs)


def test_rejects_conflicting_and_non_dataframe_exog_columns():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="cannot be used together"):
        _prepare_regression_break_design(y, exog=y, exog_cols=["x"])
    with pytest.raises(ValueError, match="DataFrame"):
        _prepare_regression_break_design(y, exog_cols=["x"])


def test_rejects_ambiguous_duplicate_dataframe_columns():
    duplicate_data = pd.DataFrame(
        np.column_stack([np.arange(10), np.arange(10) ** 2]),
        columns=["value", "value"],
    )
    with pytest.raises(ValueError, match="data DataFrame columns must be unique"):
        _prepare_regression_break_design(duplicate_data, y_col="value")

    duplicate_exog = pd.DataFrame(
        np.column_stack([np.arange(10), np.arange(10) ** 2]),
        columns=["x", "x"],
    )
    with pytest.raises(ValueError, match="exog DataFrame columns must be unique"):
        _prepare_regression_break_design(
            np.arange(10, dtype=float) ** 3,
            exog=duplicate_exog,
        )


@pytest.mark.parametrize(
    "exog",
    [
        np.column_stack([np.ones(10), np.arange(10)]),
        np.column_stack([np.arange(10), 2 * np.arange(10)]),
    ],
)
def test_rejects_duplicate_constant_and_rank_deficiency(exog):
    with pytest.raises(ValueError, match="full column rank"):
        _prepare_regression_break_design(np.arange(10, dtype=float) ** 2, exog=exog)


def test_rejects_nonfinite_rows_without_changing_break_alignment():
    y = np.arange(10, dtype=float)
    y[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _prepare_regression_break_design(y)

    exog = np.arange(10, dtype=float)
    exog[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _prepare_regression_break_design(np.arange(10, dtype=float), exog=exog)


def test_time_labels_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        _prepare_regression_break_design(
            np.arange(10, dtype=float),
            time_index=[0, 1, 2, 3, 4, 5, 5, 7, 8, 9],
        )


def test_positional_trend_does_not_depend_on_time_labels():
    y = np.arange(10, dtype=float) ** 2
    regular = _prepare_regression_break_design(y, trend="ct")
    irregular = _prepare_regression_break_design(
        y,
        trend="ct",
        time_index=np.cumsum(np.arange(1, 11)),
    )
    np.testing.assert_array_equal(regular.exog, irregular.exog)
