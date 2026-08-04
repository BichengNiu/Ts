"""Tests for lag-indexed impulse-response bar charts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Ts.TsPlots import plot_lag_response


def test_series_draws_one_bar_per_lag_and_zero_reference():
    weights = pd.Series(
        [1.0, 0.5, -0.25],
        index=pd.RangeIndex(3, name="lag"),
        name="price",
    )

    fig, ax = plot_lag_response(weights)

    assert [bar.get_height() for bar in ax.patches] == [1.0, 0.5, -0.25]
    assert ax.get_xlabel() == "Time lag"
    assert ax.get_ylabel() == "Impulse response"
    assert ax.get_title() == "price"
    assert any(np.allclose(line.get_ydata(), [0.0, 0.0]) for line in ax.lines)
    plt.close(fig)


def test_dataframe_creates_one_facet_per_input_in_column_order():
    weights = pd.DataFrame(
        {"price": [1.0, 0.5], "income": [-0.4, -0.2]},
        index=pd.RangeIndex(2, name="lag"),
    )

    fig, axes = plot_lag_response(weights, title="RDL impulse responses")

    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == ["price", "income"]
    assert fig._suptitle.get_text() == "RDL impulse responses"
    assert [bar.get_height() for bar in axes[1].patches] == [-0.4, -0.2]
    plt.close(fig)


def test_external_axis_is_reused_for_one_response():
    fig, supplied = plt.subplots()
    returned_fig, returned_ax = plot_lag_response([1.0, 0.25], ax=supplied)

    assert returned_fig is fig
    assert returned_ax is supplied
    plt.close(fig)


def test_custom_labels_note_color_and_grid_follow_contract():
    fig, ax = plot_lag_response(
        [1.0, 0.5],
        title="Dynamic multiplier",
        xtitle="Quarter",
        ytitle="Effect",
        color="#ff0000",
        note="Point estimates",
        grid=True,
    )

    assert ax.get_title() == "Dynamic multiplier"
    assert ax.get_xlabel() == "Quarter"
    assert ax.get_ylabel() == "Effect"
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))
    assert any(text.get_text() == "Point estimates" for text in fig.texts)
    assert any(line.get_visible() for line in ax.get_ygridlines())
    plt.close(fig)


def test_rgb_tuple_is_accepted_as_one_matplotlib_color():
    fig, ax = plot_lag_response([1.0, 0.5], color=(1.0, 0.0, 0.0))

    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))
    plt.close(fig)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([], "non-empty"),
        ([1.0, np.nan], "finite"),
        (pd.Series([1.0, 0.5], index=[0, 0]), "unique"),
        (pd.Series([1.0, 0.5], index=[0, -1]), "non-negative"),
        (pd.Series([1.0, 0.5], index=[0.0, 1.5]), "integer"),
    ],
)
def test_invalid_lag_responses_are_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        plot_lag_response(data)


def test_multi_response_rejects_single_external_axis():
    frame = pd.DataFrame({"x1": [1.0, 0.5], "x2": [0.2, 0.1]})
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="multiple"):
        plot_lag_response(frame, ax=ax)
    plt.close(fig)
