"""Layout measurements must work for figures created outside pyplot."""

from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
import pytest

from Ts.TsPlots import plot_series
from Ts.TsPlots.style import place_left_title_right_of_ylabel


@pytest.mark.parametrize("concrete_canvas", [False, True])
def test_series_layout_and_export_with_caller_owned_figure(concrete_canvas):
    figure = Figure()
    canvas = FigureCanvasAgg(figure) if concrete_canvas else figure.canvas
    axis = figure.add_subplot()
    values = pd.DataFrame(
        {"passengers": [100, 120, 140], "cargo": [10, 12, 14]},
        index=pd.date_range("2026-07-01", periods=3, freq="MS"),
    )

    result, _ = plot_series(
        values, ax=axis, facet=False, note="Source: public data"
    )

    assert result is figure
    if concrete_canvas:
        assert figure.canvas is canvas
    assert axis.get_legend() is not None
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_left_title_measurement_with_base_canvas():
    figure = Figure()
    axis = figure.add_subplot()
    axis.set_title("Transport", loc="left")
    axis.set_ylabel("Passengers")

    place_left_title_right_of_ylabel(axis)

    renderer = figure.canvas.get_renderer()
    title_extent = axis._left_title.get_window_extent(renderer)
    label_extent = axis.yaxis.label.get_window_extent(renderer)
    assert title_extent.x0 > label_extent.x1
