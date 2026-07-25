"""Behavior tests for the shared simulation result container."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Ts.TsSims._base import BaseSimResult


def test_base_sim_result_univariate_defaults():
    result = BaseSimResult(
        data=np.array([1.0, 2.0, 3.0]),
        residuals=np.array([0.1, 0.2, 0.3]),
        params={"nested": {"value": 1}},
    )

    assert isinstance(result.get_data(), pd.Series)
    assert result.summary() == "BaseSimResult(n=3)"

    copied = result.get_params()
    copied["nested"]["value"] = 99
    assert result.params["nested"]["value"] == 1

    fig, ax = result.plot()
    assert ax.get_title() == "BaseSimResult Simulation"
    plt.close(fig)


def test_base_sim_result_multivariate_data_and_custom_plot_title():
    result = BaseSimResult(
        data=np.arange(6, dtype=float).reshape(3, 2),
        residuals=np.zeros((3, 2)),
    )

    frame = result.get_data()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["y0", "y1"]

    fig, ax = result.plot(title="Custom simulation")
    assert ax.get_title() == "Custom simulation"
    plt.close(fig)
