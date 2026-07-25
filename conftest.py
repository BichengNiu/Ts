"""Shared pytest isolation for the complete Ts package."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close figures left open by a test so later tests remain isolated."""
    yield
    plt.close("all")
