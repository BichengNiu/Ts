"""Missing-data regression tests for migrated STL."""

import numpy as np
import pytest

from Ts.TsUtils import STL


def _seasonal_data():
    time = np.arange(120, dtype=float)
    seasonal = 2.0 * np.sin(2.0 * np.pi * time / 12.0)
    return 10.0 + 0.05 * time + seasonal


def test_stl_drop_removes_non_finite_data_and_records_positions():
    """Explicit drop preserves the migrated STL missing-data contract.

    covers: TsUtils/_stl.py::STL.__init__ [function]
    """
    data = _seasonal_data()
    data[5] = np.nan
    data[20] = np.inf

    model = STL(data, period=12, missing="drop")

    assert model.missing == "drop"
    assert model.dropped_positions == (5, 20)
    assert model.data.shape == (data.size - 2,)
    assert np.all(np.isfinite(model.data))


def test_stl_rejects_unknown_missing_policy():
    """STL accepts only the explicit raise and drop policies."""
    with pytest.raises(ValueError, match="missing must be"):
        STL(_seasonal_data(), period=12, missing="omit")
