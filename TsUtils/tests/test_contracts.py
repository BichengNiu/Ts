"""Regression tests for utility result invariants."""

import numpy as np
import pytest

from Ts.TsUtils import InterpolationResult, STL, STLResult


def test_interpolation_result_derives_remaining_mask():
    """Remaining missing values cannot drift from missing and filled masks."""
    result = InterpolationResult(
        data=np.array([1.0, 2.0, np.nan]),
        missing_mask=np.array([False, True, True]),
        filled_mask=np.array([False, True, False]),
        method="linear",
        max_gap=None,
        edge="keep",
    )

    assert result.remaining_mask.tolist() == [False, False, True]
    assert result.n_remaining == 1


def test_interpolation_result_rejects_contradictory_masks():
    """Filled positions must be originally missing and contain values."""
    with pytest.raises(ValueError, match="subset"):
        InterpolationResult(
            data=np.array([1.0]),
            missing_mask=np.array([False]),
            filled_mask=np.array([True]),
            method="linear",
            max_gap=None,
            edge="keep",
        )

    with pytest.raises(ValueError, match="filled values"):
        InterpolationResult(
            data=np.array([np.nan]),
            missing_mask=np.array([True]),
            filled_mask=np.array([True]),
            method="linear",
            max_gap=None,
            edge="keep",
        )


def test_interpolation_result_validates_recorded_options():
    """Audit metadata follows the same contract as interpolation inputs."""
    with pytest.raises(ValueError, match="method"):
        InterpolationResult(
            data=np.array([1.0]),
            missing_mask=np.array([False]),
            filled_mask=np.array([False]),
            method="invented",
            max_gap=None,
            edge="keep",
        )


def test_stl_result_rejects_incoherent_components():
    """Every decomposition component must align and carry valid config."""
    with pytest.raises(ValueError, match="same length"):
        STLResult(
            observed=np.arange(3.0),
            trend=np.arange(2.0),
            seasonal=np.arange(3.0),
            residuals=np.arange(3.0),
            weights=np.ones(3),
            period=2,
            config={
                "robust": False,
                "seasonal": 7,
                "trend": 9,
                "low_pass": 11,
            },
        )


def test_stl_tuning_arguments_are_keyword_only():
    """Long STL configuration cannot be shifted by positional mistakes."""
    data = np.sin(np.arange(24.0) * 2.0 * np.pi / 12.0)

    with pytest.raises(TypeError):
        STL(data, 12, 7)
