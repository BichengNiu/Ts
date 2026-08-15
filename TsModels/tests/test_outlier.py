"""Tests for ARIMA-residual AO/LS/IO outlier detection."""

import numpy as np
import pandas as pd
import pytest

from Ts.TsSims import simulate_sarima
from Ts.TsModels._outlier import (
    OutlierDetector,
    OutlierDetectorResult,
    _ar_ma_coefficients,
    _c_weights,
    _full_scan,
    _outlier_footprint,
    _pi_weights,
    _scan_outlier_type,
)


def _ar1_series(n=200, phi=0.7, sigma2=1.0, seed=42):
    return simulate_sarima(
        n=n,
        order=(1, 0, 0),
        ar=[phi],
        sigma2=sigma2,
        seed=seed,
    ).data.copy()


# ---------------------------------------------------------------------------
# Whitening weights
# ---------------------------------------------------------------------------


def test_pi_weights_ar1():
    pi = _pi_weights(np.array([1.0, 0.7]), np.array([1.0]), 6)
    np.testing.assert_allclose(pi, [1.0, 0.7, 0.0, 0.0, 0.0, 0.0])


def test_pi_weights_ma1_alternating_geometric():
    pi = _pi_weights(np.array([1.0]), np.array([1.0, 0.5]), 6)
    np.testing.assert_allclose(pi, [1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125])


def test_pi_weights_arma11():
    pi = _pi_weights(np.array([1.0, 0.7]), np.array([1.0, 0.4]), 6)
    expected = [1.0, 1.1, -0.44, 0.176, -0.0704, 0.02816]
    np.testing.assert_allclose(pi, expected)


def test_c_weights_ar1():
    pi = np.array([1.0, 0.7, 0.0, 0.0, 0.0])
    c = _c_weights(pi)
    np.testing.assert_allclose(c[1:], [-0.3, -0.3, -0.3, -0.3])


def test_pi_weights_include_differencing():
    ar, ma = _ar_ma_coefficients({"ar.L1": 0.7}, (1, 1, 0), (0, 0, 0, 0))
    pi = _pi_weights(ar, ma, 6)
    np.testing.assert_allclose(pi, [1.0, -0.3, -0.7, 0.0, 0.0, 0.0])


def test_ar_ma_coefficients_expand_seasonal_terms():
    params = {"ar.L1": 0.5, "ar.S.L1": 0.3, "ma.L1": 0.2}
    ar, ma = _ar_ma_coefficients(params, (1, 0, 1), (1, 0, 0, 4))
    np.testing.assert_allclose(ar, [1.0, 0.5, 0.0, 0.0, 0.3, 0.15])
    np.testing.assert_allclose(ma, [1.0, 0.2])


# ---------------------------------------------------------------------------
# Pure-outlier footprints (theory values for AR(1) with phi = 0.7)
# ---------------------------------------------------------------------------


def test_pure_outlier_footprints_ar1_omega6():
    n = 10
    pi = _pi_weights(np.array([1.0, 0.7]), np.array([1.0]), n)
    c = _c_weights(pi)
    omega = 6.0

    ao = _outlier_footprint(omega, pi, 0, n)
    np.testing.assert_allclose(ao[:3], [6.0, -4.2, 0.0])
    np.testing.assert_allclose(ao[3:], 0.0)

    ls = _outlier_footprint(omega, c, 0, n)
    np.testing.assert_allclose(ls, [6.0] + [1.8] * (n - 1))

    io = _outlier_footprint(omega, np.zeros(n), 0, n)
    np.testing.assert_allclose(io, [6.0] + [0.0] * (n - 1))


def test_outlier_footprint_places_weight_at_candidate_time():
    weights = np.array([1.0, 0.7, 0.0, 0.0])
    w = _outlier_footprint(1.0, weights, 2, 6)
    np.testing.assert_allclose(w, [0.0, 0.0, 1.0, -0.7, 0.0, 0.0])


def test_outlier_footprint_truncates_at_sample_end():
    weights = np.array([1.0, 0.7, 0.0, 0.0])
    w = _outlier_footprint(1.0, weights, 5, 6)
    np.testing.assert_allclose(w, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Estimation consistency
# ---------------------------------------------------------------------------


def test_omega_matches_explicit_ols_regression():
    rng = np.random.default_rng(7)
    n = 60
    e = rng.normal(size=n)
    pi = _pi_weights(np.array([1.0, 0.7]), np.array([1.0]), n)
    position = 15
    e = e + 6.0 * _outlier_footprint(1.0, pi, position, n)

    omega, lstat, standard_error = _scan_outlier_type(e, pi, 1.0)
    w = _outlier_footprint(1.0, pi, position, n)
    expected = np.dot(w, e) / np.dot(w, w)
    assert omega[position] == pytest.approx(expected)
    assert standard_error[position] == pytest.approx(1.0 / np.sqrt(np.dot(w, w)))
    assert lstat[position] == pytest.approx(expected * np.sqrt(np.dot(w, w)))


def test_ls_estimate_is_invariant_to_omega_scale():
    rng = np.random.default_rng(11)
    n = 80
    base = rng.normal(size=n)
    pi = _pi_weights(np.array([1.0, 0.7]), np.array([1.0]), n)
    c = _c_weights(pi)
    position = 30
    w = _outlier_footprint(1.0, c, position, n)

    estimates = []
    for scale in (1.0, 6.0):
        e = base + scale * w
        omega, _, _ = _scan_outlier_type(e, c, 1.0)
        estimates.append(omega[position])
    assert estimates[1] - estimates[0] == pytest.approx(5.0)


def test_full_scan_io_equals_raw_residual_statistics():
    e = np.array([1.0, -2.0, 3.0])
    pi = _pi_weights(np.array([1.0]), np.array([1.0]), 3)
    c = _c_weights(pi)
    lstat, omega, se = _full_scan(e, pi, c, 2.0)
    np.testing.assert_allclose(omega[2], e)
    np.testing.assert_allclose(lstat[2], e / 2.0)
    np.testing.assert_allclose(se[2], np.full(3, 2.0))


# ---------------------------------------------------------------------------
# End-to-end detection
# ---------------------------------------------------------------------------


def test_detects_single_ao():
    data = _ar1_series(n=200)
    data[100] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    assert isinstance(result, OutlierDetectorResult)
    assert len(result.events) == 1
    row = result.events.iloc[0]
    assert row["type"] == "AO"
    assert row["time"] == 100
    assert abs(row["omega"] - 6.0) < 2.0
    assert row["standard_error"] > 0.0


def test_detects_single_ls():
    data = _ar1_series(n=200)
    data[100:] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    assert len(result.events) == 1
    row = result.events.iloc[0]
    assert row["type"] == "LS"
    assert row["time"] == 100


def test_detects_single_io():
    phi = 0.7
    data = _ar1_series(n=200)
    impulse = 6.0 * np.array([phi**j for j in range(len(data) - 100)])
    data[100:] += impulse
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    assert len(result.events) == 1
    row = result.events.iloc[0]
    assert row["type"] == "IO"
    assert row["time"] == 100
    assert abs(row["omega"] - 6.0) < 2.0


def test_clean_series_detects_nothing():
    data = _ar1_series(n=120, seed=7)
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
    assert result.events.empty
    assert result.statistic < 3.5


def test_iterative_detection_of_ao_and_ls():
    data = _ar1_series(n=300)
    data[100] += 6.0
    data[200:] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    types = result.events["type"].tolist()
    times = result.events["time"].tolist()
    assert "AO" in types
    assert "LS" in types
    assert times[types.index("AO")] == 100
    assert times[types.index("LS")] == 200


def test_series_input_uses_index_labels():
    dates = pd.date_range("2020-01-01", periods=200, freq="MS")
    data = pd.Series(_ar1_series(n=200), index=dates)
    data.iloc[100] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    assert result.events.iloc[0]["time"] == dates[100]
    assert result.l_statistics.index[0] == dates[0]


def test_max_events_limits_detection():
    data = _ar1_series(n=200)
    data[100] += 6.0
    data[120] += 6.0
    result = OutlierDetector(order=(1, 0, 0), max_events=1).fit_detect(data)
    assert len(result.events) == 1
    assert len(result.scan_history) == 1


def test_result_contains_full_scan_outputs():
    data = _ar1_series(n=150)
    data[75] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    assert result.pi_weights.shape == (150,)
    assert result.c_weights.shape == (150,)
    assert result.residuals.shape == (150,)
    assert result.adjusted_residuals.shape == (150,)
    assert result.l_statistics.shape == (150, 3)
    assert list(result.l_statistics.columns) == ["AO", "LS", "IO"]
    assert result.sigma > 0.0
    assert result.critical_value == 3.5


def test_summary_reports_detected_events():
    data = _ar1_series(n=150)
    data[75] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)

    text = result.summary()
    assert "ARIMA Outlier Detection" in text
    assert "Detected Events" in text
    assert "AO" in text
    assert text == str(result)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_errors():
    with pytest.raises(TypeError, match="critical_value"):
        OutlierDetector(critical_value="3")
    with pytest.raises(ValueError, match="critical_value"):
        OutlierDetector(critical_value=0.0)
    with pytest.raises(TypeError, match="max_events"):
        OutlierDetector(max_events=1.5)
    with pytest.raises(ValueError, match="max_events"):
        OutlierDetector(max_events=0)


def test_fit_detect_rejects_invalid_series():
    with pytest.raises(ValueError, match="at least 10"):
        OutlierDetector().fit_detect(np.arange(5.0))
    with pytest.raises(ValueError, match=r"must be 1-D"):
        OutlierDetector().fit_detect(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="finite"):
        OutlierDetector().fit_detect(np.r_[np.arange(10.0), np.nan])


def test_plot_marks_events_and_l_statistics():
    import matplotlib

    matplotlib.use("Agg")
    data = _ar1_series(n=200)
    data[100:] += 6.0
    result = OutlierDetector(order=(1, 0, 0)).fit_detect(data)
    _fig, axes = result.plot()
    assert len(axes) == 2
    assert len(axes[0].collections) == len(result.events)
    assert len(axes[1].lines) >= 3
    assert any(line.get_linestyle() == "--" for line in axes[1].lines)
