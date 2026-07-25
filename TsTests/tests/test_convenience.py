"""Convenience tests for TsTests — edge cases and cross-cutting concerns."""

import matplotlib

matplotlib.use("Agg")

import numpy as np


def test_pp_critical_values_keys_have_percent():
    """PhillipsPerronTestResult critical_values keys should contain '%' for plotting."""
    from Ts.TsTests import PhillipsPerronTest

    data = np.random.randn(100).cumsum()  # I(1) series
    pp = PhillipsPerronTest(data)
    pp.fit()
    crit = pp.result_.critical_values
    for k in crit:
        assert "%" in k, f"Critical value key '{k}' missing '%' suffix"
