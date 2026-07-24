"""Tests for Ts.TsModels._compare — compare_models function."""

import matplotlib
matplotlib.use("Agg")

import pytest
from Ts.TsSims import simulate_garch


@pytest.fixture
def garch11_result():
    """Fit GARCH(1,1) and return result."""
    from Ts.TsModels._garch import GARCH

    r = simulate_garch(
        n=300, p=1, q=1, omega=0.1, alpha=[0.2], beta=[0.7],
        seed=42, burn=200,
    )
    model = GARCH(r.data, p=1, q=1)
    return model.fit()


@pytest.fixture
def arch2_result():
    """Fit ARCH(2) and return result."""
    from Ts.TsModels._garch import GARCH

    r = simulate_garch(
        n=200, p=2, q=0, omega=0.4, alpha=[0.3, 0.2],
        seed=42, burn=200,
    )
    model = GARCH(r.data, p=2, q=0)
    return model.fit()


class TestCompareModels:
    """Test compare_models function."""

    def test_compare_two_models(self, garch11_result, arch2_result):
        """compare_models with two models outputs Stata-style table."""
        from Ts.TsModels._compare import compare_models

        models = {
            "GARCH(1,1)": garch11_result,
            "ARCH(2)": arch2_result,
        }
        table = compare_models(models)
        assert isinstance(table, str)
        assert "GARCH(1,1)" in table
        assert "ARCH(2)" in table
        assert "(1)" in table
        assert "(2)" in table
        assert "t statistics in parentheses" in table
        assert "* p<0.10, ** p<0.05, *** p<0.01" in table
        assert "ARCH" in table

    def test_compare_single_model(self, garch11_result):
        """compare_models with single model works."""
        from Ts.TsModels._compare import compare_models

        models = {"GARCH(1,1)": garch11_result}
        table = compare_models(models)
        assert "GARCH(1,1)" in table
        assert "N" in table

    def test_compare_empty_dict_raises(self):
        """compare_models with empty dict raises ValueError."""
        from Ts.TsModels._compare import compare_models

        with pytest.raises(ValueError):
            compare_models({})

    def test_compare_invalid_input_raises(self):
        """compare_models with non-dict input raises TypeError."""
        from Ts.TsModels._compare import compare_models

        with pytest.raises(TypeError):
            compare_models([1, 2, 3])

    def test_compare_stata_format_groups(self, garch11_result, arch2_result):
        """compare_models groups parameters: main, ARCHM, ARCH."""
        from Ts.TsModels._compare import compare_models

        models = {
            "retDJIA": garch11_result,
            "retSP": arch2_result,
        }
        table = compare_models(models)
        assert "main" in table
        assert "ARCH" in table
        assert "L.arch" in table or "ARCH" in table
