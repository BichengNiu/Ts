"""Repository-wide contract for interactive public API help."""

from __future__ import annotations

import inspect

import pytest

from Ts import TsMetrics, TsModels, TsPlots, TsSims, TsTests, TsUtils
from Ts.TsPlots.style import apply_fonts, style_axes


PUBLIC_MODULES = (TsPlots, TsUtils, TsSims, TsModels, TsMetrics, TsTests)


def _public_objects():
    objects = {}
    for module in PUBLIC_MODULES:
        for name in module.__all__:
            objects.setdefault(name, getattr(module, name))
    return objects


PUBLIC_OBJECTS = _public_objects()
PUBLIC_OBJECTS.update({"apply_fonts": apply_fonts, "style_axes": style_axes})


def _section(doc: str, name: str) -> list[str]:
    """Return the body of one NumPy-style docstring section."""
    lines = doc.splitlines()
    for index in range(len(lines) - 1):
        if lines[index].strip() != name:
            continue
        if set(lines[index + 1].strip()) != {"-"}:
            continue
        body = []
        for position, line in enumerate(lines[index + 2 :], start=index + 2):
            if (
                position + 1 < len(lines)
                and line
                and not line.startswith(" ")
                and set(lines[position + 1].strip()) == {"-"}
            ):
                break
            body.append(line)
        return body
    return []


def _documented_names(doc: str) -> set[str]:
    names = set()
    for section_name in ("Parameters", "Attributes"):
        for line in _section(doc, section_name):
            if not line or line.startswith(" ") or ":" not in line:
                continue
            declaration = line.split(":", 1)[0]
            names.update(name.strip().lstrip("*") for name in declaration.split(","))
    return names


def _signature_parameters(obj) -> set[str]:
    try:
        parameters = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return set()
    return {
        parameter.name
        for parameter in parameters
        if parameter.name not in {"self", "cls"}
        and not parameter.name.startswith("_")
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }


@pytest.mark.parametrize("name", sorted(PUBLIC_OBJECTS))
def test_public_help_is_complete(name):
    obj = PUBLIC_OBJECTS[name]
    doc = inspect.getdoc(obj) or ""
    assert doc, f"{name} has no docstring"
    assert _section(doc, "Examples"), f"{name} has no Examples section"

    expected = _signature_parameters(obj)
    documented = _documented_names(doc)
    assert expected <= documented, (
        f"{name} does not document signature parameters: "
        f"{sorted(expected - documented)}"
    )

    if inspect.isfunction(obj):
        assert _section(doc, "Returns"), f"{name} has no Returns section"


DEMO_METHODS = {
    TsUtils.STL: ("fit", "summary"),
    TsUtils.STLResult: ("summary", "plot"),
    TsUtils.TimeSeriesSummary: ("summary", "plot"),
    TsUtils.EACFResult: ("summary",),
    TsUtils.InterpolationResult: ("summary",),
    TsTests.BaseTest: ("summary",),
    TsTests.ADFTest: ("fit",),
    TsTests.ADFTestResult: ("plot_test",),
    TsTests.PhillipsPerronTest: ("fit",),
    TsTests.PhillipsPerronTestResult: ("plot_test",),
    TsTests.KPSSTest: ("fit",),
    TsTests.KPSSTestResult: ("plot_test",),
    TsTests.PerronTest: ("fit",),
    TsTests.PerronTestResult: ("plot_test",),
    TsTests.ZivotAndrewsTest: ("fit",),
    TsTests.ZivotAndrewsTestResult: ("plot_test",),
    TsTests.LjungBoxTest: ("fit",),
    TsTests.EngleLMTest: ("fit",),
    TsTests.NormalityTest: ("fit",),
    TsTests.NormalityTestResult: ("plot_test",),
    TsTests.JohansenTest: ("fit",),
    TsTests.TodaYamamotoTest: ("fit",),
    TsSims.BaseSimResult: ("get_data", "get_params", "summary", "plot"),
    TsSims.SimSARIMAResult: ("summary", "plot"),
    TsSims.SimGARCHResult: ("summary", "plot", "to_dataframe"),
    TsSims.SimRDLResult: ("get_exog", "get_components", "summary"),
    TsSims.SimCointegratedResult: ("summary", "plot"),
    TsSims.SimTSDSResult: ("summary", "plot"),
    TsModels.BaseModel: ("summary", "oos", "backtest", "backcast"),
    TsModels.BaseModelResult: (
        "summary",
        "plot_fit",
        "plot_diagnostics",
        "test_residuals",
    ),
    TsModels.SARIMAX: ("fit",),
    TsModels.SARIMAXResult: (
        "summary",
        "predict",
        "plot_roots",
        "policy_effect",
        "weights",
        "cycle_period",
        "long_run_equilibrium",
    ),
    TsModels.GARCH: ("fit",),
    TsModels.GARCHResult: ("summary", "predict", "test_persistence"),
    TsModels.AutoModelResult: ("summary",),
    TsModels.IRFResult: ("summary", "get"),
    TsModels.FEVDResult: ("summary", "get"),
    TsModels.VAROrderResult: ("summary",),
    TsModels.VARResult: (
        "summary",
        "plot_fit",
        "plot_diagnostics",
        "test_residuals",
        "irf",
        "fevd",
        "granger_causality",
        "plot_irf",
        "plot_roots",
        "predict",
    ),
    TsModels.VAR: ("select_order", "fit"),
    TsModels.VECM: ("fit",),
    TsModels.VECMOrderResult: ("summary",),
    TsModels.VECMResult: (
        "summary",
        "irf",
        "fevd",
        "granger_causality",
        "plot_diagnostics",
        "test_residuals",
        "predict",
        "plot_roots",
    ),
    TsModels.SVAR: ("fit",),
    TsModels.SVARResult: ("summary", "irf", "plot_irf"),
    TsModels.AutoSARIMAX: ("fit",),
    TsModels.AutoGARCH: ("fit",),
}


def _public_methods():
    methods = {}
    for owner in PUBLIC_OBJECTS.values():
        if not inspect.isclass(owner):
            continue
        for method_name, raw in owner.__dict__.items():
            if method_name.startswith("_") or isinstance(raw, property):
                continue
            method = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
            if callable(method):
                methods[(owner, method_name)] = getattr(owner, method_name)

    # Include demo-facing methods inherited from private implementation bases,
    # such as GARCH.fit, in the public help contract.
    for owner, method_names in DEMO_METHODS.items():
        for method_name in method_names:
            methods[(owner, method_name)] = getattr(owner, method_name)
    return methods


PUBLIC_METHODS = _public_methods()


@pytest.mark.parametrize(
    ("owner", "method_name"),
    sorted(PUBLIC_METHODS, key=lambda item: (item[0].__name__, item[1])),
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_public_method_help_is_complete(owner, method_name):
    method = PUBLIC_METHODS[(owner, method_name)]
    doc = inspect.getdoc(method) or ""
    assert doc, f"{owner.__name__}.{method_name} has no docstring"
    assert _section(doc, "Examples"), (
        f"{owner.__name__}.{method_name} has no Examples section"
    )
    expected = _signature_parameters(method)
    documented = _documented_names(doc)
    assert expected <= documented, (
        f"{owner.__name__}.{method_name} does not document signature parameters: "
        f"{sorted(expected - documented)}"
    )
    assert _section(doc, "Returns"), (
        f"{owner.__name__}.{method_name} has no Returns section"
    )
