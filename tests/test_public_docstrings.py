"""Repository-wide contract for interactive public API help."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re

import pytest

import Ts
from Ts import TsMetrics, TsModels, TsPlots, TsSims, TsTests, TsUtils
from Ts.TsPlots.style import apply_fonts, style_axes


PUBLIC_MODULES = (TsPlots, TsUtils, TsSims, TsModels, TsMetrics, TsTests)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_MODULES = {
    "TsPlots": TsPlots,
    "TsUtils": TsUtils,
    "TsSims": TsSims,
    "TsModels": TsModels,
    "TsMetrics": TsMetrics,
    "TsTests": TsTests,
}


def _readme_text(package_name: str) -> str:
    """Return one package README as UTF-8 text."""
    return (REPOSITORY_ROOT / package_name / "README.md").read_text(
        encoding="utf-8"
    )


def _python_blocks(markdown: str) -> list[str]:
    """Extract fenced blocks explicitly labelled as executable Python."""
    return re.findall(r"```python\s*\n(.*?)```", markdown, flags=re.DOTALL)


@pytest.mark.parametrize(("package_name", "module"), README_MODULES.items())
def test_readme_mentions_every_public_export(package_name, module):
    """Every exported public name is discoverable in its package README."""
    readme = _readme_text(package_name)
    missing = [
        name
        for name in module.__all__
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            readme,
        )
        is None
    ]
    assert not missing, f"{package_name}/README.md omits public API: {missing}"


@pytest.mark.parametrize(("package_name", "module"), README_MODULES.items())
def test_readme_python_blocks_are_syntactically_valid(package_name, module):
    """Blocks labelled as Python are executable syntax, not signatures."""
    del module
    for number, block in enumerate(_python_blocks(_readme_text(package_name)), 1):
        try:
            ast.parse(block)
        except SyntaxError as error:
            pytest.fail(
                f"{package_name}/README.md Python block {number} is invalid: "
                f"line {error.lineno}: {error.msg}"
            )


@pytest.mark.parametrize(("package_name", "module"), README_MODULES.items())
def test_readme_public_imports_resolve(package_name, module):
    """Every package import shown in executable README blocks still exists."""
    stale = []
    for number, block in enumerate(_python_blocks(_readme_text(package_name)), 1):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != f"Ts.{package_name}":
                continue
            stale.extend(
                (number, alias.name)
                for alias in node.names
                if alias.name != "*" and not hasattr(module, alias.name)
            )
    assert not stale, f"{package_name}/README.md has stale imports: {stale}"


def test_top_level_exports_resolve():
    """The curated top-level namespace never advertises missing objects."""
    missing = [name for name in Ts.__all__ if not hasattr(Ts, name)]
    assert not missing, f"Ts.__all__ contains missing objects: {missing}"


def _public_objects():
    objects = {}
    for module in PUBLIC_MODULES:
        for name in module.__all__:
            if name in objects:
                raise RuntimeError(
                    f"public API name collision: {name} is exported by "
                    "more than one subpackage"
                )
            objects[name] = getattr(module, name)
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


# Only inherited demo-facing methods need manual registration: methods defined
# directly on a public class are collected automatically by _public_methods().
DEMO_METHODS = {
    TsSims.SimSARIMAResult: ("plot",),
    TsSims.SimTSDSResult: ("plot",),
    TsModels.GARCH: ("fit",),
    TsModels.VARResult: ("plot_fit", "plot_diagnostics", "test_residuals"),
    TsModels.VECMResult: ("plot_diagnostics", "test_residuals"),
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
