"""Regression tests for the lazy top-level namespace."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_PARENT = Path(__file__).resolve().parents[2]


def _loaded_ts_modules(import_statement: str) -> set[str]:
    script = (
        "import sys; "
        f"{import_statement}; "
        "print('\\n'.join(name for name in sys.modules if name.startswith('Ts.')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_PARENT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in completed.stdout.splitlines() if line}


def test_importing_root_namespace_does_not_load_subpackages():
    assert _loaded_ts_modules("import Ts") == set()


def test_importing_metrics_keeps_model_and_plot_dependencies_local():
    loaded = _loaded_ts_modules("import Ts.TsMetrics")
    assert not any(name.startswith("Ts.TsModels") for name in loaded)
    assert not any(name.startswith("Ts.TsPlots") for name in loaded)
    assert not any(name.startswith("Ts.TsTests") for name in loaded)


def test_importing_simulations_keeps_model_dependency_local():
    loaded = _loaded_ts_modules("import Ts.TsSims")
    assert not any(name.startswith("Ts.TsModels") for name in loaded)
    assert not any(name.startswith("Ts.TsPlots") for name in loaded)
