"""Audit tests for the TsPlots default colour template.

The template is the single source of truth for every colour used across the
Ts plotting stack: four main colours (黑 / 深蓝 / 灰 / 深红) lead the default
cycle, and cosmetic colours are named roles.  These tests enforce two
invariants:

1. ``TsPlots/style.py`` is the only place where a raw colour literal (hex or
   named) may appear in non-test source.
2. Downstream packages (`TsModels`, `TsTests`, `TsSims`, ...) must not access
   ``DEFAULT_PALETTE`` by a fixed numeric index — positional colour coupling
   breaks when the template changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from matplotlib.colors import is_color_like

from Ts.TsPlots.style import (
    ANNOTATION_EDGE,
    AXIS_GRAY,
    AXIS_TEXT_GRAY,
    BAND_COLOR,
    BLACK,
    DARK_BLUE,
    DARK_RED,
    DEFAULT_PALETTE,
    EXTENDED_PALETTE,
    GRAY,
    GRID_GRAY,
    INK,
    REFERENCE_LINE_COLOR,
    SHADE_COLOR,
    WHITE,
    ZERO_LINE_COLOR,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_LABELSIZE,
    TITLE_FONTSIZE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Packages owned by the Ts repo whose plotting must honour the template.
_PACKAGES = [
    "TsPlots",
    "TsModels",
    "TsTests",
    "TsSims",
    "TsUtils",
    "TsMetrics",
]

# The one file allowed to define colour values.
_ALLOWED_COLOR_FILES = {"TsPlots" / Path("style.py")}

_ROLE_COLORS = [
    BLACK,
    DARK_BLUE,
    GRAY,
    DARK_RED,
    INK,
    WHITE,
    AXIS_GRAY,
    AXIS_TEXT_GRAY,
    GRID_GRAY,
    ANNOTATION_EDGE,
    REFERENCE_LINE_COLOR,
    SHADE_COLOR,
    BAND_COLOR,
    ZERO_LINE_COLOR,
]

_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,6}\b")
_NAMED_RE = re.compile(
    r"""["'](?:black|white|red|blue|green|gray|grey|purple|orange|navy|maroon|silver)["']""",
    re.IGNORECASE,
)
# Fixed numeric access to the palette (index or literal slice).  The variable
# cycling patterns ``DEFAULT_PALETTE[i % len(...)]`` are deliberately allowed.
_INDEX_RE = re.compile(r"DEFAULT_PALETTE\s*\[\s*\d+(?:\s*:\s*\d+)?\s*\]")


def _non_test_sources():
    for package in _PACKAGES:
        root = REPO_ROOT / package
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT)
            if "tests" in relative.parts:
                continue
            yield path, relative


class TestPaletteTemplate:
    def test_default_cycle_is_led_by_requested_main_colors(self):
        assert DEFAULT_PALETTE[:4] == [BLACK, DARK_BLUE, GRAY, DARK_RED]

    def test_cycle_has_eight_distinct_colors(self):
        assert len(DEFAULT_PALETTE) == 8
        assert len(set(DEFAULT_PALETTE)) == len(DEFAULT_PALETTE)
        assert set(DEFAULT_PALETTE) == {*EXTENDED_PALETTE, BLACK, DARK_BLUE, GRAY, DARK_RED}

    def test_every_role_is_a_valid_matplotlib_color(self):
        for role in _ROLE_COLORS:
            assert is_color_like(role), f"{role!r} is not a valid matplotlib colour"

    def test_reference_and_zero_roles_follow_template(self):
        # Reference/critical lines reuse the dark-red main colour; the zero
        # baseline reuses the text ink colour — no separate hardcoded reds.
        assert REFERENCE_LINE_COLOR == DARK_RED
        assert ZERO_LINE_COLOR == INK


class TestNoHardcodedColors:
    """No raw colour literal may live outside ``TsPlots/style.py``."""

    @pytest.fixture(scope="class")
    def violations(self):
        found = []
        for path, relative in _non_test_sources():
            if relative in _ALLOWED_COLOR_FILES:
                continue
            text = path.read_text(encoding="utf-8")
            if _HEX_RE.search(text):
                found.append(f"{relative}: hex literal")
            if _NAMED_RE.search(text):
                found.append(f"{relative}: named-colour literal")
        return found

    def test_hex_colours_only_in_style_py(self, violations):
        assert not [v for v in violations if "hex" in v], violations

    def test_named_colours_only_in_style_py(self, violations):
        assert not [v for v in violations if "named" in v], violations


class TestNoPositionalPaletteAccess:
    """Downstream packages must not index ``DEFAULT_PALETTE`` by number."""

    @pytest.fixture(scope="class")
    def violations(self):
        found = []
        for path, relative in _non_test_sources():
            if relative.parts[0] == "TsPlots":
                continue  # series/bar cycling inside TsPlots is deliberate
            text = path.read_text(encoding="utf-8")
            if _INDEX_RE.search(text):
                found.append(str(relative))
        return found

    def test_no_numeric_palette_index_outside_tsplots(self, violations):
        assert not violations, f"positional DEFAULT_PALETTE access: {violations}"


class TestTypographyContract:
    """Rendered figures must honour the shared font/size constants."""

    def test_plot_series_title_labels_ticks_legend_use_constants(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from Ts.TsPlots import plot_series

        rng = np.random.default_rng(1)
        data = {"a": rng.normal(size=30), "b": rng.normal(size=30)}
        fig, ax = plot_series(data, facet=False, title="Contract")
        try:
            assert ax.title.get_fontsize() == TITLE_FONTSIZE
            assert ax.xaxis.label.get_fontsize() == AXIS_LABEL_FONTSIZE
            assert ax.yaxis.label.get_fontsize() == AXIS_LABEL_FONTSIZE
            assert {t.get_fontsize() for t in ax.get_xticklabels()} == {TICK_LABELSIZE}
            assert {t.get_fontsize() for t in ax.get_yticklabels()} == {TICK_LABELSIZE}
            assert {t.get_fontsize() for t in ax.get_legend().get_texts()} == {
                LEGEND_FONTSIZE
            }
        finally:
            plt.close(fig)

    def test_plot_acf_title_labels_use_constants(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from Ts.TsPlots import plot_acf

        fig, ax = plot_acf(np.random.default_rng(2).normal(size=50), nlags=10, title="ACF")
        try:
            assert ax.title.get_fontsize() == TITLE_FONTSIZE
            assert ax.xaxis.label.get_fontsize() == AXIS_LABEL_FONTSIZE
            assert ax.yaxis.label.get_fontsize() == AXIS_LABEL_FONTSIZE
        finally:
            plt.close(fig)

    def test_plot_scatter_annotation_size_uses_annotation_constant(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from Ts.TsPlots import plot_scatter
        from Ts.TsPlots.style import ANNOTATION_FONTSIZE

        x = np.arange(5.0)
        fig, ax = plot_scatter(x=x, y=x + 1, show_values=True, title="Scatter")
        try:
            annotations = [t for t in ax.texts if t.get_text()]
            assert annotations
            assert {t.get_fontsize() for t in annotations} == {ANNOTATION_FONTSIZE}
        finally:
            plt.close(fig)

    def test_plot_lag_response_title_uses_constant(self):
        import matplotlib.pyplot as plt
        import pandas as pd

        from Ts.TsPlots import plot_lag_response

        weights = pd.Series([1.0, 0.5], name="price")
        fig, ax = plot_lag_response(weights, title="RDL")
        try:
            assert ax.title.get_fontsize() == TITLE_FONTSIZE
        finally:
            plt.close(fig)
