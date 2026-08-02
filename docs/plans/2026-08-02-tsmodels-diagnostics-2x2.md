# TsModels 2x2 Diagnostic Plots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the shared univariate `plot_diagnostics()` figure to a 2-by-2 layout containing residuals, a residual histogram, residual ACF, and residual PACF in that order.

**Architecture:** Implement the layout once in `BaseModelResult`, which is the actual shared implementation used by SARIMAX, GARCH, and Auto model results. Preserve the existing residual sample, diagnostic-test annotation, ACF/PACF calculations, and flat positional axes access while adding one histogram panel; VAR and VECM retain their separate multivariate diagnostic implementations.

**Tech Stack:** Python 3, NumPy, Matplotlib, TsPlots, TsTests, pytest, Ruff.

---

### Task 1: Lock down the 2x2 public plotting contract with failing tests

**Files:**
- Modify: `TsModels/tests/test_base.py`
- Modify: `TsModels/tests/test_sarimax.py`
- Modify: `TsModels/tests/test_garch.py`
- Modify: `TsModels/tests/test_auto.py`

**Step 1: Change the shared result contract from three to four axes**

Update the base-result test to require a flat four-Axes tuple in row-major order:

```python
fig, axes = result.plot_diagnostics()

assert isinstance(fig, Figure)
assert len(axes) == 4
assert [ax.get_title() for ax in axes] == [
    "Residuals",
    "Residual Histogram",
    "Residual ACF",
    "Residual PACF",
]
```

The flat tuple preserves existing access such as `axes[0]` while mapping positions to the requested grid:

```text
axes[0]  axes[1]
axes[2]  axes[3]
```

**Step 2: Test that the histogram represents every diagnostic residual**

Add a test asserting that the second panel contains histogram bars and that their count heights sum to the residual sample size:

```python
histogram_count = sum(patch.get_height() for patch in axes[1].patches)
assert histogram_count == pytest.approx(len(result.residuals))
```

Keep the white-noise and normality annotation assertion on `axes[0]`.

**Step 3: Update SARIMAX positional assertions**

Require four axes, inspect ACF/PACF on `axes[2:]`, and change the diffuse-initialization ACF assertion from `axes[1]` to `axes[2]`. Continue asserting that `axes[0]` displays exactly `result.residuals`, so the new histogram cannot accidentally reintroduce state-initialization residuals.

**Step 4: Update inherited GARCH and Auto result assertions**

Change their inherited diagnostic tests from three to four axes. Do not alter VAR/VECM tests because those result classes override the shared method with multivariate layouts.

**Step 5: Run tests and verify the old implementation fails**

Run:

```powershell
python -m pytest TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_garch.py TsModels/tests/test_auto.py -q
```

Expected before implementation: failures showing three axes, no histogram panel, and the old ACF/PACF positions.

### Task 2: Implement the shared 2x2 layout

**Files:**
- Modify: `TsModels/_base.py:422`

**Step 1: Create the subplot grid and stable row-major names**

Replace the vertical three-panel construction with:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_residuals, ax_histogram, ax_acf, ax_pacf = axes.flat
diagnostic_residuals = self.residuals
```

Return a flat tuple:

```python
return fig, (ax_residuals, ax_histogram, ax_acf, ax_pacf)
```

Do not return the two-dimensional NumPy array because that would break existing `axes[0]` consumers.

**Step 2: Preserve the residual time-series panel and annotation**

Reuse `plot_series()` on `ax_residuals` with the existing title and labels. Keep the Ljung-Box white-noise and Jarque-Bera normality annotation on this panel without changing the test calculations.

**Step 3: Add the residual histogram with shared visual styling**

There is no existing TsPlots histogram API, so use the Matplotlib Axes method directly and reuse the TsPlots palette and axes style:

```python
ax_histogram.hist(
    diagnostic_residuals,
    bins="auto",
    color=DEFAULT_PALETTE[0],
    edgecolor="white",
    alpha=0.85,
)
ax_histogram.set_title("Residual Histogram")
ax_histogram.set_xlabel("Residual")
ax_histogram.set_ylabel("Frequency")
style_axes(ax_histogram)
```

Use counts rather than density because the request is for a histogram and no fitted density comparison was requested.

**Step 4: Move ACF and PACF into the second row**

Reuse the existing `plot_acf()` and `plot_pacf()` calls on `ax_acf` and `ax_pacf`. Preserve `zero_lag=False` for ACF and the existing PACF default so both correlograms begin at lag one.

**Step 5: Preserve the figure title and spacing**

Keep the current default suptitle, title font, and `tight_layout()` behavior. Do not introduce new plotting parameters or a new histogram abstraction.

**Step 6: Run the focused tests**

```powershell
python -m pytest TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_garch.py TsModels/tests/test_auto.py -q
```

Expected: all selected tests pass.

### Task 3: Update current public documentation

**Files:**
- Modify: `TsModels/README.md:97`
- Modify: `TsModels/_base.py:423`

**Step 1: Update the method docstring**

Describe a four-panel 2-by-2 figure and document the flat row-major axes return order.

**Step 2: Update the README**

Replace the current three-panel description with:

```text
第一行：残差时间序列、残差直方图；第二行：残差 ACF、残差 PACF（2×2）
```

No notebook source change is required because existing `result.plot_diagnostics()` calls automatically render the new layout.

### Task 4: Verify behavior, rendering, and repository quality

**Files:**
- Verify only; generate any smoke-test image outside the repository.

**Step 1: Run the complete TsModels suite**

```powershell
python -m pytest TsModels/tests -q
```

Expected: all TsModels tests pass.

**Step 2: Render and inspect a real SARIMAX diagnostic figure**

Fit a seeded AR(1), call `plot_diagnostics()`, save the figure to a temporary PNG, and visually verify:

- top-left: residual time series and diagnostic annotation;
- top-right: residual histogram;
- bottom-left: ACF beginning at lag one;
- bottom-right: PACF beginning at lag one;
- titles, labels, and suptitle do not overlap.

Delete only the temporary image after inspection.

**Step 3: Run checks scoped to changed files**

```powershell
python -m ruff format --check TsModels/_base.py TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_garch.py TsModels/tests/test_auto.py docs/plans/2026-08-02-tsmodels-diagnostics-2x2.md
python -m ruff check TsModels
python -m compileall -q TsModels
git diff --check
```

Expected: every command exits with code 0. The known unrelated repository-wide format backlog remains outside this change.

**Step 4: Run the current explicit repository gate**

The former `scripts/check_ts_quality.py` is absent in this checkout. Use the active `pyproject.toml` contract directly:

```powershell
python -m pytest -q --cov=TsSims --cov=TsTests --cov-branch --cov-report=term-missing --cov-fail-under=90
```

Expected: all configured package tests pass and branch coverage remains at or above 90%.

**Step 5: Inspect the final diff**

Confirm that the new changes are limited to the shared diagnostic plot, relevant tests, current README, and this plan, while preserving the previously approved default-order/default-missing edits already present in the working tree. Do not commit unless the user separately requests a commit.
