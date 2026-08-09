# Standardized Residuals Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public population-standardized residual array to every model result and make all residual diagnostic figures display it.

**Architecture:** Implement the calculation once as a read-only `BaseModelResult.standardized_residuals` property. Reuse that property in the shared univariate diagnostic plot and the existing VAR/VECM overrides; preserve `.residuals` and `test_residuals()` unchanged.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, Ruff, existing `TsModels`, `TsPlots`, and `TsTests` interfaces.

---

### Task 1: Add the shared standardized-residual property

**Files:**
- Modify: `TsModels/tests/test_base.py`
- Modify: `TsModels/_base.py:335-389`

**Step 1: Write failing property tests**

Add tests that calculate the expected values explicitly:

```python
def test_standardized_residuals_divide_by_population_std(self, result):
    original = result.residuals.copy()
    expected = original / np.std(original, ddof=0)

    np.testing.assert_allclose(result.standardized_residuals, expected)
    np.testing.assert_array_equal(result.residuals, original)


@pytest.mark.parametrize(
    "residuals",
    [np.zeros(4), np.array([0.0, 1.0, np.nan])],
)
def test_standardized_residuals_reject_invalid_scale(residuals):
    ...
```

The fixture residuals have a nonzero sample mean, so the expected expression
also guards against accidentally implementing a z-score that subtracts the
mean.

**Step 2: Run the new tests and verify failure**

Run:

```powershell
python -m pytest TsModels/tests/test_base.py -k standardized_residuals -q
```

Expected: failure because `BaseModelResult` has no `standardized_residuals`.

**Step 3: Implement the minimal shared property**

Add a documented read-only property to `BaseModelResult`:

```python
@property
def standardized_residuals(self) -> np.ndarray:
    residuals = np.asarray(self.residuals, dtype=float)
    axis = 0 if residuals.ndim == 2 else None
    scale = np.std(residuals, axis=axis, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("residual standard deviation must be positive and finite")
    return residuals / scale
```

Document the `ddof=0` convention, division without separate mean subtraction,
and column-wise two-dimensional behavior in the class docstring.

**Step 4: Run the property tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 2: Use standardized residuals in the shared diagnostic figure

**Files:**
- Modify: `TsModels/tests/test_base.py:72-167`
- Modify: `TsModels/tests/test_sarimax.py:401-462`
- Modify: `TsModels/_base.py:470-602`

**Step 1: Update tests to specify displayed values and labels**

Require:

```python
displayed = np.asarray(axes[0].lines[0].get_ydata(), dtype=float)
np.testing.assert_allclose(displayed, result.standardized_residuals)
assert axes[0].get_title() == "Standardized Residuals"
assert axes[0].get_ylabel() == "Standardized Residual"
assert axes[1].get_title() == "Standardized Residual Histogram"
assert axes[1].get_xlabel() == "Standardized Residual"
```

Keep the histogram-count assertion. In the SARIMAX burn-in test, calculate the
expected plotted values from `result.standardized_residuals`, while retaining
the assertion that `.residuals` equals the post-burn statsmodels residuals.

**Step 2: Run focused tests and verify the old plot contract fails**

Run:

```powershell
python -m pytest TsModels/tests/test_base.py TsModels/tests/test_sarimax.py -k "plot_diagnostics or diagnostics_exclude" -q
```

Expected: failures showing that raw residuals and old labels are still used.

**Step 3: Update the shared implementation**

Set `diagnostic_residuals = self.standardized_residuals`. Rename the four panel
titles to `Standardized Residuals`, `Standardized Residual Histogram`,
`Standardized Residual ACF`, and `Standardized Residual PACF`. Rename the time
plot y-axis and histogram x-axis to `Standardized Residual`.

Continue to reuse `TsPlots.plot_series()`, `plot_acf()`, `plot_pacf()`, shared
style constants, `LjungBoxTest`, and `NormalityTest`.

**Step 4: Run the focused tests**

Run the command from Step 2.

Expected: all selected tests pass; SARIMAX still excludes initialization
residuals.

### Task 3: Use column-standardized residuals in VAR and VECM diagnostics

**Files:**
- Modify: `TsModels/tests/test_var.py:419-430`
- Modify: `TsModels/tests/test_vecm.py:466-500`
- Modify: `TsModels/_var.py:995-1054`
- Modify: `TsModels/_vecm.py:648-695`

**Step 1: Add failing multivariate tests**

For fitted VAR and VECM results, assert:

```python
standardized = fitted.standardized_residuals
np.testing.assert_allclose(np.std(standardized, axis=0, ddof=0), 1.0)
for position in range(standardized.shape[1]):
    displayed = np.asarray(axes[position, 0].lines[0].get_ydata(), dtype=float)
    np.testing.assert_allclose(displayed, standardized[:, position])
```

Also assert standardized residual titles and y-axis labels. Retain the original
`.residuals` shape and existing diagnostic return-contract assertions.

**Step 2: Run focused multivariate tests and verify failure**

Run:

```powershell
python -m pytest TsModels/tests/test_var.py TsModels/tests/test_vecm.py -k plot_diagnostics -q
```

Expected: failures because the overrides still plot raw residuals.

**Step 3: Update both existing overrides**

Calculate `diagnostic_residuals = self.standardized_residuals` once per method,
then reuse the appropriate column for the time plot, ACF, and PACF. Update the
panel titles and y-axis labels without changing grid layout, variable naming,
or `TsPlots` calls.

**Step 4: Run focused multivariate tests**

Run the command from Step 2.

Expected: all selected tests pass.

### Task 4: Update public documentation

**Files:**
- Modify: `TsModels/_base.py:335-365,470-476`
- Modify: `TsModels/README.md:98-108`

**Step 1: Update documentation**

Add `.standardized_residuals` to the shared Result interface table. Define it as
residuals divided by their population standard deviation (`ddof=0`), with
per-equation scaling for multivariate results. State that diagnostic figures
display standardized residuals and that `.residuals` remains available in its
original model scale.

Explicitly state that GARCH conditional standardization by
`.conditional_volatility` is not part of this feature.

**Step 2: Run documentation-related doctests/import checks**

Run:

```powershell
python -m pytest TsModels/tests/test_base.py TsModels/tests/test_auto.py TsModels/tests/test_garch.py -q
```

Expected: all tests pass.

No notebook source change is required: existing notebook calls to
`result.plot_diagnostics()` automatically render the new public behavior.

### Task 5: Complete regression and quality verification

**Files:**
- Verify all modified source, test, documentation, and plan files.

**Step 1: Run the complete TsModels test suite**

```powershell
python -m pytest TsModels/tests -q
```

Expected: all TsModels tests pass.

**Step 2: Run the complete repository test suite**

```powershell
python -m pytest -q
```

Expected: all repository tests pass.

**Step 3: Run lint and source validation**

```powershell
python -m ruff check TsModels
python -m compileall -q TsModels
git diff --check
```

Expected: Ruff reports `All checks passed!`; compileall and diff check exit 0.

**Step 4: Inspect the final change boundary**

```powershell
git status --short
git diff --stat
git diff -- TsModels/_base.py TsModels/_var.py TsModels/_vecm.py TsModels/README.md TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_var.py TsModels/tests/test_vecm.py
```

Expected: only the approved standardized-residual implementation, tests,
documentation, and implementation plan are present.

**Step 5: Commit the verified feature**

```powershell
git add docs/plans/2026-08-09-standardized-residuals-implementation.md TsModels/_base.py TsModels/_var.py TsModels/_vecm.py TsModels/README.md TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_var.py TsModels/tests/test_vecm.py
git commit -m "feat: standardize model residual diagnostics"
```

Expected: one clean feature commit on `main`.
