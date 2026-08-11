# Correlogram Missing-Value Policy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared `missing="drop" | "raise"` policy to `plot_acf()` and `plot_pacf()`, defaulting to removal of all non-finite observations.

**Architecture:** Keep conversion and validation inside `TsPlots/acf_plot.py`: `_to_1d()` continues to normalize supported containers, and one new private helper validates the policy and applies a finite mask before either statsmodels function is called. Tests compare each plotting function with explicitly cleaned data and verify that effective sample size controls adaptive lags.

**Tech Stack:** Python 3.13, NumPy, pandas, matplotlib, statsmodels, pytest, Ruff.

---

### Task 1: Specify the missing-value contract with failing tests

**Files:**
- Modify: `TsPlots/tests/test_plots.py:305-351`

**Step 1: Write parameterized default-drop tests**

Add tests for both public functions that pass a copied NumPy array containing
`NaN`, positive infinity, and negative infinity. Compare the plotted bar heights
with a call using `data[np.isfinite(data)]`, and assert the original array is
unchanged with `equal_nan=True`.

```python
@pytest.mark.parametrize("plotter", [plot_acf, plot_pacf])
def test_default_missing_drop_matches_explicitly_cleaned_data(plotter):
    data = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0, -np.inf, 8.0, 9.0, 10.0])
    original = data.copy()

    fig, ax = plotter(data, nlags=2)
    expected_fig, expected_ax = plotter(data[np.isfinite(data)], nlags=2)

    np.testing.assert_allclose(
        [patch.get_height() for patch in ax.patches],
        [patch.get_height() for patch in expected_ax.patches],
    )
    np.testing.assert_array_equal(data, original, strict=True)
    plt.close(fig)
    plt.close(expected_fig)
```

Use separate ACF and PACF assertions where their displayed lag-zero behavior
differs.

**Step 2: Write validation tests**

Add parameterized tests asserting:

```python
with pytest.raises(ValueError, match="row positions: 1, 3"):
    plotter(np.array([1.0, np.nan, 2.0, np.inf, 3.0]), missing="raise")

with pytest.raises(ValueError, match="missing must be 'raise' or 'drop'"):
    plotter(np.arange(20.0), missing="omit")

with pytest.raises(ValueError, match="no finite observations"):
    plotter(np.array([np.nan, np.inf, -np.inf]))
```

**Step 3: Write an effective-sample adaptive-lag test**

Use a 63-element array with three non-finite values. Assert that default-drop
produces the same number of bars as the existing 60-observation adaptive-lag
tests: 18 for ACF (lags 0--17) and 17 for PACF (lags 1--17).

**Step 4: Run the focused tests and verify failure**

Run:

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m pytest TsPlots/tests/test_plots.py -k 'missing or adaptive' -q
```

Expected: new tests fail because neither public function accepts or applies the
approved missing-value policy; existing adaptive-lag tests still pass.

### Task 2: Implement one shared private cleaning path

**Files:**
- Modify: `TsPlots/acf_plot.py:61-91`
- Modify: `TsPlots/acf_plot.py:402-490`
- Modify: `TsPlots/acf_plot.py:529-613`
- Test: `TsPlots/tests/test_plots.py`

**Step 1: Add the private helper**

Immediately after `_to_1d()`, add a helper equivalent to:

```python
def _resolve_missing(data: np.ndarray, missing: str) -> np.ndarray:
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be 'raise' or 'drop'")
    finite = np.isfinite(data)
    positions = np.flatnonzero(~finite)
    if len(positions) and missing == "raise":
        joined = ", ".join(str(int(position)) for position in positions)
        raise ValueError(
            f"data contains non-finite values at row positions: {joined}"
        )
    cleaned = data[finite]
    if cleaned.size == 0:
        raise ValueError("data contains no finite observations")
    return cleaned
```

The implementation must not mutate `data` and must report positions relative to
the caller's original one-dimensional input.

**Step 2: Extend both keyword-only signatures**

Add `missing: str = "drop"` immediately after `alpha` in both functions. Pass
the `_to_1d()` result through `_resolve_missing()` before resolving colours or
calling statsmodels.

**Step 3: Update public docstrings**

Document the accepted values, default, treatment of all non-finite values,
effective-sample lag selection, interior-gap compression warning, and
`ValueError` behavior. Add one executable `missing="raise"` example or a default
drop example without disturbing existing examples.

**Step 4: Run focused tests**

Run:

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m pytest TsPlots/tests/test_plots.py -q
```

Expected: all `TsPlots/tests/test_plots.py` tests pass.

### Task 3: Document the public behavior

**Files:**
- Modify: `TsPlots/README.md:281-333`
- Test: `tests/test_public_docstrings.py`

**Step 1: Update signatures and parameter table**

Show `missing="drop"` in both abbreviated signatures and add a parameter-table
row defining `"drop"` and `"raise"`.

**Step 2: Add statistical warning and examples**

State that the default finite mask removes `NaN` and both infinities before
adaptive lag calculation. Warn that dropping an interior gap compresses time;
recommend `missing="raise"` when that would be misleading. Include:

```python
plot_acf(series, missing="drop")
plot_pacf(series, missing="raise")
```

**Step 3: Run documentation contract tests**

Run:

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m pytest tests/test_public_docstrings.py -q
```

Expected: all public docstring and README checks pass.

### Task 4: Validate and commit the complete feature

**Files:**
- Modify: `TsPlots/acf_plot.py`
- Modify: `TsPlots/tests/test_plots.py`
- Modify: `TsPlots/README.md`
- Already committed design: `docs/plans/2026-08-11-correlogram-missing-policy-design.md`
- Add plan: `docs/plans/2026-08-11-correlogram-missing-policy.md`

**Step 1: Run scoped style checks**

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m ruff format --check TsPlots/acf_plot.py TsPlots/tests/test_plots.py
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m ruff check TsPlots/acf_plot.py TsPlots/tests/test_plots.py
git diff --check
```

Expected: all commands exit zero.

**Step 2: Run TsPlots and public-contract tests**

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m pytest TsPlots/tests tests/test_public_docstrings.py -q
```

Expected: all focused and public-contract tests pass.

**Step 3: Run the full repository suite**

```powershell
& 'D:\sync\sync\forecasting with dynamic regression models\.venv\Scripts\python.exe' -m pytest -q
```

Expected: the full suite passes without new warnings or failures.

**Step 4: Audit the final diff and commit**

Confirm that only the approved ACF/PACF implementation, tests, README, and plan
are uncommitted. Then run:

```powershell
git add TsPlots/acf_plot.py TsPlots/tests/test_plots.py TsPlots/README.md docs/plans/2026-08-11-correlogram-missing-policy.md
git commit -m "feat: handle missing values in correlogram plots"
```

Expected: `main` contains the design commit and one clean feature commit; the
worktree is clean. Do not push unless separately requested.
