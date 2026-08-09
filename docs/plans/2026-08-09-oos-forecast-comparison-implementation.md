# OOS Forecast Comparison Helpers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reusable table and plot helpers to `OOSComparisonResult` for arbitrary univariate and multivariate OOS model comparisons.

**Architecture:** Extend `OOSResult` with backward-compatible optional series-name and interval-level metadata populated by the existing OOS engine. Build the comparison table inside `OOSComparisonResult`, and implement its plotting method as a thin composition over the public `TsPlots.plot_series()` interface plus same-colour interval fills.

**Tech Stack:** Python 3.13, NumPy, pandas, Matplotlib, pytest, Ruff, nbformat/nbclient.

---

### Task 1: Preserve OOS series names and interval level

**Files:**
- Modify: `TsMetrics/tests/test_contracts.py`
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/_results.py`
- Modify: `TsMetrics/_evaluation.py`
- Modify: `TsMetrics/_oos.py`

**Step 1: Write failing contract tests**

Add tests proving that:

```python
result = OOSResult(
    mean=np.ones((2, 2)),
    actual=np.ones((2, 2)),
    lower=np.zeros((2, 2)),
    upper=np.full((2, 2), 2.0),
    estimation_indices=np.arange(10),
    validation_indices=np.array([10, 11]),
    estimation_dates=None,
    validation_dates=None,
    model_type="TEST",
    target="observed",
    series_names=("output", "prices"),
    alpha=0.10,
)
assert result.series_names == ("output", "prices")
assert result.alpha == pytest.approx(0.10)
```

Also reject duplicate/empty/wrong-length names, names on a univariate result, invalid `alpha`, and `alpha` without intervals. Retain a test constructing the original ten positional arguments to prove backward compatibility.

**Step 2: Run the tests and verify failure**

Run:

```powershell
python -m pytest TsMetrics/tests/test_contracts.py -q
```

Expected: new tests fail because `OOSResult` does not accept `series_names` or `alpha`.

**Step 3: Add metadata fields and validation**

Append defaulted fields after `target`:

```python
series_names: tuple[str, ...] | None = None
alpha: float | None = None
```

Normalize `series_names` to a tuple, require one unique non-empty string per multivariate column, and reject names for one-dimensional results. Normalize `alpha` to float and require `0 < alpha < 1`; require intervals when it is set.

Add an evaluation helper that reads `model.data_names` for two-dimensional data, validates its length, and returns a copied tuple. In `oos()`, pass this tuple and the already validated `alpha` into `OOSResult`.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_contracts.py TsMetrics/tests/test_evaluation.py -q
```

Expected: all focused tests pass.

### Task 2: Add the automatic comparison table

**Files:**
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/_results.py`

**Step 1: Write failing table tests**

Cover three arbitrary model names and assert:

```python
frame = report.forecast_table()
assert frame.columns.tolist() == [
    "Actual",
    "model-a forecast",
    "model-a error",
    "model-b forecast",
    "model-b error",
    "model-c forecast",
    "model-c error",
]
np.testing.assert_allclose(
    frame["model-a error"],
    frame["model-a forecast"] - frame["Actual"],
)
```

Add cases for date/position indices, `include_errors=False`, default interval exclusion, `include_intervals=True`, and mixed interval availability. Add multivariate tests selecting by `series="output"` and `series=1`, plus errors for missing/unknown/out-of-range selections.

**Step 2: Verify tests fail**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py -q
```

Expected: failures report missing `forecast_table`.

**Step 3: Implement selection and table construction**

Add private helpers in `_results.py` to:

- validate booleans without accepting integer substitutes;
- verify shared OOS horizon, actual values, validation metadata, series names and target;
- resolve a one-dimensional values view from `series=None`, integer position or name;
- select `validation_dates` when present, otherwise `validation_indices`.

Implement `forecast_table()` so it copies all arrays into a new `DataFrame`, preserves report insertion order, computes `mean - actual`, and includes interval columns only on explicit request and only where bounds exist.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py -q
```

Expected: all tests pass and existing `report.table` assertions remain unchanged.

### Task 3: Add the automatic comparison plot

**Files:**
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/_results.py`

**Step 1: Write failing plot tests**

Use Matplotlib's non-interactive test environment and assert:

```python
fig, ax = report.plot_forecasts(
    series="output",
    colors=["#222222", "#1f77b4", "#ff7f0e"],
    title="Holdout comparison",
    freq="month",
    show_intervals=True,
)
assert ax.get_title() == "Holdout comparison"
assert [line.get_label() for line in ax.lines] == [
    "Actual",
    "model-a forecast",
    "model-b forecast",
]
assert len(ax.collections) == 2
```

Also verify that a model without bounds adds no collection, `show_intervals=False` adds none, a non-default `alpha=0.10` produces a `90% interval` legend label, supplied colours are shared by lines and fills, an existing `ax` is reused, and invalid `interval_alpha` is rejected.

**Step 2: Verify tests fail**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py -q
```

Expected: failures report missing `plot_forecasts`.

**Step 3: Implement a thin TsPlots composition**

Inside `plot_forecasts()`:

1. Call `forecast_table(series=..., include_errors=False)`.
2. Retain only `Actual` and forecast columns.
3. Lazily import `Ts.TsPlots.plot_series` and `Ts.TsPlots.style.DEFAULT_PALETTE` to avoid import cycles.
4. Resolve colours as deep gray for actual plus palette colours for models, or require one supplied colour per plotted line.
5. Call `plot_series(frame, facet=False, auto_dual_y=False, ...)` with public keyword arguments.
6. Add each available interval via `ax.fill_between()` using the corresponding forecast colour and validated opacity.
7. Rebuild the frameless legend and return `(fig, ax)`.

**Step 4: Run focused plotting tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsPlots/tests -q
```

Expected: all tests pass with no open-figure warnings.

### Task 4: Document, demonstrate, and verify the public workflow

**Files:**
- Modify: `TsMetrics/README.md`
- Modify: `TsMetrics/demo.ipynb`
- Modify: `TsMetrics/_results.py`

**Step 1: Update public documentation**

Document the short workflow:

```python
comparison = report.forecast_table()
fig, ax = report.plot_forecasts(
    title="Validation forecasts",
    xtitle="Validation month",
    ytitle="Housing starts",
    freq="month",
    grid=True,
)
```

Explain the `forecast - actual` sign, default interval exclusion from tables, automatic plot intervals, arbitrary-model behavior, and multivariate `series=` selection.

Expand result-container docstrings with the two methods, metadata fields, return types, and concise examples.

**Step 2: Update and execute the demo Notebook**

Replace the manual OOS comparison DataFrame and manual `fill_between()` calls in `TsMetrics/demo.ipynb` with `report.forecast_table()` and `report.plot_forecasts()`. Execute from a clean kernel with `nbclient`, save outputs, and verify every code cell completed without an error output.

**Step 3: Run focused and full verification**

Run:

```powershell
python -m pytest TsMetrics/tests TsPlots/tests -q
python -m pytest -q
python -m ruff check TsMetrics TsPlots
python -m compileall -q TsMetrics TsPlots
git diff --check
```

Expected: all tests pass; Ruff, compilation and whitespace checks exit zero.

**Step 4: Audit and publish**

Inspect the final diff for hard-coded model names, duplicated plotting logic, stale manual Notebook code, compatibility regressions and unrelated changes. Commit the implementation and documentation on `main`, confirm the working tree is clean and ahead of `origin/main`, then push `main` to `origin` and verify local/remote commit equality.

