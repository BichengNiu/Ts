# Rolling-Window Forecast Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing leakage-free `backtest()` result with a time-ordered table of canonical forecast-error metrics calculated separately for every complete overlapping forecast window.

**Architecture:** Keep model cloning, refitting, prediction, and failure handling in the existing `TsMetrics.backtest()` engine. Add one internal window-axis aggregator that delegates every calculation to `compute_metrics()`, then expose it through a backward-compatible `BacktestResult.metrics_by_window` property using optional calendar and series-name metadata already available from the evaluation protocol.

**Tech Stack:** Python 3.13, NumPy, pandas, pytest, Ruff, standard-library `compileall`.

---

### Task 1: Add positional univariate window metrics

**Files:**
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/_aggregation.py`
- Modify: `TsMetrics/_results.py:10-15,368-497`

**Step 1: Write the failing exact-value test**

Add:

```python
def test_backtest_metrics_by_window_are_exact_and_position_labelled():
    result = BacktestResult(
        mean=np.array([[2.0, 4.0], [3.0, 8.0]]),
        actual=np.array([[1.0, 5.0], [3.0, 4.0]]),
        lower=None,
        upper=None,
        origins=np.array([10, 11]),
        failures=[],
        model_type="TEST",
        window="expanding",
        target="observed",
    )

    frame = result.metrics_by_window

    assert frame.columns.tolist() == [
        "window_start", "window_end", "mae", "mse", "rmse",
        "mape", "smape", "theil_u1", "n",
    ]
    assert frame["window_start"].tolist() == [10, 11]
    assert frame["window_end"].tolist() == [11, 12]
    assert frame["rmse"].tolist() == pytest.approx([1.0, np.sqrt(8.0)])
    assert frame["mape"].tolist() == pytest.approx([60.0, 50.0])
    assert frame["n"].tolist() == [2, 2]
```

This fixes the public column order and proves that each row scores one complete forecast window rather than one horizon pooled across origins.

**Step 2: Run the test and verify failure**

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py::test_backtest_metrics_by_window_are_exact_and_position_labelled -q
```

Expected: FAIL because `BacktestResult` has no `metrics_by_window` attribute.

**Step 3: Add the internal aggregator**

In `TsMetrics/_aggregation.py`, add a helper whose outer list is origin and inner list is series:

```python
def backtest_metrics_by_window(actual, predicted):
    """Compute canonical metrics separately for every forecast window."""
    if actual.shape != predicted.shape:
        raise ValueError("backtest actual and predicted must have the same shape")
    if predicted.ndim == 2:
        return [
            [compute_metrics(actual[index], predicted[index])]
            for index in range(predicted.shape[0])
        ]
    if predicted.ndim == 3:
        return [
            [
                compute_metrics(
                    actual[origin, :, series],
                    predicted[origin, :, series],
                )
                for series in range(predicted.shape[2])
            ]
            for origin in range(predicted.shape[0])
        ]
    raise ValueError(
        f"unsupported backtest shape for window metrics: {predicted.shape}"
    )
```

Do not duplicate metric formulas. `compute_metrics()` remains authoritative for numerical stability, missing pairs, zero actuals, and `n`.

**Step 4: Add the minimal result property**

Import `backtest_metrics_by_window` and `ERROR_METRIC_NAMES` in `_results.py`, then add:

```python
@property
def metrics_by_window(self):
    """Return canonical metrics for each complete forecast window."""
    horizon = self.mean.shape[1]
    rows = []
    for position, metrics in enumerate(
        backtest_metrics_by_window(self.actual, self.mean)
    ):
        rows.append(
            {
                "window_start": int(self.origins[position]),
                "window_end": int(self.origins[position] + horizon - 1),
                **metrics[0],
            }
        )
    columns = ["window_start", "window_end", *ERROR_METRIC_NAMES, "n"]
    return pd.DataFrame.from_records(rows, columns=columns)
```

Task 2 generalizes this single path for dates and multivariate output.

**Step 5: Run focused tests**

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py::test_backtest_metrics_by_window_are_exact_and_position_labelled TsMetrics/tests/test_metrics.py -q
```

Expected: all selected tests PASS.

**Step 6: Commit**

```powershell
git add -- TsMetrics/_aggregation.py TsMetrics/_results.py TsMetrics/tests/test_evaluation.py
git commit -m "feat: add per-window backtest metrics"
```

---

### Task 2: Preserve dates and series names in `BacktestResult`

**Files:**
- Modify: `TsMetrics/tests/test_contracts.py`
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/_results.py:41-85,368-497`

**Step 1: Write failing metadata contract tests**

First prove the original positional signature remains valid:

```python
def test_backtest_result_metadata_defaults_preserve_positional_construction():
    result = BacktestResult(
        np.ones((1, 1)), np.ones((1, 1)), None, None,
        np.array([10]), [], "TEST", "expanding", "observed",
    )
    assert result.dates is None
    assert result.series_names is None
```

Add tests for defensive copying of a valid `DatetimeIndex`; rejection of missing, duplicate, decreasing, or too-short calendars; preservation of one unique non-empty name per multivariate series; and rejection of names on univariate or wrong-width results.

**Step 2: Run tests and verify failure**

```powershell
python -m pytest TsMetrics/tests/test_contracts.py -q
```

Expected: new tests FAIL because the fields do not exist.

**Step 3: Append optional fields and reuse validators**

Append after `target`:

```python
dates: pd.DatetimeIndex | None = None
series_names: tuple[str, ...] | None = None
```

Normalize and validate them in `__post_init__`:

```python
self.dates = (
    None if self.dates is None else pd.DatetimeIndex(self.dates).copy()
)
if self.dates is not None:
    _validate_dates("dates", self.dates, len(self.dates))
    target_indices = self.target_indices
    if target_indices.min() < 0 or target_indices.max() >= len(self.dates):
        raise ValueError("dates must cover every backtest target index")
self.series_names = _validate_series_names(self.series_names, self.mean[0])
```

Using `self.mean[0]` deliberately reuses the OOS validator because its shape is `(horizon,)` for univariate and `(horizon, n_series)` for multivariate backtests.

**Step 4: Generalize the table without pooling scales**

Use names or stable fallback labels:

```python
if self.mean.ndim == 2:
    labels = (None,)
else:
    labels = self.series_names or tuple(
        f"series_{position}" for position in range(self.mean.shape[2])
    )
```

Map positions to dates when available:

```python
starts = self.origins
ends = self.origins + self.mean.shape[1] - 1
if self.dates is not None:
    starts = self.dates.take(starts)
    ends = self.dates.take(ends)
```

Build one row per origin for univariate output and one row per origin-series pair for multivariate output. Include `series` only for multivariate output. Never add a pooled multivariate RMSE.

**Step 5: Add exact date and multivariate tests**

Assert a dated result maps `[10, 11]` to `dates[10:12]` and ends to `dates[11:13]`. For two series, assert:

```python
frame = result.metrics_by_window
assert frame["series"].tolist() == ["output", "prices", "output", "prices"]
assert len(frame) == result.mean.shape[0] * result.mean.shape[2]
```

Also assert fallback labels are `series_0`, `series_1` when names are unavailable.

**Step 6: Run focused tests**

```powershell
python -m pytest TsMetrics/tests/test_contracts.py TsMetrics/tests/test_evaluation.py -q
```

Expected: all focused tests PASS.

**Step 7: Commit**

```powershell
git add -- TsMetrics/_results.py TsMetrics/tests/test_contracts.py TsMetrics/tests/test_evaluation.py
git commit -m "feat: preserve backtest window metadata"
```

---

### Task 3: Populate metadata through the existing engine

**Files:**
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsMetrics/tests/test_contracts.py`
- Modify: `TsMetrics/_backtest.py:7-18,123-214`

**Step 1: Write the failing end-to-end overlap test**

Use the existing deterministic `_MeanModel`:

```python
def test_overlapping_dated_windows_run_through_latest_observation():
    dates = pd.date_range("2020-01-01", periods=20, freq="MS")
    model = _MeanModel(np.arange(20.0), dates=dates)
    result = model.backtest(
        initial_window=10, horizon=3, step=1, window="expanding"
    )
    frame = result.metrics_by_window

    assert len(frame) == 20 - 10 - 3 + 1
    assert frame["window_start"].tolist()[:2] == [dates[10], dates[11]]
    assert frame["window_end"].tolist()[:2] == [dates[12], dates[13]]
    assert frame.iloc[-1]["window_end"] == dates[-1]
    assert [window[-1] for window in model.fit_windows] == list(
        np.arange(9.0, 17.0)
    )
    assert model.result_ is None
```

This proves exact window count, one-period overlap, expanding refits, latest-period coverage, and caller immutability.

**Step 2: Extend failure and tuple-date tests**

For the existing `on_error="record"` result, assert:

```python
frame = result.metrics_by_window
assert frame["n"].tolist() == [0, 0]
assert frame.drop(columns=["window_start", "window_end", "n"]).isna().all().all()
```

Extend the tuple-date protocol test to call `backtest()` and prove its result owns a normalized `DatetimeIndex` whose final window ends on the final date.

**Step 3: Run and verify failure**

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py::test_overlapping_dated_windows_run_through_latest_observation TsMetrics/tests/test_evaluation.py::test_recorded_failure_does_not_leave_partial_forecast_values -q
```

Expected: the dated test FAILS because `backtest()` has not populated `dates`.

**Step 4: Reuse evaluation metadata helpers**

Import `model_series_names` alongside `model_data`, then resolve metadata once:

```python
data = model_data(model)
target = validate_model_protocol(model, "backtest")
dates = validated_model_dates(model, data)
series_names = model_series_names(model, data)
```

Pass `dates=dates` and `series_names=series_names` to the existing `BacktestResult(...)` call. Do not change origin generation, training slices, future-exogenous hooks, fitting, forecast normalization, intervals, or failure recording.

**Step 5: Run all TsMetrics tests**

```powershell
python -m pytest TsMetrics/tests -q
```

Expected: all tests PASS.

**Step 6: Commit**

```powershell
git add -- TsMetrics/_backtest.py TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py
git commit -m "feat: label rolling metrics with model metadata"
```

---

### Task 4: Prove SARIMAX and multivariate integration

**Files:**
- Modify: `TsModels/tests/test_evaluation.py:416-456`

**Step 1: Extend the existing SARIMAX test**

Reuse its fitted result:

```python
dynamic = result.metrics_by_window
assert len(dynamic) == 2
assert dynamic["window_start"].tolist() == [60, 70]
assert dynamic["window_end"].tolist() == [61, 71]
assert np.isfinite(dynamic["rmse"]).all()
```

**Step 2: Extend the existing VAR test with names**

Construct VAR with `cols=["output", "prices"]`, then assert:

```python
dynamic = result.metrics_by_window
assert result.series_names == ("output", "prices")
assert dynamic["series"].tolist() == [
    "output", "prices", "output", "prices",
]
assert dynamic.groupby("series", sort=False).size().to_dict() == {
    "output": 2,
    "prices": 2,
}
assert np.isfinite(dynamic["rmse"]).all()
```

**Step 3: Run model and cross-package tests**

```powershell
python -m pytest TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py -q
python -m pytest TsMetrics/tests TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py -q
```

Expected: all selected tests PASS.

**Step 4: Commit**

```powershell
git add -- TsModels/tests/test_evaluation.py
git commit -m "test: cover dynamic backtest metrics across models"
```

---

### Task 5: Document the workflow and result contract

**Files:**
- Modify: `TsMetrics/_results.py:368-415,484-497`
- Modify: `TsMetrics/README.md:101-122`
- Modify: `TsModels/README.md:258-303`

**Step 1: Expand the result docstring**

Document `dates`, `series_names`, and `metrics_by_window`. Include:

```python
result = model.backtest(
    initial_window=30,
    horizon=3,
    step=1,
    window="expanding",
)
dynamic = result.metrics_by_window
dynamic[["window_start", "window_end", "rmse", "mape", "n"]]
```

State that `origin` is the first target position, each row scores the complete horizon, and multivariate rows contain `series`.

**Step 2: Update both READMEs**

In `TsMetrics/README.md`, explain:

- `initial_window` is the first minimum training sample;
- `horizon=N` is the complete forecast/scoring window;
- `step=1` creates overlapping windows;
- `window="expanding"` adds each newly observed period to the next refit;
- a length-`T` sample yields `T - initial_window - N + 1` rows;
- the last window ends at the latest observation;
- canonical metric, missing-pair, zero-actual MAPE, error-recording, date, and multivariate semantics.

In `TsModels/README.md`, extend the existing backtest example:

```python
dynamic = expanding.metrics_by_window
print(dynamic[["window_start", "window_end", "rmse", "mape", "n"]])
```

For VAR/VECM/SVAR, explain that rows are separated by `series`. Do not add or document a second evaluator.

**Step 3: Run documentation and API checks**

```powershell
python -m pytest tests/test_public_docstrings.py TsMetrics/tests TsModels/tests/test_evaluation.py -q
Push-Location ..
try { python -c "import numpy as np; from Ts.TsModels import SARIMAX; r=SARIMAX(np.arange(20.0), order=(0,0,0)).backtest(initial_window=10, horizon=3, step=1); f=r.metrics_by_window; assert len(f)==8; assert f.iloc[-1]['window_end']==19; print(f[['window_start','window_end','rmse','mape','n']].tail(2).to_string(index=False))" } finally { Pop-Location }
```

Expected: tests PASS; the script exits 0, prints two rows, and ends at position 19.

**Step 4: Commit**

```powershell
git add -- TsMetrics/_results.py TsMetrics/README.md TsModels/README.md
git commit -m "docs: explain rolling forecast reliability metrics"
```

Do not modify, restore, stage, or commit the unrelated deleted `chapter6.ipynb`.

---

### Task 6: Audit the design boundary and validate the repository

**Files:**
- Inspect: `docs/plans/2026-08-09-rolling-window-forecast-reliability-design.md`
- Inspect: `docs/plans/2026-08-09-rolling-window-forecast-reliability-implementation.md`
- Inspect: `TsMetrics/_aggregation.py`
- Inspect: `TsMetrics/_results.py`
- Inspect: `TsMetrics/_backtest.py`
- Inspect: `TsMetrics/tests/test_contracts.py`
- Inspect: `TsMetrics/tests/test_evaluation.py`
- Inspect: `TsModels/tests/test_evaluation.py`
- Inspect: `TsMetrics/README.md`
- Inspect: `TsModels/README.md`

**Step 1: Audit the reuse boundary**

Confirm from source that fitting and forecasting still use `backtest()` and `fit_and_forecast()`; window aggregation calls `compute_metrics()`; no `rolling_reliability()` exists; `TsMetrics` imports no model class; multivariate dynamic RMSE is per series; trailing partial windows are absent; and the caller remains unmodified.

```powershell
rg -n "metrics_by_window|backtest_metrics_by_window|rolling_reliability" TsMetrics TsModels __init__.py
git status --short
```

Expected: only the approved extension appears; `rolling_reliability` has no match; the unrelated Notebook deletion remains unstaged.

**Step 2: Run focused and full tests**

```powershell
python -m pytest TsMetrics/tests TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py -q
python -m pytest -q
```

Expected: all tests PASS.

**Step 3: Run static checks**

```powershell
python -m ruff check TsMetrics TsModels
python -m compileall -q TsMetrics TsModels
git diff --check
```

Expected: all commands exit 0 with no reported violations.

**Step 4: Verify the acceptance equation with dates**

```powershell
Push-Location ..
try { python -c "import numpy as np, pandas as pd; from Ts.TsModels import SARIMAX; d=pd.date_range('2020-01-01', periods=20, freq='MS'); r=SARIMAX(pd.Series(np.arange(20.0), index=d), order=(0,0,0)).backtest(initial_window=10, horizon=3, step=1, window='expanding'); f=r.metrics_by_window; assert len(f)==20-10-3+1; assert f.iloc[-1]['window_end']==d[-1]; assert f['n'].eq(3).all(); print(f.tail(1).to_string(index=False))" } finally { Pop-Location }
```

Expected: exit code 0; the final row ends on `2021-08-01` and has `n == 3`.

**Step 5: Review repository state**

```powershell
git status --short
git log -8 --oneline --decorate
```

Expected: implementation commits are on `main`, no feature branch exists, and the only unrelated working-tree change is the pre-existing deletion of `chapter6.ipynb`.

If any check fails, fix the smallest owning layer, rerun its focused test, then repeat Tasks 6.2-6.5. Do not report completion while a required check remains failing.
