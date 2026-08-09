# Unified Forecast Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace every OOS/backtest-specific public API with one `evaluate_forecasts()` entrypoint driven by `Holdout` or `RollingOrigin`, while fixing one-step RDL forecasting and returning one fair multi-model comparison result.

**Architecture:** Public scheme dataclasses generate immutable internal forecast splits. One engine reuses the existing evaluation protocol to clone, fit, forecast, and score every named estimator; fixed holdout is represented as one split and rolling-origin evaluation as multiple equal-horizon splits. Existing metrics and TsPlots are composed into two unified result classes, and all old functions/classes are removed without aliases.

**Tech Stack:** Python 3.13, NumPy, pandas, statsmodels, matplotlib, pytest, Ruff.

---

### Task 1: Repair one-step rational distributed-lag forecasting

**Files:**
- Modify: `TsModels/_distributed_lag.py:840-872`
- Test: `TsModels/tests/test_distributed_lag.py`

**Step 1: Write failing direct-prediction and evaluation regression tests**

Add a small deterministic RDL fixture and assert that a one-period future prediction returns shape `(1,)`, finite values, and aligned bounds. Add a second test exercising the existing evaluation path with `horizon=1`; retain a `horizon=2` assertion to protect the working path.

Core assertion:

```python
one_step = fitted.predict(
    start=fitted.nobs,
    end=fitted.nobs,
    future_exog=np.array([[future_x]]),
)
assert one_step.mean.shape == (1,)
assert np.isfinite(one_step.mean).all()
```

**Step 2: Run the regression tests and verify failure**

Run:

```powershell
python -m pytest TsModels/tests/test_distributed_lag.py -k "one_step" -q
```

Expected: FAIL with `AxisError: axis 1 is out of bounds for array of dimension 1` from `_RationalLagSARIMAX.update()`.

**Step 3: Normalise the observation intercept before repetition**

Change the update path to make the state-space intercept explicitly two-dimensional:

```python
base_intercept = np.asarray(self.ssm["obs_intercept"]).reshape(
    self.k_endog,
    -1,
)
if base_intercept.shape[1] == 1 and self.nobs > 1:
    base_intercept = np.repeat(base_intercept, self.nobs, axis=1)
```

Keep the existing transfer-effect addition and parameter handling unchanged.

**Step 4: Run focused RDL tests**

Run:

```powershell
python -m pytest TsModels/tests/test_distributed_lag.py -q
```

Expected: all distributed-lag tests PASS, including one- and multi-step forecasts.

**Step 5: Commit**

```powershell
git add -- TsModels/_distributed_lag.py TsModels/tests/test_distributed_lag.py
git commit -m "fix: support one-step RDL forecasts"
```

### Task 2: Add explicit, immutable forecast-evaluation schemes

**Files:**
- Create: `TsMetrics/_schemes.py`
- Create: `TsMetrics/tests/test_schemes.py`

**Step 1: Write failing Holdout split tests**

Cover positional and `DatetimeIndex` boundaries, inclusive endpoints, a permitted gap between train and test, test-before-train rejection, absent dates, mixed boundary types, and empty/out-of-range periods.

Expected contract:

```python
scheme = Holdout(train=(0, 9), test=(12, 14))
splits = scheme.split(nobs=20, dates=None)
assert len(splits) == 1
assert splits[0].train_indices.tolist() == list(range(10))
assert splits[0].target_indices.tolist() == [12, 13, 14]
assert splits[0].gap == 2
```

**Step 2: Write failing RollingOrigin split tests**

Cover expanding, rolling, default/explicit `window_size`, `gap`, `step`, complete-horizon truncation, date labels, and invalid bool/non-positive integers.

Expected examples:

```python
scheme = RollingOrigin(initial_window=10, horizon=3, step=4)
assert [s.target_indices.tolist() for s in scheme.split(20, None)] == [
    [10, 11, 12],
    [14, 15, 16],
]

gapped = RollingOrigin(initial_window=10, horizon=2, gap=2)
first = gapped.split(20, None)[0]
assert first.train_indices.tolist() == list(range(10))
assert first.target_indices.tolist() == [12, 13]
```

**Step 3: Run tests and verify missing imports**

Run:

```powershell
python -m pytest TsMetrics/tests/test_schemes.py -q
```

Expected: collection FAIL because `Holdout` and `RollingOrigin` do not exist.

**Step 4: Implement scheme dataclasses and internal split records**

Create frozen dataclasses:

```python
@dataclass(frozen=True)
class ForecastSplit:
    split: int
    train_indices: np.ndarray
    target_indices: np.ndarray
    train_start: object
    train_end: object
    forecast_start: object
    forecast_end: object
    gap: int
    window: str

@dataclass(frozen=True)
class Holdout:
    train: tuple
    test: tuple

@dataclass(frozen=True)
class RollingOrigin:
    initial_window: int
    horizon: int = 1
    step: int = 1
    window: str = "expanding"
    window_size: int | None = None
    gap: int = 0
```

Use existing period/date validation helpers from `TsMetrics/_periods.py` where their closed-boundary contract matches. Do not duplicate date parsing.

**Step 5: Run scheme tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_schemes.py -q
```

Expected: all scheme tests PASS.

**Step 6: Commit**

```powershell
git add -- TsMetrics/_schemes.py TsMetrics/tests/test_schemes.py
git commit -m "feat: define forecast evaluation schemes"
```

### Task 3: Replace method-specific results with unified result containers

**Files:**
- Modify: `TsMetrics/_results.py`
- Modify: `TsMetrics/_aggregation.py`
- Test: `TsMetrics/tests/test_evaluation.py`
- Test: `TsMetrics/tests/test_contracts.py`

**Step 1: Write failing `ForecastEvaluationResult` shape tests**

Require one common result shape for holdout and rolling evaluation, exact interval alignment, increasing split identifiers, full target-index coverage, optional calendar labels, multivariate names, atomic failed splits, and defensive copies.

Construct a single-split holdout result with `(1, 3)` arrays and a rolling result with `(2, 2)` arrays. Reject one-dimensional public forecast arrays.

**Step 2: Write failing long-table and grouped-metric tests**

Require prediction rows with:

```text
model, split, origin, target_time, horizon, series,
actual, forecast, error, lower, upper, valid
```

Require `metric_table(by="horizon")`, `by="origin"`, and multivariate `by="series"` to delegate calculations to `compute_metrics()` and preserve calendar labels.

**Step 3: Write failing comparison-table fairness tests**

Build two model results with one model-specific failed split. Assert:

- raw results retain all rows and failures;
- common finite masks are identical for every ranked model;
- `table` contains every canonical metric plus `n_total`, `n_common`, `coverage`, `failures`, `rank`;
- empty common samples reject ranking;
- `ranking`, `scores`, and `best_model` are deterministic.

**Step 4: Run focused tests and verify failure**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py -k "ForecastEvaluationResult or ForecastComparisonResult or common" -q
```

Expected: collection/assertion failures because the unified classes do not exist.

**Step 5: Generalise aggregation over the split tensor**

Replace OOS/backtest-specific aggregation helpers with internal functions that slice `(split, horizon[, series])` arrays and always call `compute_metrics()`. Do not copy metric formulas.

**Step 6: Implement the two dataclasses in `_results.py`**

Retain shared validation helpers already present in `_results.py`. Add:

```python
@dataclass
class ForecastEvaluationResult:
    ...

@dataclass
class ForecastComparisonResult:
    results: dict[str, ForecastEvaluationResult]
    rank_by: str = "rmse"
```

Implement common-mask scoring in the comparison object. A single-model mapping follows the same path as multiple models.

**Step 7: Run result tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py -q
```

Expected: new unified result tests PASS. Legacy-result tests may still pass until Task 6 removes old classes.

**Step 8: Commit**

```powershell
git add -- TsMetrics/_results.py TsMetrics/_aggregation.py TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py
git commit -m "feat: unify forecast evaluation results"
```

### Task 4: Build the one evaluation engine and forward fit options

**Files:**
- Create: `TsMetrics/_engine.py`
- Modify: `TsMetrics/_evaluation.py`
- Test: `TsMetrics/tests/test_evaluation.py`
- Test: `TsMetrics/tests/test_contracts.py`

**Step 1: Write failing fixed-holdout engine tests**

Call:

```python
report = evaluate_forecasts(
    {"weak": weak, "strong": strong},
    scheme=Holdout(train=(0, 9), test=(10, 14)),
)
```

Assert one split per model, shared actual values/indices, complete table, ranking, and unchanged original estimators.

**Step 2: Write failing rolling-origin engine tests**

Assert exact expanding and rolling training windows, `gap`, full future exog/date alignment, target arrays, interval shapes, and original-estimator non-mutation.

**Step 3: Write failing fit-kwargs prevalidation tests**

Use recording dummy estimators and assert:

```python
fit_kwargs={"method": "lbfgs", "maxiter": 500}
```

reaches every clone at every split. If any named model does not accept a requested keyword, reject the full batch before the first fit and name the incompatible model and keyword.

Replace `validate_fit_method()` with a general `validate_fit_kwargs()` that inspects explicit parameters or `**kwargs`.

**Step 4: Write failing exogenous-information tests**

For any model exposing non-`None` `exog`:

- omitted `future_exog` fails before fitting;
- `future_exog="observed"` permits evaluation;
- unsupported strings/types fail;
- non-exogenous model batches reject an irrelevant non-`None` policy only if it cannot be applied consistently;
- result metadata records the conditional-information choice.

**Step 5: Write failing atomic error tests**

Verify `on_error="raise"` stops at the first exception and `"record"` stores one failure per complete split with all forecast/actual outputs for that model/split set to NaN. Confirm comparison metrics use only the intersection of finite pairs.

**Step 6: Run engine tests and verify failure**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py -k "evaluate_forecasts or fit_kwargs or future_exog" -q
```

Expected: FAIL because `evaluate_forecasts()` and general fit validation do not exist.

**Step 7: Implement `evaluate_forecasts()`**

Reuse the current helpers for model data, dates, series names, training exog/dates, prediction normalisation, and actual-target transformation. The engine loop must consume `ForecastSplit` records only; it must not branch on `Holdout` versus `RollingOrigin` while fitting.

Adjust `fit_and_forecast()` to receive validated `fit_kwargs` and explicit train/target positions. Preserve model-specific `_evaluation_predict_kwargs()` and `_evaluation_actual()` hooks.

**Step 8: Run engine and model-integration tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py TsModels/tests/test_evaluation.py -q
```

Expected: unified tests PASS; remaining legacy API tests are migrated in Task 6.

**Step 9: Commit**

```powershell
git add -- TsMetrics/_engine.py TsMetrics/_evaluation.py TsMetrics/tests/test_evaluation.py TsMetrics/tests/test_contracts.py TsModels/tests/test_evaluation.py
git commit -m "feat: add unified forecast evaluation engine"
```

### Task 5: Add unified tables and plots through TsPlots

**Files:**
- Modify: `TsMetrics/_results.py`
- Test: `TsMetrics/tests/test_evaluation.py`

**Step 1: Write failing table/plot tests**

Cover:

- single- and multi-model `table` order;
- `predictions` long-table order and defensive copies;
- `metric_table(by=...)` invalid dimensions and multivariate series selection;
- `plot_forecasts()` for one selected horizon when rolling paths overlap;
- `plot_metric("rmse", by="origin")`;
- titles, axis labels, notes, grid, colours and returned `(fig, ax)`.

Monkeypatch or inspect axes to prove both plot methods call the public `TsPlots.plot_series()` style path rather than a parallel plotting implementation.

**Step 2: Run plot tests and verify failure**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py -k "plot or metric_table or predictions" -q
```

Expected: FAIL for missing unified helpers.

**Step 3: Implement table and plot composition**

Build frames from the canonical prediction table. Call `Ts.TsPlots.plot_series()` for lines and use existing palette/style helpers only where intervals require matching fills. Never reimplement tick frequency, labels, notes, dual axes, or grid styling.

For rolling forecasts, require a selected `horizon` in `plot_forecasts()` whenever multiple forecasts exist for the same target label; raise an actionable error instead of silently drawing ambiguous duplicate paths.

**Step 4: Run result and TsPlots tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsPlots/tests/test_plots.py -q
```

Expected: all tests PASS.

**Step 5: Commit**

```powershell
git add -- TsMetrics/_results.py TsMetrics/tests/test_evaluation.py
git commit -m "feat: report unified forecast comparisons"
```

### Task 6: Remove legacy APIs and migrate every repository consumer

**Files:**
- Delete: `TsMetrics/_oos.py`
- Delete: `TsMetrics/_backtest.py`
- Delete: `TsMetrics/_compare.py`
- Modify: `TsMetrics/__init__.py`
- Modify: `__init__.py`
- Modify: `TsModels/_base.py`
- Modify: `TsMetrics/README.md`
- Modify: `TsModels/README.md`
- Modify: `TsMetrics/tests/test_contracts.py`
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsModels/tests/test_evaluation.py`
- Modify: `TsModels/tests/test_evaluation_periods.py`
- Modify: `TsModels/tests/test_sarimax.py`
- Modify: `TsModels/tests/test_sarimax_exog.py`
- Modify: `TsModels/tests/test_distributed_lag.py`
- Modify: `TsModels/tests/test_garch.py`
- Modify: `TsModels/tests/test_var.py`

**Step 1: Write the final public-export contract tests**

Assert `Ts.TsMetrics` and root `Ts` export only:

```text
Holdout, RollingOrigin, ForecastEvaluationResult,
ForecastComparisonResult, evaluate_forecasts
```

for forecast evaluation. Assert old names are absent. Assert `BaseModel` has no `oos` or `backtest` attribute.

**Step 2: Replace every live repository call**

Use `evaluate_forecasts({"model": model}, scheme=...)` for single-model tests and named mappings for comparisons. Retrieve the per-model result with `report.results["model"]` where array-level assertions are needed.

Do not edit historical `docs/plans/` files.

**Step 3: Delete old implementations and result classes**

Remove old imports, `__all__` entries, convenience methods, old result dataclasses, and private modules. Do not leave forwarding functions, aliases, `__getattr__` fallbacks, or deprecation shims.

**Step 4: Rewrite documentation around the one interface**

Both README files must include:

- complete signatures and parameter meanings;
- return types and field/method access;
- Holdout, expanding, rolling, non-overlapping block, gap and one-step examples;
- the user's RDL-versus-ARIMA pattern using `fit_kwargs`;
- overlapping-error dependence;
- conditional observed-future-exog warning;
- failure/common-sample ranking behavior;
- distinction from `TsModels.compare_models()` parameter comparison.

Update root Quick Start and all public docstrings. The checkout contains no demo notebook, so do not invent one; runnable README/docstring examples are the durable examples for this repository state.

**Step 5: Search for forbidden live references**

Run:

```powershell
rg -n -S "evaluate_models_oos|compare_forecasts|OOSResult|BacktestResult|OOSComparisonResult|ComparisonResult|\.oos\(|\.backtest\(" . -g '!docs/plans/**' -g '!build/**' -g '!Ts.egg-info/**'
```

Expected: no matches outside intentionally worded migration-history tests, which should also be avoided where possible.

**Step 6: Run all forecast/model tests**

Run:

```powershell
python -m pytest TsMetrics/tests TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py TsModels/tests/test_sarimax.py TsModels/tests/test_sarimax_exog.py TsModels/tests/test_distributed_lag.py TsModels/tests/test_garch.py TsModels/tests/test_var.py -q
```

Expected: all selected tests PASS.

**Step 7: Commit**

```powershell
git add -A -- TsMetrics TsModels __init__.py
git commit -m "refactor: replace legacy forecast evaluation APIs"
```

### Task 7: Validate behavior, documentation, and the entire repository

**Files:**
- Modify only if verification exposes a defect within the approved design.

**Step 1: Run direct smoke examples**

Run one synthetic Holdout comparison, one expanding comparison, one fixed rolling comparison, one gapped comparison, and one RDL `horizon=1` comparison. Assert ranking tables, grouped metrics, prediction rows, plots, and original-model non-mutation.

**Step 2: Run the complete test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all repository tests PASS.

**Step 3: Run static and artifact checks**

Run:

```powershell
python -m ruff check .
python -m compileall -q TsMetrics TsModels TsPlots TsSims TsTests TsUtils
git diff --check
```

Expected: all commands exit zero.

**Step 4: Audit the final plan-to-code boundary**

Inspect:

```powershell
git status --short
git diff --stat HEAD~6..HEAD
git log --oneline -8
```

Confirm:

- no branch was created;
- no old public name or fallback survives;
- no metric, plotting, validation, or model-cloning logic was duplicated;
- every new public object is exported, documented, and tested;
- no unrelated file was changed;
- historical design/implementation plans remain unchanged.

**Step 5: Commit any verification-only corrections**

If and only if checks required in-scope corrections:

```powershell
git add -- <verified in-scope files>
git commit -m "test: complete forecast evaluation migration"
```

Otherwise leave the clean sequence of task commits unchanged.
