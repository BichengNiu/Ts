# Backtesting and Backcasting Implementation Plan

> **Superseded:** Backtesting and `BacktestResult` now belong to
> `TsMetrics`; backcasting belongs to `TsModels/_backcast.py`. The former
> `TsModels/_evaluation.py` architecture and its import examples are
> historical only and must not be used as current API documentation.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add leakage-free rolling/expanding backtesting and reverse-time backcasting methods to every predictive model in `TsModels`.

**Architecture:** Put the orchestration and result containers in a new `TsModels/_evaluation.py` module. Add thin `BaseModel.backtest()` and `BaseModel.backcast()` methods that delegate to this module, shallow-clone the configured model, replace only its data window, clear `result_`, and refit without mutating the caller. Use separate `BacktestResult` and `BackcastResult` containers because `PredictResult` cannot represent multiple forecast origins or negative pre-sample indices.

**Tech Stack:** Python 3.13, NumPy, dataclasses, existing `TsModels` model/result contracts, pytest, Ruff.

---

## Confirmed semantics

### `backtest`

- This is true historical forecast evaluation: every forecast origin fits a fresh model using only observations available at that origin.
- `window="expanding"` uses `data[:origin]`.
- `window="rolling"` uses `data[origin-window_size:origin]`; `window_size` defaults to `initial_window`.
- Forecast origins are:

  ```python
  range(initial_window, len(data) - horizon + 1, step)
  ```

- The result stores one row per origin and one column per forecast horizon:

  ```text
  univariate:   (n_origins, horizon)
  multivariate: (n_origins, horizon, n_series)
  ```

- Overall metrics flatten all finite evaluated predictions. `metrics_by_horizon` and `metrics_by_series` preserve the two important analytical dimensions.
- GARCH-family predictions are conditional volatility. Their observed evaluation target is the absolute return centred using the current training-window mean:

  ```python
  abs(y_future - mean(y_train))
  ```

  The result must expose `target="absolute_demeaned_return_proxy"` so this proxy is never presented as observed volatility.
- `on_error="raise"` stops at the first failed fit. `on_error="record"` stores the origin and exception message, fills that forecast row with `NaN`, and excludes it from metrics.
- The original model and its existing `result_` are unchanged.

### `backcast`

- This means reverse-time estimation, not ordinary in-sample fitted values and not causal reconstruction.
- Reverse the observed series, fit the same configured model on the reversed data, forecast `steps` beyond that reversed sample, then reverse the returned forecast so output indices are chronological:

  ```text
  indices = [-steps, ..., -2, -1]
  ```

- Confidence bounds are reversed in the same way as point forecasts.
- For GARCH-family models the output target is conditional volatility.
- Deterministic time trends are re-estimated in reverse time. This limitation must appear in the docstring and README.
- GARCH/AutoGARCH with `exog` is rejected for both methods until `predict()` can accept future/pre-sample exogenous values explicitly.

## Public API

```python
model.backtest(
    initial_window,
    horizon=1,
    step=1,
    window="expanding",
    window_size=None,
    alpha=0.05,
    on_error="raise",
) -> BacktestResult

model.backcast(
    steps,
    alpha=0.05,
) -> BackcastResult
```

Supported through `BaseModel`: `SARIMA`, `GARCH`, `VAR`, `VECM`, `SVAR`, `AutoSARIMA`, and `AutoGARCH`. `STL` is a decomposition class and does not inherit `BaseModel`, so it intentionally receives neither method.

---

### Task 1: Lock the result contracts with failing unit tests

**Files:**

- Create: `TsModels/tests/test_evaluation.py`
- Test: `TsModels/tests/test_evaluation.py`

**Step 1: Write failing construction and metric tests**

Add tests covering:

```python
from Ts.TsModels import BackcastResult, BacktestResult


def test_backtest_result_keeps_origin_and_horizon_axes():
    result = BacktestResult(
        mean=np.array([[1.0, 2.0], [3.0, 4.0]]),
        actual=np.array([[1.5, 2.5], [3.5, 4.5]]),
        lower=None,
        upper=None,
        origins=np.array([20, 21]),
        target_indices=np.array([[20, 21], [21, 22]]),
        metrics={},
        metrics_by_horizon=[],
        metrics_by_series=[],
        failures=[],
        model_type="SARIMA",
        window="expanding",
        target="observed",
    )
    assert result.mean.shape == (2, 2)
    assert result.target_indices.tolist() == [[20, 21], [21, 22]]


def test_backcast_result_uses_negative_chronological_indices():
    result = BackcastResult(
        mean=np.array([1.0, 2.0, 3.0]),
        lower=None,
        upper=None,
        indices=np.array([-3, -2, -1]),
        model_type="SARIMA",
        target="observed",
    )
    assert result.indices.tolist() == [-3, -2, -1]
```

Also test metric helpers with `NaN`, a zero-valued actual series, multivariate arrays, and all-missing rows.

**Step 2: Verify the tests fail**

Run:

```powershell
python -m pytest TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: collection fails because `BacktestResult` and `BackcastResult` do not exist.

**Step 3: Do not commit in this workspace**

This directory is not a Git repository. Record the changed-file list after each task instead of fabricating a commit. If the plan is later executed in the source Git repository, commit this task as:

```text
test: define backtesting and backcasting contracts
```

---

### Task 2: Implement result containers, validation, and model cloning

**Files:**

- Create: `TsModels/_evaluation.py`
- Modify: `TsModels/tests/test_evaluation.py`

**Step 1: Add exact result fields**

Implement:

```python
@dataclass
class BacktestResult:
    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    origins: np.ndarray
    target_indices: np.ndarray
    metrics: dict
    metrics_by_horizon: list[dict]
    metrics_by_series: list[dict]
    failures: list[dict]
    model_type: str
    window: str
    target: str


@dataclass
class BackcastResult:
    mean: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    indices: np.ndarray
    model_type: str
    target: str
```

Arrays must be copied or normalized in `__post_init__`; result construction must reject incompatible shapes rather than broadcasting silently.

**Step 2: Add argument validation**

Implement private validation that enforces:

```python
initial_window >= 10
horizon >= 1
step >= 1
window in {"expanding", "rolling"}
window_size is None or window_size >= 10
on_error in {"raise", "record"}
0.0 < alpha < 1.0
initial_window + horizon <= len(model.data)
steps >= 1
```

For rolling windows, require `window_size >= initial_window` only when the first origin would otherwise have fewer than `initial_window` observations.

**Step 3: Add a non-mutating clone helper**

Use:

```python
def _clone_with_data(model, data, exog=None):
    cloned = copy.copy(model)
    cloned.data = np.array(data, dtype=float, copy=True)
    cloned.result_ = None
    if hasattr(cloned, "exog"):
        cloned.exog = None if exog is None else np.array(exog, copy=True)
    return cloned
```

The helper must not call the public constructor because validated configuration such as restrictions, selected columns, order ranges, and model flags already live on the instance.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: result/validation/clone tests pass; orchestration tests still fail until Tasks 3–4.

**Step 5: Commit when executed in a Git repository**

```text
feat: add forecast evaluation result contracts
```

---

### Task 3: Implement leakage-free `backtest()`

**Files:**

- Modify: `TsModels/_evaluation.py`
- Modify: `TsModels/_base.py:568`
- Modify: `TsModels/tests/test_evaluation.py`

**Step 1: Write failing behavioral tests**

Use a small deterministic `BaseModel` test double whose `fit()` records its training data and whose result forecasts the training mean. Verify:

- no forecast sees observations at or after its origin;
- expanding train lengths grow by `step`;
- rolling train lengths remain `window_size`;
- `horizon > 1` produces the documented shapes and target indices;
- `step > 1` skips origins correctly;
- `on_error="record"` preserves the row and records the exception;
- an existing `model.result_` object is unchanged;
- invalid inputs raise `ValueError` with the offending argument name.

**Step 2: Add the thin public method**

Add to `BaseModel`:

```python
def backtest(
    self,
    initial_window,
    horizon=1,
    step=1,
    window="expanding",
    window_size=None,
    alpha=0.05,
    on_error="raise",
):
    """Run leakage-free rolling-origin forecast evaluation."""
    from Ts.TsModels._evaluation import backtest_model

    return backtest_model(
        self,
        initial_window=initial_window,
        horizon=horizon,
        step=step,
        window=window,
        window_size=window_size,
        alpha=alpha,
        on_error=on_error,
    )
```

**Step 3: Implement the orchestration loop**

For each origin:

1. Resolve training start from the window mode.
2. Clone the configured model with only the training slice.
3. Slice `exog` using the same rows when present.
4. Fit the clone.
5. Require the result to expose callable `predict`.
6. Forecast with:

   ```python
   prediction = fitted.predict(
       start=fitted.nobs,
       end=fitted.nobs + horizon - 1,
       alpha=alpha,
   )
   ```

7. Store the forecast against `model.data[origin:origin+horizon]`.
8. For GARCH/AutoGARCH, square neither side: compare predicted conditional volatility with `abs(y_future - mean(y_train))` and set the target metadata accordingly.
9. Compute overall, per-horizon, and per-series metrics using only finite pairs.

**Step 4: Add real-model integration tests**

Use short deterministic datasets and narrow model specifications:

- `SARIMA(...).backtest(initial_window=40, horizon=2, step=5)`
- `VAR(...).backtest(initial_window=45, horizon=2, step=5)`
- `VECM(...).backtest(initial_window=50, horizon=1, step=10)`
- `SVAR(...).backtest(initial_window=50, horizon=1, step=10)`
- `GARCH(...).backtest(initial_window=60, horizon=2, step=10)`
- `AutoSARIMA` and `AutoGARCH` with single-value search ranges to verify inherited compatibility without making tests slow.

Assertions must cover shapes, finite values, origin indices, target metadata, and non-mutation.

**Step 5: Run focused tests**

Run:

```powershell
python -m pytest TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: all backtest tests pass.

**Step 6: Commit when executed in a Git repository**

```text
feat: add leakage-free model backtesting
```

---

### Task 4: Implement reverse-time `backcast()`

**Files:**

- Modify: `TsModels/_evaluation.py`
- Modify: `TsModels/_base.py:568`
- Modify: `TsModels/tests/test_evaluation.py`

**Step 1: Write failing tests**

Verify:

- the clone receives `model.data[::-1]`;
- the returned forecast and bounds are reversed into chronological order;
- indices are exactly `np.arange(-steps, 0)`;
- 1-D and 2-D output shapes are preserved;
- the original model/result are unchanged;
- `steps=0` and invalid `alpha` fail;
- GARCH output is positive and marked `target="conditional_volatility"`;
- GARCH/AutoGARCH with `exog` raises `NotImplementedError` before fitting.

**Step 2: Add the public method**

Add to `BaseModel`:

```python
def backcast(self, steps, alpha=0.05):
    """Estimate pre-sample values by reverse-time refitting and forecasting."""
    from Ts.TsModels._evaluation import backcast_model

    return backcast_model(self, steps=steps, alpha=alpha)
```

The docstring must explicitly state that deterministic trends are re-estimated in reverse time and that this is not causal reconstruction.

**Step 3: Implement reverse-fit-forecast-reverse**

Use:

```python
reversed_model = _clone_with_data(model, model.data[::-1])
fitted = reversed_model.fit()
prediction = fitted.predict(
    start=fitted.nobs,
    end=fitted.nobs + steps - 1,
    alpha=alpha,
)
mean = np.asarray(prediction.mean)[::-1].copy()
lower = None if prediction.lower is None else np.asarray(prediction.lower)[::-1].copy()
upper = None if prediction.upper is None else np.asarray(prediction.upper)[::-1].copy()
```

Do not swap lower and upper; only reverse their time axis.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: all backtest and backcast tests pass.

**Step 5: Commit when executed in a Git repository**

```text
feat: add reverse-time model backcasting
```

---

### Task 5: Export and document the API

**Files:**

- Modify: `TsModels/__init__.py:65-110`
- Modify: `TsModels/README.md`
- Modify: `TsModels/demo.ipynb`
- Test: `TsModels/tests/test_evaluation.py`

**Step 1: Add failing public-import tests**

```python
def test_result_types_are_public():
    from Ts.TsModels import BackcastResult, BacktestResult

    assert BacktestResult.__name__ == "BacktestResult"
    assert BackcastResult.__name__ == "BackcastResult"
```

**Step 2: Export the result types**

Import both classes from `._evaluation` and add them to `TsModels.__all__`. Do not add standalone `backtest` or `backcast` functions; the requested contract is method-based.

**Step 3: Update the README**

Add:

- the exact signatures;
- expanding and rolling examples;
- result shapes and metric definitions;
- the GARCH volatility proxy warning;
- the reverse-time interpretation and deterministic-trend warning;
- the exogenous-GARCH limitation.

Use examples:

```python
model = SARIMA(y, order=(1, 0, 0))
bt = model.backtest(initial_window=80, horizon=4, step=1)
bc = model.backcast(steps=12)
```

**Step 4: Update and execute the demo notebook**

Add one deterministic SARIMA section demonstrating both methods. Execute the complete notebook with the same kernel/environment currently used by the project and verify no new execution error.

**Step 5: Commit when executed in a Git repository**

```text
docs: document backtesting and backcasting
```

---

### Task 6: Full verification and audit

**Files:**

- Verify: `TsModels/_evaluation.py`
- Verify: `TsModels/_base.py`
- Verify: `TsModels/__init__.py`
- Verify: `TsModels/tests/test_evaluation.py`
- Verify: `TsModels/README.md`
- Verify: `TsModels/demo.ipynb`

**Step 1: Run the focused suite**

```powershell
python -m pytest TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: all new tests pass.

**Step 2: Run the complete regression suite**

```powershell
python -m pytest . -p no:cacheprovider -q
```

Baseline before implementation: `600 passed, 93 warnings`.

Expected after implementation: baseline tests plus all new tests pass; no new warning category is introduced.

**Step 3: Run scoped Ruff**

The full-repository Ruff baseline is already red because existing notebooks contain malformed/incorrectly typed cells. Do not hide that pre-existing condition and do not expand this feature into notebook repair.

Run only the changed Python scope:

```powershell
ruff check TsModels/_evaluation.py TsModels/_base.py TsModels/__init__.py TsModels/tests/test_evaluation.py
```

Expected: `All checks passed!`

**Step 4: Compile changed modules**

```powershell
python -m py_compile TsModels/_evaluation.py TsModels/_base.py TsModels/__init__.py TsModels/tests/test_evaluation.py
```

Expected: exit code 0 and no output.

**Step 5: Run public API smoke tests**

From `C:\Users\NIU\Desktop`:

```powershell
python -c "from Ts.TsModels import BacktestResult, BackcastResult, SARIMA; print('evaluation-api-ok')"
```

Expected:

```text
evaluation-api-ok
```

**Step 6: Final handoff**

Report:

- exact changed files;
- new test count and full-suite result;
- scoped Ruff and compile results;
- the GARCH proxy definition;
- reverse-time backcasting limitations;
- confirmation that the original model is never mutated;
- that Git commit was skipped because this workspace has no `.git` repository.
