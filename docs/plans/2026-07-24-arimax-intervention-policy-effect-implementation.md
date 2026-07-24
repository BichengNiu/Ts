# ARIMAX Intervention and Policy Effect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `SARIMA` into a date-aware ARIMAX implementation with automatic future exogenous data, multiple forecast scenarios, event interventions, policy-effect inference, and leakage-free exogenous OOS/backtesting.

**Architecture:** Keep model estimation and forecast orchestration in `TsModels/_sarima.py`. Put immutable event specifications, date mapping, event design matrices, policy-effect result objects, and uncertainty engines in `TsModels/_intervention.py`. Extend the `BaseModel`/`TsMetrics` evaluation protocol only with generic training-date and future-prediction context hooks, so metrics code never needs to understand event columns.

**Tech Stack:** Python 3.13, NumPy, pandas, SciPy, statsmodels 0.14.x, Matplotlib, pytest, Ruff.

**Approved design:** `docs/plans/2026-07-24-arimax-intervention-policy-effect-design.md`

**Execution skills:** Use `@executing-plans` for task-by-task execution. After Task 10 passes, use `@code-simplifier` for Task 11 without changing public behavior.

---

## Global implementation rules

- Work test-first and commit after every green task.
- Do not add deprecated aliases, permissive fallbacks, silent sorting, implicit filling, or event-column overrides.
- Keep one canonical implementation for date alignment and one canonical implementation for event design matrices.
- Preserve `SARIMA(data, order=...)` as the no-exogenous-variable case, not as a compatibility shim.
- Run commands from `C:\Users\NIU\Desktop\Ts`.
- If imports fail because the package parent is absent from `sys.path`, run the same command with:

  ```powershell
  $env:PYTHONPATH = (Resolve-Path '..').Path
  ```

- Before Task 1, record the baseline:

  ```powershell
  python -m pytest TsModels/tests/test_sarima.py TsMetrics/tests/test_evaluation.py -p no:cacheprovider -q
  python -m ruff check TsModels TsMetrics
  git status --short
  ```

  Expected: existing focused tests and Ruff pass; working tree contains only this uncommitted plan if it has not yet been committed.

---

### Task 1: Define and validate event specifications

**Files:**

- Create: `TsModels/_intervention.py`
- Create: `TsModels/tests/test_intervention.py`

**Step 1: Write failing `EventSpec` tests**

Add tests equivalent to:

```python
import pytest

from Ts.TsModels._intervention import EventSpec


def test_event_spec_defaults_to_period_mapping():
    event = EventSpec(name="policy", dates=["2025-03-15"], kind="pulse")
    assert event.date_rule == "period"
    assert event.window is None
    assert event.reference is None


@pytest.mark.parametrize("kind", ["other", "", None])
def test_event_spec_rejects_unknown_kind(kind):
    with pytest.raises(ValueError, match="kind"):
        EventSpec(name="policy", dates=["2025-03-15"], kind=kind)


def test_step_rejects_dynamic_window():
    with pytest.raises(ValueError, match="window.*pulse"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="step",
            window=(-2, 4),
            reference=-1,
        )


def test_window_requires_reference_inside_window():
    with pytest.raises(ValueError, match="reference"):
        EventSpec(
            name="policy",
            dates=["2025-03-15"],
            kind="pulse",
            window=(-2, 4),
            reference=5,
        )
```

Also cover:

- empty/whitespace event names;
- empty date lists;
- duplicate dates within one event;
- invalid `date_rule`;
- non-integer or reversed windows;
- `reference` without `window`;
- `window` without `reference`;
- `reference=0`, which is valid and removes the contemporaneous column.

**Step 2: Run tests and verify failure**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: collection fails because `TsModels._intervention` does not exist.

**Step 3: Implement the immutable contract**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

DateRule = Literal["exact", "period", "next", "previous"]
EventKind = Literal["pulse", "step"]


@dataclass(frozen=True)
class EventSpec:
    name: str
    dates: Sequence[object]
    kind: EventKind
    window: tuple[int, int] | None = None
    reference: int | None = None
    date_rule: DateRule = "period"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("event name must be a non-empty string")
        if self.kind not in {"pulse", "step"}:
            raise ValueError("event kind must be 'pulse' or 'step'")
        if self.date_rule not in {"exact", "period", "next", "previous"}:
            raise ValueError(
                "date_rule must be 'exact', 'period', 'next', or 'previous'"
            )
        parsed = tuple(pd.Timestamp(value) for value in self.dates)
        if not parsed:
            raise ValueError("event dates must not be empty")
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"event {name!r} contains duplicate dates")
        if self.kind == "step" and self.window is not None:
            raise ValueError("window is only valid for pulse events")
        if (self.window is None) != (self.reference is None):
            raise ValueError("window and reference must be specified together")
        if self.window is not None:
            start, end = self.window
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start > end
            ):
                raise ValueError("window must be an ordered pair of integers")
            if self.reference not in range(start, end + 1):
                raise ValueError("reference must lie inside window")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dates", parsed)
```

Use explicit checks rather than coercing an invalid window or rule.

**Step 4: Run tests**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: all Task 1 tests pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_intervention.py TsModels/tests/test_intervention.py
git commit -m "feat: define intervention event contract"
```

---

### Task 2: Implement event-date mapping and design matrices

**Files:**

- Modify: `TsModels/_intervention.py`
- Modify: `TsModels/tests/test_intervention.py`

**Step 1: Write failing mapping and coding tests**

Use a monthly-start index and assert:

```python
import numpy as np
import pandas as pd

from Ts.TsModels._intervention import EventSpec, build_event_matrix


def test_period_maps_within_month_to_monthly_observation():
    dates = pd.date_range("2025-01-01", periods=5, freq="MS")
    matrix, metadata = build_event_matrix(
        dates,
        [EventSpec("policy", ["2025-03-15"], "pulse")],
    )
    assert matrix["event__policy"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert metadata["policy"].columns == ("event__policy",)


def test_repeated_step_dates_create_cumulative_staircase():
    dates = pd.date_range("2025-01-01", periods=6, freq="MS")
    matrix, _ = build_event_matrix(
        dates,
        [
            EventSpec(
                "policy",
                ["2025-02-10", "2025-04-20"],
                "step",
            )
        ],
    )
    assert matrix["event__policy"].tolist() == [0, 1, 1, 2, 2, 2]


def test_event_window_excludes_reference_and_counts_overlap():
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    event = EventSpec(
        "announcement",
        ["2025-01-03", "2025-01-04"],
        "pulse",
        window=(-1, 2),
        reference=-1,
        date_rule="exact",
    )
    matrix, metadata = build_event_matrix(dates, [event])
    assert "event__announcement__m1" not in matrix
    assert matrix.loc["2025-01-04", "event__announcement__p1"] == 1.0
    assert matrix.loc["2025-01-04", "event__announcement__p0"] == 1.0
    assert metadata["announcement"].relative_periods == (0, 1, 2)


def test_step_before_target_slice_remains_active():
    calendar = pd.date_range("2025-01-01", periods=6, freq="MS")
    target = calendar[3:]
    matrix, _ = build_event_matrix(
        target,
        [EventSpec("policy", ["2025-02-15"], "step")],
        calendar=calendar,
    )
    assert matrix["event__policy"].tolist() == [1.0, 1.0, 1.0]
```

Add parametrized tests for:

- `exact`, `next`, and `previous`;
- monthly-start and monthly-end `period`;
- quarterly and daily `period`;
- an irregular index failing under `period`;
- events before and after the target slice projected against a full calendar;
- an event outside the current calendar contributing zero until extension;
- an `exact` event inside calendar bounds but absent from the calendar failing;
- duplicate event names;
- collision with a supplied ordinary exogenous column name;
- timezone mismatch.

**Step 2: Verify tests fail**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: failures because `build_event_matrix` and metadata do not exist.

**Step 3: Implement one canonical mapping path**

Add:

```python
@dataclass(frozen=True)
class EventColumns:
    name: str
    columns: tuple[str, ...]
    relative_periods: tuple[int, ...] | None
    mapped_positions: tuple[int, ...]


def build_event_matrix(
    target_dates: pd.DatetimeIndex,
    events: Sequence[EventSpec],
    *,
    calendar: pd.DatetimeIndex | None = None,
    reserved_names: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, EventColumns]]:
    ...
```

Implementation requirements:

1. Normalize `target_dates` and `calendar` once with
   `_validate_datetime_index`; default `calendar` to `target_dates`.
2. Require every target date to occur exactly once in the calendar.
3. Map each event timestamp against the complete calendar with
   `_map_event_position`.
4. For `period`, use the inferred/fixed pandas offset and a normalized period
   alias; do not emulate `period` with unconditional nearest-neighbour matching.
5. Generate column names with:

   ```python
   def _relative_suffix(relative: int) -> str:
       if relative < 0:
           return f"m{abs(relative)}"
       if relative > 0:
           return f"p{relative}"
       return "p0"
   ```

6. Project mapped calendar positions into `target_dates` and add with `+= 1.0`,
   preserving overlap counts.
7. For a step event, count every mapped occurrence at or before each target
   date, so events before the target slice remain active.
8. An event outside calendar bounds contributes zero for that calendar. An
   event inside calendar bounds that cannot satisfy its rule fails explicitly.
9. Reject duplicate event names and reserved-name collisions here. Reject
   all-zero event columns only when validating the fitting design in Task 4,
   because a forecast slice may legitimately contain no occurrence.
10. Return columns in event-list order and relative-period order.

Keep `_map_event_position` private and test only through `build_event_matrix`,
except where a direct test is necessary to distinguish date rules.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
python -m ruff check TsModels/_intervention.py TsModels/tests/test_intervention.py
```

Expected: all pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_intervention.py TsModels/tests/test_intervention.py
git commit -m "feat: build dated intervention matrices"
```

---

### Task 3: Normalize dated endogenous and ordinary exogenous data

**Files:**

- Modify: `TsModels/_sarima.py:1-460`
- Create: `TsModels/tests/test_sarima_exog.py`

**Step 1: Write failing construction tests**

Add:

```python
import numpy as np
import pandas as pd
import pytest

from Ts.TsModels import SARIMA


def test_dataframe_exog_splits_history_and_default_future():
    y_dates = pd.date_range("2025-01-01", periods=12, freq="MS")
    all_dates = pd.date_range("2025-01-01", periods=15, freq="MS")
    y = pd.Series(np.arange(12.0), index=y_dates)
    exog = pd.DataFrame({"price": np.arange(15.0)}, index=all_dates)

    model = SARIMA(y, exog=exog, order=(0, 0, 0), trend="n")

    assert model.exog.shape == (12, 1)
    assert model.exog_names == ("price",)
    assert model.dates.equals(y_dates)
    assert model.future_exog.index.equals(all_dates[12:])


def test_ndarray_exog_requires_names_and_equal_length():
    with pytest.raises(ValueError, match="exog_names"):
        SARIMA(np.arange(12.0), exog=np.ones((12, 1)))
    with pytest.raises(ValueError, match="12 observations"):
        SARIMA(
            np.arange(12.0),
            exog=np.ones((11, 1)),
            exog_names=["x"],
        )


def test_missing_drop_removes_y_exog_and_dates_jointly():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    y = pd.Series(np.arange(12.0), index=dates)
    x = pd.DataFrame({"x": np.arange(12.0)}, index=dates)
    y.iloc[2] = np.nan
    x.iloc[4, 0] = np.nan

    model = SARIMA(y, exog=x, missing="drop")

    assert len(model.data) == 10
    assert dates[2] not in model.dates
    assert dates[4] not in model.dates
    assert model.exog.shape == (10, 1)
```

Also test:

- explicit `dates=` with ndarray `data`;
- a Series plus conflicting `dates=` fails;
- duplicate, descending, and timezone-inconsistent dates fail;
- a DataFrame missing one historical date fails and names that date;
- future DataFrame rows must begin after the last `y` date;
- nonnumeric or infinite exog fails;
- `missing="raise"` is the default;
- unknown missing policy fails;
- one-dimensional exog is rejected rather than reshaped silently;
- no exog/no dates still works.

**Step 2: Verify tests fail**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
```

Expected: `SARIMA.__init__` rejects new keywords or fails new assertions.

**Step 3: Add a private normalized input object**

In `TsModels/_sarima.py`, add:

```python
@dataclass(frozen=True)
class _SARIMAInputs:
    endog: np.ndarray
    dates: pd.DatetimeIndex | None
    exog: np.ndarray | None
    exog_names: tuple[str, ...]
    future_exog: pd.DataFrame | None


def _normalise_sarima_inputs(
    data,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    missing="raise",
) -> _SARIMAInputs:
    ...
```

Implement these exact invariants:

- pandas Series supplies the authoritative index;
- ndarray data must be one-dimensional before conversion;
- DataFrame exog aligns by exact historical labels, never by row position;
- array exog requires a two-dimensional exact-length matrix and unique
  `exog_names`;
- the joint finite mask is applied only for `missing="drop"`;
- copied float arrays are stored so later caller mutation cannot alter a model;
- the future DataFrame retains its date index and the same ordered columns.

Update the constructor signature to:

```python
def __init__(
    self,
    data,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    events=None,
    missing="raise",
):
```

Store `events` without building them yet; Task 4 integrates event design.

**Step 4: Run focused and regression tests**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py TsModels/tests/test_sarima.py -p no:cacheprovider -q
python -m ruff check TsModels/_sarima.py TsModels/tests/test_sarima_exog.py
```

Expected: new normalization tests and existing SARIMA tests pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_sarima.py TsModels/tests/test_sarima_exog.py
git commit -m "feat: normalize dated ARIMAX inputs"
```

---

### Task 4: Fit ARIMAX with ordinary and event regressors

**Files:**

- Modify: `TsModels/_sarima.py`
- Modify: `TsModels/tests/test_sarima_exog.py`
- Modify: `TsModels/tests/test_intervention.py`

**Step 1: Write failing estimation tests**

Create deterministic synthetic fixtures:

```python
def arimax_fixture(seed=42, n=300):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    error = np.empty(n)
    error[0] = rng.normal()
    for index in range(1, n):
        error[index] = 0.45 * error[index - 1] + rng.normal(scale=0.4)
    y = 1.75 * x + error
    return (
        pd.Series(y, index=dates),
        pd.DataFrame({"x": x}, index=dates),
    )


def test_arimax_recovers_exogenous_coefficient():
    y, x = arimax_fixture()
    result = SARIMA(
        y,
        exog=x,
        order=(1, 0, 0),
        trend="n",
    ).fit()
    assert result.params["x"] == pytest.approx(1.75, abs=0.15)
```

Add tests that:

- a pulse coefficient is recovered from synthetic data;
- a cumulative step coefficient is recovered;
- a window omits the reference column and fits all remaining relative columns;
- `result.exog_names` and `result.event_names` retain the public names;
- duplicate ordinary/event columns and rank-deficient designs fail before
  statsmodels fitting;
- an all-zero event column fails;
- adding `trend="c"` with an explicit constant exog fails clearly.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: coefficient/event assertions fail because `SARIMAX` does not yet
receive the combined design.

**Step 3: Build and validate the combined design once**

In `SARIMA.__init__`:

```python
event_frame, event_metadata = build_event_matrix(
    inputs.dates,
    events,
    reserved_names=inputs.exog_names,
)
combined_exog = _combine_design(inputs.exog, event_frame)
_validate_design_matrix(
    combined_exog,
    column_names=inputs.exog_names + tuple(event_frame.columns),
    trend=trend,
)
```

Require dated data when `events` is not empty. Pass `combined_exog` to:

```python
SARIMAX(
    self.data,
    exog=self.design_matrix,
    dates=self.dates,
    order=self.order,
    seasonal_order=self.seasonal_order,
    trend=self.trend,
    enforce_stationarity=self.enforce_stationarity,
    enforce_invertibility=self.enforce_invertibility,
)
```

Store in `SARIMAResult`:

```python
_dates: pd.DatetimeIndex | None
_ordinary_exog: np.ndarray | None
_ordinary_exog_names: tuple[str, ...]
_event_specs: tuple[EventSpec, ...]
_event_metadata: dict[str, EventColumns]
_design_columns: tuple[str, ...]
_design_matrix: np.ndarray | None
_default_future_exog: pd.DataFrame | None
_model_kwargs: dict
```

Expose read-only copied properties for dates, ordinary exog names, and event
names. Do not expose mutable internal matrices.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py TsModels/tests/test_intervention.py TsModels/tests/test_sarima.py -p no:cacheprovider -q
python -m ruff check TsModels/_sarima.py TsModels/_intervention.py
```

Expected: all pass, including existing no-exog SARIMA tests.

**Step 5: Commit**

```powershell
git add -- TsModels/_sarima.py TsModels/tests/test_sarima_exog.py TsModels/tests/test_intervention.py
git commit -m "feat: estimate SARIMA with exogenous events"
```

---

### Task 5: Add strict future-exogenous scenarios

**Files:**

- Modify: `TsModels/_sarima.py`
- Modify: `TsModels/tests/test_sarima_exog.py`

**Step 1: Write failing scenario-result tests**

Add:

```python
from Ts.TsModels import PredictResult, ScenarioForecastResult


def test_default_future_exog_returns_one_predict_result():
    model = dated_model_with_three_future_exog_rows()
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
    )
    assert isinstance(result, PredictResult)
    assert result.mean.shape == (3,)


def test_mapping_returns_default_and_named_scenarios():
    model = dated_model_with_three_future_exog_rows()
    future_dates = model.future_exog.index
    custom = {
        "high": pd.DataFrame({"x": [2.0, 2.0, 2.0]}, index=future_dates),
        "low": pd.DataFrame({"x": [-2.0, -2.0, -2.0]}, index=future_dates),
    }
    result = model.fit().predict(
        start=len(model.data),
        end=len(model.data) + 2,
        future_exog=custom,
    )
    assert isinstance(result, ScenarioForecastResult)
    assert tuple(result.scenarios) == ("default", "high", "low")
    assert result.default_name == "default"
    assert np.all(result["high"].mean > result["low"].mean)
```

Also test:

- a DataFrame/array is named `"custom"`;
- custom-only scenarios give `default_name=None`;
- `"default"` cannot appear in a user mapping;
- empty scenario mapping fails;
- scenario names must be nonempty unique strings;
- all frames have exact columns in fitted order;
- array scenarios require `future_dates`;
- wrong dates, missing dates, extra rows, NaN/inf and wrong length fail;
- an event-like column in `future_exog` fails;
- a request beyond stored/default coverage lists missing dates;
- caller mutation after prediction does not alter result data;
- `ScenarioForecastResult` validates identical dates and lengths;
- `result["unknown"]` raises `KeyError`;
- `plot()` returns `(fig, ax)` and labels all scenarios.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
```

Expected: missing class/new keyword failures.

**Step 3: Implement the result and normalizer**

Add:

```python
@dataclass
class ScenarioForecastResult:
    scenarios: dict[str, PredictResult]
    default_name: str | None
    dates: pd.DatetimeIndex | None

    def __post_init__(self) -> None:
        ...

    def __getitem__(self, name: str) -> PredictResult:
        return self.scenarios[name]

    def summary(self) -> str:
        ...

    def plot(self, title=None):
        ...
```

Add one private normalizer:

```python
def _normalise_future_scenarios(
    future_exog,
    *,
    future_dates,
    expected_dates,
    exog_names,
    default_future_exog,
) -> tuple[dict[str, pd.DataFrame], str | None]:
    ...
```

This function is the only place that interprets DataFrame, array, or mapping
scenario inputs. It must return copied DataFrames in deterministic insertion
order and never accept generated event columns.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
python -m ruff check TsModels/_sarima.py TsModels/tests/test_sarima_exog.py
```

Expected: all pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_sarima.py TsModels/tests/test_sarima_exog.py
git commit -m "feat: add ARIMAX forecast scenarios"
```

---

### Task 6: Make prediction date-aware and scenario-aware

**Files:**

- Modify: `TsModels/_sarima.py:60-159`
- Modify: `TsModels/tests/test_sarima.py`
- Modify: `TsModels/tests/test_sarima_exog.py`

**Step 1: Write failing prediction tests**

Test:

```python
def test_predict_accepts_date_bounds_and_generates_future_events():
    result = fitted_dated_event_model()
    prediction = result.predict(
        start="2025-10-01",
        end="2026-03-01",
    )
    assert prediction.mean.shape == (6,)
    assert prediction.is_oos.tolist() == [False, False, False, True, True, True]


def test_exogenous_forecast_without_coverage_fails():
    result = fitted_model_without_future_exog()
    with pytest.raises(ValueError, match="future exog.*2026-01-01"):
        result.predict(start=result.nobs, end=result.nobs + 1)
```

Add mixed in-sample/out-of-sample tests that compare each scenario against a
direct statsmodels `get_prediction`/`get_forecast` call with the exact combined
design matrix. Also test `dynamic`, alpha validation, skipped early future
periods, inferred future dates, and mandatory `future_dates` on irregular data.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_sarima.py TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
```

Expected: date bounds and future exog paths fail.

**Step 3: Refactor prediction into one-scenario primitive**

Change the public signature:

```python
def predict(
    self,
    start=0,
    end=None,
    dynamic=False,
    alpha=0.05,
    *,
    future_exog=None,
    future_dates=None,
):
    ...
```

Extract:

```python
def _predict_one(
    self,
    window,
    *,
    dynamic,
    alpha,
    future_ordinary_exog,
    prediction_dates,
) -> PredictResult:
    ...
```

Rules:

- date bounds resolve to integer positions before calling the existing
  `_resolve_prediction_window`;
- in-sample prediction uses fitted design already stored by statsmodels;
- out-of-sample prediction combines normalized ordinary exog with event columns
  built with `target_dates=prediction_dates` and a calendar containing all
  historical dates plus the complete required future range, preserving step
  carryover and future event timing;
- `get_forecast(steps=..., exog=combined_future_design)` receives all skipped
  earlier future rows, then the output is sliced with `forecast_skip`;
- each scenario calls `_predict_one`; no fit is repeated;
- one scenario returns its `PredictResult`, multiple scenarios return
  `ScenarioForecastResult`;
- `PredictResult` keeps its existing numeric positional plotting contract;
  `ScenarioForecastResult.dates` carries explicit date labels.

Remove the broad `except Exception: pass` around full-sample intervals. Catch
only documented statsmodels interval exceptions if a real failure case exists;
otherwise let the error surface.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_sarima.py TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
python -m ruff check TsModels/_sarima.py
```

Expected: all existing and new prediction tests pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_sarima.py TsModels/tests/test_sarima.py TsModels/tests/test_sarima_exog.py
git commit -m "feat: forecast dated ARIMAX scenarios"
```

---

### Task 7: Implement policy-effect point estimates and result contract

**Files:**

- Modify: `TsModels/_intervention.py`
- Modify: `TsModels/_sarima.py`
- Modify: `TsModels/tests/test_intervention.py`

**Step 1: Write failing effect-contract tests**

Use a fitted event-window model and assert:

```python
from Ts.TsModels import PolicyEffectResult


def test_policy_effect_is_event_design_contrast():
    fitted = fitted_policy_model()
    effect = fitted.policy_effect(
        events="policy",
        start="2025-01-01",
        end="2025-12-01",
        method="delta",
    )
    metadata = fitted._event_metadata["policy"]
    positions = [fitted._design_columns.index(name) for name in metadata.columns]
    beta = np.asarray(fitted._statsmodels_result.params)[positions]
    event_design = fitted._design_matrix[:, positions]
    expected = event_design @ beta

    assert isinstance(effect, PolicyEffectResult)
    np.testing.assert_allclose(effect.effect.to_numpy(), expected)
    np.testing.assert_allclose(
        effect.factual_mean - effect.counterfactual_mean,
        effect.effect,
    )
    assert effect.cumulative_effect == pytest.approx(expected.sum())
    assert "因果" in effect.identification_note
```

Also test:

- one event name and a list of event names;
- unknown or duplicate requested names;
- a date interval outside the prediction range;
- factual/counterfactual series share the same index;
- nonselected events remain in both paths;
- a cumulative step gives the expected increasing contrast;
- `summary()` contains coefficients, cumulative effect, method, and the
  identification note;
- `plot()` returns `(fig, ax)`.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: missing `PolicyEffectResult` and `policy_effect`.

**Step 3: Implement selection and contrast**

Add:

```python
@dataclass
class PolicyEffectResult:
    coefficients: pd.DataFrame
    factual_mean: pd.Series
    counterfactual_mean: pd.Series
    effect: pd.Series
    lower: pd.Series
    upper: pd.Series
    cumulative_effect: float
    cumulative_lower: float
    cumulative_upper: float
    pretrend_test: dict | None
    method: str
    identification_note: str

    def __post_init__(self) -> None:
        ...

    def summary(self) -> str:
        ...

    def plot(self, title=None):
        ...
```

Add the engine:

```python
def estimate_policy_effect(
    result,
    *,
    events,
    start,
    end,
    method,
    alpha,
    n_draws,
    seed,
) -> PolicyEffectResult:
    ...
```

Point-estimate implementation:

```python
contrast = selected_event_design
beta = full_parameter_vector[selected_parameter_positions]
effect = contrast @ beta
factual = result.predict(start=start, end=end).mean
counterfactual = factual - effect
```

For out-of-sample dates, construct the selected event design for the requested
date range with the same canonical `build_event_matrix` function and the same
historical-plus-future calendar used by prediction. Ordinary exogenous variables
and nonselected event columns remain unchanged and cancel from the contrast.

Add the thin delegation:

```python
def policy_effect(
    self,
    events,
    *,
    start=0,
    end=None,
    method="simulation",
    alpha=0.05,
    n_draws=2_000,
    seed=None,
):
    return estimate_policy_effect(...)
```

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_intervention.py TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
python -m ruff check TsModels/_intervention.py TsModels/_sarima.py
```

Expected: point-estimate and result-contract tests pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_intervention.py TsModels/_sarima.py TsModels/tests/test_intervention.py
git commit -m "feat: estimate conditional policy effects"
```

---

### Task 8: Add delta, parameter simulation, and pretrend inference

**Files:**

- Modify: `TsModels/_intervention.py`
- Modify: `TsModels/tests/test_intervention.py`

**Step 1: Write failing uncertainty tests**

Add direct numerical tests using a known contrast and covariance matrix:

```python
def test_delta_interval_uses_full_event_covariance():
    contrast = np.array([[1.0, 2.0]])
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    standard_error = _contrast_standard_errors(contrast, covariance)
    assert standard_error[0] == pytest.approx(np.sqrt(44.0))


def test_simulation_is_reproducible_and_keeps_joint_covariance():
    first = fitted_policy_model().policy_effect(
        "policy", method="simulation", n_draws=500, seed=123
    )
    second = fitted_policy_model().policy_effect(
        "policy", method="simulation", n_draws=500, seed=123
    )
    pd.testing.assert_series_equal(first.lower, second.lower)
    pd.testing.assert_series_equal(first.upper, second.upper)
    assert first.cumulative_lower == second.cumulative_lower


def test_event_leads_receive_joint_wald_pretrend_test():
    effect = fitted_window_model().policy_effect(
        "announcement", method="delta"
    )
    assert effect.pretrend_test["df"] == 2
    assert 0.0 <= effect.pretrend_test["p_value"] <= 1.0
```

Also test:

- `alpha`, `n_draws`, and `seed` validation;
- parameter positions are taken by exact statsmodels names;
- cumulative delta variance uses the summed contrast, not the sum of marginal
  standard errors;
- empirical simulation bounds use `alpha / 2` quantiles;
- no lead columns gives `pretrend_test=None`;
- reference-period coefficient is shown as fixed zero in the coefficient table.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: interval and pretrend assertions fail.

**Step 3: Implement exact inference helpers**

Add:

```python
def _contrast_standard_errors(
    contrast: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    variances = np.einsum("ij,jk,ik->i", contrast, covariance, contrast)
    return np.sqrt(np.clip(variances, 0.0, None))


def _delta_intervals(...):
    ...


def _simulation_intervals(...):
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(beta, covariance, size=n_draws)
    paths = draws @ contrast.T
    ...


def _pretrend_wald_test(...):
    ...
```

Use `scipy.stats.norm` for delta intervals and `scipy.stats.chi2.sf` for the
joint Wald p value. Always subset the joint covariance matrix by exact selected
parameter positions; do not assume event parameters are contiguous.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_intervention.py -p no:cacheprovider -q
python -m ruff check TsModels/_intervention.py
```

Expected: all delta, simulation, and pretrend tests pass.

**Step 5: Commit**

```powershell
git add -- TsModels/_intervention.py TsModels/tests/test_intervention.py
git commit -m "feat: quantify intervention uncertainty"
```

---

### Task 9: Add parameterized bootstrap with a hard success threshold

**Files:**

- Modify: `TsModels/_intervention.py`
- Modify: `TsModels/tests/test_intervention.py`

**Step 1: Write failing bootstrap tests**

Use small, deterministic draw counts in unit tests and monkeypatch only the
statsmodels simulation/refit boundary:

```python
def test_bootstrap_is_reproducible():
    fitted = fitted_policy_model()
    first = fitted.policy_effect(
        "policy",
        method="bootstrap",
        n_draws=30,
        seed=7,
    )
    second = fitted.policy_effect(
        "policy",
        method="bootstrap",
        n_draws=30,
        seed=7,
    )
    assert first.cumulative_lower == pytest.approx(second.cumulative_lower)
    assert first.cumulative_upper == pytest.approx(second.cumulative_upper)


def test_bootstrap_rejects_less_than_eighty_percent_success(monkeypatch):
    fitted = fitted_policy_model()
    force_refit_failures(monkeypatch, failures=3, attempts=10)
    with pytest.raises(RuntimeError, match="7/10.*80%"):
        fitted.policy_effect(
            "policy",
            method="bootstrap",
            n_draws=10,
            seed=9,
        )
```

Also assert that individual failure type/message records are retained in the
raised error or a private diagnostic object, without exposing partial intervals.

**Step 2: Verify failures**

```powershell
python -m pytest TsModels/tests/test_intervention.py -k bootstrap -p no:cacheprovider -q
```

Expected: bootstrap method is not implemented.

**Step 3: Implement bootstrap**

For each draw:

1. derive a child RNG from `np.random.SeedSequence(seed).spawn(n_draws)`;
2. simulate `nobs` values from the fitted statsmodels SARIMAX specification,
   using the original full combined design matrix;
3. construct the same statsmodels `SARIMAX` specification with simulated
   endogenous data and the unchanged design;
4. fit with `disp=False`;
5. extract selected event coefficients by exact parameter name;
6. calculate the same effect contrast and cumulative effect;
7. catch fit/simulation errors per draw and record type/message.

After all attempts:

```python
success_rate = len(successful_paths) / n_draws
if success_rate < 0.80:
    raise RuntimeError(
        f"bootstrap produced {len(successful_paths)}/{n_draws} "
        "successful refits, below the required 80%"
    )
```

Use empirical quantiles only after the threshold passes. Do not retry failed
draws until the requested number of successes is reached, because that hides
the model's failure rate.

**Step 4: Run focused tests**

```powershell
python -m pytest TsModels/tests/test_intervention.py -k bootstrap -p no:cacheprovider -q
python -m ruff check TsModels/_intervention.py
```

Expected: bootstrap tests pass within a bounded runtime.

**Step 5: Commit**

```powershell
git add -- TsModels/_intervention.py TsModels/tests/test_intervention.py
git commit -m "feat: bootstrap policy effect intervals"
```

---

### Task 10: Pass future exogenous data through OOS and backtesting

**Files:**

- Modify: `TsModels/_base.py:604-616`
- Modify: `TsModels/_sarima.py`
- Modify: `TsMetrics/_common.py:82-104,166-183`
- Modify: `TsMetrics/_oos.py`
- Modify: `TsMetrics/_backtest.py`
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: `TsModels/tests/test_evaluation.py`

**Step 1: Write failing leakage and alignment tests**

Add an instrumented ARIMAX model/result or monkeypatch `SARIMA.fit` to record:

```python
def test_oos_passes_holdout_exog_without_holdout_y():
    dates = pd.date_range("2020-01-01", periods=30, freq="MS")
    x = pd.DataFrame({"x": np.arange(30.0)}, index=dates)
    y = pd.Series(2.0 * x["x"].to_numpy(), index=dates)
    model = SARIMA(y, exog=x, order=(0, 0, 0), trend="n")

    result = model.oos(
        estimation_period=(dates[0], dates[19]),
        validation_period=(dates[20], dates[29]),
    )

    assert result.mean.shape == (10,)
    assert result.metrics["rmse"] < 1e-5
    assert model.result_ is None


def test_backtest_uses_each_origins_matching_future_exog():
    ...


def test_record_mode_reports_missing_future_exog_dates():
    ...
```

The backtest test must prove:

- each cloned model receives only training `y`;
- training exog matches the same training dates;
- forecast exog matches exactly `[origin:origin+horizon]`;
- rolling windows slice dates and exog from the same `train_start`;
- events are regenerated from retained `EventSpec`;
- the original model/result is unchanged.

Add a regression test proving GARCH with exog remains explicitly unsupported
rather than receiving an unexpected keyword.

**Step 2: Verify failures**

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsModels/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: ARIMAX evaluation fails for missing future exog.

**Step 3: Extend the generic protocol**

Change `BaseModel._clone_for_evaluation` to accept and copy dates:

```python
def _clone_for_evaluation(self, data, exog=None, *, dates=None):
    cloned = copy.copy(self)
    cloned.data = np.array(data, dtype=float, copy=True)
    cloned.result_ = None
    if hasattr(cloned, "exog"):
        cloned.exog = None if exog is None else np.array(exog, dtype=float, copy=True)
    if hasattr(cloned, "dates"):
        cloned.dates = None if dates is None else dates.copy()
    return cloned
```

Add a default prediction-context hook:

```python
def _evaluation_predict_kwargs(self, start, stop):
    del start, stop
    return {}
```

Override in `SARIMA`:

```python
def _clone_for_evaluation(self, data, exog=None, *, dates=None):
    return type(self)(
        data,
        order=self.order,
        seasonal_order=self.seasonal_order,
        trend=self.trend,
        enforce_stationarity=self.enforce_stationarity,
        enforce_invertibility=self.enforce_invertibility,
        dates=dates,
        exog=exog,
        exog_names=self.exog_names if exog is not None else None,
        events=self.events,
        missing="raise",
    )


def _evaluation_predict_kwargs(self, start, stop):
    kwargs = {}
    if self.exog is not None:
        kwargs["future_exog"] = np.array(
            self.exog[start:stop], dtype=float, copy=True
        )
    if self.dates is not None:
        kwargs["future_dates"] = self.dates[start:stop].copy()
    return kwargs
```

The explicit SARIMA override must reconstruct every derived design field from
the sliced training data. It retains the full immutable event specifications,
but it must not retain the original full `y`, design matrix, or fitted result.
Events outside the training calendar contribute zero until the calendar is
extended; if a selected event has no estimable occurrence and therefore leaves
an all-zero training column, fitting fails as unidentifiable.

Update `fit_and_forecast`:

```python
def fit_and_forecast(
    model,
    train_data,
    exog,
    dates,
    predict_kwargs,
    horizon,
    alpha,
    expected_shape,
):
    cloned = model._clone_for_evaluation(
        train_data,
        exog=exog,
        dates=dates,
    )
    fitted = cloned.fit()
    prediction = fitted.predict(
        start=fitted.nobs,
        end=fitted.nobs + horizon - 1,
        alpha=alpha,
        **predict_kwargs,
    )
    return fitted, prediction_arrays(prediction, expected_shape)
```

Add `training_dates(model, start, stop)` and call
`model._evaluation_predict_kwargs(origin, origin + horizon)` in OOS/backtest.
Do not pass empty future kwargs to models that do not support them.

For `on_error="record"`, retain the existing all-NaN row behavior and include
the precise missing dates in the failure message.

**Step 4: Run evaluation and model regressions**

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsModels/tests/test_evaluation.py TsModels/tests/test_garch.py -p no:cacheprovider -q
python -m ruff check TsModels/_base.py TsModels/_sarima.py TsMetrics
```

Expected: ARIMAX OOS/backtest pass without target leakage; existing model
evaluation tests remain green.

**Step 5: Commit**

```powershell
git add -- TsModels/_base.py TsModels/_sarima.py TsMetrics/_common.py TsMetrics/_oos.py TsMetrics/_backtest.py TsMetrics/tests/test_evaluation.py TsModels/tests/test_evaluation.py
git commit -m "feat: evaluate forecasts with future exog"
```

---

### Task 11: Export, document, review, simplify, and validate

**Files:**

- Modify: `TsModels/__init__.py`
- Modify: `__init__.py`
- Modify: `TsModels/README.md`
- Modify: `TsMetrics/README.md`
- Modify: `TsModels/tests/test_sarima_exog.py`
- Modify as required by review: files changed in Tasks 1-10 only

**Step 1: Write failing public-export tests**

Add:

```python
def test_new_arimax_types_are_public():
    from Ts import EventSpec, PolicyEffectResult, ScenarioForecastResult
    from Ts.TsModels import (
        EventSpec as ModelEventSpec,
        PolicyEffectResult as ModelPolicyEffectResult,
        ScenarioForecastResult as ModelScenarioForecastResult,
    )

    assert EventSpec is ModelEventSpec
    assert PolicyEffectResult is ModelPolicyEffectResult
    assert ScenarioForecastResult is ModelScenarioForecastResult
```

**Step 2: Export types and update documentation**

Export:

- `EventSpec`;
- `PolicyEffectResult`;
- `ScenarioForecastResult`.

Update `TsModels/README.md` with executable examples for:

1. ARIMAX estimation with indexed controls;
2. automatic default future exog;
3. two named future scenarios;
4. pulse, cumulative step, and event window;
5. `policy_effect()` with all three methods;
6. causal-identification caveat.

Update `TsMetrics/README.md` to state:

- OOS/backtest uses future exog corresponding to each holdout window;
- this assumes those values were known at the forecast origin;
- no future `y` enters fitting;
- missing future exog fails or is recorded according to `on_error`.

**Step 3: Run documentation/API tests**

```powershell
python -m pytest TsModels/tests/test_sarima_exog.py -p no:cacheprovider -q
python -m ruff check TsModels/__init__.py __init__.py
```

Expected: export and documentation-related tests pass.

**Step 4: Perform systematic code review**

Review the complete diff against the approved design:

```powershell
git diff --check
git diff --stat 02e9551..HEAD
git diff 02e9551..HEAD -- TsModels TsMetrics __init__.py
```

Explicitly inspect:

- date-index ownership and caller-mutation safety;
- ordinary/event column ordering;
- statsmodels parameter-name matching;
- mixed in-sample/out-of-sample prediction slicing;
- reserved scenario names;
- covariance subsetting and cumulative variance;
- bootstrap reproducibility/failure accounting;
- OOS/backtest target leakage;
- broad exception handlers;
- duplicate normalization or event-building code;
- accidental changes outside the approved files.

Fix every verified issue with a regression test before changing code.

**Step 5: Run `@code-simplifier`**

Limit simplification to files changed by this feature. Preserve all approved
public contracts and tests. Specifically prefer:

- one future-scenario normalizer;
- one event matrix builder;
- short validation helpers with precise errors;
- no wrapper that only renames another private helper;
- no duplicated delta/simulation contrast construction;
- no comments that restate obvious code.

After simplification:

```powershell
git diff --check
python -m pytest TsModels/tests/test_intervention.py TsModels/tests/test_sarima_exog.py TsMetrics/tests/test_evaluation.py -p no:cacheprovider -q
```

Expected: focused suite passes unchanged.

**Step 6: Run full validation**

```powershell
python -m pytest -p no:cacheprovider -q
python -m ruff check .
python -m compileall -q TsModels TsMetrics
git diff --check
git status --short
```

Expected:

- full pytest suite passes;
- Ruff reports no errors;
- compileall exits 0;
- diff check exits 0;
- status contains only intended files before the final commit.

**Step 7: Commit**

```powershell
git add -- TsModels/__init__.py __init__.py TsModels/README.md TsMetrics/README.md TsModels TsMetrics
git commit -m "docs: publish ARIMAX intervention workflow"
git status --short --branch
```

Expected: the branch is clean after the commit.

---

## Final acceptance checklist

- [ ] Ordinary exogenous coefficients are estimated and exposed by name.
- [ ] pandas-indexed exog automatically stores post-`y` rows as `"default"`.
- [ ] `future_exog` supports one custom path or multiple named paths.
- [ ] Multiple scenarios never refit model parameters.
- [ ] Event columns cannot be supplied through `future_exog`.
- [ ] Pulse, cumulative step, event windows, reference periods, and four date
      rules have direct tests.
- [ ] Policy effects include coefficients, factual/counterfactual paths,
      period effects, cumulative effects, uncertainty, and identification note.
- [ ] Delta, joint parameter simulation, and parameterized bootstrap are tested.
- [ ] Bootstrap fails below 80% successful refits.
- [ ] Pre-event lead coefficients receive a joint Wald test.
- [ ] OOS/backtest passes correct future exog and never fits on holdout `y`.
- [ ] Missing or misaligned data produces precise errors with no fallback.
- [ ] Existing no-exog SARIMA and other model tests remain green.
- [ ] Full pytest, Ruff, compileall, diff check, systematic review, and
      code-simplify all pass.
