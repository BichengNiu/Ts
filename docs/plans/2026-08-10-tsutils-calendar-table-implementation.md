# TsUtils Calendar Table Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `calendar_table()` to reshape one regularly dated time series into a calendar-oriented pandas DataFrame that is ready for Excel or CSV export.

**Architecture:** Extract the generic time-index and frequency resolution already used by `seasonal_dummies()` into a private shared helper, then implement the new reshape as a thin pandas `reindex()` plus `unstack()` adapter. Keep file writing outside TsUtils, preserve missing values, and use natural-calendar coordinates only.

**Tech Stack:** Python 3.13, pandas 3, NumPy, pytest, Ruff

---

### Task 1: Establish the public monthly contract and shared calendar validation

**Files:**
- Create: `TsUtils/_calendar.py`
- Create: `TsUtils/_calendar_table.py`
- Create: `TsUtils/tests/test_calendar_table.py`
- Modify: `TsUtils/_seasonal_dummies.py`
- Modify: `TsUtils/__init__.py`
- Modify: `__init__.py`

**Step 1: Write failing public API, selection, and monthly-schema tests**

Add tests that:

```python
from Ts import calendar_table as root_calendar_table
from Ts.TsUtils import calendar_table


def test_public_exports_resolve_to_the_same_function():
    assert root_calendar_table is calendar_table


def test_monthly_series_has_fixed_month_rows_and_year_columns():
    index = pd.date_range("2023-11-01", periods=5, freq="MS")
    source = pd.Series([11.0, 12.0, 1.0, np.nan, 3.0], index=index)

    result = calendar_table(source)

    assert result.index.equals(pd.Index(range(1, 13), name="month"))
    assert result.columns.equals(pd.Index([2023, 2024], name="year"))
    assert result.loc[11, 2023] == 11.0
    assert pd.isna(result.loc[2, 2024])


def test_dataframe_selection_requires_col_only_when_ambiguous():
    index = pd.date_range("2024-01-01", periods=3, freq="MS")
    frame = pd.DataFrame({"sales": [1.0, 2.0, 3.0]}, index=index)
    pd.testing.assert_frame_equal(
        calendar_table(frame),
        calendar_table(frame, col="sales"),
    )

    ambiguous = frame.assign(price=[4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="col is required"):
        calendar_table(ambiguous)
```

Also assert that `col` is rejected for Series input and that a missing or duplicate DataFrame column is rejected.

**Step 2: Run the tests and verify the missing API failure**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -q
```

Expected: collection fails because `calendar_table` is not exported.

**Step 3: Extract shared private calendar helpers**

Create `TsUtils/_calendar.py` with private helpers that:

```python
def resolve_time_index(data) -> pd.DatetimeIndex | pd.PeriodIndex:
    """Return a validated dated index from an index, Series, or DataFrame."""


def resolve_frequency(
    index: pd.DatetimeIndex | pd.PeriodIndex,
    *,
    freq: str | None = None,
):
    """Return a unit pandas offset from explicit, stored, or inferred frequency."""
```

Move the existing empty, `NaT`, uniqueness, monotonicity, `index.freq`, and `pd.infer_freq()` behavior out of `_seasonal_dummies.py`. Keep operation-specific unsupported-frequency errors inside `seasonal_dummies()` so its public messages and supported boundary do not change.

**Step 4: Implement the minimal monthly reshape and exports**

In `_calendar_table.py`, implement:

```python
def calendar_table(data, *, col=None, freq=None, label_style="numeric") -> pd.DataFrame:
    series = _select_series(data, col=col)
    index = resolve_time_index(series)
    offset = resolve_frequency(index, freq=freq)
    family = _classify_frequency(offset)
    complete = _complete_series(series, offset)
    return _reshape_monthly(complete)
```

At this step `_classify_frequency()` may accept only monthly offsets and raise for the remaining families. `_reshape_monthly()` must use calendar month and year coordinates, `unstack()`, and `reindex(range(1, 13))`; it must not loop over cells.

Export `calendar_table` from `TsUtils/__init__.py` and the curated root `__init__.py`.

**Step 5: Run focused and regression tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py TsUtils/tests/test_seasonal_dummies.py -q
```

Expected: the new monthly tests and all existing seasonal-dummy tests pass.

**Step 6: Commit**

```powershell
git add TsUtils/_calendar.py TsUtils/_calendar_table.py TsUtils/_seasonal_dummies.py TsUtils/__init__.py __init__.py TsUtils/tests/test_calendar_table.py
git commit -m "feat: add monthly calendar table reshape"
```

### Task 2: Complete annual, natural-quarterly, daily, and business-daily schemas

**Files:**
- Modify: `TsUtils/_calendar_table.py`
- Modify: `TsUtils/tests/test_calendar_table.py`

**Step 1: Write failing schema tests**

Cover:

```python
def test_natural_quarterly_table_uses_q1_to_q4_rows():
    index = pd.period_range("2023Q4", periods=3, freq="Q-DEC")
    result = calendar_table(pd.Series([4.0, 1.0, 2.0], index=index))
    assert result.index.equals(pd.Index(range(1, 5), name="quarter"))
    assert result.loc[4, 2023] == 4.0
    assert result.loc[1, 2024] == 1.0


def test_daily_table_uses_day_rows_and_year_month_columns():
    index = pd.date_range("2024-01-30", periods=4, freq="D")
    result = calendar_table(pd.Series(range(4), index=index, dtype=float))
    assert result.index.equals(pd.Index(range(1, 32), name="day"))
    assert result.columns.names == ["year", "month"]
    assert result.loc[30, (2024, 1)] == 0.0
    assert result.loc[2, (2024, 2)] == 3.0
```

Add equivalent annual and business-daily assertions. Annual output has one row labelled `"value"` and columns named `year`. Business-daily completion must not invent weekend observations; its fixed day rows still expose weekend dates as `NaN`.

**Step 2: Run the focused tests and verify unsupported-family failures**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -q
```

Expected: the newly added frequency-family tests fail because only monthly is implemented.

**Step 3: Generalize frequency classification and reshaping**

Classify unit pandas offsets in this order:

```text
BusinessDay -> business_daily
Day -> daily
Week -> weekly
business/calendar month begin/end -> monthly
calendar quarter begin/end -> quarterly
calendar year begin/end -> annual
```

Require calendar quarter-start January or quarter-end December. Require calendar year-start January or year-end December. Reject multiplied offsets (`offset.n != 1`), fiscal anchors, intraday offsets, and every unknown family.

Build one long frame containing `value`, row coordinate, and natural-calendar column coordinates, assert coordinate uniqueness, and use `unstack()` to create the table. Reindex to the canonical row schema for each family.

**Step 4: Run focused tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -q
```

Expected: all annual, quarterly, monthly, daily, and business-daily tests pass; weekly tests are not yet present.

**Step 5: Commit**

```powershell
git add TsUtils/_calendar_table.py TsUtils/tests/test_calendar_table.py
git commit -m "feat: support calendar table frequency schemas"
```

### Task 3: Implement Wednesday-based weekly attribution

**Files:**
- Modify: `TsUtils/_calendar_table.py`
- Modify: `TsUtils/tests/test_calendar_table.py`

**Step 1: Write failing weekly-anchor tests**

Use at least `W-SUN` and `W-MON`. Include a boundary case where the weekly timestamp label and the Wednesday fall in different months:

```python
def test_weekly_month_is_determined_by_wednesday_not_timestamp_label():
    index = pd.date_range("2024-02-04", periods=2, freq="W-SUN")
    result = calendar_table(pd.Series([10.0, 20.0], index=index))

    # The week ending Sunday 4 February has Wednesday 31 January.
    assert result.loc[5, (2024, 1)] == 10.0
    assert result.loc[1, (2024, 2)] == 20.0
```

Assert fixed rows 1--5, two-level columns named `year` and `month`, `PeriodIndex` support, and correct behavior for another weekly anchor.

**Step 2: Run the weekly tests and verify failure**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -k weekly -q
```

Expected: weekly frequency is unsupported or attributed from the wrong date.

**Step 3: Implement weekly interval and Wednesday derivation**

For a weekly `PeriodIndex`, use each period's start and end boundaries. For a weekly `DatetimeIndex`, interpret each anchored label as the end of its seven-day interval, derive the start six days earlier, and select the unique Wednesday inside the interval. Compute the row as:

```python
within_month_week = ((wednesday.day - 1) // 7) + 1
```

Use the Wednesday's natural year and month for the two column levels. Do not split, aggregate, or classify from the weekly label alone.

**Step 4: Run focused and all TsUtils tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -q
python -m pytest TsUtils/tests -q
```

Expected: all calendar-table tests and the complete TsUtils suite pass.

**Step 5: Commit**

```powershell
git add TsUtils/_calendar_table.py TsUtils/tests/test_calendar_table.py
git commit -m "feat: classify weekly tables by Wednesday"
```

### Task 4: Add explicit gap completion, label styles, and failure paths

**Files:**
- Modify: `TsUtils/_calendar.py`
- Modify: `TsUtils/_calendar_table.py`
- Modify: `TsUtils/tests/test_calendar_table.py`

**Step 1: Write failing gap, presentation, and validation tests**

Add tests for:

```python
def test_explicit_monthly_frequency_completes_missing_timestamp_with_nan():
    index = pd.DatetimeIndex(["2024-01-01", "2024-03-01"])
    result = calendar_table(
        pd.Series([1.0, 3.0], index=index),
        freq="MS",
    )
    assert pd.isna(result.loc[2, 2024])


def test_text_labels_are_presentation_ready():
    index = pd.date_range("2024-01-01", periods=2, freq="MS")
    result = calendar_table(
        pd.Series([1.0, 2.0], index=index),
        label_style="text",
    )
    assert result.index[:2].tolist() == ["1月", "2月"]
    assert result.columns.tolist() == ["2024年"]
```

Cover text labels for quarterly, weekly, and daily rows and for both levels of weekly/daily columns. Cover invalid `label_style`, inability to infer a gapped index without `freq`, explicit frequency/date mismatch, fiscal quarter/year, multiplied and intraday frequencies, empty input, `NaT`, duplicate/decreasing indexes, unsupported containers, non-numeric/Boolean/complex values, infinity, and non-mutation.

**Step 2: Run tests and verify the new failures**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py -q
```

Expected: new gap, label, and validation assertions fail before implementation.

**Step 3: Complete explicit frequency validation and gap completion**

When `freq` is explicit, require a string accepted by pandas, construct the complete `date_range` or `period_range` between the observed boundaries, and verify every observed label belongs to it. Reject incompatible labels rather than coercing them. When `freq` is absent, keep the explicit-metadata then `infer_freq()` order and raise if unresolved.

Preserve ordinary missing values during `reindex()`. Reject infinities using only non-missing observations; do not reject `NaN` or `pd.NA`.

**Step 4: Apply label styles after reshaping**

Keep all internal coordinates numeric. As the final step, map rows and columns for `label_style="text"`:

```text
year -> "2024年"
month -> "1月"
quarter row -> "Q1"
month row -> "1月"
week row -> "第1周"
day row -> "1日"
```

Retain stable axis names (`year`, `month`, `quarter`, `week`, `day`) in both styles.

**Step 5: Run focused and regression tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_calendar_table.py TsUtils/tests/test_seasonal_dummies.py -q
python -m pytest TsUtils/tests -q
```

Expected: all tests pass.

**Step 6: Commit**

```powershell
git add TsUtils/_calendar.py TsUtils/_calendar_table.py TsUtils/tests/test_calendar_table.py
git commit -m "feat: validate and label calendar tables"
```

### Task 5: Document, audit, and validate the complete public feature

**Files:**
- Modify: `TsUtils/README.md`
- Modify: `tests/test_public_docstrings.py` only if method/docstring contract coverage requires an explicit entry
- Verify: `TsUtils/_calendar_table.py`
- Verify: `TsUtils/__init__.py`
- Verify: `__init__.py`

**Step 1: Add executable public documentation**

Document:

- the full signature and input-selection rules;
- every frequency schema;
- natural-year-only behavior;
- Wednesday-based weekly attribution;
- numeric and text label examples;
- explicit `freq` for gapped input;
- missing-value preservation;
- `.to_excel()` and `.to_csv()` examples.

The public docstring must contain executable monthly and weekly examples and complete `Parameters`, `Returns`, and `Raises` sections.

**Step 2: Run documentation and public-export checks**

Run:

```powershell
python -m pytest tests/test_public_docstrings.py TsUtils/tests/test_calendar_table.py -q
```

Expected: every public export is mentioned in the TsUtils README, top-level imports resolve, and docstring examples pass.

**Step 3: Run static checks**

Run:

```powershell
python -m ruff check TsUtils/_calendar.py TsUtils/_calendar_table.py TsUtils/_seasonal_dummies.py TsUtils/tests/test_calendar_table.py TsUtils/__init__.py __init__.py
python -m ruff format --check TsUtils/_calendar.py TsUtils/_calendar_table.py TsUtils/_seasonal_dummies.py TsUtils/tests/test_calendar_table.py TsUtils/__init__.py __init__.py
python -m compileall -q TsUtils
git diff --check
```

Expected: every command exits successfully with no findings.

**Step 4: Run the complete test suite**

Run:

```powershell
python -m pytest -q
```

Expected: the full repository suite passes without regressions.

**Step 5: Audit plan-to-code and failure paths**

Confirm that implementation, tests, README, docstring, both public exports, every supported schema, every rejection path, missing-value behavior, and Wednesday rule match the approved design. Confirm no file-writing dependency, aggregation path, interpolation path, compatibility alias, or parallel calendar-validation implementation was added.

**Step 6: Commit**

```powershell
git add TsUtils/README.md TsUtils/_calendar_table.py tests/test_public_docstrings.py
git commit -m "docs: document calendar table utility"
```
