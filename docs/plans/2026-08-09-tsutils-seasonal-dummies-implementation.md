# TsUtils Seasonal Dummies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `seasonal_dummies()` to generate fixed-schema recurring calendar dummy variables from regular dated time-series indexes.

**Architecture:** Create one focused TsUtils module that validates and extracts a pandas time index, classifies its pandas offset family, derives canonical seasonal positions, and delegates one-hot encoding to pandas. Export the function through `Ts.TsUtils` and `Ts`, without coupling it to model-specific intervention machinery.

**Tech Stack:** Python, pandas 2.3.3, NumPy, pytest, Ruff, existing TsUtils public-export conventions.

---

### Task 1: Validate dated inputs and classify supported frequencies

**Files:**
- Create: `TsUtils/_seasonal_dummies.py`
- Create: `TsUtils/tests/test_seasonal_dummies.py`

**Step 1: Write failing validation and frequency tests**

Cover all accepted containers and rejected index states:

```python
@pytest.mark.parametrize("container", ["series", "frame", "index"])
def test_accepts_dated_public_inputs(container):
    index = pd.date_range("2024-01-01", periods=12, freq="MS")
    ...
    result = seasonal_dummies(data)
    assert result.index.equals(index)


@pytest.mark.parametrize(
    "data, match",
    [
        (pd.Series([1, 2, 3]), "DatetimeIndex or PeriodIndex"),
        (pd.DatetimeIndex([]), "must not be empty"),
        (pd.DatetimeIndex(["2024-01-01", None]), "missing dates"),
        (pd.DatetimeIndex(["2024-01-01", "2024-01-01"]), "unique"),
    ],
)
def test_rejects_invalid_time_indexes(data, match):
    with pytest.raises((TypeError, ValueError), match=match):
        seasonal_dummies(data)
```

Also test descending, irregular, annual, intraday, multiplied offsets such as
`2MS`, and non-boolean `drop_first` inputs.

**Step 2: Run tests to verify the missing public API fails**

```powershell
python -m pytest TsUtils/tests/test_seasonal_dummies.py -q
```

Expected: collection/import failure because the function does not exist.

**Step 3: Implement input extraction and frequency classification**

In `TsUtils/_seasonal_dummies.py`, add private helpers that:

- accept Series, DataFrame, DatetimeIndex, or PeriodIndex;
- require a non-empty, missing-free, unique, increasing time index;
- use `index.freq` first and `pd.infer_freq(index)` second;
- convert the frequency with `pandas.tseries.frequencies.to_offset`;
- require `offset.n == 1`;
- classify `Day`, `BusinessDay`, `Week`, month begin/end variants, and quarter
  begin/end variants;
- reject annual, intraday, irregular, and unsupported offsets with the detected
  frequency included in the error.

Define the public function shell and boolean validation, but return no final
dummy values until Task 2.

**Step 4: Run validation tests**

Run the command from Step 2.

Expected: validation and classification tests pass; value-schema tests remain
pending or xfailed only if explicitly separated.

### Task 2: Generate fixed-schema seasonal dummy matrices

**Files:**
- Modify: `TsUtils/_seasonal_dummies.py`
- Modify: `TsUtils/tests/test_seasonal_dummies.py`

**Step 1: Write failing exact-schema tests**

Specify these contracts:

```python
def test_monthly_defaults_to_january_reference_and_fixed_schema():
    index = pd.date_range("2024-02-01", periods=3, freq="MS")
    result = seasonal_dummies(index)
    assert list(result) == [f"month_{month:02d}" for month in range(2, 13)]
    assert result.loc[index[0], "month_02"] == 1


def test_drop_first_false_keeps_complete_quarter_schema():
    index = pd.period_range("2024Q1", periods=4, freq="Q-DEC")
    result = seasonal_dummies(index, drop_first=False)
    assert list(result) == ["quarter_Q1", "quarter_Q2", "quarter_Q3", "quarter_Q4"]
    np.testing.assert_array_equal(result.sum(axis=1), 1)
```

Add exact tests for calendar daily weekdays, business weekdays, ISO week 01–53,
month begin/end variants, fiscal `Q-MAR` PeriodIndex, `QS-APR` DatetimeIndex,
index preservation, `int8` values, and absent-category all-zero columns.

**Step 2: Run schema tests and verify failure**

```powershell
python -m pytest TsUtils/tests/test_seasonal_dummies.py -q
```

Expected: failures where the function has not produced the contracted matrix.

**Step 3: Implement category derivation and pandas encoding**

Derive positions as follows:

- daily: `dayofweek`, using Monday–Sunday or Monday–Friday canonical labels;
- weekly: ISO week number from timestamps (or PeriodIndex start timestamps);
- monthly: `.month`;
- quarterly: calculate Q1–Q4 relative to the `QuarterBegin.startingMonth` or
  `QuarterEnd.startingMonth` anchor.

Construct an ordered `pd.Categorical` with the complete canonical category
list, call `pd.get_dummies(..., dtype=np.int8)`, assign the original index, and
drop the fixed first column only when `drop_first=True`.

The docstring must state inputs, formula-free category rules, fixed schema,
reference categories, errors, return type, and include executable monthly and
quarterly examples.

**Step 4: Run all seasonal-dummy tests**

Run the command from Step 2.

Expected: all tests pass.

### Task 3: Publish and document the new TsUtils API

**Files:**
- Modify: `TsUtils/__init__.py`
- Modify: `__init__.py`
- Modify: `TsUtils/README.md`
- Modify: `TsUtils/tests/test_seasonal_dummies.py`

**Step 1: Add failing public-import tests**

```python
def test_public_imports_expose_same_function():
    from Ts import seasonal_dummies as top_level
    from Ts.TsUtils import seasonal_dummies as utils_level

    assert top_level is utils_level
```

**Step 2: Run the public-import test and verify failure**

```powershell
python -m pytest TsUtils/tests/test_seasonal_dummies.py -k public_imports -q
```

Expected: import failure until exports are added.

**Step 3: Add both public exports**

Import `seasonal_dummies` in `TsUtils/__init__.py`, add it to `TsUtils.__all__`,
and mirror the symbol in the root package import block and `__all__`.

**Step 4: Update README**

Change the TsUtils capability count, add the function to the public import
example, and add a `## 季节虚拟变量` section containing monthly and quarterly
examples, fixed-column behavior, default references, supported frequencies,
and explicit annual/intraday/irregular exclusions.

**Step 5: Run public tests and doctests**

```powershell
python -m pytest TsUtils/tests/test_seasonal_dummies.py -q
python -m pytest --doctest-modules TsUtils/_seasonal_dummies.py -q
```

Expected: all tests and doctests pass.

### Task 4: Complete regression, quality, and change-boundary verification

**Files:**
- Verify all new and modified source, tests, docs, and plans.

**Step 1: Run the full TsUtils suite**

```powershell
python -m pytest TsUtils/tests -q
```

Expected: all TsUtils tests pass.

**Step 2: Run the full repository suite**

```powershell
python -m pytest -q
```

Expected: all repository tests pass.

**Step 3: Run lint, compile, and public smoke checks**

```powershell
python -m ruff check TsUtils __init__.py
python -m compileall -q TsUtils
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd().parent)); from Ts import seasonal_dummies; from Ts.TsUtils import seasonal_dummies as u; assert seasonal_dummies is u"
git diff --check
```

Expected: Ruff prints `All checks passed!`; all other commands exit 0.

**Step 4: Audit the final diff and repository state**

```powershell
git status --short
git diff --stat
git diff -- TsUtils/_seasonal_dummies.py TsUtils/tests/test_seasonal_dummies.py TsUtils/__init__.py __init__.py TsUtils/README.md
```

Expected: only the approved seasonal-dummy feature, tests, docs, and
implementation plan are present.

**Step 5: Commit the verified feature**

```powershell
git add docs/plans/2026-08-09-tsutils-seasonal-dummies-implementation.md TsUtils/_seasonal_dummies.py TsUtils/tests/test_seasonal_dummies.py TsUtils/__init__.py __init__.py TsUtils/README.md
git commit -m "feat: add seasonal dummy generation"
```

Expected: one clean feature commit on `main`.
