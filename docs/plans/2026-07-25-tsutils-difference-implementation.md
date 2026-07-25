# TsUtils Difference Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add one composable `difference()` utility for first/second ordinary,
log, year-over-year, and year-over-year log differences on pandas Series and
DataFrame objects.

**Architecture:** Put the transformation and its validation in a dedicated
`TsUtils._difference` module. Represent all eight requested operations through
three orthogonal keyword arguments: `order`, `log`, and `lag`; a year-over-year
operation uses an explicit observations-per-year lag so the function never
guesses the sampling frequency. Preserve pandas metadata and shape, propagate
missing values under standard pandas differencing semantics, and reject inputs
that make the requested calculation undefined.

**Tech Stack:** Python 3.13, pandas 2.2, NumPy, pytest, Ruff.

---

### Task 1: Specify the public transformation contract

**Files:**

- Create: `TsUtils/tests/test_difference.py`

**Step 1: Write failing tests for all eight transformations**

Use monthly Series and two-column DataFrame fixtures to verify:

- `order=1, log=False, lag=1`;
- `order=1, log=True, lag=1`;
- `order=2, log=False, lag=1`;
- `order=2, log=True, lag=1`;
- `order=1, log=False, lag=12`;
- `order=1, log=True, lag=12`;
- `order=2, log=False, lag=12`;
- `order=2, log=True, lag=12`.

Expected values must be computed independently with the explicit formulas:

```python
first = values - values.shift(lag)
second = values - 2 * values.shift(lag) + values.shift(2 * lag)
```

For log variants, apply the formulas to `np.log(values)`.

**Step 2: Write failing contract tests**

Cover:

- Series/DataFrame container and metadata preservation;
- output shape and index preservation, including leading missing values;
- no mutation of caller data;
- missing-value propagation;
- rejection of non-pandas input, non-numeric/boolean columns, infinities,
  non-positive values for log transforms, invalid `order`, and invalid `lag`;
- exports from both `Ts.TsUtils` and `Ts`.

**Step 3: Run the new test module and verify the API is absent**

Run:

```powershell
python -m pytest TsUtils/tests/test_difference.py -p no:cacheprovider -q
```

Expected: collection fails because `difference` is not exported.

### Task 2: Implement and integrate `difference`

**Files:**

- Create: `TsUtils/_difference.py`
- Modify: `TsUtils/__init__.py`
- Modify: `__init__.py`

**Step 1: Implement strict input and option validation**

Public signature:

```python
def difference(data, *, order=1, log=False, lag=1):
    ...
```

The implementation must:

- accept only pandas Series and DataFrame;
- reject empty inputs, duplicate DataFrame column labels, non-numeric columns,
  and boolean dtypes;
- allow missing values but reject positive/negative infinity;
- require `order` to be the non-boolean integer `1` or `2`;
- require `lag` to be a positive non-boolean integer;
- require `log` to be a boolean;
- require every non-missing value to be strictly positive when `log=True`.

**Step 2: Implement the transformation**

Copy/convert the input to floating-point values, optionally apply the natural
logarithm, and call pandas `diff(periods=lag)` exactly `order` times. Return the
same pandas container type with the original index, Series name, and DataFrame
columns. Do not drop rows or mutate the caller.

For `order=2` and lag `s`, the result is:

```text
(1 - L^s)^2 x_t = x_t - 2 x_(t-s) + x_(t-2s)
```

The log form applies the same operator to `log(x)`.

**Step 3: Export the public function**

Add `difference` to `TsUtils.__init__`, the package root imports, and both
`__all__` declarations.

**Step 4: Run the targeted tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_difference.py -p no:cacheprovider -q
```

Expected: all tests pass.

### Task 3: Document and verify the completed feature

**Files:**

- Modify: `TsUtils/README.md`

**Step 1: Document the API and all eight mappings**

Add:

- the signature and behavior contract;
- a table mapping the eight operations to arguments;
- explicit monthly (`lag=12`), quarterly (`lag=4`), and annual (`lag=1`)
  year-over-year examples;
- the second year-over-year difference formula;
- log positivity and missing-value behavior.

**Step 2: Run TsUtils regression tests**

Run:

```powershell
python -m pytest TsUtils/tests -p no:cacheprovider -q
```

Expected: all TsUtils tests pass.

**Step 3: Run the full repository regression suite**

Run:

```powershell
python -m pytest . -p no:cacheprovider -q
```

Expected: all tests pass.

**Step 4: Run static checks**

Run:

```powershell
python -m ruff check TsUtils __init__.py
```

Expected: Ruff reports no errors.

**Step 5: Verify public imports**

Run:

```powershell
python -c "from Ts import difference as root; from Ts.TsUtils import difference as local; assert root is local; print('difference-api-ok')"
```

Expected: `difference-api-ok`.

**Step 6: Review the final diff**

Run:

```powershell
git status --short
git diff --check
git diff --stat
```

Expected: the pre-existing deletion of `scripts/check_ts_quality.py` remains
untouched; task changes are limited to the plan, difference module, tests,
public exports, and TsUtils documentation; no whitespace errors are reported.
