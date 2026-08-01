# TsUtils Box-Cox Transformation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add an auditable Box-Cox transformation for pandas Series and
DataFrame inputs, including automatic or explicit lambda values.

**Architecture:** Add a dedicated `TsUtils._boxcox` module that delegates the
statistical calculation to SciPy and preserves pandas metadata. Always return a
`BoxCoxResult` containing transformed data and the lambda values used, so fitted
parameters can be inspected and reused without changing the return type.

**Tech Stack:** Python 3.13, pandas, NumPy, SciPy, pytest, Ruff.

---

### Task 1: Specify the public contract with tests

**Files:**

- Create: `TsUtils/tests/test_boxcox.py`

**Step 1: Write failing numerical tests**

Test Series transformations against `scipy.stats.boxcox` for automatic lambda,
an explicit lambda, and `lmbda=0`. Test DataFrame columns independently with
automatically estimated, shared scalar, and per-column lambda values.

**Step 2: Write failing contract tests**

Verify a stable `BoxCoxResult`, metadata and shape preservation, no input
mutation, missing-value preservation, reusable fitted lambdas, and exports from
both `Ts.TsUtils` and `Ts`. Reject non-pandas, empty, duplicate-column,
non-numeric, boolean, complex, infinite, and non-positive inputs. Reject invalid
lambda specifications and reject insufficient or constant data during automatic
estimation.

**Step 3: Verify that the new API is initially absent**

Run:

```powershell
python -m pytest TsUtils/tests/test_boxcox.py -p no:cacheprovider -q
```

Expected: test collection fails because `boxcox` and `BoxCoxResult` are not yet
exported.

### Task 2: Implement and export Box-Cox transformation

**Files:**

- Create: `TsUtils/_boxcox.py`
- Modify: `TsUtils/__init__.py`
- Modify: `__init__.py`

**Step 1: Implement the result contract**

Create:

```python
@dataclass(frozen=True)
class BoxCoxResult:
    data: pd.Series | pd.DataFrame
    lmbda: float | pd.Series
```

For a Series, `lmbda` is a float. For a DataFrame, it is a float Series indexed
by the original columns, including when one shared scalar was supplied.

**Step 2: Implement validation and transformation**

Implement:

```python
def boxcox(data, *, lmbda=None):
    ...
```

Only accept non-empty Series/DataFrame objects with unique DataFrame columns and
real numeric non-boolean values. Preserve NaNs but reject infinities and require
every observed value to be strictly positive. For automatic estimation, require
at least two observed, non-constant values per series. A Series accepts `None`
or one finite real lambda. A DataFrame accepts `None`, one finite real lambda,
or a pandas Series/mapping whose labels exactly match the columns. For automatic
estimation, pass only observed values to `scipy.stats.boxcox` and reconstruct
missing positions afterward so the implementation remains compatible with the
declared SciPy 1.13 minimum. For fixed lambda values, delegate directly to
`scipy.stats.boxcox`. Reconstruct the original pandas container without mutating
the input.

**Step 3: Export the API**

Export `boxcox` and `BoxCoxResult` from `TsUtils.__init__` and the root package,
and include both in their `__all__` declarations.

**Step 4: Run targeted tests**

Run:

```powershell
python -m pytest TsUtils/tests/test_boxcox.py -p no:cacheprovider -q
```

Expected: all Box-Cox tests pass.

### Task 3: Document and verify the feature

**Files:**

- Modify: `TsUtils/README.md`
- Modify: `TsUtils/demo.ipynb`

**Step 1: Document behavior and examples**

Document automatic and explicit lambda usage, DataFrame per-column behavior,
the `BoxCoxResult` fields, strict positivity, missing-value preservation, and
the rule that the utility never adds a shift automatically.

Rebuild the demo as an executable API tutorial for all six TsUtils capabilities.
For every function, document the full signature, every parameter, return type,
all public result fields/properties/methods, basic and parameter-variant calls,
result access patterns, constraints, and common statistical pitfalls. The
Box-Cox examples must estimate lambda on training data, reuse it on held-out
observations, and cover DataFrame lambdas. Recheck every example while executing
the notebook top-to-bottom.

**Step 2: Execute and validate the demo notebook**

Run:

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace TsUtils/demo.ipynb
```

Expected: every cell executes successfully and the final check prints
`All TsUtils parameter, return-object, and usage checks passed.`

**Step 3: Run regression tests**

Run:

```powershell
python -m pytest TsUtils/tests -p no:cacheprovider -q
python -m pytest . -p no:cacheprovider -q
```

Expected: all tests pass.

**Step 4: Run static and public-contract checks**

Run:

```powershell
python -m ruff check TsUtils __init__.py
python -c "from Ts import BoxCoxResult, boxcox; from Ts.TsUtils import boxcox as local; assert boxcox is local; print('boxcox-api-ok')"
git diff --check
```

Expected: Ruff and whitespace checks report no errors, and the import smoke test
prints `boxcox-api-ok`.
