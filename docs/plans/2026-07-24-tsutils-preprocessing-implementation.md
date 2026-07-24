# TsUtils Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add a `TsUtils` preprocessing subpackage that owns STL decomposition and provides auditable missing-value interpolation, without any seasonal-adjustment/X-13 API.

**Architecture:** Move STL out of `TsModels` with no compatibility re-export, keeping its existing public behavior while replacing its dependency on `TsModels._base` with a local validation helper. Implement interpolation as a type-preserving function returning a structured result with masks and counts; true interpolation remains the default, while edge filling is explicit.

**Tech Stack:** Python 3.13, NumPy, pandas, SciPy-backed pandas interpolation, statsmodels STL, pytest, Ruff.

---

### Task 1: Create TsUtils and hard-migrate STL

**Files:**
- Create: `TsUtils/__init__.py`
- Create: `TsUtils/_validation.py`
- Create: `TsUtils/_stl.py`
- Create: `TsUtils/tests/__init__.py`
- Create: `TsUtils/tests/test_stl.py`
- Delete: `TsModels/_stl.py`
- Delete: `TsModels/tests/test_stl.py`
- Modify: `TsModels/__init__.py`
- Modify: `TsModels/tests/test_missing_policy.py`

**Step 1: Move the STL tests and change public imports**

Copy the existing STL behavior tests into `TsUtils/tests/test_stl.py`, import
`STL` and `STLResult` from `Ts.TsUtils`, and add explicit coverage for
`missing="drop"` and `dropped_positions`.

**Step 2: Run tests to verify the new package is initially missing**

Run:

```powershell
python -m pytest Ts/TsUtils/tests/test_stl.py -p no:cacheprovider -q
```

Expected: collection fails because `Ts.TsUtils` does not exist.

**Step 3: Implement the package and migrate STL**

Create a private `TsUtils._validation._resolve_missing_rows` helper with the
same strict `raise`/`drop` contract currently used by STL. Move the STL and
STLResult implementation unchanged except for importing that local helper.

Remove the STL module, documentation, imports and `__all__` entries from
`TsModels`. Remove STL from the cross-model missing-policy parameterization;
its missing behavior is now tested in `TsUtils/tests/test_stl.py`.

**Step 4: Run the migrated STL tests**

Run:

```powershell
python -m pytest Ts/TsUtils/tests/test_stl.py Ts/TsModels/tests/test_missing_policy.py -p no:cacheprovider -q
```

Expected: all selected tests pass and `Ts.TsModels` has no STL symbol.

### Task 2: Implement missing-value interpolation

**Files:**
- Create: `TsUtils/_interpolation.py`
- Create: `TsUtils/tests/test_interpolation.py`
- Modify: `TsUtils/__init__.py`

**Step 1: Write failing tests for the public contract**

Cover:

- one- and two-dimensional NumPy inputs;
- pandas Series/DataFrame type, index, name and column preservation;
- `linear`, `time`, `nearest`, and `cubic` methods;
- no mutation of caller data;
- `max_gap` protection for long internal gaps;
- `edge="keep"` and explicit `edge="nearest"`;
- rejection of infinities, invalid dimensions, invalid options and invalid
  time indexes;
- all-missing inputs remaining unfilled;
- result masks, counts, `complete`, and `summary()`.

**Step 2: Run tests to verify the API is missing**

Run:

```powershell
python -m pytest Ts/TsUtils/tests/test_interpolation.py -p no:cacheprovider -q
```

Expected: collection fails because `interpolate_missing` and
`InterpolationResult` do not exist.

**Step 3: Implement the result and interpolation engine**

Public signature:

```python
def interpolate_missing(
    data,
    method="linear",
    *,
    max_gap=None,
    edge="keep",
) -> InterpolationResult:
    ...
```

The implementation must:

- accept numeric ndarray, Series, and DataFrame inputs only;
- treat `NaN`/`pd.NA` as missing and reject positive/negative infinity;
- interpolate down the observation axis independently per series;
- require a unique, increasing DatetimeIndex/TimedeltaIndex for
  `method="time"`;
- retain gaps longer than `max_gap`;
- never extrapolate by default;
- fill only eligible leading/trailing gaps with the nearest observation when
  `edge="nearest"`;
- return a copied output in the same container type and same logical shape.

`InterpolationResult` stores `data`, `missing_mask`, `filled_mask`,
`remaining_mask`, `method`, `max_gap`, and `edge`, and exposes
`n_missing`, `n_filled`, `n_remaining`, `complete`, and `summary()`.

**Step 4: Run interpolation tests**

Run:

```powershell
python -m pytest Ts/TsUtils/tests/test_interpolation.py -p no:cacheprovider -q
```

Expected: all interpolation tests pass.

### Task 3: Integrate public API, documentation, and verification

**Files:**
- Create: `TsUtils/README.md`
- Modify: `__init__.py`
- Modify: `TsModels/README.md`

**Step 1: Add public exports and documentation**

Export `STL`, `STLResult`, `interpolate_missing`, and
`InterpolationResult` from both `Ts.TsUtils` and `Ts`. Update the root package
description from five to six subpackages. Remove current STL ownership and
examples from `TsModels/README.md`; document STL and interpolation in
`TsUtils/README.md`, including assumptions and edge behavior.

**Step 2: Run targeted tests**

Run:

```powershell
python -m pytest Ts/TsUtils/tests Ts/TsModels/tests/test_missing_policy.py -p no:cacheprovider -q
```

Expected: all selected tests pass.

**Step 3: Run full regression and static checks**

Run:

```powershell
python -m pytest Ts -p no:cacheprovider -q
python -m ruff check Ts/TsUtils Ts/TsModels/__init__.py Ts/TsModels/tests/test_missing_policy.py Ts/__init__.py
```

Expected: the full suite passes and Ruff reports no errors.

**Step 4: Verify public and removed APIs**

Run:

```powershell
python -c "from Ts import STL, STLResult, interpolate_missing, InterpolationResult; from Ts.TsUtils import STL as USTL; from Ts import TsModels; assert not hasattr(TsModels, 'STL'); print('tsutils-api-ok')"
```

Expected: `tsutils-api-ok`.

**Step 5: Review the final diff**

Run:

```powershell
git status --short
git diff --check
git diff --stat
```

Expected: only the plan and requested TsUtils migration/integration files are
changed; no whitespace errors are reported.
