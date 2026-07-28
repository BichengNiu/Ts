# Structural Break Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use @executing-plans to implement this plan task-by-task.

**Goal:** Correct the existing Perron/Zivot-Andrews structural-break unit-root contracts; add Chow and OLS-residual CUSUM; add a complete Bai-Perron multiple-unknown-break regression test and a Lee-Strazicich two-unknown-break LM unit-root test.

**Architecture:** Keep structural-break unit-root tests separate from regression parameter-stability tests. Reuse the existing `BaseTest`/`BaseTestResult` lifecycle, add one small regression-design helper, and expose one class plus one result dataclass per statistical method. Use `statsmodels` only where its published diagnostic implementation supplies a valid reference contract; keep breakpoint mapping, validation, summaries, and plots inside `TsTests`.

**Tech Stack:** Python 3.13, NumPy, SciPy, pandas, statsmodels 0.14.x, matplotlib, pytest, Ruff.

**Status:** Approved by the user on 2026-07-28 for production-code execution.

---

## Scope and statistical contracts

### Existing methods to correct

- `PerronTest`: known single structural break in a unit-root test.
- `ZivotAndrewsTest`: unknown single structural break in a unit-root test.

Perron model regressors must be distinct:

```python
PERRON_DUMMIES = {
    "intercept": ("DL", "DP"),
    "slope": ("DL", "DT"),
    "both": ("DL", "DP", "DT"),
}

ZIVOT_ANDREWS_DUMMIES = {
    "intercept": ("DL",),
    "slope": ("DT",),
    "both": ("DL", "DT"),
}
```

Perron critical values must be selected by break fraction
`lambda = break_index / nobs`, not by sample size. The supported table spans
`lambda = 0.1, ..., 0.9`; interpolate only between adjacent tabulated break
fractions.

### New methods

| Class | Null hypothesis | Break location | Main inference |
|---|---|---|---|
| `ChowTest` | all selected OLS coefficients are stable | known | classical F statistic and p-value |
| `CUSUMTest` | OLS regression parameters are stable | unknown | supremum of scaled cumulative OLS residuals, Brownian-bridge p-value |
| `BaiPerronTest` | regression coefficients are stable across all regimes | multiple unknown | global dynamic-programming partition, break-count selection, sup-F family and break-date intervals |
| `LeeStrazicichTwoBreakTest` | the series has a unit root while allowing two breaks under both null and alternative | two unknown | minimum LM statistic over admissible break pairs |

The first release deliberately excludes:

- Hansen (1992): the installed statsmodels implementation returns only a sparse
  5% critical-value table and contains an upstream TODO about the table.
- Quandt-Andrews / sup-Wald: the break date is unidentified under the null and
  the limiting distribution is non-standard; a plain F-distribution p-value is
  invalid.
- Recursive-residual CUSUM: useful as a path diagnostic, but lower priority
  after adding direct coverage for multiple unknown breaks. It remains a
  documented follow-up rather than part of this release.

---

### Task 1: Lock down current behavior and reproduce the defects

**Files:**

- Modify: `TsTests/tests/test_perron.py`
- Modify: `TsTests/tests/test_zivot.py`
- Modify: `TsTests/tests/test_input_validation.py`
- Create: `TsTests/tests/test_structural_break_reference.py`

**Step 1: Add failing Perron specification tests**

```python
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("intercept", {"DL", "DP"}),
        ("slope", {"DL", "DT"}),
        ("both", {"DL", "DP", "DT"}),
    ],
)
def test_perron_models_use_published_dummy_sets(model, expected):
    assert set(_make_perron_break_dummies(100, 50, model)) == expected
```

**Step 2: Add failing critical-value tests**

```python
@pytest.mark.parametrize(
    ("model", "break_fraction", "alpha", "expected"),
    [
        ("intercept", 0.1, 0.05, -3.68),
        ("intercept", 0.5, 0.05, -3.76),
        ("slope", 0.5, 0.05, -3.96),
        ("both", 0.5, 0.05, -4.24),
        ("both", 0.9, 0.01, -4.41),
    ],
)
def test_perron_critical_values_depend_on_break_fraction(
    model, break_fraction, alpha, expected
):
    assert _perron_crit(model, break_fraction, alpha) == pytest.approx(expected)
```

**Step 3: Add failing validation tests**

Cover:

- negative or boolean `lags` / `max_lags`;
- non-positive or non-finite `lag_crit`;
- `trim <= 0` or `trim >= 0.5`;
- non-increasing or duplicate `time_index`;
- a Perron break label absent from `time_index`;
- a Perron break fraction outside the supported `[0.1, 0.9]` range;
- insufficient observations or rank-deficient regressions.

**Step 4: Add Zivot-Andrews reference parity tests**

For fixed `lags=1`, compare all three model specifications to
`statsmodels.tsa.stattools.zivot_andrews(..., autolag=None)`:

```python
MODEL_MAP = {"intercept": "c", "slope": "t", "both": "ct"}
assert ours.statistic == pytest.approx(upstream[0], abs=1e-10)
assert ours.break_index == upstream[4]
```

**Step 5: Run the focused tests and record expected failures**

Run:

```powershell
$env:PYTHONUTF8='1'
python -m pytest TsTests/tests/test_perron.py TsTests/tests/test_zivot.py TsTests/tests/test_input_validation.py TsTests/tests/test_structural_break_reference.py -q
```

Expected: failures for Perron slope dummies, break-fraction critical values,
and invalid-parameter acceptance; existing Zivot-Andrews fixed-lag parity
tests pass.

---

### Task 2: Correct Perron dummy construction and critical values

**Files:**

- Modify: `TsTests/_break_utils.py`
- Modify: `TsTests/_critical_values.py`
- Modify: `TsTests/_perron.py`
- Modify: `TsTests/_zivot.py`
- Test: `TsTests/tests/test_perron.py`
- Test: `TsTests/tests/test_zivot.py`
- Test: `TsTests/tests/test_structural_break_reference.py`

**Step 1: Split ambiguous dummy construction**

Replace the shared `include_pulse` switch with two explicit helpers:

```python
def _make_perron_break_dummies(T: int, break_idx: int, model: str):
    t = np.arange(T)
    dl = (t > break_idx).astype(float)
    dp = (t == break_idx + 1).astype(float)
    dt = np.maximum(t - break_idx, 0).astype(float)
    mapping = {
        "intercept": {"DL": dl, "DP": dp},
        "slope": {"DL": dl, "DT": dt},
        "both": {"DL": dl, "DP": dp, "DT": dt},
    }
    return mapping[model]


def _make_zivot_break_dummies(T: int, break_idx: int, model: str):
    t = np.arange(T)
    dl = (t > break_idx).astype(float)
    dt = np.maximum(t - break_idx, 0).astype(float)
    mapping = {
        "intercept": {"DL": dl},
        "slope": {"DT": dt},
        "both": {"DL": dl, "DT": dt},
    }
    return mapping[model]
```

Do not retain `_make_break_dummies` as a compatibility alias: it is private,
and retaining it would preserve the ambiguity that caused the model-B error.

**Step 2: Replace sample-size critical values**

Define Perron critical-value tables keyed by break fraction `0.1` through
`0.9` for all three models and significance levels `0.01`, `0.025`, `0.05`,
and `0.10`.

```python
def _perron_crit(model: str, break_fraction: float, significance: float) -> float:
    table = _PERRON_CRIT_MAP[model]
    fractions = np.asarray(sorted(table))
    if not 0.1 <= break_fraction <= 0.9:
        raise ValueError("break_fraction must be between 0.1 and 0.9")
    values = np.asarray([table[f][significance] for f in fractions])
    return float(np.interp(break_fraction, fractions, values))
```

Delete `_interpolate_crit`; it encodes the wrong sample-size contract and has
no other caller.

**Step 3: Pass the actual break fraction into the result**

Add `break_fraction: float` to `PerronTestResult`. Store the actual matched
time label rather than the unverified requested value.

Correct both Perron and Zivot-Andrews conclusion text so the null is stated
consistently:

```python
"Reject H0 (unit root); evidence favors stationarity around a breaking trend"
```

The current phrase `"Reject H0 (stationary with break)"` reverses the stated
null hypothesis.

**Step 4: Use positional deterministic trend**

In ADF-style break regressions, build the deterministic trend from
`np.arange(T, dtype=float)`. Treat `time_index` as an observation label only.
This prevents irregular calendar labels from changing the statistical model.

**Step 5: Run focused tests**

Run:

```powershell
python -m pytest TsTests/tests/test_perron.py TsTests/tests/test_zivot.py TsTests/tests/test_structural_break_reference.py -q
```

Expected: all Perron specification and critical-value tests pass; the
Zivot-Andrews fixed-lag reference results remain unchanged.

---

### Task 3: Harden existing structural-break input contracts

**Files:**

- Modify: `TsTests/_break_utils.py`
- Modify: `TsTests/_perron.py`
- Modify: `TsTests/_zivot.py`
- Modify: `TsTests/tests/test_input_validation.py`

**Step 1: Add explicit validators**

```python
def _validate_nonnegative_int(value, *, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _validate_time_axis(time_index):
    if np.any(np.diff(time_index) <= 0):
        raise ValueError("time_index must be strictly increasing and unique")


def _locate_known_break(time_index, break_year):
    matches = np.flatnonzero(np.isclose(time_index, break_year, rtol=0.0, atol=1e-12))
    if len(matches) != 1:
        raise ValueError("break_year must match exactly one time_index value")
    return int(matches[0])
```

**Step 2: Validate test-specific configuration during construction**

- Perron: known break must resolve exactly and its fraction must be in
  `[0.1, 0.9]`.
- Zivot-Andrews: `0 < trim < 0.5`.
- Both: validate `lags`, `max_lags`, `lag_crit`, finite non-constant data,
  minimum residual degrees of freedom, and full-rank design.

**Step 3: Reject invalid numerical fits**

After each OLS fit, require finite statistic, standard error, residuals, and
positive residual degrees of freedom. Expected numerical failures may skip a
Zivot-Andrews candidate; unexpected programming errors must still propagate.

**Step 4: Run input and break-test suites**

Run:

```powershell
python -m pytest TsTests/tests/test_input_validation.py TsTests/tests/test_perron.py TsTests/tests/test_zivot.py -q
```

Expected: all tests pass with no runtime warnings from invalid negative-lag or
boundary-break regressions.

---

### Task 4: Add a shared regression-stability design helper

**Files:**

- Create: `TsTests/_regression_break_utils.py`
- Modify: `TsTests/_base.py`
- Create: `TsTests/tests/test_regression_break_utils.py`
- Modify: `TsTests/tests/test_shared_infrastructure.py`

**Step 1: Allow non-lag tests to report honest metadata**

Change:

```python
lags: int | None
```

and format `None` as `N/A` in `BaseTestResult._format_conclusion`. Existing
test results continue to return integer lag counts.

**Step 2: Define one shared design builder**

```python
@dataclass(frozen=True)
class RegressionBreakDesign:
    endog: np.ndarray
    exog: np.ndarray
    time_index: np.ndarray
    column_names: tuple[str, ...]


def _prepare_regression_break_design(
    data,
    *,
    exog=None,
    time_index=None,
    trend="c",
    y_col=None,
    time_col=None,
    exog_cols=None,
) -> RegressionBreakDesign:
    ...
```

Contract:

- `trend` is one of `"n"`, `"c"`, `"ct"`;
- `exog` is explicit 1-D or 2-D numeric data;
- `exog_cols` is valid only for DataFrame `data` and cannot be combined with
  `exog`;
- no NaN/Inf rows are silently removed because removal changes breakpoint
  alignment;
- positional trend is used; `time_index` is a display/mapping label;
- the final design must have full column rank and `nobs > nparams`.

**Step 3: Test array and DataFrame paths**

Cover 1-D/2-D exog, named DataFrame columns, deterministic-only designs,
misalignment, non-finite input, duplicate constant columns, rank deficiency,
and conflicting `exog`/`exog_cols`.

**Step 4: Run helper tests**

Run:

```powershell
python -m pytest TsTests/tests/test_regression_break_utils.py TsTests/tests/test_shared_infrastructure.py -q
```

Expected: all tests pass.

---

### Task 5: Implement the known-break Chow test

**Files:**

- Create: `TsTests/_chow.py`
- Create: `TsTests/tests/test_chow.py`

**Step 1: Write result-contract and manual-formula tests**

```python
result = ChowTest(y, break_year=50, exog=x, time_index=t).fit()
assert result.statistic == pytest.approx(manual_f)
assert result.pvalue == pytest.approx(scipy.stats.f.sf(manual_f, q, n - 2 * q))
assert result.break_index == 50
assert result.df_num == q
assert result.df_denom == n - 2 * q
```

Also test a stable simulation does not systematically reject and a seeded,
large coefficient shift is detected near the expected p-value range.

**Step 2: Implement the classical Chow statistic**

```python
rss_pooled = pooled.ssr
rss_split = left.ssr + right.ssr
df_num = nparams
df_denom = nobs - 2 * nparams
statistic = ((rss_pooled - rss_split) / df_num) / (rss_split / df_denom)
pvalue = scipy.stats.f.sf(statistic, df_num, df_denom)
```

Break semantics remain consistent with Perron: the first regime includes
`break_index`; the second starts at `break_index + 1`.

Reject breaks that leave either regime with `nobs_regime <= nparams`.

**Step 3: Define result fields**

`ChowTestResult` extends `BaseTestResult` and adds:

- `break_year`, `break_index`;
- `df_num`, `df_denom`;
- `rss_pooled`, `rss_split`;
- `coefficients_pooled`, `coefficients_before`, `coefficients_after`;
- `fitted_pooled`, `fitted_split`;
- `time_index`.

Set `lags=None`. Document that the classical F reference distribution assumes
independent, homoskedastic errors and a break date specified before inspecting
the data.

**Step 4: Add a diagnostic plot**

`plot_test()` shows observed `y`, pooled fitted values, split-regime fitted
values, and the known break line against `time_index`.

**Step 5: Run Chow tests**

Run:

```powershell
python -m pytest TsTests/tests/test_chow.py -q
```

Expected: all tests pass.

---

### Task 6: Implement the OLS-residual CUSUM test

**Files:**

- Create: `TsTests/_cusum.py`
- Create: `TsTests/tests/test_cusum.py`

**Step 1: Write upstream parity tests**

Fit the same OLS model and compare to
`statsmodels.stats.diagnostic.breaks_cusumolsresid`:

```python
sup_b, pvalue, critical_values = breaks_cusumolsresid(resid, ddof=nparams)
assert result.statistic == pytest.approx(sup_b)
assert result.pvalue == pytest.approx(pvalue)
assert result.critical_values["5%"] == pytest.approx(1.36)
```

**Step 2: Implement `CUSUMTest`**

Use the shared design builder, fit OLS once, then call the statsmodels
diagnostic with `ddof=nparams`. Store the actual normalized cumulative process:

```python
scale = np.sqrt(np.sum(resid**2) * nobs / (nobs - nparams))
cusum = np.cumsum(resid) / scale
```

**Step 3: Define result and plotting contracts**

`CUSUMTestResult` adds:

- `critical_values`;
- `cusum`;
- `time_index`;
- `coefficients`, `fitted`;
- `plot_test(alpha=0.05)`.

The plot shows the scaled cumulative residual path and symmetric horizontal
critical limits. The summary states `H0: regression parameters are stable`.

**Step 4: Run CUSUM tests**

Run:

```powershell
python -m pytest TsTests/tests/test_cusum.py -q
```

Expected: exact numerical parity with statsmodels and passing stable/broken
seeded examples.

---

### Task 7: Implement the Bai-Perron multiple-unknown-break test

**Files:**

- Create: `TsTests/_bai_perron.py`
- Create: `TsTests/tests/test_bai_perron.py`

**Step 1: Test global partitioning rather than greedy detection**

Use seeded regressions with zero, one, two, and three breaks. Include a case
where greedy binary segmentation chooses the wrong partition but global
dynamic programming recovers the minimum-SSR partition.

**Step 2: Implement segment costs and dynamic programming**

Precompute admissible segment SSR values from cumulative cross-products, then
solve the globally optimal partition for every break count from zero through
`max_breaks`. Enforce `min_segment_size`/`trim`, full column rank, and positive
residual degrees of freedom for every segment.

**Step 3: Select the break count and expose inference**

Return all candidate partitions and criteria. Support BIC and LWZ break-count
selection. Report supF(1|0), sequential supF(l+1|l), UDmax and WDmax. For
supF(l+1|l), hold the globally estimated l-break partition fixed and search
each of its l+1 segments for one additional admissible break; do not compare
two separately re-optimized partitions. Because their distributions are
non-standard and the published tables cover only a finite grid of trimming,
regressor-count, and break-count combinations, use a reproducible Rademacher
wild bootstrap for p-values and critical values; never substitute ordinary
pointwise F p-values.

**Step 4: Estimate regimes and break-date intervals**

Store break indices, original time labels, segment coefficients, residuals,
fitted values, SSR, and confidence intervals. Construct ordered break-date
intervals from the selected partition's wild-bootstrap distribution. State
explicitly that this is heteroskedasticity robust but not serial-correlation
robust.

**Step 5: Cross-check against an independent implementation**

Compare the dynamic-programming partitions to independent exhaustive
enumeration for fixed one- and two-break datasets, and keep the indexing and
BIC/LWZ contracts aligned with `strucchange::breakpoints`. R remains an
algorithmic reference, not a runtime dependency.

**Step 6: Run Bai-Perron tests**

Run:

```powershell
python -m pytest TsTests/tests/test_bai_perron.py -q
```

Expected: global partitions, criteria, inference metadata, intervals, and
original-label mapping all pass.

---

### Task 8: Implement the Lee-Strazicich two-unknown-break LM unit-root test

**Files:**

- Create: `TsTests/_lee_strazicich.py`
- Create: `TsTests/tests/test_lee_strazicich.py`

**Step 1: Lock down Model A and Model C designs**

Model A includes two level-shift dummies. Model C includes two level shifts
and two trend shifts. Construct the detrended LM series exactly from
first-differenced deterministic regressors before estimating the augmented
LM regression.

**Step 2: Search all admissible break pairs**

Search both break indices jointly over the trimmed interval. Enforce the
model-specific minimum separation and enough observations for the selected
lag order. The reported statistic is the minimum t-statistic on the lagged
detrended level.

**Step 3: Implement explicit lag selection**

Support fixed lags plus AIC, BIC, and general-to-specific t-statistic selection
over `0..max_lags`. Record the selected lag at the minimizing break pair.

**Step 4: Apply the published critical-value contract**

For Model A, use the break-location-invariant Lee-Strazicich critical values.
For Model C, select critical values from the two-break-fraction cells in the
published table. Expose the matched cell and both break fractions in the
result.

**Step 5: Cross-check fixed reference cases**

Compare statistics, lags, break pairs, and critical values with an independent
Lee-Strazicich implementation on fixed datasets. Store the reference numbers
in tests; do not add a runtime dependency on that implementation.

**Step 6: Run Lee-Strazicich tests**

Run:

```powershell
python -m pytest TsTests/tests/test_lee_strazicich.py -q
```

Expected: Model A/Model C design, two-dimensional search, lag selection,
critical-value cell selection, and original-label mapping all pass.

---

### Task 9: Publish API and documentation

**Files:**

- Modify: `TsTests/__init__.py`
- Modify: `TsTests/README.md`
- Modify: `__init__.py`
- Create: `TsTests/tests/test_structural_break_exports.py`

**Step 1: Export all new classes**

```python
from ._chow import ChowTest, ChowTestResult
from ._cusum import CUSUMTest, CUSUMTestResult
from ._bai_perron import BaiPerronTest, BaiPerronTestResult
from ._lee_strazicich import (
    LeeStrazicichTwoBreakTest,
    LeeStrazicichTwoBreakTestResult,
)
```

Add the same eight names to both `__all__` lists.

**Step 2: Correct package taxonomy**

Document two separate groups:

1. structural-break unit-root tests: Perron and Zivot-Andrews;
2. regression parameter-stability tests: Chow, OLS-CUSUM, and Bai-Perron.

Do not describe Perron/Zivot-Andrews as direct tests that a generic regression
coefficient changed.

Also reconcile current README/API drift: `ZivotAndrewsTest` defaults to
`model="intercept"` and accepts `"slope"`, not the documented
`model="both"` / `"trend"` combination.

**Step 3: Document assumptions and method selection**

Include a compact decision table:

- known date + coefficient stability -> Chow;
- unknown instability + asymptotic p-value -> OLS-residual CUSUM;
- multiple unknown regression breaks -> Bai-Perron;
- unit root allowing one known break -> Perron;
- unit root allowing one unknown break -> Zivot-Andrews.
- unit root allowing two unknown breaks -> Lee-Strazicich.

State the Chow homoskedasticity/independence assumptions and the CUSUM
regressor-distribution caveat. Add executable examples for array and DataFrame
input.

**Step 4: Run export and documentation examples**

Run:

```powershell
python -m pytest TsTests/tests/test_structural_break_exports.py -q
python -c "import doctest; import Ts.TsTests as m; raise SystemExit(doctest.testmod(m).failed)"
```

Expected: all public names import from both `Ts.TsTests` and `Ts`.

---

### Task 10: Run package-wide verification

**Files:**

- No production changes unless a test exposes a defect caused by this work.

**Step 1: Format check**

```powershell
python -m ruff format --check TsTests __init__.py
```

Expected: exit code 0.

**Step 2: Lint**

```powershell
python -m ruff check TsTests __init__.py
```

Expected: exit code 0.

**Step 3: TsTests suite**

```powershell
python -m pytest TsTests/tests -q
```

Expected: all tests pass; no unexpected warnings.

**Step 4: TsTests branch coverage**

```powershell
python -m pytest TsTests/tests --cov=TsTests --cov-branch --cov-report=term-missing --cov-fail-under=90 -q
```

Expected: at least 90% branch coverage.

**Step 5: Full repository suite**

```powershell
python -m pytest -q
```

Expected: all repository tests pass; only previously characterized external
library warnings, if any, remain.

**Step 6: Diff integrity**

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only the approved structural-break files and
this plan are modified. Do not commit unless the user separately requests it.
