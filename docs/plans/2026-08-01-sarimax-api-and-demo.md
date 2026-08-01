# SARIMAX API and Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the TsModels SARIMA estimation API with a single SARIMAX API, add exogenous-aware automatic order selection, and make the TsModels demo complete, accurate, and executable.

**Architecture:** Rename the existing statsmodels-backed estimator and result to `SARIMAX` and `SARIMAXResult`, with no aliases, wrappers, deprecation warnings, or fallback imports for the removed TsModels names. Extend automatic selection as `AutoSARIMAX` by reusing the same input-normalisation, exogenous-design, prediction, and evaluation contracts as the manual estimator. Keep `TsSims.simulate_sarima()` and `SimSARIMAResult` because they describe a separate SARIMA data-generating process rather than a compatibility layer for the estimator.

**Tech Stack:** Python 3.13, NumPy, pandas, statsmodels 0.14.5, pytest, Ruff, Jupyter nbclient.

---

## Confirmed scope

- Remove the public TsModels symbols `SARIMA`, `SARIMAResult`, and `AutoSARIMA`.
- Do not add aliases, wrappers, `__getattr__` fallbacks, deprecated imports, or warning-based migration paths.
- Add and export `SARIMAX`, `SARIMAXResult`, and `AutoSARIMAX` from both `Ts.TsModels` and `Ts`.
- Set fitted and automatic-selection `model_type` to `"SARIMAX"` for every estimator configuration, including the no-exogenous special case.
- Rename estimator implementation and estimator-test files from `sarima` to `sarimax` where the file owns the estimation API.
- Keep the simulation API `simulate_sarima()` / `SimSARIMAResult` unchanged.
- Update the notebook source and regenerate all outputs from a clean top-to-bottom execution.
- Do not alter historical plans under `docs/plans/` merely to rewrite old terminology.

## Read-only audit baseline

- Relevant tests: `180 passed, 81 warnings`.
- Relevant Ruff check: `All checks passed!`.
- `TsModels/demo.ipynb` contains 98 cells: 46 code and 52 Markdown.
- All 46 saved code cells have execution counts and the saved notebook contains no error output, but those outputs were generated on 2026-07-24 and are stale.
- A current in-memory execution fails in code cell 41 because it reads `OOSResult.target_indices`; the current OOS contract exposes `validation_indices`.
- The notebook contains no occurrence of `exog`, `exog_names`, `future_exog`, `EventSpec`, `ScenarioForecastResult`, `PolicyEffectResult`, `policy_effect`, or dated SARIMAX input.
- Cells 19, 20, 79, and 80 contain stray `</cell ...>` / DSML editor artifacts.

---

### Task 1: Lock the breaking public API contract with tests

**Files:**
- Create: `TsModels/tests/test_sarimax_api.py`
- Modify: `TsModels/tests/test_contracts.py` if a shared public-export assertion already belongs there

**Step 1: Write failing public-surface tests**

Add tests equivalent to:

```python
def test_sarimax_types_are_the_only_public_arima_estimators():
    import Ts
    import Ts.TsModels as models

    assert Ts.SARIMAX is models.SARIMAX
    assert Ts.SARIMAXResult is models.SARIMAXResult
    assert Ts.AutoSARIMAX is models.AutoSARIMAX
    for removed in ("SARIMA", "SARIMAResult", "AutoSARIMA"):
        assert not hasattr(Ts, removed)
        assert not hasattr(models, removed)


def test_sarimax_fit_uses_canonical_result_and_model_type(ar1_data):
    from Ts.TsModels import SARIMAX, SARIMAXResult

    result = SARIMAX(ar1_data, order=(1, 0, 0)).fit()

    assert isinstance(result, SARIMAXResult)
    assert result.model_type == "SARIMAX"
    assert "SARIMAX Model Estimation Result" in result.summary()
```

Also assert that importing `Ts.TsModels._sarima` fails after the module move and that `Ts.TsModels._sarimax` exposes only the new estimator/result names.

**Step 2: Run tests and verify the intended failures**

Run:

```powershell
$env:PYTHONUTF8='1'
python -m pytest TsModels/tests/test_sarimax_api.py -p no:cacheprovider -q
```

Expected: fail because `SARIMAX`, `SARIMAXResult`, and `AutoSARIMAX` are not yet exported and the removed names still exist.

---

### Task 2: Rename the estimator implementation without compatibility paths

**Files:**
- Rename: `TsModels/_sarima.py` to `TsModels/_sarimax.py`
- Rename: `TsModels/tests/test_sarima.py` to `TsModels/tests/test_sarimax.py`
- Rename: `TsModels/tests/test_sarima_exog.py` to `TsModels/tests/test_sarimax_exog.py`
- Modify: `TsModels/_intervention.py`
- Modify: all Python imports listed by `rg -n "TsModels\._sarima|\bSARIMAResult\b|\bSARIMA\(" TsModels TsMetrics __init__.py -g "*.py"`

**Step 1: Rename canonical symbols and internal identifiers**

In `TsModels/_sarimax.py`:

```python
class SARIMAXResult(BaseModelResult): ...


class SARIMAX(BaseModel): ...
```

Rename `_SARIMAInputs` to `_SARIMAXInputs` and `_normalise_sarima_inputs()` to `_normalise_sarimax_inputs()`. Update docstrings, return annotations, plot titles, summaries, and coverage declarations. Do not define the removed public names anywhere in production code.

**Step 2: Make result identity unambiguous**

Construct:

```python
result = SARIMAXResult(
    model_type="SARIMAX",
    ...,
)
```

The order summary must use `Order: SARIMAX(...)`. Mathematical prose may state that a no-exogenous specification is the SARIMA special case, but API examples and result labels must use SARIMAX.

**Step 3: Update all estimator tests mechanically and semantically**

- Import from `Ts.TsModels._sarimax` or the public `Ts.TsModels` surface.
- Rename estimator/result test classes and coverage paths.
- Expect `model_type == "SARIMAX"` and SARIMAX headers.
- Preserve all existing input, sparse-lag, root, forecast, intervention, policy-effect, and evaluation assertions.

**Step 4: Run the renamed estimator suite**

Run:

```powershell
$env:PYTHONUTF8='1'
python -m pytest TsModels/tests/test_sarimax_api.py TsModels/tests/test_sarimax.py TsModels/tests/test_sarimax_exog.py TsModels/tests/test_intervention.py -p no:cacheprovider -q
```

Expected: all tests pass; no import resolves through `_sarima`.

---

### Task 3: Implement exogenous-aware AutoSARIMAX

**Files:**
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/tests/test_auto.py`
- Modify: `TsModels/tests/test_evaluation.py`
- Modify: `TsModels/tests/test_evaluation_periods.py`
- Modify: `TsModels/tests/test_missing_policy.py`

**Step 1: Write failing automatic-selection tests**

Cover:

```python
def test_auto_sarimax_passes_exog_to_every_candidate(dated_regression_data):
    y, exog = dated_regression_data
    result = AutoSARIMAX(
        y,
        exog=exog,
        p=(0, 1),
        d=(0, 0),
        q=(0, 0),
        trend="n",
    ).fit()

    assert result.model_type == "SARIMAX"
    assert result.best_result.exog_names == tuple(exog.columns)
    assert set(exog.columns) <= set(result.best_result.params)
```

Add separate tests for:

- DataFrame historical/future exogenous splitting;
- array exogenous input requiring `exog_names`;
- joint `missing="drop"` alignment across `y`, dates, and exogenous rows;
- event-design propagation to every candidate;
- default future exogenous prediction from `best_result`;
- named future scenarios delegated by `AutoModelResult.predict()`;
- OOS and backtest training/future exogenous windows without target leakage;
- precise early validation when exogenous data are invalid, rather than one failure message per grid candidate.

**Step 2: Extend the constructor explicitly**

Use one canonical constructor shaped as:

```python
class AutoSARIMAX(_BaseAutoModel):
    def __init__(
        self,
        data,
        p=(0, 3),
        d=(0, 1),
        q=(0, 3),
        P=(0, 1),
        D=(0, 1),
        Q=(0, 1),
        s=0,
        trend="c",
        criterion="aic",
        method="grid",
        *,
        dates=None,
        exog=None,
        exog_names=None,
        events=None,
        enforce_stationarity=True,
        enforce_invertibility=True,
        missing="raise",
    ): ...
```

Validate endogenous data, dates, ordinary exogenous data, future exogenous data, and missing rows once through the same helpers used by `SARIMAX`. Store copied arrays/frames and immutable names/specifications.

**Step 3: Build every candidate through SARIMAX**

The grid factory must call `SARIMAX(...)` with the candidate orders plus the validated trend, dates, exogenous design, events, and enforcement settings. Preserve default future exogenous rows on the selected `SARIMAXResult`.

**Step 4: Rebuild evaluation windows explicitly**

Implement `_clone_for_evaluation()` and `_evaluation_predict_kwargs()` so automatic re-selection uses only the training `y`, while passing exactly aligned historical and future exogenous rows. Do not rely on the shallow-copy default in `BaseModel` for derived design state.

**Step 5: Run automatic-selection and evaluation tests**

Run:

```powershell
$env:PYTHONUTF8='1'
python -m pytest TsModels/tests/test_auto.py TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py TsModels/tests/test_missing_policy.py -p no:cacheprovider -q
```

Expected: all tests pass, including AutoSARIMAX exogenous OOS/backtest coverage.

---

### Task 4: Publish only the new estimator API

**Files:**
- Modify: `TsModels/__init__.py`
- Modify: `__init__.py`
- Modify: `TsModels/_base.py`
- Modify: `TsMetrics/tests/test_evaluation.py`
- Modify: any remaining runtime/tests returned by the legacy-symbol scan

**Step 1: Replace exports and examples**

Export only:

```python
from ._auto import AutoGARCH, AutoModelResult, AutoSARIMAX
from ._sarimax import ARCycleResult, SARIMAX, SARIMAXResult, ScenarioForecastResult
```

Update both `__all__` lists and quick-start examples. Update generic model-type examples where they refer specifically to the renamed estimator.

**Step 2: Add removal assertions**

Run from the repository parent:

```powershell
$env:PYTHONUTF8='1'
python -c "import Ts, Ts.TsModels as m; assert all(hasattr(m, n) for n in ('SARIMAX','SARIMAXResult','AutoSARIMAX')); assert all(not hasattr(m, n) for n in ('SARIMA','SARIMAResult','AutoSARIMA')); assert all(not hasattr(Ts, n) for n in ('SARIMA','SARIMAResult','AutoSARIMA')); print('sarimax-public-api-ok')"
```

Expected: `sarimax-public-api-ok`.

---

### Task 5: Rebuild the README and notebook as verified SARIMAX documentation

**Files:**
- Modify: `TsModels/README.md`
- Modify: `TsMetrics/README.md`
- Modify: `TsModels/demo.ipynb`

**Step 1: Correct the documentation model**

Use `SARIMAX` consistently for the estimator. Explain once that setting `exog=None` yields the SARIMA special case; this is a statistical relationship, not an API fallback. Keep `simulate_sarima()` in examples only as a data generator.

**Step 2: Remove notebook corruption and stale outputs**

- Remove stray editor markup from cells 19, 20, 79, and 80.
- Clear all execution counts, outputs, and stale execution timestamps before rebuilding examples.
- Fix the OOS example to use `evaluation.validation_indices`, not `target_indices`.
- Describe inclusive estimation/validation bounds consistently with the actual OOS contract.

**Step 3: Cover the complete SARIMAX workflow with compact executable sections**

The notebook must include executable examples for:

1. SARIMAX without external regressors as the SARIMA special case;
2. seasonal and sparse AR/MA orders;
3. root, stationarity, invertibility, cycle, equilibrium, residual, and prediction diagnostics;
4. dated pandas `Series` plus DataFrame `exog` and named coefficient access;
5. a DataFrame containing both historical and future exogenous rows;
6. automatic default future-exogenous forecasting;
7. named baseline/stress future scenarios and `ScenarioForecastResult` access/plotting;
8. pulse, step, and event-window `EventSpec` designs;
9. `policy_effect()` with a fast deterministic example plus concise method/identification notes;
10. `AutoSARIMAX` with ordinary exogenous variables;
11. leakage-free dated OOS and backtest with correctly aligned exogenous context;
12. explicit missing/alignment contract notes and one concise validation-error example.

Do not duplicate the full API reference in every example. Each code cell must demonstrate one contract and assert at least one important result where practical.

**Step 4: Execute the complete notebook from a clean kernel**

Execute in memory first:

```powershell
$env:MPLBACKEND='Agg'
$env:PYTHONUTF8='1'
Push-Location ..
python -c "import nbformat; from nbclient import NotebookClient; p='Ts/TsModels/demo.ipynb'; nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=240, kernel_name='python3', resources={'metadata': {'path': '.'}}, allow_errors=False).execute(); print('tsmodels-demo-ok')"
Pop-Location
```

Expected: `tsmodels-demo-ok` with no `CellExecutionError`.

After that passes, execute and write the refreshed outputs to the notebook using the repository's available Jupyter tooling. Re-open the JSON and assert:

- every code cell has an execution count;
- no output has `output_type == "error"`;
- no cell contains `</cell`, `DSML`, or `parameter name=`;
- estimator imports use `SARIMAX` / `AutoSARIMAX` only.

---

### Task 6: Run repository-wide validation

**Files:**
- Verify all modified files

**Step 1: Scan for forbidden compatibility code**

Run:

```powershell
rg -n "class SARIMA\b|class SARIMAResult\b|class AutoSARIMA\b|from .* import .*\bSARIMA\b|from .* import .*\bAutoSARIMA\b|TsModels\._sarima" TsModels TsMetrics __init__.py -g "*.py"
```

Expected: no matches. `simulate_sarima` and `SimSARIMAResult` are explicitly allowed.

**Step 2: Run formatting and static checks**

```powershell
$env:PYTHONUTF8='1'
python -m ruff format --check .
python -m ruff check .
python -m compileall -q TsModels TsMetrics
git diff --check
```

Expected: every command exits 0.

**Step 3: Run all tests**

```powershell
$env:PYTHONUTF8='1'
python -m pytest -p no:cacheprovider -q
```

Expected: all tests pass. Report the exact pass/warning count; do not reuse the historical baseline.

**Step 4: Inspect final scope**

```powershell
git status --short --branch
git diff --stat
```

Expected: only SARIMAX implementation, dependent tests/docs, the rebuilt TsModels notebook, and this plan are modified. Do not create commits unless the user separately requests them.
