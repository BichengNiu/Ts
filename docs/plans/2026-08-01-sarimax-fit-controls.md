# SARIMAX Fit Controls Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Make SARIMAX optimization configurable and convergence observable without changing existing `fit()` callers.

**Architecture:** Keep model specification in `SARIMAX.__init__()` and expose only the fitting controls needed by statsmodels through keyword-only arguments. Preserve statsmodels warnings by default, add an opt-in convergence gate, and surface optimizer metadata from the retained statsmodels result. Restore an inferable `DatetimeIndex` frequency before constructing statsmodels so regular dated data does not emit avoidable frequency warnings.

**Tech Stack:** Python, NumPy, pandas, statsmodels, pytest, Ruff.

---

### Task 1: Lock the public fitting contract with tests

**Files:**
- Modify: `TsModels/tests/test_sarimax.py`

**Steps:**

1. Add a test that monkeypatches `StatsmodelsSARIMAX` and verifies `method`, `maxiter`, copied `start_params`, and `disp=False` are forwarded.
2. Add parameterized validation tests for an empty/non-string method, non-positive/boolean `maxiter`, and invalid/non-finite/wrong-length `start_params`.
3. Add tests that `require_convergence=True` raises on a non-converged statsmodels result while the default remains backward compatible.
4. Add a test that a regular DatetimeIndex with missing `.freq` is passed to statsmodels with its inferred frequency attached.
5. Run the focused tests and verify they fail for the expected missing API.

### Task 2: Implement fit controls and frequency restoration

**Files:**
- Modify: `TsModels/_sarimax.py`

**Steps:**

1. Add small private normalizers for `method`, `maxiter`, `start_params`, and `require_convergence`.
2. Change the public signature to `fit(*, start_params=None, method="lbfgs", maxiter=50, require_convergence=False)`.
3. Infer and attach frequency to regular model dates; retain the existing positional-index fallback when frequency cannot be inferred.
4. Forward the validated controls to `statsmodels.SARIMAX.fit()` with `disp=False`.
5. Raise `RuntimeError` with method, iteration limit, and optimizer return details only when convergence is required and not achieved.

### Task 3: Surface optimization diagnostics

**Files:**
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/tests/test_sarimax.py`

**Steps:**

1. Add read-only `converged`, `optimizer`, and copied `optimization_details` properties to `SARIMAXResult`.
2. Add `Optimizer` and `Converged` lines to `SARIMAXResult.summary()`.
3. Test defensive copying of optimizer details, including NumPy arrays.
4. Run all SARIMAX tests.

### Task 4: Validate the real failure case and package quality

**Files:**
- No additional production files.

**Steps:**

1. Refit the housing-permits quarterly dummy model with stable starting parameters and `require_convergence=True`; expect convergence and no frequency warning.
2. Run `python -m pytest TsModels/tests/test_sarimax.py TsModels/tests/test_sarimax_api.py TsModels/tests/test_sarimax_exog.py -p no:cacheprovider -q`.
3. Run `python scripts/check_ts_quality.py` if present; otherwise run full pytest plus Ruff on the changed files.
4. Run `git diff --check` and inspect the final diff without modifying unrelated user changes.

No commits are included because the worktree already contains unrelated user edits and no commit was requested.
