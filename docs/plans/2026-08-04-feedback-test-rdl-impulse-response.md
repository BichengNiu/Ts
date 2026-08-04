# Feedback Test and RDL Impulse Response Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a conditional feedback F test and RDL impulse-response bar charts that reuse the existing SARIMAX, rational-lag, and TsPlots contracts.

**Architecture:** `TsTests` owns the standalone OLS feedback test and structured results. `TsModels` composes that test from stored SARIMAX response/exogenous data and composes `TsPlots.plot_lag_response` from the existing RDL `weights(steps)` output. No transfer-function or causality algorithm is duplicated.

**Tech Stack:** Python 3.13, NumPy, pandas, statsmodels 0.14.5, Matplotlib, pytest, Ruff.

---

### Task 1: Feedback-test contract

**Files:**
- Create: `TsTests/_feedback.py`
- Create: `TsTests/tests/test_feedback.py`
- Modify: `TsTests/__init__.py`
- Modify: `__init__.py`

**Steps:**
1. Write failing tests for named inputs, lag design, complete-case handling, exact OLS coefficients, exact joint F statistics, multi-input results, summaries, and invalid inputs.
2. Run `python -m pytest -q TsTests/tests/test_feedback.py` and verify collection/import failure.
3. Implement `FeedbackTest`, `FeedbackTestResult`, and `FeedbackEquationResult` using `BaseTest`, `BaseMultiTestResult`, statsmodels OLS, and `RegressionResults.f_test`.
4. Export the three public classes from `TsTests` and the package root.
5. Re-run the focused test and expect all tests to pass.

### Task 2: SARIMAX integration

**Files:**
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/tests/test_sarimax_exog.py`

**Steps:**
1. Write failing tests for `SARIMAXResult.feedback_test()` using stored original exogenous inputs, target subsetting, no-exog errors, and AutoModelResult delegation.
2. Run the exact new tests and verify failure.
3. Implement thin composition methods without reproducing feedback-test logic.
4. Re-run the exact tests and expect pass.

### Task 3: Shared lag-response plot

**Files:**
- Create: `TsPlots/lag_plot.py`
- Create: `TsPlots/tests/test_lag_plot.py`
- Modify: `TsPlots/__init__.py`
- Modify: `__init__.py`

**Steps:**
1. Write failing tests for Series/DataFrame input, zero reference line, integer lag ticks, facets, external axes, style, and invalid data.
2. Run `python -m pytest -q TsPlots/tests/test_lag_plot.py` and verify import failure.
3. Implement `plot_lag_response()` with existing `TsPlots.style` helpers.
4. Export it from `TsPlots` and the package root.
5. Re-run the focused plot tests and expect pass.

### Task 4: RDL plotting composition

**Files:**
- Modify: `TsModels/_distributed_lag.py`
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/tests/test_distributed_lag.py`

**Steps:**
1. Write failing tests for single-input `RationalLagResult.plot_impulse_response`, multi-input SARIMAX facets, selection/order, no-RDL errors, and AutoModelResult delegation.
2. Implement all methods by passing existing `weights(steps)` results to `TsPlots.plot_lag_response()`.
3. Run `python -m pytest -q TsModels/tests/test_distributed_lag.py` and expect pass.

### Task 5: Public help and documentation

**Files:**
- Modify: `TsTests/README.md`
- Modify: `TsPlots/README.md`
- Modify: `TsModels/README.md`
- Modify: `tests/test_public_docstrings.py`

**Steps:**
1. Document the feedback equation, joint null, predictive-not-structural interpretation, result accessors, and SARIMAX shortcut.
2. Document `plot_lag_response()` and RDL impulse-response examples.
3. Register new demo-facing methods in the public docstring contract.
4. Run `python -m pytest -q tests/test_public_docstrings.py` and expect pass.

### Task 6: Final verification

**Files:**
- Verify all changed files.

**Steps:**
1. Run focused tests for `TsTests`, the new plot module, RDL, SARIMAX exogenous behavior, and public docstrings.
2. Run `python -m ruff check TsTests TsPlots TsModels tests __init__.py`.
3. Run `python -m compileall -q TsTests TsPlots TsModels`.
4. Run `python -m pytest -q`; baseline before changes is `1550 passed`.
5. Run `git diff --check` and inspect `git diff --stat` plus `git status --short`.
6. Do not commit unless the user explicitly requests a commit.
