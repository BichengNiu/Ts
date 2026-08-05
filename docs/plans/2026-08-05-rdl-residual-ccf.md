# RDL Residual Cross-Correlation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Box-Jenkins residual cross-correlation diagnostics for fitted rational distributed-lag models, including lag-wise inference, the joint S* test, plots, and AutoSARIMAX delegation.

**Architecture:** Put all residual-CCF calculation and structured statistical results in `TsTests`. Reuse and expose the existing `TsPlots` correlogram renderer for precomputed correlations, while `SARIMAXResult` only validates fitted input prewhitening models, aligns innovations, counts fitted transfer-function parameters, and composes the test. Input ARIMA models remain explicit so diagnostics never hide model selection.

**Tech Stack:** Python 3.13, NumPy, pandas, SciPy, statsmodels 0.14.5, Matplotlib, pytest, Ruff.

---

### Task 1: Statistical core and result contract

**Files:**
- Create: `TsTests/_residual_ccf.py`
- Create: `TsTests/tests/test_residual_ccf.py`
- Modify: `TsTests/__init__.py`

**Steps:**
1. Write failing tests for the Box-Jenkins lag direction, biased cross-covariance convention, standard errors, confidence limits, S* statistic, chi-square degrees of freedom, multi-input access, summaries, and invalid inputs.
2. Run `python -m pytest TsTests/tests/test_residual_ccf.py -q -p no:cacheprovider` and verify collection/import failure.
3. Implement `ResidualCCFInputResult`, `ResidualCCFTestResult`, and `ResidualCCFTest` by composing `statsmodels.tsa.stattools.ccf(adjusted=False)` and `scipy.stats` inference.
4. Export the public classes and rerun the focused tests.

### Task 2: Shared correlogram plotting

**Files:**
- Modify: `TsPlots/acf_plot.py`
- Modify: `TsPlots/__init__.py`
- Modify: `TsPlots/tests/test_plots.py`
- Modify: `TsTests/_residual_ccf.py`
- Modify: `TsTests/tests/test_residual_ccf.py`

**Steps:**
1. Write failing tests for a public precomputed-correlogram plot, scalar and lag-varying bands, external axes, and multi-input facets.
2. Extract the existing ACF/PACF renderer into public `plot_correlogram` without changing ACF/PACF output.
3. Add `plot_test()` methods to residual-CCF results and verify bar values, bands, axes, titles, and input selection.
4. Run `python -m pytest TsPlots/tests/test_plots.py TsTests/tests/test_residual_ccf.py -q -p no:cacheprovider`.

### Task 3: SARIMAX and AutoSARIMAX integration

**Files:**
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/tests/test_distributed_lag.py`
- Modify: `TsModels/tests/test_auto.py`

**Steps:**
1. Write failing tests for `SARIMAXResult.residual_ccf_test()` and Auto delegation.
2. Validate that selected inputs are fitted RDL inputs and every supplied prewhitening model is a converged, univariate, input-only SARIMAX result fitted to the exact historical input path and calendar.
3. Align post-burn input innovations with final-model residuals at their common sample end and pass per-input RDL parameter counts to `ResidualCCFTest`.
4. Cover finite/rational/sparse RDL specifications, multiple inputs, mismatched series/dates, missing models, nonconvergence, and insufficient degrees of freedom.
5. Run the focused TsModels tests.

### Task 4: Public help and documentation

**Files:**
- Modify: `TsTests/README.md`
- Modify: `TsModels/README.md`
- Modify: `TsPlots/README.md`
- Modify: `tests/test_public_docstrings.py`

**Steps:**
1. Document the identification-versus-diagnostic prewhitening distinction, lag sign, `1/sqrt(n)` approximation, S* formula, per-input degrees of freedom, and non-causal interpretation.
2. Add runnable NumPy-style `Examples` for all new public objects and methods.
3. Add the new APIs to the public-help method audit.
4. Run public docstring, doctest, and IPython help smoke tests.

### Task 5: Numerical and regression validation

**Files:**
- Modify as required by failures only within the approved files above.

**Steps:**
1. Run all focused residual-CCF, RDL, AutoSARIMAX, plotting, and public-help tests.
2. Verify an omitted-lag simulation produces the expected residual-CCF peak and a correctly specified simulation does not systematically reject.
3. Render single- and multi-input plots under `MPLBACKEND=Agg` and inspect their artists and labels.
4. Run full pytest, Ruff, compileall, and `git diff --check`.
5. Audit plan-to-code coverage, failure paths, public exports, exact diff scope, and preservation of untracked user files.
6. Commit the verified change directly on `main`.
