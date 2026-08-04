# Public API Help Docstrings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every top-level public Ts callable show complete parameter, return-object, and runnable usage documentation through IPython `?name`/`name?` and Python `help(name)`, then remove the five demo notebooks after they have served as verified migration fixtures.

**Architecture:** Keep documentation on the existing function, class, and public-method definitions; add no runtime documentation registry or duplicate implementation. Use one repository-level contract test to enforce NumPy-style sections and signature coverage, migrate each distinct notebook workflow to its owning API, and delete notebooks only after their clean execution and all final checks pass.

**Tech Stack:** Python docstrings, NumPy documentation convention, `inspect`, `doctest`, pytest, IPython, nbformat/nbclient, Ruff.

---

### Task 1: Freeze the public-documentation contract

**Files:**
- Create: `tests/test_public_docstrings.py`
- Modify: `pyproject.toml`

1. Add a test that imports every distinct name exported by the six subpackage
   `__all__` lists, plus the two style helpers used by the plotting demo, and
   requires every callable to have a summary and `Examples` section.
2. For functions and public constructors, compare `inspect.signature()` parameters with the NumPy-style `Parameters` section, excluding `self`, `cls`, `*args`, and `**kwargs` only when appropriate.
3. Require functions to document `Returns` unless they return `None`; require result containers to document `Attributes`.
4. Automatically cover every public method defined by an exported class, and
   explicitly include demo-facing methods inherited from private implementation
   bases (`fit`, prediction, summaries, plotting, diagnostics, evaluation, and
   result accessors).
5. Add `tests` to pytest discovery and run the new test, expecting failure before docstrings are migrated.

### Task 2: Complete TsPlots and TsUtils help

**Files:**
- Modify: `TsPlots/acf_plot.py`, `TsPlots/sc_plot.py`, `TsPlots/ts_plot.py`
- Modify: `TsUtils/_boxcox.py`, `TsUtils/_difference.py`, `TsUtils/_eacf.py`, `TsUtils/_interpolation.py`, `TsUtils/_stl.py`, `TsUtils/_summary.py`
- Modify: `TsPlots/README.md`, `TsUtils/README.md`

1. Move the plotting examples currently visible only at module level into the four public plotting functions.
2. Add complete examples for all public utility functions, constructors, result attributes, and demonstrated result methods.
3. Preserve existing signatures and behavior.
4. Run the focused docstring tests and existing `TsPlots`/`TsUtils` tests.

### Task 3: Complete TsTests help

**Files:**
- Modify public definitions in `TsTests/_base.py`, `_adf.py`, `_bai_perron.py`, `_chow.py`, `_cusum.py`, `_engle_lm.py`, `_johansen.py`, `_kpss.py`, `_lee_strazicich.py`, `_ljungbox.py`, `_normality.py`, `_perron.py`, `_phillips_perron.py`, `_toda_yamamoto.py`, `_zivot.py`
- Modify: `TsTests/README.md`

1. Document every exported test and result class, including hypotheses, constructor parameters, result attributes, and `fit`/`summary`/`plot_test` examples.
2. Cover public tests not present in the notebook with minimal deterministic examples.
3. Run the focused docstring tests and all `TsTests` tests.

### Task 4: Complete TsSims help

**Files:**
- Modify public definitions in `TsSims/_base.py`, `_cointegration.py`, `_garch.py`, `_garch_ext.py`, `_garch_result.py`, `_rdl.py`, `_sarima.py`, `_ts_ds.py`
- Modify: `TsSims/README.md`

1. Add deterministic examples for every exported simulation function and result/specification class.
2. Document shared result inspection methods (`get_data`, `get_params`, `summary`, `plot`) and specialized accessors.
3. Run the focused docstring tests and all `TsSims` tests.

### Task 5: Complete TsMetrics help

**Files:**
- Modify public definitions in `TsMetrics/_metrics.py`, `_oos.py`, `_backtest.py`, `_compare.py`, `_results.py`
- Modify: `TsMetrics/README.md`

1. Expand all metric one-line docstrings into full parameter/return/edge-case documentation.
2. Document evaluation functions and all exported result containers, including tables, rankings, period fields, and plotting examples.
3. Run the focused docstring tests and all `TsMetrics` tests.

### Task 6: Complete TsModels help without overwriting existing work

**Files:**
- Modify public definitions in `TsModels/_base.py`, `_auto.py`, `_backcast.py`, `_compare.py`, `_distributed_lag.py`, `_garch.py`, `_garch_result.py`, `_intervention.py`, `_sarimax.py`, `_svar.py`, `_var.py`, `_vecm.py`
- Modify: `TsModels/README.md`

1. Preserve the existing uncommitted Series/one-dimensional exogenous-variable changes in `_auto.py`, `_sarimax.py`, their tests, README, and notebook.
2. Add complete model constructor and result-object documentation.
3. Add examples for fitting, forecasting, diagnostics, roots, intervention effects, distributed lags, VAR/VECM/SVAR analysis, automatic selection, OOS/backtest/backcast, and result inspection.
4. Run the focused docstring tests and all `TsModels` tests.

### Task 7: Execute examples and verify interactive help

**Files:**
- Modify only docstrings/tests if verification exposes defects.

1. Execute docstring examples with a non-interactive Matplotlib backend.
2. Run IPython help smoke checks for representative function, model, result, and method objects and require `Parameters`/`Returns` or `Attributes`/`Examples` in the rendered text.
3. Execute all five `demo.ipynb` notebooks from clean kernels using nbclient without changing their saved outputs.
4. Run the complete pytest suite, Ruff, compileall, and `git diff --check`.

### Task 8: Remove superseded demos and repeat final verification

**Files:**
- Delete after Task 7 passes: `TsPlots/demo.ipynb`, `TsUtils/demo.ipynb`, `TsTests/demo.ipynb`, `TsSims/demo.ipynb`, `TsModels/demo.ipynb`
- Modify: package READMEs only where demo links must be replaced by interactive-help guidance.

1. Verify the five exact deletion targets are inside the repository and are the notebooks already executed in Task 7.
2. Delete only those five notebooks.
3. Remove or replace stale README links.
4. Repeat docstring tests, full pytest, Ruff, compileall, import/export checks, IPython help smoke checks, and `git diff --check`.
5. Report preserved pre-existing changes separately from documentation changes and deleted notebooks.

## Final verification

- 109 public callables and 128 public methods are protected by 237 structural
  help tests.
- 1,191 docstring statements executed successfully from isolated doctest
  namespaces.
- All five notebooks executed from clean kernels before deletion (324 cells,
  zero execution errors).
- The post-deletion repository passed 1,550 pytest tests, Ruff, compileall,
  IPython help smoke checks, and `git diff --check`.
