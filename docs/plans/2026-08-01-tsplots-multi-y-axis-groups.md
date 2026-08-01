# TsPlots Multi Y-Axis Groups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Extend `plot_series(..., facet=False)` from at most two Y axes to automatic or user-defined scale groups with multiple right-side axes.

**Architecture:** Preserve the current robust scale metric and `auto_dual_y` compatibility, but replace the binary split with an ordered list of axis groups. Automatic mode groups adjacent scales whose ratio is below the threshold and merges the closest neighboring groups when the configured axis limit is exceeded. Manual grouping supplies an exact series-to-group mapping and overrides automatic detection.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, pytest, Jupyter/nbformat.

---

### Task 1: Define the multi-axis grouping contract with failing tests

**Files:**
- Modify: `TsPlots/tests/test_plots.py`

**Steps:**
1. Add tests proving three distinct scales create three axes by default.
2. Add tests proving similar scales remain on one axes group.
3. Add tests proving `axis_groups` overrides automatic scale grouping.
4. Add tests for `max_y_axes`, first-series-left ordering, `ax.right_ax`, `ax.extra_y_axes`, and shifted right spines.
5. Add validation tests for missing/unknown manual labels, unhashable group identifiers, excessive manual groups, invalid `max_y_axes`, and `facet=True` conflicts.
6. Run `python -m pytest TsPlots/tests/test_plots.py::TestPlotSeries -p no:cacheprovider -q`; expect failures caused by missing new parameters and multi-axis behavior.

### Task 2: Implement automatic and manual axis groups

**Files:**
- Modify: `TsPlots/ts_plot.py`
- Verify: `TsModels/_base.py`
- Verify: `TsModels/_var.py`
- Verify: `TsSims/_base.py`

**Steps:**
1. Add `axis_groups=None` and `max_y_axes: int = 3` to `plot_series`.
2. Validate `max_y_axes` as an integer of at least 1.
3. Validate manual mappings against final series labels; every label must appear exactly once and manual group count must not exceed `max_y_axes`.
4. Replace the binary scale split with automatic adjacent-ratio grouping; merge the closest neighboring groups until the count is within `max_y_axes`.
5. Keep the first series group on the primary axes. Create remaining axes with `twinx()`, move additional right spines outward, and reserve figure space.
6. Expose the first right axes as `ax.right_ax` and all right axes as `ax.extra_y_axes`.
7. Plot each series on its assigned axes, apply unit/ymin/style to every axes, and combine all line handles in the primary legend.
8. Run the targeted tests, then `python -m pytest TsPlots/tests -p no:cacheprovider -q` and the affected model/simulation tests; expect all to pass.

### Task 3: Document, demonstrate, and verify

**Files:**
- Modify: `TsPlots/README.md`
- Modify: `TsPlots/demo.ipynb`
- Modify: `docs/plans/2026-08-01-tsplots-multi-y-axis-groups.md`

**Steps:**
1. Document `axis_groups`, `max_y_axes`, automatic grouping/merging, manual precedence, validation, and access to every axes.
2. Add executable demo cells for automatic three-axis grouping and manual grouping.
3. Execute the notebook top-to-bottom with `python -m jupyter execute TsPlots/demo.ipynb --inplace --timeout=180` and verify no error outputs.
4. Run `python -m pytest -p no:cacheprovider -q`; expect the full repository to pass.
5. Run `python -m ruff check TsPlots TsModels TsSims`, `python -m ruff format --check TsPlots TsModels TsSims`, and `git diff --check`; expect all checks to pass.
6. Review `git status --short` and confirm the pre-existing `TsUtils/demo.ipynb` and Box-Cox plan changes remain untouched.

## Implementation Result

- Automatic grouping now creates up to `max_y_axes` scale groups and merges the closest adjacent groups when necessary.
- `axis_groups` provides strict manual grouping and overrides automatic detection.
- The first right axes remains available as `ax.right_ax`; all right axes are exposed through `ax.extra_y_axes`.
- Multiple right spines are shifted outward and the figure reserves additional right margin.
- `TsPlots/demo.ipynb` contains 49 cells (22 executable) and runs top-to-bottom without errors.
- Final checks: `1228 passed`, Ruff lint/format passed, and `git diff --check` passed.
