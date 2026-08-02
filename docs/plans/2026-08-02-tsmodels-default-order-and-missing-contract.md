# TsModels Default Order and Missing Contract Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change `SARIMAX`'s default non-seasonal order to `(0, 0, 0)` and change the TsModels-wide public missing-data default to `missing="drop"`, while preserving explicit strict failure through `missing="raise"`.

**Architecture:** Treat this as a public API contract migration, not a local workaround. Update every public TsModels constructor and the public VAR/VECM order-selection entry points, keep already-clean internal clone/candidate paths strict, and lock the behavior down with cross-model regression tests plus documentation.

**Tech Stack:** Python 3, NumPy, pandas, statsmodels, pytest, Ruff.

---

### Task 1: Write regression tests for the new public defaults

**Files:**
- Modify: `TsModels/tests/test_sarimax.py`
- Modify: `TsModels/tests/test_sarimax_exog.py`
- Modify: `TsModels/tests/test_missing_policy.py`

**Step 1: Add the SARIMAX default-order test**

Add a constructor-level assertion that omitting `order` stores `(0, 0, 0)`:

```python
def test_default_order_is_zero_zero_zero(ar1_data):
    model = SARIMAX(ar1_data)
    assert model.order == (0, 0, 0)
```

Keep existing tests that explicitly request `(1, 0, 0)` unchanged; they test AR(1), not the default.

**Step 2: Make the missing-policy factories exercise actual defaults**

Refactor the univariate and multivariate test factories so they accept `**kwargs`. Test these three distinct contracts for `SARIMAX`, `AutoSARIMAX`, `GARCH`, `AutoGARCH`, `VAR`, `SVAR`, and `VECM`:

```python
model = factory(data)
assert model.missing == "drop"
assert model.dropped_positions == (4, 17)
assert np.all(np.isfinite(model.data))

with pytest.raises(ValueError, match="row positions: 4, 17"):
    factory(data, missing="raise")
```

Retain the invalid-policy test for `missing="omit"`.

**Step 3: Test joint SARIMAX endog/exog/date deletion under the default**

Change the existing SARIMAX exogenous-data test to omit `missing="drop"` and assert that missing rows in either `y` or historical `exog` are jointly removed, dates stay synchronized, and `dropped_positions` records the original zero-based positions. Split the old default-raise test into an explicit `missing="raise"` test covering both `NaN` and infinity.

**Step 4: Test public order-selection defaults**

Update `VAR.select_order(...)` to assert default row deletion and explicit `missing="raise"` failure. Add the same contract test for `VECM.select_order(...)` so constructors and selectors do not diverge.

**Step 5: Run the new tests and verify they fail against the old contract**

Run:

```powershell
python -m pytest TsModels/tests/test_sarimax.py TsModels/tests/test_sarimax_exog.py TsModels/tests/test_missing_policy.py -q
```

Expected before implementation: failures showing default `order == (1, 0, 0)` and default missing-data rejection instead of deletion.

### Task 2: Change the production defaults without weakening internal invariants

**Files:**
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/_garch.py`
- Modify: `TsModels/_garch_base.py`
- Modify: `TsModels/_var.py`
- Modify: `TsModels/_svar.py`
- Modify: `TsModels/_vecm.py`

**Step 1: Change SARIMAX's public defaults**

In `SARIMAX.__init__`, change:

```python
order = (0, 0, 0)
missing = "drop"
```

Align the private `_normalise_sarimax_inputs(..., missing="drop")` boundary so direct internal use cannot drift from the constructor contract.

**Step 2: Change every other public model constructor to default-drop**

Change `missing="raise"` to `missing="drop"` in these public constructors:

- `AutoSARIMAX`
- `GARCH`
- `AutoGARCH`
- `VAR`
- `SVAR`
- `VECM`

Also align `_BaseVolModel`'s private constructor default because it is the shared GARCH/AutoGARCH normalization boundary.

**Step 3: Change public selector defaults**

Change both `VAR.select_order(...)` and `VECM.select_order(...)` to default to `missing="drop"`.

**Step 4: Preserve explicit strict checks on cleaned internal data**

Do not mechanically replace these internal calls:

- `SARIMAX._clone_for_evaluation(..., missing="raise")`
- `AutoSARIMAX._clone_for_evaluation(..., missing="raise")`
- AutoSARIMAX candidate construction with `missing="raise"`

Those paths receive data already normalized by the public boundary. Keeping them strict detects internal alignment defects instead of silently shortening an evaluation window or candidate sample.

**Step 5: Run the focused tests**

Run:

```powershell
python -m pytest TsModels/tests/test_sarimax.py TsModels/tests/test_sarimax_exog.py TsModels/tests/test_missing_policy.py -q
```

Expected: all selected tests pass; explicit `missing="raise"` still rejects `NaN` and infinity.

### Task 3: Update the public documentation contract

**Files:**
- Modify: `TsModels/README.md`
- Modify: `TsModels/_sarimax.py`
- Modify: `TsModels/_auto.py`
- Modify: `TsModels/_garch.py`
- Modify: `TsModels/_var.py`
- Modify: `TsModels/_svar.py`
- Modify: `TsModels/_vecm.py`

**Step 1: Update model docstrings**

Document `SARIMAX.order` as defaulting to `(0, 0, 0)`. For every public model class, state that `missing="drop"` is the default and that `missing="raise"` remains the explicit fail-fast option.

**Step 2: Update the README's unified contract**

Revise the top-level missing-data section to say:

- `NaN` and infinite rows are dropped by default;
- row deletion is joint across modeled variables/exogenous regressors;
- original zero-based positions remain available through `dropped_positions`;
- callers who must prevent silent sample changes should pass `missing="raise"`.

Update all current API signatures and parameter tables, including `SARIMAX`, `AutoSARIMAX`, `GARCH`, `AutoGARCH`, `VAR`, `SVAR`, and `VECM`.

**Step 3: Preserve historical and explicit examples**

Do not rewrite old files under `docs/plans/`. Do not change examples that explicitly use `order=(1, 0, 0)` to estimate an AR(1); only default-signature documentation changes.

**Step 4: Check for stale current-contract text**

Run:

```powershell
rg -n 'Default ``"raise"``|default is.*raise|missing="raise"|order=\(1, 0, 0\)' TsModels --glob '*.py' --glob '*.md'
```

Expected: remaining `missing="raise"` references are intentional strict examples/internal calls, and remaining explicit `(1, 0, 0)` references describe AR(1) models rather than the constructor default.

### Task 4: Validate the complete change

**Files:**
- Verify only; no new files expected.

**Step 1: Run all TsModels tests**

```powershell
python -m pytest TsModels/tests -q
```

Expected: exit code 0 with no failures.

**Step 2: Run static checks**

```powershell
python -m ruff check TsModels
python -m compileall -q TsModels
git diff --check
```

Expected: all commands exit with code 0 and `git diff --check` prints no whitespace errors.

**Step 3: Run the repository quality gate**

```powershell
python scripts/check_ts_quality.py
```

Expected: formatting, lint, tests, and configured coverage gates all pass.

**Step 4: Inspect the final diff**

Confirm that the diff contains only the approved default-contract implementation, regression tests, current documentation, and this plan. Do not commit unless the user separately requests a commit.
