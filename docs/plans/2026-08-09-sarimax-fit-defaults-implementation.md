# SARIMAX Fit Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Make `maxiter=500`, `cov_type="oim"`, and `require_convergence=True` the shared defaults for ordinary SARIMAX, RDL, and every AutoSARIMAX candidate.

**Architecture:** Reuse the single public `SARIMAX.fit()` estimation path already shared by ordinary SARIMAX and RDL. Change only its keyword defaults and documentation; keep the existing validators, statsmodels/private-RDL forwarding, convergence gate, and AutoSARIMAX no-argument candidate fit calls.

**Tech Stack:** Python, NumPy, statsmodels, pytest, Ruff.

---

### Task 1: Lock ordinary SARIMAX defaults and override behavior

**Files:**
- Modify: `TsModels/tests/test_sarimax.py`

**Step 1: Write the failing default-forwarding test**

Add a monkeypatched statsmodels backend test beside
`test_fit_forwards_optimizer_controls_and_copies_start_params`:

```python
def test_fit_uses_reliable_defaults(self, ar1_data, monkeypatch):
    import Ts.TsModels._sarimax as sarimax_module

    captured = {}

    class StopAfterCapture(RuntimeError):
        pass

    class CapturingSARIMAX:
        def __init__(self, *args, **kwargs):
            self.k_params = 3

        def fit(self, **kwargs):
            captured.update(kwargs)
            raise StopAfterCapture

    monkeypatch.setattr(sarimax_module, "StatsmodelsSARIMAX", CapturingSARIMAX)
    model = sarimax_module.SARIMAX(ar1_data, order=(1, 0, 0))

    with pytest.raises(StopAfterCapture):
        model.fit()

    assert captured["method"] == "bfgs"
    assert captured["maxiter"] == 500
    assert captured["cov_type"] == "oim"
    assert captured["disp"] is False
```

Change `test_require_convergence_rejects_optimizer_failure` so its call omits
`require_convergence=True`; retain explicit `method="powell"` and `maxiter=7`
to keep the error-message assertion focused on the default convergence gate.

**Step 2: Run the ordinary tests and verify failure**

Run:

```powershell
python -m pytest TsModels/tests/test_sarimax.py::TestSARIMAX::test_fit_uses_reliable_defaults TsModels/tests/test_sarimax.py::TestSARIMAX::test_require_convergence_rejects_optimizer_failure -p no:cacheprovider -q
```

Expected: the default-forwarding assertion reports `50`/`"opg"`, and the
non-converged call does not raise at the convergence gate.

**Step 3: Preserve the explicit override regression**

Keep the existing forwarding test with explicit `maxiter=250` and
`cov_type="oim"`. Add a lightweight complete fake fitted result if needed to
assert that `require_convergence=False` permits a deliberately non-converged
result to reach result construction; do not add a new public helper solely for
testing.

### Task 2: Lock RDL and AutoSARIMAX reuse

**Files:**
- Modify: `TsModels/tests/test_distributed_lag.py`
- Modify: `TsModels/tests/test_auto.py`

**Step 1: Write the failing RDL test**

Extend `test_rdl_require_convergence_rejects_optimizer_failure` so the patched
RDL backend captures keyword arguments and the public call is simply
`model.fit()`. Assert before returning the non-converged fake that the backend
received:

```python
assert kwargs["method"] == "bfgs"
assert kwargs["maxiter"] == 500
assert kwargs["cov_type"] == "oim"
assert kwargs["disp"] is False
```

The existing `RuntimeError` assertion then proves that convergence is required
by default for RDL as well.

**Step 2: Write the AutoSARIMAX inheritance test**

Monkeypatch `Ts.TsModels._sarimax.SARIMAX.fit` with a recording method that
raises a sentinel exception. Run a one-candidate `AutoSARIMAX.fit()`, expect
the existing final `RuntimeError("No model converged during grid search")`,
and assert the candidate called `fit()` without overriding keyword arguments.
This locks the intended reuse: the candidate receives the public SARIMAX
defaults instead of a duplicate AutoSARIMAX configuration.

**Step 3: Run the new tests and verify failure**

Run:

```powershell
python -m pytest TsModels/tests/test_distributed_lag.py::test_rdl_require_convergence_rejects_optimizer_failure TsModels/tests/test_auto.py::TestAutoSARIMAX::test_candidates_inherit_sarimax_fit_defaults -p no:cacheprovider -q
```

Expected: the RDL default assertions fail before implementation; the Auto test
passes as a contract guard for the existing no-argument delegation.

### Task 3: Change the single public fit contract

**Files:**
- Modify: `TsModels/_sarimax.py:2638`

**Step 1: Update the public signature**

Change only the three defaults:

```python
def fit(
    self,
    *,
    start_params=None,
    method="bfgs",
    maxiter=500,
    cov_type="oim",
    require_convergence=True,
):
```

Do not branch on `distributed_lags` and do not duplicate defaults in
`AutoSARIMAX`.

**Step 2: Update the docstring defaults and semantics**

Document `maxiter` as default 500, `cov_type` as default `"oim"`, and
`require_convergence` as default True. State that callers can explicitly pass
`False` to inspect a non-converged result.

**Step 3: Run all three focused test modules**

Run:

```powershell
python -m pytest TsModels/tests/test_sarimax.py TsModels/tests/test_distributed_lag.py TsModels/tests/test_auto.py -p no:cacheprovider -q
```

Expected: all tests pass. Any real fixture that intentionally tolerates a
non-converged fit must opt out explicitly with `require_convergence=False`;
do not weaken the new public default.

### Task 4: Synchronize user-facing documentation

**Files:**
- Modify: `TsModels/README.md:405`
- Modify: `TsModels/README.md:423`
- Modify: `TsModels/README.md:454`

**Step 1: Add the fit signature and default table**

Immediately after the SARIMAX constructor table, document:

```python
SARIMAX(...).fit(
    method="bfgs",
    maxiter=500,
    cov_type="oim",
    require_convergence=True,
)
```

Explain that these defaults apply equally to ordinary SARIMAX, RDL, and
AutoSARIMAX candidates, while explicit arguments override them.

**Step 2: Simplify the RDL examples**

Where an RDL example repeats exactly the new defaults, replace the multiline
`.fit(...)` block with `.fit()`. Retain explicit fitting arguments only where
the example is intentionally demonstrating override behavior.

**Step 3: Check documentation references**

Run:

```powershell
rg -n 'maxiter=50|cov_type="opg"|require_convergence=False|default 50|default False' TsModels docs -g '*.py' -g '*.md'
```

Expected: no stale statement claims the old `SARIMAX.fit()` defaults. Historical
design documents may describe their original contract and must not be rewritten.

### Task 5: Validate quality and isolate this change from existing work

**Files:**
- Verify: `TsModels/_sarimax.py`
- Verify: `TsModels/tests/test_sarimax.py`
- Verify: `TsModels/tests/test_distributed_lag.py`
- Verify: `TsModels/tests/test_auto.py`
- Verify: `TsModels/README.md`

**Step 1: Run focused quality checks**

Run:

```powershell
python -m ruff check TsModels/_sarimax.py TsModels/tests/test_sarimax.py TsModels/tests/test_distributed_lag.py TsModels/tests/test_auto.py
python -m compileall -q TsModels
git diff --check
```

Expected: all commands exit 0.

**Step 2: Run the package test suite**

Run:

```powershell
python -m pytest TsModels/tests -p no:cacheprovider -q
```

Expected: all tests pass with no unexpected non-convergence failures.

**Step 3: Audit the final diff**

Inspect `git diff` and confirm the default-value task touches only the planned
lines. Preserve all pre-existing `log=True` worktree changes. Do not stage or
commit unrelated hunks; if exact staging cannot separate overlapping hunks,
leave the implementation uncommitted and report that explicitly.
