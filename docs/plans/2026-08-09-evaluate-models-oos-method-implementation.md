# `evaluate_models_oos` Optimizer Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow callers of `evaluate_models_oos()` to choose a supported model optimizer through `method=...` while preserving every estimator's current default when it is omitted.

**Architecture:** Extend the existing `evaluate_models_oos()` → `oos()` → `fit_and_forecast()` chain instead of adding a second fitting path. A shared introspection helper validates whether an estimator accepts `fit(method=...)`; the batch entry validates every named estimator before fitting any of them, while the model remains responsible for validating concrete optimizer names.

**Tech Stack:** Python 3.13, `inspect.signature`, NumPy, pytest, existing TsMetrics evaluation protocols, existing `SARIMAX.fit(method=...)`.

---

### Task 1: Add failing optimizer-forwarding contract tests

**Files:**
- Modify: `TsMetrics/tests/test_evaluation.py:49-89`
- Modify: `TsMetrics/tests/test_evaluation.py:477-518`

**Step 1: Add a method-aware test estimator**

Add a subclass beside `_MeanModel` whose shared list survives the shallow evaluation clone:

```python
class _OptimizerMeanModel(_MeanModel):
    """Mean model that records the optimizer selected for its clone."""

    def __init__(self, data, dates=None, *, methods=None, **kwargs):
        super().__init__(data, dates=dates, **kwargs)
        self.methods = [] if methods is None else methods

    def fit(self, *, method="bfgs"):
        self.methods.append(method)
        return super().fit()
```

**Step 2: Add tests for direct and batch forwarding**

Add tests proving that:

```python
def test_oos_forwards_explicit_fit_method():
    model = _OptimizerMeanModel(np.arange(15.0))
    result = oos(model, (0, 9), (10, 14), method="lbfgs")
    assert result.mean.shape == (5,)
    assert model.methods == ["lbfgs"]


def test_evaluate_models_oos_forwards_method_to_every_model():
    first = _OptimizerMeanModel(np.arange(15.0))
    second = _OptimizerMeanModel(np.arange(15.0), forecast_bias=6.5)
    report = evaluate_models_oos(
        {"first": first, "second": second},
        estimation_period=(0, 9),
        validation_period=(10, 14),
        method="powell",
    )
    assert report.table.shape[0] == 2
    assert first.methods == ["powell"]
    assert second.methods == ["powell"]
```

Use stable `OOSResult` assertions rather than depending on optimizer-specific numerical estimates.

**Step 3: Add the atomic incompatibility test**

Put the compatible model first and an existing `_MeanModel` second:

```python
def test_evaluate_models_oos_rejects_unsupported_method_before_fitting():
    supported = _OptimizerMeanModel(np.arange(15.0))
    unsupported = _MeanModel(np.arange(15.0))

    with pytest.raises(TypeError, match=r"unsupported.*method"):
        evaluate_models_oos(
            {"supported": supported, "unsupported": unsupported},
            estimation_period=(0, 9),
            validation_period=(10, 14),
            method="lbfgs",
        )

    assert supported.methods == []
    assert unsupported.fit_windows == []
```

The exact assertion must require the error to contain the incompatible mapping key `unsupported`.

**Step 4: Run tests and verify the new API is missing**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py -k "method or optimizer" -q
```

Expected: new tests fail because `oos()` and `evaluate_models_oos()` do not accept `method`; existing selected tests remain unaffected.

**Step 5: Commit the failing tests**

```powershell
git add -- TsMetrics/tests/test_evaluation.py
git commit -m "test: define OOS optimizer selection contract"
```

### Task 2: Thread `method` through the shared OOS fitting path

**Files:**
- Modify: `TsMetrics/_protocols.py:33-37`
- Modify: `TsMetrics/_evaluation.py:3-8`
- Modify: `TsMetrics/_evaluation.py:84-123`
- Modify: `TsMetrics/_oos.py:30-67`
- Modify: `TsMetrics/_oos.py:82-102`
- Modify: `TsMetrics/_compare.py:154-226`
- Modify: `TsModels/_base.py:891-930`

**Step 1: Add one shared fit-method compatibility helper**

Import `Parameter` and `signature` from `inspect` in `_evaluation.py`. Add:

```python
def validate_fit_method(model, method, *, model_name=None):
    """Return fit kwargs after validating explicit optimizer support."""
    if method is None:
        return {}
    if not isinstance(method, str):
        raise TypeError("method must be a string or None")
    fit = getattr(model, "fit", None)
    if not callable(fit):
        raise TypeError("model must provide fit()")
    try:
        parameters = signature(fit).parameters.values()
    except (TypeError, ValueError) as error:
        label = type(model).__name__ if model_name is None else repr(model_name)
        raise TypeError(
            f"model {label} does not expose an inspectable fit() supporting method"
        ) from error
    accepts_method = any(
        parameter.name == "method" or parameter.kind == Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not accepts_method:
        label = type(model).__name__ if model_name is None else repr(model_name)
        raise TypeError(f"model {label} does not support fit(method=...)")
    return {"method": method}
```

Do not validate the optimizer value against SARIMAX's optimizer list here; reuse the target model's existing validation.

**Step 2: Keep `fit_and_forecast()` as the only execution point**

Add a final optional `fit_kwargs=None` argument. Replace `fitted = fit()` with:

```python
fitted = fit(**({} if fit_kwargs is None else fit_kwargs))
```

Update `EvaluationCloneProtocol.fit()` to accept `**kwargs: Any`. Existing backtest calls omit `fit_kwargs`, so their behavior stays equivalent at the call boundary.

**Step 3: Extend direct OOS evaluation**

Change the public signature to:

```python
def oos(model, estimation_period, validation_period, *, alpha=0.05, method=None):
```

Call `validate_fit_method(model, method)` before the evaluation clone is fitted and pass the returned dictionary to `fit_and_forecast()`. Document `method` as an optional optimizer forwarded to `fit()` and state that `None` preserves model defaults.

**Step 4: Add batch preflight and forwarding**

Change `evaluate_models_oos()` to accept `method=None`. After mapping/name/rank validation and before the evaluations comprehension, run:

```python
for name, model in models.items():
    validate_fit_method(model, method, model_name=name)
```

Then pass `method=method` to every `oos()` call. This guarantees that an incompatible named model is reported before any earlier model is fitted.

**Step 5: Keep the estimator convenience method consistent**

Add `method=None` to `BaseModel.oos()` and forward it to `TsMetrics.oos()`. Update its NumPy-style parameter documentation and example.

**Step 6: Run focused tests**

Run:

```powershell
python -m pytest TsMetrics/tests/test_evaluation.py TsModels/tests/test_evaluation.py -q
```

Expected: all tests pass; the default `_MeanModel.fit()` tests prove that `method=None` does not force a keyword argument.

**Step 7: Commit implementation**

```powershell
git add -- TsMetrics/_protocols.py TsMetrics/_evaluation.py TsMetrics/_oos.py TsMetrics/_compare.py TsModels/_base.py
git commit -m "feat: select optimizer in OOS evaluation"
```

### Task 3: Update public documentation and executable examples

**Files:**
- Modify: `TsMetrics/README.md:160-183`
- Modify: `TsModels/README.md:207-270`
- Modify: `docs/plans/2026-08-09-evaluate-models-oos-method-design.md:64-68`

**Step 1: Document the batch parameter and supported SARIMAX choices**

Add `method="lbfgs"` to the `evaluate_models_oos()` example and explain:

- omission/`None` retains each model's `fit()` default;
- SARIMAX accepts `newton`, `nm`, `bfgs`, `lbfgs`, `powell`, `cg`, `ncg`, and `basinhopping`;
- every model in one batch must accept `fit(method=...)`, otherwise the named model is rejected before fitting;
- AutoSARIMAX's constructor `method` is a search strategy and is not silently treated as a likelihood optimizer.

**Step 2: Document direct `oos()` forwarding**

Update the TsModels evaluation section to show:

```python
evaluation = model.oos(
    estimation_period=(0, 79),
    validation_period=(80, 99),
    method="lbfgs",
)
```

Explain that the parameter is forwarded only when explicit.

**Step 3: Correct the design verification wording**

Replace “返回结果记录所选优化器” with “验证所选优化器实际到达克隆模型的底层拟合调用”. `OOSResult` intentionally stores forecasts and scores, not fitted optimizer metadata.

**Step 4: Run public help and README checks**

Run:

```powershell
python -m pytest tests/test_public_docstrings.py TsMetrics/tests/test_contracts.py -q
```

Expected: all public signatures are fully documented; all README Python blocks parse; all exports remain unchanged.

**Step 5: Commit documentation**

```powershell
git add -- TsMetrics/README.md TsModels/README.md docs/plans/2026-08-09-evaluate-models-oos-method-design.md
git commit -m "docs: explain OOS optimizer selection"
```

### Task 4: Run regression and repository validation

**Files:**
- Verify only; no planned source changes.

**Step 1: Run all evaluation and SARIMAX tests**

```powershell
python -m pytest TsMetrics/tests TsModels/tests/test_evaluation.py TsModels/tests/test_sarimax.py -q
```

Expected: all selected tests pass without new warnings or skipped failures.

**Step 2: Run the complete suite**

```powershell
python -m pytest -q
```

Expected: full repository suite passes.

**Step 3: Run static and artifact checks**

```powershell
python -m ruff check TsMetrics TsModels tests
python -m compileall -q TsMetrics TsModels
git diff --check
git status --short --branch
```

Expected: Ruff, byte-compilation, and whitespace checks pass; status contains only the user's pre-existing untracked `tmp/` plus intentional commits on `main`.

**Step 4: Audit the final public call chain**

Verify from the diff that:

- no optimizer validation algorithm was copied from `SARIMAX.fit()`;
- `method=None` reaches `fit()` as no keyword argument;
- the only model fitting remains in `fit_and_forecast()`;
- unsupported batch models fail before the first fit;
- backtest behavior and signatures are unchanged.

**Step 5: Record completion**

If verification required no fixes, no extra commit is necessary. If a test reveals an in-scope defect, add the smallest regression test and fix, rerun the affected scope and full suite, then commit only those files with a focused message.
