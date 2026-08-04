# RDL automatic initialization and level-intercept correction plan

## Objective

Make the one-call public `SARIMAX(..., distributed_lags=...).fit()` path fit
finite distributed-lag models on their scientifically valid sample without
user-created lag columns, constant columns, private backends, or hand-tuned
starting values.  The textbook preliminary LTF model must be expressed as:

```python
SARIMAX(
    sales.iloc[:140],
    exog=leading_indicator.iloc[:140].rename("leading_indicator"),
    order=(1, 0, 0),
    trend="c",
    distributed_lags={
        "leading_indicator": RationalLagSpec(numerator=15, denominator=0)
    },
).fit()
```

`trend="c"` remains the only requested constant.  Raw statsmodels parameters
remain available for compatibility, while the Ts result explicitly exposes
the original-level intercept `C` implied by the SARIMAX parameterization.

## Existing capabilities reviewed and reused

- `TsModels._sarimax.SARIMAX` remains the only public estimator and continues
  to own data normalization, ordinary exogenous inputs, fitting, diagnostics,
  forecasting, and result construction.
- `TsModels._distributed_lag._RationalLagSARIMAX` remains the single private
  joint-likelihood backend.  No generated public lag-column implementation is
  added.
- Existing statsmodels `loglikelihood_burn` is reused for conditional
  likelihood and for Ts fitted-value / residual masking.
- Existing reduced AR polynomial and root diagnostics are reused to transform
  the state-equation intercept into the original-level regression intercept.
- Explicit legacy `initialization="zero"` and `"steady_state"` contracts stay
  available; only the default becomes model-aware `"auto"`.

## Statistical contract

For a finite transfer function with maximum input lag `K`, the first `K`
rows do not have complete observed input history.  Under `initialization="auto"`
Ts therefore:

1. constructs the same finite transfer effect internally;
2. excludes the incomplete input-history rows plus the disturbance model's
   required recursion depth from the likelihood;
3. estimates regression starts on that same valid sample; and
4. initializes the ARMA disturbance parameters from the preliminary-regression
   residuals, avoiding the near-unit-root local solution caused by unrelated
   default starts.

For a rational denominator, observed data cannot identify the infinite
pre-sample input path.  `"auto"` resolves to the existing steady-state
assumption and reports that resolution.  Users can still request explicit
`"zero"` or `"steady_state"` assumptions when subject-matter knowledge dictates.

For an undifferenced stationary model with `trend="c"`, statsmodels estimates
the state intercept `c`.  Ts exposes the original-level intercept as
`level_intercept = c / A(1)`, where `A(B)` is the complete reduced AR
polynomial.  This is the `C` in `Y_t = C + H(B)X_t + N_t`; it is not an extra
exogenous constant and does not alter the optimizer parameter vector.

## Test-first implementation batches

### Batch 1: public specification and backward compatibility

- Assert `RationalLagSpec()` defaults to `initialization="auto"`.
- Assert invalid initialization values still fail clearly.
- Keep explicit `"zero"` finite-lag equivalence to the old zero-padded design.
- Assert rational `"auto"` resolves to steady-state filtering.

### Batch 2: conditional finite-lag likelihood and starts

- Add a deterministic finite-lag + AR(1) regression test.
- Assert automatic likelihood burn equals maximum input history plus the AR
  recursion depth.
- Compare estimates against an explicit complete-history regression fit.
- Assert public residuals and plotted fitted values exclude the same burn.
- Keep forecast, clone/OOS, rational Koyck, and ordinary SARIMAX tests passing.

### Batch 3: original-level intercept

- Add `SARIMAXResult.level_intercept`.
- Assert it equals `params["intercept"] / sum(polynomial_reduced_ar)` for a
  stationary undifferenced constant model.
- Assert `long_run_equilibrium()` reuses the same value.
- Report both the level intercept and the raw state intercept meaning in the
  summary without changing `params`, covariance, or starting-parameter order.
- Return `None` where an original-level constant is not defined (differencing,
  time trend, no stationarity); retain the existing log-scale guard.

### Batch 4: documentation and textbook verification

- Update the TsModels README with the one-call textbook form, automatic
  initialization policy, effective sample, and intercept interpretation.
- Run focused RDL/SARIMAX tests, the complete repository suite, Ruff,
  `compileall`, and `git diff --check`.
- Refit the first 140 observations from the supplied `data.xlsx` and compare
  `C`, all 16 finite-lag coefficients, AR(1) coefficient, and innovation
  standard deviation with Table 5.2.

## Acceptance criteria

- The textbook model is fitted through one public `SARIMAX()` call.
- No user-generated constant or lag columns and no private estimator calls.
- Default finite RDL estimation uses only complete lag history automatically.
- Explicit initialization choices preserve their previous numerical meaning.
- Raw statsmodels parameters remain traceable, and `level_intercept` reports
  the textbook-level `C` directly.
- Tests and the external textbook replication both pass, with any remaining
  differences explained by likelihood/rounding rather than hidden tuning.
