# SARIMAX rational distributed lags implementation plan

## Objective

Extend the existing public `SARIMAX` / `SARIMAXResult` contract with optional
rational distributed lags (transfer functions), including multiple inputs,
sparse numerator and denominator lag specifications, structured coefficient
access, steady-state gains, impulse weights, and future-input forecasting.

The approved sparse-lag rule is:

- `numerator=3` estimates numerator lags 0, 1, 2, and 3.
- `numerator=(0, 1, 3)` estimates lags 0, 1, and 3 and fixes lag 2 at zero.
- `denominator=3` estimates denominator lags 1, 2, and 3.
- `denominator=(1, 3)` estimates lags 1 and 3 and fixes lag 2 at zero.

These restrictions apply to polynomial coefficients, not to the recursively
generated impulse weights.

## Existing capabilities reviewed and reused

- `TsModels._sarimax.SARIMAX` input/date/missing-value normalisation, event
  design, fitting controls, future-exogenous scenarios, and clone/OOS hooks.
- `TsModels._sarimax.SARIMAXResult` prediction, convergence, root diagnostics,
  display burn-in masking, log-response restoration, and summaries.
- `TsModels._base.BaseModel` and `BaseModelResult` common estimation,
  diagnostics, backtest, and result contracts.
- Existing sparse AR/MA active-lag convention and fixed-zero reporting.
- `AutoSARIMAX` candidate construction; RDL specifications remain fixed across
  SARIMA candidates and are not automatically searched.
- Existing intervention covariance utilities as the pattern for delta-method
  derived inference; no duplicate generic result or plotting stack is added.
- `TsSims.BaseSimResult` and `simulate_sarima` for a matching RDL generator;
  transfer specifications and filtering reuse the canonical `TsModels`
  definitions rather than duplicating the estimator's equations.

Statsmodels' ordinary `exog` cannot represent an unknown rational denominator.
A private `StatsmodelsSARIMAX` extension is therefore required to add the
parameter-dependent transfer effect to the observation intercept while
retaining the existing SARIMA state-space likelihood.

## Statistical contract

For input `i`:

`H_i(B) = B^delay * omega_i(B) / delta_i(B)`

with `delta_i(B) = 1 - sum(delta_ij * B^j)`. The impulse weights obey

`psi_k = omega_k + sum(delta_j * psi_(k-j))`.

For a stable denominator, the steady-state gain is

`sum(omega) / (1 - sum(delta))`.

- Denominator stability is enforced when requested and always diagnosed from
  the complete polynomial, including omitted coefficients fixed at zero.
- Recursive filters require consecutive, equally spaced observations. RDL
  fitting rejects samples where `missing="drop"` removed rows.
- `initialization="zero"` treats pre-sample inputs/effects as zero;
  `initialization="steady_state"` assumes the first input level prevailed
  before the sample. The assumption is visible in the result summary.
- Gains for `log=True` are reported on the fitted log-response scale.
- Future predictions require every original exogenous input, including RDL
  inputs, for every forecast period.

## Public API

```python
spec = RationalLagSpec(
    numerator=(0, 2),
    denominator=(1, 3),
    delay=1,
    initialization="zero",
)

model = SARIMAX(
    y,
    exog=X,
    distributed_lags={"price": spec},
)
result = model.fit(require_convergence=True)
```

Results expose:

- flattened free estimates in `params`, `std_errors`, and `p_values`;
- all polynomial rows, including fixed zeros, in
  `distributed_lag_coefficients`;
- `fixed_params` entries for omitted numerator/denominator lags;
- per-input `steady_state_gains` with delta-method inference;
- `weights(steps)` and per-input structured results;
- automatic RDL sections in `summary()`.

## Implementation batches

1. Add the plan, shared lag normalisation, `RationalLagSpec`, coefficient/gain
   result structures, recursive weights, roots, and focused tests.
2. Integrate a private joint-MLE backend into `SARIMAX.fit()`, split ordinary
   and RDL inputs, expose structured results, and test finite, rational,
   sparse, and multiple-input recovery.
3. Extend prediction, future scenarios, clone/OOS, and `AutoSARIMAX` fixed-spec
   propagation.
4. Add `TsSims.simulate_rdl`, structured inputs/components, multiple-input
   parameter-recovery tests, and public simulation documentation.
5. Update public exports, README, API documentation, and executable demos.
6. Run focused tests, the complete repository suite, Ruff, compileall,
   notebook execution, and `git diff --check`.

## Verification requirements

- No-RDL SARIMAX behavior and all existing tests remain compatible.
- A denominator-free RDL equals an explicitly lag-expanded regression.
- Simulated Koyck and multiple-input models recover parameters and gains.
- Sparse omitted coefficients are absent from the optimizer vector and shown
  as fixed zeros in structured output and summaries.
- Weight recursion and gain formulas match closed-form cases.
- Unstable/near-boundary denominators, gaps, invalid names/specifications,
  future-path omissions, and non-convergence are explicit failures.
- Forecast scenarios use the full historical plus future input path.
