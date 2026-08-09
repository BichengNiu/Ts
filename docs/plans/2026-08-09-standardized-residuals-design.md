# Standardized Residuals Design

## Goal

Add a reusable standardized-residual interface to model result objects and use
it throughout residual diagnostic plots while preserving the original residual
series and residual-test contract.

## Public contract

`BaseModelResult.standardized_residuals` is a read-only property defined as

\[
e_t^* = \frac{e_t}{\operatorname{std}(e)}, \qquad \text{ddof}=0.
\]

The residual mean is not subtracted separately. One-dimensional results use one
standard deviation for the complete statistically valid residual sample.
Two-dimensional VAR/VECM results use a separate standard deviation for each
equation (`axis=0`). The property returns a new floating-point array and does
not mutate `.residuals`.

If any required standard deviation is zero or non-finite, the property raises a
clear `ValueError`, because the requested standardized residual is undefined.

## Architecture and reuse

The calculation belongs on `BaseModelResult`, the existing shared public result
contract. SARIMAX, GARCH, Auto, VAR, VECM, and SVAR result objects inherit this
behavior. No parallel utility or model-specific implementation is added.

The shared univariate `BaseModelResult.plot_diagnostics()` and the existing
VAR/VECM diagnostic overrides consume `.standardized_residuals`. Existing
`TsPlots.plot_series()`, `plot_acf()`, and `plot_pacf()` remain responsible for
rendering and styling.

GARCH uses the same whole-sample residual standard deviation in this change.
Conditional standardization by `conditional_volatility` is explicitly deferred.

## Diagnostic behavior

All panels in a diagnostic figure consume the standardized residual array so
the figure has one coherent data contract:

- the residual time-series y-axis becomes `Standardized Residual`;
- the histogram x-axis becomes `Standardized Residual`;
- ACF and PACF retain the same numerical values because common positive scaling
  does not change correlations;
- Jarque-Bera and Ljung-Box annotations retain the same statistical meaning and
  numerical results, apart from possible floating-point roundoff.

`test_residuals()` continues to use `.residuals`. Its documented raw-residual
contract is unchanged, and the included tests are invariant to a common
positive scale.

## Validation

Tests will verify the exact `ddof=0` formula, absence of mean subtraction,
non-mutation, zero/non-finite scale errors, SARIMAX burn-in preservation,
univariate plot data and labels, and per-equation VAR/VECM behavior. Existing
Auto and GARCH inherited diagnostic tests provide compatibility coverage.

The change also updates `TsModels/README.md` and relevant public docstrings.
Focused tests, the full suite, Ruff, and Git diff checks complete verification.
