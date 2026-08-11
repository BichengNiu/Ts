# Correlogram Missing-Value Policy Design

## Goal

Give `plot_acf()` and `plot_pacf()` one explicit and consistent policy for
non-finite observations. Both functions should produce useful plots from the
ordinary boundary `NaN` values created by differencing or lagging, while callers
who require a fixed sample can opt into fail-fast validation.

## Public contract

Both existing functions gain one keyword-only argument:

```python
plot_acf(data, nlags=None, *, missing="drop", ...)
plot_pacf(data, nlags=None, *, missing="drop", ...)
```

`missing` accepts only `"drop"` and `"raise"`:

- `"drop"` is the default and removes every non-finite observation (`NaN`,
  positive infinity, and negative infinity) before computing the statistic.
- `"raise"` rejects non-finite input and reports its original zero-based row
  positions.

The input object is never mutated. All other plotting defaults and return values
remain unchanged, including ACF lag zero, PACF starting at lag one, confidence
bands, styles, and `(fig, ax)` returns.

## Statistical semantics

Dropping an interior observation compresses the series and makes the observations
on either side adjacent. This is intentional only when the caller selects the
default `"drop"` policy. Documentation will warn callers to use
`missing="raise"` when an interior gap must not be silently compressed, and to
interpolate or otherwise model the gap before plotting when calendar spacing is
substantive.

The effective sample after cleaning controls statsmodels' adaptive `nlags`
selection. An explicit `nlags` continues to use statsmodels' validation against
that effective sample. Empty effective samples fail with a clear error before
statsmodels is called.

## Architecture and reuse

`TsPlots/acf_plot.py` keeps its existing `_to_1d()` conversion boundary. A new
private helper immediately after conversion validates `missing`, builds one
finite mask with `numpy.isfinite()`, reports original positions for `"raise"`,
and returns the filtered array for `"drop"`. Both public plotting functions call
that helper, so ACF and PACF cannot drift.

The existing `TsUtils._validation._resolve_missing_rows()` policy was inspected.
`TsPlots` will not import it because `TsUtils._summary` already imports
`TsPlots`; reversing that dependency would introduce a circular package import.
No new public function, result class, or parallel plotting path is added.

## Errors

- An unknown policy raises `ValueError("missing must be 'raise' or 'drop'")`.
- `missing="raise"` reports every original non-finite row position.
- `missing="drop"` raises a clear `ValueError` if no finite observations remain.
- Existing shape and multi-column DataFrame errors remain unchanged.
- Existing statsmodels errors for an excessive explicit PACF lag remain intact.

## Documentation and tests

The NumPy-style docstrings and `TsPlots/README.md` will document the new
parameter, default, treatment of infinity, effective-sample lag selection, and
the interior-gap warning.

Focused tests will cover:

- default deletion of `NaN` for ACF and PACF;
- deletion of positive and negative infinity;
- equality with plots produced from an explicitly cleaned array;
- adaptive lag counts based on the cleaned sample;
- `missing="raise"` with original positions;
- rejection of an unknown policy;
- rejection of an all-non-finite sample;
- preservation of the caller-owned pandas or NumPy input;
- retention of existing signatures, return types, and explicit-lag validation.

Validation proceeds through focused `TsPlots` tests, public docstring/API tests,
scoped Ruff checks, and the full repository test suite.
