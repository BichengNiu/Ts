# TsUtils Seasonal Dummies Design

## Goal

Add a public `seasonal_dummies()` preprocessing function that automatically
generates recurring calendar-position dummy variables from a dated time-series
index.

## Public contract

```python
from Ts.TsUtils import seasonal_dummies

dummies = seasonal_dummies(data)
dummies = seasonal_dummies(data, drop_first=False)
```

The function accepts a pandas Series or DataFrame with a `DatetimeIndex` or
`PeriodIndex`, or accepts either index directly. It uses only the time index,
does not inspect or mutate data values, and returns a numeric DataFrame aligned
to the original index.

`drop_first=True` is the default to avoid exact collinearity with a model
intercept. `drop_first=False` retains the complete category set. The argument
must be a boolean.

## Frequency and category rules

The function uses an index's explicit frequency when available and otherwise
uses `pandas.infer_freq()`. Supported frequency families and fixed schemas are:

| Input frequency | Categories | Default reference |
|---|---|---|
| Calendar daily | Monday through Sunday | Monday |
| Business daily | Monday through Friday | Monday |
| Anchored weekly | ISO weeks 01 through 53 | Week 01 |
| Month begin/end and business variants | Months 01 through 12 | January |
| Quarter begin/end and business variants | Quarters Q1 through Q4 | Q1 |

All canonical category columns are returned even if some categories are absent
from the supplied sample. This keeps historical and future exogenous matrices
schema-compatible. Weekly categories use ISO week numbering. Fiscal-quarter
positions follow the frequency anchor rather than calendar-quarter labels.

Annual, intraday, irregular, unsupported, or unidentifiable frequencies raise
a clear `ValueError`; annual data intentionally has no seasonal-dummy contract.

## Architecture and reuse

The implementation belongs in `TsUtils/_seasonal_dummies.py` and reuses pandas
index frequency metadata, `infer_freq()`, categorical values, and
`get_dummies()`. It does not duplicate the model-specific `EventSpec` pulse and
step machinery, which solves a different problem.

The function is exported from both `Ts.TsUtils` and the unified `Ts` namespace.
No result dataclass is added because the returned DataFrame already carries the
original index and explicit column schema.

## Validation

The input index must be non-empty, free of missing timestamps, unique, and
monotonically increasing. Plain positional indexes are rejected. Tests cover
Series, DataFrame, DatetimeIndex, PeriodIndex, each supported frequency family,
anchored fiscal quarters, full and dropped-first schemas, short samples with
absent categories, index preservation, public imports, and all failure paths.

Documentation is added to `TsUtils/README.md` and the public docstring. The
repository's executable docstrings and README remain the demonstration surface;
no parallel notebook implementation is needed.
