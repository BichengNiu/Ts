# TsUtils Calendar Table Design

## Goal

Add a public `calendar_table()` utility that reshapes one dated time series
into a rectangular calendar table suitable for direct export through pandas to
Excel, CSV, or another tabular format.

The function performs deterministic reshaping only. It does not interpolate,
aggregate, format files, or mutate caller-owned data.

## Public contract

```python
from Ts.TsUtils import calendar_table

table = calendar_table(
    data,
    col=None,
    freq=None,
    label_style="numeric",
)
```

`data` accepts a pandas `Series` or `DataFrame` with a `DatetimeIndex` or
`PeriodIndex`. A Series must use `col=None`. A one-column DataFrame is selected
automatically when `col=None`; a multi-column DataFrame requires the unique
column label supplied through `col`.

`freq=None` first uses index frequency metadata and then `pandas.infer_freq()`.
If neither identifies a supported frequency, the function raises. An explicit
pandas-compatible `freq` may be supplied for a gapped series; it is validated
against the observed timestamps and controls completion of the regular index
between the first and last observation.

`label_style` accepts only `"numeric"` and `"text"`. Numeric labels are the
default because they remain convenient for sorting and calculation. Text labels
are intended for presentation-oriented exports.

The function returns a pandas `DataFrame`. File output remains the caller's
responsibility:

```python
table.to_excel("output.xlsx")
table.to_csv("output.csv")
```

## Output schemas

Only calendar-year semantics are supported.

| Frequency | Rows | Columns |
|---|---|---|
| Annual | one `value` row | natural year |
| Quarterly | quarters 1--4 | natural year |
| Monthly | months 1--12 | natural year |
| Weekly | within-month weeks 1--5 | two levels: natural year, month |
| Daily | days 1--31 | two levels: natural year, month |
| Business daily | days 1--31 | two levels: natural year, month |

Rows always retain their complete canonical schema. Columns are chronological
and cover the completed calendar between the first and last observation.
Original missing values and newly completed periods remain `NaN`.

In numeric mode, period, year, and month labels are integers. In text mode they
become presentation labels such as `"1月"`, `"第1周"`, `"1日"`, `"2024年"`,
and `"Q1"`. Weekly and daily outputs use named two-level columns so Excel and
CSV naturally serialize them as two header rows.

## Weekly attribution

Every weekly observation is assigned according to the calendar month containing
the Wednesday of its seven-day period, regardless of whether the source index is
anchored as `W-SUN`, `W-MON`, or another weekly variant. Its row is the ordinal
of that Wednesday within the month, from 1 through 5.

The implementation derives the actual weekly interval from the resolved pandas
frequency and locates the Wednesday within that interval. It does not classify
the observation solely from the timestamp label and does not split or aggregate
weekly values.

## Supported and rejected frequencies

The first version supports unit annual, calendar-quarterly, monthly, weekly,
calendar-daily, and business-daily frequencies. It rejects:

- fiscal-quarter and fiscal-year definitions;
- intraday frequencies;
- multiplied frequencies such as two-month intervals;
- irregular or unidentified frequencies without an explicit `freq`;
- explicit frequencies incompatible with the observed timestamps.

Natural-quarter inputs include calendar quarter-end and calendar quarter-start
variants. Fiscal anchors are not silently converted because that can change the
meaning of year and quarter labels.

## Architecture and reuse

The implementation lives in `TsUtils/_calendar_table.py`. General time-index
validation and frequency-resolution logic currently embedded in
`TsUtils/_seasonal_dummies.py` is extracted into a private shared calendar
helper. Both public functions use that helper, while `seasonal_dummies()` keeps
its existing signature, supported-frequency boundary, return schema, and error
behavior.

The reshape is implemented with pandas regular-index construction, `reindex()`,
calendar accessors, `MultiIndex`, and `unstack()`. No parallel matrix algorithm,
file writer, or result container is introduced.

The data flow is:

1. Select one Series from the public input without mutating it.
2. Validate the numeric values and time index.
3. Resolve and validate the frequency.
4. Complete the regular index between the observed boundaries with `NaN` values.
5. Derive natural calendar coordinates, including Wednesday-based weekly ones.
6. Reject any coordinate collision rather than aggregating.
7. Unstack into the canonical schema and apply the selected label style.

## Validation and errors

The function rejects empty inputs, unsupported containers, missing date labels,
non-unique or decreasing indexes, ambiguous DataFrame selection, duplicate
column labels, non-numeric/Boolean/complex data, and positive or negative
infinity. Ordinary missing values are valid and remain missing.

Errors identify the invalid argument or frequency. A request never silently
sorts, aggregates, interpolates, or chooses one of multiple DataFrame columns.

## Public exports and documentation

`calendar_table` is exported from both `Ts.TsUtils` and the curated `Ts`
namespace. Its public docstring includes executable monthly and weekly examples.
`TsUtils/README.md` documents every schema, both label styles, explicit frequency
use for gapped inputs, Wednesday attribution, and pandas export examples.

## Tests

Tests cover:

- public imports from `Ts.TsUtils` and `Ts`;
- Series, one-column DataFrame, and explicit `col` selection;
- annual, natural-quarterly, monthly, weekly, daily, and business-daily schemas;
- weekly anchors and a week whose Wednesday determines a different month from
  its timestamp label;
- numeric and text labels;
- explicit frequency completion with `NaN` gaps;
- preservation of original missing values and non-mutation;
- `DatetimeIndex` and `PeriodIndex` paths;
- every validation and unsupported-frequency failure;
- executable docstrings and README public-name coverage;
- all existing `seasonal_dummies()` tests after shared-helper extraction.

Validation proceeds from focused tests to all TsUtils tests, public API/docstring
tests, lint/format checks, and the full repository test suite.
