"""Unified result containers for historical forecast evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._metrics import ERROR_METRIC_NAMES, compute_metrics
from ._schemes import ForecastSplit
from Ts.TsUtils._validation import validate_alpha


PARAMETER_ESTIMATE_COLUMNS = (
    "split",
    "parameter",
    "estimate",
    "std_error",
    "p_value",
)
PARAMETER_TABLE_COLUMNS = (
    "split",
    "train_start",
    "train_end",
    "n_train",
    "window",
    "forecast_start",
    "parameter",
    "estimate",
    "std_error",
    "p_value",
)


def _optional_array(values):
    return None if values is None else np.array(values, dtype=float, copy=True)


def _validate_alpha(alpha):
    if alpha is None:
        return None
    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError("alpha must be a number strictly between 0 and 1")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise TypeError("alpha must be a number strictly between 0 and 1") from error
    return validate_alpha(alpha)


def _normalise_names(names, count, label):
    if isinstance(names, str):
        raise TypeError(f"{label} must be a sequence of strings")
    try:
        names = tuple(names)
    except TypeError as error:
        raise TypeError(f"{label} must be a sequence of strings") from error
    if len(names) != count:
        raise ValueError(f"{label} must contain one name per series")
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError(f"{label} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique")
    return names


def _validate_series_names(names, mean):
    if names is None:
        return None
    if mean.ndim != 3:
        raise ValueError("series_names is only valid for multivariate results")
    return _normalise_names(names, mean.shape[2], "series_names")


def _same_split(left, right):
    fields = (
        "split",
        "train_start",
        "train_end",
        "forecast_start",
        "forecast_end",
        "gap",
        "window",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields) and (
        np.array_equal(left.train_indices, right.train_indices)
        and np.array_equal(left.target_indices, right.target_indices)
    )


def _rank(scores):
    return sorted(
        scores,
        key=lambda name: (
            not np.isfinite(scores[name]),
            scores[name] if np.isfinite(scores[name]) else np.inf,
            name,
        ),
    )


def _normalise_parameter_estimates(values, n_splits):
    """Return an owned, validated long-form parameter-estimate table."""
    if values is None:
        return pd.DataFrame(columns=PARAMETER_ESTIMATE_COLUMNS)
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
    else:
        try:
            frame = pd.DataFrame.from_records(values)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "parameter_estimates must be tabular records or None"
            ) from error
    if frame.empty:
        return pd.DataFrame(columns=PARAMETER_ESTIMATE_COLUMNS)
    missing = set(PARAMETER_ESTIMATE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            "parameter_estimates is missing columns: " + ", ".join(sorted(missing))
        )
    frame = frame.loc[:, list(PARAMETER_ESTIMATE_COLUMNS)].copy()
    split_values = []
    for value in frame["split"]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("parameter estimate split must be an integer")
        value = int(value)
        if not 0 <= value < n_splits:
            raise ValueError("parameter estimate split is out of range")
        split_values.append(value)
    frame["split"] = split_values
    if (
        not frame["parameter"]
        .map(lambda value: isinstance(value, str) and bool(value))
        .all()
    ):
        raise TypeError("parameter names must be non-empty strings")
    if frame.duplicated(["split", "parameter"]).any():
        raise ValueError("parameter estimates must be unique by split and parameter")
    for column in ("estimate", "std_error", "p_value"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"parameter estimate {column} must be numeric") from error
    return frame.reset_index(drop=True)


def _select_parameter_names(parameters, available):
    """Validate a parameter selector and retain the caller's order."""
    available = tuple(dict.fromkeys(available))
    if parameters is None:
        return available
    if isinstance(parameters, str):
        selected = (parameters,)
    else:
        try:
            selected = tuple(parameters)
        except TypeError as error:
            raise TypeError(
                "parameters must be a parameter name, a sequence, or None"
            ) from error
    if not selected:
        raise ValueError("parameters must not be empty")
    if not all(isinstance(name, str) and name for name in selected):
        raise TypeError("parameters must contain non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("parameters must not contain duplicates")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"unknown parameter(s): {unknown}")
    return selected


@dataclass
class ForecastEvaluationResult:
    """Forecasts from one estimator over all splits in one scheme.

    Arrays always retain the split and forecast-horizon axes. Univariate
    arrays have shape ``(split, horizon)``; multivariate arrays add a final
    series axis.

    Parameters
    ----------
    mean : ndarray
        Point forecasts with split and horizon axes.
    actual : ndarray
        Observed scoring targets with the same shape as ``mean``.
    lower : ndarray or None
        Lower interval bounds, when available.
    upper : ndarray or None
        Upper interval bounds, when available.
    splits : sequence of ForecastSplit
        Resolved training and target metadata for every result row.
    failures : list of dict
        Atomic failed-split records.
    model_type : str
        Fitted model label.
    target : str
        Name of the observable scoring target.
    dates : sequence of datetime-like, optional
        Full source calendar.
    series_names : sequence of str, optional
        Names for the final multivariate axis.
    alpha : float, optional
        Significance level used for intervals.
    uses_observed_future_exog : bool, default False
        Whether realized future exogenous paths conditioned the forecasts.
    parameter_estimates : tabular records, optional
        Scalar fitted parameter estimates and diagnostics retained by split.

    Examples
    --------
    >>> split = RollingOrigin(10).split(11)
    >>> result = ForecastEvaluationResult(
    ...     mean=[[1.0]], actual=[[1.5]], lower=None, upper=None,
    ...     splits=split, failures=[], model_type="demo", target="observed",
    ... )
    >>> result.mean.shape
    (1, 1)
    """

    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    splits: tuple[ForecastSplit, ...]
    failures: list[dict]
    model_type: str
    target: str
    dates: pd.DatetimeIndex | None = None
    series_names: tuple[str, ...] | None = None
    alpha: float | None = None
    uses_observed_future_exog: bool = False
    parameter_estimates: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=PARAMETER_ESTIMATE_COLUMNS)
    )

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.actual = np.array(self.actual, dtype=float, copy=True)
        self.lower = _optional_array(self.lower)
        self.upper = _optional_array(self.upper)
        if self.mean.ndim not in (2, 3):
            raise ValueError(
                "mean must have shape (split, horizon) or "
                "(split, horizon, series)"
            )
        if not self.mean.size:
            raise ValueError("mean must contain at least one forecast value")
        if self.actual.shape != self.mean.shape:
            raise ValueError("actual must have the same shape as mean")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper must both be set or both be None")
        if self.lower is not None:
            if self.lower.shape != self.mean.shape or self.upper.shape != self.mean.shape:
                raise ValueError("lower and upper must have the same shape as mean")
            finite = np.isfinite(self.lower) & np.isfinite(self.upper)
            if np.any(self.lower[finite] > self.upper[finite]):
                raise ValueError("lower must not exceed upper")

        try:
            self.splits = tuple(self.splits)
        except TypeError as error:
            raise TypeError("splits must be a sequence") from error
        if len(self.splits) != self.mean.shape[0]:
            raise ValueError("splits must contain one entry per forecast split")
        if not all(isinstance(split, ForecastSplit) for split in self.splits):
            raise TypeError("splits must contain only ForecastSplit values")
        if [split.split for split in self.splits] != list(range(len(self.splits))):
            raise ValueError("split identifiers must be contiguous from zero")
        if any(len(split.target_indices) != self.mean.shape[1] for split in self.splits):
            raise ValueError("every split target must match the forecast horizon")

        if not isinstance(self.model_type, str) or not self.model_type:
            raise TypeError("model_type must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise TypeError("target must be a non-empty string")
        self.series_names = _validate_series_names(self.series_names, self.mean)
        self.alpha = _validate_alpha(self.alpha)
        if not isinstance(self.uses_observed_future_exog, (bool, np.bool_)):
            raise TypeError("uses_observed_future_exog must be a boolean")
        self.uses_observed_future_exog = bool(self.uses_observed_future_exog)
        self.parameter_estimates = _normalise_parameter_estimates(
            self.parameter_estimates,
            len(self.splits),
        )

        self.dates = None if self.dates is None else pd.DatetimeIndex(self.dates).copy()
        targets = np.concatenate([split.target_indices for split in self.splits])
        if targets.min() < 0:
            raise ValueError("split target indices must be non-negative")
        if self.dates is not None:
            if self.dates.hasnans or not self.dates.is_unique:
                raise ValueError("dates must be complete and unique")
            if not self.dates.is_monotonic_increasing:
                raise ValueError("dates must be strictly increasing")
            if targets.max() >= len(self.dates):
                raise ValueError("dates must cover every forecast target index")

        self.failures = [dict(failure) for failure in self.failures]
        seen = set()
        for failure in self.failures:
            if not {"split", "error_type", "message"}.issubset(failure):
                raise ValueError(
                    "each failure must contain split, error_type, and message"
                )
            split = failure["split"]
            if isinstance(split, (bool, np.bool_)) or not isinstance(
                split, (int, np.integer)
            ):
                raise TypeError("failure split must be an integer")
            split = int(split)
            if not 0 <= split < len(self.splits) or split in seen:
                raise ValueError("failure splits must uniquely identify result rows")
            if not np.isnan(self.mean[split]).all() or not np.isnan(
                self.actual[split]
            ).all():
                raise ValueError("failed splits must retain all-NaN result rows")
            if self.lower is not None and (
                not np.isnan(self.lower[split]).all()
                or not np.isnan(self.upper[split]).all()
            ):
                raise ValueError("failed split intervals must be all NaN")
            failure["split"] = split
            seen.add(split)

    @property
    def metrics(self):
        """Return canonical metrics over finite forecast pairs."""
        return compute_metrics(self.actual, self.mean)

    @property
    def predictions(self):
        """Return one long-form row per split, horizon, and series."""
        multivariate = self.mean.ndim == 3
        labels = (
            self.series_names
            or tuple(f"series_{index}" for index in range(self.mean.shape[2]))
            if multivariate
            else (None,)
        )
        rows = []
        for split_position, split in enumerate(self.splits):
            for horizon_position, target_position in enumerate(split.target_indices):
                target_time = (
                    int(target_position)
                    if self.dates is None
                    else self.dates[int(target_position)]
                )
                for series_position, series in enumerate(labels):
                    index = (
                        (split_position, horizon_position, series_position)
                        if multivariate
                        else (split_position, horizon_position)
                    )
                    actual = float(self.actual[index])
                    forecast = float(self.mean[index])
                    valid = bool(np.isfinite(actual) and np.isfinite(forecast))
                    rows.append(
                        {
                            "split": split.split,
                            "origin": split.forecast_start,
                            "target_time": target_time,
                            "horizon": horizon_position + 1,
                            "series": series,
                            "actual": actual,
                            "forecast": forecast,
                            "error": forecast - actual if valid else np.nan,
                            "lower": np.nan if self.lower is None else float(self.lower[index]),
                            "upper": np.nan if self.upper is None else float(self.upper[index]),
                            "valid": valid,
                        }
                    )
        return pd.DataFrame.from_records(rows)

    @property
    def split_table(self):
        """Return one row describing each resolved training/forecast split."""
        return pd.DataFrame.from_records(
            [
                {
                    "split": split.split,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "forecast_start": split.forecast_start,
                    "forecast_end": split.forecast_end,
                    "n_train": len(split.train_indices),
                    "gap": split.gap,
                    "window": split.window,
                }
                for split in self.splits
            ]
        )

    def parameter_table(self, parameters=None):
        """Return parameter estimates for every resolved training sample.

        Parameters
        ----------
        parameters : str or sequence of str, optional
            Parameter names to retain. By default, all observed parameters are
            returned.

        Returns
        -------
        pandas.DataFrame
            One row per split and parameter, including training-sample bounds,
            estimate, standard error, and p-value. Failed splits are retained
            with missing estimates.

        Examples
        --------
        >>> table = result.parameter_table("ar.L1")
        >>> ["train_start", "train_end", "estimate"]
        ['train_start', 'train_end', 'estimate']
        """
        available = self.parameter_estimates["parameter"].tolist()
        selected = _select_parameter_names(parameters, available)
        if not selected:
            return pd.DataFrame(columns=PARAMETER_TABLE_COLUMNS)
        pairs = pd.DataFrame.from_records(
            [
                {"split": split.split, "parameter": parameter}
                for split in self.splits
                for parameter in selected
            ]
        )
        estimates = pairs.merge(
            self.parameter_estimates,
            on=["split", "parameter"],
            how="left",
            validate="one_to_one",
        )
        table = self.split_table.merge(
            estimates,
            on="split",
            how="inner",
            validate="one_to_many",
        )
        return table.loc[:, list(PARAMETER_TABLE_COLUMNS)].copy()


@dataclass
class ForecastComparisonResult:
    """Comparable forecasts, fair common-sample metrics, and rankings.

    Parameters
    ----------
    results : dict of str to ForecastEvaluationResult
        Named results sharing splits, targets, calendars, and actual values.
    rank_by : str, default "rmse"
        Canonical error metric used for ascending ranking.

    Examples
    --------
    >>> report = ForecastComparisonResult({"model": result})
    >>> report.ranking
    ['model']
    """

    results: dict[str, ForecastEvaluationResult]
    rank_by: str = "rmse"

    def __post_init__(self):
        if not isinstance(self.results, dict):
            raise TypeError("results must be a dict of names to forecast results")
        if not self.results:
            raise ValueError("results must not be empty")
        if not all(isinstance(name, str) and name for name in self.results):
            raise TypeError("result names must be non-empty strings")
        if not all(
            isinstance(result, ForecastEvaluationResult)
            for result in self.results.values()
        ):
            raise TypeError("results must contain only ForecastEvaluationResult values")
        if self.rank_by not in ERROR_METRIC_NAMES:
            raise ValueError(f"rank_by must be one of {list(ERROR_METRIC_NAMES)}")
        self.results = dict(self.results)
        reference = next(iter(self.results.values()))
        for name, result in self.results.items():
            if result.target != reference.target:
                raise ValueError("all results must use the same target")
            if result.mean.shape != reference.mean.shape:
                raise ValueError("all results must have the same forecast shape")
            if result.series_names != reference.series_names:
                raise ValueError(f"result {name!r} does not use shared series_names")
            if len(result.splits) != len(reference.splits) or any(
                not _same_split(left, right)
                for left, right in zip(result.splits, reference.splits, strict=True)
            ):
                raise ValueError("all results must use the same forecast splits")
            if (result.dates is None) != (reference.dates is None):
                raise ValueError("all results must use the same date metadata")
            if result.dates is not None and not result.dates.equals(reference.dates):
                raise ValueError("all results must use the same dates")
            observed = np.isfinite(result.actual) & np.isfinite(reference.actual)
            if not np.array_equal(result.actual[observed], reference.actual[observed]):
                raise ValueError("all results must use the same actual values")

    @property
    def _common_mask(self):
        reference = next(iter(self.results.values()))
        mask = np.ones(reference.mean.shape, dtype=bool)
        for result in self.results.values():
            mask &= np.isfinite(result.actual) & np.isfinite(result.mean)
        return mask

    @property
    def target(self):
        """Return the shared forecast target name."""
        return next(iter(self.results.values())).target

    @property
    def scores(self):
        """Return common-sample values of the ranking metric."""
        mask = self._common_mask
        return {
            name: float(compute_metrics(result.actual[mask], result.mean[mask])[self.rank_by])
            for name, result in self.results.items()
        }

    @property
    def ranking(self):
        """Return model names ordered by ascending forecast error."""
        return _rank(self.scores)

    @property
    def best_model(self):
        """Return the best finite-scoring model name."""
        ranking = self.ranking
        if not ranking or not np.isfinite(self.scores[ranking[0]]):
            return None
        return ranking[0]

    @property
    def table(self):
        """Return common-sample metrics, coverage, failures, and rank."""
        mask = self._common_mask
        n_total, n_common = int(mask.size), int(mask.sum())
        rows = {}
        for name, result in self.results.items():
            metrics = compute_metrics(result.actual[mask], result.mean[mask])
            valid = np.isfinite(result.actual) & np.isfinite(result.mean)
            rows[name] = {
                **{metric: metrics[metric] for metric in ERROR_METRIC_NAMES},
                "n_total": n_total,
                "n_common": n_common,
                "coverage": float(valid.sum() / n_total),
                "failures": len(result.failures),
            }
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "model"
        frame["rank"] = pd.Series(
            {name: position for position, name in enumerate(self.ranking, start=1)},
            dtype=int,
        )
        return frame.loc[self.ranking].copy()

    @property
    def predictions(self):
        """Return canonical long-form forecasts for every model."""
        frames = []
        for name, result in self.results.items():
            frame = result.predictions
            frame.insert(0, "model", name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @property
    def splits(self):
        """Return the shared resolved split table."""
        return next(iter(self.results.values())).split_table.copy()

    @property
    def failures(self):
        """Return model-labelled failure records."""
        rows = []
        for name, result in self.results.items():
            rows.extend({"model": name, **failure} for failure in result.failures)
        return pd.DataFrame.from_records(
            rows, columns=["model", "split", "error_type", "message"]
        )

    def parameter_table(self, *, model=None, parameters=None):
        """Return rolling parameter estimates with training-sample metadata.

        Parameters
        ----------
        model : str, optional
            One model name from ``results``. By default all models are included.
        parameters : str or sequence of str, optional
            Parameter names to retain. By default all observed parameters are
            included.

        Returns
        -------
        pandas.DataFrame
            Model-labelled estimates by split and training-sample range.

        Examples
        --------
        >>> table = report.parameter_table(model="ARIMA", parameters="ar.L1")
        >>> "estimate" in table
        True
        """
        if model is None:
            model_names = tuple(self.results)
        elif not isinstance(model, str) or not model:
            raise TypeError("model must be a non-empty string or None")
        elif model not in self.results:
            raise ValueError(f"unknown model {model!r}")
        else:
            model_names = (model,)

        frames = []
        for name in model_names:
            frame = self.results[name].parameter_table()
            if frame.empty:
                continue
            frame.insert(0, "model", name)
            frames.append(frame)
        columns = ("model", *PARAMETER_TABLE_COLUMNS)
        if not frames:
            _select_parameter_names(parameters, ())
            return pd.DataFrame(columns=columns)
        table = pd.concat(frames, ignore_index=True)
        selected = _select_parameter_names(parameters, table["parameter"].tolist())
        table = table.loc[table["parameter"].isin(selected)].copy()
        parameter_order = {name: position for position, name in enumerate(selected)}
        model_order = {name: position for position, name in enumerate(model_names)}
        table["_model_order"] = table["model"].map(model_order)
        table["_parameter_order"] = table["parameter"].map(parameter_order)
        table = table.sort_values(
            ["_model_order", "_parameter_order", "split"],
            kind="stable",
        ).drop(columns=["_model_order", "_parameter_order"])
        return table.loc[:, list(columns)].reset_index(drop=True)

    def plot_parameters(
        self,
        parameters=None,
        *,
        model=None,
        x="train_end",
        title=None,
        xtitle=None,
        ytitle=None,
        freq=None,
        note=None,
        grid=False,
        ax=None,
    ):
        """Plot parameter estimates across rolling training samples.

        Parameters
        ----------
        parameters : str or sequence of str, optional
            Parameter names to plot. By default all observed parameters are
            shown.
        model : str, optional
            One model name from ``results``. By default all models are included.
        x : {"split", "train_end", "forecast_start"}
            Horizontal axis identifying each estimation sample.
        title, xtitle, ytitle, freq, note, grid, ax : optional
            Plot options forwarded to :func:`TsPlots.plot_series`.

        Returns
        -------
        tuple
            Matplotlib figure and axes.

        Examples
        --------
        >>> fig, ax = report.plot_parameters(
        ...     "ar.L1", model="ARIMA", x="train_end"
        ... )
        """
        from Ts.TsPlots import plot_series

        x_labels = {
            "split": "Split",
            "train_end": "Training sample end",
            "forecast_start": "Forecast origin",
        }
        if x not in x_labels:
            raise ValueError(f"x must be one of {list(x_labels)}")
        table = self.parameter_table(model=model, parameters=parameters)
        if table.empty:
            raise ValueError("no fitted parameter estimates are available to plot")
        model_names = table["model"].drop_duplicates().tolist()
        parameter_names = table["parameter"].drop_duplicates().tolist()
        if len(model_names) == 1:
            table["_series"] = table["parameter"]
            series_order = parameter_names
        elif len(parameter_names) == 1:
            table["_series"] = table["model"]
            series_order = model_names
        else:
            table["_series"] = table["model"] + ": " + table["parameter"]
            series_order = [
                f"{name}: {parameter}"
                for name in model_names
                for parameter in parameter_names
                if not table.loc[
                    (table["model"] == name) & (table["parameter"] == parameter)
                ].empty
            ]
        frame = table.pivot(index=x, columns="_series", values="estimate").reindex(
            columns=series_order
        )
        return plot_series(
            frame,
            facet=False,
            auto_dual_y=False,
            title=title,
            xtitle=x_labels[x] if xtitle is None else xtitle,
            ytitle="Estimate" if ytitle is None else ytitle,
            freq=freq,
            note=note,
            grid=grid,
            ax=ax,
        )

    def metric_table(self, *, by):
        """Return common-sample metrics grouped by one evaluation axis.

        Parameters
        ----------
        by : {"origin", "horizon", "series"}
            Evaluation axis used to form metric groups.

        Returns
        -------
        pandas.DataFrame
            Model-labelled canonical metrics for every group.

        Examples
        --------
        >>> report.metric_table(by="horizon")["horizon"].tolist()
        [1]
        """
        if by not in {"horizon", "origin", "series"}:
            raise ValueError("by must be 'horizon', 'origin', or 'series'")
        reference = next(iter(self.results.values()))
        if by == "horizon":
            groups = [
                (position + 1, (slice(None), position))
                for position in range(reference.mean.shape[1])
            ]
        elif by == "origin":
            groups = [
                (split.forecast_start, (position, slice(None)))
                for position, split in enumerate(reference.splits)
            ]
        elif reference.mean.ndim == 2:
            groups = [(reference.target, (slice(None), slice(None)))]
        else:
            labels = reference.series_names or tuple(
                f"series_{index}" for index in range(reference.mean.shape[2])
            )
            groups = [
                (label, (slice(None), slice(None), position))
                for position, label in enumerate(labels)
            ]
        rows = []
        common = self._common_mask
        for name, result in self.results.items():
            for label, index in groups:
                group_mask = common[index]
                metrics = compute_metrics(
                    result.actual[index][group_mask], result.mean[index][group_mask]
                )
                rows.append({"model": name, by: label, **metrics})
        return pd.DataFrame.from_records(
            rows, columns=["model", by, *ERROR_METRIC_NAMES, "n"]
        )

    def _series_label(self, series):
        reference = next(iter(self.results.values()))
        if reference.mean.ndim == 2:
            if series not in {None, 0}:
                raise ValueError("series must be None or 0 for univariate results")
            return None
        labels = reference.series_names or tuple(
            f"series_{index}" for index in range(reference.mean.shape[2])
        )
        if series is None:
            raise ValueError("series is required for multivariate forecasts")
        if isinstance(series, (bool, np.bool_)):
            raise TypeError("series must be an integer position or name")
        if isinstance(series, (int, np.integer)):
            if not 0 <= int(series) < len(labels):
                raise IndexError("series position is out of range")
            return labels[int(series)]
        if not isinstance(series, str) or not series:
            raise TypeError("series must be an integer position or name")
        if series not in labels:
            raise ValueError(f"unknown series {series!r}")
        return series

    def plot_forecasts(
        self,
        *,
        horizon=None,
        series=None,
        title=None,
        xtitle=None,
        ytitle=None,
        freq=None,
        note=None,
        grid=False,
        show_intervals=True,
        interval_alpha=0.12,
        ax=None,
    ):
        """Plot aligned actuals and forecasts through ``TsPlots.plot_series``.

        Parameters
        ----------
        horizon : int, optional
            One forecast step to show when rolling windows overlap.
        series : int or str, optional
            Multivariate series position or name.
        title : str, optional
            Figure title.
        xtitle : str, optional
            Horizontal-axis title.
        ytitle : str, optional
            Vertical-axis title.
        freq : str, optional
            Date-axis display frequency passed to ``plot_series``.
        note : str, optional
            Figure note.
        grid : bool, default False
            Whether to show the plot grid.
        show_intervals : bool, default True
            Whether to shade available forecast intervals.
        interval_alpha : float, default 0.12
            Forecast-interval fill opacity.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on.

        Returns
        -------
        tuple
            Matplotlib figure and axes.

        Examples
        --------
        >>> fig, ax = report.plot_forecasts(horizon=1)
        """
        from Ts.TsPlots import plot_series

        reference = next(iter(self.results.values()))
        max_horizon = reference.mean.shape[1]
        if horizon is None and len(reference.splits) > 1 and max_horizon > 1:
            raise ValueError("horizon is required when rolling forecasts overlap")
        if horizon is None:
            horizon = 1 if max_horizon == 1 else None
        elif isinstance(horizon, (bool, np.bool_)) or not isinstance(
            horizon, (int, np.integer)
        ):
            raise TypeError("horizon must be an integer")
        elif not 1 <= int(horizon) <= max_horizon:
            raise ValueError(f"horizon must be between 1 and {max_horizon}")
        else:
            horizon = int(horizon)
        series_label = self._series_label(series)
        if not isinstance(show_intervals, (bool, np.bool_)):
            raise TypeError("show_intervals must be a boolean")
        if isinstance(interval_alpha, (bool, np.bool_)):
            raise TypeError("interval_alpha must be a number between 0 and 1")
        interval_alpha = float(interval_alpha)
        if not np.isfinite(interval_alpha) or not 0 <= interval_alpha <= 1:
            raise ValueError("interval_alpha must be between 0 and 1")

        values = self.predictions
        if horizon is not None:
            values = values.loc[values["horizon"] == horizon]
        if reference.mean.ndim == 3:
            values = values.loc[values["series"] == series_label]
        actual = values.groupby("target_time", sort=True)["actual"].first()
        forecasts = values.pivot(index="target_time", columns="model", values="forecast")
        frame = pd.concat(
            [actual.rename("actual"), forecasts.reindex(columns=list(self.results))],
            axis=1,
        )
        old_lines = 0 if ax is None else len(ax.lines)
        fig, ax = plot_series(
            frame,
            facet=False,
            auto_dual_y=False,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            freq=freq,
            note=note,
            grid=grid,
            show_legend=False,
            ax=ax,
        )
        lines = ax.lines[old_lines : old_lines + len(frame.columns)]
        if show_intervals:
            for line, (name, result) in zip(lines[1:], self.results.items(), strict=True):
                if result.lower is None:
                    continue
                rows = values.loc[values["model"] == name].sort_values("target_time")
                confidence = "" if result.alpha is None else f" {(1-result.alpha)*100:g}%"
                ax.fill_between(
                    rows["target_time"],
                    rows["lower"],
                    rows["upper"],
                    color=line.get_color(),
                    alpha=interval_alpha,
                    label=f"{name}{confidence} interval",
                )
        ax.legend(frameon=False, ncol=2)
        return fig, ax

    def plot_metric(
        self,
        metric="rmse",
        *,
        by="origin",
        title=None,
        xtitle=None,
        ytitle=None,
        freq=None,
        note=None,
        grid=False,
        ax=None,
    ):
        """Plot one metric by origin or horizon via ``TsPlots.plot_series``.

        Parameters
        ----------
        metric : str, default "rmse"
            Canonical error metric to plot.
        by : {"origin", "horizon"}, default "origin"
            Evaluation axis shown horizontally.
        title : str, optional
            Figure title.
        xtitle : str, optional
            Horizontal-axis title.
        ytitle : str, optional
            Vertical-axis title.
        freq : str, optional
            Date-axis display frequency passed to ``plot_series``.
        note : str, optional
            Figure note.
        grid : bool, default False
            Whether to show the plot grid.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on.

        Returns
        -------
        tuple
            Matplotlib figure and axes.

        Examples
        --------
        >>> fig, ax = report.plot_metric("rmse", by="origin")
        """
        from Ts.TsPlots import plot_series

        if metric not in ERROR_METRIC_NAMES:
            raise ValueError(f"metric must be one of {list(ERROR_METRIC_NAMES)}")
        if by not in {"origin", "horizon"}:
            raise ValueError("by must be 'origin' or 'horizon' for metric plots")
        table = self.metric_table(by=by)
        frame = table.pivot(index=by, columns="model", values=metric).reindex(
            columns=list(self.results)
        )
        return plot_series(
            frame,
            facet=False,
            auto_dual_y=False,
            title=title,
            xtitle=xtitle,
            ytitle=metric.upper() if ytitle is None else ytitle,
            freq=freq,
            note=note,
            grid=grid,
            ax=ax,
        )


__all__ = ["ForecastComparisonResult", "ForecastEvaluationResult"]
