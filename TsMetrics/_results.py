"""Result containers for forecast performance evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._aggregation import (
    backtest_metrics_by_series,
    metrics_by_horizon,
    oos_metrics_by_series,
)
from ._metrics import ERROR_METRIC_NAMES, compute_metrics


def _optional_float_array(values):
    """Return a copied float array while preserving None."""
    if values is None:
        return None
    return np.array(values, dtype=float, copy=True)


def _validate_interval_pair(lower, upper, expected_shape):
    """Require complete, aligned, and ordered prediction intervals."""
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must both be set or both be None")
    if lower is None:
        return
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError(
            "lower and upper must have the same shape as mean, got "
            f"{lower.shape}, {upper.shape}, and {expected_shape}"
        )
    finite = np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[finite] > upper[finite]):
        raise ValueError("lower must not exceed upper")


def _validate_increasing(name, values):
    """Require a non-empty, strictly increasing one-dimensional index."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def _validate_dates(name, values, expected_length):
    """Require complete, unique, increasing date metadata."""
    if len(values) != expected_length:
        raise ValueError(f"{name} must align with its positional indices")
    if values.hasnans:
        raise ValueError(f"{name} must not contain missing dates")
    if not values.is_unique or not values.is_monotonic_increasing:
        raise ValueError(f"{name} must be unique and strictly increasing")


def _validate_result_labels(model_type, target):
    """Require meaningful model and evaluation-target labels."""
    if not isinstance(model_type, str) or not model_type:
        raise TypeError("model_type must be a non-empty string")
    if not isinstance(target, str) or not target:
        raise TypeError("target must be a non-empty string")


def _rank_scores(scores):
    """Return deterministic ascending error-score ranking."""
    return sorted(
        scores,
        key=lambda name: (
            not np.isfinite(scores[name]),
            scores[name] if np.isfinite(scores[name]) else np.inf,
            name,
        ),
    )


@dataclass
class OOSResult:
    """Leakage-free evaluation over explicit estimation and validation periods."""

    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    estimation_indices: np.ndarray
    validation_indices: np.ndarray
    estimation_dates: pd.DatetimeIndex | None
    validation_dates: pd.DatetimeIndex | None
    model_type: str
    target: str

    def __post_init__(self):
        """Normalise arrays and validate the period metadata contract."""
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.actual = np.array(self.actual, dtype=float, copy=True)
        self.lower = _optional_float_array(self.lower)
        self.upper = _optional_float_array(self.upper)
        self.estimation_indices = np.array(
            self.estimation_indices,
            dtype=int,
            copy=True,
        )
        self.validation_indices = np.array(
            self.validation_indices,
            dtype=int,
            copy=True,
        )
        self.estimation_dates = (
            None
            if self.estimation_dates is None
            else pd.DatetimeIndex(self.estimation_dates).copy()
        )
        self.validation_dates = (
            None
            if self.validation_dates is None
            else pd.DatetimeIndex(self.validation_dates).copy()
        )
        if self.mean.ndim not in (1, 2):
            raise ValueError("mean must have shape (horizon,) or (horizon, n_series)")
        if self.mean.shape[0] == 0 or (self.mean.ndim == 2 and self.mean.shape[1] == 0):
            raise ValueError("mean must contain at least one forecast value")
        if self.actual.shape != self.mean.shape:
            raise ValueError(
                "actual must have the same shape as mean, got "
                f"{self.actual.shape} and {self.mean.shape}"
            )
        _validate_increasing("estimation_indices", self.estimation_indices)
        _validate_increasing("validation_indices", self.validation_indices)
        if np.any(np.diff(self.estimation_indices) != 1):
            raise ValueError("estimation_indices must be contiguous")
        if np.any(np.diff(self.validation_indices) != 1):
            raise ValueError("validation_indices must be contiguous")
        if self.validation_indices.shape != (self.mean.shape[0],):
            raise ValueError(
                "validation_indices must contain one entry per validation period"
            )
        if self.validation_indices[0] <= self.estimation_indices[-1]:
            raise ValueError(
                "validation indices must be strictly later than estimation indices"
            )
        if (self.estimation_dates is None) != (self.validation_dates is None):
            raise ValueError(
                "estimation_dates and validation_dates must both be set or both be None"
            )
        if self.estimation_dates is not None:
            _validate_dates(
                "estimation_dates",
                self.estimation_dates,
                len(self.estimation_indices),
            )
            _validate_dates(
                "validation_dates",
                self.validation_dates,
                len(self.validation_indices),
            )
            if self.validation_dates[0] <= self.estimation_dates[-1]:
                raise ValueError(
                    "validation dates must be strictly later than estimation dates"
                )
        _validate_interval_pair(self.lower, self.upper, self.mean.shape)
        _validate_result_labels(self.model_type, self.target)

    @property
    def metrics(self):
        """Compute overall metrics from the current result arrays."""
        return compute_metrics(self.actual, self.mean)

    @property
    def metrics_by_series(self):
        """Compute metrics for each endogenous series."""
        return oos_metrics_by_series(self.actual, self.mean)


@dataclass
class BacktestResult:
    """Rolling-origin forecast evaluation result."""

    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    origins: np.ndarray
    failures: list[dict]
    model_type: str
    window: str
    target: str

    def __post_init__(self):
        """Normalise arrays and reject incompatible result shapes."""
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.actual = np.array(self.actual, dtype=float, copy=True)
        self.lower = _optional_float_array(self.lower)
        self.upper = _optional_float_array(self.upper)
        self.origins = np.array(self.origins, dtype=int, copy=True)
        self.failures = [dict(failure) for failure in self.failures]

        if self.mean.ndim not in (2, 3):
            raise ValueError(
                "mean must have shape (n_origins, horizon) or "
                "(n_origins, horizon, n_series)"
            )
        if (
            self.mean.shape[0] == 0
            or self.mean.shape[1] == 0
            or (self.mean.ndim == 3 and self.mean.shape[2] == 0)
        ):
            raise ValueError("mean must contain at least one forecast value")
        if self.actual.shape != self.mean.shape:
            raise ValueError(
                "actual must have the same shape as mean, got "
                f"{self.actual.shape} and {self.mean.shape}"
            )
        _validate_interval_pair(self.lower, self.upper, self.mean.shape)
        _validate_increasing("origins", self.origins)
        if len(self.origins) != self.mean.shape[0]:
            raise ValueError("origins must have one entry per forecast origin")
        if self.window not in {"expanding", "rolling"}:
            raise ValueError("window must be either 'expanding' or 'rolling'")
        failed_origins = set()
        for failure in self.failures:
            required = {"origin", "error_type", "message"}
            if not required.issubset(failure):
                raise ValueError(
                    "each failure must contain origin, error_type, and message"
                )
            origin = failure["origin"]
            matches = np.flatnonzero(self.origins == origin)
            if matches.size != 1 or origin in failed_origins:
                raise ValueError("failure origins must uniquely identify result rows")
            row = int(matches[0])
            if (
                not np.isnan(self.mean[row]).all()
                or not np.isnan(self.actual[row]).all()
            ):
                raise ValueError("failed origins must retain all-NaN result rows")
            failed_origins.add(origin)
        _validate_result_labels(self.model_type, self.target)

    @property
    def target_indices(self):
        """Derive target positions from forecast origins and horizon."""
        horizon = np.arange(self.mean.shape[1], dtype=int)
        return self.origins[:, None] + horizon

    @property
    def metrics(self):
        """Compute overall metrics from the current result arrays."""
        return compute_metrics(self.actual, self.mean)

    @property
    def metrics_by_horizon(self):
        """Compute metrics for each forecast horizon."""
        return metrics_by_horizon(self.actual, self.mean)

    @property
    def metrics_by_series(self):
        """Compute metrics for each endogenous series."""
        return backtest_metrics_by_series(self.actual, self.mean)


@dataclass
class ComparisonResult:
    """Ranking of comparable forecast evaluation results."""

    metric: str
    scores: dict[str, float]
    target: str

    def __post_init__(self):
        """Copy mappings and require every model to appear once in ranking."""
        if not isinstance(self.metric, str) or not self.metric:
            raise TypeError("metric must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise TypeError("target must be a non-empty string")
        if not all(isinstance(name, str) for name in self.scores):
            raise TypeError("score names must be strings")
        if not self.scores:
            raise ValueError("scores must not be empty")
        self.scores = {name: float(value) for name, value in self.scores.items()}
        if any(np.isfinite(score) and score < 0.0 for score in self.scores.values()):
            raise ValueError("finite error scores must be non-negative")

    @property
    def ranking(self):
        """Return model names sorted by ascending finite score."""
        return _rank_scores(self.scores)


@dataclass
class OOSComparisonResult:
    """Multi-model OOS evaluations and their complete metric table."""

    evaluations: dict[str, OOSResult]
    rank_by: str

    def __post_init__(self):
        """Copy public state and validate the report's structural contract."""
        if not isinstance(self.evaluations, dict):
            raise TypeError("evaluations must be a dict of names to OOSResult values")
        if not self.evaluations:
            raise ValueError("evaluations must not be empty")
        if not all(isinstance(name, str) for name in self.evaluations):
            raise TypeError("evaluation names must be strings")
        if not all(
            isinstance(evaluation, OOSResult)
            for evaluation in self.evaluations.values()
        ):
            raise TypeError("evaluations must contain only OOSResult values")
        if self.rank_by not in ERROR_METRIC_NAMES:
            raise ValueError(
                f"rank_by must be one of {list(ERROR_METRIC_NAMES)}, "
                f"got {self.rank_by!r}"
            )
        self.evaluations = dict(self.evaluations)

    @property
    def target(self):
        """Return the shared forecast target."""
        return next(iter(self.evaluations.values())).target

    @property
    def scores(self):
        """Return one score per model for the selected ranking metric."""
        return {
            name: float(evaluation.metrics[self.rank_by])
            for name, evaluation in self.evaluations.items()
        }

    @property
    def ranking(self):
        """Return model names ordered by the selected error metric."""
        return _rank_scores(self.scores)

    @property
    def best_model(self):
        """Return the best finite-scoring model, or None if no score is finite."""
        ranking = self.ranking
        if not ranking or not np.isfinite(self.scores[ranking[0]]):
            return None
        return ranking[0]

    @property
    def table(self):
        """Return a ranking-ordered DataFrame containing every error metric."""
        frame = pd.DataFrame.from_dict(
            {
                name: evaluation.metrics
                for name, evaluation in self.evaluations.items()
            },
            orient="index",
        )
        frame = frame.loc[:, [*ERROR_METRIC_NAMES, "n"]]
        frame.index.name = "model"
        ranks = {name: rank for rank, name in enumerate(self.ranking, start=1)}
        frame["rank"] = pd.Series(ranks, dtype=int)
        return frame.loc[self.ranking].copy()
