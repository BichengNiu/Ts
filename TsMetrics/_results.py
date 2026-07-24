"""Result containers for forecast performance evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _optional_float_array(values):
    """Return a copied float array while preserving None."""
    if values is None:
        return None
    return np.array(values, dtype=float, copy=True)


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
    metrics: dict
    metrics_by_series: list[dict]
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
        self.metrics = dict(self.metrics)
        self.metrics_by_series = [
            dict(metrics) for metrics in self.metrics_by_series
        ]

        if self.mean.ndim not in (1, 2):
            raise ValueError(
                "mean must have shape (horizon,) or (horizon, n_series)"
            )
        if self.actual.shape != self.mean.shape:
            raise ValueError(
                "actual must have the same shape as mean, got "
                f"{self.actual.shape} and {self.mean.shape}"
            )
        if self.estimation_indices.ndim != 1 or not len(
            self.estimation_indices
        ):
            raise ValueError("estimation_indices must be a non-empty 1-D array")
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
            if len(self.estimation_dates) != len(self.estimation_indices):
                raise ValueError(
                    "estimation_dates must align with estimation_indices"
                )
            if len(self.validation_dates) != len(self.validation_indices):
                raise ValueError(
                    "validation_dates must align with validation_indices"
                )
            if self.validation_dates[0] <= self.estimation_dates[-1]:
                raise ValueError(
                    "validation dates must be strictly later than estimation dates"
                )
        for name, values in (("lower", self.lower), ("upper", self.upper)):
            if values is not None and values.shape != self.mean.shape:
                raise ValueError(
                    f"{name} must have the same shape as mean, got "
                    f"{values.shape} and {self.mean.shape}"
                )

@dataclass
class BacktestResult:
    """Rolling-origin forecast evaluation result."""

    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    origins: np.ndarray
    target_indices: np.ndarray
    metrics: dict
    metrics_by_horizon: list[dict]
    metrics_by_series: list[dict]
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
        self.target_indices = np.array(
            self.target_indices,
            dtype=int,
            copy=True,
        )
        self.metrics = dict(self.metrics)
        self.metrics_by_horizon = [
            dict(metrics) for metrics in self.metrics_by_horizon
        ]
        self.metrics_by_series = [
            dict(metrics) for metrics in self.metrics_by_series
        ]
        self.failures = [dict(failure) for failure in self.failures]

        if self.mean.ndim not in (2, 3):
            raise ValueError(
                "mean must have shape (n_origins, horizon) or "
                "(n_origins, horizon, n_series)"
            )
        if self.actual.shape != self.mean.shape:
            raise ValueError(
                "actual must have the same shape as mean, got "
                f"{self.actual.shape} and {self.mean.shape}"
            )
        for name, values in (("lower", self.lower), ("upper", self.upper)):
            if values is not None and values.shape != self.mean.shape:
                raise ValueError(
                    f"{name} must have the same shape as mean, got "
                    f"{values.shape} and {self.mean.shape}"
                )
        if self.origins.ndim != 1 or len(self.origins) != self.mean.shape[0]:
            raise ValueError(
                "origins must have one entry per forecast origin"
            )
        if self.target_indices.shape != self.mean.shape[:2]:
            raise ValueError(
                "target_indices must have shape (n_origins, horizon), got "
                f"{self.target_indices.shape}"
            )


@dataclass
class ComparisonResult:
    """Ranking of comparable forecast evaluation results."""

    metric: str
    scores: dict[str, float]
    ranking: list[str]
    target: str

    def __post_init__(self):
        """Copy mappings and require every model to appear once in ranking."""
        if not all(isinstance(name, str) for name in self.scores):
            raise TypeError("score names must be strings")
        if not all(isinstance(name, str) for name in self.ranking):
            raise TypeError("ranking names must be strings")
        self.scores = {
            name: float(value) for name, value in self.scores.items()
        }
        self.ranking = list(self.ranking)
        if (
            len(self.ranking) != len(self.scores)
            or set(self.ranking) != set(self.scores)
        ):
            raise ValueError("ranking must contain every scored model once")
