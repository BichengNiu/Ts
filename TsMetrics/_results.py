"""Result containers for forecast performance evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _optional_float_array(values):
    """Return a copied float array while preserving None."""
    if values is None:
        return None
    return np.array(values, dtype=float, copy=True)


@dataclass
class OOSResult:
    """Single-split, leakage-free holdout evaluation result."""

    mean: np.ndarray
    actual: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    target_indices: np.ndarray
    metrics: dict
    metrics_by_series: list[dict]
    model_type: str
    target: str
    split: int

    def __post_init__(self):
        """Normalise arrays and validate the holdout shape contract."""
        self.mean = np.array(self.mean, dtype=float, copy=True)
        self.actual = np.array(self.actual, dtype=float, copy=True)
        self.lower = _optional_float_array(self.lower)
        self.upper = _optional_float_array(self.upper)
        self.target_indices = np.array(
            self.target_indices,
            dtype=int,
            copy=True,
        )
        self.metrics = dict(self.metrics)
        self.metrics_by_series = [
            dict(metrics) for metrics in self.metrics_by_series
        ]
        self.split = int(self.split)

        if self.mean.ndim not in (1, 2):
            raise ValueError(
                "mean must have shape (horizon,) or (horizon, n_series)"
            )
        if self.actual.shape != self.mean.shape:
            raise ValueError(
                "actual must have the same shape as mean, got "
                f"{self.actual.shape} and {self.mean.shape}"
            )
        if self.target_indices.shape != (self.mean.shape[0],):
            raise ValueError(
                "target_indices must contain one entry per holdout period"
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
