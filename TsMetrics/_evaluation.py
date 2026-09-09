"""Model validation and isolated forecast execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._results import _normalise_names

def model_series_names(model, data):
    """Return validated display names for a multivariate model, when available."""
    if data.ndim == 1:
        return None
    names = getattr(model, "data_names", None)
    if names is None:
        return None
    return _normalise_names(names, data.shape[1], "model.data_names")


def evaluation_actual(model, observed, train_data, expected_shape):
    """Return and validate one evaluation target window."""
    actual = np.asarray(
        model._evaluation_actual(observed, train_data),
        dtype=float,
    )
    if actual.shape != expected_shape:
        raise ValueError(
            f"evaluation target has shape {actual.shape}, expected {expected_shape}"
        )
    return actual


def training_exog(model, start, stop):
    """Return a copied exogenous training window when the model has one."""
    exog = getattr(model, "exog", None)
    if exog is None:
        return None
    if hasattr(exog, "iloc"):
        return exog.iloc[start:stop].copy()
    return np.array(np.asarray(exog)[start:stop], dtype=float, copy=True)


def training_dates(dates, start, stop):
    """Return a copied date-index training window when available."""
    if dates is None:
        return None
    return dates[start:stop].copy()


def validate_model_protocol(
    model: Any,
    context: str,
) -> str:
    """Validate the evaluation contract and return its target name."""
    required = (
        "_clone_for_evaluation",
        "_evaluation_actual",
        "_evaluation_predict_kwargs",
        "_validate_evaluation",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(
            "model does not implement the evaluation protocol: " + ", ".join(missing)
        )
    target = getattr(model, "_evaluation_target_name", None)
    if not isinstance(target, str) or not target:
        raise TypeError(
            "model does not implement the evaluation protocol: _evaluation_target_name"
        )
    model._validate_evaluation(context)
    return target
