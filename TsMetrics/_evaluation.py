"""Model validation and isolated forecast execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from Ts.TsUtils._forecast import model_data
from Ts.TsUtils._forecast import normalise_forecast_result
from Ts.TsUtils._forecast import parameter_snapshot
from Ts.TsUtils._forecast import prediction_arrays
from ._results import _normalise_names


def evaluation_data(model):
    """Return validated one- or two-dimensional data owned by an estimator."""
    return model_data(model)


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


def forecast_for_evaluation(
    model,
    train_data,
    *,
    exog,
    dates,
    horizon,
    alpha,
    fit_kwargs,
    evaluation_start,
    evaluation_stop,
    expected_shape,
):
    """Run the new model seam, or adapt the historical structural protocol."""
    runner = getattr(model, "_forecast_for_evaluation", None)
    if callable(runner):
        return normalise_forecast_result(
            runner(
                train_data,
                exog=exog,
                dates=dates,
                horizon=horizon,
                alpha=alpha,
                fit_kwargs=fit_kwargs,
                evaluation_start=evaluation_start,
                evaluation_stop=evaluation_stop,
            ),
            expected_shape,
        )

    clone = getattr(model, "_clone_for_evaluation", None)
    predict_kwargs = getattr(model, "_evaluation_predict_kwargs", None)
    if not callable(clone) or not callable(predict_kwargs):
        raise TypeError(
            "model does not implement the evaluation execution protocol: "
            "_forecast_for_evaluation or legacy clone/predict hooks"
        )
    cloned = clone(train_data, exog=exog, dates=dates)
    fit = getattr(cloned, "fit", None)
    if not callable(fit):
        raise TypeError("_clone_for_evaluation() must return an object with fit()")
    fitted = fit(**fit_kwargs)
    nobs = getattr(fitted, "nobs", None)
    if (
        isinstance(nobs, (bool, np.bool_))
        or not isinstance(nobs, (int, np.integer))
        or nobs < 1
    ):
        raise TypeError("fit() must return a result with a positive integer nobs")
    model_type = getattr(fitted, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise TypeError("fit() must return a result with a non-empty model_type")
    predict = getattr(fitted, "predict", None)
    if not callable(predict):
        raise NotImplementedError(f"{type(model).__name__} does not provide predict()")
    prediction = predict(
        start=int(nobs),
        end=int(nobs) + horizon - 1,
        alpha=alpha,
        **predict_kwargs(evaluation_start, evaluation_stop),
    )
    return normalise_forecast_result(
        (
            model_type,
            parameter_snapshot(fitted),
            prediction_arrays(prediction, expected_shape),
        ),
        expected_shape,
    )


def validate_model_protocol(
    model: Any,
    context: str,
) -> str:
    """Validate the evaluation contract and return its target name."""
    required = ("_evaluation_actual", "_validate_evaluation")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    has_new_protocol = callable(getattr(model, "_forecast_for_evaluation", None))
    has_legacy_protocol = all(
        callable(getattr(model, name, None))
        for name in ("_clone_for_evaluation", "_evaluation_predict_kwargs")
    )
    if not has_new_protocol and not has_legacy_protocol:
        missing.append(
            "_forecast_for_evaluation or "
            "(_clone_for_evaluation and _evaluation_predict_kwargs)"
        )
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
