"""SARIMA-error backend for the ARDL result prediction contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Ts.TsModels._base import PredictResult


class _ARDLSARIMAErrorBridge:
    """Keep SARIMA-error design and recursive forecasting in one adapter.

    ``ARDLResult`` remains the public result surface.  This adapter owns the
    effective-design and future-recursion details needed only when a manually
    configured SARIMA error is present.
    """

    def __init__(self, result):
        self.result = result

    def _build_future_design(self, steps, future_exog, future_dates):
        result = self.result
        error_result = result._error_result
        if error_result is None or result._ardl_model is None:
            raise RuntimeError("SARIMA error state is unavailable")
        future_inputs = None
        if result.exog_names:
            future_inputs = result._normalise_future_exog(steps, future_exog)
        elif future_exog is not None:
            raise ValueError(
                "future_exog is invalid because the model has no explanatory inputs"
            )

        index = future_dates.copy() if future_dates is not None else pd.RangeIndex(steps)
        columns = tuple(error_result.exog_names)
        future_design = pd.DataFrame(index=index, columns=columns, dtype=float)

        deterministics = getattr(result._ardl_model, "_deterministics", None)
        if deterministics is not None:
            if future_dates is None:
                deterministic = deterministics.out_of_sample(steps)
            else:
                deterministic = deterministics.out_of_sample(
                    steps,
                    forecast_index=future_dates,
                )
            for name in deterministic.columns:
                if name in future_design.columns:
                    future_design[name] = np.asarray(deterministic[name], dtype=float)

        historical_response = np.asarray(result._model_data, dtype=float)
        response_path = np.full(result.nobs + steps, np.nan, dtype=float)
        response_path[: result.nobs] = historical_response
        exog_path = None
        if result._exog is not None:
            exog_path = np.vstack(
                [
                    result._exog.to_numpy(dtype=float),
                    future_inputs.to_numpy(dtype=float),
                ]
            )

        lag_columns = {f"y.L{lag}": ("response", lag) for lag in result.ar_lags}
        for name, lags in result.distributed_lags.items():
            for lag in lags:
                lag_columns[f"{name}.L{lag}"] = (name, lag)

        for step in range(steps):
            for column, (source, lag) in lag_columns.items():
                if column not in future_design.columns:
                    continue
                source_position = result.nobs + step - lag
                if source == "response":
                    value = response_path[source_position]
                else:
                    if exog_path is None:
                        raise RuntimeError("fitted explanatory input history is missing")
                    value = exog_path[
                        source_position,
                        result.exog_names.index(source),
                    ]
                future_design.loc[index[step], column] = value

            partial_design = future_design.iloc[: step + 1].copy()
            future_prediction = error_result.predict(
                start=error_result.nobs,
                end=error_result.nobs + step,
                dynamic=False,
                alpha=0.05,
                future_exog=partial_design,
                future_dates=(
                    None if future_dates is None else future_dates[: step + 1]
                ),
            )
            response_path[result.nobs + step] = float(future_prediction.mean[-1])

        if not np.all(np.isfinite(future_design.to_numpy(dtype=float))):
            raise RuntimeError("future ARDL design contains non-finite values")
        return future_design

    def predict(self, window, *, dynamic, alpha, future_exog, future_dates):
        """Predict the ARDL response through the fitted SARIMA error backend."""
        result = self.result
        error_result = result._error_result
        if error_result is None:
            raise RuntimeError("SARIMA error state is unavailable")

        mean = np.full(window.size, np.nan, dtype=float)
        lower = np.full(window.size, np.nan, dtype=float)
        upper = np.full(window.size, np.nan, dtype=float)

        if window.in_sample_size:
            valid_start = max(window.start, result.hold_back)
            valid_end = min(window.end, result.nobs - 1)
            if valid_start <= valid_end:
                prediction = error_result.predict(
                    start=valid_start - result.hold_back,
                    end=valid_end - result.hold_back,
                    dynamic=dynamic,
                    alpha=alpha,
                    response_log=result.log,
                )
                destination = valid_start - window.start
                size = valid_end - valid_start + 1
                mean[destination : destination + size] = prediction.mean[:size]
                lower[destination : destination + size] = prediction.lower[:size]
                upper[destination : destination + size] = prediction.upper[:size]

        if window.has_forecast:
            resolved_dates = error_result.resolve_future_dates(
                window.forecast_steps,
                future_dates,
            )
            future_design = self._build_future_design(
                window.forecast_steps,
                future_exog,
                resolved_dates,
            )
            prediction = error_result.predict(
                start=error_result.nobs,
                end=error_result.nobs + window.forecast_steps - 1,
                dynamic=dynamic,
                alpha=alpha,
                future_exog=future_design,
                future_dates=resolved_dates,
                response_log=result.log,
            )
            destination = window.in_sample_size
            source = window.forecast_skip
            mean[destination:] = prediction.mean[source:]
            lower[destination:] = prediction.lower[source:]
            upper[destination:] = prediction.upper[source:]

        full = error_result.predict(
            start=0,
            end=error_result.nobs - 1,
            alpha=alpha,
            response_log=result.log,
        )
        full_lower = np.full(result.nobs, np.nan, dtype=float)
        full_upper = np.full(result.nobs, np.nan, dtype=float)
        if full._full_lower is not None:
            full_lower[result.hold_back :] = full._full_lower
            full_upper[result.hold_back :] = full._full_upper
        return PredictResult(
            mean=mean,
            lower=lower,
            upper=upper,
            is_oos=np.arange(window.start, window.end + 1) >= result.nobs,
            _full_data=result.data,
            _full_fitted=result.fitted_values,
            _full_lower=full_lower,
            _full_upper=full_upper,
            _start=window.start,
        )
