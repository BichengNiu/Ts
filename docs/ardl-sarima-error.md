# ARDL 的 SARIMA 误差结构

`TsModels.ARDL` 的目标变量滞后和输入变量滞后始终由 `lags`、`order` 手动指定。
需要相关误差项时，再显式提供 `error_order=(p, d, q)` 以及可选的
`error_seasonal_order=(P, D, Q, s)`：

```python
from Ts.TsModels import ARDL

result = ARDL(
    y,
    lags=[1, 2],
    exog=x,
    order={"x1": [0, 2], "x2": [1]},
    error_order=(1, 0, 1),
    error_seasonal_order=(0, 0, 1, 12),
    error_enforce_stationarity=True,
    error_enforce_invertibility=True,
).fit(
    error_method="bfgs",
    error_maxiter=500,
    error_cov_type="oim",
)
```

误差项通过 Ts 现有的 `SARIMAX` 状态空间实现估计。`ARDLResult` 仍保留
`ar_lags`、`distributed_lags` 和 `ardl_order`，并额外提供
`error_order`、`error_seasonal_order`、`error_likelihood_burn`、
`error_arroots` 和 `error_marroots`。省略 `error_order` 时保持原有条件 OLS
ARDL 行为；此时不能单独指定非零的季节误差阶数。

带外生变量的预测仍需通过 `future_exog` 提供完整未来路径；日期频率无法推断
时可通过 `future_dates` 提供未来日期。
