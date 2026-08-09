# Ts/TsMetrics

时间序列预测误差指标与统一的历史泛化能力评估。

## 公共接口

```python
from Ts.TsMetrics import (
    mae, mse, rmse, mape, smape, theil_u1, compute_metrics,
    Holdout, RollingOrigin,
    ForecastEvaluationResult, ForecastComparisonResult,
    evaluate_forecasts,
)
```

旧的 `oos`、`backtest`、`compare_forecasts`、`evaluate_models_oos` 以及对应结果类型已经
彻底删除，不提供兼容别名。固定留出、扩展窗口和固定滚动窗口都使用同一个函数；区别只由
`scheme` 表达。

## 误差指标

所有指标要求 `actual` 与 `predicted` 形状一致。默认忽略任一侧非有限的配对；
`nan_policy="raise"` 可改为立即报错。

| 指标 | 含义 | 方向 |
|---|---|---|
| `mae` | 平均绝对误差 | 越小越好 |
| `mse` | 均方误差 | 越小越好 |
| `rmse` | 均方根误差 | 越小越好 |
| `mape` | 平均绝对百分比误差 | 越小越好；实际值为零的项不参与 |
| `smape` | 对称平均绝对百分比误差 | 越小越好 |
| `theil_u1` | Theil U1 不平等系数 | 越小越好 |

`compute_metrics(actual, predicted)` 一次返回全部指标和有效配对数 `n`。

## 固定留出评估

位置或日期边界都采用闭区间：

```python
from Ts.TsMetrics import Holdout, evaluate_forecasts

report = evaluate_forecasts(
    {"AR(1)": ar1_model, "AR(2)": ar2_model},
    scheme=Holdout(train=(0, 79), test=(80, 99)),
    rank_by="rmse",
    fit_kwargs={"method": "lbfgs", "maxiter": 500},
)
```

测试期可以晚于训练期，中间形成 `gap`。引擎会从训练期末连续预测跨过间隔，但只对测试期
评分。日期边界必须真实存在；训练期至少包含 10 个观测。

## 滚动起点评估

扩展窗口：

```python
from Ts.TsMetrics import RollingOrigin

report = evaluate_forecasts(
    {"ARIMA": arima_model, "RDL": rdl_model},
    scheme=RollingOrigin(
        initial_window=400,
        horizon=2,
        step=1,
        window="expanding",
    ),
    fit_kwargs={"method": "lbfgs", "maxiter": 500},
    future_exog="observed",
)
```

固定长度滚动窗口：

```python
scheme = RollingOrigin(
    initial_window=80,
    horizon=4,
    step=4,
    window="rolling",
    window_size=60,
    gap=1,
)
```

同一个方案还覆盖两种常见泛化检验，无需增加新接口：

```python
# 逐期在线 / prequential：每次只预测下一期
online = RollingOrigin(initial_window=80, horizon=1, step=1)

# 非重叠历史测试区块：下一起点正好接在上一预测块之后
blocked = RollingOrigin(initial_window=80, horizon=4, step=4)
```

每个起点都克隆并重新拟合模型，只使用该起点之前、方案允许的训练数据。原模型及其
`result_` 不会被修改。只有完整的预测窗口进入结果。

## 外生变量与条件预测

只要任一模型使用外生变量，就必须显式声明：

```python
report = evaluate_forecasts(
    models,
    scheme=scheme,
    future_exog="observed",
)
```

这表示评价使用历史中实际实现的未来外生变量路径，因此是条件预测比较。真实部署时若这些
变量在预测起点未知，应先预测它们或建立情景，不能把事后实现值解释为实时可知信息。

## 拟合参数与失败策略

`fit_kwargs` 是传给每个模型、每个拆分的统一拟合参数。引擎会在第一次拟合之前验证所有模型
是否接受这些参数，避免比较执行到一半才发现某个模型不支持优化器。

- `on_error="raise"`：任一拆分失败时立即抛出异常；
- `on_error="record"`：整个失败拆分保留为 NaN，并写入 `report.failures`。

排名只使用所有模型共同具有有限实际值和预测值的配对。`report.table` 同时报告 `n_total`、
`n_common`、`coverage` 和 `failures`，使缺失预测不会通过缩小计分样本获得优势。

## 统一结果

无论方案类型，返回值都是 `ForecastComparisonResult`：

```python
report.table
report.ranking
report.best_model
report.results["RDL"].metrics
report.predictions
report.splits
report.failures
report.metric_table(by="origin")
report.metric_table(by="horizon")
report.metric_table(by="series")
```

单模型结果数组形状固定为：

- 单变量：`(n_splits, horizon)`；
- 多变量：`(n_splits, horizon, n_series)`。

固定留出只是 `n_splits == 1`，不会退化成另一套一维结果契约。

当 `step < horizon` 时，相邻窗口会包含相同目标时点，但代表不同起点或预测步长下的预测
任务；pooled RMSE、MAPE 等会保留这些误差。它们不是相互独立的观测，因此统计解释时应
同时查看 `metric_table(by="origin")` 与 `metric_table(by="horizon")`，并明确重叠依赖。

`TsModels.compare_models()` 比较的是已经拟合的参数估计结果（例如 AIC、BIC 和系数表）；
`evaluate_forecasts()` 比较的才是时间有序历史切分上的泛化预测误差，两者不能互换。

## 绘图

绘图复用 `TsPlots.plot_series()` 的样式：

```python
fig, ax = report.plot_forecasts(
    horizon=1,
    title="Rolling-origin forecasts",
    freq="month",
    grid=True,
)

fig, ax = report.plot_metric(
    "rmse",
    by="origin",
    title="Rolling RMSE",
)
```

多变量结果必须指定 `series`。多个滚动窗口且 `horizon > 1` 时，预测图必须指定一个
`horizon`，避免把重叠目标期的不同步长预测混成一条含义不清的曲线。

## 运行测试

```powershell
python -m pytest TsMetrics/tests -q
```
