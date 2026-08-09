# Ts/TsMetrics

时间序列预测性能指标与无信息泄漏评估工具包。

`TsMetrics` 只负责度量和评估；模型拟合、样本内预测和未来预测由
`TsModels` 负责。评估引擎通过 `BaseModel` 协议调用模型，不导入任何具体
模型类。

## 交互式帮助

公共指标、评估函数及结果对象的参数、返回值和可执行样例均已写入 docstring。
在 IPython/Jupyter 中输入 `?rmse`（或 `rmse?`），也可在 Python 中调用
`help(rmse)` 查看完整帮助。

## 公共接口

```python
from Ts.TsMetrics import (
    mae,
    mse,
    rmse,
    mape,
    smape,
    theil_u1,
    compute_metrics,
    oos,
    backtest,
    compare_forecasts,
    evaluate_models_oos,
    OOSResult,
    BacktestResult,
    ComparisonResult,
    OOSComparisonResult,
)
```

这些符号也从 `Ts` 顶层命名空间导出。

## 点预测指标

所有指标要求 `actual` 与 `predicted` 形状完全相同。默认
`nan_policy='omit'`，只剔除任一侧非有限的配对；使用
`nan_policy='raise'` 可以让非有限值立即报错。

| 指标 | 含义 | 说明 |
|---|---|---|
| `mae` | Mean Absolute Error | 原量纲，越小越好 |
| `mse` | Mean Squared Error | 平方量纲，越小越好 |
| `rmse` | Root Mean Squared Error | 原量纲，对大误差更敏感 |
| `mape` | Mean Absolute Percentage Error | 实际值为零的配对不参与计算 |
| `smape` | Symmetric MAPE | 实际值与预测值均为零时该项记为零 |
| `theil_u1` | Theil U1 inequality coefficient | 范围为 0 到 1；名称不与 Theil U2 混用 |

```python
metrics = compute_metrics(actual, predicted)
```

返回字典包含上述六项指标和有效配对数 `n`。

## 显式估计期与验证期 OOS

位置型数据使用零基、闭区间边界：

```python
evaluation = oos(
    model,
    estimation_period=(0, 79),
    validation_period=(80, 99),
)
# 等价便利入口：model.oos(...)
```

带日期索引的数据必须使用真实存在的日期边界：

```python
evaluation = model.oos(
    estimation_period=("2018-01-01", "2022-12-01"),
    validation_period=("2023-03-01", "2023-12-01"),
)
```

两个期间均为闭区间。验证期必须严格晚于估计期；允许中间存在间隔，
但模型会从估计期末端开始连续预测并跨过该间隔，只对验证期评分。
日期边界必须精确存在，不会自动吸附到最近日期。越界、逆序、重叠、
估计期不足 10 个观测或外生变量无法覆盖完整预测桥接区间都会直接报错。

引擎只使用估计期数据克隆并拟合模型。原模型及其 `result_` 不会被修改。
`split` 参数已删除，不存在弃用或兼容路径。

`predict(oos_start=...)` 不存在。固定全样本参数再把尾部标成 OOS 会把
验证期信息带入参数估计，因此不作为性能评估接口。

`OOSResult` 提供：

- `mean`、`actual`、`lower`、`upper`；
- `estimation_indices` 与 `validation_indices`；
- 日期模型同时提供 `estimation_dates` 与 `validation_dates`；
- 向量模型提供 `series_names`，所有模型保留区间请求的 `alpha`；
- 总体 `metrics` 与逐变量 `metrics_by_series`；
- `target`，用于标明被评价的可观测对象。
## 滚动历史回测

```python
evaluation = backtest(
    model,
    initial_window=80,
    horizon=4,
    step=1,
    window="expanding",
)
# 等价便利入口：model.backtest(...)
```

每个预测起点都只使用此前可见的数据重新拟合。`window='rolling'` 时可用
`window_size` 指定固定训练窗口。结果按预测起点、预测期和变量保存，并
同时提供总体、逐预测期和逐变量指标。

`window_size` 只对 `window='rolling'` 有效；扩展窗口传入该参数会明确
抛出 `ValueError`，不会静默忽略。固定滚动窗口必须满足
`10 <= window_size <= initial_window`，因此从第一个预测原点开始就保持
同一训练长度，不会先扩展再滚动。

## 模型性能比较

当多个模型需要使用同一个估计期和验证期时，使用批量入口：

```python
report = evaluate_models_oos(
    {
        "AR(1)": ar1_model,
        "AR(2)": ar2_model,
    },
    estimation_period=(0, 79),
    validation_period=(80, 99),
    rank_by="rmse",
)

print(report.table)
print(report.ranking)
print(report.best_model)
ar1_evaluation = report.evaluations["AR(1)"]
```

`report.table` 按排名返回模型 × 指标表，列为 MAE、MSE、RMSE、MAPE、
sMAPE、Theil U1、有效配对数 `n` 和 `rank`。`report.evaluations` 保留每个
模型的完整 `OOSResult`，可以继续读取预测值、区间和日期元数据。

真实值、各模型预测值和误差可以直接生成一张逐期对比表：

```python
forecast_comparison = report.forecast_table()
```

列顺序为 `Actual`，随后按模型传入顺序排列 `<model> forecast` 和
`<model> error`。误差固定定义为 `forecast - actual`。预测区间默认不进入
表格；确实需要上下界时显式使用：

```python
forecast_comparison = report.forecast_table(include_intervals=True)
```

真实值、全部模型预测值和已有预测区间也可以直接绘制：

```python
fig, ax = report.plot_forecasts(
    title="Validation forecasts",
    xtitle="Validation month",
    ytitle="Housing starts",
    freq="month",
    grid=True,
    note="Models use one shared estimation and validation split.",
)
```

绘图复用 `TsPlots.plot_series()` 的样式和日期轴，真实值与所有模型预测值
保持在同一纵轴。只有实际提供上下界的模型才会显示区间；可以通过
`show_intervals=False` 隐藏。区间标签使用 OOS 评价时的 `alpha`，例如
`alpha=0.10` 显示为 90% interval。

以上接口遍历 `report.evaluations`，不限定模型类型或模型数量。对于 VAR、
VECM、SVAR 等多变量结果，必须按名称或零基位置选择一个变量：

```python
prices_table = report.forecast_table(series="prices")
fig, ax = report.plot_forecasts(series=1)
```

批量比较要求每个模型的实际值和点预测在整个验证期内均为有限值。
任何模型存在缺失或无限预测都会直接报错，不能通过少算困难观测获得更优
排名。模型之间还必须具有相同预测目标、估计期、验证期和逐元素相同的
实际观测值。

已有 OOS 或 backtest 结果仍可单独比较一个指定指标：

```python
comparison = compare_forecasts(
    {
        "AR(1)": ar1_evaluation,
        "AR(2)": ar2_evaluation,
    },
    metric="rmse",
)
print(comparison.ranking)
```

比较前强制检查：

- 使用相同评估方法；
- 模型名称必须是字符串且不能因隐式转换发生覆盖；
- `target` 相同；
- 估计期与验证期的索引和日期元数据相同；
- 实际观测值逐元素完全相同。

这可以阻止把均值预测与波动率预测，或不同留出期的结果放进同一排名。

## GARCH 评价目标

GARCH 输出条件波动率，但真实条件波动率不可直接观测。OOS 和 backtest
统一使用当前训练窗口均值中心化后的绝对收益：

```python
abs(y_future - mean(y_train))
```

结果明确标记：

```python
target == "absolute_demeaned_return_proxy"
```

该代理不被表述为已观测的真实波动率。GARCH/AutoGARCH 设置 `exog` 时，
如果没有明确提供未来或样本前外生变量，OOS、backtest 和 backcast 会抛出
`NotImplementedError`。

## 运行测试

```bash
python -m pytest Ts/TsMetrics/tests -p no:cacheprovider -q
```

## SARIMAX 的 OOS 与历史回测

对带普通外生变量的 SARIMAX/SARIMAX，OOS 和 backtest 会在每个预测起点：

- 只使用该起点之前的 `y`、外生变量和日期重新估计模型；
- 将目标窗口对应的未来外生变量传给预测，不把未来 `y` 传入拟合；
- 在滚动窗口中同步切片训练期 `y`、外生变量和日期；
- 根据保留的 `EventSpec` 在训练期和预测期日历上重新生成事件列。

这种评价方式假设目标窗口的外生变量在预测起点已经可知。若它们实际上也需
预测，应先建立相应的信息集或情境，不能把事后实现值当作实时已知信息。

缺少或错位的未来外生变量不会使用回退值。`on_error="raise"` 会立即抛出
异常；backtest 使用 `on_error="record"` 时，该预测起点保留为全 `NaN`，
并在 `failures` 中记录缺失日期和错误信息。
