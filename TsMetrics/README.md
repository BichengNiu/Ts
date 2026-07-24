# Ts/TsMetrics

时间序列预测性能指标与无信息泄漏评估工具包。

`TsMetrics` 只负责度量和评估；模型拟合、样本内预测和未来预测由
`TsModels` 负责。评估引擎通过 `BaseModel` 协议调用模型，不导入任何具体
模型类。

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
    OOSResult,
    BacktestResult,
    ComparisonResult,
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

## 真实留出法 OOS

```python
evaluation = oos(model, split=80)
# 等价便利入口：model.oos(split=80)
```

引擎只使用 `model.data[:split]` 克隆并拟合模型，然后预测
`model.data[split:]`。原模型及其 `result_` 不会被修改。

`predict(oos_start=...)` 不存在。固定全样本参数再把尾部标成 OOS 会把
保留期信息带入参数估计，因此不作为性能评估接口。

`OOSResult` 提供：

- `mean`、`actual`、`lower`、`upper`；
- `target_indices` 和 `split`；
- 总体 `metrics` 与逐变量 `metrics_by_series`；
- `target`，用于标明被评价的可观测对象。

## 滚动历史回测

```python
evaluation = backtest(
    model,
    initial_window=80,
    horizon=4,
    step=1,
    window='expanding',
)
# 等价便利入口：model.backtest(...)
```

每个预测起点都只使用此前可见的数据重新拟合。`window='rolling'` 时可用
`window_size` 指定固定训练窗口。结果按预测起点、预测期和变量保存，并
同时提供总体、逐预测期和逐变量指标。

`window_size` 只对 `window='rolling'` 有效；扩展窗口传入该参数会明确
抛出 `ValueError`，不会静默忽略。

## 模型性能比较

```python
comparison = compare_forecasts(
    {
        'AR(1)': ar1_evaluation,
        'AR(2)': ar2_evaluation,
    },
    metric='rmse',
)
print(comparison.ranking)
```

比较前强制检查：

- 使用相同评估方法；
- 模型名称必须是字符串且不能因隐式转换发生覆盖；
- `target` 相同；
- 目标索引相同；
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
target == 'absolute_demeaned_return_proxy'
```

该代理不被表述为已观测的真实波动率。GARCH/AutoGARCH 设置 `exog` 时，
如果没有明确提供未来或样本前外生变量，OOS、backtest 和 backcast 会抛出
`NotImplementedError`。

## 运行测试

```bash
python -m pytest Ts/TsMetrics/tests -p no:cacheprovider -q
```

## ARIMAX 的 OOS 与历史回测

对带普通外生变量的 SARIMA/ARIMAX，OOS 和 backtest 会在每个预测起点：

- 只使用该起点之前的 `y`、外生变量和日期重新估计模型；
- 将目标窗口对应的未来外生变量传给预测，不把未来 `y` 传入拟合；
- 在滚动窗口中同步切片训练期 `y`、外生变量和日期；
- 根据保留的 `EventSpec` 在训练期和预测期日历上重新生成事件列。

这种评价方式假设目标窗口的外生变量在预测起点已经可知。若它们实际上也需
预测，应先建立相应的信息集或情境，不能把事后实现值当作实时已知信息。

缺少或错位的未来外生变量不会使用回退值。`on_error="raise"` 会立即抛出
异常；backtest 使用 `on_error="record"` 时，该预测起点保留为全 `NaN`，
并在 `failures` 中记录缺失日期和错误信息。
