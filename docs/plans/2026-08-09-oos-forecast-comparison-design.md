# OOS 预测对比表与对比图设计

## 背景

`TsMetrics.evaluate_models_oos()` 已经负责在同一估计期和验证期上重新拟合任意数量的模型，并通过 `OOSComparisonResult` 保存各模型的 `OOSResult`、评价指标和排名。当前用户仍需手工提取每个模型的真实值、预测值、误差和区间，再自行拼接 `DataFrame`、调用 `TsPlots.plot_series()` 并逐个添加置信区间。

本功能消除这段重复的 Notebook 胶水代码，但不改变模型估计、预测或评价算法。

## 已有能力与复用结论

- `TsMetrics.evaluate_models_oos()` 已保证模型共享验证期、真实值和预测目标，因此新接口不重复执行一致性判断。
- `TsMetrics.OOSComparisonResult` 已是多模型样本外比较的公共结果容器，适合承载表格和绘图入口。
- `TsMetrics.OOSResult` 已保存预测均值、真实值、区间、位置和日期，但尚未保存区间显著性水平与多变量名称。
- `TsPlots.plot_series()` 已统一颜色、线型、标记、字体、日期刻度、图题和注释；新绘图必须组合该公共函数，不创建平行样式实现。
- VAR、VECM 和 SVAR 估计器已通过 `data_names` 保存变量名称，可以在 OOS 评价时传入结果对象。

## 公共接口

### 预测对比表

在 `OOSComparisonResult` 增加：

```python
report.forecast_table(
    series=None,
    *,
    include_errors=True,
    include_intervals=False,
)
```

单变量结果默认返回以下列，模型顺序与传给 `evaluate_models_oos()` 的映射顺序一致：

```text
Actual
<model 1> forecast
<model 1> error
<model 2> forecast
<model 2> error
...
```

误差定义固定为 `forecast - actual`。`include_intervals=False` 是默认值；启用时，仅为实际包含完整区间的模型加入 `<model> lower` 和 `<model> upper`。

日期感知结果使用 `validation_dates` 作为索引，否则使用 `validation_indices`。已有 `report.table` 继续表示指标排名表，不改变语义。

### 预测对比图

在 `OOSComparisonResult` 增加：

```python
report.plot_forecasts(
    series=None,
    *,
    colors=None,
    title=None,
    xtitle=None,
    ytitle="Value",
    freq=None,
    note=None,
    grid=False,
    show_intervals=True,
    interval_alpha=0.12,
    ax=None,
)
```

绘图按“真实值在前、各模型预测值在后”的顺序调用 `TsPlots.plot_series(..., facet=False, auto_dual_y=False)`。所有序列强制共用一个纵轴。模型有区间且 `show_intervals=True` 时，通过同色 `fill_between()` 添加区间；无区间模型只显示预测线。

默认颜色中真实值使用深灰色，模型颜色从 `TsPlots` 公共调色板依次取得；显式 `colors` 同时决定折线和对应区间颜色。返回值保持 `(fig, ax)`。

## 任意模型与多变量支持

接口只遍历 `report.evaluations`，不识别或硬编码 RDL、ARIMA、SARIMAX、VAR 等具体模型类型。只要模型符合现有 OOS 评价协议，就能进入同一流程。

- 单变量结果不要求 `series`。
- 多变量结果要求通过整数位置或名称选择一个变量，例如 `series=0` 或 `series="starts"`。
- 未选择多变量序列时抛出明确错误，避免隐式选择第一列。
- 名称来自新增的 `OOSResult.series_names`；数组型向量模型已有的 `y0`、`y1` 等名称同样保留。
- 同一报告中各模型的预测维度和变量名必须一致。

## 元数据扩展

在 `OOSResult` 尾部增加两个具有默认值的字段，保持已有位置参数构造兼容：

```python
series_names: tuple[str, ...] | None = None
alpha: float | None = None
```

`oos()` 将从估计器的 `data_names` 获取多变量名称，并保存经验证的 `alpha`。手工构造、未提供 `alpha` 的结果仍可绘图，其区间图例使用通用的 `interval`；由 `oos()` 生成的结果按 `(1 - alpha) * 100` 正确标记区间水平。

## 错误处理

- `series` 类型必须是整数或非空字符串，布尔值不作为整数接受。
- 单变量结果拒绝无意义的非零/非名称选择。
- 多变量结果缺少 `series`、名称未知或位置越界时给出明确错误。
- 报告中的真实值形状、验证索引、验证日期、变量名或 `alpha` 不一致时拒绝制表或绘图。
- `include_errors`、`include_intervals`、`show_intervals` 必须是布尔值。
- `interval_alpha` 必须位于 `[0, 1]`。
- `colors` 数量不足时沿用 `TsPlots.plot_series()` 的现有校验契约。

## 测试与交付

- 单变量任意数量模型：列顺序、误差方向、日期/位置索引。
- 默认排除区间；显式包含区间；有区间与无区间模型混合。
- 多变量按名称和位置选择；无选择、非法名称和越界位置。
- 绘图复用 `plot_series` 样式、同轴显示、区间颜色与非默认置信水平标签。
- `OOSResult` 旧位置参数构造、`report.table` 和现有 OOS 评价行为保持兼容。
- 更新 `TsMetrics/README.md`、公共 docstring 和 `TsMetrics/demo.ipynb`，并执行相关测试、全量测试、Ruff、编译检查、Notebook 执行和 `git diff --check`。

