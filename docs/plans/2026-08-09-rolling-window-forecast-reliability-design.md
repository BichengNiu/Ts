# 滚动窗口预测可靠性设计

## 目标

在现有无信息泄漏的滚动起点回测基础上，直接报告模型在连续重叠预测窗口上的动态可靠性。用户指定最小训练样本和 `N` 期预测窗口后，模型从第一个可用预测起点开始，每次向前滚动一期、重新拟合并预测未来完整 `N` 期，直到最后一个窗口结束于最新一期样本。每个窗口分别报告 RMSE、MAPE 等现有规范指标。

## 现有能力与复用结论

已检索 `TsMetrics` 的公共 API、README、回测实现、结果容器、聚合函数、评价协议和测试，以及 `TsModels.BaseModel.backtest()` 的便利入口与模型评价测试。

现有 `backtest()` 已经负责：

- 从 `initial_window` 指定的最小训练样本开始；
- 在每个预测起点克隆并重新拟合模型，避免修改调用者和引入未来目标信息；
- 通过 `horizon` 指定每个起点的完整预测期数；
- 通过 `step` 控制相邻预测起点的距离；
- 支持扩展训练窗和固定长度滚动训练窗；
- 保存预测起点、逐预测期的预测值、真实值、区间和失败信息；
- 复用 `compute_metrics()` 计算统一的 MAE、MSE、RMSE、MAPE、sMAPE、Theil U1 和有效配对数。

因此本功能不新增滚动评估引擎，也不新增与 `backtest()` 重复的公共函数。最终方案扩展现有 `BacktestResult`，把已经生成的预测结果按预测窗口聚合为动态指标表。

## 用户契约

推荐调用保持为现有公共接口：

```python
result = model.backtest(
    initial_window=20,
    horizon=3,
    step=1,
    window="expanding",
)

dynamic = result.metrics_by_window
```

参数在本功能中的含义：

- `initial_window`：第一次拟合时可见的最小训练样本量；
- `horizon=N`：每个预测起点向前预测并评价的完整 `N` 期窗口；
- `step=1`：相邻预测窗口重叠，并且每次只向前移动一期；
- `window="expanding"`：第一次使用最小训练样本，之后每期加入最新可见观测并重新拟合；
- 现有 `window="rolling"` 与 `window_size` 继续可用，但不是本需求的推荐默认口径。

只评价完整的 `N` 期窗口。对于长度为 `T` 的样本，`step=1` 时窗口数量为：

```text
T - initial_window - N + 1
```

最后一个窗口的结束位置为 `T - 1`，即最新一期样本。

## 结果结构

`BacktestResult` 新增只读属性 `metrics_by_window`。单变量结果返回按时间排列的 `pandas.DataFrame`：

| window_start | window_end | mae | mse | rmse | mape | smape | theil_u1 | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-09 | 2022-11 | ... | ... | ... | ... | ... | ... | 3 |
| 2022-10 | 2022-12 | ... | ... | ... | ... | ... | ... | 3 |

日期模型使用真实日期作为 `window_start` 和 `window_end`；无日期模型使用零基观测位置。为了支持这一输出，`BacktestResult` 保留可选的日期和变量名元数据。新增元数据字段提供默认值，以保持已有手工构造方式兼容。

多变量模型按“窗口 × 变量”报告指标，不把不同量纲的序列混合计算 RMSE。变量名称优先使用模型已有 `data_names`；没有名称时使用稳定的位置标签。

现有 `metrics`、`metrics_by_horizon`、`metrics_by_series`、预测数组、失败信息和构造方式保持不变。

## 数据流与职责边界

```text
原始模型
  -> backtest() 在每个预测起点截取当时可见的训练样本
  -> 通过现有评价协议克隆并重新拟合模型
  -> 预测未来完整 N 期
  -> BacktestResult 保存预测、真实值、区间和元数据
  -> metrics_by_window 按每个预测窗口调用 compute_metrics()
  -> 返回动态可靠性指标表
```

职责保持为：

- `TsModels` 继续负责模型估计和预测，只通过 `BaseModel.backtest()` 委托评价；
- `TsMetrics/_backtest.py` 继续负责回测编排，只补充向结果对象传递已有日期和变量名；
- `TsMetrics/_aggregation.py` 负责按窗口聚合，并复用 `compute_metrics()`；
- `TsMetrics/_results.py` 负责结果验证、元数据保存和动态指标表接口；
- 不在 `TsPlots`、`TsTests`、`TsUtils` 或具体模型中复制指标计算。

## 指标与异常语义

- 每行只使用该预测起点未来完整 `N` 期的预测值和真实值；
- 相邻窗口可以包含同一目标日期，但这些预测来自不同起点或预测步长；
- 指标严格复用 `compute_metrics()`，不复制公式；
- 部分预测值或真实值非有限时，沿用 `nan_policy="omit"` 的有效配对语义，并通过 `n` 报告实际样本数；
- 窗口内实际值全部为零时，MAPE 沿用现有契约返回 `NaN`；
- `on_error="raise"` 时，窗口拟合或预测失败立即抛出原异常；
- `on_error="record"` 时，失败窗口继续保留，动态指标为 `NaN`、`n=0`，错误详情仍写入 `failures`；
- 不生成不足 `N` 期的尾部窗口，也不把短窗口与完整窗口混在同一动态序列中。

## 实现边界

计划修改：

- `TsMetrics/_aggregation.py`：增加按预测窗口聚合现有指标的共享函数；
- `TsMetrics/_results.py`：扩展 `BacktestResult` 元数据和 `metrics_by_window`；
- `TsMetrics/_backtest.py`：传递日期和多变量名称；
- `TsMetrics/tests/test_evaluation.py` 与 `TsMetrics/tests/test_contracts.py`：覆盖聚合和结果契约；
- `TsModels/tests/test_evaluation.py`：覆盖真实模型便利入口和兼容行为；
- `TsMetrics/README.md`、`TsModels/README.md` 与公共 docstring：增加完整说明和可执行用例。

不计划：

- 新增 `rolling_reliability()` 或其他重复公共入口；
- 新增第二种模型克隆、拟合或滚动切窗实现；
- 修改模型参数估计与预测算法；
- 增加动态指标绘图接口；
- 修改或恢复工作区中已处于删除状态、且与本功能无关的 `chapter6.ipynb`。

## 测试设计

1. 使用可手算的预测误差逐窗口核对全部规范指标。
2. 验证 `horizon=N`、`step=1` 时窗口逐期重叠。
3. 验证窗口数为 `T - initial_window - N + 1`，且最后一个 `window_end` 为最新样本。
4. 验证日期索引保留真实窗口起止日期。
5. 验证无日期模型返回正确的零基位置。
6. 验证多变量结果按变量分别计算，不跨变量聚合。
7. 验证 `on_error="record"` 的失败窗口返回 `NaN` 指标和 `n=0`，并保留失败详情。
8. 覆盖实际值含零、全部为零和缺失配对时的 MAPE 与 `n` 语义。
9. 验证新增可选元数据不破坏已有 `BacktestResult` 构造方式和属性。
10. 至少通过 SARIMAX 和一个多变量模型验证 `model.backtest()` 集成。

## 文档与验收

更新 `TsMetrics/README.md`、`TsModels/README.md` 和 `BacktestResult` docstring，说明参数含义、重叠窗口机制、返回对象、字段读取方法、多变量行为和限制。当前仓库没有与该工作流对应的 `TsMetrics` 演示 Notebook，因此不创建或修改无关 Notebook；README 和 docstring 提供可执行示例。

最终验证命令：

```powershell
python -m pytest TsMetrics/tests TsModels/tests/test_evaluation.py TsModels/tests/test_evaluation_periods.py -q
python -m pytest -q
python -m ruff check TsMetrics TsModels
python -m compileall -q TsMetrics TsModels
git diff --check
```

验收标准：给定最小训练样本和 `N`，用户一次调用现有 `backtest()` 即可获得从第一个预测窗口到最新一期的动态可靠性指标表；所有窗口均完整、按时间有序、无未来目标信息泄漏，且现有公共行为保持兼容。
