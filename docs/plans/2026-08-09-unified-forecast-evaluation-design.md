# 统一预测泛化评估接口设计

## 目标

Ts 只保留一个公共预测评估入口 `evaluate_forecasts()`。固定留出、扩展窗口、
固定长度滚动窗口、非重叠历史区块、逐期在线评估和带间隔评估都由显式的
时间切分方案对象表达。模型克隆、无泄漏重估、预测、指标计算、模型排名和
绘图继续复用现有 TsMetrics 与 TsPlots 能力，不建立平行实现。

本次是明确的破坏性迁移，不保留旧名称、弃用别名或兼容回退。

## 删除的公共契约

- `BaseModel.oos()` 和 `BaseModel.backtest()`；
- `TsMetrics.oos()`、`TsMetrics.backtest()`；
- `evaluate_models_oos()` 和 `compare_forecasts()`；
- `OOSResult`、`BacktestResult`、`ComparisonResult`、
  `OOSComparisonResult` 及其顶层导出。

历史 `docs/plans/` 文件是当时设计记录，不回写或伪造其历史内容。

## 唯一公共入口

```python
report = evaluate_forecasts(
    models,
    *,
    scheme,
    rank_by="rmse",
    alpha=0.05,
    fit_kwargs=None,
    on_error="raise",
    future_exog=None,
)
```

`models` 必须是非空的字符串名称到未拟合 Ts 估计器的映射。所有模型必须
共享相同目标值、观测日历和评估切分。评估前先验证全部模型、方案、排名指标
和拟合参数，避免批处理中途才发现结构错误。

`fit_kwargs` 是传给每次隔离克隆 `fit()` 的共同关键字参数。它解决当前
backtest 无法选择优化器的问题，同时避免为不同模型再引入包装类。本次不增加
per-model 拟合配置；异构模型若不能接受同一显式参数，应使用各自默认拟合行为。

## 时间切分方案

### `Holdout`

```python
Holdout(train=(train_start, train_end), test=(test_start, test_end))
```

训练期和测试期均为闭区间，可使用位置或日期边界。测试期必须严格位于训练期
之后；中间允许存在未评分间隔。一个固定留出产生一个预测切分，统一结果张量
的第一维长度为一。

### `RollingOrigin`

```python
RollingOrigin(
    initial_window,
    horizon=1,
    step=1,
    window="expanding",
    window_size=None,
    gap=0,
)
```

- `window="expanding"`：训练起点固定，训练终点随预测起点前移；
- `window="rolling"`：训练长度固定为 `window_size`，默认等于
  `initial_window`；
- `gap`：训练终点与第一个预测目标之间保留的观测数量；
- 只生成具有完整 `horizon` 的切分，不保留尾部不完整窗口。

该对象已经能表达：

- expanding rolling-origin；
- fixed rolling window；
- `horizon=1, step=1` 的逐期 prequential 历史评估；
- `step=horizon` 的非重叠测试区块；
- `gap>0` 的发布滞后或 embargo 评估。

不新增随机 K 折时间序列交叉验证，因为它会破坏信息时间顺序。不新增
`Prequential` 或 `BlockedCV` 别名，因为它们没有独立于
`RollingOrigin` 的契约。

## 内部切分契约

两种方案都解析为不可变的内部 `ForecastSplit`：

- `split`；
- 训练位置数组；
- 目标位置数组；
- 训练起止标签；
- 预测起止标签；
- `gap`、`horizon` 和窗口类型。

评估引擎只消费 `ForecastSplit`，不根据方案类型分支拟合和预测逻辑。

## 统一执行流程

1. 验证模型映射、名称、评估协议、共同目标和共同日历；
2. 验证 `rank_by`、`alpha`、`on_error`、`fit_kwargs` 和外生变量口径；
3. 由方案一次性生成所有切分；
4. 对每个模型、每个切分调用现有 `_clone_for_evaluation()`；
5. 只传训练窗口的目标、外生变量和日期；
6. 传递共同 `fit_kwargs` 并拟合隔离克隆；
7. 使用切分对应的未来上下文预测；
8. 通过 `_evaluation_actual()` 取得正确尺度的评分目标；
9. 原子提交完整切分预测，或按 `on_error` 抛出/记录失败；
10. 在共同有效预测对上构建指标和排名。

原始估计器及其已有 `result_` 始终不变。

## 外生变量口径

带外生变量的历史评估必须显式传入：

```python
future_exog="observed"
```

这表示每个历史预测窗口使用随后实现的实际外生变量路径，是条件预测评估。
若任一模型包含外生变量而调用者未显式确认，评估在任何拟合前报错。结果保存
`uses_observed_future_exog` 元数据，文档和摘要必须说明该结果不等同于未知未来
外生变量下的部署预测。

本次不实现外生变量预测 provider。该能力需要独立设计信息集、辅助模型训练窗
口和多变量路径一致性，不能作为一个未经验证的回调参数附带加入。

## 结果对象

### `ForecastEvaluationResult`

保存一个模型的所有预测切分：

- `mean`、`actual`、`lower`、`upper`；
- `splits`、`failures`；
- `model_type`、`target`、`dates`、`series_names`；
- `uses_observed_future_exog`。

数组形状统一为：

- 单变量 `(n_splits, horizon)`；
- 多变量 `(n_splits, horizon, n_series)`。

### `ForecastComparisonResult`

`evaluate_forecasts()` 始终返回比较结果，即使只传入一个模型。公开能力为：

- `results`：模型名到 `ForecastEvaluationResult`；
- `table`、`scores`、`ranking`、`best_model`；
- `predictions`：统一长表；
- `splits`、`failures`；
- `metric_table(by="horizon" | "origin" | "series")`；
- `plot_forecasts()` 和 `plot_metric()`。

`predictions` 至少包含模型、切分、预测起点、目标时点、预测步长、序列、实际
值、预测值、误差、区间和有效性。`splits` 单独保存训练/预测边界，避免在每个
预测行重复切分元数据。

## 指标与公平排名

继续调用现有 `compute_metrics()` 计算 MAE、MSE、RMSE、MAPE、sMAPE、
Theil U1 和有效数量，不复制公式。

总体指标在全部有效“切分 × 预测步长 × 序列”预测对上 pooled 计算。重叠
滚动预测中的同一目标时点在不同 horizon 下属于不同预测任务，因此保留；结果
必须同时提供按 horizon 和 origin 的分组指标，并明确这些误差不相互独立。

`on_error="record"` 保留每个模型的失败切分，但比较排名只在所有模型共同有限
的预测对上计算。总体表增加 `n_total`、`n_common`、`coverage`、`failures` 和
`rank`。共同样本为空时拒绝排名，不能让模型通过遗漏困难窗口获得优势。

## RDL 单步预测前置修复

`_RationalLagSARIMAX.update()` 当前在单期扩展模型上把一维 singleton
`obs_intercept` 沿 `axis=1` 复制，导致 `horizon=1` 抛出 `AxisError`。
应先按 `k_endog` 规范为二维矩阵，再根据 `nobs` 扩展。

修复必须覆盖直接单步预测、统一 Holdout、统一 RollingOrigin、日期索引、普通
外生变量与 RDL 并存，并保留多步预测行为。

## 代码组织

- 新增 `TsMetrics/_schemes.py`：公共方案对象和内部 split 生成；
- 新增 `TsMetrics/_engine.py`：唯一公共入口和统一执行循环；
- 重构现有 `TsMetrics/_results.py`：以两个统一结果类替换四个旧类；
- 重构 `TsMetrics/_aggregation.py`：统一 split/horizon/series 聚合；
- 复用 `TsMetrics/_evaluation.py`：模型协议、拟合参数验证、预测规范化；
- 删除不再使用的 `_oos.py`、`_backtest.py`、`_compare.py`；
- 删除 `BaseModel` 的两个旧便利方法；
- 更新 TsMetrics、根包导出、README、docstring 和所有调用测试。

## 验收标准

- 旧函数和旧结果类无法再导入或调用；
- 固定留出和所有 rolling-origin 形式只通过 `evaluate_forecasts()`；
- 所有模型共享目标、日历、切分和共同评分样本；
- `fit_kwargs={"method": "lbfgs"}` 传到每次重估；
- RDL `horizon=1` 与多步预测都通过；
- 外生变量条件评估必须显式确认；
- 按总体、horizon、origin 和 series 输出一致；
- 失败切分不会造成不同样本上的不公平排名；
- 原始模型不被修改；
- 公共 README、docstring、Quick Start 和测试无旧 API 残留；
- 定向测试、全仓 pytest、Ruff、compileall 和 `git diff --check` 全部通过。
