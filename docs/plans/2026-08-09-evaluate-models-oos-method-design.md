# `evaluate_models_oos` 优化器参数设计

## 背景

`TsMetrics.evaluate_models_oos()` 当前通过 `oos()` 和共享的
`fit_and_forecast()` 克隆模型，然后无参数调用克隆对象的 `fit()`。因此
`SARIMAX` 在样本外比较中只能采用 `SARIMAX.fit()` 的默认优化器 BFGS，用户
无法在批量评价入口选择 L-BFGS、Powell 等已由 `SARIMAX.fit()` 支持的优化器。

该入口同时接受 SARIMAX 之外的 Ts 估计器。不同模型的 `fit()` 并不共享
优化器参数契约，且 `AutoSARIMAX.method` 表示搜索策略而不是似然优化器，因而
不能按参数名猜测或静默忽略。

## 已检查并复用的现有能力

- `TsMetrics.evaluate_models_oos()`：保留批量比较、共同样本和排名职责。
- `TsMetrics.oos()`：保留单次隔离估计和验证窗口职责。
- `TsMetrics._evaluation.fit_and_forecast()`：继续作为克隆、拟合和预测的唯一
  共享实现，不新增平行拟合路径。
- `SARIMAX.fit(method=...)`：复用现有优化器规范化、支持列表和错误校验，不在
  TsMetrics 重复验证具体优化器名称。

## 公共契约

为 `evaluate_models_oos()` 增加仅限关键字参数 `method=None`：

- `None`：不向 `fit()` 传入 `method`，完整保持现有行为和向后兼容性。
- 字符串：将 `method` 作为拟合参数传给每一个模型克隆的 `fit()`。
- 在开始任何模型拟合前，检查全部模型克隆的 `fit()` 是否明确接受 `method`
  或 `**kwargs`。如有模型不支持，抛出包含模型名称的 `TypeError`，不产生部分
  比较结果，也不静默回退默认优化器。

示例：

```python
report = evaluate_models_oos(
    {"AR(1)": ar1_model, "AR(2)": ar2_model},
    estimation_period=(0, 79),
    validation_period=(80, 99),
    method="lbfgs",
)
```

## 调用链与边界

`evaluate_models_oos()` 将拟合选项传给 `oos()`，`oos()` 再传给
`fit_and_forecast()`；只有共享拟合函数执行 `fit(**fit_kwargs)`。具体优化器的
合法值仍由模型自身负责。

本次只增加用户要求的 `method`，不扩展为任意 `fit_kwargs`，也不改变
`SARIMAX.fit()` 的 BFGS 默认值。这样既满足明确需求，也避免提前扩大所有模型
的通用拟合配置协议。

## 方案比较

1. **严格统一透传（采用）**：契约明确；混合不兼容模型在拟合前失败，避免
   用户误以为所有模型使用了同一优化器。
2. **仅对 SARIMAX 透传并忽略其他模型**：调用方便，但比较报告会混用不同
   拟合策略且不易察觉，因此拒绝。
3. **把优化器保存到模型构造器或克隆配置中**：可绕过评价 API 的拟合参数，
   但重复 `fit()` 已有职责，并需要扩大多个模型的构造契约，因此拒绝。

## 错误处理与验证

- 验证 `method=None` 仍调用无参数 `fit()`。
- 验证显式 `method` 到达克隆模型的 `fit()`，原模型仍不被拟合或修改。
- 验证不支持 `method` 的命名模型在任何拟合发生前报错，错误包含模型名。
- 验证所选优化器实际到达克隆模型的底层拟合调用。
- 更新 `evaluate_models_oos()` docstring 和 `TsMetrics/README.md` 的参数说明与
  示例，并运行聚焦测试、公开文档测试及完整测试集。
