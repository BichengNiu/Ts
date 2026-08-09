# SARIMAX 拟合默认值调整设计

## 目标

将 `SARIMAX.fit()` 的默认拟合控制统一调整为：

```python
maxiter=500
cov_type="oim"
require_convergence=True
```

该默认契约同时适用于普通 SARIMAX、带 `distributed_lags` 的 RDL，以及
`AutoSARIMAX` 构造并拟合的全部候选模型。调用者显式传入的参数继续覆盖默认值；
默认优化器保持为 `"bfgs"`。

## 复用与边界

RDL 已通过 `SARIMAX(..., distributed_lags=...).fit()` 使用同一估计入口，
`AutoSARIMAX` 也通过该入口拟合候选模型。因此只修改现有 `SARIMAX.fit()`，
不新增 RDL 估计器、包装函数或配置对象，也不改变 `RationalLagSpec` 的职责。

## 公共行为

- `fit()` 的公开签名直接显示 `maxiter=500`、`cov_type="oim"` 和
  `require_convergence=True`。
- 普通 SARIMAX 和 RDL 默认都使用 observed-information covariance。
- 任一模型在默认 500 次迭代内未报告收敛时，沿用现有错误路径抛出
  `RuntimeError`。
- 显式指定 `maxiter`、`cov_type` 或 `require_convergence=False` 时，沿用现有
  参数验证和转发逻辑。
- `AutoSARIMAX` 不新增重复的拟合参数；其所有候选通过现有无参数 `.fit()`
  调用继承新默认契约，失败候选继续由现有搜索错误处理逻辑处置。

## 测试与文档

测试覆盖普通 SARIMAX 的三个新默认值、RDL 后端的三个新默认值、默认拒绝
未收敛结果、显式覆盖兼容性，以及 `AutoSARIMAX` 候选继承新默认值。同步更新
`SARIMAX.fit()` docstring、`TsModels/README.md` 中的签名和说明。

实现和验证必须保留工作区中已有的 `log=True` 相关未提交改动，只精确暂存本任务
新增或修改的行。
