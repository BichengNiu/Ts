# SARIMAX 与 ARCH/GARCH 参数相关矩阵

## 目标与范围

为 `SARIMAXResult` 和 `GARCHResult` 提供带参数名的估计参数相关矩阵及统一热图。
覆盖 SARIMA/ARIMA/SARIMAX、外生变量、事件、RDL、ARCH、GARCH、GJR-GARCH、
EGARCH、GARCH-M 和 IGARCH。VAR、SVAR、VECM 本期不接入协方差。

参数相关矩阵用于检查联合估计中的强相关、冗余和辨识脆弱性，不替代已有的
AR/MA 根、VAR 特征根、GARCH 持久性或回归参数随时间稳定性检验。

## 现有能力检索与复用

- `BaseModelResult` 已统一参数、标准误、p 值和诊断方法，因此相关矩阵的转换、
  参数筛选与模型绘图入口放在该基类，不在 SARIMAX/GARCH 中复制。
- SARIMAX 的 statsmodels 结果已保留 `param_names` 与 `cov_params()`；RDL 的
  delta-method 推断也已使用同一完整协方差，直接复用其顺序和数值。
- ARCH/GARCH 的 arch 结果已保留带索引的 `param_cov`，`test_persistence()` 已
  使用该矩阵，直接复用。
- IGARCH 已在 `_igarch_std_errors()` 中计算数值 Hessian 逆矩阵，但此前只保留
  对角标准误。本功能保留完整自由参数协方差，并通过公开参数约束 Jacobian
  转换，不另写第二套 Hessian。
- `TsPlots.style` 已统一字体、字号和坐标轴样式；新热图复用这些契约。现有
  `TsPlots` 没有矩阵绘图公共 API，因此新增通用 `plot_correlation_matrix()`，
  模型结果只负责委托。

## 公共契约

- `result.parameter_correlation(parameters=None)` 返回 `pandas.DataFrame`；行列
  使用相同的估计参数名，支持显式子集和顺序。
- `result.plot_parameter_correlation(...)` 委托
  `TsPlots.plot_correlation_matrix()`。
- 固定为零的 SARIMAX 稀疏滞后不进入矩阵，因为它们不是估计参数。
- IGARCH 最后一个 beta 是精确约束推导参数，因此完整相关矩阵是奇异矩阵；
  这是正确的约束结果，不作为估计失败处理。
- 未接入的模型抛出明确的 `NotImplementedError`。

## 测试与文档

测试覆盖共享矩阵换算、参数筛选、输入验证、热图契约、SARIMAX/标准 GARCH
与底层协方差一致性、IGARCH Jacobian 约束、Auto 模型委托，以及未接入模型的
兼容行为。同步更新 `TsModels/README.md`、`TsPlots/README.md` 和公共 docstring。
