# TsUtils 赫斯特指数

## 已确定的设计

- 后端公共接口属于 `TsUtils`，HTFA 只负责选择变量、调用和展示。
- 使用经典重标极差（R/S）估计：对多个 2 的幂次分块计算平均 R/S，再对
  `log(R/S)` 与 `log(块长度)` 做最小二乘斜率估计。
- 默认删除缺失值，不插值；也提供 `missing="raise"` 供严格调用方使用。
- 至少需要 20 个有效观测；常数序列和无有效分块序列明确报错。
- `hurst_exponent()` 返回浮点数；解释只作为描述性启发，不视为统计检验结论。

## 复用检查

`TsUtils._summary._as_numeric_series` 已统一处理数组、Series、DataFrame 列选择、
数值类型、布尔/复数和无穷值；`TsUtils._validation.validate_choice` 负责缺失值
策略校验。新功能复用这两个接口，不复制已有输入验证。

## 验证

重点覆盖独立 R/S 参考值、可重复性、持续序列与白噪声的相对关系、缺失值策略、
DataFrame 选择、无效输入和顶层导出。实现完成后运行 `TsUtils/tests` 与公共
docstring 契约测试。
