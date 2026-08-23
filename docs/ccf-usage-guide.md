# Ts 包 CCF 使用指南

本文档说明当前 Ts 仓库中 CCF（cross-correlation function，交叉相关函数）
相关功能的实际定义、调用方式、结果解释和常见限制。

## 1. 先明确：当前包提供的是哪一种 CCF

当前包没有面向任意两条原始序列的通用 `ccf(x, y)` 公共统计接口。
公开的 CCF 统计功能是 **传递函数模型的残差 CCF 诊断**：

\[
r_k = \operatorname{corr}(a_t, e_{t-k}), \quad k=0,1,\ldots,K
\]

其中：

- `a_t` 是最终传递函数/RDL 模型的输出残差；
- `e_t` 是输入单变量 ARIMA 预白化模型的新息；
- 正滞后 `k` 表示当前输出残差与 `k` 期以前输入新息的相关；
- `K` 是用户指定的最大滞后。

因此，CCF 在 Ts 中主要回答的是：

> 输入经过预白化后，最终模型残差中是否还保留了系统性的输入—输出动态关系？

它是模型充分性诊断，不是结构性因果检验。

相关实现和公共导出位于：

- [`TsTests/_residual_ccf.py`](../TsTests/_residual_ccf.py)
- [`TsTests/__init__.py`](../TsTests/__init__.py)
- [`TsModels/_sarimax.py`](../TsModels/_sarimax.py)
- [`TsPlots/acf_plot.py`](../TsPlots/acf_plot.py)

## 2. 三层公共接口

| 层次 | 公共接口 | 职责 |
|---|---|---|
| 统计核心 | `TsTests.ResidualCCFTest` | 计算逐阶残差 CCF 和联合 Box–Jenkins `S*` 检验 |
| 统计结果 | `ResidualCCFInputResult`、`ResidualCCFTestResult` | 保存逐输入结果、显著滞后、检验表和摘要 |
| 模型集成 | `SARIMAXResult.residual_ccf_test()` | 从已拟合 RDL 模型和显式输入预白化模型构造残差 CCF |
| 自动模型集成 | `AutoModelResult.residual_ccf_test()` | 转交给自动模型的 `best_result` |
| 绘图 | `TsPlots.plot_correlogram()` | 绘制调用方已经计算好的相关系数和置信带 |

最常用的入口是：

```python
from Ts.TsTests import ResidualCCFTest

from Ts.TsModels import SARIMAXResult

from Ts.TsPlots import plot_correlogram
```

`SARIMAXResult` 和 `plot_correlogram` 不是必需的；直接使用
`ResidualCCFTest` 时只需要 `TsTests`。

## 3. 直接使用 `ResidualCCFTest`

### 3.1 最小示例

```python
import numpy as np

from Ts.TsTests import ResidualCCFTest

rng = np.random.default_rng(42)
output_residuals = rng.normal(size=200)
input_innovations = rng.normal(size=200)

test = ResidualCCFTest(
    output_residuals,
    input_innovations,
    lags=12,
)
result = test.fit()

print(result.tests)
print(result.get("x").summary())
fig, ax = result.plot_test()
```

一维数组输入默认命名为 `"x"`。也可以使用 `Series` 显式命名：

```python
import pandas as pd

input_innovations = pd.Series(input_innovations, name="price")
result = ResidualCCFTest(
    output_residuals,
    input_innovations,
    lags=12,
).fit()

price_result = result.get("price")
```

### 3.2 多个输入

字典和 `DataFrame` 的列名会作为输入名：

```python
import pandas as pd

input_innovations = pd.DataFrame(
    {
        "price": price_innovations,
        "income": income_innovations,
    }
)

result = ResidualCCFTest(
    output_residuals,
    input_innovations,
    lags=12,
    transfer_params={"price": 2, "income": 1},
).fit()

print(result.input_names)
print(result.tests)
fig, axes = result.plot_test()
```

二维 NumPy 数组必须提供 `input_names`：

```python
result = ResidualCCFTest(
    output_residuals,
    input_matrix,
    lags=12,
    input_names=["price", "income"],
).fit()
```

输入格式规则如下：

| `input_residuals` 类型 | `input_names` 规则 |
|---|---|
| 一维数组 | 可省略；默认名称为 `x` |
| `Series` | 可省略；优先使用 `Series.name`，没有名称时使用 `x` |
| 二维数组 | 必须提供一个与列数相同的名称序列 |
| `DataFrame` | 使用列名；不能再提供 `input_names` |
| mapping | 使用键名；不能再提供 `input_names` |

所有输入必须是一维、数值、有限且不能是常数序列。当前实现不把缺失值
静默删除；`NaN` 和无穷值会被拒绝。

## 4. 参数和样本对齐规则

构造函数为：

```python
ResidualCCFTest(
    output_residuals,
    input_residuals,
    lags,
    *,
    input_names=None,
    transfer_params=0,
    alpha=0.05,
)
```

### `lags`

`lags` 必须是正整数，统计系数包含 `0` 到 `lags`，因此总共有
`lags + 1` 个 CCF 系数。

例如：

```python
ResidualCCFTest(y_resid, x_innov, lags=6)
```

会计算 `lag=0, 1, 2, 3, 4, 5, 6`。

有效对齐样本数必须大于最大滞后。输出残差和输入新息长度不同时，当前实现
按共同样本末端右对齐：

```text
output: [......................a1 a2 a3 a4 a5]
input :             [e1 e2 e3 e4 e5]
                       └── 共同样本末端 ──┘
```

对每个输入分别使用：

```python
nobs = min(len(output_residuals), len(input_innovations))
```

这适合处理输入 ARIMA 模型因初始化或 burn-in 而产生较短新息序列的情况。
多输入结果中的 `result.nobs` 是各输入有效样本数的最小值；每个
`result.get(name).nobs` 才是该输入自己的有效样本数。

### `transfer_params`

`transfer_params` 是用于联合检验自由度修正的传递函数参数数目：

```python
# 所有输入使用同一个 m
transfer_params=2

# 每个输入单独指定 m
transfer_params={"price": 2, "income": 1}
```

它不包括：

- 输入预白化模型的 ARIMA 参数；
- 输出扰动模型的 ARIMA 参数；
- 固定 delay；
- 稀疏规格中固定为零的系数。

每个输入必须满足：

\[
df = K + 1 - m > 0
\]

如果自由度不为正，构造检验对象时会直接报错。

### `alpha`

`alpha` 必须位于 `(0, 1)`，默认值为 `0.05`。它同时用于：

- 逐阶置信带；
- 联合 `S*` 检验的拒绝判断 `pvalue < alpha`。

当前实现没有额外的多重比较校正，因此 `significant_lags` 是逐阶置信带
意义下的显著滞后，不应机械解释为经过多重检验控制后的结论。

## 5. 结果对象怎么读

### 5.1 汇总结果：`ResidualCCFTestResult`

```python
result = ResidualCCFTest(...).fit()

result.input_names  # 输入名称元组
result.tests        # 每个输入一行的联合检验表
result.summary()    # 所有输入的文本摘要
result.get("price") # 取出某一个输入的详细结果
```

`result.tests` 包含以下列：

| 列 | 含义 |
|---|---|
| `input` | 输入名称 |
| `s_statistic` | 联合 Box–Jenkins `S*` 统计量 |
| `df` | 卡方近似自由度 `lags + 1 - transfer_params` |
| `p_value` | 联合检验 p 值 |
| `reject` | 是否拒绝“0 到 K 阶残差 CCF 联合为 0” |
| `nobs` | 该输入的有效对齐样本数 |
| `transfer_params` | 该输入的传递函数参数数目 |

### 5.2 单输入结果：`ResidualCCFInputResult`

```python
item = result.get("price")

item.correlations       # pandas.Series，索引为 lag=0..K
item.standard_errors    # 每阶标准误，当前均为 1/sqrt(nobs)
item.confidence_limits  # DataFrame，包含 lower / upper
item.statistic          # S* 统计量
item.pvalue             # 联合检验 p 值
item.df                 # 联合检验自由度
item.nobs               # 有效样本数
item.reject             # item.pvalue < item.alpha
item.significant_lags   # 超出逐阶置信带的滞后元组
item.summary()          # 文本摘要
```

例如查看每阶结果：

```python
item = result.get("price")
table = item.correlations.rename("ccf").to_frame()
table["lower"] = item.confidence_limits["lower"]
table["upper"] = item.confidence_limits["upper"]
print(table)
```

### 5.3 绘图

统计结果自带绘图方法：

```python
# 单输入或全部输入
fig, axes = result.plot_test()

# 只绘制某一个输入
fig, ax = result.plot_test(inputs="price")

# 选择多个输入
fig, axes = result.plot_test(inputs=["price", "income"])
```

`ResidualCCFInputResult.plot_test()` 也可以直接绘制单个输入：

```python
fig, ax = result.get("price").plot_test()
```

底层 `TsPlots.plot_correlogram()` 只负责绘图，不计算相关系数：

```python
from Ts.TsPlots import plot_correlogram

fig, ax = plot_correlogram(
    item.correlations,
    confidence_band=item.confidence_limits["upper"],
    title="Price input residual CCF",
    ytitle="Residual CCF",
)
```

传入 `Series` 或一维数组返回单轴；传入 `DataFrame` 或二维数组时按列生成
分面。`confidence_band` 表示置信带的半宽，必须是非负值。

## 6. RDL/SARIMAX 集成用法

在 RDL 模型场景下，不建议手工从最终模型对象拼接内部数组；应使用
`SARIMAXResult.residual_ccf_test()`，让模型接口负责：

- 检查当前结果是否为已收敛 RDL 模型；
- 检查输入模型是否覆盖所选 RDL 输入；
- 检查输入数据和日期索引是否完全一致；
- 取得输入模型新息；
- 根据每个 `RationalLagSpec` 的活动参数数量计算自由度。

### 6.1 单输入示例

```python
import numpy as np
import pandas as pd

from Ts.TsModels import RationalLagSpec, SARIMAX

rng = np.random.default_rng(42)
x = pd.Series(rng.normal(size=180), name="x")
y = 0.8 * x.to_numpy() + rng.normal(scale=0.3, size=180)

# 最终模型：x 作为 RDL 输入
fitted = SARIMAX(
    y,
    exog=x,
    trend="n",
    distributed_lags={"x": RationalLagSpec()},
).fit(method="bfgs", maxiter=300, require_convergence=True)

# 输入预白化模型：必须拟合到同一条 x，且不能再带 exog / RDL
input_model = SARIMAX(
    x,
    trend="n",
).fit()

diagnostic = fitted.residual_ccf_test(
    {"x": input_model},
    lags=6,
)

print(diagnostic.tests)
print(diagnostic.get("x").significant_lags)
fig, ax = diagnostic.plot_test(inputs="x")
```

`RationalLagSpec()` 默认包含一个活动 numerator 参数，因此在 `lags=6`
时通常对应：

```text
transfer_params = 1
df = 6 + 1 - 1 = 6
```

### 6.2 多输入和子集

```python
models = {
    "price": price_input_model,
    "income": income_input_model,
}

# 检验全部 RDL 输入
all_result = fitted.residual_ccf_test(models, lags=8)

# 只检验一个输入
price_result = fitted.residual_ccf_test(
    models,
    lags=8,
    inputs="price",
)
```

`inputs` 可以是一个名称或名称序列。`input_models` 必须是以 RDL 输入名
为键的 mapping。

### 6.3 `AutoSARIMAX`

如果自动模型的 `best_result` 是支持该诊断的 RDL `SARIMAXResult`，可以直接
调用：

```python
diagnostic = auto_result.residual_ccf_test(
    {"x": input_model},
    lags=6,
)
```

自动模型不会替用户自动生成输入预白化模型；输入模型仍然必须显式传入。

## 7. 统计计算定义

当前实现使用固定 `n` 分母的有偏样本 CCF。对每个滞后，近似标准误为：

\[
SE(r_k) = \frac{1}{\sqrt n}
\]

双侧置信带为：

\[
\left[-\frac{z_{1-\alpha/2}}{\sqrt n},
\frac{z_{1-\alpha/2}}{\sqrt n}\right]
\]

联合 Box–Jenkins 检验为：

\[
S^*=n^2\sum_{k=0}^{K}\frac{r_k^2}{n-k}
\]

其 p 值使用自由度为 `K + 1 - m` 的卡方近似。

原假设是：

> 该输入对应的 0 到 K 阶残差交叉相关联合为 0。

因此：

- `reject=True` 表示检测到剩余的输入—输出交叉相关证据；
- `reject=False` 只能表示当前样本和滞后范围内无法拒绝原假设；
- `reject=False` 不证明模型已经真实正确；
- 某一个 `significant_lags` 只表示逐阶峰值超过置信带。

## 8. 推荐的诊断流程

```text
拟合输入预白化模型
        │
        ▼
拟合最终 RDL/传递函数模型
        │
        ▼
运行 residual_ccf_test()
        │
        ├── 查看 result.tests：联合 S* 结论
        ├── 查看 get(name).significant_lags：定位滞后峰值
        └── 查看 plot_test()：检查峰值形状和方向
```

建议按以下顺序解释结果：

1. 先看每个输入的联合 `p_value` 和 `reject`；
2. 再看 `significant_lags`，识别剩余动态出现在哪些滞后；
3. 对照 `correlations` 的符号和大小，而不是只看是否越过置信带；
4. 检查输入预白化模型是否合理；
5. 检查 RDL 的 numerator、denominator、delay 和输入集合；
6. 如果存在反馈、输入之间高度相关或遗漏输入，不能把峰值简单归因于
   某一个被遗漏的传递函数滞后。

显著峰值是模型修正的线索，不是“自动增加同阶 numerator 参数”的命令。

## 9. 识别阶段和诊断阶段不能混用

传递函数建模通常包含两个不同阶段：

### 识别阶段

把输入和输出都通过输入预白化滤波器处理，用于观察潜在的传递动态。

### 诊断阶段

在最终模型已经拟合后，用最终输出残差和输入预白化模型的新息计算残差 CCF。
`ResidualCCFTest` 属于这个阶段。

不要把诊断阶段的 `output_residuals` 替换成原始输出，也不要把输入预白化
模型的原始输入替换成未经滤波的输入序列。

## 10. 常见错误

| 现象 | 原因和处理 |
|---|---|
| `lags must be >= 1` | `lags` 必须是正整数 |
| `Need at least ... observations` | 对齐样本数不足以覆盖 `0..K` 阶 |
| `degrees of freedom` | `lags + 1 - transfer_params` 不为正；应增加样本/滞后或修正参数计数 |
| `input_names is required for two-dimensional array input` | 二维数组缺少列名 |
| `input_names must not be provided with a mapping` | mapping/DataFrame 已经提供输入名，不能重复传入 |
| `must contain only finite values` | 输入含 `NaN` 或无穷值；先按项目的数据缺失策略处理 |
| `requires a fitted rational distributed-lag model` | 对普通 SARIMAX 结果调用了 RDL 专用接口 |
| `missing fitted input model` | `input_models` 没有覆盖被检验的 RDL 输入 |
| `exact historical input` | 输入预白化模型不是拟合到最终 RDL 使用的同一条输入序列 |
| `calendar` | 两个模型的日期索引不完全一致 |
| `input-only` | 预白化模型包含 `exog`、事件或 RDL |
| `untransformed input scale` | 预白化模型使用了 `log=True`，与 RDL 输入尺度不一致 |

## 11. 结果解释边界

残差 CCF 显著可能来自多种原因：

- RDL numerator 或 denominator 动态设定不足；
- 输入预白化模型没有充分去除输入自身动态；
- 输入变量遗漏或输入之间存在相关；
- 输出模型中存在反馈或其他未建模结构；
- 样本对齐、日期、变换尺度或模型收敛存在问题。

所以应将 CCF 作为传递函数模型的诊断证据，与模型设定、残差白噪声检验、
反馈检验和经济/业务机制一起判断。它本身不识别结构性因果关系，也不替代
模型稳定性、参数显著性或样本外预测检验。

## 12. 可复现验证

在仓库根目录运行：

```powershell
python -m pytest TsTests/tests/test_residual_ccf.py -q
python -m pytest TsModels/tests/test_distributed_lag.py -k residual_ccf -q
python -m pytest TsModels/tests/test_auto.py -k residual_ccf -q
python -m pytest TsPlots/tests/test_plots.py -k precomputed -q
```

对应的回归测试覆盖：

- Box–Jenkins 公式和滞后方向；
- 不同长度序列的右对齐；
- 标准误、置信带和显著滞后；
- 多输入结果和输入子集；
- RDL 参数计数与稀疏滞后；
- 输入数据、日期、收敛状态和变换尺度校验；
- 统一 CCF 绘图和分面行为。
