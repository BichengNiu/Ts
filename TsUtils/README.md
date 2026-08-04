# Ts/TsUtils

时间序列预处理与识别诊断工具包。当前提供六类能力：

- STL（Seasonal-Trend decomposition using LOESS）分解；
- 缺失值内插及可审计的填补结果；
- 普通、对数及显式周期的差分；
- 可审计、可复用参数的 Box-Cox 幂变换；
- 时间序列统计摘要与诊断图；
- 用于非季节 ARMA 阶数识别的扩展自相关函数（EACF）。

`TsUtils` 只负责进入模型前的数据处理与识别诊断，不估计预测模型。本版本不提供
X-12/X-13 或其他季节调整接口；STL 只做分解，不等同于官方统计口径的季节调整。

## 公共接口

```python
from Ts.TsUtils import (
    BoxCoxResult,
    EACFResult,
    STL,
    STLResult,
    boxcox,
    difference,
    eacf,
    interpolate_missing,
    InterpolationResult,
)
```

这些符号也可以从 `Ts` 顶层导入。

## 交互式帮助

公共接口的参数、返回值和可执行样例均已写入 docstring。在 IPython/Jupyter
中输入 `?boxcox`（或 `boxcox?`），也可在 Python 中调用 `help(boxcox)`
查看完整帮助。

## EACF 阶数识别

`eacf()` 在拟合模型前生成非季节 ARMA 的扩展自相关表：

```python
from Ts.TsUtils import eacf

result = eacf(series, ar_max=7, ma_max=13)
print(result.summary())

numeric_table = result.values
coded_table = result.symbols
```

结果的行对应 AR 阶数 `p`，列对应 MA 阶数 `q`。`x` 表示该位置的扩展自相关
显著偏离零，`o` 表示其绝对值不超过：

\[
\frac{2}{\sqrt{n-p-q-1}}
\]

从左上角开始出现的 `o` 三角形可用于提出 `(p, q)` 候选，但它是识别启发式，
不自动决定唯一模型；候选仍应通过参数估计、残差诊断和样本外评价复核。

输入接受一维数组、`Series` 和单列 `DataFrame`。多列 `DataFrame` 必须显式指定
`variable`：

```python
result = eacf(dataframe, ar_max=5, ma_max=8, variable="GDP")
```

EACF 不自动差分、删除或插补观测。输入必须是完整、有限、非常数的数值序列；
所需最少样本量还要保证递推中的最高阶 AR 回归可识别；不足时会报告明确的下限。
当前接口只识别非季节 ARMA 阶数，
季节项应结合季节差分、季节相关图和拟合后诊断另行判断。

## 差分

`difference()` 用三个正交参数组合普通差分、对数差分和同比差分：

```python
from Ts.TsUtils import difference

first = difference(series)
second_log = difference(series, order=2, log=True)
monthly_yoy = difference(series, lag=12)
quarterly_yoy_log = difference(frame, log=True, lag=4)
```

公共签名：

```python
difference(data, *, order=1, log=False, lag=1)
```

| 操作 | 参数 |
|---|---|
| 一阶差分 | `order=1, log=False, lag=1` |
| 一阶对数差分 | `order=1, log=True, lag=1` |
| 二阶差分 | `order=2, log=False, lag=1` |
| 二阶对数差分 | `order=2, log=True, lag=1` |
| 同比差分（月度） | `order=1, log=False, lag=12` |
| 同比对数差分（月度） | `order=1, log=True, lag=12` |
| 同比二阶差分（月度） | `order=2, log=False, lag=12` |
| 同比二阶对数差分（月度） | `order=2, log=True, lag=12` |

同比周期不自动猜测，应按数据频率显式设置：月度 `lag=12`、季度
`lag=4`、年度 `lag=1`。`lag` 表示观测位置差，不按日历标签自动对齐。

设同比周期为 \(s\)，同比二阶差分定义为：

\[
(1-L^s)^2x_t=x_t-2x_{t-s}+x_{t-2s}
\]

对数版本先计算自然对数，再应用相同的差分算子。因此，`log=True` 要求全部
非缺失观测严格大于零。

输入只接受数值型 `pandas.Series` 或 `pandas.DataFrame`。DataFrame 按列独立
计算；返回值保持原容器类型、索引、Series 名称、DataFrame 列名和行数，不修改
调用方数据。差分产生的前置缺失值不会被删除，输入中的缺失值按 pandas 的标准
差分规则传播；正负无穷、布尔值、复数和非数值列会被拒绝。

## Box-Cox 变换

`boxcox()` 使用 SciPy 的最大似然方法自动估计幂参数 λ，并始终返回包含转换后
数据和实际 λ 的 `BoxCoxResult`：

```python
from Ts.TsUtils import boxcox

fitted = boxcox(training_series)
transformed = fitted.data
estimated_lmbda = fitted.lmbda

# 将训练集估计的参数复用于后续观测，避免使用未来信息重新估计。
future_transformed = boxcox(future_series, lmbda=estimated_lmbda).data
```

公共签名：

```python
boxcox(data, *, lmbda=None)
```

对于 `Series`，`result.lmbda` 是一个浮点数。对于 `DataFrame`，每列独立转换，
`result.lmbda` 是以原列名为索引的 `Series`。也可以给所有列指定同一个 λ，或按列
传入参数：

```python
automatic = boxcox(frame)
shared = boxcox(frame, lmbda=0.5)
per_column = boxcox(frame, lmbda={"output": 0.0, "prices": 0.25})
```

`lmbda=0` 对应自然对数变换。自动估计要求每列至少有两个非缺失且不完全相同的
观测。缺失值在估计时被忽略，但会保留在输出中的原位置；输入索引、名称、列名和
长度保持不变，调用方数据不会被修改。

Box-Cox 只定义于严格正值。该接口拒绝零、负值和无穷值，不会自动添加平移常数，
因为平移会改变转换结果及 λ 的统计含义。如果业务上确有平移依据，应由调用方先
显式完成并记录该处理。

## STL 分解

```python
import numpy as np
from Ts.TsUtils import STL

time = np.arange(120)
monthly = 10 + 0.05 * time + 2 * np.sin(2 * np.pi * time / 12)

result = STL(monthly, period=12, robust=True).fit()
print(result.summary())
result.plot()
```

`STLResult` 提供：

- `observed`：进入分解的观测序列；
- `trend`：趋势项；
- `seasonal`：季节项；
- `residuals`：残差；
- `weights`：稳健分解权重；
- `fitted_values`：`trend + seasonal`。

默认 `missing="raise"`，遇到 `NaN` 或无穷值立即报错。显式设置
`missing="drop"` 才会删除相应位置，并在分解器的 `dropped_positions`
中记录原始零基位置。删除会改变时间间隔，因此规则时间序列通常应先插值，而不是
在 STL 中删行。

## 缺失值插值

```python
import numpy as np
from Ts.TsUtils import interpolate_missing

data = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
result = interpolate_missing(
    data,
    method="linear",
    max_gap=2,
    edge="keep",
)

filled = result.data
print(result.summary())
```

### 支持的输入

- 一维或二维 NumPy 数组；
- `pandas.Series`；
- `pandas.DataFrame`。

二维数据按列独立插值。返回数据保持原容器类型和逻辑形状；Series 名称、索引以及
DataFrame 索引、列名均会保留。调用方数据不会被原地修改。

### 插值方法

| `method` | 含义 | 约束 |
|---|---|---|
| `"linear"` | 按观测位置线性插值 | 默认方法 |
| `"time"` | 按实际时间间隔线性插值 | 需要唯一、递增的 `DatetimeIndex` 或 `TimedeltaIndex` |
| `"nearest"` | 最近邻插值 | 依赖 SciPy |
| `"cubic"` | 三次插值 | 依赖 SciPy；需要足够的有效观测，可能产生过冲 |

`NaN` 和 `pd.NA` 视为缺失。正负无穷通常代表计算溢出或数据错误，因此不会被静默
插补，而是直接报错。

### 长缺口和边界

`max_gap` 是允许填补的最大连续缺失长度。超过该长度的缺口会完整保留，不会从两端
部分填补：

```python
result = interpolate_missing(data, max_gap=2)
```

默认 `edge="keep"`，只做真正的内插，序列首尾缺失保持不变。若业务上确认可以使用
最近观测延伸边界，可显式设置：

```python
result = interpolate_missing(data, edge="nearest")
```

边界填补同样受 `max_gap` 限制。全缺失序列不会被人为赋予水平。

### 审计信息

`InterpolationResult` 提供：

- `missing_mask`：原始缺失位置；
- `filled_mask`：本次成功填补的位置；
- `remaining_mask`：仍然缺失的位置；
- `n_missing`、`n_filled`、`n_remaining`；
- `complete`：是否已填补全部原始缺失值；
- `summary()`：方法、限制和数量摘要。

插值是确定性预处理，不提供插补不确定性。长缺口、结构突变或需要推断标准误的场景，
应使用状态空间模型、多重插补或其他与数据生成过程匹配的方法。

## 时间序列统计摘要

`TimeSeriesSummary` 使用 pandas 的描述统计，并调用 `TsPlots` 已有的
`plot_acf`、`plot_pacf` 生成水平值和一阶差分的诊断图：

```python
from Ts.TsUtils import TimeSeriesSummary

analysis = TimeSeriesSummary(series, nlags=20)
print(analysis.summary())

figure = analysis.figure_
axes = analysis.axes_
```

单列 `DataFrame` 会自动选择唯一一列。多列 `DataFrame` 必须指定指标名称：

```python
analysis = TimeSeriesSummary(dataframe, variable="GDP", nlags=20)
print(analysis.summary())
```

统计摘要和诊断图仅使用 `variable` 指定的列。指标不存在或列名不唯一时会明确报错。

摘要包括样本量、有效观测数、频率、起止索引、统计五数、均值、标准差，
以及缺失值数量、比例和全部缺失时间戳。没有时间索引的数组改为报告缺失位置。

存在缺失值时，`summary()` 仍返回完整统计摘要，但不会擅自删除或插补数据；
四个相关图面板会说明 ACF/PACF 未计算。需要绘图时，应先显式调用
`interpolate_missing()` 或采用其他符合数据生成过程的缺失值处理方法。

`summary(plot=False)` 只生成文本。`plot()` 可显式生成或重新生成诊断图。

## 运行测试

从 `Ts` 的父目录运行：

```powershell
python -m pytest Ts/TsUtils/tests -p no:cacheprovider -q
```
