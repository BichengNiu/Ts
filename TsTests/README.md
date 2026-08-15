# `TsTests` — 时间序列统计检验包

提供时间序列分析中常用的统计检验，涵盖标准单位根检验、结构突变单位根检验、
回归参数稳定性检验、ARCH 效应检验、正态性检验、协整、因果检验与传递函数诊断。

## 交互式帮助

公共检验及结果对象的参数、返回值和可执行样例均已写入 docstring。在
IPython/Jupyter 中输入 `?ADFTest`（或 `ADFTest?`），也可在 Python 中调用
`help(ADFTest)` 查看完整帮助。

## 模块结构

```
TsTests/
├── __init__.py           # 统一导出接口
├── _base.py              # BaseTest (ABC) + BaseTestResult (dataclass) — 统一接口
├── _utils.py             # 通用工具函数（输入解析、模型验证）
├── _break_utils.py       # 结构突变专用工具（虚拟变量、滞后选择、回归构建）
├── _critical_values.py   # 临界值表 + 插值函数
├── _unitroot_plot.py     # 单位根绘图薄再导出（实现位于 TsPlots/unitroot_plot.py）
├── _adf.py               # ADFTest + ADFTestResult
├── _phillips_perron.py   # PhillipsPerronTest + PhillipsPerronTestResult
├── _kpss.py              # KPSSTest + KPSSTestResult
├── _perron.py            # PerronTest + PerronTestResult
├── _zivot.py             # ZivotAndrewsTest + ZivotAndrewsTestResult
├── _lee_strazicich.py    # LeeStrazicichTwoBreakTest + Result
├── _regression_break_utils.py  # 回归稳定性检验共享设计构建器
├── _chow.py              # ChowTest + ChowTestResult
├── _cusum.py             # CUSUMTest + CUSUMTestResult
├── _bai_perron.py        # BaiPerronTest + BaiPerronTestResult
├── _ljungbox.py          # LjungBoxTest + LjungBoxTestResult
├── _engle_lm.py          # EngleLMTest + EngleLMTestResult
├── _normality.py         # NormalityTest + NormalityTestResult
├── _johansen.py          # JohansenTest + JohansenTestResult
├── _toda_yamamoto.py     # TodaYamamotoTest + TodaYamamotoTestResult
├── _feedback.py          # FeedbackTest + per-input OLS/F-test results
├── _residual_ccf.py      # ResidualCCFTest + 逐输入 CCF/S* 结果
├── tests/                # 单元测试
│   ├── test_adf.py
│   ├── test_phillips_perron.py
│   ├── test_kpss.py
│   ├── test_perron.py
│   ├── test_zivot.py
│   ├── test_lee_strazicich.py
│   ├── test_chow.py
│   ├── test_cusum.py
│   ├── test_bai_perron.py
│   ├── test_ljungbox.py
│   ├── test_engle_lm.py
│   ├── test_normality.py
│   ├── test_johansen.py
│   ├── test_toda_yamamoto.py
│   └── test_convenience.py
└── README.md
```

## 快速开始

```python
from Ts.TsTests import (
    ADFTest,
    PhillipsPerronTest,
    KPSSTest,
    PerronTest,
    ZivotAndrewsTest,
    LeeStrazicichTwoBreakTest,
    ChowTest,
    CUSUMTest,
    BaiPerronTest,
    LjungBoxTest,
    EngleLMTest,
    NormalityTest,
    FeedbackTest,
    ResidualCCFTest,
)

# ADF 检验
adf = ADFTest(data, trend="c", max_lags=8, autolag="AIC")
print(adf.summary())
adf.result_.plot_test()

# Phillips-Perron 检验
pp = PhillipsPerronTest(data, trend="c")
print(pp.summary())
pp.result_.plot_test()

# KPSS 检验（H0: 平稳，与 ADF/PP 相反）
kpss = KPSSTest(data, trend="c")
print(kpss.summary())

# Perron 检验（已知断点）
pt = PerronTest(data, break_year=1929, time_index=years, model="intercept")
print(pt.summary())
pt.result_.plot_test()

# Zivot-Andrews 检验（未知断点）
za = ZivotAndrewsTest(data, time_index=years, model="intercept", max_lags=8)
za.fit()
za.result_.plot_test()

# Lee-Strazicich 检验（两个未知断点；H0 中也允许突变）
ls = LeeStrazicichTwoBreakTest(data, time_index=years, model="C", max_lags=8)
ls_result = ls.fit()
print(ls_result.break_years)

# 已知日期的回归系数稳定性：Chow
chow = ChowTest(y, break_year=years[50], exog=x, time_index=years)
print(chow.fit())

# 未知位置的整体参数不稳定：OLS 残差 CUSUM
cusum = CUSUMTest(y, exog=x, time_index=years)
print(cusum.fit())

# 多个未知回归断点：Bai-Perron
bp = BaiPerronTest(
    y,
    exog=x,
    time_index=years,
    max_breaks=3,
    n_bootstrap=199,
    random_state=42,
)
print(bp.fit().break_years)

# Ljung-Box Q 检验（ARCH 效应检测）
lb = LjungBoxTest(returns, lags=10)
print(lb.summary())

# Engle LM 检验（ARCH 效应检测）
lm = EngleLMTest(returns, lags=10)
print(lm.summary())

# Jarque-Bera 正态性检验
jb = NormalityTest(residuals)
print(jb.summary())

# Toda-Yamamoto 格兰杰因果检验（无需单位根/协整预检验）
ty = TodaYamamotoTest(data2d, p=2)
print(ty.summary())

# 分布滞后输入的条件反馈检验
feedback = FeedbackTest(y, X, lags=4).fit()
print(feedback.summary())       # 每个输入的完整 OLS + 联合 F 检验
print(feedback.tests)           # 紧凑的逐输入检验表

# 传递函数残差 CCF：output_residuals 为最终模型残差，input_innovations 为输入新息
ccf_result = ResidualCCFTest(
    output_residuals,
    {"price": input_innovations},
    lags=12,
    transfer_params={"price": 2},
).fit()
print(ccf_result.tests)
ccf_result.plot_test()
```

## 统一接口

所有检验类继承 `BaseTest`（ABC），遵循统一契约：

| 方法/属性 | 说明 |
|-----------|------|
| `__init__(data, ...)` | 接受时间序列 + 检验参数 |
| `fit()` | 执行检验，返回 Result 对象 |
| `summary()` | 返回格式化的字符串报告（自动调用 `fit`） |
| `result_` | 存储 `fit()` 执行后的结果对象 |

所有 Result 类继承 `BaseTestResult`，共享字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `statistic` | `float` | 检验统计量 |
| `pvalue` | `float \| None` | p 值；仅临界值推断的方法可为 `None` |
| `lags` | `int \| None` | 使用的滞后阶数 / 带宽；非滞后检验为 `None` |
| `nobs` | `int` | 有效观测数 |
| `residuals` | `ndarray \| None` | 残差序列 |

### 公共复合结果对象

部分检验包含多个方程或多个输入，因此使用分层结果对象。以下类型均属于
`TsTests.__all__` 的正式公共接口：

| 结果类 | 说明 |
|--------|------|
| `BaseMultiTestResult` | 多方程、多输入检验结果的公共基类 |
| `LeeStrazicichTwoBreakTestResult` | 两个未知结构突变的单位根检验结果 |
| `FeedbackEquationResult` | 单个输入对应的完整条件反馈 OLS 方程和联合检验 |
| `FeedbackTestResult` | 汇总全部输入条件反馈方程的结果容器 |
| `ResidualCCFInputResult` | 单个输入的逐阶残差 CCF 与联合 S* 检验 |
| `ResidualCCFTestResult` | 汇总全部输入残差 CCF 诊断的结果容器 |

## 检验对照

| 检验 | 类名 | H0 | 类型 | 临界值来源 |
|------|------|----|------|-----------|
| ADF | `ADFTest` | 存在单位根 | 标准单位根 | MacKinnon (1994) |
| Phillips-Perron | `PhillipsPerronTest` | 存在单位根 | 标准单位根 | MacKinnon / Hamilton |
| KPSS | `KPSSTest` | 平稳 | 标准单位根 | Kwiatkowski et al. (1992) |
| Perron (1989) | `PerronTest` | 含结构突变的单位根 | 已知断点 | Table IV.B（有限样本） |
| Zivot-Andrews (1992) | `ZivotAndrewsTest` | 含结构突变的单位根 | 未知断点 | Table 2（渐近） |
| Lee-Strazicich (2003) | `LeeStrazicichTwoBreakTest` | 含两个结构突变的单位根 | 两个未知断点 | 发表的 Model A/C 渐近临界值 |
| Chow | `ChowTest` | 指定日期前后回归系数稳定 | 已知断点 | 经典 F 分布 |
| OLS-CUSUM | `CUSUMTest` | 回归参数稳定 | 未知不稳定位置 | Brownian bridge 渐近分布 |
| Bai-Perron | `BaiPerronTest` | 回归系数在所有区间稳定 | 多个未知断点 | Rademacher wild bootstrap |
| Ljung-Box Q-test | `LjungBoxTest` | 无自相关（残差平方） | ARCH 效应 | chi-squared 分布 |
| Engle LM test | `EngleLMTest` | 无 ARCH 效应 | ARCH 效应 | chi-squared / F 分布 |
| Jarque-Bera | `NormalityTest` | 正态分布 | 正态性 | chi-squared(2) — 支持 `plot_test()` |
| Johansen 迹检验 | `JohansenTest` | 协整秩 <= r | 协整 | Osterwald-Lenum (1992) — 支持 `summary(alpha_idx=N)` |
| Johansen 最大特征根 | `JohansenTest` | 协整秩 = r | 协整 | Osterwald-Lenum (1992) |
| Toda-Yamamoto | `TodaYamamotoTest` | 无格兰杰因果 | 因果检验 | chi-squared（Wald 检验） |
| Conditional feedback | `FeedbackTest` | 因变量的 K 阶滞后系数联合为 0 | 输入外生性诊断 | 经典 OLS F 检验 |
| Residual CCF | `ResidualCCFTest` | 0–K 阶残差互相关联合为 0 | 传递函数充分性 | Box–Jenkins S* 的 chi-squared 近似 |

## 结构突变方法选择

| 研究问题 | 推荐方法 |
|----------|----------|
| 单位根，已知一个断点 | `PerronTest` |
| 单位根，未知一个断点 | `ZivotAndrewsTest` |
| 单位根，未知两个断点，且断点进入原假设 | `LeeStrazicichTwoBreakTest` |
| 回归系数稳定性，断点日期事先指定 | `ChowTest` |
| 回归参数是否存在未知位置的不稳定 | `CUSUMTest` |
| 回归中存在几个未知断点及其位置 | `BaiPerronTest` |

单位根突变检验和回归参数稳定性检验回答不同问题，不能互相替代。

## 标准单位根检验参数

### ADFTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `trend` | str | `"c"` | 趋势项: `"c"`, `"ct"`, `"n"` |
| `max_lags` | int | `8` | 最大滞后阶数 |
| `lags` | int \| None | `None` | 固定滞后阶数（覆盖自动选择） |
| `autolag` | str | `"AIC"` | 滞后选择准则: `"AIC"`, `"BIC"`, `"t-stat"` |

### PhillipsPerronTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `trend` | str | `"c"` | 趋势项: `"c"`, `"ct"`, `"n"` |
| `lags` | int \| None | `None` | Newey-West 带宽（None 自动选择） |

### KPSSTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `trend` | str | `"c"` | 趋势项: `"c"`, `"ct"` |
| `nlags` | str \| int | `"auto"` | 带宽: `"auto"`, `"legacy"` 或整数 |

## 结构突变检验参数

### PerronTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `break_year` | float | — | 已知断点日期 |
| `time_index` | array-like | `None` | 时间索引（None 则从 0 开始） |
| `model` | str | `"intercept"` | 突变模型: `"intercept"`, `"slope"`, `"both"` |
| `lags` | int \| None | `None` | 固定滞后阶数（None 自动选择） |
| `max_lags` | int | `8` | 最大滞后阶数 |
| `lag_method` | str | `"tstat"` | 滞后选择: `"tstat"`, `"aic"`, `"bic"` |

### ZivotAndrewsTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `time_index` | array-like | `None` | 时间索引 |
| `model` | str | `"intercept"` | 突变模型: `"intercept"`, `"slope"`, `"both"` |
| `max_lags` | int | `8` | 最大滞后阶数 |
| `lag_method` | str | `"tstat"` | 滞后选择: `"tstat"`, `"aic"`, `"bic"` |

### LeeStrazicichTwoBreakTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `time_index` | array-like | `None` | 仅用于断点标签，不改变统计模型 |
| `model` | str | `"A"` | `"A"` 为两个水平突变；`"C"` 为两个水平和趋势突变 |
| `lags` | int \| None | `None` | 固定滞后阶数；`None` 时自动选择 |
| `max_lags` | int | `8` | 自动选择的最大滞后阶数 |
| `lag_method` | str | `"tstat"` | `"tstat"`, `"aic"`, `"bic"` |
| `trim` | float | `0.10` | 两端搜索修剪比例 |

## 回归参数稳定性检验参数

`ChowTest`、`CUSUMTest` 和 `BaiPerronTest` 共用回归输入契约：

- 数组输入：`data=y`，解释变量通过 `exog=x` 指定；
- DataFrame 输入：通过 `y_col`、`time_col`、`exog_cols` 明确列；
- `trend` 支持 `"n"`、`"c"`、`"ct"`；
- 不自动删除或插补缺失值，因为这会改变断点位置。

DataFrame 示例：

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
frame = pd.DataFrame(
    {
        "year": np.arange(2000, 2040),
        "outcome": rng.normal(size=40),
        "driver": rng.normal(size=40),
    }
)
result = ChowTest(
    frame,
    break_year=2019,
    y_col="outcome",
    time_col="year",
    exog_cols=["driver"],
).fit()
```

### ChowTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like / DataFrame | — | 因变量或包含回归数据的表 |
| `break_year` | float | — | 事先指定、且必须精确匹配 `time_index` 的断点 |
| `exog` | array-like | `None` | 数组路径下的解释变量 |
| `trend` | str | `"c"` | 确定项规格 |

经典 Chow F 推断要求误差独立且同方差；断点必须在查看结果前指定。

### CUSUMTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like / DataFrame | — | 因变量或包含回归数据的表 |
| `exog` | array-like | `None` | 数组路径下的解释变量 |
| `trend` | str | `"c"` | 确定项规格 |

这是基于全样本 OLS 残差的 CUSUM，不是 Brown–Durbin–Evans 递归残差
CUSUM；其 Brownian-bridge 推断对回归量分布有经典渐近条件。

### BaiPerronTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like / DataFrame | — | 因变量或包含回归数据的表 |
| `exog` | array-like | `None` | 数组路径下的解释变量 |
| `max_breaks` | int | `5` | 最大未知断点数 |
| `breaks` | int \| None | `None` | 固定断点数；`None` 时用 BIC/LWZ 选择 |
| `criterion` | str | `"bic"` | 断点数选择准则：`"bic"` 或 `"lwz"` |
| `trim` | float | `0.15` | 最小区间比例 |
| `n_bootstrap` | int | `99` | wild bootstrap 重复次数，至少 19 |
| `random_state` | int \| None | `None` | 可复现随机种子 |

实现使用全局动态规划而不是贪心二分，返回 supF、UDmax、WDmax 和断点区间。
wild bootstrap 对异方差稳健，但当前版本不对序列相关稳健。

## ARCH 效应检验参数

### LjungBoxTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 残差序列 |
| `lags` | int | `10` | 滞后阶数 |
| `apply_squared` | bool | `True` | 是否对残差平方（m^2 检验） |

### NormalityTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 残差序列（需 >= 8 obs） |

### EngleLMTest

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 残差序列 |
| `lags` | int | `10` | 滞后阶数 |

## 突变模型的确定项

| 检验 | model | 突变确定项 |
|------|-------|------------|
| Perron | `"intercept"` | DL, DP |
| Perron | `"slope"` | DL, DT |
| Perron | `"both"` | DL, DP, DT |
| Zivot-Andrews | `"intercept"` | DL |
| Zivot-Andrews | `"slope"` | DT |
| Zivot-Andrews | `"both"` | DL, DT |
| Lee-Strazicich | `"A"` | 两个 DU |
| Lee-Strazicich | `"C"` | 两个 DU 和两个 DT |

## 滞后选择方法（结构突变检验）

| method | 说明 |
|--------|------|
| `"tstat"` | 一般到特殊的 t 统计量法（默认） |
| `"aic"` | 最小化 AIC |
| `"bic"` | 最小化 BIC |

## 绘图功能

标准单位根、结构突变单位根和回归参数稳定性检验的 Result 类均提供
`plot_test()`。图形分别展示统计量与临界值、估计断点、拟合区间或累计残差路径。

```python
# 所有单位根检验 Result 均支持 plot_test()
adf.result_.plot_test()  # ADF
pp.result_.plot_test()  # Phillips-Perron
kpss.result_.plot_test()  # KPSS
pt.result_.plot_test()  # Perron
za.result_.plot_test()  # Zivot-Andrews（含 t 统计量和 IC 曲线）
ls.result_.plot_test()  # Lee-Strazicich（两个未知断点）
chow.result_.plot_test()  # Chow（断点前后拟合）
cusum.result_.plot_test()  # CUSUM（累计残差）
bp.result_.plot_test()  # Bai-Perron（多个未知断点和区间拟合）
```

## 协整检验参数

### JohansenTest

```python
JohansenTest(data, lags=2, trend="constant", cols=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like (nobs, k) | — | 多变量时间序列，k >= 2 |
| `lags` | int | `2` | VAR 水平滞后阶数 |
| `trend` | str | `"constant"` | 确定项规格：`"none"`, `"constant"`, `"trend"` |
| `cols` | list of str | `None` | 变量列名（None 自动生成 `"y0"`, `"y1"`, ...） |

`trend` 与底层 `statsmodels.coint_johansen` 支持的确定项规格一致：

| trend | `det_order` | 说明 |
|-------|-------------|------|
| `"none"` | `-1` | 无确定项 |
| `"constant"` | `0` | 常数项（默认） |
| `"trend"` | `1` | 线性趋势 |

受约束常数和受约束趋势不能由 `coint_johansen` 的 `det_order`
准确表达，因此不作为本接口的可选值。

`summary(alpha_idx=1)` 控制输出临界值的显著性水平：
`0` = 90%, `1` = 95%, `2` = 99%。

## 格兰杰因果检验参数

### TodaYamamotoTest

```python
TodaYamamotoTest(data, p, d_max=None, trend="c", cols=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like (nobs, k) | — | 多变量时间序列，k >= 2 |
| `p` | int | — | 用于因果检验的 VAR 滞后阶数 |
| `d_max` | int or None | `None` | 最大单整阶数 (0/1/2)；None 时 ADF 自动检测 |
| `trend` | str | `"c"` | 确定性趋势: `"c"`, `"ct"`, `"n"` |
| `cols` | list of str | `None` | 变量列名（None 自动生成 `"y0"`, `"y1"`, ...） |

Toda-Yamamoto 方法通过估计 VAR(p + d_max) 并仅对前 p 阶滞后做 Wald 检验，
使得即使变量为 I(1) 或存在协整关系也能进行有效的格兰杰因果推断。
输出为 `TodaYamamotoTestResult`，包含每对变量的 chi2 统计量、自由度和 p 值，
以及每方程的联合 ALL 检验。格式与 `TsModels.GrangerCausalityResult` 一致。

### FeedbackTest

```python
FeedbackTest(
    y,
    exog,
    lags=K,
    exog_names=None,
    tested_inputs=None,
    trend="c",
    missing="raise",
    alpha=0.05,
)
```

对每个受检输入 `x_i`，估计当前 `x_i` 对所有输入的 1–K 阶滞后及
`y` 的 1–K 阶滞后的 OLS 回归，并检验 `H0: y.L1 = ... = y.LK = 0`。
`tested_inputs` 只选择左侧受检方程；其他输入仍作为滞后控制变量保留。

`result.regressions` 保存完整 statsmodels OLS 结果，`result.tests` 汇总 F 值、
自由度、p 值和拒绝结论，`result.summary()` 同时报告完整回归和联合检验。
该检验衡量条件预测反馈，不把统计预测关系表述为结构性因果。

`missing="drop"` 在构造滞后矩阵之后删除不完整行，不会先压缩原序列并把
缺口两侧错误地当作相邻观测。

### ResidualCCFTest

该检验用于传递函数估计后的诊断阶段。它把最终 DR/RDL 模型残差
\(a_t\) 与输入单变量 ARIMA 的新息 \(\hat a_t\) 进行交叉相关：正滞后
\(k>0\) 表示当前输出残差 \(a_t\) 与过去输入新息
\(\hat a_{t-k}\) 的相关。识别阶段则是将输入 ARIMA 滤波器同时作用于
原始 X 和 Y；两者不能混为同一步骤。

```python
ResidualCCFTest(
    output_residuals,
    input_residuals,
    lags=12,
    input_names=None,
    transfer_params=0,
    alpha=0.05,
)
```

实现使用 Box–Jenkins 的固定 \(n\) 分母样本 CCF。单阶近似标准误为
\(1/\sqrt n\)，置信带为
\(\pm z_{1-\alpha/2}/\sqrt n\)；教材常写的 95% `±2/sqrt(n)` 是其近似。
联合检验为

\[
S^*=n^2\sum_{k=0}^{K}\frac{r_k^2}{n-k},
\qquad df=K+1-m,
\]

其中 \(m\) 只包含该输入传递函数中实际估计的 numerator/denominator
参数，不包括扰动 ARIMA 参数、固定 delay 或稀疏规格中固定为零的系数。
`result.tests` 汇总每个输入的 S*、自由度、p 值与结论；`get(name)` 返回
逐阶相关、标准误、置信限与显著峰值；`plot_test()` 使用 TsPlots 统一样式。

显著峰值是遗漏动态的定位线索，不是机械增加同阶 numerator 参数的命令。
输入之间的相关、反馈、错误的输入预白化或 denominator 设定也可能造成峰值。
短滞后的 `1/sqrt(n)` 近似通常偏保守。方法依据 Box 与 Jenkins（1976，
pp. 395–396）；[SAS ARIMA Procedure 官方说明](https://documentation.sas.com/api/collections/pgmsascdc/v_016/docsets/etsug/content/arima.pdf?locale=en)
将同一方法称为 “Cross-correlation Check of Residuals”。

## 与 TsPlots 的衔接

检验结果绘图通过 `Ts.TsPlots.style` 获取统一的色板（`DEFAULT_PALETTE`）、图尺寸
（`FIGSIZE`）和轴样式（`style_axes`），确保所有图表风格一致。

## 项目依赖

- `numpy`, `pandas`, `scipy`, `statsmodels`
- `arch`（Phillips-Perron 检验）
- 绘图功能依赖 `matplotlib` 和 `TsPlots`

## 运行测试

```bash
python -m pytest code/python/Ts/TsTests/tests/ -v
```
