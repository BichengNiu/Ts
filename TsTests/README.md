# `TsTests` — 时间序列统计检验包

提供时间序列分析中常用的统计检验，涵盖五大类：标准单位根检验、结构性突变单位根检验、
ARCH 效应检验、正态性检验和协整检验。

## 模块结构

```
TsTests/
├── __init__.py           # 统一导出接口
├── _base.py              # BaseTest (ABC) + BaseTestResult (dataclass) — 统一接口
├── _utils.py             # 通用工具函数（输入解析、模型验证）
├── _break_utils.py       # 结构突变专用工具（虚拟变量、滞后选择、回归构建）
├── _critical_values.py   # 临界值表 + 插值函数
├── _unitroot_plot.py     # 单位根检验共享绘图工具
├── _adf.py               # ADFTest + ADFTestResult
├── _phillips_perron.py   # PhillipsPerronTest + PhillipsPerronTestResult
├── _kpss.py              # KPSSTest + KPSSTestResult
├── _perron.py            # PerronTest + PerronTestResult
├── _zivot.py             # ZivotAndrewsTest + ZivotAndrewsTestResult
├── _ljungbox.py          # LjungBoxTest + LjungBoxTestResult
├── _engle_lm.py          # EngleLMTest + EngleLMTestResult
├── _normality.py         # NormalityTest + NormalityTestResult
├── _johansen.py          # JohansenTest + JohansenTestResult
├── _toda_yamamoto.py     # TodaYamamotoTest + TodaYamamotoTestResult
├── tests/                # 单元测试
│   ├── test_adf.py
│   ├── test_phillips_perron.py
│   ├── test_kpss.py
│   ├── test_perron.py
│   ├── test_zivot.py
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
    LjungBoxTest,
    EngleLMTest,
    NormalityTest,
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
za = ZivotAndrewsTest(data, time_index=years, model="both", max_lags=8)
za.fit()
za.result_.plot_test()

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
| `pvalue` | `float \| None` | p 值（结构突变检验为 None） |
| `lags` | `int` | 使用的滞后阶数 / 带宽 |
| `nobs` | `int` | 有效观测数 |
| `residuals` | `ndarray \| None` | 残差序列 |

## 检验对照

| 检验 | 类名 | H0 | 类型 | 临界值来源 |
|------|------|----|------|-----------|
| ADF | `ADFTest` | 存在单位根 | 标准单位根 | MacKinnon (1994) |
| Phillips-Perron | `PhillipsPerronTest` | 存在单位根 | 标准单位根 | MacKinnon / Hamilton |
| KPSS | `KPSSTest` | 平稳 | 标准单位根 | Kwiatkowski et al. (1992) |
| Perron (1989) | `PerronTest` | 含结构突变的单位根 | 已知断点 | Table IV.B（有限样本） |
| Zivot-Andrews (1992) | `ZivotAndrewsTest` | 含结构突变的单位根 | 未知断点 | Table 2（渐近） |
| Ljung-Box Q-test | `LjungBoxTest` | 无自相关（残差平方） | ARCH 效应 | chi-squared 分布 |
| Engle LM test | `EngleLMTest` | 无 ARCH 效应 | ARCH 效应 | chi-squared / F 分布 |
| Jarque-Bera | `NormalityTest` | 正态分布 | 正态性 | chi-squared(2) — 支持 `plot_test()` |
| Johansen 迹检验 | `JohansenTest` | 协整秩 <= r | 协整 | Osterwald-Lenum (1992) — 支持 `summary(alpha_idx=N)` |
| Johansen 最大特征根 | `JohansenTest` | 协整秩 = r | 协整 | Osterwald-Lenum (1992) |
| Toda-Yamamoto | `TodaYamamotoTest` | 无格兰杰因果 | 因果检验 | chi-squared（Wald 检验） |

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
| `model` | str | `"both"` | 突变模型: `"intercept"`, `"trend"`, `"both"` |
| `max_lags` | int | `8` | 最大滞后阶数 |
| `lag_method` | str | `"tstat"` | 滞后选择: `"tstat"`, `"aic"`, `"bic"` |

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

## 模型类型（结构突变检验）

| model | 含义 | 突变虚拟变量 |
|-------|------|-------------|
| `"intercept"` | 截距突变（Crash Model） | DL, DP |
| `"slope"` | 趋势斜率突变（Changing-Growth） | DL, DP, DT |
| `"both"` | 截距 + 斜率同时突变 | DL, DP, DT |

## 滞后选择方法（结构突变检验）

| method | 说明 |
|--------|------|
| `"tstat"` | 一般到特殊的 t 统计量法（默认） |
| `"aic"` | 最小化 AIC |
| `"bic"` | 最小化 BIC |

## 绘图功能

标准单位根检验（ADF/PP/KPSS）和结构突变检验（Perron/Zivot-Andrews）的 Result
类均提供 `plot_test()` 方法，绘制检验统计量与临界值的对比图（由 `TsTsPlots.style`
统一风格）。

```python
# 所有单位根检验 Result 均支持 plot_test()
adf.result_.plot_test()  # ADF
pp.result_.plot_test()  # Phillips-Perron
kpss.result_.plot_test()  # KPSS
pt.result_.plot_test()  # Perron
za.result_.plot_test()  # Zivot-Andrews（含 t 统计量和 IC 曲线）
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
