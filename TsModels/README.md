# Ts/TsModels

时间序列模型估计工具包。提供 SARIMA、GARCH、VAR、SVAR、VECM 的统一使用接口，结果对象与 TsPlots、TsMetrics、TsTests 衔接。

## Missing-data contract

All public model constructors accept `missing="raise"` or `missing="drop"`.
The default is `"raise"`, which rejects both `NaN` and infinite values.
Explicit `"drop"` removes affected observations and records their original
zero-based row positions in `dropped_positions`. Multivariate models remove
complete rows. No model imputes raw observations.

## Date-index contract

所有预测模型构造器都接受 `dates=None`。`pandas.Series` 或
`pandas.DataFrame` 的 `DatetimeIndex` 会自动保留；数组数据可通过 `dates=`
显式提供日期。日期数量必须与原始观测一致，且不得包含缺失、重复或逆序值。
使用 `missing="drop"` 删除观测时，对应日期会同步删除。

## 模块结构

```
TsModels/
├── __init__.py         # 统一导出接口
├── _base.py            # BaseModel (ABC) + BaseModelResult + ResidualTestResults (dataclass)
├── _sarima.py          # SARIMA + SARIMAResult
├── _garch_result.py    # GARCHResult (dataclass) + 参数缩放辅助函数
├── _garch_base.py      # _BaseVolModel — 参数验证 + fit 调度 + IGARCH MLE
├── _garch.py           # GARCH — 公开 API 入口
├── _var.py             # VAR + VARResult — 向量自回归
├── _svar.py            # SVAR + SVARResult — 结构向量自回归
├── _vecm.py            # VECM + VECMResult — 向量误差修正模型
├── _auto.py            # AutoSARIMA + AutoGARCH + AutoModelResult
├── _compare.py         # compare_models — Stata 风格对比表格
├── tests/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_sarima.py
│   ├── test_garch.py
│   ├── test_auto.py
│   ├── test_compare.py
│   ├── test_garch_refactor.py
│   ├── test_var.py
│   ├── test_svar.py
│   └── test_vecm.py
└── README.md
```

## 快速开始

```python
import numpy as np

from Ts.TsModels import SARIMA, GARCH, VAR
from Ts.TsSims import simulate_sarima, simulate_garch

# AR(1) 估计
data = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42).data
model = SARIMA(data, order=(1, 0, 0))
result = model.fit()
print(result.summary())
result.plot_diagnostics()

# ARCH(2) 估计 (GARCH with q=0)
data = simulate_garch(n=200, p=2, q=0, seed=42).data
model = GARCH(data, p=2, q=0)
result = model.fit()

# GARCH(1,1) 估计
data = simulate_garch(n=300, p=1, q=1, seed=42).data
model = GARCH(data, p=1, q=1)
result = model.fit()
result.test_residuals(lags=10)
```

## 统一接口

估计模型继承 `BaseModel`，并通过结构化 Result 对象提供统一的拟合、预测和诊断接口：

| 方法/属性 | 说明 |
|-----------|------|
| `__init__(data, ...)` | 接受时间序列 + 模型参数 |
| `fit()` | 执行分解或估计，返回 Result 对象 |
| `summary()` | 返回格式化的字符串报告（自动调用 fit） |
| `result_` | 存储 `fit()` 执行后的结果对象 |
| `backtest(initial_window, ...)` | 逐预测起点重新拟合的滚动/扩展窗口历史回测 |
| `backcast(steps, alpha)` | 反转时间后重新拟合并估计样本前数值 |

估计模型的 Result 类继承 `BaseModelResult`，共享方法和字段：

| 方法/字段 | 说明 |
|-----------|------|
| `.summary()` | 参数估计表 + AIC/BIC |
| `.plot_fit()` | 实际值与拟合值对比图 |
| `.plot_diagnostics()` | 诊断图：残差 (含检验结果) + ACF + PACF（3-panel） |
| `.test_residuals(lags)` | 四项残差检验：白噪音 + 正态性 + Ljung-Box + Engle LM |
| `.params` | 估计参数 dict |
| `.aic` / `.bic` | 信息准则 |
| `.residuals` | 残差序列 |

## 模型专属方法

| 模型 | 方法 | 说明 |
|------|------|------|
| `SARIMA` / `SARIMAResult` | `.predict(start, end, dynamic, alpha)` | 样本内预测与未来预测；性能评估由 `TsMetrics` 负责 |
| | `.arroots` | AR 多项式特征根 (ndarray) |
| | `.maroots` | MA 多项式特征根 (ndarray) |
| | `.is_stationary` | AR 多项式的全部根是否位于单位圆外 |
| | `.is_invertible` | MA 多项式的全部根是否位于单位圆外 |
| | `.ar_lags` / `.ma_lags` | 实际参与估计的非季节 AR/MA 滞后 |
| | `.fixed_params` | 因稀疏滞后设定而固定为 0 的系数 |
| | `.plot_roots(title)` | AR/MA 逆根单位圆图 |
| | `.long_run_equilibrium()` | 无条件均值 (平稳 ARMA, d=0, trend="c" → float；否则 None) |
| `GARCH` / `GARCHResult` | `.predict(start, end, dynamic, alpha)` | 条件波动率预测；性能评估由 `TsMetrics` 负责 |
| | `.conditional_volatility` | 条件波动率 σ_t |
| | `.test_persistence()` | IGARCH 持久性 Wald 检验 |
| | `.long_run_equilibrium()` | 无条件方差 (协方差平稳 GARCH/GJR → float；EGARCH/IGARCH → None) |
| `VAR` / `VARResult` | `.irf(periods, orth, alpha)` | 脉冲响应函数，返回 `IRFResult`（含置信带） |
| | `.oirf(periods, alpha)` | 正交化脉冲响应函数，`irf(orth=True)` 的便捷封装 |
| | `.fevd(periods, alpha, n_draws, seed)` | 预测误差方差分解，返回 `FEVDResult`（含 Monte Carlo 置信带） |
| | `.plot_irf(periods, orth, alpha)` | IRF 图（含置信带） |
| | `.granger_causality(caused, causing, kind)` | Granger 因果检验（单对/联合/全部两两）——无参数时运行全部检验（等价于 Stata `vargranger`）|
| | `.predict(start, end, dynamic, alpha)` | 多步预测与置信区间；`dynamic=True` 尚不支持并会明确报错；性能评估由 `TsMetrics` 负责 |
| | `.is_stable` | 逆特征根是否全部在单位圆内 (bool) |
| | `.plot_roots(title)` | 逆特征根单位圆图 |
| | `.long_run_equilibrium()` | 无条件均值向量 (stable, trend="c"/"n" → ndarray(k,)；否则 None) |
| `SVAR` / `SVARResult` | `.irf(periods, orth=True)` | 结构性脉冲响应函数 (Θ_h = Ψ_h × A⁻¹B)，返回 `IRFResult` |
| | `.A` / `.B` | 估计的结构矩阵 (ndarray k×k) |
| | `.structural_residuals` | 结构冲击序列 (ndarray nobs×k) |
| | `.sigma_u` | 简化式残差协方差矩阵 (ndarray k×k) |
| `VECM` / `VECMResult` | `.irf(periods, orth, alpha)` | 脉冲响应函数，返回 `IRFResult` |
| | `.fevd(periods, alpha)` | 预测误差方差分解，返回 `FEVDResult` |
| | `.predict(start, end, dynamic, alpha)` | 多步预测；`dynamic=True` 尚不支持，当前不提供预测区间 |
| | `.granger_causality(...)` | Granger 因果检验 |
| | `.alpha` / `.beta` / `.gamma` | VECM 参数矩阵 |
| | `.sigma_u` | 残差协方差矩阵 (ndarray k×k) |
| | `.coint_rank` | 协整秩 |
| | `.is_stable` | 基于 companion matrix 的稳定性检查 (bool) |
| | `.plot_roots(title)` | VECM 逆特征根单位圆图 |

## 性能评估与期间接口

预测性能指标、显式期间 OOS、滚动历史回测和模型性能排序统一由
`TsMetrics` 定义。`BaseModel.oos()` 与 `BaseModel.backtest()` 只是指向
`TsMetrics` 规范实现的便利方法；`PredictResult` 不保存实际值或性能指标。

位置型数据使用零基、闭区间位置：

```python
evaluation = model.oos(
    estimation_period=(0, 79),
    validation_period=(80, 99),
)
print(evaluation.metrics)
```

带日期索引的数据使用精确日期边界：

```python
evaluation = model.oos(
    estimation_period=("2018-01-01", "2022-12-01"),
    validation_period=("2023-03-01", "2023-12-01"),
    alpha=0.05,
)
```

验证期必须严格晚于估计期。两者可以不相邻；模型会连续预测中间间隔，
但只对验证期评分。日期必须唯一、严格递增且边界真实存在。越界、逆序、
重叠、估计样本不足或外生变量未覆盖预测桥接区间都会直接失败。

`OOSResult` 保存 `estimation_indices`、`validation_indices`，日期模型还保存
`estimation_dates`、`validation_dates`。旧 `split` 参数和结果字段已经删除，
不存在弃用期或兼容路径。`predict(oos_start=...)` 伪样本外路径同样不存在。

所有继承 `BaseModel` 的预测模型（SARIMA、GARCH、VAR、VECM、SVAR 及 Auto 模型）
共享该接口。

```python
model.oos(estimation_period, validation_period, alpha=0.05)

model.backtest(
    initial_window,
    horizon=1,
    step=1,
    window='expanding',
    window_size=None,
    alpha=0.05,
    on_error='raise',
)

model.backcast(steps, alpha=0.05)
```
### Backtesting：无未来信息泄漏的滚动起点回测

`backtest()` 在每个预测起点重新拟合模型。扩展窗口使用起点前的全部历史；滚动窗口仅使用最近 `window_size` 个观测。原模型及其已有 `result_` 不会被修改。
`window_size` 仅适用于 `window='rolling'`；扩展窗口传入该参数会明确报错，不会静默忽略。

```python
from Ts.TsModels import SARIMA

model = SARIMA(y, order=(1, 0, 0))

# 扩展窗口：每次加入新观测
expanding = model.backtest(
    initial_window=80,
    horizon=4,
    step=1,
)

# 固定长度滚动窗口
rolling = model.backtest(
    initial_window=80,
    horizon=4,
    step=4,
    window='rolling',
    window_size=80,
    on_error='record',
)
```

`BacktestResult.mean` 和 `.actual` 的形状为：

- 单变量：`(n_origins, horizon)`；
- 多变量：`(n_origins, horizon, n_series)`。

结果同时提供总体 `metrics`、逐预测期 `metrics_by_horizon`、逐变量 `metrics_by_series`。规范指标为 MAE、MSE、RMSE、MAPE、sMAPE、Theil U1 和有效配对数 `n`。`on_error='record'` 会将失败窗口保留为 NaN，并把起点和异常写入 `failures`；默认 `on_error='raise'` 会立即抛出异常。完整契约见 `TsMetrics/README.md`。

GARCH 的预测对象是条件波动率，无法直接与原始收益比较。回测使用当前训练窗口均值中心化后的绝对收益 `abs(y_future - mean(y_train))` 作为可观测代理，并通过 `target='absolute_demeaned_return_proxy'` 明确标记；该代理不等于真实观测波动率。

### Backcasting：反向时间估计样本前数值

`backcast()` 将观测序列反转，使用相同配置重新拟合，向反转序列的未来预测，再把结果还原为 `[-steps, ..., -1]` 的历史顺序：

```python
backcast = model.backcast(steps=12)
print(backcast.indices)  # [-12, ..., -2, -1]
print(backcast.mean)
```

这是反向时间统计估计，不是对未观测历史的因果重建。带线性时间趋势的模型会在反向时间上重新估计趋势，因此解释时必须保留这一限制。GARCH 返回条件波动率并标记 `target='conditional_volatility'`。

当前 GARCH/AutoGARCH 的 `predict()` 尚不能显式接收样本外或样本前外生变量，因此设置 `exog` 时，`backtest()` 与 `backcast()` 会明确抛出 `NotImplementedError`，不会静默猜测外生变量。

## 接口衔接

| 方向 | 衔接方式 |
|------|----------|
| TsModels -> TsPlots | 估计模型 `.plot_fit()` 调用 `plot_series()`；`.plot_diagnostics()` 调用 `plot_series()`, `plot_acf()`, `plot_pacf()` |
| TsModels -> TsTests | `test_residuals()` 自动运行 4 项检验：白噪音 (Ljung-Box raw) + 正态性 (Jarque-Bera) + ARCH (Ljung-Box squared) + ARCH (Engle LM) |
| TsSims -> TsModels | 验证脚本：TsSims 生成数据 -> TsModels 估计 -> 比较真实参数 |

## 模型参数

### SARIMA

```python
SARIMA(data, order=(1,0,0), seasonal_order=(0,0,0,0), trend="c", *, dates=None, exog=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `order` | tuple | `(1,0,0)` | 非季节阶数 `(p,d,q)`；`p`、`q` 可为整数或实际参与估计的滞后列表 |
| `seasonal_order` | tuple | `(0,0,0,0)` | 季节阶数 `(P,D,Q,s)` |
| `trend` | str | `"c"` | 趋势设定：`"n"`, `"c"`, `"t"`, `"ct"` |
| `enforce_stationarity` | bool | `True` | 强制 AR 多项式平稳性 |
| `enforce_invertibility` | bool | `True` | 强制 MA 多项式可逆性 |

稀疏滞后列表用于把未列出的中间阶系数严格固定为 0：

```python
# AR(3)，但固定 ar.L2 = 0
ar_result = SARIMA(data, order=([1, 3], 0, 0)).fit()

# ARMA(1,3)，但固定 ma.L2 = 0
arma_result = SARIMA(data, order=(1, 0, [1, 3])).fit()
```

`summary()` 会报告实际 AR/MA 滞后、固定为 0 的系数、根的最小模、
AR 多项式平稳性、MA 多项式可逆性，以及估计时是否强制这些条件。
对于 `d > 0` 或 `D > 0` 的模型，平稳性结论针对差分后的 AR 多项式，
不表示原始水平序列平稳。

### GARCH

```python
GARCH(data, p=1, q=1, o=0, vol="GARCH", mean="Constant", dist="normal",
      garch_m=False, garch_m_form="vol", ar_lags=None, exog=None, dates=None, missing="raise")
```

纯 ARCH(p) 模型通过 `q=0` 实现：`GARCH(data, p=2, q=0)` 等价于原 ARCH(2)。

GJR-GARCH（非对称 GARCH）通过 `o>=1` 实现：`GARCH(data, p=1, o=1, q=1)`。

EGARCH（指数 GARCH）通过 `vol="EGARCH"` 实现：`GARCH(data, p=1, o=1, q=1, vol="EGARCH")`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `p` | int | `1` | ARCH 阶数 |
| `q` | int | `1` | GARCH 阶数（0 = 纯 ARCH） |
| `o` | int | `0` | 非对称 (GJR) 阶数（>=1 = GJR-GARCH / 非对称 EGARCH） |
| `vol` | str | `"GARCH"` | 波动率模型类型：`"GARCH"`（含 ARCH/GARCH/GJR-GARCH）或 `"EGARCH"` |
| `mean` | str | `"Constant"` | 均值方程类型：`"Constant"`, `"Zero"`, `"AR"` 等 |
| `dist` | str | `"normal"` | 新息分布：`"normal"`, `"t"`, `"skewt"`, `"ged"` |
| `garch_m` | bool | `False` | 启用 GARCH-M，条件波动率进入均值方程（不支持 EGARCH） |
| `garch_m_form` | str | `"vol"` | GARCH-M 中 sigma 形式：`"vol"` (sigma), `"var"` (sigma^2), `"log"` (log sigma^2) |
| `ar_lags` | int/list | `None` | 均值方程 AR 滞后阶数（仅 garch_m=True 时有效） |
| `exog` | array-like | `None` | 外生解释变量 (nobs,) 或 (nobs, k) |
| `igarch` | bool | `False` | IGARCH 约束估计，强制 sum(alpha)+sum(beta)=1 |

#### EGARCH（指数 GARCH）

```python
# EGARCH(1,1,1): 对数方差建模，天然保证方差为正
model = GARCH(data, p=1, o=1, q=1, vol="EGARCH")
result = model.fit()
print(result.summary())

# 对称 EGARCH(1,1): 无杠杆效应
model = GARCH(data, p=1, q=1, vol="EGARCH")
result = model.fit()
```

#### GJR-GARCH（非对称 GARCH）

```python
# GJR-GARCH(1,1,1): 杠杆效应，负面冲击对波动率影响更大
model = GARCH(data, p=1, o=1, q=1)
result = model.fit()
print(result.summary())

# GJR-GARCH(1,1,1) with Student's t
model = GARCH(data, p=1, o=1, q=1, dist="t")
result = model.fit()
```

#### GARCH-M (ARCH-in-Mean)

```python
# GARCH-M(1,1): 条件波动率 sigma_t 进入均值方程
model = GARCH(data, p=1, q=1, garch_m=True)
result = model.fit()

# GARCH-M with variance form
model = GARCH(data, p=1, q=1, garch_m=True, garch_m_form="var")

# GARCH-M with exogenous regressors
model = GARCH(data, p=1, q=1, garch_m=True, exog=X)

# Standard GARCH with exogenous regressors (no GARCH-M)
model = GARCH(data, p=1, q=1, exog=X)
```

#### IGARCH（Integrated GARCH）

```python
# IGARCH(1,1): 约束 alpha+beta=1，波动率冲击具有永久性
model = GARCH(data, p=1, q=1, igarch=True)
result = model.fit()
print(result.summary())  # 模型标签显示 IGARCH(1,1)

# IGARCH(1,1) 预测：方差随预测期线性增长（无均值回归）
pr = result.predict(start=result.nobs, end=result.nobs + 9)
```

IGARCH 通过参数变换 beta_q = 1 - sum(alpha) - sum(beta_{1..q-1}) 在约束下
估计模型。与标准 GARCH 不同，IGARCH 的波动率预测随 horizon 线性增长而非
回归到无条件均值。

**限制**：
- 不支持 `vol="EGARCH"`（抛出 ValueError）
- 不支持 `garch_m=True`（抛出 ValueError）
- `q` 必须 >= 1（GARCH 成分）

### compare_models — 多模型结果对比

```python
from Ts.TsModels import compare_models

models = {
    "GARCH(1,1)": result1,
    "GARCH-M(1,1)": result2,
}
table = compare_models(models)
print(table)
```

输出 Stata 风格的回归表格：参数按 main / ARCHM / ARCH 分组，
mu/_cons、kappa/sigma2、alpha[1]/L.arch、beta[1]/L.garch
自动映射为 Stata 命名，显示 t 统计量和显著性星号。

## 自动最优参数选择

`AutoSARIMA` 和 `AutoGARCH` 通过网格搜索自动选择最优模型阶数。

```python
from Ts.TsModels import AutoSARIMA, AutoGARCH
from Ts.TsSims import simulate_sarima, simulate_garch

# 自动选择最优 ARIMA 阶数
data = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42).data
auto = AutoSARIMA(data, p=(0, 3), d=(0, 1), q=(0, 3), criterion="aic")
result = auto.fit()
print(result.summary())
# 显示: 搜索方式, 选择准则, 最优阶数, 成功/尝试模型数, 最优模型参数表

# 自动选择最优 GARCH 阶数
data = simulate_garch(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1, 4), q=(0, 4), criterion="bic")
result = auto.fit()

# 最优模型结果可通过 best_result 访问
best = result.best_result  # SARIMAResult 或 GARCHResult
```

### AutoSARIMA

```python
AutoSARIMA(data, p=(0,3), d=(0,1), q=(0,3),
           P=(0,1), D=(0,1), Q=(0,1), s=0,
           trend="c", criterion="aic", method="grid", dates=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p`, `d`, `q` | `(min, max)` | `(0,3)`, `(0,1)`, `(0,3)` | 非季节阶数范围 |
| `P`, `D`, `Q` | `(min, max)` | `(0,1)` | 季节阶数范围（s=0 时忽略） |
| `s` | int | `0` | 季节周期（0=无季节性） |
| `criterion` | str | `"aic"` | 选择准则: `aic`, `bic`, `hqic`, `aicc` |

### AutoGARCH

```python
AutoGARCH(data, p=(1,4), q=(0,4), o=(0,0),
          vol="GARCH", mean="Constant", dist="normal",
          criterion="aic", method="grid",
          igarch=False, garch_m=False, garch_m_form="vol",
          ar_lags=None, exog=None, dates=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | `(min, max)` | `(1,4)` | ARCH 阶数范围 |
| `q` | `(min, max)` | `(0,4)` | GARCH 阶数范围（`(0,0)` = 纯 ARCH） |
| `o` | `(min, max)` | `(0,0)` | 非对称 (GJR) 阶数范围（`(0,1)` = 比对标与非对称） |
| `vol` | str | `"GARCH"` | 波动率模型：`"GARCH"`（含 ARCH/GARCH/GJR-GARCH）或 `"EGARCH"` |
| `mean` | str | `"Constant"` | 均值方程：`"Constant"`, `"Zero"`, `"AR"` 等 |
| `dist` | str | `"normal"` | 新息分布：`"normal"`, `"t"`, `"skewt"`, `"ged"` |
| `criterion` | str | `"aic"` | 选择准则: `aic`, `bic`, `hqic`, `aicc` |
| `igarch` | bool | `False` | IGARCH 约束估计（与 EGARCH / GARCH-M 互斥） |
| `garch_m` | bool | `False` | GARCH-M（ARCH-in-mean，与 EGARCH / IGARCH 互斥） |
| `garch_m_form` | str | `"vol"` | GARCH-M 形式：`"vol"`, `"var"`, `"log"` |
| `ar_lags` | int/list | `None` | 均值方程 AR 滞后（仅 garch_m=True） |
| `exog` | array-like | `None` | 外生解释变量 |

**使用示例**：

```python
from Ts.TsModels import AutoGARCH
from Ts.TsSims import simulate_egarch, simulate_igarch, simulate_garch_m

# EGARCH 自动选择
data = simulate_egarch(n=300, p=1, q=1, o=1, seed=42).data
auto = AutoGARCH(data, p=(1,2), q=(1,2), o=(1,1), vol="EGARCH", criterion="aic")
result = auto.fit()

# IGARCH 自动选择
data = simulate_igarch(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1,2), q=(1,2), igarch=True, criterion="bic")
result = auto.fit()

# GARCH-M 自动选择
data = simulate_garch_m(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1,2), q=(1,2), garch_m=True, criterion="aic")
result = auto.fit()
```

### AutoModelResult

继承 `BaseModelResult` 的全部字段和方法，增加：

| 字段 | 类型 | 说明 |
|------|------|------|
| `best_result` | `BaseModelResult` | 最优模型的完整结果对象 |
| `best_order` | `tuple` | 最优参数组合 |
| `candidate_results` | `list` | 所有成功拟合的模型结果 |
| `candidate_orders` | `list` | 所有成功的参数组合 |
| `selection_criterion` | `str` | 使用的选择准则 |
| `.long_run_equilibrium()` | — | 委托给 `best_result.long_run_equilibrium()` |

### VAR

```python
VAR(data, lags=1, trend="c", cols=None, dates=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like (2-D) | — | 多变量时间序列，shape (nobs, k) |
| `lags` | int | `1` | 滞后阶数 (>= 1) |
| `trend` | str | `"c"` | 趋势设定：`"n"` (无), `"c"` (常数), `"ct"` (常数+趋势), `"ctt"` (常数+二次趋势) |
| `cols` | list of str | `None` | 纳入模型的变量列名，同时用于显示。DataFrame 时自动选列；ndarray 时纯命名。None 时 DataFrame 取全部列名，ndarray 自动生成 `"y0"`, `"y1"`, ... |

VAR 模型通过方程逐一 OLS 估计，封装 `statsmodels.tsa.vector_ar.var_model.VAR`。

```python
from Ts.TsModels import VAR

# 估计 VAR(2)
model = VAR(data_2d, lags=2, cols=["gnp", "m1"])
result = model.fit()
print(result.summary())

# 滞后阶数选择
order_info = VAR.select_order(data_2d, max_lags=8, criterion="aic")

# 脉冲响应函数（返回 IRFResult，含 .values / .lower / .upper / .summary() / .get()）
irf_result = result.irf(periods=10, orth=False)
print(irf_result.summary())          # Stata 风格表格
irf_data = irf_result.get("gnp", "m1")  # 提取单对数据

# 正交化 IRF
oirf_result = result.oirf(periods=10)

# IRF 图
fig, axes = result.plot_irf(periods=10, orth=True)

# 方差分解（返回 FEVDResult，含 Monte Carlo 置信区间）
fevd_result = result.fevd(periods=10)
print(fevd_result.summary())         # Stata 风格表格
fevd_data = fevd_result.get("gnp", "m1")  # 提取单对数据

# Granger 因果检验
gc = result.granger_causality(caused="gnp", causing="m1")
print(gc)  # 格式化表格输出 + 显著性星号
gc_all = result.granger_causality(kind="chi2")  # 无参数 → 全部两两检验
print(gc_all)

# 预测
pr = result.predict(start=result.nobs, end=result.nobs + 3)

# 单位根稳定性诊断
result.is_stable        # True = 平稳
result.plot_roots()     # 逆特征根单位圆图
```

| 方法 | 说明 |
|------|------|
| `VAR.select_order(data, max_lags, criterion)` | 静态方法：基于信息准则选最优滞后阶数 |
| `result.irf(periods, orth, alpha)` | 脉冲响应函数，返回 `IRFResult`（含 values/lower/upper/summary/get） |
| `result.oirf(periods, alpha)` | 正交化 IRF，等价于 `irf(orth=True)` |
| `result.fevd(periods, alpha, n_draws, seed)` | 预测误差方差分解，返回 `FEVDResult`（含 Monte Carlo CI / summary / get） |
| `result.plot_irf(periods, orth, alpha)` | 绘制 k x k IRF 子图矩阵，含置信带 |
| `result.granger_causality(caused, causing, kind)` | Granger 因果检验，返回 GrangerCausalityResult（无参数时全部两两检验）|
| `result.predict(start, end, ...)` | 统一预测接口，返回 `PredictResult`；不计算性能指标 |
| `result.long_run_equilibrium()` | 无条件均值向量（平稳且无时间趋势时返回 ``ndarray(k,)``） |

### SVAR

```python
SVAR(data, lags=1, A=None, B=None, C_lr=None, trend="c", cols=None, dates=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like (2-D) | — | 多变量时间序列，shape (nobs, k) |
| `lags` | int | `1` | 滞后阶数 (>= 1) |
| `A` | np.ndarray (k,k) | `None` | 短期约束矩阵 A，``np.nan`` = 自由参数 |
| `B` | np.ndarray (k,k) | `None` | 短期约束矩阵 B，``np.nan`` = 自由参数 |
| `C_lr` | np.ndarray (k,k) | `None` | 长期约束矩阵（Blanchard-Quah），零 = 零约束 |
| `trend` | str | `"c"` | 趋势设定：`"n"`, `"c"`, `"ct"`, `"ctt"` |
| `cols` | list of str | `None` | 变量列名 |

SVAR 在简化式 VAR 基础上施加识别约束，恢复结构性冲击。短期约束 (A/B)
通过 MLE (scipy BFGS) 估计，长期约束通过 Blanchard-Quah 闭式解计算。

```python
from Ts.TsModels import SVAR
from Ts.TsModels import VAR
import numpy as np

# 先拟合简化式 VAR 获取数据
data = np.random.randn(200, 2)

# === 短期约束 AB-model（递归识别 / Cholesky） ===
A = np.array([[1, 0], [np.nan, 1]])         # 下三角 A（1 个自由参数）
B = np.array([[np.nan, 0], [0, np.nan]])     # 对角 B（2 个自由参数）
svar_ab = SVAR(data, lags=2, A=A, B=B)
result_ab = svar_ab.fit()
print(result_ab.summary())                   # 显示 A/B 矩阵 + VAR 参数

# === 长期约束（Blanchard-Quah） ===
C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])  # C[0,1]=0 长期零约束
svar_lr = SVAR(data, lags=2, C_lr=C_lr)
result_lr = svar_lr.fit()

# === 结构性脉冲响应 (SIRF) ===
sirf_result = result_ab.irf(periods=10, orth=True)  # Theta_h = Psi_h * A^{-1} B
print(sirf_result.summary())

# === 继承全部 VARResult 方法 ===
result_ab.irf(periods=10, orth=False)        # 简化式 IRF
result_ab.fevd(periods=10)                   # 简化式 FEVD
result_ab.granger_causality()                # Granger 因果检验
result_ab.predict()                          # 预测
result_ab.is_stable                          # 稳定性
result_ab.plot_diagnostics()                 # 残差诊断图
```

| 方法 | 说明 |
|------|------|
| `SVAR.fit()` | 先拟合简化式 VAR，再估计结构参数 |
| `SVARResult.A` / `SVARResult.B` | 估计的结构矩阵 |
| `SVARResult.irf(periods, orth=True)` | 结构性脉冲响应函数 |
| `SVARResult.structural_residuals` | 结构冲击序列 (nobs, k) |
| `SVARResult.sigma_u` | 简化式残差协方差矩阵 |
| `SVARResult.summary()` | 含 A/B 矩阵的格式化输出 |

SVAR 不支持直接通过 ``svar A B`` 进行**过度识别检验** (LR test) ——
仅提供恰好识别模型的估计。如需检验过度识别限制的合理性，可在 Stata 中
使用 `svar` 命令或手动计算 LR 统计量。

### VECM

```python
VECM(data, lags=2, coint_rank=1, trend="c", cols=None, dates=None, missing="raise")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like (2-D) | — | 多变量时间序列，shape (nobs, k) |
| `lags` | int | `2` | VAR 水平滞后阶数 (>= 1) |
| `coint_rank` | int | `1` | 协整秩 (1 <= r < k) |
| `trend` | str | `"c"` | 趋势设定：`"n"` (无), `"c"` (常数), `"ct"` (线性趋势) |
| `cols` | list of str | `None` | 变量列名 |

VECM 通过 statsmodels VECM 采用 MLE 估计，包含误差修正项和短期动态。

```python
from Ts.TsModels import VECM
import numpy as np

# 生成协整数据
np.random.seed(42)
n = 300
x = np.cumsum(np.random.randn(n))
y = 2.0 * x + np.random.randn(n) * 0.5
z = 3.0 * x + np.random.randn(n) * 0.5
data = np.column_stack([z, y, x])

# 估计 VECM (coint_rank=2)
model = VECM(data, lags=2, coint_rank=2, trend="c", cols=["Z", "Y", "X"])
result = model.fit()
print(result.summary())

# 脉冲响应
irf = result.irf(periods=10, orth=True)

# 方差分解
fevd = result.fevd(periods=10)

# Granger 因果检验
gc = result.granger_causality(caused="Z", causing="Y", kind="chi2")

# 预测
pr = result.predict(start=result.nobs, end=result.nobs + 3)

# 残差诊断
result.plot_diagnostics()
result.test_residuals(lags=10)
```

## 依赖

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `statsmodels` (SARIMA / VAR / VECM 估计)
- `arch` (GARCH 估计)
- `TsPlots` (绘图风格)
- `TsTests` (残差诊断检验)

## 运行测试

```powershell
$env:PYTHONPATH = (Resolve-Path ..)
python -m pytest TsModels/tests -v
```

## 公共接口验证

```powershell
$env:PYTHONPATH = (Resolve-Path ..)
python -c "from Ts.TsModels import SARIMA, GARCH, VAR; print(SARIMA, GARCH, VAR)"
```

## ARIMAX、事件与政策效果

下面的例子同时演示带日期索引的控制变量、自动保存的未来默认路径、
多个预测情境以及事件设计：

```python
import numpy as np
import pandas as pd

from Ts.TsModels import EventSpec, SARIMA

rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=36, freq="MS")
all_dates = pd.date_range("2020-01-01", periods=42, freq="MS")
controls = pd.DataFrame(
    {
        "rate": np.sin(np.arange(42) / 6),
        "income": np.linspace(0, 1, 42),
    },
    index=all_dates,
)

policy_level = (np.arange(36) >= 18).astype(float)
y = pd.Series(
    2.0
    + 0.8 * controls.loc[dates, "rate"].to_numpy()
    + 1.5 * policy_level
    + rng.normal(scale=0.1, size=36),
    index=dates,
)

events = [
    EventSpec(
        "announcement",
        ["2021-04-01"],
        "pulse",
        date_rule="exact",
    ),
    EventSpec(
        "policy",
        ["2021-07-01"],
        "step",
        date_rule="exact",
    ),
    EventSpec(
        "implementation",
        ["2021-07-01"],
        "pulse",
        window=(-2, 2),
        reference=-1,
        date_rule="exact",
    ),
]

model = SARIMA(
    y,
    exog=controls,
    events=events,
    order=(0, 0, 0),
    trend="c",
)
fitted = model.fit()
print(fitted.params)
```

因为 `controls` 比 `y` 多六期，超出 `y` 末期的部分会自动保存为默认未来
外生变量。使用默认路径时不需要再次传入：

```python
default_forecast = fitted.predict(start=36, end=41)
print(default_forecast.mean)
```

也可以一次提供多个命名情境。此时返回结果同时包含自动保存的
`"default"` 路径以及用户指定的情境：

```python
future_dates = all_dates[36:]
baseline = controls.loc[future_dates].copy()
stress = baseline.copy()
stress["rate"] += 0.5

scenario_forecast = fitted.predict(
    start=36,
    end=41,
    future_exog={
        "baseline": baseline,
        "stress": stress,
    },
)
print(scenario_forecast.summary())
print(scenario_forecast["default"].mean)
print(scenario_forecast["baseline"].mean)
print(scenario_forecast["stress"].mean)
```

`policy_effect()` 只改变选定事件的设计列；普通外生变量和未选事件在事实与
反事实路径中保持相同。支持 Delta、联合参数模拟和参数化 Bootstrap：

```python
delta = fitted.policy_effect("policy", method="delta")
simulation = fitted.policy_effect(
    "policy",
    method="simulation",
    n_draws=2_000,
    seed=7,
)
bootstrap = fitted.policy_effect(
    "policy",
    method="bootstrap",
    n_draws=200,
    seed=7,
)

print(delta.summary())
print(simulation.cumulative_effect)
print(bootstrap.cumulative_lower, bootstrap.cumulative_upper)
```

这里报告的是给定模型、控制变量和事件设定下的条件效果。只有外生性、
无同期未控制冲击、模型设定正确以及反事实稳定等识别条件成立时，才能将其
解释为政策的因果效果。`PolicyEffectResult.identification_note` 会保留这一
限制。
