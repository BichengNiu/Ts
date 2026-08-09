# Ts/TsModels

时间序列模型估计工具包。提供 SARIMAX、GARCH、VAR、SVAR、VECM 的统一使用接口，结果对象与 TsPlots、TsMetrics、TsTests 衔接。

## 交互式帮助

公共模型、结果对象及其 demo 用法的参数、返回值和可执行样例均已写入
docstring。在 IPython/Jupyter 中输入 `?SARIMAX`（或 `SARIMAX?`），也可在
Python 中调用 `help(SARIMAX)` 查看完整帮助。

## Missing-data contract

All public model constructors and the VAR/VECM order-selection helpers accept
`missing="raise"` or `missing="drop"`. The default is `"drop"`, which removes
rows containing either `NaN` or infinite values and records their original
zero-based positions in `dropped_positions`. Endogenous variables, historical
exogenous regressors, multivariate columns, and dates are filtered jointly.
Use `missing="raise"` when any automatic sample change must fail immediately.
No model imputes raw observations.

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
├── _sarimax.py          # SARIMAX + SARIMAXResult
├── _distributed_lag.py # RDL 规格、联合 MLE 后端与派生结果
├── _garch_result.py    # GARCHResult (dataclass) + 参数缩放辅助函数
├── _garch_base.py      # _BaseVolModel — 参数验证 + fit 调度 + IGARCH MLE
├── _garch.py           # GARCH — 公开 API 入口
├── _var.py             # VAR + VARResult — 向量自回归
├── _svar.py            # SVAR + SVARResult — 结构向量自回归
├── _vecm.py            # VECM + VECMResult — 向量误差修正模型
├── _auto.py            # AutoSARIMAX + AutoGARCH + AutoModelResult
├── _compare.py         # compare_models — Stata 风格对比表格
├── tests/
│   ├── __init__.py
│   ├── test_base.py
│   ├── test_sarimax.py
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

from Ts.TsModels import SARIMAX, GARCH, VAR
from Ts.TsSims import simulate_sarima, simulate_garch

# AR(1) 估计
data = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42).data
model = SARIMAX(data, order=(1, 0, 0))
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
| `.plot_diagnostics()` | 标准化残差诊断图：第一行为标准化残差时间序列和直方图（含 Normality/Jarque–Bera），第二行为 ACF（含 White Noise/Ljung–Box）和 PACF；VAR/VECM 保持逐方程三列布局 |
| `.test_residuals(lags)` | 四项残差检验：白噪音 + 正态性 + Ljung-Box + Engle LM |
| `.params` | 估计参数 dict |
| `.aic` / `.bic` | 信息准则 |
| `.residuals` | 原始模型尺度上的有效残差序列 |
| `.standardized_residuals` | 标准化残差 `residuals / np.std(residuals, ddof=0)`；不另行减均值，多变量结果按方程分别计算 |

## 模型专属方法

| 模型 | 方法 | 说明 |
|------|------|------|
| `SARIMAX` / `SARIMAXResult` | `.predict(start, end, dynamic, alpha)` | 样本内预测与未来预测；性能评估由 `TsMetrics` 负责 |
| | `.arroots` | AR 多项式特征根 (ndarray) |
| | `.maroots` | MA 多项式特征根 (ndarray) |
| | `.is_stationary` | AR 多项式的全部根是否位于单位圆外 |
| | `.is_invertible` | MA 多项式的全部根是否位于单位圆外 |
| | `.ar_lags` / `.ma_lags` | 实际参与估计的非季节 AR/MA 滞后 |
| | `.fixed_params` | 因稀疏滞后设定而固定为 0 的系数 |
| | `.distributed_lag_coefficients` | 所有 RDL 多项式系数，包括固定为 0 的缺省阶 |
| | `.steady_state_gains` | 各输入长期增益及 delta-method 区间 |
| | `.weights(steps)` | 各输入的前 `steps` 个递归 impulse weights |
| | `.plot_impulse_response(steps, inputs, sample_weights)` | 绘制 RDL weights；传入基础模型 weights 时以柱表示 sample response、以线表示最终传递函数响应 |
| | `.feedback_test(lags, inputs)` | 对原始外生输入运行条件反馈 OLS 与因变量滞后联合 F 检验 |
| | `.residual_ccf_test(input_models, lags, inputs)` | 用显式输入 ARIMA 新息运行逐阶残差 CCF 与联合 S* 充分性检验 |
| | `.plot_roots(title)` | AR/MA 逆根单位圆图 |
| | `.cycle_period(seasonal=False)` | 检验 AR(2) 复根和平稳性条件；返回 `ARCycleResult`，周期以原始观测间隔计 |
| | `.likelihood_burn` / `.effective_nobs` | 状态/RDL 初始化所丢弃的期数与实际参与似然的样本量 |
| | `.level_intercept` | `trend="c"` 在未差分拟合响应尺度上的截距 `C`；自动完成状态截距到水平截距的变换。`log=True` 时结果位于自然对数响应尺度 |
| | `.level_intercept_inference(alpha)` | 拟合响应尺度截距 `C` 的 delta-method 标准误、统计量、p 值与区间；`log=True` 时均位于自然对数尺度 |
| | `.unconditional_log_variance` | `log=True` 平稳模型的无条件对数响应方差；通过状态空间离散 Lyapunov 方程精确计算，其他模型返回 `None` |
| | `.long_run_equilibrium()` | 零外生输入基线的长期均值；`log=True` 时返回 `exp(C + 0.5 * unconditional_log_variance)` 的原始尺度均值 |
| `GARCH` / `GARCHResult` | `.predict(start, end, alpha)` | 条件波动率预测；性能评估由 `TsMetrics` 负责 |
| | `.conditional_volatility` | 条件波动率 σ_t |
| | `.test_persistence()` | IGARCH 持久性 Wald 检验 |
| | `.long_run_equilibrium()` | 无条件方差 (协方差平稳 GARCH/GJR → float；EGARCH/IGARCH → None) |
| `VAR` / `VARResult` | `.irf(periods, orth, alpha)` | 脉冲响应函数，返回 `IRFResult`（含置信带） |
| | `.oirf(periods, alpha)` | 正交化脉冲响应函数，`irf(orth=True)` 的便捷封装 |
| | `.fevd(periods, alpha, n_draws, seed)` | 预测误差方差分解，返回 `FEVDResult`（含 Monte Carlo 置信带） |
| | `.plot_irf(periods, orth, alpha)` | IRF 图（含置信带） |
| | `.granger_causality(caused, causing, kind)` | Granger 因果检验（单对/联合/全部两两）——无参数时运行全部检验（等价于 Stata `vargranger`）|
| | `.predict(start, end, alpha)` | 多步预测与置信区间；性能评估由 `TsMetrics` 负责 |
| | `.is_stable` | 逆特征根是否全部在单位圆内 (bool) |
| | `.plot_roots(title)` | 逆特征根单位圆图 |
| | `.long_run_equilibrium()` | 无条件均值向量 (stable, trend="c"/"n" → ndarray(k,)；否则 None) |
| `SVAR` / `SVARResult` | `.irf(periods, orth=True)` | 结构性脉冲响应函数 (Θ_h = Ψ_h × A⁻¹B)，返回 `IRFResult` |
| | `.A` / `.B` | 估计的结构矩阵 (ndarray k×k) |
| | `.structural_residuals` | 结构冲击序列 (ndarray nobs×k) |
| | `.sigma_u` | 简化式残差协方差矩阵 (ndarray k×k) |
| `VECM` / `VECMResult` | `.irf(periods, orth, alpha)` | 脉冲响应函数，返回 `IRFResult` |
| | `.fevd(periods)` | 预测误差方差分解，返回 `FEVDResult` |
| | `.predict(start, end, alpha)` | 多步预测；当前不提供预测区间 |
| | `.granger_causality(...)` | Granger 因果检验 |
| | `.alpha` / `.beta` / `.gamma` | VECM 参数矩阵 |
| | `.sigma_u` | 残差协方差矩阵 (ndarray k×k) |
| | `.coint_rank` | 协整秩 |
| | `.is_stable` | 基于 companion matrix 的稳定性检查 (bool) |
| | `.plot_roots(title)` | VECM 逆特征根单位圆图 |

### 公共结果对象索引

下列结果类也是 `TsModels.__all__` 的正式公共接口。它们通常由模型方法返回，
不需要用户直接构造：

| 结果类 | 来源 | 说明 |
|--------|------|------|
| `BackcastResult` | `model.backcast()` | 样本前预测值、区间和负位置索引 |
| `ScenarioForecastResult` | `SARIMAXResult.predict()` | 多个未来外生变量情境的命名预测结果 |
| `RationalLagResult` | SARIMAX 的 RDL 派生结果 | 单个输入的多项式系数、权重、根与长期增益 |
| `VAROrderResult` | `VAR.select_order()` | VAR 滞后阶数选择指标与推荐阶数 |
| `VECMOrderResult` | `VECM.select_order()` | VECM 滞后阶数选择指标与推荐阶数 |

当前 GARCH 结果的 `.standardized_residuals` 也遵循共享定义，即除以残差的
整体标准差；逐期除以 `.conditional_volatility` 的条件标准化不在本次接口内。

### AR(2) 周期识别

对普通 AR(2)，直接调用 `result.cycle_period()`；对季节 AR(2)，调用
`result.cycle_period(seasonal=True)`。方法要求所选分量包含连续的第一、第二
个 AR 滞后。设估计系数为 \(\phi_1, \phi_2\)，只有完整 AR 多项式平稳且

\[
\phi_1^2 + 4\phi_2 < 0
\]

时才识别为阻尼周期。普通 AR(2) 的周期为

\[
T = \frac{2\pi}{\arccos\left(\phi_1/(2\sqrt{-\phi_2})\right)}.
\]

季节 AR(2) 的结果再乘季节长度 \(s\)，因此始终以原始观测间隔计。返回的
`ARCycleResult` 同时保留系数、判别式、复根条件、平稳性和 `period`；条件
不满足时 `identified=False` 且 `period=None`。

```python
diagnostic = result.cycle_period()
if diagnostic.identified:
    print(diagnostic.period)
```

## 性能评估与期间接口

预测性能指标、显式期间 OOS、滚动历史回测和模型性能排序统一由
`TsMetrics` 定义。`BaseModel.oos()` 与 `BaseModel.backtest()` 只是指向
`TsMetrics` 规范实现的便利方法；`PredictResult` 不保存实际值或性能指标。

位置型数据使用零基、闭区间位置：

```python
evaluation = model.oos(
    estimation_period=(0, 79),
    validation_period=(80, 99),
    method="lbfgs",
)
print(evaluation.metrics)
```

`method` 仅在显式指定时转交给模型的 `fit()`；省略或设为 `None` 会保留该模型
自己的默认拟合行为。不接受 `fit(method=...)` 的模型会明确报错，不静默忽略。

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

所有继承 `BaseModel` 的预测模型（SARIMAX、GARCH、VAR、VECM、SVAR 及 Auto 模型）
共享该接口。

### 多模型统一窗口 OOS 比较

多个模型需要在相同估计期和验证期比较全部误差指标时，使用 `TsMetrics`
提供的批量入口：

```python
from Ts.TsMetrics import evaluate_models_oos

report = evaluate_models_oos(
    {
        "AR(1)": ar1_model,
        "AR(2)": ar2_model,
    },
    estimation_period=(0, 79),
    validation_period=(80, 99),
    method="lbfgs",
    rank_by="rmse",
)

print(report.table)
print(report.ranking)
print(report.best_model)
```

`report.table` 包含 MAE、MSE、RMSE、MAPE、sMAPE、Theil U1、有效配对数
`n` 和排名；`report.evaluations` 保留每个模型的完整 `OOSResult`。批量比较
强制检查预测目标、估计期、验证期和实际观测一致，并拒绝验证期内的非有限
实际值或预测值，避免模型使用不同有效样本获得不公平排名。

批量 `method` 省略或设为 `None` 时，每个模型保持自己的默认值。显式指定时，
所有命名模型必须接受 `fit(method=...)`，否则在任何拟合开始前失败。SARIMAX
可选 `newton`、`nm`、`bfgs`、`lbfgs`、`powell`、`cg`、`ncg` 和
`basinhopping`。AutoSARIMAX 构造器的同名参数是模型搜索策略，不是似然优化器。

```python
model.oos(estimation_period, validation_period, alpha=0.05, method=None)

model.backtest(
    initial_window,
    horizon=1,
    step=1,
    window="expanding",
    window_size=None,
    alpha=0.05,
    on_error="raise",
)

model.backcast(steps, alpha=0.05)
```
### Backtesting：无未来信息泄漏的滚动起点回测

`backtest()` 在每个预测起点重新拟合模型。扩展窗口使用起点前的全部历史；滚动窗口仅使用最近 `window_size` 个观测。原模型及其已有 `result_` 不会被修改。
`window_size` 仅适用于 `window='rolling'`；扩展窗口传入该参数会明确报错，不会静默忽略。

```python
from Ts.TsModels import SARIMAX

model = SARIMAX(y, order=(1, 0, 0))

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
    window="rolling",
    window_size=80,
    on_error="record",
)
```

`BacktestResult.mean` 和 `.actual` 的形状为：

- 单变量：`(n_origins, horizon)`；
- 多变量：`(n_origins, horizon, n_series)`。

结果同时提供总体 `metrics`、逐预测期 `metrics_by_horizon`、逐变量 `metrics_by_series`。规范指标为 MAE、MSE、RMSE、MAPE、sMAPE、Theil U1 和有效配对数 `n`。`on_error='record'` 会将失败窗口保留为 NaN，并把起点和异常写入 `failures`；默认 `on_error='raise'` 会立即抛出异常。完整契约见 `TsMetrics/README.md`。

动态评估模型在连续 `N` 期预测窗口上的可靠性时，设置 `horizon=N`、
`step=1`，然后读取 `metrics_by_window`：

```python
expanding = model.backtest(
    initial_window=80,
    horizon=4,
    step=1,
    window="expanding",
)

dynamic = expanding.metrics_by_window
print(dynamic[["window_start", "window_end", "rmse", "mape", "n"]])
```

每行指标只评价该起点向前预测的完整 `N` 期；训练样本随后增加一期，预测
起点也前移一期。最后一个窗口结束于最新观测。对于 VAR、VECM 和 SVAR，
结果增加 `series` 列并逐变量计算，避免把不同量纲混成一个动态 RMSE。

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
| TsModels -> TsPlots | `.plot_fit()`/`.plot_diagnostics()` 复用序列与 ACF/PACF 图；RDL impulse response 复用 `plot_lag_response()`，残差 CCF 复用 `plot_correlogram()` |
| TsModels -> TsTests | `test_residuals()` 运行残差诊断；SARIMAX 组合独立的 `FeedbackTest` 与 `ResidualCCFTest` |
| TsSims -> TsModels | 验证脚本：TsSims 生成数据 -> TsModels 估计 -> 比较真实参数 |

## 模型参数

### SARIMAX

```text
SARIMAX(
    data,
    order=(0, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
    *,
    dates=None,
    exog=None,
    exog_names=None,
    events=None,
    missing="drop",
    distributed_lags=None,
    enforce_distributed_lag_stability=True,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like | — | 时间序列 |
| `order` | tuple | `(0,0,0)` | 非季节阶数 `(p,d,q)`；`p`、`q` 可为整数或实际参与估计的滞后列表 |
| `seasonal_order` | tuple | `(0,0,0,0)` | 季节阶数 `(P,D,Q,s)` |
| `trend` | str | `"c"` | 趋势设定：`"n"`, `"c"`, `"t"`, `"ct"` |
| `enforce_stationarity` | bool | `True` | 强制 AR 多项式平稳性 |
| `enforce_invertibility` | bool | `True` | 强制 MA 多项式可逆性 |
| `dates` | datetime-like | `None` | 数组输入的显式日期；`Series` 自动使用其 `DatetimeIndex` |
| `exog` | Series/DataFrame/array-like | `None` | 普通外生变量；命名 Series 或一维数组表示单个输入；带日期的 Series/DataFrame 可同时包含历史和未来行 |
| `exog_names` | sequence[str] | `None` | 数组和未命名 Series 的必填列名；命名 Series/DataFrame 使用自身名称 |
| `events` | sequence[EventSpec] | `None` | 脉冲、阶跃或事件窗口设计 |
| `missing` | `"raise"`/`"drop"` | `"drop"` | 内生和历史外生变量的联合缺失行策略；严格模式显式传入 `"raise"` |
| `distributed_lags` | `dict[str, RationalLagSpec]` | `None` | 以外生列名为键的 RDL/transfer-function 规格；支持多个输入 |
| `enforce_distributed_lag_stability` | bool | `True` | 约束并复核完整分母多项式稳定性 |

`fit()` 对普通 SARIMAX、RDL 和 `AutoSARIMAX` 的全部候选统一使用以下默认值：

```python
result = model.fit(
    method="bfgs",
    maxiter=500,
    cov_type="oim",
    require_convergence=True,
)
```

因此默认最多进行 500 次迭代、使用 observed-information covariance，并在
优化器未报告收敛时抛出 `RuntimeError`。调用者仍可显式覆盖任一参数；如需检查
未收敛结果，必须明确传入 `require_convergence=False`。优化器只控制似然函数的
数值搜索，不会改变模型规格。

稀疏滞后列表用于把未列出的中间阶系数严格固定为 0：

```python
# AR(3)，但固定 ar.L2 = 0
ar_result = SARIMAX(data, order=([1, 3], 0, 0)).fit()

# ARMA(1,3)，但固定 ma.L2 = 0
arma_result = SARIMAX(data, order=(1, 0, [1, 3])).fit()
```

`summary()` 会报告实际 AR/MA 滞后、固定为 0 的系数、根的最小模、
AR 多项式平稳性、MA 多项式可逆性，以及估计时是否强制这些条件。
对于 `d > 0` 或 `D > 0` 的模型，平稳性结论针对差分后的 AR 多项式，
不表示原始水平序列平稳。

### Rational distributed lags

```python
from Ts.TsModels import RationalLagSpec, SARIMAX

model = SARIMAX(
    y,
    exog=X,  # DataFrame columns: price, income, control
    order=(1, 0, 0),
    distributed_lags={
        "price": RationalLagSpec(
            numerator=(0, 2),      # omega.L1 = 0
            denominator=(1, 3),    # delta.L2 = 0
            delay=1,
        ),
        "income": RationalLagSpec(numerator=1, denominator=1),
    },
)
result = model.fit(maxiter=300)
```

整数阶数表示连续活动滞后：`numerator=2` 为 L0–L2，`denominator=2`
为 L1–L2。序列表示稀疏活动滞后，遗漏的中间阶严格固定为 0；这些是
传递多项式约束，不是最终 impulse weights 约束。未列入 `distributed_lags`
的 `control` 仍作为普通静态外生变量联合估计。

`RationalLagSpec` 默认 `initialization="auto"`。有限分布滞后会自动从似然中
排除没有完整输入历史的期数，并再留出 ARMA 扰动递推所需的初始化深度；
回归起点和扰动起点也只使用同一有效样本。含递归分母的无限分布滞后无法由
样本识别无限期的样本前输入，`auto` 因而采用首期输入水平的稳态假设。
`summary()` 会明确显示解析后的策略、`Likelihood Burn` 和有效样本量。
如有明确的样本前知识，仍可显式选择 `initialization="zero"` 或
`initialization="steady_state"`，其原有数值含义不变。

教材式有限 LTF 不需要自行生成 16 个滞后列或常数列：

```python
result = SARIMAX(
    sales.iloc[:140],
    exog=leading_indicator.iloc[:140].rename("leading_indicator"),
    order=(1, 0, 0),
    trend="c",
    distributed_lags={
        "leading_indicator": RationalLagSpec(numerator=15, denominator=0)
    },
).fit()
```

这里无需重复传入拟合控制：默认的 `cov_type="oim"` 与教材表格的
observed-information 标准误口径一致，并默认要求优化器收敛。

这里 `trend="c"` 对应拟合响应尺度上的水平截距。Statsmodels 底层为了
状态空间递推估计的是状态截距 `c`；Ts 在 `result.level_intercept` 和
`summary()` 中自动报告 `C = c / A(1)`，并通过
`level_intercept_inference()` 对完整协方差矩阵做 delta-method 变换。
当模型使用 `log=True` 时，`C` 及其推断保留在自然对数响应尺度，摘要标记为
`Log-response Intercept C`。`exp(C)` 是原始尺度中位数，不是均值；
`unconditional_log_variance` 通过拟合状态空间的离散 Lyapunov 方程精确求得
对数响应的平稳方差，`long_run_equilibrium()` 返回
`exp(C + 0.5 * unconditional_log_variance)`。普通外生变量、事件和 RDL 输入
均按零输入基线解释；逐期预测仍使用各预测期自己的方差修正。
底层原始 `params["intercept"]` 仍保留，便于复核似然、协方差和优化器参数顺序；
用户不应再向 `exog` 手工加入常数列。

`result.distributed_lags["price"]` 提供结构化单输入结果；
`distributed_lag_coefficients` 自动列出估计值、标准误、p 值和 fixed 标记；
`steady_state_gains` 自动计算 `sum(omega) / (1 - sum(delta))` 及区间；
`weights(steps)` 返回递归权重。未来预测必须为每个原始外生列提供连续路径，
日期模型可传带相同未来日期的 DataFrame，无日期模型可按位置传二维数组。

RDL 的 impulse response 就是 `weights(steps)`：传递函数对单位脉冲在各个
time lag 上的响应。绘图不重新计算权重。默认将当前模型 weights 画成柱；若提供
preliminary finite-lag 模型的 `sample_weights`，则 sample weights 为柱，当前
rational transfer-function weights 为实线：

```python
# 所有 RDL 输入按拟合顺序分面
fig, axes = result.plot_impulse_response(steps=20)

# 仅绘制 price；等价于绘制 result.distributed_lags["price"].weights(20)
fig, ax = result.plot_impulse_response(20, inputs="price")

# Figure C5.14/C5.15 风格：基础 LTF weights 为柱，最终 RDL weights 为线
fig, axes = final_result.plot_impulse_response(
    steps=16,
    sample_weights=preliminary_result.weights(16),
)
```

对随机性输入，还可检查过去的模型输出是否反馈到当前输入：

```python
feedback = result.feedback_test(lags=4)
print(feedback.summary())
print(feedback.tests)
```

每个输入方程控制自身及其他全部输入的 1–K 阶滞后，并对 `y.L1`–`y.LK`
执行联合 F 检验。显著结果是条件预测反馈证据，不等同于结构性因果证明。

传递函数估计后，用显式拟合的输入 ARIMA 模型检查是否仍有遗漏的输入动态：

```python
input_models = {
    "price": SARIMAX(price, order=(1, 0, 0), trend="c").fit(),
    "income": AutoSARIMAX(income).fit(),
}
residual_ccf = result.residual_ccf_test(input_models, lags=12)
print(residual_ccf.tests)
print(residual_ccf.get("price").correlations)
fig, axes = residual_ccf.plot_test()
```

输入模型必须是对当前 RDL 同一历史输入和同一日历拟合的、已收敛的单变量
input-only `SARIMAXResult`，或其 `AutoModelResult`。接口不在诊断内部猜测
ARIMA 阶数。不同模型 burn 后的新息按共同样本末端对齐；每个输入的
`df = K + 1 - m` 自动从其活动 RDL numerator/denominator 参数计算。

### GARCH

```python
GARCH(
    data,
    p=1,
    q=1,
    o=0,
    vol="GARCH",
    mean="Constant",
    dist="normal",
    garch_m=False,
    garch_m_form="vol",
    ar_lags=None,
    exog=None,
    dates=None,
    missing="drop",
    distributed_lags=None,
    enforce_distributed_lag_stability=True,
)
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

### compare_models — 参数估计结果对比

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

`compare_models()` 比较已拟合模型的参数和显著性，不衡量样本外预测性能。
比较统一窗口下的预测误差应使用 `TsMetrics.evaluate_models_oos()`。

## 自动最优参数选择

`AutoSARIMAX` 和 `AutoGARCH` 通过网格搜索自动选择最优模型阶数。

```python
from Ts.TsModels import AutoSARIMAX, AutoGARCH
from Ts.TsSims import simulate_sarima, simulate_garch

# 自动选择最优 SARIMAX 阶数
data = simulate_sarima(n=200, order=(1, 0, 0), ar=[0.7], seed=42).data
auto = AutoSARIMAX(data, p=(0, 3), d=(0, 1), q=(0, 3), criterion="aic")
result = auto.fit()
print(result.summary())
# 显示: 搜索方式, 选择准则, 最优阶数, 成功/尝试模型数, 最优模型参数表

# 自动选择最优 GARCH 阶数
data = simulate_garch(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1, 4), q=(0, 4), criterion="bic")
result = auto.fit()

# 最优模型结果可通过 best_result 访问
best = result.best_result  # SARIMAXResult 或 GARCHResult
```

### AutoSARIMAX

```text
AutoSARIMAX(
    data,
    p=(0, 3),
    d=(0, 1),
    q=(0, 3),
    P=(0, 1),
    D=(0, 1),
    Q=(0, 1),
    s=0,
    trend="c",
    criterion="aic",
    method="grid",
    *,
    dates=None,
    exog=None,
    exog_names=None,
    events=None,
    enforce_stationarity=True,
    enforce_invertibility=True,
    missing="drop",
    log=False,
    distributed_lags=None,
    enforce_distributed_lag_stability=True,
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p`, `d`, `q` | `(min, max)` | `(0,3)`, `(0,1)`, `(0,3)` | 非季节阶数范围 |
| `P`, `D`, `Q` | `(min, max)` | `(0,1)` | 季节阶数范围（s=0 时忽略） |
| `s` | int | `0` | 季节周期（0=无季节性） |
| `criterion` | str | `"aic"` | 选择准则: `aic`, `bic`, `hqic`, `aicc` |
| `dates` | datetime-like | `None` | 严格观测日期 |
| `exog` | Series/DataFrame/array-like | `None` | 每个候选模型共享的普通外生变量；单输入可直接传命名 Series 或一维数组 |
| `exog_names` | sequence[str] | `None` | 数组和未命名 Series 的必填列名 |
| `events` | sequence[EventSpec] | `None` | 每个候选模型共享的事件设计 |
| `enforce_stationarity` | bool | `True` | 所有候选模型是否强制平稳性 |
| `enforce_invertibility` | bool | `True` | 所有候选模型是否强制可逆性 |
| `missing` | `"raise"`/`"drop"` | `"drop"` | 搜索前统一处理内生与外生缺失行；严格模式显式传入 `"raise"` |
| `log` | bool | `False` | 是否让所有候选模型拟合响应变量的自然对数；这是固定设置，不是搜索维度。输入必须严格为正，预测自动返回原尺度并进行对数正态均值偏差修正 |
| `distributed_lags` | `dict[str, RationalLagSpec]` | `None` | 所有候选共享的固定 RDL 规格；不自动搜索其阶数 |
| `enforce_distributed_lag_stability` | bool | `True` | 所有候选统一使用的分母稳定性规则 |

对于严格为正的响应变量，可统一在对数尺度搜索候选阶数，同时让最终结果和
预测自动返回原尺度：

```python
auto = AutoSARIMAX(positive_data, log=True)
result = auto.fit()
assert result.log is True
```

### AutoGARCH

```python
AutoGARCH(
    data,
    p=(1, 4),
    q=(0, 4),
    o=(0, 0),
    vol="GARCH",
    mean="Constant",
    dist="normal",
    criterion="aic",
    method="grid",
    igarch=False,
    garch_m=False,
    garch_m_form="vol",
    ar_lags=None,
    exog=None,
    dates=None,
    missing="drop",
)
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
auto = AutoGARCH(data, p=(1, 2), q=(1, 2), o=(1, 1), vol="EGARCH", criterion="aic")
result = auto.fit()

# IGARCH 自动选择
data = simulate_igarch(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1, 2), q=(1, 2), igarch=True, criterion="bic")
result = auto.fit()

# GARCH-M 自动选择
data = simulate_garch_m(n=300, p=1, q=1, seed=42).data
auto = AutoGARCH(data, p=(1, 2), q=(1, 2), garch_m=True, criterion="aic")
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
| `log` | `bool` | 所选模型是否在响应变量的自然对数尺度拟合 |
| `level_intercept` | `float`/`None` | 委托所选 SARIMAX 模型的拟合响应尺度截距；`log=True` 时位于自然对数响应尺度 |
| `.level_intercept_inference(alpha)` | `dict`/`None` | 委托所选 SARIMAX 模型的截距 delta-method 推断 |
| `unconditional_log_variance` | `float`/`None` | 委托所选 SARIMAX 模型的无条件对数响应方差 |
| `.long_run_equilibrium()` | — | 委托给 `best_result.long_run_equilibrium()` |
| `.cycle_period(seasonal=False)` | `ARCycleResult` | 委托给 `AutoSARIMAX` 选中的 `best_result` |

### VAR

```python
VAR(data, lags=1, trend="c", cols=None, dates=None, missing="drop")
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
print(irf_result.summary())  # Stata 风格表格
irf_data = irf_result.get("gnp", "m1")  # 提取单对数据

# 正交化 IRF
oirf_result = result.oirf(periods=10)

# IRF 图
fig, axes = result.plot_irf(periods=10, orth=True)

# 方差分解（返回 FEVDResult，含 Monte Carlo 置信区间）
fevd_result = result.fevd(periods=10)
print(fevd_result.summary())  # Stata 风格表格
fevd_data = fevd_result.get("gnp", "m1")  # 提取单对数据

# Granger 因果检验
gc = result.granger_causality(caused="gnp", causing="m1")
print(gc)  # 格式化表格输出 + 显著性星号
gc_all = result.granger_causality(kind="chi2")  # 无参数 → 全部两两检验
print(gc_all)

# 预测
pr = result.predict(start=result.nobs, end=result.nobs + 3)

# 单位根稳定性诊断
result.is_stable  # True = 平稳
result.plot_roots()  # 逆特征根单位圆图
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
SVAR(
    data,
    lags=1,
    A=None,
    B=None,
    C_lr=None,
    trend="c",
    cols=None,
    dates=None,
    missing="drop",
)
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
A = np.array([[1, 0], [np.nan, 1]])  # 下三角 A（1 个自由参数）
B = np.array([[np.nan, 0], [0, np.nan]])  # 对角 B（2 个自由参数）
svar_ab = SVAR(data, lags=2, A=A, B=B)
result_ab = svar_ab.fit()
print(result_ab.summary())  # 显示 A/B 矩阵 + VAR 参数

# === 长期约束（Blanchard-Quah） ===
C_lr = np.array([[np.nan, 0], [np.nan, np.nan]])  # C[0,1]=0 长期零约束
svar_lr = SVAR(data, lags=2, C_lr=C_lr)
result_lr = svar_lr.fit()

# === 结构性脉冲响应 (SIRF) ===
sirf_result = result_ab.irf(periods=10, orth=True)  # Theta_h = Psi_h * A^{-1} B
print(sirf_result.summary())

# === 继承全部 VARResult 方法 ===
result_ab.irf(periods=10, orth=False)  # 简化式 IRF
result_ab.fevd(periods=10)  # 简化式 FEVD
result_ab.granger_causality()  # Granger 因果检验
result_ab.predict()  # 预测
result_ab.is_stable  # 稳定性
result_ab.plot_diagnostics()  # 残差诊断图
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
VECM(data, lags=2, coint_rank=1, trend="c", cols=None, dates=None, missing="drop")
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
- `statsmodels` (SARIMAX / VAR / VECM 估计)
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
python -c "from Ts.TsModels import SARIMAX, GARCH, VAR; print(SARIMAX, GARCH, VAR)"
```

## SARIMAX、事件与政策效果

下面的例子同时演示带日期索引的控制变量、自动保存的未来默认路径、
多个预测情境以及事件设计：

```python
import numpy as np
import pandas as pd

from Ts.TsModels import EventSpec, SARIMAX

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

model = SARIMAX(
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
