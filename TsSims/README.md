# Ts/TsSims

时间序列模拟工具包。提供 SARIMA、GARCH、协整系统和 TS/DS 过程的合成数据生成，结果以结构化对象封装。

## 模块结构

```
TsSims/
├── __init__.py         # 统一导出接口
├── _base.py            # BaseSimResult — 所有结果的公共基类
├── _sarima.py          # SARIMA 模拟 + SimSARIMAResult
├── _garch.py           # GARCH / IGARCH 模拟
├── _garch_result.py    # SimGARCHResult — GARCH 族结果容器
├── _garch_core.py      # 共享模拟引擎和辅助函数
├── _garch_ext.py       # GJR-GARCH / EGARCH / GARCH-M 模拟
├── _cointegration.py   # 协整系统模拟 + SimCointegratedResult
├── _ts_ds.py           # TS/DS 模拟 + SimTSDSResult
├── tests/              # 单元测试
└── README.md
```

## 快速开始

```python
from Ts.TsSims import (
    simulate_sarima,
    simulate_garch,
    simulate_gjr_garch,
    simulate_egarch,
    simulate_garch_m,
)
from Ts.TsSims import simulate_cointegrated

# AR(1) 过程
r = simulate_sarima(n=100, order=(1, 0, 0), ar=[0.5], seed=42)
r.plot()
print(r.summary())

# ARCH(1) 过程 (GARCH with q=0)
r = simulate_garch(n=200, p=1, q=0, omega=0.4, alpha=[0.5], seed=42)
r.plot()
print(r.summary())

# GARCH(1,1) 过程
r = simulate_garch(n=200, p=1, q=1, omega=0.2, alpha=[0.3], beta=[0.5], seed=42)
df = r.to_dataframe()

# GJR-GARCH(1,1,1) — 杠杆效应
r = simulate_gjr_garch(
    n=300, p=1, q=1, o=1, omega=0.05, alpha=[0.10], gamma=[0.15], beta=[0.70], seed=42
)

# EGARCH(1,1,1) — 对数方差建模
r = simulate_egarch(
    n=300, p=1, q=1, o=1, omega=0.0, alpha=[0.20], gamma=[0.10], beta=[0.30], seed=42
)

# GARCH-M(1,1) — 波动率进入均值
r = simulate_garch_m(
    n=300, p=1, q=1, omega=0.10, alpha=[0.20], beta=[0.60], garch_m_kappa=0.20, seed=42
)
```

## 结果对象

三个结果类 (`SimSARIMAResult`, `SimGARCHResult`, `SimTSDSResult`) 均继承
`BaseSimResult`，提供统一接口：

| 方法 | 返回 | 说明 |
|------|------|------|
| `.get_data()` | `pd.Series` | 生成的时间序列 |
| `.get_params()` | `dict` | 所有模拟参数（深拷贝） |
| `.plot()` | `(fig, ax)` | 时间序列图，调用 TsPlots.style 统一风格 |

`SimGARCHResult` 额外提供：

| 方法 | 返回 | 说明 |
|------|------|------|
| `.to_dataframe()` | `pd.DataFrame` | data, errors, volatility 三列 |

## `simulate_sarima` — SARIMA 过程

```python
simulate_sarima(
    n=200,
    order=(1, 0, 0),              # (p, d, q)
    seasonal_order=(0, 0, 0, 0),  # (P, D, Q, s)
    ar=None, ma=None,              # 非季节系数列表（None 则用默认值）
    seasonal_ar=None, seasonal_ma=None,
    const=0.0, sigma2=1.0,
    seed=None, burn=100,
) -> SimSARIMAResult
```

系数支持两种传入方式：
- 仅指定 `order=(1,1,0)`，自动使用默认 AR 系数 `[0.5]`
- 显式传入 `ar=[0.7]`，覆盖默认值

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数（burn-in 后） |
| `order` | tuple | `(1,0,0)` | 非季节阶数 `(p, d, q)` |
| `seasonal_order` | tuple | `(0,0,0,0)` | 季节阶数 `(P, D, Q, s)` |
| `ar` | list | None | AR 系数 `[phi1, phi2, ...]` |
| `ma` | list | None | MA 系数 `[theta1, theta2, ...]` |
| `seasonal_ar` | list | None | 季节 AR 系数 `[Phi1, ...]` |
| `seasonal_ma` | list | None | 季节 MA 系数 `[Theta1, ...]` |
| `const` | float | `0.0` | 常数项 |
| `sigma2` | float | `1.0` | 新息方差 |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

## `simulate_garch` — GARCH(p,q) 过程

处理纯 ARCH（q = 0）和 GARCH（q >= 1）两种过程。

```python
simulate_garch(
    n=200, p=1, q=1,
    omega=0.4, alpha=None, beta=None,  # 方差方程参数
    mean_model="constant",
    mean_ar=None, mean_const=0.0,
    dist="normal", dist_params=None,
    seed=None, burn=100,
) -> SimGARCHResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数 |
| `p` | int | `1` | ARCH 阶数 |
| `q` | int | `1` | GARCH 阶数，`q=0` 表示纯 ARCH(p) |
| `omega` | float | `0.4` | 方差方程常数项 |
| `alpha` | list | None | ARCH 系数 `[alpha1, ...]` |
| `beta` | list | None | GARCH 系数 `[beta1, ...]` |
| `mean_model` | str | `"constant"` | 均值方程类型 |
| `mean_const` | float | `0.0` | 均值常数 |
| `mean_ar` | list | None | 均值 AR 系数 |
| `dist` | str | `"normal"` | 新息分布：`"normal"` 或 `"t"` |
| `dist_params` | dict | None | 分布参数，如 `{"df": 5}` |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

## `simulate_igarch` — IGARCH(p,q) 过程

生成满足 sum(alpha) + sum(beta) = 1 约束的 IGARCH 数据。约束通过
自动调整最后一个 beta 系数实现。

```python
simulate_igarch(
    n=200, p=1, q=1,
    omega=0.10, alpha=None, beta=None,
    mean_const=0.0, dist="normal",
    seed=None, burn=100,
) -> SimGARCHResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数 |
| `p` | int | `1` | ARCH 阶数 (>= 1) |
| `q` | int | `1` | GARCH 阶数 (>= 1) |
| `omega` | float | `0.10` | 方差方程常数项 |
| `alpha` | list | None | ARCH 系数 `[alpha1, ...]`，默认 `[0.2]*p` |
| `beta` | list | None | GARCH 系数 `[beta1, ...]`，默认 `[0.5]*q`；最后一个自动调整以满足约束 |
| `mean_const` | float | `0.0` | 均值常数 |
| `dist` | str | `"normal"` | 新息分布：`"normal"` 或 `"t"` |
| `dist_params` | dict | None | 分布参数，如 `{"df": 5}` |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

```python
from Ts.TsSims import simulate_igarch

# IGARCH(1,1): alpha=0.3, beta auto-adjusted to 0.7
r = simulate_igarch(n=300, p=1, q=1, omega=0.05, alpha=[0.30], seed=42)
params = r.get_params()
# params["alpha"] = [0.30], params["beta"] = [0.70]
assert sum(params["alpha"]) + sum(params["beta"]) == 1.0
r.plot()
```

## `simulate_gjr_garch` — GJR-GARCH(p,o,q) 过程

生成带有杠杆效应的非对称 GARCH 数据。负面冲击对波动率的影响大于正面冲击。

```python
simulate_gjr_garch(
    n=200, p=1, q=1, o=1,
    omega=0.10, alpha=None, gamma=None, beta=None,
    mean_model="constant", mean_const=0.0,
    dist="normal", dist_params=None,
    seed=None, burn=100,
) -> SimGARCHResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数 |
| `p` | int | `1` | ARCH 阶数 |
| `q` | int | `1` | GARCH 阶数，`q=0` 表示纯 ARCH with leverage |
| `o` | int | `1` | 非对称 (GJR) 阶数，`o=0` 退化为标准 GARCH |
| `omega` | float | `0.10` | 方差方程常数项 |
| `alpha` | list | None | ARCH 系数，默认 `[0.10]*p` |
| `gamma` | list | None | 杠杆系数 `[gamma1, ...]`，默认 `[0.10]*o` |
| `beta` | list | None | GARCH 系数，默认 `[0.70]*q` |
| `mean_model` | str | `"constant"` | 均值方程类型 |
| `mean_const` | float | `0.0` | 均值常数 |
| `dist` | str | `"normal"` | 新息分布：`"normal"` 或 `"t"` |
| `dist_params` | dict | None | 分布参数，如 `{"df": 5}` |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

## `simulate_egarch` — EGARCH(p,o,q) 过程

生成指数 GARCH 数据，通过对数方差建模天然保证方差为正。

```python
simulate_egarch(
    n=200, p=1, q=1, o=1,
    omega=0.0, alpha=None, gamma=None, beta=None,
    mean_model="constant", mean_const=0.0,
    dist="normal", dist_params=None,
    seed=None, burn=100,
) -> SimGARCHResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数 |
| `p` | int | `1` | ARCH (magnitude) 阶数 |
| `q` | int | `1` | GARCH (persistence) 阶数 |
| `o` | int | `1` | 非对称 (leverage) 阶数，`o=0` 为对称 EGARCH |
| `omega` | float | `0.0` | 对数方差方程常数项 |
| `alpha` | list | None | 幅度系数，默认 `[0.15]*p` |
| `gamma` | list | None | 符号系数，默认 `[0.05]*o` |
| `beta` | list | None | 持续性系数，默认 `[0.30]*q` |
| `mean_model` | str | `"constant"` | 均值方程类型 |
| `mean_const` | float | `0.0` | 均值常数 |
| `dist` | str | `"normal"` | 新息分布：`"normal"` 或 `"t"` |
| `dist_params` | dict | None | 分布参数，如 `{"df": 5}` |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

## `simulate_garch_m` — GARCH-M (ARCH-in-Mean) 过程

生成 GARCH-in-Mean 数据，条件波动率进入均值方程。

```python
simulate_garch_m(
    n=200, p=1, q=1,
    omega=0.10, alpha=None, beta=None,
    garch_m_kappa=0.20, garch_m_form="vol",
    mean_model="constant", mean_const=0.0,
    dist="normal", dist_params=None,
    seed=None, burn=100,
) -> SimGARCHResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数 |
| `p` | int | `1` | ARCH 阶数 |
| `q` | int | `1` | GARCH 阶数，`q=0` 表示纯 ARCH-M |
| `omega` | float | `0.10` | 方差方程常数项 |
| `alpha` | list | None | ARCH 系数，默认 `[0.20]*p` |
| `beta` | list | None | GARCH 系数，默认 `[0.60]*q` |
| `garch_m_kappa` | float | `0.20` | ARCH-in-mean 系数 |
| `garch_m_form` | str | `"vol"` | 均值方程中 sigma 形式：`"vol"`, `"var"`, `"log"` |
| `mean_model` | str | `"constant"` | 均值方程类型 |
| `mean_const` | float | `0.0` | 均值常数 |
| `dist` | str | `"normal"` | 新息分布：`"normal"` 或 `"t"` |
| `dist_params` | dict | None | 分布参数，如 `{"df": 5}` |
| `seed` | int | None | 随机种子 |
| `burn` | int | `100` | 预热期数 |

## 绘图集成

所有 `plot()` 方法内部调用 `TsPlots.plot_series()` 和 `TsPlots.style` 模块的常量（`DEFAULT_PALETTE`、`style_axes()`），确保与项目中其他图表风格一致。

## 依赖

- `numpy`, `pandas`, `scipy`, `matplotlib`
- `statsmodels` (SARIMA 依赖 `ArmaProcess`)
- `TsPlots` (绘图风格)

## `simulate_cointegrated` — 协整多变量过程

通过 VECM 表示生成 k 维协整时间序列：

```
Delta Y_t = alpha @ beta.T @ Y_{t-1} + epsilon_t
```

```python
simulate_cointegrated(
    n=200, k=2, coint_rank=1,
    alpha=None, beta=None,
    sigma=1.0,
    seed=None, burn=100,
) -> SimCointegratedResult
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | `200` | 生成观测数（burn-in 后） |
| `k` | int | `2` | 变量数 (>= 2) |
| `coint_rank` | int | `1` | 协整秩 (1 <= r < k) |
| `alpha` | ndarray \| None | `None` | 调整系数 (k, r)，默认 -0.5 * I_r |
| `beta` | ndarray \| None | `None` | 协整向量 (k, r)，默认 [I_r; 0] |
| `sigma` | float | `1.0` | 新息标准差 |
| `seed` | int \| None | `None` | 随机种子 |
| `burn` | int | `100` | 预热期数 |

```python
from Ts.TsSims import simulate_cointegrated
import numpy as np

# k=2, r=1: 两个变量，一个协整关系（spread 平稳）
alpha = np.array([[-0.3], [0.0]])
beta = np.array([[1.0], [-1.0]])
r = simulate_cointegrated(n=500, k=2, coint_rank=1, alpha=alpha, beta=beta, seed=42)
df = r.get_data()
r.plot()
```

`SimCointegratedResult` 继承 `BaseSimResult`，额外提供：

| 方法 | 返回 | 说明 |
|------|------|------|
| `.get_data()` | `pd.DataFrame` | 多变量时间序列，列名 `"y0"`, `"y1"`, ... |
| `.get_params()` | `dict` | 所有模拟参数（深拷贝） |
| `.summary()` | `str` | 参数总结报告 |
| `.plot()` | `(fig, axes)` | 每变量一个子图，调用 TsPlots 统一风格 |

## 示例

```python
import numpy as np
from Ts.TsSims import (
    simulate_sarima,
    simulate_garch,
    simulate_gjr_garch,
    simulate_egarch,
    simulate_garch_m,
)

# SARIMA(1,1,1) — 带漂移的差分ARMA
r = simulate_sarima(
    n=200,
    order=(1, 1, 1),
    ar=[0.3],
    ma=[0.5],
    const=0.1,
    seed=123,
)
print(r.summary())
r.plot()

# ARCH(2) — GARCH with q=0
r = simulate_garch(
    n=500,
    p=2,
    q=0,
    omega=0.2,
    alpha=[0.3, 0.2],
    seed=42,
)
r.plot()

# GARCH(1,1) — 厚尾 Student's t 新息
r = simulate_garch(
    n=300,
    p=1,
    q=1,
    omega=0.1,
    alpha=[0.2],
    beta=[0.7],
    dist="t",
    dist_params={"df": 5},
    seed=99,
)
r.plot()

# GJR-GARCH(1,1,1) — 杠杆效应
r = simulate_gjr_garch(
    n=300,
    p=1,
    q=1,
    o=1,
    omega=0.05,
    alpha=[0.10],
    gamma=[0.15],
    beta=[0.70],
    seed=42,
)
r.plot()

# EGARCH(1,1,1) — 对数方差
r = simulate_egarch(
    n=300,
    p=1,
    q=1,
    o=1,
    omega=0.0,
    alpha=[0.20],
    gamma=[0.10],
    beta=[0.30],
    seed=42,
)
r.plot()

# GARCH-M(1,1) — ARCH-in-Mean
r = simulate_garch_m(
    n=300,
    p=1,
    q=1,
    omega=0.10,
    alpha=[0.20],
    beta=[0.60],
    garch_m_kappa=0.20,
    seed=42,
)
r.plot()
```
