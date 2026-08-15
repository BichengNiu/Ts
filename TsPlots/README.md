# Ts/TsPlots

适用于时间序列计量经济学学习的 Python 绘图工具包。提供统一的字体、配色和坐标轴风格，
支持时间序列折线图、散点图、自相关函数图、预计算相关图和相关矩阵热图。

## 交互式帮助

公共接口的参数、返回值和可执行样例均已写入 docstring。在 IPython/Jupyter
中输入 `?plot_series`（或 `plot_series?`），也可在 Python 中调用
`help(plot_series)` 查看完整帮助。

## 模块结构

```
TsPlots/
├── __init__.py   # 统一导出所有公开接口
├── style.py      # 共享样式：字体、调色板、辅助函数
├── ts_plot.py    # 时间序列绘图：plot_series
├── sc_plot.py    # 散点图绘图：plot_scatter
├── acf_plot.py   # 相关图：plot_acf, plot_pacf, plot_correlogram
├── lag_plot.py   # 滞后柱线响应图：plot_lag_response
└── matrix_plot.py # 相关矩阵热图：plot_correlation_matrix
```

## 安装 / 导入

将 `TsPlots/` 目录放在工作目录下，随后直接导入：

```python
from Ts.TsPlots import (
    plot_series, plot_scatter, plot_acf, plot_pacf,
    plot_correlogram, plot_correlation_matrix, plot_lag_response,
)
```

依赖：`matplotlib`、`numpy`、`pandas`、`statsmodels`（均为标准数据分析环境）。

---

## `plot_series` — 时间序列折线图

```text
from Ts.TsPlots import plot_series

fig, result = plot_series(data, x=None, y=None, *, facet=True, ...)
```

### 接受的输入格式

| 输入类型 | 说明 |
|----------|------|
| `pd.DataFrame` | 每列为一条系列；索引或指定列为时间轴 |
| `pd.Series` | 单条系列；索引为时间轴 |
| `dict` | `{标签: 值数组}` 形式的多系列 |
| 1D array | 单条系列，自动生成 0-based 时间轴 |
| 2D array | 每列为一条系列 |

### 常用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | 见上表 | — | 数据 |
| `x` | str / array | `None` | 时间列名或时间值数组；`None` 则用索引 |
| `y` | str / list | `None` | 仅绘制指定列（DataFrame） |
| `title` | str | `None` | 图表标题 |
| `xtitle` | str | `None` | x 轴标签（默认自动检测） |
| `ytitle` | str | `"Value"` | y 轴标签 |
| `unit` | str | `None` | y 轴单位，显示为「（单位：XX）」 |
| `ymin` | float / None | `None` | y 轴下限；`None` 为自动 |
| `freq` | str | `None` | datetime 轴频率：`'day'` / `'week'` / `'month'` / `'quarter'` / `'year'` |
| `year_ruler` | bool | `False` | 月度数据的严格刻度：只标数据中实际出现的 3 的倍数月份（标签为「3月」，不旋转），并在 x 轴下方为每个出现的年份绘制年度标尺；x 轴两端各留 10 天。与 `freq` 互斥使用，替代其 locator |
| `xtick_step` | int | `None` | 数值 x 轴刻度间隔 |
| `max_ticks` | int | `12` | 自动刻度上限 |
| `linewidth` | float | `3` | 线宽 |
| `markersize` | float | `7` | 标记大小 |
| `colors` | list | `None` | 覆盖默认配色（按系列顺序） |
| `labels` | list | `None` | 覆盖系列标签 |
| `show_legend` | bool | `True` | 是否显示图例 |
| `legend_labels` | list | `None` | 覆盖图例文字（渲染时替换） |
| `legend_loc` | str | `"best"` | 图例位置 |
| `legend_bbox` | tuple | `None` | 图例 `bbox_to_anchor`，如 `(1.02, 1)` |
| `grid` | bool | `False` | 是否显示网格 |
| `show_values` | bool | `False` | 是否在每个数据点标注数值；标注自动放在局部极小值上方、局部极大值下方，避免与相邻线段重叠 |
| `value_decimals` | int | `1` | 数值标注小数位数 |
| `vlines` | float / list | `None` | 垂直参考线位置；落在数据范围之外的位置自动跳过 |
| `bar_series` | str / list | `None` | 以柱状图绘制的系列标签；其余系列仍为折线。柱色继承 `colors`，所在轴强制从 0 起 |
| `bar_width` | float | `None` | 柱宽（数据单位；日期轴单位为天）。`None` 时取相邻 x 间距中位数的 60% |
| `bar_edge_color` | str | `None` | 柱边框色；`None` 时与柱同色 |
| `bar_edge_linewidth` | float | `0.6` | 柱边框线宽 |
| `bar_alpha` | float | `1.0` | 柱透明度（0–1） |
| `shade` | tuple / list | `None` | 阴影区间，如 `(2008, 2009)` 或 `[(2008,2009),(2020,2021)]` |
| `note` | str | `None` | 图表左下角注释文字；当 `title_position="bottom"` 时，note 显示在 bottom title **上方** |
| `title_position` | str | `"top"` | `"top"` 或 `"bottom"` |
| `facet` | bool | `True` | 两条及以上序列是否按序列纵向分面；单序列不受影响 |
| `sharex` | bool | `True` | 分面子图是否统一 X 轴标度 |
| `sharey` | bool | `False` | 分面子图是否统一 Y 轴标度 |
| `auto_dual_y` | bool | `True` | `facet=False` 且未手动分组时，是否按稳健尺度自动建立多个 Y 轴；保留原参数名以兼容旧代码 |
| `scale_ratio_threshold` | float | `10.0` | 自动分组时划分相邻尺度组的最小尺度比，必须为正有限数 |
| `axis_groups` | mapping | `None` | 手动指定 `{序列标签: 组标识}`；相同组标识共用一个 Y 轴，并覆盖自动判断 |
| `max_y_axes` | int | `3` | Y 轴总数上限（含左轴）；自动模式超限时合并最接近的组，手动模式超限时报错 |
| `ax` | Axes | `None` | 传入已有坐标轴；多序列分面不接受单个 `ax`，此时应设 `facet=False` |

### 返回对象

| 场景 | 返回值 | 如何查看 |
|------|--------|----------|
| 单序列 | `(fig, ax)` | `ax.lines`、`ax.get_xlabel()`、`ax.get_ylabel()` |
| 多序列且 `facet=True` | `(fig, axes)` | `axes` 是一维 `numpy.ndarray`；用 `axes[0]`、`axes[1]` 访问各分面 |
| 多序列且 `facet=False`，只有一个尺度组 | `(fig, ax)` | 所有线都在 `ax.lines` 中；`ax.extra_y_axes == []` |
| 多序列且 `facet=False`，存在多个尺度组 | `(fig, ax)` | 左轴为 `ax`；第一个右轴为 `ax.right_ax`；全部右轴为 `ax.extra_y_axes`；全部轴为 `fig.axes` |

自动模式下，每条序列的稳健尺度取“绝对值 95% 分位数”和“5%–95% 分位距”中的较大者。按尺度排序后，相邻尺度比小于 `scale_ratio_threshold` 的序列归入同组；达到或超过阈值时建立新组。若组数超过 `max_y_axes`，函数依次合并尺度差距最小的相邻组。包含第一条输入序列的组始终使用左轴，其余轴依次放在右侧。

手动模式要求 `axis_groups` 的键与最终绘制标签完全一致，组标识必须可哈希，且组数不能超过 `max_y_axes`。手动分组优先于 `auto_dual_y`；多序列分面模式下不能同时指定 `axis_groups`。

### 示例

```python
import numpy as np, pandas as pd
from Ts.TsPlots import plot_series

# 基础用法：DataFrame 多序列，默认按序列纵向分面
t = np.arange(2000, 2026)
df = pd.DataFrame(
    {
        "GDP增长率": np.random.normal(6.5, 1, 26),
        "CPI增长率": np.random.normal(2.3, 0.8, 26),
    },
    index=t,
)
fig, axes = plot_series(
    df,
    title="宏观经济指标",
    ytitle="增长率（%）",
    grid=True,
)

# axes 是一维数组：分别调取两个分面
gdp_ax = axes[0]
cpi_ax = axes[1]
print(gdp_ax.get_title(loc="left"), cpi_ax.get_title(loc="left"))

# 分面时分别控制坐标轴标度：X 轴不统一，Y 轴统一
fig, axes = plot_series(df, sharex=False, sharey=True)

# 关闭分面：按相近尺度自动分组，可创建多个右侧 Y 轴
scale_data = {
    "增长率（%）": [2.1, 2.8, 3.0, 2.6],
    "GDP（亿元）": [12000, 13500, 15100, 16800],
    "人口（人）": [1_200_000, 1_230_000, 1_260_000, 1_290_000],
}
fig, ax = plot_series(scale_data, facet=False, title="不同量级序列")
left_ax = ax
first_right_ax = ax.right_ax
all_right_axes = ax.extra_y_axes
print(len(fig.axes), len(all_right_axes))

# 用户指定分组：增长率单独一组，GDP 和人口分别使用其他 Y 轴
manual_groups = {
    "增长率（%）": "比率",
    "GDP（亿元）": "经济总量",
    "人口（人）": "人口规模",
}
fig, ax = plot_series(
    scale_data,
    facet=False,
    axis_groups=manual_groups,
    max_y_axes=3,
)

# 如需强制所有序列画在同一 Y 轴，关闭自动分组
fig, ax = plot_series(scale_data, facet=False, auto_dual_y=False)

# 单序列仍返回一个 Axes；可叠加阴影、参考线和注释
dates = pd.date_range("2020-01", periods=36, freq="MS")
s = pd.Series(np.cumsum(np.random.normal(0, 1, 36)), index=dates, name="指数")
fig, ax = plot_series(
    s,
    freq="month",
    title="月度指数",
    shade=[(pd.Timestamp("2020-06"), pd.Timestamp("2020-09"))],
    vlines=pd.Timestamp("2021-01"),
    note="数据来源：模拟数据",
)

# 月度数据使用严格月刻度 + 年度标尺
fig, ax = plot_series(
    s,
    year_ruler=True,
    title="月度指数（严格刻度）",
)

# 柱状图：产量柱在左轴，钻机数折线在右轴
frame = pd.DataFrame(
    {
        "原油产量": [300, 310, 320, 315],
        "活跃钻机数": [45, 47, 50, 52],
    },
    index=dates[:4],
)
fig, ax = plot_series(
    frame,
    facet=False,
    axis_groups={"原油产量": "left", "活跃钻机数": "right"},
    bar_series=["原油产量"],
    colors=["#B8BDC6", "#1F4E79"],
    bar_edge_color="#6B7280",
    bar_alpha=0.72,
    grid=True,
    year_ruler=True,
    title="产量与钻机数",
)
# 柱状图参考线：落在数据范围之外的 vlines 自动跳过
fig, ax = plot_series(
    frame["原油产量"],
    bar_series=["原油产量"],
    facet=False,
    vlines=pd.Timestamp("2021-03"),
)
```

---

## `plot_scatter` — 散点图

```text
from Ts.TsPlots import plot_scatter

fig, ax = plot_scatter(data=None, x=None, y=None, *, ...)
```

### 接受的输入格式

| 输入类型 | 说明 |
|----------|------|
| `pd.DataFrame` | `x` 和 `y` 指定列名；`group` 指定分组列 |
| `dict` | `{标签: (x数组, y数组)}` |
| 2D array | 第 0 列为 x，第 1 列为 y |
| 两个数组 | 直接传 `x=...`, `y=...` |

### 常用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | 见上表 | `None` | 数据；直接传数组时可省略 |
| `x` | str / array | `None` | DataFrame 时为列名，否则为 x 数组 |
| `y` | str / array | `None` | DataFrame 时为列名，否则为 y 数组 |
| `group` | str | `None` | DataFrame 分组列名，每个值单独成系列 |
| `title` | str | `None` | 图表标题 |
| `xtitle` | str | `None` | x 轴标签 |
| `ytitle` | str | `None` | y 轴标签 |
| `x_unit` | str | `None` | x 轴单位，显示为「（单位：XX）」 |
| `y_unit` | str | `None` | y 轴单位，显示为「（单位：XX）」 |
| `fit_line` | bool | `False` | 是否添加 OLS 趋势线（每系列独立） |
| `alpha` | float | `0.7` | 散点透明度 |
| `markersize` | float | `7` | 标记大小 |
| `colors` | list | `None` | 覆盖默认配色 |
| `labels` | list | `None` | 覆盖系列标签 |
| `show_legend` | bool | `True` | 是否显示图例 |
| `legend_labels` | list | `None` | 覆盖图例文字 |
| `legend_bbox` | tuple | `None` | 图例 `bbox_to_anchor` |
| `hlines` | float / list | `None` | 水平参考线位置 |
| `vlines` | float / list | `None` | 垂直参考线位置 |
| `shade` | tuple / list | `None` | 垂直阴影区间 |
| `xtick_step` | float | `None` | x 轴刻度间隔 |
| `ytick_step` | float | `None` | y 轴刻度间隔 |
| `ymin` | float / None | `None` | y 轴下限（默认自动） |
| `equal_aspect` | bool | `False` | 是否等比例坐标轴 |
| `show_values` | bool | `False` | 是否标注每点坐标 `(x, y)`；标注方向自动偏向离最近邻居最远的方向，减少相互遮盖 |
| `grid` | bool | `False` | 是否显示网格 |
| `note` | str | `None` | 图表左下角注释文字；当 `title_position="bottom"` 时，note 显示在 bottom title **上方** |
| `title_position` | str | `"top"` | `"top"` 或 `"bottom"` |
| `ax` | Axes | `None` | 传入已有坐标轴 |

### 示例

```python
import numpy as np, pandas as pd
from Ts.TsPlots import plot_scatter

# 基础用法：DataFrame + 趋势线 + 单位标签
df = pd.DataFrame(
    {
        "收入": np.random.normal(50, 10, 80),
        "消费": np.random.normal(35, 8, 80),
        "地区": np.random.choice(["东部", "西部", "中部"], 80),
    }
)
fig, ax = plot_scatter(
    df,
    x="收入",
    y="消费",
    fit_line=True,
    x_unit="千元",
    y_unit="千元",
    title="收入与消费关系",
)

# 分组散点图
fig, ax = plot_scatter(
    df,
    x="收入",
    y="消费",
    group="地区",
    fit_line=True,
    legend_bbox=(1.02, 1),
    legend_loc="upper left",
)

# 直接传入数组
fig, ax = plot_scatter(
    x=np.random.normal(0, 1, 100),
    y=np.random.normal(0, 1, 100),
    hlines=0,
    vlines=0,
    fit_line=True,
)
```

---

## `plot_acf` 和 `plot_pacf` — 自相关函数图

```text
from Ts.TsPlots import plot_acf, plot_pacf

fig, ax = plot_acf(data, nlags=None, *, alpha=0.05, missing="drop", ...)
fig, ax = plot_pacf(data, nlags=None, *, alpha=0.05, missing="drop", ...)
```

基于 `statsmodels.tsa.stattools.acf` / `pacf` 绘制样本自相关和偏自相关函数图，
以柱状图 + 置信带形式展示。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like / Series / 单列 DataFrame | — | 一维时间序列 |
| `nlags` | int / None | `None` | 计算并显示的滞后阶数；不设置时由 statsmodels 根据样本量自适应选择 |
| `alpha` | float | `0.05` | 显著性水平（0.05 = 95% 置信带） |
| `missing` | `"drop"` / `"raise"` | `"drop"` | 删除全部非有限观测，或报告其原始行位置并报错 |
| `bartlett_confint` | bool | `True` | ACF: 使用 Bartlett 滞后变化公式（`True`）或均匀 ±z/√n 公式（`False`） |
| `zero_lag` | bool | `True` | ACF: 是否包含滞后 0（ACF ≡ 1） |
| `method` | str | `"ywm"` | PACF: 估计方法，可选 `"ywm"`、`"ols"`、`"ld"` |
| `bar_color` | str | `None` | 柱状颜色，默认 `DEFAULT_PALETTE[0]`（深蓝） |
| `band_color` | str | `"#d0d0d0"` | 置信带填充色 |
| `band_alpha` | float | `0.4` | 置信带透明度 |
| `max_ticks` | int | `12` | x 轴最大刻度数 |
| `grid` | bool | `False` | 是否显示网格 |
| `note` | str | `None` | 图表左下角注释 |
| `title_position` | str | `"top"` | `"top"` 或 `"bottom"` |
| `title` | str | `None` | 图表标题 |
| `xtitle` | str | `"滞后期数"` | x 轴标签 |
| `ytitle` | str | `"ACF值"` / `"PACF值"` | y 轴标签 |
| `ax` | Axes | `None` | 传入已有坐标轴（子图嵌入时使用） |

### 缺失值处理

- 默认 `missing="drop"` 在计算前删除 `NaN`、正无穷和负无穷，并使用清理后的有效样本量选择自适应 `nlags`。
- `missing="raise"` 遇到任何非有限值时报告其在原始一维输入中的行位置，适合禁止静默改变样本的分析。
- 删除内部缺口会压缩时间，使缺口两侧的观测被视为相邻时点。如果日历间隔具有统计含义，应使用 `missing="raise"`，并先插值或以其他方式显式处理缺口。

### 置信带说明

- **ACF**：默认使用 Bartlett 公式（`bartlett_confint=True`），置信带宽度随滞后变化。
- **PACF**：使用均匀置信带 ±z/√n。
- **默认滞后阶数**：`nlags=None` 时直接使用 statsmodels 的自适应规则。ACF 为 `min(int(10 * log10(n)), n - 1)`；PACF 为 `min(int(10 * log10(n)), n // 2 - 1)`。用户显式传入整数时严格使用该值，PACF 超过样本量限制时会明确报错。
- `alpha=0.05` 对应 95% 置信带，`alpha=0.01` 对应 99%，`alpha=0.10` 对应 90%。

### 示例

```python
from Ts.TsPlots import plot_acf, plot_pacf

# 基础用法
fig, ax = plot_acf(residuals, nlags=20, missing="drop")
fig, ax = plot_pacf(residuals, nlags=20, alpha=0.01, missing="raise")

# 嵌入子图网格
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(series, ax=ax1)
plot_pacf(series, ax=ax2)
```

### `plot_correlogram` — 绘制预计算相关系数

`plot_correlogram(data, confidence_band, ...)` 不重新计算统计量，只负责按
TsPlots 样式绘制调用方提供的 lag-indexed 相关系数和零假设置信带。Series
返回单轴；DataFrame 按列生成分面，适合 `ResidualCCFTestResult.plot_test()`
等统计结果复用。

```python
from Ts.TsPlots import plot_correlogram

fig, ax = plot_correlogram(
    residual_ccf,
    confidence_band=1.96 / np.sqrt(nobs),
    ytitle="Residual CCF",
)
```

---

## `plot_correlation_matrix` — 相关矩阵热图

该函数只负责验证和绘制已计算好的相关矩阵，统一使用 `[-1, 1]` 色阶。传入
DataFrame 时自动采用索引和列标签；数组输入可通过 `labels=` 指定标签。

```python
from Ts.TsPlots import plot_correlation_matrix

matrix = [[1.0, -0.7], [-0.7, 1.0]]
fig, ax = plot_correlation_matrix(
    matrix,
    labels=["ar.L1", "ma.L1"],
    annotate=True,
)
```

---

## `plot_lag_response` — 滞后柱线响应图

```python
from Ts.TsPlots import plot_lag_response

fig, ax = plot_lag_response(rdl_result.weights(20))
fig, axes = plot_lag_response(sarimax_result.weights(20))

# preliminary sample weights 为柱，最终传递函数隐含 weights 为线
fig, axes = plot_lag_response(
    preliminary_result.weights(16),
    line_data=final_result.weights(16),
)
```

横轴为非负整数 time lag，纵轴为 impulse-response weight。Series 或一维数组
返回单轴；多列 DataFrame 或二维数组按列生成分面，并保持输入顺序。函数统一
使用 TsPlots 的字体、色板、零参考线、网格和注释样式，也支持单响应外部 `ax`。
传入 `line_data` 时，`data` 绘制为 sample impulse-response 柱，`line_data`
绘制为 transfer-function weights 实线；两者的 lag 索引和响应名称必须完全一致。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data` | — | lag-indexed Series/DataFrame 或一/二维数组 |
| `line_data` | `None` | 与 `data` 对齐的可选实线响应数据 |
| `title` | `None` | 单图标题或多图总标题 |
| `xtitle` | `"Time lag"` | 横轴标题 |
| `ytitle` | `"Impulse response"` | 纵轴标题 |
| `color` | TsPlots 色板 | 单色或每响应一种颜色 |
| `line_color` | `"black"` | 可选响应线颜色 |
| `zero_line` | `True` | 绘制零响应参考线 |
| `grid` | `True` | 显示共享虚线网格 |
| `max_ticks` | `15` | 刻度过密时的最大整数刻度数 |

---

## `TsPlots.style` — 样式常量与辅助函数

通过 `from Ts.TsPlots.style import ...` 可访问所有样式接口。

### 常量

| 名称 | 说明 |
|------|------|
| `DEFAULT_PALETTE` | 8 色防色盲调色板 |
| `DEFAULT_LINESTYLES` | 8 种线型（黑白打印可区分） |
| `DEFAULT_MARKERS` | 8 种标记形状 |
| `LATIN_FONT` | Latin 字体名（Times New Roman） |
| `CHINESE_FONT_CANDIDATES` | 中文字体候选列表（仿宋系列） |
| `HEITI_FONT_CANDIDATES` | 黑体字体候选列表（标题用） |
| `SELECTED_CHINESE_FONT` | 运行时选定的中文字体名 |
| `FIGSIZE` | 默认图形尺寸 `(10, 5.5)` |
| `TITLE_FONTSIZE` | 标题字号 `14` |
| `AXIS_LABEL_FONTSIZE` | 坐标轴标签字号 `15` |
| `TICK_LABELSIZE` | 刻度标签字号 `14` |
| `LEGEND_FONTSIZE` | 图例字号 `14` |
| `NOTE_FONTSIZE` | 注释字号 `9` |

### 辅助函数

| 函数 | 说明 |
|------|------|
| `apply_fonts(latin, chinese_candidates)` | 重新配置 matplotlib 字体 |
| `style_axes(ax, *, grid, tick_labelsize)` | 隐藏上/右边框，设置刻度字号 |
| `draw_shade(ax, shade, color, alpha)` | 绘制垂直阴影区 |
| `draw_vlines(ax, vlines, color, linestyle, linewidth)` | 绘制垂直参考线 |
| `draw_hlines(ax, hlines, color, linestyle, linewidth)` | 绘制水平参考线 |
| `draw_legend(ax, *, ...)` | 绘制无边框图例 |
| `draw_unit_label(ax, unit, *, axis)` | 在坐标轴标签后追加单位 |
| `draw_note_and_bottom_title(fig, *, ...)` | 在图形底部放置标题或注释 |

---

## 设计说明

- **防色盲配色**：`DEFAULT_PALETTE` 基于 Okabe-Ito 方案，8 种颜色。
- **黑白可区分**：系列同时使用颜色、线型和标记三重编码；偶数索引系列使用实心标记，奇数索引使用空心标记。
- **中英文混排**：Latin 字符用 Times New Roman，CJK 字符用仿宋（自动回退）。
- **坐标轴风格**：隐藏上边框和右边框，刻度向外。
