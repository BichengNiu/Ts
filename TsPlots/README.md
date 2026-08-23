# Ts/TsPlots

适用于时间序列计量经济学学习的 Python 绘图工具包。提供统一的字体、配色和坐标轴风格，
支持时间序列折线图、分类/分组柱状图、散点图、自相关函数图、预计算相关图和相关矩阵热图。

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
├── bar_plot.py   # 分类/分组柱状图：plot_bar
├── sc_plot.py    # 散点图绘图：plot_scatter
├── acf_plot.py   # 相关图：plot_acf, plot_pacf, plot_correlogram
├── lag_plot.py   # 滞后柱线响应图：plot_lag_response
└── matrix_plot.py # 相关矩阵热图：plot_correlation_matrix
```

## 安装 / 导入

将 `TsPlots/` 目录放在工作目录下，随后直接导入：

```python
from Ts.TsPlots import (
    plot_series, plot_bar, plot_scatter, plot_acf, plot_pacf,
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
| `ytitle` | str | `None` | y 轴标签；**不传时轴标题只显示单位**（`unit` 给出则显示「（单位：XX）」，否则不显示）；显式传入时使用之（有 `unit` 则单位追加在末尾） |
| `unit` | str | `None` | y 轴单位，显示为「（单位：XX）」；默认作为主 y 轴与所有右侧双轴的轴标题，不再自动拼接变量名 |
| `ymin` | float / None | `None` | y 轴下限；`None` 为自动 |
| `year_ruler` | bool | `False` | 月度数据的严格 locator：只标数据中实际出现的 3 的倍数月份（标签为「3月」，不旋转），并在 x 轴下方为每个出现的年份绘制年度标尺；x 轴两端各留 10 天 |
| `xtick_step` | int | `None` | 数值 x 轴刻度间隔 |
| `max_ticks` | int | `12` | 自动刻度上限 |
| `linewidth` | float | `3` | 线宽 |
| `markersize` | float | `7` | 标记大小 |
| `colors` | list | `None` | 覆盖默认配色（按系列顺序） |
| `labels` | list | `None` | 覆盖系列标签 |
| `show_legend` | bool | `True` | 是否显示图例 |
| `legend_labels` | list | `None` | 覆盖图例文字（渲染时替换） |
| `legend_loc` | str | `None` | 图例位置；**不传时图例绘制在时间轴下方、绘图区外的底部边距**（`year_ruler=True` 时贴在年份标签之下），图注紧随其下；传任意位置（如 `"best"`、`"upper left"`）则图例回到绘图区内 |
| `legend_bbox` | tuple | `None` | 图例 `bbox_to_anchor`；留空时由 `legend_loc` 自动定位，如 `(1.02, 1)` |
| `legend_size` | float | `None` | 图例字号（磅）；留空时按图形物理尺寸自动缩放，并同步调整标记与图例句柄 |
| `grid` | bool | `False` | 是否显示网格 |
| `show_values` | bool | `False` | 是否在每个数据点标注数值；标注自动放在局部极小值上方、局部极大值下方，避免与相邻线段重叠 |
| `value_decimals` | int | `1` | 数值标注小数位数 |
| `vlines` | float / list | `None` | 垂直参考线位置；落在数据范围之外的位置自动跳过 |
| `bar_series` | str / list | `None` | 以柱状图绘制的系列标签；其余系列仍为折线。柱色继承 `colors`，所在轴强制从 0 起 |
| `bar_width` | float | `None` | 柱宽（数据单位；日期轴单位为天）。`None` 时取相邻 x 间距中位数的 60% |
| `bar_edge_color` | str | `BAR_EDGE_COLOR` | 柱边框色；默认浅灰（`BAR_EDGE_COLOR`），显式传 `None` 时与柱同色 |
| `bar_edge_linewidth` | float | `0.6` | 柱边框线宽 |
| `bar_alpha` | float | `1.0` | 柱透明度（0–1） |
| `shade` | tuple / list | `None` | 阴影区间，如 `(2008, 2009)` 或 `[(2008,2009),(2020,2021)]` |
| `note` | str | `None` | 图表左下角注释文字；当 `title_position="bottom"` 时，note 显示在 bottom title **下方** |
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
| `legend_loc` | str | `None` | 图例位置；不传时在 x 轴下方（绘图区外），传任意位置则回到绘图区内 |
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
| `note` | str | `None` | 图表左下角注释文字；当 `title_position="bottom"` 时，note 显示在 bottom title **下方** |
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

## `plot_bar` — 分类/分组柱状图

```text
from Ts.TsPlots import plot_bar

fig, ax = plot_bar(data, x=None, y=None, *, group=None, horizontal=False, stacked=False, ...)
```

针对离散分类的通用柱状图：支持多系列在分类槽内**并列**（分组柱图）、**堆叠**、
**横向**（分类在 y 轴）三种布局，以及柱顶数值标注、分类刻度抽稀和统一 TsPlots 样式。
`plot_series` 的 `bar_series` 一次只能在同一 x 位置画一个柱系列，因此分类并列/堆叠
柱图应使用本函数。

### 接受的输入格式

| 输入类型 | 说明 |
|----------|------|
| `pd.Series` | 索引为分类标签，值为一个柱系列（名称为系列标签） |
| `pd.DataFrame` | 宽表：索引为分类，其余每列为一个系列；`x` 指定分类列名（替代索引）；`y` 限定系列列 |
| `pd.DataFrame` + `group` | 长表：`x` 为分类列、`y` 为数值列、`group` 为系列列，每个组值一个系列 |
| `dict` | `{系列名: 值数组}`；分类默认 0..n-1，可用 `x` 数组指定 |
| 1D array | 单系列，分类 0..n-1 |
| 2D array | 每列一个系列，分类 0..n-1 |

### 常用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `horizontal` | bool | `False` | 横向柱图（分类在 y 轴） |
| `stacked` | bool | `False` | 系列互相堆叠而非并列 |
| `title` / `xtitle` / `ytitle` | str | `None` | 图标题 / 分类轴标题 / 数值轴标题（未传时自动检测） |
| `x_unit` / `y_unit` | str | `None` | 轴单位，显示为「（单位：XX）」 |
| `bar_width` | float | `0.6` | 柱宽（分类槽单位）。单系列柱宽即为该值；n 系列并列时每根柱 `bar_width/n`，整组占 `bar_width` |
| `bar_edge_color` | str | `BAR_EDGE_COLOR` | 柱边框色；默认浅灰（`BAR_EDGE_COLOR`），显式传 `None` 时与柱同色 |
| `bar_edge_linewidth` | float | `0.6` | 柱边框线宽 |
| `bar_alpha` | float | `1.0` | 柱透明度 |
| `colors` / `labels` | list | `None` | 覆盖系列配色 / 系列标签 |
| `show_legend` | bool | `True` | 是否显示图例 |
| `legend_loc` | str | `None` | 图例位置；不传时在分类轴下方（绘图区外），传任意位置则回到绘图区内 |
| `legend_title` / `legend_cols` | str / int | `None` | 图例标题 / 图例列数 |
| `grid` | bool | `False` | 是否显示网格（默认只画横向网格线 `grid_axis="y"`） |
| `ymin` | float | `None` | 数值轴下限（横向模式作用于 x 轴） |
| `show_values` | bool | `False` | 在每根柱顶端标注数值 |
| `value_decimals` | int | `1` | 数值标注小数位数 |
| `max_ticks` | int | `12` | 分类刻度标签上限；分类过多时均匀抽稀（柱仍全部绘制） |
| `tick_rotation` | float | `0` | 分类刻度标签旋转角度 |
| `hlines` / `vlines` | float / list | `None` | 数值轴 / 分类轴参考线 |
| `shade` | tuple / list | `None` | 分类区间阴影（仅纵向模式） |
| `note` / `title_position` | str | `None` / `"top"` | 图注与标题位置（同 `plot_series`） |
| `ax` | Axes | `None` | 传入已有坐标轴 |

### 示例

```python
import numpy as np, pandas as pd
from Ts.TsPlots import plot_bar

# 单系列：Series 索引即分类
s = pd.Series([120, 150, 90], index=["东部", "中部", "西部"], name="产量")
fig, ax = plot_bar(s, title="分地区产量", ytitle="产量", y_unit="万吨")

# 分组柱图：宽表 DataFrame，列即系列
df = pd.DataFrame(
    {"2023": [120, 90, 150], "2024": [135, 95, 160]},
    index=["东部", "中部", "西部"],
)
fig, ax = plot_bar(df, title="分年份分地区产量", grid=True)

# 长表 + group 列：分类/数值/系列三列
long = pd.DataFrame({
    "地区": ["东部", "东部", "中部", "中部"],
    "年份": ["2023", "2024", "2023", "2024"],
    "产量": [120, 135, 90, 95],
})
fig, ax = plot_bar(long, x="地区", y="产量", group="年份")

# 横向分组柱图 + 数值标注
fig, ax = plot_bar(df, horizontal=True, show_values=True)

# 堆叠柱图
fig, ax = plot_bar(df, stacked=True, ytitle="产量", y_unit="万吨")

# 数值标注、阴影、参考线
fig, ax = plot_bar(
    s,
    show_values=True,
    shade=(0, 1),
    hlines=100,
    note="数据来源：模拟数据",
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
| `bar_color` | str | `None` | 柱状颜色，默认 `DEFAULT_PALETTE[0]`（黑） |
| `band_color` | str | `BAND_COLOR` | 置信带填充色（浅灰） |
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
| `line_color` | `INK` | 可选响应线颜色（黑） |
| `zero_line` | `False` | 绘制零响应参考线（默认关闭，需要时显式开启） |
| `grid` | `True` | 显示共享虚线网格 |
| `max_ticks` | `15` | 刻度过密时的最大整数刻度数 |

---

## `TsPlots.style` — 样式常量与辅助函数

通过 `from Ts.TsPlots.style import ...` 可访问所有样式接口。

### 常量

| 名称 | 说明 |
|------|------|
| `BLACK` / `DARK_BLUE` / `GRAY` / `DARK_RED` | 调色模板主色：黑、深蓝、灰、深红 |
| `EXTENDED_PALETTE` | 4 个派生扩展色（≥5 系列时循环） |
| `DEFAULT_PALETTE` | 默认 8 色循环：`[黑, 深蓝, 灰, 深红, *扩展]` |
| `INK` | 文字色（黑）：标题、图注、热力图低对比文字 |
| `WHITE` | 空心标记填充、注释底、热力图高对比文字 |
| `AXIS_GRAY` / `AXIS_TEXT_GRAY` | 年轴刻度线与年轴文字 |
| `GRID_GRAY` | 网格 / 零参考线 |
| `ANNOTATION_EDGE` | 注释框边线 |
| `ZERO_LINE_COLOR` | 响应图可选零基线（`zero_line=True` 时使用；相关图已不再自动画 0 线） |
| `REFERENCE_LINE_COLOR` | 参考/关键断点线（深红） |
| `SHADE_COLOR` / `BAND_COLOR` | 阴影区 / 置信带填充（浅灰） |
| `DEFAULT_LINESTYLES` | 8 种线型（黑白打印可区分） |
| `DEFAULT_MARKERS` | 8 种标记形状 |
| `LATIN_FONT` | Latin 字体名（Times New Roman） |
| `CHINESE_FONT_CANDIDATES` | 中文字体候选列表（黑体族：微软雅黑 / 黑体，全包统一使用） |
| `HEITI_FONT_CANDIDATES` | 同一字体族的兼容别名（指向 `CHINESE_FONT_CANDIDATES`） |
| `SELECTED_CHINESE_FONT` | 运行时选定的中文字体名 |
| `FIGSIZE` | 默认图形尺寸 `(10, 5.5)` |
| `TITLE_FONTSIZE` | 标题字号 `14` |
| `AXIS_LABEL_FONTSIZE` | 坐标轴标签字号 `15` |
| `TICK_LABELSIZE` | 刻度标签字号 `14` |
| `LEGEND_FONTSIZE` | 图例字号 `14` |
| `NOTE_FONTSIZE` | 注释字号（与刻度一致） `14` |
| `LEGEND_BELOW_OFFSET` | 底部图例顶部相对 x 轴的锚点偏移（普通轴，`-0.17`） |
| `LEGEND_BELOW_YEAR_RULER_OFFSET` | 底部图例锚点偏移（`year_ruler=True`，避开年份标尺，`-0.24`） |

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
| `draw_note_and_bottom_title(fig, *, ...)` | 在图形底部放置标题、注释或底部图例（`BottomLegend`） |
| `draw_suptitle(fig, title, *, ...)` | 以统一字体族放置图级标题；全仓 `fig.suptitle` 必须经由它 |
| `BottomLegend(handles, labels, *, ...)` | 描述「时间轴下方、绘图区外」的底部图例：`draw_note_and_bottom_title` 将其锚定在时间轴/年份标签之下，图注紧随其下，并按实际高度精确撑开底部边距 |

---

## 设计说明

- **统一分层角色**：所有图表的堆叠层级由 `style.py` 的**角色常量**统一配置，绘图模块**禁止出现裸数字 zorder**。自底向上：背景填充 `ZORDER_BACKGROUND=0`（阴影/置信带）< 网格 `ZORDER_GRID=0.5` < 柱 `ZORDER_BAR=1` < 拟合线 `ZORDER_FIT=2` < 数据线/散点 `ZORDER_LINE=3` < 标注线 `ZORDER_REFERENCE=4`（`vlines` / `hlines` / 临界线 / 零值基准线）< 关键点高亮 `ZORDER_HIGHLIGHT=5`（检验统计量、特征根）。柱线混合图因此默认**网格在柱后、线在柱前、标注线在最前**，不依赖 matplotlib 默认值。
- **柱绘制单一实现**：`bar_plot._draw_bars` 是柱绘制的唯一实现，`plot_bar` 与 `plot_series(bar_series=...)` 的柱部分都经由它，柱样式契约（`bar_alpha` / `bar_edge_color` / `bar_edge_linewidth`，默认柱边为浅灰 `BAR_EDGE_COLOR`）保持一致。
- **默认调色模板**：`DEFAULT_PALETTE` 以 **黑 / 深蓝 / 灰 / 深红** 四主色引导，扩展 4 个同族派生色；主色与装饰色全部通过 `TsPlots.style.py` 中的**具名角色**引用。绘图代码（含 `TsModels` / `TsTests` / `TsSims` 的绘图方法）**禁止出现裸色值**——如需改色，只改 `style.py`。
- **黑白可区分**：系列同时使用颜色、线型和标记三重编码；偶数索引系列使用实心标记，奇数索引使用空心标记。
- **统一字体**：所有文字——图标题、轴标题、图例、刻度、图注、数值标注——共用同一字体族：Latin 用 Times New Roman，CJK 用**黑体族**（微软雅黑 / 黑体，自动回退，`CHINESE_FONT_CANDIDATES`），仅字号按角色区分；**图标题统一不加粗**（常规字重）。正文与标题不再使用不同字体族（仿宋正文 / 黑体标题的双字体族设计已废弃）。
- **坐标轴风格**：隐藏上边框和右边框，刻度向外。
- **底部排布**：图例默认绘制在时间轴（x 轴）下方、绘图区外的底部边距里（`year_ruler=True` 时贴在年份标尺标签之下），图注（`note`）紧跟图例下方、不再悬在图形最底边；分面图整图共享一个底部图例。显式传入 `legend_loc` 时图例回到绘图区内。
- **轴标题与图例角色标注**：轴标题默认只显示单位（`unit`），不再自动携带变量名（右轴亦然，`second_axis_title` / `third_axis_title` 可覆盖）；双轴（多轴）叠加图的图例文字统一为「变量名（左轴/右轴）」，单轴图不带括号后缀，显式传 `legend_labels` 时按原文不加后缀。
