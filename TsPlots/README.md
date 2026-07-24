# Ts/TsPlots

适用于时间序列计量经济学学习的 Python 绘图工具包。提供统一的字体、配色和坐标轴风格，
支持时间序列折线图、散点图和自相关函数图。

## 模块结构

```
TsPlots/
├── __init__.py   # 统一导出所有公开接口
├── style.py      # 共享样式：字体、调色板、辅助函数
├── ts_plot.py    # 时间序列绘图：plot_series
├── sc_plot.py    # 散点图绘图：plot_scatter
└── acf_plot.py   # 自相关函数绘图：plot_acf, plot_pacf
```

## 安装 / 导入

将 `TsPlots/` 目录放在工作目录下，随后直接导入：

```python
from Ts.TsPlots import plot_series, plot_scatter, plot_acf, plot_pacf
```

依赖：`matplotlib`、`numpy`、`pandas`、`statsmodels`（均为标准数据分析环境）。

---

## `plot_series` — 时间序列折线图

```python
from Ts.TsPlots import plot_series

fig, ax = plot_series(data, x=None, y=None, *, ...)
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
| `ymin` | float / None | `0` | y 轴下限；`None` 为自动 |
| `freq` | str | `None` | datetime 轴频率：`'day'` / `'week'` / `'month'` / `'quarter'` / `'year'` |
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
| `vlines` | float / list | `None` | 垂直参考线位置 |
| `shade` | tuple / list | `None` | 阴影区间，如 `(2008, 2009)` 或 `[(2008,2009),(2020,2021)]` |
| `note` | str | `None` | 图表左下角注释文字；当 `title_position="bottom"` 时，note 显示在 bottom title **上方** |
| `title_position` | str | `"top"` | `"top"` 或 `"bottom"` |
| `ax` | Axes | `None` | 传入已有坐标轴（多子图时使用） |

### 示例

```python
import numpy as np, pandas as pd
from Ts.TsPlots import plot_series

# 基础用法：DataFrame，多系列
t = np.arange(2000, 2026)
df = pd.DataFrame({
    "GDP增长率": np.random.normal(6.5, 1, 26),
    "CPI增长率": np.random.normal(2.3, 0.8, 26),
}, index=t)
fig, ax = plot_series(df, title="宏观经济指标", ytitle="增长率（%）", grid=True)

# 高级用法：阴影 + 参考线 + 注释
fig, ax = plot_series(
    df,
    shade=[(2008, 2009), (2020, 2021)],
    vlines=2015,
    note="数据来源：模拟数据",
    ymin=None,
)

# datetime 索引 + 月度频率
dates = pd.date_range("2020-01", periods=36, freq="MS")
s = pd.Series(np.cumsum(np.random.normal(0, 1, 36)), index=dates, name="指数")
fig, ax = plot_series(s, freq="month", title="月度指数", ymin=None)
```

---

## `plot_scatter` — 散点图

```python
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
df = pd.DataFrame({
    "收入": np.random.normal(50, 10, 80),
    "消费": np.random.normal(35, 8, 80),
    "地区": np.random.choice(["东部", "西部", "中部"], 80),
})
fig, ax = plot_scatter(df, x="收入", y="消费",
                       fit_line=True, x_unit="千元", y_unit="千元",
                       title="收入与消费关系")

# 分组散点图
fig, ax = plot_scatter(df, x="收入", y="消费",
                       group="地区", fit_line=True,
                       legend_bbox=(1.02, 1), legend_loc="upper left")

# 直接传入数组
fig, ax = plot_scatter(
    x=np.random.normal(0, 1, 100),
    y=np.random.normal(0, 1, 100),
    hlines=0, vlines=0, fit_line=True,
)
```

---

## `plot_acf` 和 `plot_pacf` — 自相关函数图

```python
from Ts.TsPlots import plot_acf, plot_pacf

fig, ax = plot_acf(data, nlags=40, *, ...)
fig, ax = plot_pacf(data, nlags=40, *, ...)
```

基于 `statsmodels.tsa.stattools.acf` / `pacf` 绘制样本自相关和偏自相关函数图，
以柱状图 + 置信带形式展示。

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | array-like / Series / 单列 DataFrame | — | 一维时间序列 |
| `nlags` | int | `40` | 计算并显示的滞后阶数 |
| `alpha` | float | `0.05` | 显著性水平（0.05 = 95% 置信带） |
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

### 置信带说明

- **ACF**：默认使用 Bartlett 公式（`bartlett_confint=True`），置信带宽度随滞后变化。
- **PACF**：使用均匀置信带 ±z/√n。
- `alpha=0.05` 对应 95% 置信带，`alpha=0.01` 对应 99%，`alpha=0.10` 对应 90%。

### 示例

```python
from Ts.TsPlots import plot_acf, plot_pacf

# 基础用法
fig, ax = plot_acf(residuals, nlags=20)
fig, ax = plot_pacf(residuals, nlags=20, alpha=0.01)

# 嵌入子图网格
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(series, ax=ax1)
plot_pacf(series, ax=ax2)
```

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
