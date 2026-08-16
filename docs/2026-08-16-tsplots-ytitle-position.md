# TsPlots y 轴标题排布参数 `ytitle_position`

日期：2026-08-16

## 需求

业务方希望图表默认采用「丁字」式 y 轴标题：标题横排在 y 轴的上端点，
轴线从标题下方垂下（竖是轴、横是标题），替代传统的轴侧竖排；同时该排布
必须可由调用方选择（可选回传统样式），且所有 TsPlots 图表（时序图、
ACF/PACF 相关图）统一生效。

## 设计

在 `TsPlots` 的公共绘图函数上新增向后兼容参数：

- `ytitle_position : {"top", "side"}, default "top"`
  - `"top"`：标题横排（rotation=0）并居中于轴顶端上方、正对 y 轴轴心
    （左轴 x=0、右侧双轴 x=1，y=1.05），轴线顶端落在标题横笔正中，
    形成「丁字」布局。
  - `"side"`：传统布局，标题竖排在轴侧（matplotlib 默认行为，不做事后
    调整）。
- 非法取值抛出 `ValueError`（消息包含 `ytitle_position`）。

## 实现

复用优先：新增的公共助手 `place_ylabel_at_top(ax)` 放在
`TsPlots/style.py`（统一绘图样式契约模块），对任意轴（含 twinx 右轴）
原地调整 y 轴标题；无标题文本的轴不做任何改动。

应用点：

- `TsPlots/ts_plot.py::plot_series`
  - 单轴路径：`ax.set_ylabel` 之后。
  - 分面路径：每个 `panel_ax.set_ylabel` 之后。
  - 自动双轴路径：每个 `right_axis.set_ylabel` 之后。
- `TsPlots/acf_plot.py`
  - `plot_correlogram` / `plot_acf` / `plot_pacf` 全部透传参数到
    `_draw_correlogram`，在 `set_ylabel` 之后应用。

## 兼容性

- 参数默认值为 `"top"`（新默认布局），需要传统排布的调用方显式传入
  `ytitle_position="side"`。
- 原有参数与返回契约不变；`plot_series` 的多轴、分面、`extra_y_axes`
  行为不受影响。
- 公共 API 的 docstring 已同步补充参数说明（仓库
  `test_public_help_is_complete` 契约要求每个签名参数都被文档化）。

## 与图标题的避让

图标题靠左（`title_loc="left"`，含分面面板的左上系列名标题）且 y 标题
置顶时，二者同处轴顶端会重叠。处理规则：

- **y 轴标题位置不变**：仍正对轴心（x=0 / x=1，y=1.05）。
- **图标题右移**：`place_left_title_right_of_ylabel` 测量 y 标题的实际
  宽度，把左上标题的左缘移到 y 标题右缘 + 间隙（单轴 8pt、分面面板
  6pt）处，二者互不重叠。
- 单轴：仅当 `title and title_position == "top" and title_loc == "left"`
  时右移；分面：面板标题恒为左上角，恒右移。
- 右侧双轴不受左侧图标题影响。

## 参考线与阴影的日期字符串兼容

matplotlib 3.11 的 `mdates.date2num` 不再接受纯字符串（把字符串当作
0-d 数组索引报 ``too many indices``）。`_clip_vlines_to_data` 对字符串
位置先用 ``pd.to_datetime`` 归一化为 `Timestamp` 再转换；日期类型的
参考线/阴影区间在 3.11 下正常绘制。

## 测试

`tests/test_ytitle_position.py` 覆盖：

- `plot_series` 默认（top）：旋转 0、位置在轴顶端外侧。
- `plot_series(ytitle_position="side")`：旋转 90、传统位置。
- 分面面板同样应用 top 布局。
- `plot_acf` 默认 top、`plot_pacf` side 传统。
- 非法取值抛出 `ValueError`。
