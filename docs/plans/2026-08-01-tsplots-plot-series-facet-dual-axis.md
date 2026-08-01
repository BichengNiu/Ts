# `plot_series` 分面与自动双 Y 轴实施计划

## 目标

扩展 `TsPlots.plot_series`：多序列默认纵向分面展示，并允许分别控制 X/Y 轴是否共享标度；关闭分面时，在序列尺度差异显著时自动切换为双 Y 轴，同时保持单序列和现有模型绘图接口兼容。

## 已确认接口

```python
plot_series(
    data,
    ...,
    facet=True,
    sharex=True,
    sharey=False,
    auto_dual_y=True,
    scale_ratio_threshold=10.0,
)
```

- 单序列：行为不变，返回 `(fig, ax)`。
- 多序列且 `facet=True`：每个序列一个纵向子图，返回 `(fig, axes)`；`axes` 为一维 `numpy.ndarray`。
- `sharex`、`sharey` 分别控制分面子图的 X/Y 轴统一标度。
- 多序列且 `facet=False`：序列叠加显示；若稳健尺度比达到阈值，自动使用左右双 Y 轴。
- `auto_dual_y=False` 禁用自动双轴；`scale_ratio_threshold` 必须为正有限数。
- 三个及以上序列按最大的相邻尺度断点分组；包含第一个序列的组保留在左轴。
- 双轴模式仍返回主轴 `ax`，右轴同时可通过 `ax.right_ax` 和 `fig.axes[1]` 获取。
- 多序列分面模式不接受单个外部 `ax`，应明确报错并提示使用 `facet=False`。

## 尺度判定

1. 对每个序列仅使用有限数值。
2. 稳健尺度取 `max(abs(x) 的 95% 分位数, x 的 5%-95% 分位距)`。
3. 忽略无法计算或尺度为零的序列；按有效尺度升序排列。
4. 最大相邻尺度比大于或等于 `scale_ratio_threshold` 时启用双轴，并在该断点分组。

## 实施步骤

### 任务 1：建立契约测试

- 在 `TsPlots/tests/test_plots.py` 增加：默认分面、`sharex`/`sharey`、单序列兼容、自动双轴、禁用双轴、自定义阈值、三序列分组、外部 `ax` 冲突、返回对象访问方式等测试。
- 先运行新增测试，确认当前实现按预期失败。

### 任务 2：实现功能并修复内部调用

- 在 `TsPlots/ts_plot.py` 增加参数校验、稳健尺度分组、分面绘制和双轴绘制逻辑。
- 保持原有颜色、标签、刻度、参考线、阴影、数值标注和图例参数有效。
- 在 `TsModels/_base.py` 和 `TsModels/_var.py` 的 Actual/Fitted 叠加图中显式传入 `facet=False`，保持返回单个 `Axes` 的既有契约。
- 运行 `TsPlots` 和相关模型测试。

### 任务 3：更新文档并全量验证

- 更新 `TsPlots/README.md` 的参数说明、返回对象、分面/双轴用例和访问方式，并修正已有参数表与实际签名不一致之处。
- 运行格式检查、静态检查、包级测试和仓库全量测试。
- 检查最终 diff，确认不包含对既有 `TsUtils/demo.ipynb` 与 Box-Cox 计划文件的修改。

## 验证命令

```powershell
python -m pytest TsPlots/tests -p no:cacheprovider -q
python -m pytest TsModels/tests TsSims/tests TsUtils/tests -p no:cacheprovider -q
python -m pytest -p no:cacheprovider -q
python -m ruff check TsPlots TsModels
python -m ruff format --check TsPlots TsModels
git diff --check
```

## 实施结果

- `plot_series` 已实现默认分面、`sharex`/`sharey`、自动双 Y 轴和稳健尺度分组。
- `BaseModelResult`、`VARResult`、`BaseSimResult` 的既有单轴返回契约已显式保留。
- `TsPlots/README.md` 与 `TsPlots/demo.ipynb` 已补充参数、返回对象、访问方式和可执行用例。
- `TsPlots/demo.ipynb` 已使用 `jupyter execute --inplace` 自上而下执行，无错误输出。
- 最终验证：`1217 passed`；Ruff lint/format 与 `git diff --check` 全部通过。
