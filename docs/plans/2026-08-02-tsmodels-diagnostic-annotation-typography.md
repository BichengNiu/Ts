# TsModels 诊断图注释分配与字体统一实施计划

**目标：** 按统计含义重新分配残差诊断检验结果，并让直方图的标题和轴标题完全遵循 `TsPlots` 的统一排版契约。

**范围：** 修改 `BaseModelResult.plot_diagnostics()` 的共享实现，因此 SARIMAX、GARCH 和 Auto 模型结果同步生效；VAR/VECM 使用独立的多变量诊断实现，不在本次范围内。

**已确认原因：** `Residuals`、`Residual ACF` 和 `Residual PACF` 通过 `TsPlots` 绘图函数设置标题与轴标题，而 `Residual Histogram` 直接调用 Matplotlib。`style_axes()` 只统一字体族、刻度、网格和边框，不设置标题/轴标题字号或标题粗细。实际对象属性为：其他三个面板标题 `14/bold`、轴标题 `15`，直方图标题 `12/normal`、轴标题 `10`。

## Task 1：先锁定新的公共契约

**文件：** `TsModels/tests/test_base.py`

1. 修改检验注释测试，明确要求：
   - `Residuals` 面板不再承载 White Noise 或 Normality 结果；
   - `Residual ACF` 面板包含 White Noise/Ljung-Box 的 `Q` 与 `p` 值；
   - `Residual Histogram` 面板包含 Normality/Jarque-Bera 的 `JB` 与 `p` 值；
   - `Residual PACF` 不重复检验结果。
2. 新增直方图排版回归断言：
   - 标题使用 `TITLE_FONTSIZE`、粗体和与其他诊断面板相同的标题字体族；
   - x/y 轴标题使用 `AXIS_LABEL_FONTSIZE` 和统一正文字体族。
3. 运行定向测试，确认测试在实现前因旧行为失败。

## Task 2：修改共享诊断图实现

**文件：** `TsModels/_base.py`

1. 保留现有 Ljung-Box 与 Jarque-Bera 计算和数值格式，不重复计算检验。
2. 拆分当前合并注释：
   - White Noise 注释放到 `ax_acf`；
   - Normality 注释放到 `ax_histogram`；
   - 两个注释沿用同一套位置、字号和文本框样式。
3. 直方图显式复用 `TsPlots.style` 的排版常量：
   - `TITLE_FONTSIZE`、`fontweight="bold"`、`pad=12`；
   - `AXIS_LABEL_FONTSIZE` 用于 x/y 轴标题；
   - 继续调用 `style_axes()` 统一字体族、刻度和边框。
4. 更新方法 docstring，准确说明每个检验结果所在面板。

## Task 3：同步文档

**文件：** `TsModels/README.md`、`AGENTS.md`

更新 `.plot_diagnostics()` 的表格说明：Residuals 为纯时间序列，Normality 位于 Histogram，White Noise 位于 ACF，PACF 保持无检验注释。

在根目录新增仓库级 `AGENTS.md`，固化“复用优先”开发契约：开发任何绘图、检验、估计或其他新功能前，必须先检索并优先继承/组合 Ts 包已有公共函数、基类与模块；只有现有能力无法满足明确契约时才新增实现，并用测试说明不能复用或扩展现有能力的原因。

## Task 4：验证

1. 定向回归：

   `python -m pytest TsModels/tests/test_base.py TsModels/tests/test_sarimax.py TsModels/tests/test_garch.py TsModels/tests/test_auto.py -q`

2. 用固定随机种子拟合真实 SARIMAX，保存临时 PNG 并目视确认：2×2 布局、注释归属、无文字遮挡、四个面板排版一致；检查后删除临时图。
3. 模块回归：

   `python -m pytest TsModels -q`

4. 静态检查：

   `python -m ruff format --check TsModels/_base.py TsModels/tests/test_base.py`

   `python -m ruff check TsModels`

   `python -m compileall -q TsModels`

5. 全库回归与覆盖率门槛：

   `python -m pytest -q --cov=TsSims --cov=TsTests --cov-branch --cov-report=term-missing --cov-fail-under=90`

6. 检查 `git diff --check` 和最终变更范围。本轮不提交，除非用户另行要求提交。
