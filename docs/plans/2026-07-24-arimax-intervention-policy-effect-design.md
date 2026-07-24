# ARIMAX、事件干预与政策效果估计设计

**目标：** 扩展 `SARIMA`，使其完整支持带日期索引的普通外生变量、自动保存的未来外生变量、多情境预测、事件干预变量和政策效果估计，并同步使 `TsMetrics` 的 OOS 与滚动回测正确处理外生变量。

**架构：** `SARIMA` 继续作为统一的 SARIMA/ARIMAX 模型入口，负责数据规范化、statsmodels `SARIMAX` 拟合及预测。新增 `TsModels/_intervention.py`，集中保存事件定义、事件设计矩阵、政策效果结果与三种不确定性计算。`SARIMAResult.policy_effect()` 仅作为便利入口委托给干预模块。`TsMetrics` 通过明确的评估协议取得每个预测起点对应的样本内与未来外生变量，不在指标层复制模型逻辑。

**技术栈：** Python、NumPy、pandas、statsmodels 0.14.x、SciPy、Matplotlib、pytest、Ruff。

---

## 1. 范围与边界

本功能包括：

- 普通外生变量的样本内 ARIMAX 估计；
- 带日期数据的严格对齐；
- 自动识别并保存 `y` 末期之后的默认未来外生变量；
- 单一或多个用户情境的未来外生变量预测；
- `pulse`、累计式 `step` 和事件研究窗口；
- 事件系数推断、条件反事实路径、逐期效果、累计效果及不确定性；
- delta、参数模拟和参数化 bootstrap；
- 前趋势联合 Wald 检验；
- OOS 与滚动回测中的外生变量切片。

本功能不承诺：

- 仅凭事件日期自动建立因果识别；
- 自动控制同期政策、遗漏变量或反向因果；
- 自动预测未知的普通外生变量；
- 自动插补缺失日期或缺失外生变量；
- 将情境模拟重新拟合为不同参数模型；
- 对旧的错误调用方式提供弃用别名或兼容回退。

普通无外生变量的 `SARIMA(data, order=...)` 仍然是该模型的合法子情形，但不保留任何已经删除或不再成立的伪 OOS、隐式错位、静默填充等旧行为。

## 2. 公共 API

### 2.1 模型构造

```python
from Ts.TsModels import EventSpec, SARIMA

model = SARIMA(
    data=y,
    dates=dates,
    exog=controls,
    exog_names=["oil_price"],
    events=[
        EventSpec(
            name="rate_cut",
            dates=["2025-03-15"],
            kind="pulse",
            date_rule="period",
        ),
        EventSpec(
            name="new_policy",
            dates=["2025-06-01", "2026-01-01"],
            kind="step",
        ),
        EventSpec(
            name="announcement",
            dates=["2025-04-10"],
            kind="pulse",
            window=(-2, 5),
            reference=-1,
        ),
    ],
    order=(1, 0, 1),
)
result = model.fit()
```

`data` 支持：

- 带唯一、单调 `DatetimeIndex` 的 pandas `Series`；
- 一维数值数组，此时可通过 `dates=` 提供等长日期；
- 没有事件且不需要日期预测时，仍可使用纯数值数组。

`exog` 支持：

- 带日期索引的 pandas `DataFrame`；
- 二维数值数组；数组长度必须等于 `data`，列名来自 `exog_names`。

### 2.2 预测与多情境

```python
default_prediction = result.predict(
    start="2025-01-01",
    end="2026-12-01",
)

scenario_predictions = result.predict(
    start="2025-01-01",
    end="2026-12-01",
    future_exog={
        "high_price": high_price_frame,
        "low_price": low_price_frame,
    },
    future_dates=future_dates,
)
```

规则：

- 带日期 `exog` 中与 `y` 日期重叠的部分用于拟合；
- 超出 `y` 末期的连续部分自动保存为 `"default"` 未来情境；
- `future_exog=DataFrame/array` 表示单一 `"custom"` 情境；
- `future_exog={name: frame_or_array}` 表示多个具名情境；
- 若存在自动保存的数据，多情境结果同时包含 `"default"`；
- `"default"` 为保留名称，用户不得覆盖；
- 所有情境必须具有相同普通外生变量列、预测日期和长度；
- 事件列由 `EventSpec` 自动生成，禁止通过 `future_exog` 提供；
- 各情境复用同一组估计参数，不重新拟合。

只有一个默认预测时返回现有 `PredictResult`。出现多个情境时返回：

```python
@dataclass
class ScenarioForecastResult:
    scenarios: dict[str, PredictResult]
    default_name: str | None
    dates: pd.DatetimeIndex | None
```

该结果支持按情境名称索引、`summary()` 和 `plot()`。若没有自动默认数据，仅有用户情境，则 `default_name=None`。

### 2.3 政策效果

```python
effect = result.policy_effect(
    events="new_policy",
    start="2025-01-01",
    end="2026-12-01",
    method="simulation",
    alpha=0.05,
    n_draws=2_000,
    seed=42,
)
```

`events` 接受一个事件名或事件名序列。返回：

```python
@dataclass
class PolicyEffectResult:
    coefficients: pd.DataFrame
    factual_mean: pd.Series
    counterfactual_mean: pd.Series
    effect: pd.Series
    lower: pd.Series
    upper: pd.Series
    cumulative_effect: float
    cumulative_lower: float
    cumulative_upper: float
    pretrend_test: dict | None
    method: str
    identification_note: str
```

另提供 `summary()` 与 `plot()`。

## 3. 事件编码

`EventSpec` 是不可变数据类：

```python
@dataclass(frozen=True)
class EventSpec:
    name: str
    dates: Sequence[DateLike]
    kind: Literal["pulse", "step"]
    window: tuple[int, int] | None = None
    reference: int | None = None
    date_rule: Literal["exact", "period", "next", "previous"] = "period"
```

编码语义：

- `pulse`：映射后的事件期取 1，其他期取 0；
- 同名多日期 `pulse` 若映射或窗口重叠，使用计数而不是截断为 1；
- `step`：每次事件发生后永久增加 1；同名多日期形成 `0 → 1 → 2 → ...` 的累计阶梯；
- `step` 系数解释为每次事件发生带来的边际永久变化；
- `window=(a, b)` 仅适用于 `pulse`，生成从相对期 `a` 到 `b` 的列；
- `reference` 对应列不进入模型，其系数固定为 0；
- 不允许 `window` 与 `step` 同时使用。

列名由内部统一编码器生成，并使用保留前缀，避免与普通外生变量冲突。事件名称必须非空且唯一。

### 3.1 日期映射

- `exact`：事件日期必须与一个观测日期完全一致；
- `period`：按观测序列的频率映射到事件所属期间，为默认值；
- `next`：映射到不早于事件日期的第一期观测；
- `previous`：映射到不晚于事件日期的最后一期观测。

无法推断频率、映射结果超出可用日期范围或时区不一致时明确失败。多个事件日期合法映射到同一期时，按上述计数语义编码。

## 4. 数据规范化与验证

模型内部保存一个规范化后的联合数据对象，至少包含：

- 一维 `endog`；
- 样本内普通外生变量；
- 样本内事件矩阵；
- 合并后的样本内设计矩阵；
- 日期索引；
- 自动保存的默认未来普通外生变量；
- 普通外生变量列名与事件列元数据。

验证规则：

- 日期必须唯一、单调递增且时区一致；
- 普通外生变量必须为有限数值，列名必须唯一；
- 数组型 `exog` 必须是二维，长度必须等于 `data`；
- 带日期 `exog` 必须完整覆盖 `y` 的每个日期；
- 超出 `y` 的默认未来数据必须从下一期开始连续覆盖；
- 普通外生变量列不能与事件保留列名冲突；
- 设计矩阵不能存在完全重复列、全零列或确定性的秩亏；
- 常数项、趋势项、事件和普通外生变量之间的完全共线性在拟合前报错。

缺失值策略：

- 默认 `missing="raise"`；
- 可选 `missing="drop"`，联合删除 `y`、普通外生变量和事件矩阵的对应样本行；
- 不允许只删除 `y` 后继续使用未同步切片的外生变量；
- 不自动插值或用最后观测值填充。

## 5. 估计与预测数据流

`SARIMA.fit()` 将规范化后的 `endog` 和合并设计矩阵传入 statsmodels `SARIMAX`。拟合结果保留：

- 普通外生变量和事件参数的位置；
- 参数协方差矩阵；
- 完整日期和频率；
- 原始事件定义；
- 自动保存的默认未来普通外生变量；
- 构建设计矩阵所需的不可变元数据。

`SARIMAResult.predict()`：

1. 解析样本内和样本外日期范围；
2. 为每个情境严格对齐普通外生变量；
3. 根据事件定义自动生成对应日期的事件列；
4. 按拟合时的固定列顺序合并设计矩阵；
5. 调用 statsmodels 预测接口；
6. 将每个情境规范化为 `PredictResult`；
7. 必要时组装 `ScenarioForecastResult`。

当日期频率无法可靠推断时，样本外预测必须显式传入 `future_dates`。缺少未来外生变量时，错误必须列出缺失列和日期。

## 6. 政策效果定义

政策效果是拟合模型下的条件反事实差异：

- 事实设计矩阵保留所选事件列；
- 反事实设计矩阵把所选事件列设为 0；
- 普通外生变量、其他事件、趋势、ARIMA 参数及同一组误差状态保持不变；
- 逐期效果为事实条件均值减去反事实条件均值；
- 累计效果为请求区间内逐期效果之和。

statsmodels `SARIMAX` 的外生变量部分是“回归项 + SARIMA 误差”。因此，在保持相同误差状态时，事件效应由事实与反事实事件设计矩阵之差对事件系数的线性组合给出。事件窗口提供跨期动态系数；不能把 ARIMA 误差相关性误写成政策冲击的结构性传播机制。

系数表包含：

- 估计值；
- 标准误；
- z 统计量；
- p 值；
- 置信区间；
- 事件名和相对期。

若事件包含参考期之前的 lead 列，`pretrend_test` 对这些系数执行联合 Wald 检验。参考期系数固定为 0，不参与估计。

结果必须始终包含识别说明：除非外生性、无同期未控冲击、模型设定和反事实稳定性等条件成立，否则输出是模型条件下的关联性政策效果，而不是自动成立的因果效应。

## 7. 不确定性

### 7.1 Delta method

对逐期和累计线性对比使用事件系数协方差矩阵：

```text
Var(Cβ) = C Var(β) C'
```

区间使用正态近似。该方法快，适合作为诊断和大样本近似。

### 7.2 参数模拟

默认 `method="simulation"`：

- 从事件系数的联合渐近正态分布抽样；
- 保留事件系数之间的估计相关性；
- 每次抽样计算完整逐期和累计效果；
- 使用经验分位数形成区间；
- `seed` 保证可复现。

该方法不模拟新的数据生成过程，表达的是给定模型下的参数不确定性。

### 7.3 参数化 bootstrap

`method="bootstrap"`：

1. 从拟合 SARIMAX 生成参数化时间序列；
2. 保持设计矩阵和事件安排；
3. 用相同模型规格重新拟合；
4. 重新计算事件系数和效果；
5. 用成功抽样的经验分位数形成区间。

记录每次失败及原因。成功率低于 80% 时整体失败，不以少量成功样本生成区间。

## 8. OOS 与滚动回测

`TsMetrics` 的模型评估协议增加对外生变量预测上下文的支持：

- 每个预测起点联合切分 `y`、日期和样本内普通外生变量；
- 预测窗口取得严格对应的未来普通外生变量；
- 事件矩阵由该训练模型保存的 `EventSpec` 和预测日期重新生成；
- 模型克隆不改变原模型及其 `result_`；
- 测试期 `y` 只用于评分，不能进入拟合或状态初始化；
- 未来普通外生变量可以进入预测，但必须明确说明这些值被假设为在预测起点已知。

任一预测起点缺少所需外生变量时：

- `on_error="raise"` 立即失败；
- `on_error="record"` 记录起点、缺失列和缺失日期，并使该预测行全部为 `NaN`。

多情境预测不能自动选择最优情境参与评分。若评估接口以后允许多情境，必须由调用者显式指定情境；本次实现中 OOS 与回测使用模型自动保存的默认外生变量路径。

## 9. 错误处理

以下情况直接抛出具体异常，不兼容回退：

- 日期重复、逆序或时区不一致；
- 事件日期无法按指定规则映射；
- 无法推断期间频率却使用 `date_rule="period"`；
- 数组型外生变量长度或列数错误；
- 情境列、日期或长度不一致；
- 覆盖默认保留情境名；
- 用户提供事件列；
- 事件定义冲突或事件名重复；
- 合并设计矩阵秩亏；
- 未来外生变量覆盖不足；
- bootstrap 有效重估比例不足；
- 试图对不存在的事件计算政策效果。

异常信息必须说明失败对象、预期值和实际值，不进行静默截断、广播、排序、填充或降级。

## 10. 文件边界

- `TsModels/_sarima.py`
  - 扩展 `SARIMA` 和 `SARIMAResult`；
  - 保存规范化数据和外生变量元数据；
  - 实现单情境与多情境预测；
  - 定义或承载 `ScenarioForecastResult`。
- `TsModels/_intervention.py`
  - `EventSpec`；
  - 事件日期映射和设计矩阵；
  - `PolicyEffectResult`；
  - delta、参数模拟、bootstrap；
  - 前趋势检验。
- `TsModels/_base.py`
  - 仅增加评估所需的明确外生变量协议；
  - 不放入事件统计逻辑。
- `TsMetrics/_common.py`
  - 联合训练切片和未来外生变量预测上下文。
- `TsMetrics/_oos.py`、`TsMetrics/_backtest.py`
  - 使用扩展协议，不直接理解事件列。
- `TsModels/__init__.py`
  - 导出新的公开类型。
- 测试与 README
  - 分别覆盖模型合同、干预统计、评估协议和完整使用示例。

## 11. 测试与验收

### 模型和数据合同

- 合成 ARIMAX 数据能恢复普通外生变量系数；
- `Series + DatetimeIndex` 与 `ndarray + dates` 结果等价；
- `missing="drop"` 联合删除，`missing="raise"` 明确失败；
- 日期错位、重复、乱序、时区冲突明确失败；
- 普通外生变量与事件列碰撞明确失败。

### 事件编码

- 单日期与多日期 `pulse`；
- 多日期累计式 `step`；
- 相对期窗口和参考期；
- 重叠窗口使用计数；
- `exact`、`period`、`next`、`previous`；
- 边界外映射和不可推断频率失败。

### 预测情境

- 自动保存并使用默认未来外生变量；
- 单个 `"custom"` 情境；
- 多个具名情境；
- 默认情境与用户情境并存；
- 禁止覆盖 `"default"`；
- 缺列、错列、错日期和覆盖不足失败；
- 各情境使用同一组估计参数。

### 政策效果

- 合成数据恢复已知事件系数；
- 事实与反事实逐期差等于事件设计矩阵对比；
- 累计效果正确；
- delta 区间正确；
- simulation 与 bootstrap 在固定种子下可复现；
- bootstrap 失败比例门槛有效；
- lead 联合 Wald 前趋势检验正确；
- 输出始终包含识别限制说明。

### 评估与回归

- OOS 和 backtest 每个起点取得正确外生变量切片；
- 测试期 `y` 不进入拟合；
- 外生变量缺失遵守 `on_error`；
- 无外生变量 SARIMA 的现有正常合同不回归；
- 全量 `pytest` 通过；
- Ruff 检查通过；
- README 示例可完整执行。

## 12. 完成定义

只有同时满足以下条件才算完成：

- 所有新公开合同有文档和类型说明；
- 单元测试、集成测试和现有测试全部通过；
- OOS/backtest 无目标泄漏且外生变量对齐有测试证据；
- 三种不确定性方法均有数值与失败路径测试；
- 无兼容别名、静默降级或隐式填充；
- 示例清楚区分预测情境、条件政策效果与因果识别；
- 代码经过系统审查和 code-simplify，不保留重复的数据对齐或事件编码逻辑。
