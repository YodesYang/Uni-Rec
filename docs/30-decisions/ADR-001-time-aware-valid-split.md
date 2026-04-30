# ADR-001 · 验证集按 Row Group 时间序切分（vs 行级切分）

> **Status**: Accepted
> **Date**: 2026-04-30
> **Related Tasks**: [T8](../20-tasks/active.md#t8--把验证集切分从-random-改为按时间切)
> **Related Experiments**: 待平台 job 后回填

---

## Context（背景与触发原因）

### 触发证据

- **官方答复（2026-04-26 "basic questions on dataset"）**：「test set timestamps do not overlap with training set and are from a later time period」
- **当前 baseline 行为**：`dataset.py::get_pcvr_data` 用 `valid_ratio=0.1` 取**最后 10% Row Group** 作 valid，但**没人验证过 `sorted(glob)+RG-index` 顺序是否等于时间序**
- **隐含风险**：若 baseline 切法不是时间序 → 离线 valid 与训练集时间分布相同，而真实测试集是未来 → 离线 AUC 与公榜 AUC 必然有 gap → 调参方向被噪声带偏

### 问题域

- **不能纯改成 random**：当前已经不是 random（被原 TODO.md 误判为 random，实际是 sorted-glob 顺序）
- **不能直接改成"按时间切"而不留兜底**：万一平台 RG 顺序本来就近似时间序，强切反而引入额外打乱
- **必须先回答"当前切法实际的时间对齐情况"**，再决定是否切换策略

### 时间压力

- 距初赛 ddl 25 天，每天评估额度 3 次串行 → **不能为了诊断浪费一次评估额度**

---

## Decision（最终决定做什么）

新增 `--valid_split_strategy {filename, time}` CLI flag，**默认 `filename` 完全保留 baseline 行为**：

| 模式 | 切分依据 | 行为 |
|------|----------|------|
| `filename`（默认） | `sorted(glob) + RG-index` | 完全等同 baseline 原行为 |
| `time` | 每个 RG 的 `max(timestamp)` ASC | 严格时间序，保证 valid 在 train 之后 |

**关键设计**：**不论哪种策略**，切完都打印 train/valid ts 范围 + `overlap_sec` sanity log。这样**跑一次 baseline 就能直接读出来当前切法的实际时间对齐**，不消耗评估额度。

### 关键技术细节

- 文件：`baseline/dataset.py` 加 `_rg_ts_range` helper + `get_pcvr_data` 加 `valid_split_strategy` 参数
- 文件：`baseline/train.py` 加 `--valid_split_strategy {filename,time}` CLI flag + 透传
- diff：~85 行，3 个文件，0 lint 错误
- sanity log 实现：`filename` 模式只采 4 个边界 RG（避免扫全部 1000 RG），`time` 模式自然扫全部
- 时间排序数据源：优先 `parquet metadata.row_group(i).column.statistics`（零 IO），缺失时回落到读 `timestamp` 列（每 RG ~15KB，总 ~15MB IO，~2 秒）

---

## Alternatives Considered（备选方案）

### A. 直接默认改成 `time` 模式

- **Pros**：不用配开关，所有 baseline 自动获益
- **Cons**：违反 user rule「默认行为不变」红线；万一 baseline RG 顺序本来就近似时间序，强切反而引入额外乱序；无法对照
- **结论**：拒绝

### B. 直接做行级切分（按 row 的 `timestamp` 与 cutoff 比较）

- **Pros**：理论上最干净，valid 严格在 train 之后
- **Cons**：要改 `PCVRParquetDataset._convert_batch` 内部，diff 会扩散到训练 hot path；性能开销不可控；与现有 `row_group_range` 参数冲突
- **结论**：先做 RG 级（本 ADR），保留**升级路径**（T8.5）— 若 `time` 模式 overlap_sec 仍 > 0 再开行级方案

### C. 完全不动，只加 sanity log

- **Pros**：最小改动
- **Cons**：诊断能力够，但**没有可执行的 fix**；下次发现问题还要再改一轮
- **结论**：拒绝（既然要改，就一次给出诊断 + fix 两条路径）

---

## Consequences（影响与外溢）

### 正向

- **零评估额度消耗**：跑一次 baseline 就能从 sanity log 直接看出当前切法的时间对齐情况
- **可逆**：默认行为不变，强制 opt-in 才启用 `time` 模式 → 完全符合 user rule「默认行为不变 / 配置开关」红线
- **未来 ADR-NNN 可以引用**：T8.5（行级切）启动时直接复用 `_rg_ts_range` helper

### 负向 / 待观察

- **demo 数据特殊性**：demo 1k 行被人工 `row_group_size=100` 切成 10 RG，每个 RG 内部跨几乎全时间范围（780s）→ 即使 `time` 排序也消不掉 overlap（filename overlap=765s / time overlap=770s）。**这是 demo 数据问题，不是 ADR 设计问题**
- **平台数据可能也有 overlap**：若平台 RG 也跨大时间段，本 ADR 在 RG 粒度无解 → 必须升 T8.5（行级切）
- **新增 `_rg_ts_range` 函数**：在 RG 数量超大（>10K）时元数据扫描可能变慢；当前 1000 RG 量级毫无问题

---

## Verification（如何验证决策正确）

### 已验证（本地 demo）

- ✅ 代码无 lint 错误
- ✅ `python -c` 内联跑 + 完整 train.py 1 epoch 端到端验证
- ✅ AUC 0.6761 与 EXP-001 baseline 完全一致 → 训练逻辑零副作用
- ✅ sanity log 按预期输出：`[T8] train ts: [..725001, ..725767]  valid ts: [..725002, ..725675]  overlap_sec: 765`
- ✅ warning 也按预期触发：`[T8] valid_split_strategy='filename' yields ts overlap with train; consider --valid_split_strategy time`

### 待验证（下一次 platform job）

| 平台 `filename` overlap_sec | 平台 `time` overlap_sec | 决策路径 |
|---|---|---|
| ≤ 0 | — | baseline 切法本来就是时间序 → 关闭这条线，T8 直接 `[x]` 结案 |
| > 0 | ≤ 0 | 切到 `--valid_split_strategy time` → 预期公榜 +0.005~0.010 AUC |
| > 0 | > 0 | RG 跨时间段 → 升 T8.5（行级切），写 ADR-XXX 继任本 ADR |

### 验证方案（user rule "影响数据/采样的改动须给出验证方案"）

- **离线 vs 在线一致性**：改前/改后用同一 ckpt 跑两次 eval，看离线 AUC 数字与公榜 AUC 的 gap 缩小多少（预期：改前 gap 0.01~0.03，改后 gap < 0.005；若反之 → 说明 row group 不是时间序，回滚）
- **AUC 绝对值预期**：改后离线 AUC **可能下降** 0.005~0.015（因为切走了"近期"训练样本），但公榜 AUC 期望**上升** 0.003~0.01 — 这是健康信号
- **Sanity**：已内置 sanity log 自动打印 train / valid 时间区间 + overlap_sec
