# EXP-026 · P3 Valid Split Gap（15%）Half-Train

**日期**：2026-05-05
**作者**：Agent
**状态**：⚠️ **Funnel 未真正起效**（2026-05-05 · 诊断结果见下文 · gap 区间时间戳与 train/valid 严重重叠）

---

## 背景

EXP-024 方法论复盘（2026-05-04 晚）的核心诊断：

> Dev AUC 测的是"train 前 95% → train 尾部 5%"的 generalization；LB AUC 测的是"train 100% → 远未来 test"的 generalization。**两个不同任务**，相关性不稳定。

7 次历史提交 dev→LB gap 稳定在 **+0.021~+0.030**（EXP-018 §历史 Gap 扫描）。当前 time-split 的尾部 valid 与 train 尾部紧邻（覆盖最后 ~1~2 天），test 则是**更远未来**（官方："test timestamps do not overlap with training set and are from a later time period"）。

这是 ADR-005 "Valid Split Redesign" 的方案 C（"留时间空窗"）路线的第一次真实数据验证。

---

## 假设

在 train 尾部和 dev 之间**丢弃一段时间窗**（gap），模拟 test 相对 train 的跳跃。布局：

```
time axis: [ train 80% ] [ gap 15% (dropped) ] [ dev 5% ]
```

Gap 丢弃的数据既不进 train 也不进 dev。Dev 距 train 尾部多了 15% 时间跨度的"退让"，预期：

1. **Dev AUC 绝对值会下降**（train 数据少 15% + dev 更难）—— 这是**必然**，不是失败
2. **Dev→LB gap 会缩小**（dev 任务与 LB 任务更接近）
3. 判决标准**不是 dev 绝对值**，而是：
   - `icq-half 的 dev→LB gap = 0.868619 − 0.84637 = +0.022`
   - 本实验如果 dev→LB gap < +0.018 → **P3 成立**（funnel 改善）
   - 如果 gap ≥ +0.020 → P3 无效

---

## 配置

**Job Name**：`gap15-icq-halftrain-swiglu-vr005-bs128`

与 `icq-t13tf-halftrain-swiglu-vr005-bs128`（EXP-019, LB 0.84637 SOTA）**单变量差异**：

| 参数 | icq-half baseline | 本 job | 变化 |
|---|---|---|---|
| `--valid_gap_ratio` | 0.0 | **0.15** | +P3 gap |
| `--logloss_rebound_patience` | 未传（0）| 2 | 与 EXP-025 对齐 |
| 其他 | 完全一致 | 完全一致 | — |

完整 flag：

```
--item_conditioned_query
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05 --valid_gap_ratio 0.15
--logloss_rebound_patience 2 --logloss_rebound_delta 0.0
--batch_size 128 --num_epochs 4
```

**明确未叠加**（single-variable discipline）：
- `--id_mask_prob`（P1，独立 EXP-025 同时在跑）
- `--target_item_seq_injection`（T16，独立 full train 在跑）
- `--holdout_ratio`（独立成本未核查前不叠）

---

## 代码落地摘要

- `baseline/dataset.py::get_pcvr_data`：新增 `valid_gap_ratio` 参数，time-sorted RG 切分顺序变为 `[train | gap (dropped) | valid | holdout]`。参数校验：`valid_gap_ratio > 0` 必须配 `valid_split_strategy='time'`；与 `enable_row_time_cutoff` 互斥；`valid + holdout + gap < 1.0`。
- `baseline/train.py`：新增 CLI `--valid_gap_ratio`，透传 `get_pcvr_data`。
- Gap 区间的 RG **既不进 train dataset 也不进 valid dataset**，完全丢弃。
- 日志新增 `[P3] gap ts (dropped): [min, max]  effective train→valid distance = X sec`，便于平台日志核查 gap 时间区间。

本地 dataset-only smoke 5/5 通过（与 EXP-025 共享 smoke 脚本），其中 smoke 2 直接验证三路切分：10 RG × `valid_ratio=0.2 + holdout_ratio=0.1 + valid_gap_ratio=0.2` → `7 train (700 rows) / 1 gap (dropped, 100 rows) / 1 valid (100 rows) / 1 holdout (100 rows)` ✓

---

## 判决阈值（**gap 是主指标，不是 dev 绝对值**）

| 场景 | E4 dev AUC | E4 dev→LB gap | 判决 |
|---|---|---|---|
| **P3 大成** | 任何值 | **< +0.015** | 方案 C 落地，ADR-005 Accepted；下轮默认带 gap |
| **P3 成功** | 任何值 | < +0.020 | 方案 C 可行；下轮保留但需与 P1 / T16 做联合优化 |
| **P3 模糊** | ≥ 0.860 | 0.020~0.023 | gap 效应太弱，尝试 gap=20% 或 25% |
| **P3 失败** | < 0.855 | — | 训练数据损失代价 > valid 信号改善，放弃方案 C 切方案 A（triple-split）|

**关于 LB evaluation 的优先级**：P3 的首要价值是"funnel 可信度"，而不是"LB 突破"。即使 LB < SOTA，只要 gap 缩小，P3 就**值得 evaluate 一次来校准 dev→LB 经验公式**。

但 evaluation quota 今日已用 3/4，剩余 1 slot 优先给 **E4 dev 最高的那个 job**（P1 或 P3）。

---

## 回流数据

| Epoch | Step | Val AUC | Val LogLoss | vs icq-half same epoch |
|---|---|---|---|---|
| E1 | 13342 | 0.862648 | 0.22878 | icq-half E1=0.864992 (Δ **−0.002344**) |
| E2 | 26684 | 0.865276 | 0.22617 | icq-half E2=0.867557 (Δ **−0.002281**) |
| E3 | 40026 | 0.865914 | 0.22568 | icq-half E3=0.867965 (Δ **−0.002051**) |
| **E4** | **53368** | **0.867124** | **0.22506** | icq-half E4=0.868619 (Δ **−0.001495**) |

**best = E4 @ step 53368, local AUC = 0.867124** (stop_reason=max_epochs_completed)
总训练时长 ≈ 5h52m（vs icq-half 5h19m，因为 train 行数减少 15%）

### Row Group 切分实际结果

```
Row Group split (time): 836 train (1666033 rows), 156 gap (dropped, 324948 rows), 52 valid (108975 rows)
```

看似正确的三路切分，但时间戳分析暴露**致命问题**：

```
[T8]  train ts: [1772238909, 1772588918]   (跨度 4.04 天)
[P3]  gap   ts: [1772247536, 1772588958]   (跨度 3.95 天)  ← 与 train 严重重叠！
      valid ts: [1772247872, 1772588990]   (跨度 3.95 天)  ← 与 train 严重重叠！
```

`effective train→valid distance (valid_min - train_max) = -341046 sec` ⚠️

**负值意味着 valid_min < train_max**，即 train 尾部和 valid 起点时间上**完全反向**。gap 根本没起到"时间退让"的作用。

---

## Funnel 失效诊断（关键 insight）

### 为什么 gap 未真正隔离时间？

**根源**：`valid_split_strategy='time'` 的排序键是 **每个 RG 的 `max(timestamp)`**，但**单个 RG 内部时间跨度 4 天 vs 总跨度 4.04 天**——即：

- **每个 RG 的时间范围都几乎覆盖整个 train 时段**（RG 内时间乱序 / 广泛交织）
- 按 `max(ts)` 排序后取尾部 15% 做 gap，**这些尾部 RG 的 min(ts) 仍能小到 2/28 左右**
- 所以 "gap RGs" 丢弃的是 "max(ts) 较新但内部包含跨 4 天全部时期的样本"
- **实际时间上 train / gap / valid 三方几乎完全共享同一时间窗**

### 对比 EXP-024 诊断

EXP-024 已指出"dev AUC 测 `train 前 95%` → `train 尾部 5%`，LB 测 `train 100%` → `远未来 test`"，两者是不同任务。**我们原本以为加 gap 能让 dev 接近 LB 任务**。但因为 RG 内部时间乱序，**加 gap 只是减少了 15% 训练数据，没缩小 dev/test 任务差**。

### 证据：dev AUC 降 0.0015 符合"减 15% train data"的纯预期

- 15% train data 减少 → dev AUC 约降 **0.001~0.002**（经验估算）
- 实测 E4 Δ dev = **−0.001495** ← 完美落在该预期内
- 如果 gap **真的** 起到时间退让作用，dev 应额外降 **0.003~0.005**（test 远未来任务更难）
- **实测 dev 降幅只有"数据损失"那一档，没有"任务变难"那一档** → 证明 gap 未起效

---

## 结论与修复方案

### 方案 C（RG-level gap）在当前数据下**天然失效**

除非：① RG 内部时间是单调的（已证否）；② 使用 **row-level** cutoff（类似 T8.5 的实现）。

### 下一步（不立即启动）

**T22-fix · Row-level gap**：在 `_row_ts_cutoff` 的基础上实现 row-level gap cutoff，即：

```
train row: timestamp < T_cutoff_1
gap row:   T_cutoff_1 <= timestamp < T_cutoff_2   (丢弃)
dev row:   timestamp >= T_cutoff_2
```

需要：
- `_row_ts_cutoff` 扩展为双阈值版本
- `PCVRParquetDataset.timestamp_min/max` 新增"排除范围"语义
- 与现有 `enable_row_time_cutoff`（单阈值）兼容

代码量估算 ~50 行，但**优先级排在 T23 之后**：先验证 seq_c disable 方向是否有 LB 收益，再决定是否投入修 P3 的工程成本。

### Evaluation 决策

**不 evaluate gap15**。本轮 E4 dev 0.867124 落在 icq-half E4 减数据的预期内，无法提供独立 LB 校准价值。Evaluation quota 已优先用于 idmaskc15（结果 LB 0.842176 已证伪）。

