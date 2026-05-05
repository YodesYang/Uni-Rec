# EXP-026 · P3 Valid Split Gap（15%）Half-Train

**日期**：2026-05-05
**作者**：Agent
**状态**：🔄 In Progress（platform running 中，等 E4 回流）

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

## 回流数据（待填）

| Epoch | Step | Val AUC | Val LogLoss | vs icq-half same epoch |
|---|---|---|---|---|
| E1 | TBD | TBD | TBD | icq-half E1=0.864992 |
| E2 | TBD | TBD | TBD | icq-half E2=0.867557 |
| E3 | TBD | TBD | TBD | icq-half E3=0.867965 |
| **E4** | **TBD** | **TBD** | TBD | icq-half E4=**0.868619** |

预期轨迹：E1~E4 dev AUC 应低于 icq-half 对应 epoch **约 0.002~0.005**，即 E4 落在 **[0.863, 0.866]** 区间为"正常"。

---

## 与 EXP-025（P1 id_mask）并行关系

- 两个 job 都基于 icq-half SOTA 做**单变量 ablation**
- EXP-025 改**训练信号**（token-level mask）
- EXP-026 改**评价信号**（valid 定义）
- **正交**，互不干扰
- 两个 job E4 回流后可做交叉分析：
  - 若 P1 正向（LB ≥ 0.846）但 P3 gap 没缩 → 说明问题主要是 id 过拟合，不是 valid 定义
  - 若 P3 gap 缩了但 P1 LB 没涨 → 说明 funnel 方向对，但 P1 强度需要调
  - 若两者都正向 → **下一轮叠加** 成为新默认 config

---

## 下一步

1. E4 数据回流 → 对照 icq-half 判断 dev 轨迹健康度
2. 与 EXP-025 交叉分析 gap vs id_mask 哪个改善更大
3. 如 dev 轨迹健康 + evaluation quota 允许 → 消耗 1 eval 拿 LB，校准 `dev→LB` 经验公式
4. 若 gap 明显缩小 → 下一轮 config 默认加 `--valid_gap_ratio 0.15`（或 adjusted ratio），与 P1 / T16 叠加做 full train
