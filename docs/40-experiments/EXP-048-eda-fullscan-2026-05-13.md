# EXP-048 · EDA Full Scan 1044 RG · 2.10M rows · 2026-05-13

> **Status**: ✅ 已完成 · 数据落档 + 方法学优先级重排 · CEST 22:15
> **日期**: 2026-05-13 CEST 22:00 起草
> **关联**: [TODO T31](../20-tasks/active.md) · [team-memory line 关于 user_id 一人一行](../../.cursor/rules/ai-readme/team-memory.mdc) · 原始 JSON `docs/99-raw/eda/eda-fullscan-2026-05-13.json`

---

## TL;DR

平台跑 `src/eda_quantile.py` 全扫 1044 row groups · 2.1M rows · 213k positive · 总耗时 16 分 33 秒 · 数据通过 base64+gzip blob 从 stdout 完整回流。

**3 条 EDA 直接给的方法学结论**:

1. **`user_id` distinct=500k · max_freq=1 · 100% 一人一行实锤**（社区 Seed 主楼情报已证实 · 这次 EDA 数字层面再确认 · `mean_freq=1.00`）
2. **`item_id` distinct=231k · 高频 (>10) 占 13.6% distinct · 长尾 (=1) 占 45.4%** · 与 `user_id` 完全不同 · 是有效信号源
3. **`item_int` 14 个字段 missing pattern signal volume 全部 ≥ `user_int` 同档** · 这是当前模型未显式利用的最大金矿

**3 条方向反转**:

- ❌ **time-aware 类 trick (T30 全局桶 / time-of-day / recency 加权) ROI 重新评估为低**: 实验数据 (EXP-030 gap=0.05 dev→LB gap +0.002~+0.003) 证明时间漂移本身**不是** valid→LB gap 主因
- ❌ **user-side feature engineering ROI 极低**: gap 普遍 < 0.03 · 上界 LogLoss reduction < 0.001
- ❌ **T34 user_id structural disable 是空操作**: 当前 `dataset.py` 已经不把 user_id 作为 model 输入（只用作 inference row id）

**3 条 Tier 0 方法学方向**:

- ✅ **T31 is-missing**: item-side 优先 · 信号体量 vol top 5 全在 item · 预期 LB +0.002~+0.005
- ✅ **seq_d truncation 扩长**: 当前 256 / p50=2706 · 浪费 > 80% rows · 预期 LB +0.002~+0.005 (条件: throughput 撑得住)
- ✅ **T33 NLIR (M18 已跑)**: 与 EDA 正交 · 架构 bet · 预期 LB +0.001~+0.003

---

## EDA 关键数据

### Q1 · user_id / item_id 频次分布

| 指标 | user_id | item_id |
|---|---|---|
| distinct | 500,000 | 231,479 |
| freq=1 占比 | **100.0%** | 45.4% |
| freq 2–10 | 0.0% | 41.0% |
| freq >10 | 0.0% | **13.6%** |
| max_freq | 1 | 7,591 |
| mean_freq | 1.00 | 9.07 |
| total_observations | 500,000 | 2,099,956 |

### Q2 · 按 signal volume (`n_null × |gap|`) 排序 TOP 10

| col | n_null | null_rate | gap | sig_vol | tier |
|---|---|---|---|---|---|
| **item_int_feats_83** | 1,862,640 | 88.7% | −0.020 | **36,451** | 软 bitmap |
| **item_int_feats_84** | 1,862,640 | 88.7% | −0.020 | **36,451** | 软 bitmap |
| **item_int_feats_85** | 1,862,640 | 88.7% | −0.020 | **36,451** | 软 bitmap |
| user_int_feats_109 | 1,837,178 | 87.5% | −0.015 | 27,678 | 软 bitmap |
| **item_int_feats_7** | 315,944 | 15.1% | +0.066 | 20,838 | 软 bitmap |
| **item_int_feats_12** | 315,944 | 15.1% | +0.066 | 20,838 | 软 bitmap |
| **item_int_feats_81** | 1,485,470 | 70.7% | +0.014 | 20,154 | 软 bitmap |
| user_int_feats_99 | 1,702,441 | 81.1% | −0.012 | 20,221 | 软 bitmap |
| **item_int_feats_8** | 1,296,932 | 61.8% | +0.013 | 16,492 | 软 bitmap |
| **item_int_feats_6** | 91,466 | 4.4% | +0.159 | 14,560 | 软 bitmap |

按 |gap| 排序 TOP 5 (硬信号 · n_null 小但 cvr_null 极端):

| col | n_null | cvr_null | cvr_notnull | gap | tier |
|---|---|---|---|---|---|
| item_int_feats_5 | 1,401 | 99.29% | 10.11% | +0.892 | 硬 vocab+1 |
| item_int_feats_9 | 1,401 | 99.29% | 10.11% | +0.892 | 硬 vocab+1 |
| item_int_feats_10 | 1,401 | 99.29% | 10.11% | +0.892 | 硬 vocab+1 |
| item_int_feats_13 | 1,406 | 99.22% | 10.11% | +0.891 | 硬 vocab+1 |
| user_int_feats_1 | 292 | 95.89% | 10.16% | +0.857 | 硬 vocab+1 |
| item_int_feats_16 | 1,979 | 70.24% | 10.11% | +0.601 | 硬 vocab+1 |

### Q4 · 序列长度 vs 当前 truncation

EDA 看的是 parquet 原始数组里的真实非 padding 长度 (max 反映 schema hard ceiling · 不是当前 train config)。

| domain | p50 | p75 | p90 | p95 | p99 | parquet max | 当前 train max_len | rows 被进一步截 |
|---|---|---|---|---|---|---|---|---|
| seq_a | 555 | 1,155 | 1,889 | 1,957 | 1,984 | 2,000 | 256 | **>50%** |
| seq_b | 531 | 1,108 | 1,905 | 1,959 | 1,988 | 2,000 | 256 | **>50%** |
| seq_c | 375 | 824 | 1,207 | 1,445 | 2,579 | 3,999 | 512 | 25-50% |
| seq_d | **2,706** | 3,430 | 3,712 | 3,845 | 3,923 | 4,000 | 512 | **>80%** |

### Q5 · time_diff 分布 (p50/p99 按天)

| domain | p25 | p50 | p75 | p95 | p99 | events/day (p50 推算) | 业务推断 |
|---|---|---|---|---|---|---|---|
| seq_a | 40 天 | **80 天** | 127 天 | 173 天 | **505 天** | 6.9 | 中频跨季 |
| seq_b | 31 天 | **99 天** | 136 天 | 169 天 | 176 天 | 5.4 | 中频跨季 |
| seq_c | 131 天 | **274 天** | 437 天 | 594 天 | **629 天** | 1.4 | 低频跨年 |
| seq_d | 8 天 | **19 天** | 33 天 | 51 天 | 56 天 | **143** | 高频近期 |

p99 跨度 11.2× 倍 (seq_d 56 天 vs seq_c 629 天)。

---

## 业务语义重新解读 (资深算法研究员视角)

### 1. 数据集本质 = "Domain Adaptation under 100% User Cold Start"

`user_id` 单次曝光 + train/test 池不重叠（Seed 主楼证实）→ 这不是 production pCVR 的简化版 · 是**极端化的 covariate shift**:

- 经典 cold start: 部分 test user 是新的
- 本赛题: **100% test user 是新的**
- model 学的是 **`(user_int, item_int, 4 序列) → CVR` 的纯函数** · 不能依赖任何 user history retrieval / lifetime cohort

**直接推论**:
- user_id embedding 在数学上必然过拟合 (与 single label 一一映射) · 当前 dataset.py 已 disable user_id 进 model · 正确处理
- T34 user_id structural disable 是空操作 · 撤回

### 2. 4 序列的真实业务语义

| domain | 长度密度 | time 跨度 | 业务推断 |
|---|---|---|---|
| seq_a / seq_b | 中长 (p50≈540) | 中期 (80~99 天 p50) | 中频跨季曝光记录 · A/B 可能冗余 |
| seq_c | 短稀 (p50=375) | 长期 (274 天 p50) | lifetime item anchor (Seed: feat47=item_id) |
| seq_d | **超长高密** (p50=2706) | 短期 (19 天 p50) | session-level fine-grained event log |

**seq_d 是当前最大的"被浪费"信号源**: 截到 256 / p50=2706 · 看到的不到 1/5 · 同时是 events/day=143 的高密度行为 · 业务上最接近 immediate intent。

### 3. valid→LB gap (0.022~0.030) 的真实来源

EXP-030 gap=0.05 配置实测 dev→LB gap +0.002~+0.003 · 即:

- 大部分 valid→LB gap 来自 **valid 在 train 时间末端共享局部模式** (train 总跨度 4 天 · valid 切尾 5%)
- 一旦 row-level gap 隔离 · gap 缩到 +0.002~+0.003 · 这才是真正的 covariate shift 大小
- 而 **真正 covariate shift 的来源是 user 池差异** · 不是时间漂移

**这条颠覆了 time-aware 类 trick 的优先级判断**:
- T30 time bucket boundaries (per-domain 或全局) ROI 上限受限
- recency reweight ROI 受限
- T31 is-missing / seq_d 扩长 / T33 NLIR 等**不依赖时间维度**的方向才是主战场

### 4. T31 实施细节修订

之前 EXP-044 默认所有 missing 都用 Scheme D (embedding +1 + bitmap)。EDA 数据揭示**应该分两段处理**:

| Tier | 触发条件 | 推荐方式 |
|---|---|---|
| **软 bitmap** | n_null × |gap| 大 (vol > 10k) | 加 bitmap 列 · 让 model 显式学 binary indicator |
| **硬 vocab+1** | n_null 小但 |gap| ≥ 0.5 | 仅扩 vocab +1 slot · embedding 自然学 anchor · 不加 bitmap |

按这个分类 · M19 实施清单:

**软 bitmap (item-side · vol 大优先)**:
`item_6, 7, 12, 8, 11, 81, 83, 84, 85` (9 字段 · 累计 vol ≈ 190k)

**硬 vocab+1 (item-side · 稀有强信号 · 仅扩 slot 不加 bitmap)**:
`item_5, 9, 10, 13, 16` (5 字段)

**user-side 留 M20+ 单独验证** (相关于 M19 的 item-side · 需要独立归因)

### 5. seq_d 扩长的可行性边界

- 当前 SwiGLU encoder 在 seq_d=512 下 half-train ~5h
- seq_d=1024 估算 FLOPs ×2 · 但 attention 是 O(N²) → 实际可能 ×3~4
- half-train 4 epoch wall-time 估算 8~12h · 在平台 ~10h 限制边缘
- **保险方案**: M20 启动 seq_d=1024 · 如果 throughput 撑不下 · 接受 E2/E3 ckpt 也算
- **风险下界**: 即使只跑到 E2 · seq_d=1024 vs seq_d=512 同 epoch dev 比较 · 仍可判方向

---

## 数据来源

- 原始 JSON: `docs/99-raw/eda/eda-fullscan-2026-05-13.json` (22.8 KB)
- 平台 job: `EDA-quantile-fullscan-1044rg` · 启动 2026-05-14 01:23 · 完成 01:39 · 16 min 33 s
- 实施: `src/eda_quantile.py` (vectorized · base64+gzip stdout blob 回流)
- 解码工具: `tools/decode_eda_blob.py`

---

## 后续动作

1. ✅ 落档 EXP-048 + 原始 JSON · commit
2. ✅ team-memory 加 3 条铁律 (user_id 一人一行 / item-side 主信号 / time 漂移温和)
3. → M19: T31 item-side is-missing (软 bitmap 9 字段 + 硬 vocab+1 5 字段)
4. → M20: seq_d truncation 256 → 1024 (单变量 · 复刻 M9 SOTA gap=0.05 path)
5. → M18 (T33 NLIR) 等明早回流 · 与 M19/M20 形成 3 个独立 Tier 0 方向并跑
