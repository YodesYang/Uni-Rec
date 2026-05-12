# EXP-040 · M5 gap=0 × T25 DIN bypass × seed=3 Half-train · 2026-05-12

> **Status**: ⚠️ **负向证伪**（seed=3 在 gap=0 × DIN bypass 下 LB 退步 Δ−0.0035）· M9 seed=42 补回遗憾 training 已起
> **日期**: 2026-05-12 CEST 12:53 启动 · CEST 23:28 完成 · CEST 18:38 LB 回流
> **关联**：[EXP-019](EXP-019-icq-halftrain-2026-05-04.md)（SOTA 0.84637）· [EXP-036 A 过期](EXP-036-gap0-t25-seed-sweep-tokenization-2026-05-10.md)（seed=42 同组合 Dev 0.868924 永久丢失）· [EXP-030 icq_seed3](../40-experiments/)（seed=3 no-DIN LB 0.845329）

---

## TL;DR

消化"**EXP-036 A Dev 0.868924 过期丢失**"的战略缺口 + 昨日 LB 回流发现 "Dev→LB gap 扩大是方法学负向最强信号"，今日启动 M5 填补 gap=0 × DIN × seed=3 数据点。

**结果**：**LB 0.841812** vs 同 seed=3 × no-DIN 基线 0.845329 = **Δ−0.003517 明确负向**。

**三大发现**：
1. **T25 bypass 的"黄金三元组"** = bypass + seed=42 + gap=0.05 · 偏离任一维度显著变差
2. **Dev→LB gap 是方法学诚实度的尺** · M5 gap +0.0266（史上最高）确证负向
3. **Seed 对 DIN × gap 交互极度敏感** · seed=3 在 gap=0.05 × DIN 仍 work（T25 bypass s3 LB 0.843069），但切到 gap=0 × DIN 立刻 −0.004 LB

---

## M5 训练详情

### Config (`M5-t25bypass-gap0-seed3-4ep`)

```bash
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2
--num_queries 2 --ns_groups_json ""
--seq_encoder_type swiglu --seq_hash_vocab 500000
--enable_time_of_day_features
--valid_split_strategy time --valid_ratio 0.05
--item_conditioned_query --enable_din_interest         # T25 DIN bypass
--seed 3
--batch_size 128 --num_epochs 4
--logloss_rebound_delta 0.003 --logloss_rebound_patience 2  # 松 guard
--enable_torch_compile --enable_amp --amp_dtype bfloat16
# 注意: 无 --enable_row_valid_gap → gap=0 领地
```

### 训练轨迹

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.863486 | — |
| 2 | 31882 | 0.866514 | — |
| 3 | 47823 | 0.867485 | — |
| **4** ⭐ | **63764** | **0.868450** | **0.22329** |

**stop_reason**: `max_epochs_completed`（单调上升，无 guard 触发）
**Wall-time**: 4h37m（compile+AMP on）

### LB

- **eval_id**: 1778602914951
- **LB AUC**: **0.841812** / inference 514.22s

---

## 严格判决对照表

**历史关键锚点**（全部 Dev 在 gap=0 / no-gap 领地）:

| Ckpt | Seed | DIN | Gap | Dev | LB | Dev→LB gap |
|---|---|---|---|---|---|---|
| **EXP-019 icq-t13tf** | 42 | ❌ | 0 | 0.868619 | **0.84637** ⭐ SOTA | +0.0223 |
| **EXP-030 icq_seed3** | 3 | ❌ | 0 | ~0.8689 | **0.845329** #2 | ~+0.0236 |
| EXP-036 A (过期丢失) | 42 | ✅ bypass | 0 | **0.868924** | 永久丢失 | — |
| EXP-036 B (过期丢失) | 3 | ✅ bypass | 0 | 0.867661 (guard E3 停) | 永久丢失 | — |
| **M5 今日** ⚠️ | **3** | ✅ bypass | 0 | **0.868450** | **0.841812** ❌ | **+0.0266** 史上最高 |

### Finding 1: M5 vs EXP-030 icq_seed3 (加 DIN 单变量)

相同：seed=3 + gap=0 + ICQ + hash500k + timefeat
差异：M5 加了 T25 DIN bypass

- Dev: 0.8689 → 0.868450 (Δ**−0.00045**) · 近持平
- LB: 0.845329 → 0.841812 (Δ**−0.003517**) · 明确负向

**结论**：**T25 DIN bypass 在 gap=0 × seed=3 上不 work 甚至负向**。

### Finding 2: M5 vs EXP-019 SOTA (seed + DIN 双变量)

- Dev: 0.868619 → 0.868450 (Δ−0.00017) · 持平
- LB: 0.84637 → 0.841812 (Δ**−0.004558**) · 严重负向

**结论**：seed=3 × gap=0 × DIN 叠加 = 比 SOTA 差近半 percent。

### Finding 3: Dev→LB gap 史上最高档 (+0.0266)

**gap 扩大对照**：

| Ckpt | Dev→LB gap | 备注 |
|---|---|---|
| T25 bypass s42 gap=0.05 (T25 B) | **+0.000263** ⭐ 最小 | 黄金 |
| T29 vocabtier s42 gap=0.05 (M3) | +0.001026 | 第 2 小 |
| EXP-030 baseline gap=0.05 | +0.003176 | 正常 |
| T28 integrated s3 gap=0.05 (M1) | +0.003390 | 扩 13× vs T25 bypass |
| **M5 DIN bypass s3 gap=0** | **+0.0266** | **扩 100× vs T25 bypass · 项目史上最高** |
| EXP-019 baseline gap=0 | +0.0223 | gap=0 领地原生偏高 |

**机制**：
- gap=0 领地 dev 本来就含"训练期污染"（EXP-022 C6 证 dev 0.868 实为 dev 0.838 加 +0.030 memorization）
- DIN × seed=3 × gap=0 **进一步放大这种污染**（memorization 从 +0.022 变 +0.027）
- DIN bypass 在 seed=3 下学到了"近未来 target pattern"但在远未来 test 失效

### Finding 4: M5 vs M1 (T28 integrated s3 gap=0.05) 的惊人巧合

- M1 T28 integrated s3 gap=0.05: LB 0.841807 / Dev 0.845197
- M5 T25 bypass s3 gap=0: LB **0.841812** / Dev 0.868450

**LB 差 0.000005**（几乎 bit-identical）!

**机制假说**（待更多数据验证）：
- seed=3 在 **任何 DIN 变体 + 任何 gap** 配置下 LB 收敛到 ~0.8418
- **seed=3 对 test 分布存在系统性敏感**（可能是某些 sparse embedding 初始化路径在 seed=3 下映射到 test 不友好的子空间）
- "DIN × seed=3 = LB 0.8418" 是一个需要注意的**稳定吸引子**

---

## 核心洞察：T25 bypass 的"黄金三元组"

**已知所有 DIN bypass 相关 LB 结果**：

| Config | LB | 判决 |
|---|---|---|
| T25 bypass × seed=42 × gap=0.05 (T25 B) | **0.843433** | ⭐ 黄金 |
| T25 bypass × seed=3 × gap=0.05 | 0.843069 | ✅ 仍 work |
| T25 bypass × seed=7 × gap=0.05 | 0.842373 | ⚪ 持平 |
| T25 bypass × seed=11 × gap=0.05 | 0.841317 | ⚪ 持平 |
| T25 bypass × seed=5 × gap=0.05 | 0.840166 | ⚪ 持平 |
| **T25 bypass × seed=3 × gap=0 (M5)** | **0.841812** | ❌ **负向** |
| T25 bypass × seed=42 × gap=0 (EXP-036 A 过期) | 丢失 | 未知 |

**三元组规律**：
- `bypass + 任何 seed + gap=0.05` → **全部 LB ≥ 0.840**（稳定正向）
- `bypass + seed=42 + gap=0.05` → **LB 0.843433 上沿**（黄金配方）
- `bypass + seed=3 + gap=0` → **LB 0.841812 负向退步**（偏离三元组 2 维）
- `bypass + seed=42 + gap=0` → **唯一未知**（EXP-036 A 过期），M9 正在补回

### 铁律候选

1. **DIN bypass 的最优领地是 gap=0.05**，gap=0 下的迁移充满不确定性
2. **seed=42 是 DIN 方向的"幸运 seed"**（gap=0.05 上沿 / gap=0 EXP-036 A Dev 项目最高），其他 seed 在 DIN × gap=0 下表现不稳定
3. **Dev→LB gap 扩 2× 以上 = 方法学负向最强信号**（团队已有铁律，M5 再次证实）

---

## 战略含义：M9 的绝对必要性

**M9 = M5 的 seed=42 版本，唯一能答复"T25 bypass 在 gap=0 × seed=42 到底 work 不 work"**。

情景分析：

### M9 LB ≥ 0.847 → 破 SOTA 成立
- 说明 "DIN × gap=0 × seed=42" 是 work 的特殊组合
- seed=3 是 gap=0 下的坏种子（M5 证实）
- 起 M10 三重叠加（DIN × T29 × gap=0 × seed=42）冲 **0.849~0.851**

### M9 LB ∈ [0.844, 0.847] → 持平 SOTA
- DIN 在 gap=0 有增益但不大
- 战略：继续 gap=0.05 叠加路线 + seed sweep 验证

### M9 LB < 0.844 → DIN × gap=0 方向**彻底关闭**
- "黄金三元组" 铁律固化：DIN bypass 仅在 gap=0.05 × 特定 seed 下 work
- Top 20 只能靠 gap=0.05 叠加 + FaFE 等新方向实现
- 预期 ceiling 降到 0.847~0.848（T25 + T29 叠加），Top 20 难度大增

---

## 下一步

- ✅ **M9 已启动**（CEST 18:45）· `M9-t25bp-gap0-seed42-4ep`
- ⏳ 等 M6/M7/M8 今晚/明早陆续回流（gap=0.05 × 叠加 / FaFE）
- ⏳ 明早 CEST ~04:00 M9 回流 · 根据上述情景分析决定

---

## 新铁律候选（待团队确认后入 team-memory）

1. **DIN bypass 的 "黄金三元组" 铁律**：bypass + seed=42 + gap=0.05 是 DIN 方向**唯一已验证 + 稳定**的配置。偏离任一维度都需要新的实验验证，不能外推。
2. **gap=0 × DIN 交互的 seed 敏感性铁律**：在 gap=0.05 上不同 seed 的 DIN LB 跨度 ~0.003（仍全部正向），但切到 gap=0 可能跨度 >0.005 且可能某些 seed 负向。**新方法学 × gap=0 至少需要 2 seed 验证才能下 "work" 判决**。
3. **"DIN × seed=3 = 0.8418 吸引子" 假说**：M1 和 M5 LB 差 0.000005 的惊人巧合值得更多实验验证；若下一个 seed=3 × DIN 变体仍 ~0.842，则吸引子假说成立。
