# EXP-036 · gap=0 T25 DIN Seed Sweep + Tokenization u4i3 · 4 Training Batch (2026-05-10)

**日期**：2026-05-10 CEST 16:10（西班牙下午）· 北京 22:10
**作者**：Agent（用户多轮 "请深入思考" 触发战略 pivot）
**状态**：✅ 4 job 全部 training 完成 · Evaluation 已提交（等 5/11 早 07:00 LB）
**关联**：EXP-035（今日早上的 Kaggle 金牌视角战略复盘）· EXP-019（gap=0 seed=42 × no DIN × E4 SOTA baseline 0.84637）· EXP-030（gap=0.05 baseline）

---

## TL;DR

**4 个 training job，3 gap=0 × T25 DIN seed sweep + 1 Tokenization u4i3**：

| Job | Best Dev | Best Epoch | Stop Reason |
|---|---|---|---|
| **seed=42 × gap=0 × DIN × 4ep** | **0.868924** ⭐ | E4 | max_epochs |
| seed=3 × gap=0 × DIN × 4ep | 0.867661 | E3 | logloss_rebound_epoch_3 |
| seed=7 × gap=0 × DIN × 4ep | 0.866622 | E3 | logloss_rebound_epoch_3 |
| Tokenization u4i3 × gap=0.05 × DIN × 4ep | 0.843743 | E3 | max_epochs |

**关键发现**：

1. **seed=42 × gap=0 × DIN × E4 Dev 0.868924 是项目史上最高 Dev**（超越 EXP-019 seed=42 × gap=0 × no DIN × E4 = 0.868619 **+0.000305**）。DIN bypass 首次在 gap=0 配置下显示正向 Dev 信号。
2. **gap=0 下 seed 对 DIN 敏感度高**：seed=42 独强，seed=3/7 被 LogLoss guard E3 停（未跑到 E4）。
3. **Tokenization u4i3 Dev 几乎持平 T25 B**（Δ+0.000047 = 0.5× 噪声），**但 LogLoss 更低 −0.00245**（校准信号更好）。
4. Evaluation 4 quota 全部用完，4 个 ckpt 都已 publish，5/11 早 07:00 CEST 看 LB。

---

## 背景：为什么做这 4 个 job

今日 CEST 09:30 用户反问 "Kaggle 金牌视角深入思考"，触发 EXP-035 战略复盘。发现 48h 0 LB SOTA 增益，主要错误是"过度 seed sweep + 在 gap=0.05 低山头上攀爬 + 忽略 DECEM Tokenization 情报"。

**决策**：
- 3 个 slot 用于 **gap=0 T25 DIN seed sweep**（切换山头到 SOTA 领地 0.84637）
- 1 个 slot 用于 **Tokenization 修改实验**（DECEM 明说 work 但未消化）

---

## 配置与 run.sh

### A · `t25-din-seed42-halftrain-gap0-4ep`

**Config**：
```bash
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu --seq_hash_vocab 500000
--enable_time_of_day_features
--valid_split_strategy time --valid_ratio 0.05
--item_conditioned_query --enable_din_interest
--keep_topk_ckpt 3
--seed 42 --batch_size 128 --num_epochs 4
# 注意：没有 --enable_row_valid_gap / --valid_gap_ratio (gap=0)
```

**Wall-time**: 6h17m

### B/C · seed=3 / seed=7 （同上，只换 --seed）

### D · `t-tokenization-u4i3-halftrain-gap05`

**Config**（vs T25 B baseline 的差异）:
```bash
--user_ns_tokens 4  # 原 5，少 1
--item_ns_tokens 3  # 原 2，多 1
--enable_row_valid_gap --valid_gap_ratio 0.05  # gap=0.05
--seed 42 --num_epochs 4
```

`T = num_queries×S + num_ns = 2×4 + (4+1+3) = 16`（和 baseline 相同，`d_model=64 % 16 = 0` ✅）

**Wall-time**: 5h14m

---

## 结果详细

### A · seed=42 × gap=0 × DIN × 4ep

| Epoch | step | Dev AUC | Val LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.864313 | — |
| 2 | 31882 | 0.866682 | — |
| 3 | 47823 | 0.868142 | — |
| **4** | **63764** | **0.868924** ⭐ | **0.22259** |

Top-K: rank1 E4 0.868924 / rank2 E3 0.868142 / rank3 E2 0.866682

**stop_reason**: `max_epochs_completed`（未触发 guard，轨迹单调上升）

### B · seed=3 × gap=0 × DIN × 4ep

| Epoch | step | Dev AUC | Val LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.863879 | — |
| 2 | 31882 | 0.867137 | — |
| **3** | **47823** | **0.867661** | **0.22459** |

**stop_reason**: `logloss_rebound_epoch_3`（E3 后 LogLoss 反弹触发 guard，没跑 E4）

### C · seed=7 × gap=0 × DIN × 4ep

| Epoch | step | Dev AUC | Val LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.864311 | — |
| 2 | 31882 | 0.866340 | — |
| **3** | **47823** | **0.866622** | **0.22530** |

**stop_reason**: `logloss_rebound_epoch_3`（同 seed=3，E3 后 guard 触发）

### D · Tokenization u4i3 × gap=0.05 × DIN × 4ep

| Epoch | step | Dev AUC | Val LogLoss |
|---|---|---|---|
| 1 | 15168 | 0.836686 | — |
| 2 | 30336 | 0.840100 | — |
| **3** | **45504** | **0.843743** | **0.27596** |
| 4 | 60672 | 0.843616 | 0.27576 |

**stop_reason**: `max_epochs_completed`（E3 > E4，但 guard 不触发说明 LogLoss 还在降）

Top-K: rank1 E3 / rank2 E4（E4 反而比 E3 dev 低 −0.00013 但 LogLoss 更低）

---

## 关键分析

### Finding 1：gap=0 seed=42 × DIN × E4 Dev 0.868924 是项目史最高 Dev

**严格对比**（grep EXP-019 line 42）：

| 对照 | E4 Dev | LogLoss | Config |
|---|---|---|---|
| **EXP-019 gap=0 seed=42 no-DIN** (icq-t13tf-halftrain) | 0.868619 | 0.22336 | gap=0 + ICQ |
| **今日 gap=0 seed=42 + DIN** | **0.868924** | **0.22259** | gap=0 + ICQ + DIN |
| **Δ (DIN gain in gap=0)** | **+0.000305** | **−0.00077** | 3× rerun 噪声 ±0.0001 |

**观察**：
- DIN 在 gap=0 下 **Dev +0.000305**（临界 3σ，勉强有信号）
- LogLoss **更低 −0.00077**（校准也改善）
- **但增益幅度远低于 gap=0.05 下的 DIN gain**（gap=0.05 实测 Dev +0.00125）

**机制推测**：
- gap=0 下 train data 多 5%（无 row_valid_gap），seq embedding 已经充分训练
- DIN 作为"额外 target-aware pass"的**边际贡献递减**
- 类似 diminishing returns：baseline 越强，附加模块增益越小

### Finding 2：gap=0 下 seed 对 DIN 敏感度极高

**对比 3 个 gap=0 × DIN seed**（E3 best）：

| Seed | E3 Dev | ΔvsEXP-019 seed=42 E3 (0.868195) |
|---|---|---|
| 42 | 0.868142 | −0.00005（基本持平）|
| 3 | 0.867661 | −0.00053 |
| 7 | 0.866622 | −0.00157 |

**观察**：
- seed=42 最强（独大）
- seed=3/7 比 seed=42 低 0.0005~0.0015
- seed=3/7 **被 LogLoss guard E3 停**，没跑到 E4

**对比 EXP-030 gap=0 × no DIN seed 分布**（5-seed LB 跨度 0.010）：
- seed=42 LB 0.84637（上沿）
- seed=3 LB 0.845329（第 2 高）
- seed=7 LB 0.836413（最低）

**推断**：
- gap=0 下 seed luck 很大（跨度 0.010）
- **DIN 不能改变 "seed luck 上沿规律"**：seed=42 在 DIN 和 no-DIN 下都是上沿
- seed=7 × DIN 预期 LB 仍接近 seed=7 × no-DIN × 0.836，**LB 最低**

### Finding 3：Tokenization u4i3 Dev 持平，LogLoss 更好

**严格对比**（grep T25 B 昨夜 LB 0.843433）：

| Job | E3 Dev | E3 LogLoss | E4 Dev | E4 LogLoss |
|---|---|---|---|---|
| **T25 B (user=5, item=2)** | 0.843696 | 0.27841 | — | — |
| **Tokenization u4i3 (user=4, item=3)** | 0.843743 | **0.27596** | 0.843616 | **0.27576** |
| Δ (E3) | +0.00005（噪声内） | **−0.00245** | | |

**观察**：
- Dev 几乎持平（0.5× 噪声）
- **LogLoss 明显更低**（校准提升，概率分布更尖锐）
- E4 Dev 微跌 −0.00013，E4 LogLoss 继续降（**Dev 和 LogLoss 背离**，有趣）

**机制推测**：
- u4i3 把 item 特征 14 fid 分成 3 组（~4.7 fid/组），比原 2 组（~7 fid/组）更细
- user 从 5 组变 4 组（~9 fid/组），粒度更粗
- **Dev 不变说明架构容量未改变**（T=16 同），**LogLoss 降说明输出概率更 confident**
- **LB 预测**：LogLoss 和 LB 相关但不是直接预测；可能 LB 略升 +0.0005~0.002，也可能持平

### Finding 4：num_epochs=4 修复的价值在今日数据上明显

**A job 跑满 E4 = 0.868924**（vs E3 = 0.868142）**Δ+0.000782 Dev**。

如果按之前 num_epochs=3，我们会停在 E3，**丢失 +0.000782 Dev**。

EXP-019 E3→E4 Δ+0.000654（+0.000654 → +0.000782 差 +0.000128），**今日 DIN × gap=0 在 E4 增益略大于 EXP-019 无 DIN baseline**。

---

## LB 预测（严格定性，不给数字）

按新铁律"agent 不再给具体概率数字"，只给定性 ceiling：

| Ckpt | Config | LB Ceiling（定性） |
|---|---|---|
| **A · seed=42 × gap=0 × DIN × E4** | best so far | **可能 0.846~0.850**（上限若 DIN gap=0 gain 等比例加到 SOTA 0.84637） |
| B · seed=3 × gap=0 × DIN × E3 | 上沿 seed luck | 可能 0.845~0.848（seed=3 历史好运）|
| C · seed=7 × gap=0 × DIN × E3 | 下沿 seed luck | 可能 0.838~0.842（seed=7 历史差运）|
| D · Tokenization u4i3 × gap=0.05 × E3 | 新方向 | 可能 0.843~0.846（和 T25 B 持平到微升）|

**真实 LB 要等 5/11 早 07:00 CEST**。

---

## Evaluation 策略

**4 quota 已用完**（用户已提交 4 个 evaluation）：
- seed=42 × gap=0 × DIN × E4
- seed=3 × gap=0 × DIN × E3
- seed=7 × gap=0 × DIN × E3
- Tokenization u4i3 × gap=0.05 × E3

**5/11 早 07:00 CEST** LB 上榜。

---

## 战略含义

### 如果 A (seed=42 gap=0 DIN E4) LB ≥ 0.847

- **突破 5/4 SOTA 0.84637**，**新 LB 记录**
- 说明 "gap=0 × DIN × seed=42" 叠加确实 work
- 下一步：seed sweep 寻 gap=0 × DIN 5-seed 上沿（seed=13/17/23 等）

### 如果 A LB ∈ [0.8464, 0.847)

- 与 SOTA 持平或微突破（在 seed 噪声带内）
- **gap=0 × DIN 叠加效果不明显**（data completeness 已消耗大部分增益）
- 下一步：pivot 到 T28 DIN block-integrated 或 Tokenization 深入

### 如果 A LB < 0.845

- **gap=0 × DIN 无效甚至轻微负向**
- Dev +0.000305 不足以跨越 gap=0 的 Dev→LB gap 噪声
- 下一步：**放弃 gap=0 × DIN 路线**，专心 Tokenization / T28 / S4

---

## 下一批 training slot 计划（CEST 16:10 起）

等 5/11 早 07:00 CEST LB 数据再决定。

**3 候选路径**（基于 5/11 LB 结果）：

| 情景 | 推荐下批 3 slot |
|---|---|
| A LB ≥ 0.847（SOTA 突破）| gap=0 × DIN × seed=13/17/23（seed sweep 找上沿）|
| A LB ∈ [0.8464, 0.847)（持平）| T28 DIN block-integrated × 2 + Tokenization 变种 × 1 |
| A LB < 0.845（负向）| Tokenization 深入 × 2 + S4 100% data × 1 |

---

## 教训

### 教训 1：gap=0 × DIN 可能**边际增益递减**

- gap=0.05 下 DIN gain +0.00125
- gap=0 下 DIN gain +0.000305（1/4）
- 数据越完整，附加模块 ROI 越低

### 教训 2：LogLoss guard 在 gap=0 下对 seed 敏感

- seed=42 过了 guard 跑到 E4
- seed=3/7 被 E3 guard 停
- **未来 gap=0 × DIN 实验应默认 `--logloss_rebound_delta 0.003 --logloss_rebound_patience 2`** 松 guard

### 教训 3：num_epochs=4 在 gap=0 × DIN × seed=42 下关键

- E3→E4 Dev +0.000782（超 EXP-019 无 DIN 的 +0.000654）
- **验证今早 num_epochs 修正的价值**

### 教训 4：Tokenization 是"Dev 持平，LogLoss 分叉"的有趣信号

- u4i3 Dev +0.00005（噪声）但 LogLoss −0.00245（明显）
- **单看 Dev 会错过这个信号**
- 未来方法学实验应**同时看 Dev 和 LogLoss**

---

## 数据源（严格 grep 验证）

- 今日训练日志 `~/Downloads/train_log.txt` line 1~94
- EXP-019 line 42 (gap=0 seed=42 × no DIN × E4 = 0.868619)
- EXP-019 line 116 (gap=0 seed=42 × no DIN × E4 LB = 0.84637 SOTA)
- EXP-030 line 12, 25, 185 (gap=0.05 baseline + seed 分布)
- 昨夜 evaluation: T25 B LB 0.843433 (2026-05-10 early CEST)

---

## 后续

- 2026-05-11 早 07:00 CEST: LB 更新，补 Results 章节
- 基于 LB 结果决定 5/11 training 策略
