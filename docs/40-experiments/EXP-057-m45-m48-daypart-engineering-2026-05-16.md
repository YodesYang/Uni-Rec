# EXP-057 · M45-M48 Daypart Engineering 4-job Batch · 2026-05-16

> **Status**: 🟡 In-flight (5/16 CEST 10:50 起 · 4 platform jobs · 预期 ~CEST 14:00~16:00 dev 回流)
> **Date**: 2026-05-16 CEST 10:55 起草
> **Trigger**: M23 LB 0.836631 强负 (路径 A 死透) → EXP-056 EDA Q6/Q7 揭示 weekday 是真实 daypart 信号 (周末 CVR 0.024~0.034 vs 工作日 0.10~0.12 · 5× gap)
> **Related**: EXP-056 (Tier 0 EDA · 元宵节假设证伪) · EXP-052 (M22 SOTA) · EXP-055 §16~17 (path A 死透 audit)

---

## TL;DR · Path A 死透后 daypart engineering 4-job batch

**EXP-056 EDA Q6 实证**:
- 周末 (02-28/03-01 · 7.2% rows) CVR 0.024~0.034
- 工作日 (03-02/03/04) CVR 0.10~0.12
- **5× CVR gap = data-driven low-noise filter target**

**4 jobs 沿 daypart filter / reweight 维度展开**:

| Job | Config | Train data | 期望 LB | 假设 |
|---|---|---|---|---|
| **M45** | M22 + weekend filter (cutoff=1772409600) | 92.8% (1.95M) | +0.0005~+0.001 | 剔除 noise 周末 → CVR base rate 更准 |
| **M46** | M45 + temporal_weight 2.5 | 92.8% (reweight) | +0.0005~+0.0012 | 双 daypart stack 增量叠加 |
| **M47** | M45 + temporal_weight 5.0 | 92.8% (reweight) | +0.0005~+0.0015 | alpha grid 强化版 |
| **M48** | M22 + 03-04 only filter (cutoff=1772582400) | 24.6% (515k) | +0.001 / −0.005 | 极端 daypart match · 高 variance |

**全部 4 jobs · seed=42 · gap=0 · M22 base config · 仅在 daypart 维度变化** (合 5/16 早立的 "5/22 前 seed=42 lock" 铁律)。

---

## 1. EXP-056 EDA mechanism summary (引用)

### 1.1 周末数据是 noise · 不是真信号

| Day | Rows | Pct | CVR |
|---|---|---|---|
| 02-28 (周六) | 86,691 | 4.1% | **0.024** |
| 03-01 (周日) | 65,039 | 3.1% | **0.034** |
| 03-02 (周一) | 119,861 | 5.7% | 0.119 |
| 03-03 (周二) | 1,312,781 | **62.5%** | 0.101 |
| 03-04 (周三) | 515,584 | 24.6% | **0.122** ⭐ test 同日 |

**Insight**: 周末 CVR 0.024~0.034 比工作日 0.10~0.12 低 **3~5×** · 这是真实 distribution shift · 与 test (周三 工作日 CVR ~0.12) 完全不同。

### 1.2 Test daypart hour 落 train 空白区

EXP-056 EDA Q7 显示:
- Train 主峰: 北京 06~09 + 22~00 (合计 51% rows)
- Test 时段: 北京 09:50~14:49 (5h)
- Train 同时段覆盖 = **仅 13.7% rows**
- 北京 10/11 hour 几乎空 (0.1%/0.2%)

→ **数据本身限制** · model 对 test daypart 的预测能力有 ceiling。

---

## 2. 4 Jobs Configuration

### M45 · weekend filter alone (单变量 daypart cleanup)

```bash
--user_train_timestamp_min 1772409600  # UTC 03-02 00:00
# 其余 = M22 baseline
```

**Train rows**: 1.95M (剔除 02-28 + 03-01 共 7.2%)

**机制**: 直接剔除周末 noise · model 学到的 CVR base rate ≈ 工作日 0.10~0.12 · 与 test 一致

**期望 dev**: 持平 M22 (~0.8685 ± 0.0005) · 因 valid set 5% 尾部 (主要 03-04 末段) 不受影响
**期望 LB**: +0.0005~+0.001 (基于 weekend 7.2% noise removal)

---

### M46 · M45 + temporal_weight 2.5

```bash
--user_train_timestamp_min 1772409600
--temporal_weight_alpha 2.5
```

**机制**: 在 M45 weekend filter 基础上 · 再 reweight 让 train 末尾 (03-04 同日) 权重更高

**期望 LB**: +0.0005~+0.0012 (M45 + M41 reweight 部分增量重叠 · stack 假设互不冲突)

⚠️ **stack 风险**: 5/14~5/16 6/6 M22+X stack 失败 (除 M41 +0.0003)。但 M46 = M45 (weekend 同方向 daypart) + temporal_weight (同方向 daypart) · 两者机制都是 daypart filter · 增量可能加性。

---

### M47 · M45 + temporal_weight 5.0

```bash
--user_train_timestamp_min 1772409600
--temporal_weight_alpha 5.0
```

**机制**: alpha=5.0 让 train 末尾 03-04 weight × ~150 (vs alpha=2.5 × ~12) · 极强 reweight

**期望 LB**: +0.0005~+0.0015 · 但 alpha 太大可能 saturate · 也可能 overfit 03-04

**Insight**: alpha grid (1.0 · 2.5 · 5.0 · 10.0) 三点扫 · 找最优值

---

### M48 · M22 + 仅保留 03-04 train (极端 daypart filter)

```bash
--user_train_timestamp_min 1772582400  # UTC 03-04 00:00
```

**Train rows**: 515k (24.6% · 仅 03-04)

**机制**: 完全剔除非 test 同日数据 · model 仅学 03-04 daypart pattern

**期望**: 高 variance · +0.001 / −0.005

**双 hypothesis**:
- (a) 03-04 同分布 train 让 model 完美 match test → +0.001 LB (信号 alignment 极佳)
- (b) train data 缩到 24.6% (515k rows) · 模型欠拟合 → −0.005 LB (data insufficiency)

**信息价值**: 极高 · 决定 daypart match vs data quantity 谁 dominant。

⚠️ **wall-time 异常**: 24.6% data → 训练时间 ~1h15m (vs 4h45m baseline) · 早回流。

---

## 3. 决策预案 · 5/16 晚 1 evaluation quota

| 场景 | 5/16 晚 evaluate 选 | 决断 |
|---|---|---|
| M45 dev > 0.8688 (强真信号) | **M45 ckpt LB** | weekend filter alone 真信号 |
| M46/M47 dev > M45 dev 显著 | 取 dev 高者 | 双 daypart stack work |
| M48 dev > M45 dev (in-band) | **M48 ckpt LB** | 极端 daypart match 信号 |
| M48 dev < 0.866 (data 不足) | 不评 M48 · 评 M45 | data quantity 重要 |
| 全部持平 M22 | **M45 ckpt LB** | binary 判决 weekend filter |

**默认决断** = M45 ckpt LB (基础数据点 · 决 weekend filter 是否 work)

---

## 4. 5/17 早 4 quota 预案 (基于 5/16 晚 1 LB)

### 场景 A · M45 LB > 0.848 (真信号)

5/17 早 4 quota:
- M46 ckpt LB (验 stack)
- M47 ckpt LB (验 alpha 强化)
- M48 ckpt LB (验极端 daypart)
- 备选 M22 cross-seed s11 (5/22 前禁 cross-seed 铁律例外: 已 LB 破 SOTA 阈值)

### 场景 B · M45 LB ∈ [0.847, 0.848] (弱信号 · band 内)

5/17 早 4 quota:
- M46 ckpt LB
- M48 ckpt LB
- 备选: M30 (Trick 6 alone) / M37 (NLIR) 闭环老候选

### 场景 C · M45 LB ≤ 0.847 (持平 M22)

weekend filter alone 不 work · 但 M46/M47 stack 仍可能。

5/17 早 4 quota:
- M46 ckpt LB
- M47 ckpt LB
- M48 ckpt LB
- 备选: M22 cross-seed (5/22 前 final 阶段例外可考虑)

### 场景 D · M45 LB < 0.846 (强负 · weekend filter 反向)

→ EDA-mechanism 数据驱动也失败 · 必须切 path B

5/17 早 4 quota:
- M46/M47/M48 LB (闭环 daypart 路径)
- 5/17 下午起 path B preparation (UE 分离 ADR-014 reactivate)

---

## 5. 真实 Top 20 概率估算 (5/16 CEST 10:55)

- 5/14 EXP-051: 25~35%
- 5/15 EXP-055: 10~15%
- 5/16 09:00 (M40/M41 LB 后): 10~15%
- 5/16 10:30 (M23 LB −0.011 后): 5~8%
- **5/16 10:55 (4 jobs 启动后)**: **5~10%** (M45/46/47/48 都 work 才能破)

**距 0.85 还差**: 0.847846 (M41 SOTA) → 0.85 = **+0.0022**

**M45/46/47 累加增量上限** (假设全 work):
- M45: +0.001
- M46 - M45: +0.0005 (temporal_weight 2.5 增量 部分重叠)
- M47 - M46: +0.0003 (alpha 5 vs 2.5 边际收益)
- 累加: **+0.0018**

→ 4 jobs 全 work + cross-seed luck = 接近 0.85 · 但**非常乐观估计**。

---

## 6. 风险清单

### Risk 1 · stack 第 2 步连续失败模式可能复现 (M46/M47)

5/14~5/16 历史:
- M22 + Trick 1 (M28): −0.0011
- M22 + Trick 6 (M34): TBD
- M22 + NLIR (M37): TBD
- **M22 + DCN cross (M23): −0.0110 ⚠️ 强负**
- M22 + 5-cluster (M33): −0.0029
- **M22 + temporal_weight (M41): +0.0003 ⭐ 唯一 work**

→ 6/6 stack · 1/6 弱信号 · M46/M47 大概率持平或微负。

### Risk 2 · M48 极端 filter data 不足

24.6% rows (515k) 可能不够学 user/item embedding · dev 端就显著负面。

### Risk 3 · cutoff 算错 (历史教训)

我之前给的 1772554800 是错的 · commit `83efe1d` 已修正。但用户上传时**必须 verify Args dict**:
- M45/46/47: `user_train_timestamp_min: 1772409600` ⭐
- M48: `user_train_timestamp_min: 1772582400` ⭐
- 错误值会导致 silent bug (剔除 75% data 但仍跑完)

### Risk 4 · 平台 ckpt 24h 自动清理

- M48 最快 (~CEST 12:15 完成) · 必须 ~CEST 24:15 5/16 前 publish
- M45/46/47 ~CEST 16:00 完成 · 必须 ~5/17 早 publish

---

## 7. References

- EXP-056 §1-2 (EDA Q6/Q7 完整数据)
- EXP-055 §16-17 (path A 死透 audit + M23 LB 重大判读)
- src/dataset.py line 1761-1762 (timestamp_min row filter 实现)
- src/train.py line 185-198 (--user_train_timestamp_min CLI · commit 83efe1d 含 cutoff fix)
- M41 LB +0.000254 (唯一 M22 + X stack work 的弱信号 · 机制 = weekend reweight)
- M23 LB 0.836631 (DCN cross 路径强负 · path A 死透触发本 EXP)
