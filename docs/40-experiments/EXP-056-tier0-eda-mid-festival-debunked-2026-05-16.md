# EXP-056 · Tier 0 EDA · 元宵节假设证伪 + Weekday/Hour 真实 daypart 机制 · 2026-05-16

> **Status**: ✅ EDA 完成 · 5/15 元宵节假设彻底证伪 · 真实 daypart shift = weekday × hour
> **Date**: 2026-05-16 CEST 10:15
> **Trigger**: 用户 5/16 09:35 决断 "EDA 才是数据科学第一步" · agent 扩展 src/eda_quantile.py 加 Q6/Q7 · platform job 16 min · pretty summary 完整捕获
> **Related**: EXP-048 (Q1/Q2/Q4/Q5 fullscan) · EXP-052 §15 元宵节认知 (本 EXP 证伪) · M40 (LB 半负 · 现在解释为 dow 真信号被删) · M41 (LB +0.000254 · 机制重新解读)

---

## TL;DR · 三大反转

1. **元宵节假设证伪**: 03-03 (元宵节) CVR=**0.10072** · **不是 peak**。**03-04 节后翌日 CVR=0.12164 = 项目最高**。
2. **真实 CVR 机制 = Weekday vs Weekend**: 周末 (02-28/03-01) CVR=**0.024~0.034** · 工作日 CVR=**0.10~0.12** · gap **3~5×**!
3. **Hour 严重 daypart shift**: Train 主峰 = 北京 06~09 + 22~00 (合计 51%+) · **Test 时段 (北京 09:50~14:49) = train 仅 13.7% rows · 北京 10/11 几乎空 (0.1%/0.2%)**

→ **5/15 23:30 元宵节认知整体方向是错的** · M40 LB 半负是因为**删了真实有用信号 dow**(区分 weekday vs weekend) · M41 LB +0.000254 真信号是因为 reweight 把**周末低 CVR 数据 weight 降下来**。

---

## 1. Q6 · CVR by Day 完整数据

| 日期 | 农历 | weekday | cnt% | CVR |
|---|---|---|---|---|
| 02-28 | 正月十二 | **周六** (5) | 4.1% (86,691) | **0.02412** ⚠️ 极低 |
| 03-01 | 正月十三 | **周日** (6) | 3.1% (65,039) | **0.03378** ⚠️ 低 |
| 03-02 | 正月十四 | 周一 (0) | 5.7% (119,861) | 0.11926 |
| **03-03** ⭐ | **正月十五元宵节** | **周二** (1) | **62.5%** (1,312,781) | **0.10072** |
| **03-04** | 正月十六 | **周三** (2) | **24.6%** (515,584) | **0.12164** ⭐ test 同日 |

**关键**:
- 03-03 元宵节占 62.5% rows = sampling artifact (**不是流量峰** · 而是**采样偏差**)
- 03-04 (test 同日 · 占 24.6%) CVR **0.122 = 项目最高**
- 周末 02-28/03-01 CVR **5× 低于工作日**

---

## 2. Q7 · CVR by Hour-of-Day (北京时间) 完整数据

| 北京 hour | cnt | cnt% | CVR |
|---|---|---|---|
| 08:00 | 361,027 | **17.2%** ⭐ peak1 | 0.1104 |
| 07:00 | 324,851 | **15.5%** | 0.0979 |
| 09:00 | 229,203 | 10.9% | 0.1117 |
| 06:00 | 222,456 | 10.6% | 0.1014 |
| 00:00 | 178,390 | 8.5% | 0.0960 |
| 05:00 | 125,299 | 6.0% | 0.0958 |
| 23:00 | 118,577 | 5.6% | 0.0910 |
| 01:00 | 95,380 | 4.5% | 0.0996 |
| 04:00 | 73,652 | 3.5% | 0.0985 |
| 02:00 | 68,509 | 3.3% | 0.0945 |
| 03:00 | 62,437 | 3.0% | 0.1025 |
| 21:00 | 34,254 | 1.6% | 0.0977 |
| 20:00 | 31,059 | 1.5% | 0.0996 |
| 22:00 | 28,593 | 1.4% | 0.0867 |
| 19:00 | 20,813 | 1.0% | 0.1034 |
| 17:00 | 17,308 | 0.8% | 0.0999 |
| 16:00 | 17,206 | 0.8% | 0.0900 |
| **13:00** | 17,184 | **0.8%** | **0.1168** ← test peak |
| 18:00 | 16,863 | 0.8% | 0.1203 |
| **12:00** | 16,541 | **0.8%** | **0.1269** ⭐ CVR 最高但 cnt 低 |
| **14:00** | 16,202 | **0.8%** | 0.0982 |
| 15:00 | 16,120 | 0.8% | 0.0993 |
| **11:00** | **4,915** | **0.2%** | **0.0612** ⚠️ 几乎空 |
| **10:00** | **3,117** | **0.1%** | **0.0638** ⚠️ 几乎空 |

**Test 时段 (北京 09:50 ~ 14:49)** vs train 同时段:
- Train 北京 09~14 hour 合计: **287,162 rows = 13.7% of train**
- Train 北京 10/11 几乎空 (0.1%/0.2%) · model 学不充分
- Test 100% rows 都在 09~14 · 落入 train 数据空白区

---

## 3. Train CVR Profile 综合

### 全 train CVR
- Total: 213,518 / 2,099,956 = **10.17%** (global mean)

### Train CVR by daypart bucket
- 工作日 (03-02/03/04 · 92.7% rows): **0.106 avg**
- 周末 (02-28/03-01 · 7.2% rows): **0.028 avg** (5× 低)

### Train CVR by hour bucket (北京)
- Test 时段 (09~14 · 13.7% rows): **0.111 avg** (略高于 mean)
- 早高峰 (06~09 · 54.2% rows): **0.105 avg** (接近 mean)
- 凌晨 (00~05 · 26% rows): **0.097 avg** (略低)
- 晚 (15~23 · 22% rows): **0.097 avg** (略低)

→ **CVR 波动主要在 weekday vs weekend** · hour 内波动较小

---

## 4. 假设证伪 · 5/15 元宵节认知整体错误

### 5/15 23:30 元宵节认知 (现在证伪)

> "03-03 元宵节 = train 流量峰 (63%) → 元宵节高 CVR · test 节后翌日 = 元宵节余热未退"

**实际**:
- 03-03 占 63% 是**采样偏差** (sampling design) · 不是流量峰
- 03-03 CVR 0.10 · **平均水平** · 不是峰值
- 03-04 CVR 0.122 · **比元宵节高** · 不是"节后回归低 CVR"

→ "元宵节 daypart shift" 概念**完全错** · 真实是 **weekday daypart**

### 修订 5/15 EXP-055 §15 元宵节认知

| 5/15 23:30 假设 | 5/16 EDA 实证 |
|---|---|
| 元宵节 CVR 高 → train 主体偏 | ❌ 元宵节 CVR = 平均 (0.10) |
| 03-04 = 元宵节翌日 = 节后回归低 CVR | ❌ 03-04 CVR 0.122 = 最高 |
| dow 学到 dow=1 元宵节 pattern → 误导信号 | ❌ dow 学到 dow=5/6 周末低 CVR · **真信号** |
| temporal_weight 修正 03-04 underweight | ⚠️ 部分对: weight 末尾 = weight 工作日 + weight 03-04 |
| daypart shift mechanism = 节日 pattern | ❌ 真正 mechanism = weekday + hour |

### 修订 M40 LB −0.000661 解读

之前 (元宵节假设): "M40 删 dow 移除元宵节 overfit 信号 · LB 应持平或微涨 · 实际 LB 跌 → dow 不是纯误导"

**真实 (5/16 EDA)**:
- dow 在 train 上学到 **weekday=5/6 (周末) → CVR 0.024~0.034 (5× 低于工作日)**
- test 是 weekday=2 (周三 · 工作日) · model 应预测 CVR ≈ 工作日 mean 0.10~0.12
- 删 dow → model 不知 weekday · 用全 train mean 0.10 · **失去 weekday discrimination**
- LB 跌 −0.000661 · 完全合理 · **dow 是真信号 · 不是误导**

→ **`enable_hour_only_features` 应该 deprecated** · weekend flag 必须保留 (test 全是工作日 · weekend=0 · 但 train 周末 weekend=1 · 删了 weekend 失去这个 binary discrimination)

### 修订 M41 LB +0.000254 解读

之前: "reweight 让 03-04 (test 同日) sample weight 升 · model 学得偏 03-04 daypart"

**真实**:
- temporal_weight_alpha=2.5 按 timestamp linear weight (0~1 normalized)
- timestamp 末尾 = 03-04 (CVR 0.122) + 03-03 末段
- timestamp 开头 = 02-28 / 03-01 (周末 · CVR 0.024/0.034)
- → reweight 的真实作用 = **降低周末 (低 CVR) sample 权重**
- → model 学得偏 工作日 mean (0.10) → test 工作日 → 略好

→ **M41 真信号机制 = weekday filter** (不是元宵节 daypart) · **+0.000254 LB 与机制吻合**

---

## 5. 战略修订 (5/16 CEST 10:15)

### 已 deprecated 的方向 (基于 5/15 元宵节认知 · 现证伪)

- ❌ **`--enable_hour_only_features`** (M40 路径): dow 是真信号 · 不应删
- ❌ **"daypart-aware item embedding"** 类 path B 设计: hour shift 是数据稀疏问题 · 不是 model 表达力问题
- ❌ **更深的 temporal_weight tune** (e.g. alpha=5 / alpha=10): 当前 2.5 已经 +0.000254 · 单调加 alpha 大概率 saturate

### 新方向候选 (基于 5/16 EDA 真实 mechanism)

#### 候选 A · weekend-only filter (M41 极简版)

把 train 的 02-28/03-01 (周末 · 7.2% rows · CVR 0.024~0.034) 直接**剔除**:
- 实施: 1 行 dataset filter `timestamp >= ts_03_02_start`
- 期望增量: 类 M41 但更直接 · 0.0005~+0.001 LB
- 风险: train data 减少 7%

#### 候选 B · hour-aware sample weighting (真正 daypart-aware)

按 train hour bucket reweight:
- 北京 09~14 (test 同时段) train sample weight × 5
- 北京 10/11 (train 几乎空) train sample weight × 20 (但 cnt 太少 · 实际效果有限)
- 其他 hour weight × 1 或降权

实施: 修 dataset.py temporal_weight 改为 hour-bucket-based · ~30 行

期望增量: +0.001~+0.002 LB (假设 hour-aware 比 timestamp-aware 强)

风险: 北京 10/11 hour cnt 太少 (3117/4915 rows) · 即使 weight × 20 · model 仍学不充分

#### 候选 C · 接受现状 · 不再 daypart engineering

- M41 +0.000254 弱信号已实测 · 复杂 daypart-aware 增量有限
- 真正的 daypart shift mechanism = train 时段不覆盖 test 时段 · 这是**数据本身限制** · 不是 model 能修复
- 改投其他方向 (path B UE 分离 · 或 final ckpt × s11/s2 上沿)

---

## 6. 5/16 下午行动建议 (修订)

### Tier 1 候选 · 基于 EDA 真实 mechanism

| 候选 | 实施成本 | 期望 LB 增量 | 备注 |
|---|---|---|---|
| **A · weekend filter** | 5 行 + 1 train job | +0.0005~+0.001 | 与 M41 部分重叠 · 但更直接 |
| B · hour-aware reweight | ~30 行 + 1 train job | +0.001~+0.002 | 真正 daypart-aware · 但 hour 11 数据不足 |
| C · 不做 daypart · M23 LB 决方向 | 0 | 0 | 等 M23 retry LB 出 · 决 cross 路径 |

### 5/16 下午 4 slot 推荐用法 (修订)

**前提**: M23 retry LB 出 (~CEST 10:25)

| Slot | Job | 期望 |
|---|---|---|
| 1 | **M45 = M22 + weekend filter** (剔除 02-28/03-01 train) | +0.0005~+0.001 LB · 干净 daypart-aware |
| 2 | TBD (基于 M23 LB 决) | 如 M23 LB > 0.848 → M22 + DCN cross stack;否则留空 |
| 3 | TBD | 同上 |
| 4 | 留空 | 5/16 晚 LB 后决 |

---

## 7. References

- Q6/Q7 EDA platform job: `EDA-quantile-tier0-1044rg-2026-05-16` (5/16 CEST 10:00 起 · 16 min · 2,099,956 rows · 213,518 pos)
- 原始 log: `~/Downloads/train_log.txt` (含完整 pretty summary)
- Decoded JSON: 不需 (pretty summary 全在 log)
- EXP-052 §15 元宵节认知 (本 EXP 证伪)
- EXP-048 §1 user_id 一人一行 (仍然 valid)
