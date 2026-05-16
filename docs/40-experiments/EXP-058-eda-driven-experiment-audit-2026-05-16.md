# EXP-058 · EDA 反推 8 大错误 Audit + 修订路径 · 2026-05-16

> **Status**: 🔵 Documented (5/16 CEST 11:55 落档)
> **Date**: 2026-05-16 CEST 11:55
> **Trigger**: 用户要求 "重新审视过去几天所有实验决策合理性" · M23 LB 0.836631 强负引发深度 audit
> **Related**: EXP-055 (5/15 retrospective) · EXP-056 (Tier 0 EDA · 元宵节证伪) · EXP-049 (correlation EDA) · EXP-057 (M45-M48 daypart batch)

---

## TL;DR · 16 天累计 11 次 LB 数据 · EDA 反推 4 个真错 + 4 个可能误判

**4 个真错 (already-evaluated · 已损失 quota)**:
1. **M40** (hour_only · 删 dow+weekend) → EDA-056 证 weekday 是 5× CVR gap 真信号
2. **M23** (DCN cross) → 100% item OOV 下 train 内 cross 共现 反向放大 noise
3. **M28** (全局 time bucket) → 4 domain time_diff 11× 差 · 应 per-domain bucket
4. **M33** (5-cluster 替换 vocabtier) → 应叠加而非替换 · 失去 high-vocab item id isolation

**4 个可能误判 (publish-ready · LB 待评)**:
- M37 (NLIR) · M30 (Trick 6) · M38 v2 (Multi-Emb K=2) · M39 v2 (Global Token)
- 共性: dev 弱负 → 之前判 "大概率 LB 弱负" · EDA 后重审多数为 architecture-level 不依赖 train 期 stat bias · LB 可能持平甚至 dev↓ LB↑ (类 M41)

**3 个真未消化方向** (因发现的 misimplementation):
- per-domain time bucket (修复 M28)
- vocabtier + user-side 5-cluster 双 stack (修复 M33)
- NLIR + UE split 互不依赖 stack (M37 闭环)

---

## 1. 16 天累计 LB 数据 (严格 grep 重写)

| Job | Method | LB | Δ vs M22 (0.847592) | 当时归因 | 5/16 EDA 后重审 |
|---|---|---|---|---|---|
| **M22** ⭐ | vocabtier ns_groups | 0.847592 | (anchor) | T29 vocab tier 真信号 | ✅ 仍真信号 (high-vocab item id 单独 chunk) |
| **M22-rerun** | same seed rerun | 0.847324 | −0.000268 | rerun band ±0.0003 | ✅ stable signal |
| **M9** | T25 DIN bypass + timefeat + hash500k | 0.846710 | −0.000882 | DIN bypass + timefeat | 🔍 timefeat 含 dow/weekend · 与 EDA 真信号一致 |
| **M28** | M22 + Trick 1 (time bucket) | 0.846514 | −0.001078 | stack 第 2 步失败 | 🔍 时间桶冗余 vocabtier · 应 per-domain |
| **M29** | M22 × seed=3 | 0.845312 | −0.002280 | seed 噪声 | ✅ cross-seed band ±0.005 内 |
| **M33** | M22 + 5-cluster ns_groups (替换) | 0.844710 | −0.002882 | dev/LB 反向 27× | 🔍 5-cluster 替换 vocabtier 是实施错 |
| **M40** | M22 删 dow + weekend (hour_only) | 0.846931 | −0.000661 | dow 半证伪 | ⚠️ EDA 证 dow/weekend 是真信号 · M40 删了 |
| **M41** | M22 + temporal_weight 2.5 | 0.847846 | +0.000254 | dev↓ LB↑ 反例 | ✅ 真实机制 = 降低周末低 CVR weight |
| **M36** | M22 × seed=7 | 0.836413 | −0.011179 | seed=7 极端低沿 | ✅ EXP-028 5 seed 跨度 0.010 outlier |
| **M23** | M22 + DCN cross (gap=0) | 0.836631 | −0.010961 | DCN cross 路径死 | ⚠️ EDA 给的 cross 候选 OOV 反向放大 40× |

**结构性观察**:
- 11/11 LB jobs 中 **仅 M41 一个 stack 工作** (+0.000254 弱信号)
- 6/6 M22 + X stack 失败 (除 M41) · stack 第 2 步是 hard
- cross-seed band ±0.005 (M29 −0.0023 / M36 −0.0112) 主导 dev=42 单点对比的 confidence

---

## 2. 4 个真错 deep audit

### Error A · M40 删 dow+weekend = 删了真信号

**问题**: 5/15 23:30 元宵节认知错误推论 dow 是误导信号 → M40 hour_only → LB 跌 −0.000661

**EDA-056 反证** (weekday × CVR):

| Day | Rows | Pct | CVR |
|---|---|---|---|
| 02-28 (周六) | 86,691 | 4.1% | **0.024** |
| 03-01 (周日) | 65,039 | 3.1% | **0.034** |
| 03-02 (周一) | 119,861 | 5.7% | 0.119 |
| 03-03 (周二) | 1,312,781 | 62.5% | 0.101 |
| 03-04 (周三) | 515,584 | 24.6% | **0.122** ⭐ test 同日 |

**dow / weekend flag 是 5× CVR gap 的核心 signal** · 删了等于让 model 把周末 noise 当 normal · 在 test (周三 工作日 CVR ~0.12) 上 base rate 偏低。

**教训**: 凭印象 (元宵节假设) 删特征 · 没先 EDA 验证 → 损失 1 个 evaluation quota。已立铁律 (team-memory): 任何"删特征"决策必须基于 EDA 数据驱动 · 不凭文化/季节印象。

---

### Error B · M23 DCN cross 失败 deep audit

**EDA-049 给定 cross 候选** (top user × item Cramér V):
- (user_1, item_10) CV=0.278 / MI=0.138
- (user_1, item_5) CV=0.273 / MI=0.129
- (user_49, item_6) CV=0.272 / MI=0.080

**M23 LB 实测**: −0.011 (强负 · 反向放大 40×)

**为什么 EDA 给的 "强 user × 强 item" cross 失败**:

(a) user_1 + user_49 是 user-side **低基数** (V=6 / V=3) · 大概率是设备/渠道/性别 · test 上**仍可能见到相同值**
(b) item_5/6/7/10/12 是 item-side 高 MI · 但**本质 vocab 大** (3xx~9xx range) · test 上**新 item id** OOV 高
(c) DCN cross 学的是 user_1 × item_5 共现 pattern · **(user_1=2, item_5=新值) 在 train 上没见过** · model 学到的 cross pattern 完全无用
(d) **不只无用 · 还反向放大 noise**: cross feature 给 model 一个"我懂这个 pair"的虚假信心 · 在 OOV 场景下系统性误导

**根因**: EXP-049 §3 "DCN cross 是 Tier 1 方向"是错判 · **EDA 数据本身没错** · 错在外推到 test 的逻辑 · **只看了 train 内共现 · 没看 test transferability**

**铁律 (新立)**: 任何 cross / interaction feature 候选 · 必须验证 **(test 端候选 cross 值的 OOV 率)**。如果 item-side OOV > 50%,cross 路径全死。

---

### Error C · M28 全局 time bucket 不适配 4 domain time scale

**M28 LB**: −0.001078

**EDA-056 Q5 数据**: time_diff p99
- seq_d (短期/快): **56 天**
- seq_c (长期/慢): **629 天**
- 量级跨 **11×**

**Trick 1 设计**: 全局 64-bucket log-scale boundaries (DECEM 论文方案)

**问题**:
- 在 seq_d (短期高密) 上 bucket 太粗 · 学不细
- 在 seq_c (长期低频) 上 bucket 浪费 · 大量 bucket 落在稀疏区
- **全局 bucket 是 11× scale gap 上的中庸方案 · 都不好**

**未消化方向**: per-domain time bucket
- seq_d: 64-bucket 覆盖 [0, 56 天]
- seq_c: 64-bucket 覆盖 [0, 629 天]
- seq_a/b: 各自 fit
- 实施成本: dataset.py time_bucket_boundaries 改 List[List[float]] · ~50 LOC

→ **M28 不是 Trick 1 错 · 是实施细节错** · per-domain bucket 未消化

---

### Error D · M33 5-cluster 替换 vocabtier · 失去 high-vocab item id isolation

**M33 LB**: −0.002882

**问题**: 5-cluster 是 EXP-049 EDA-correlation cluster (基于 user × user CV) · 但**替换 vocabtier 是错的**
- vocabtier (EXP-038) 按 vocab size 分 · item-side 维度
- 5-cluster (EXP-049) 按 user × user correlation 分 · user-side 维度
- 两者**正交**

**正确实施**: vocabtier (item-side 保留) + 5-cluster (user-side 重组) · 而不是 vocabtier 整体替换

**M33 实际做的**: ns_groups_5cluster.json 完全替换 ns_groups_vocab_tier.json → high-vocab item id (M22 真信号源) **单独 chunk 没了** → 退回到 baseline ns_groups 类似状态

→ **M33 是实施 mistake** · 不是 5-cluster 路径错 · vocabtier × 5-cluster 双 stack **未真正测过**

---

## 3. 4 个可能误判 (publish-ready · LB 待评)

### Error E · M37 (NLIR · vocabtier + Linear gating)

**当前 status**: dev 0.868311 (−0.000265) · LB 待评

**5/15 早 agent 推断**: NLIR 路径关闭 · dev 弱负 = 大概率 LB 也弱负

**EDA 后重审**:
- NLIR (Linear gating) 是 architecture-level · **不依赖 train 期统计量**
- 与 weekday/hour daypart shift **不直接交互**
- 类似 M41 的反例机制可能 (dev 弱负 → LB 持平/反向涨)

→ **M37 不应过早判 deprecated** · 候选 dev↓ LB↑ 反例之一

---

### Error F · M38 v2 (Multi-Emb K=2)

**当前 status**: dev 0.868403 (−0.000173) · LB 待评

**EDA 后重审**:
- Multi-Emb K=2 增加 embedding 容量 · **不引入 train 期 statistical bias**
- 与 covariate shift 正交
- LB 可能在 noise band 内 (持平)

→ **M38 v2 仍是闭环候选** · 但 ROI 低 (大概率持平不破 SOTA)

---

### Error G · M39 v2 (Global Token)

**当前 status**: dev 0.867963 (−0.000613) · LB 待评

**EDA 后重审**:
- Global Token 在 user_dense slot 加 mean-pool 全局信号
- 与 ADR-014 UE split 在**同一 user_dense slot 路径冲突**
- mean-pool 把 fid_61 (256-dim UE) + count features 一起 mean → **信号稀释**

→ **M39 v2 大概率不破 SOTA · 可不评**

---

### Error H · M30 (Trick 6 · Q_init=MLP(item_emb))

**当前 status**: dev 0.868438 (−0.000138) · LB 待评

**EDA 后重审**:
- Q_init 改 query 初始化 · architecture-level · 不依赖 train stat
- 类 M37 性质 · LB 可能接近 noise band (持平)

→ **M30 仍可考虑闭环** · 给 M34 stack 失败归因 · 但优先级低于 M23/M37

---

## 4. 真正未消化的方向 (因 misimplementation 发现)

### 方向 1 · per-domain time bucket (修复 M28)

- 实施: dataset.py 改 time_bucket_boundaries 为 4-domain 各自 fit
- 预期: dev +0.0002~+0.0005 · LB +0.0003~+0.001
- 成本: ~50 LOC + smoke + 1 platform job
- ROI: 中 (M22 + per-domain bucket 是干净 stack)

### 方向 2 · vocabtier + user-side 5-cluster 双 stack (修复 M33)

- 实施: 用 ns_groups_vocab_tier.json 同时叠加 5-cluster user-side 分组 · 需新生成 ns_groups_vocabtier_x_5cluster.json
- 预期: 不确定 · 5-cluster 可能与 vocabtier 互不冲突也可能干扰
- 成本: ~30 min ns_groups 文件生成 + 1 platform job
- ROI: 中低 (5-cluster 单独信号未验证)

### 方向 3 · NLIR + UE split 互不依赖 stack

- 假设 M37 LB 持平/微正 (验证后)
- M37 (NLIR) + ADR-014 (UE split) 走不同路径 · 可 stack
- 但 ADR-014 path B 当前 suspended · 需要 reactivate

---

## 5. 5/16 晚 evaluation quota 推荐 (基于本 audit)

**当前剩余**: 2 quota (晚 21:00 用)

**修订推荐顺序** (基于本 audit + EXP-057 daypart batch):

### Slot 1 · M45 LB (5/16 下午 daypart batch ckpt 回流后判)

- 优先级: ⭐⭐⭐
- 决断价值: weekend filter alone work 不 work · daypart engineering 路径核心信号

### Slot 2 候选 (取决于 5/16 下午 M45-M48 dev)

(a) **M37 LB** (NLIR) · 如本 audit 反推成立 · 可能反例 dev↓ LB 持平
(b) **M38 v2 LB** (Multi-Emb K=2) · RankUp 闭环
(c) M46/M47/M48 dev 高者 · 如 daypart batch 出强真信号

**默认决断**: M45 + 取决于下午数据决 slot 2

---

## 6. 真实状态评估 (5/16 CEST 11:55)

**M22 vocabtier (LB 0.847592) 仍是 SOTA**:
- 11/11 LB 中 M22 / M41 是 top 2 · 都基于 vocabtier
- vocabtier 路径**接近局部最优** · 单纯改 ns_groups 难破

**LB 上沿估算** (M22 baseline cross-seed):
- s42: 0.847592 (已知)
- 上沿 s11/s2 估计: 0.848~0.849 (cross-seed band ±0.0015)
- 但 5/22 前 seed=42 lock 铁律 · 不做

**距 0.85 (Top 20) 还差**: +0.0024
**距 DECEM Rank 1 (0.852)**: +0.0044

**真实 Top 20 概率** (5/16 11:55):
- daypart batch 4 jobs 全 work: 5~10%
- 部分 work: 3~5%
- 全部失败: <2%

---

## 7. 经验总结 (新立 / 强化的纪律)

### 新立铁律 1 · cross/interaction feature 必须验证 OOV transferability

任何 cross / interaction feature 候选 · 必须验证 **(test 端候选 cross 值的 OOV 率)**:
- 如 item-side OOV > 50% · cross 路径**系统性死**
- 不要只看 train 内 Cramér V / MI · 那只是"train 内有信号" · 不代表 test transferable

### 新立铁律 2 · 删特征决策必须 EDA 数据驱动

任何"删特征"决策必须基于:
- EDA 量化数据 (CVR gap / 分布变化 / 信号强度)
- **不凭文化/季节/直觉印象** (M40 元宵节假设教训)
- 即使打算删 · 先用 dev split 验证不删的版本 vs 删的版本 dev 差距

### 强化铁律 3 · stack 第 2 步极难 work · 默认期望 = 持平

5/14~5/16 11/11 LB 数据中 6/6 M22 + X stack 失败 · 仅 M41 work (+0.000254 弱信号)。
- stack 默认期望 = 持平 · 不预设 +0.001 增量
- 选 stack 候选时 · 必须证明 X 与 baseline 路径**机制正交** (M41 = temporal_weight 与 vocabtier 正交 work · 其他 stack X 大概率与 vocabtier 路径 overlap)

### 强化铁律 4 · 实施错误归因不要外推路径死

M33 5-cluster 替换 vocabtier 是**实施 mistake** · 不代表 5-cluster 路径错。
- 任何 LB 强负 first 检查 implementation diff · 不要默认路径死
- 路径判死必须基于**正确实施 + 多次实验**

---

## 8. References

- EXP-049 §3 (DCN cross Tier 1 错判 · 已被 M23 LB 实证)
- EXP-052 (M22 SOTA 0.847592)
- EXP-053 (M28/M29 LB stack audit)
- EXP-055 §16-17 (path A 死透 audit · M23 LB 重大判读)
- EXP-056 (Tier 0 EDA · 元宵节证伪 · weekday 真信号)
- EXP-057 (M45-M48 daypart batch · 当前 in-flight)
- team-memory line 175 (training slot vs evaluation quota 区分)
- ADR-014 (UE split path B · 5/16 reactivated)
- src/eda_quantile.py (Q6 CVR by day · Q7 CVR by hour)
