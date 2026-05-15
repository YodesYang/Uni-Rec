# 5/16 早 quota refresh 决策矩阵 (CEST 06:00) · 基于今晚 4 dev 场景

> **创建**: 2026-05-15 CEST 17:33 · 用户 17:30 反问"为什么总做 cross-seed" + 17:32 同意 slot 留空后写
> **目的**: 5/16 早 quota refresh 时 · 基于今晚 4 dev 数据 · 严格按"信号驱动 ≠ 焦虑驱动"决评估顺序 + 训练设计

---

## 现有事实底 (严格 grep)

### LB 数据点 (已实测 · 截至 5/15 17:33)

| Job | Config | LB | Dev | Dev→LB gap |
|---|---|---|---|---|
| M9 | gap=0 baseline | 0.84671 | 0.868244 | +0.0215 |
| **M22** ⭐ SOTA | M9 + vocabtier (s42) | **0.847592** | 0.868576 | +0.0210 |
| M28 | M22 + Trick 1 | 0.846514 | 0.868565 | +0.0220 (扩 0.001) |
| M29 | M9 + vocabtier (s3) | 0.845312 | 0.868374 | +0.0231 (扩 0.0016) |

**Dev→LB gap 跨方法**: 0.0210 ~ 0.0231 · 平均 ~0.022 · std ~0.001

### Dev 数据点 (已实测 · 待 LB)

| Job | Dev | LogLoss | 状态 |
|---|---|---|---|
| **M33** (5-cluster × s42) ⭐ | **0.868682** | 0.22259 | 已 publish · 待评 |
| M30 (Trick 6 alone) | 0.868438 | 0.22319 | 已 publish · 待评 |
| M34 (vocabtier+Trick 6) | 0.868422 | 0.22334 | 已 publish · 待评 |
| **M37 (vocabtier+NLIR)** | 0.868311 | 0.22407 | 已 publish · 待评 |
| M27-v2 (T31+user_15) | 0.868166 | — | 已 publish · 待评 |
| M23 (T34 DCN gap=0) | 0.868311 | — | 已 publish · 待评 |
| M17 (seq_d=768) | 0.867794 | — | 已 publish · 待评 |
| M36 (vocabtier × s7) | 0.867579 | 0.22397 | 已 publish · 待评 |
| M31 (T16) | 0.867646 | — | 已 publish · 待评 |
| M21 (T34 gap=0.05) | 0.843551 | — | 已 publish · 待评 |

### 待今晚回流 (~CEST 18:30 / 21:00)

| Job | Dev 待 | 战略价值 |
|---|---|---|
| **M38** (Multi-Emb K=2) | ~CEST 18:30 | 唯一机制完全独立 stack 候选 |
| **M22-rerun** | ~CEST 21:00 | 同 seed=42 同 config rerun · 测 noise std |
| **M33-rerun** | ~CEST 21:00 | 同 s42 · 测 5-cluster noise std |

---

## 关键判决依据 · "+0.000106 是真信号还是 noise"

**核心问题**: M33 dev 0.868682 vs M22 dev 0.868576 = +0.000106 = 真信号 (5-cluster > vocabtier) 还是 rerun noise?

**判决依据 = M22-rerun + M33-rerun 跨度**:

| M22-rerun vs M22 dev 跨度 | 推论 |
|---|---|
| ≤ 0.0001 | 0.000106 是真信号 · M33 > M22 · **5-cluster 是新 baseline** |
| 0.0001 ~ 0.0003 | 边界 · 需要 cross-seed 进一步 |
| ≥ 0.0003 | 噪声 · M33 vs M22 难分胜负 · 看 LB |

---

## 5/16 早 决策矩阵 (4 quota · 基于今晚 4 dev 数据场景)

### 场景定义维度

**维度 1**: M38 dev (Multi-Emb K=2)
- A1: M38 dev > 0.8688 (强真信号)
- A2: M38 dev ∈ [0.8682, 0.8688] (持平 vocabtier)
- A3: M38 dev < 0.8682 (失败)

**维度 2**: M22-rerun + M33-rerun 跨度
- B1: 都 ≤ 0.0001 跨度 (noise 极小 · M33 真信号)
- B2: 跨度 0.0001~0.0003 (边界 · cross-seed 进一步)
- B3: 跨度 ≥ 0.0003 (noise 大 · M33 vs M22 难分)

---

### 9 大场景 (3 × 3) + 推荐

#### 场景 1 · A1 + B1 (M38 强真信号 + noise 极小) · **理想态 · 5%**

**dev 状态**: M38 > M33 > M22 · 全部真信号
**5/16 4 quota**:
1. M38-best (确认 LB)
2. M33-best (5-cluster 真 SOTA 候选)
3. M22-rerun (校准 noise · 即使是 rerun 也 LB)
4. **新 stack** = M22 baseline + Multi-Emb K=2 (M38 配置堆到 vocabtier 上)

**5/16 4 slot 训练**:
1. M38 + 5-cluster (双 stack · 现在 stack 第二步**有理由**)
2. M22 + Multi-Emb K=2 cross-seed (× s3)
3. M33 cross-seed (× s3)
4. 留 backup

#### 场景 2 · A1 + B2/B3 · 罕见但需谨慎

**5/16 4 quota**:
1. M38-best
2. M33-best
3. M22-rerun
4. M30 (Trick 6 alone · 给 stack 归因)

**5/16 4 slot**: 等 LB 数据后决 · 不盲推 stack

#### 场景 3 · A2 + B1 · **最常见情境 · 30%**

**dev 状态**: M38 持平 vocabtier · M33 真信号 +0.0001
**5/16 4 quota**:
1. **M33-best** ⭐ (5-cluster 真 SOTA 候选 · 必评)
2. M22-rerun (noise 校准锚)
3. M38-best (即使 dev 持平 · 1 LB 数据点判 Multi-Emb K=2 真假信号)
4. M30 (Trick 6 alone · 给 M34 stack 归因)

**5/16 4 slot 训练**:
1. **M33 × seed=3** (5-cluster cross-seed 第 1 数据点 · 最高 ROI)
2. **M22 × seed=11** (vocabtier 4th seed · 锁 noise std)
3. **M33 + Multi-Emb K=2 stack** (5-cluster 真信号 + 唯一独立机制)
4. 留 backup

#### 场景 4 · A2 + B2 · **次常见 · 25%**

**dev 状态**: M38 持平 · noise 边界
**5/16 4 quota**:
1. M33 (即使可能 noise · 也评 1 个数据)
2. M22-rerun + M33-rerun (noise 校准 · 双数据点)
3. **保留 1 quota 兜底**

**5/16 4 slot**:
1. M22 × seed=11 (vocabtier 4th 数据)
2. M33 × seed=3 (5-cluster 2nd seed)
3. 留 2 backup

#### 场景 5 · A2 + B3 · **危险 · 25%** (vocabtier 跨 seed std ≥ 0.0003)

**含义**: M22 LB 0.847592 大概率 seed luck · M33 +0.000106 完全噪声
**5/16 4 quota**:
1. M22-rerun (确认 anchor)
2. M33 (虽然可能 noise · 但 1 个数据点决断)
3. M30 (Trick 6 单变量 · 即使是噪声也提供归因数据)
4. 保留 1 quota

**5/16 4 slot**: **不起新 stack** · 集中 cross-seed 校准·或评估更多 ckpt

#### 场景 6 · A3 + B1/B2 · M38 失败 + noise 小

**含义**: Multi-Emb K=2 不 work · 5/15 唯一独立机制候选失败
**5/16 4 quota**:
1. M33-best (5-cluster 真信号)
2. M22-rerun (noise)
3. M30 / M37 (单变量归因)
4. 保留

**5/16 4 slot**:
1. M33 × seed=3 (cross-seed)
2. M22 × seed=11 (vocabtier 4th)
3. **新方向探索**: T16 alpha=0.05 (M31 alpha=0.1 失败·更小 alpha 试) / DECEM Trick 1 时间桶 single-var (T30)
4. 留 backup

#### 场景 7 · A3 + B3 · **最坏情境 · 5%** · M38 失败 + noise 大

**含义**: 一切都不稳·M22 LB 0.847592 也是 seed luck
**5/16 4 quota**:
1. M22-rerun (确认 anchor)
2. M33-rerun (5-cluster 也 cross-seed)
3-4. 保守评 (M30 / M17)

**5/16 4 slot**: **彻底重审 · 考虑大改动 (杭仔路线)** 或 **退守 S4 final** · **概率 5%** 但要预案

---

## 总览 · 9 场景概率分布 (10 年研究员估算)

| 场景 | 概率 | 5/16 主路径 |
|---|---|---|
| 1 (A1 + B1) | 5% | 双 stack + cross-seed |
| 2 (A1 + B2/B3) | 5% | 谨慎 stack + cross-seed |
| **3 (A2 + B1)** | **30%** ⭐ | **M33 cross-seed + 5-cluster + Multi-Emb K=2 stack** |
| **4 (A2 + B2)** | **25%** ⭐ | cross-seed 校准 + 留 backup |
| **5 (A2 + B3)** | **25%** | noise 主导 · 不起 stack · 集中校准 |
| 6 (A3 + B1/B2) | 5% | 5-cluster + 新方向 |
| 7 (A3 + B3) | 5% | 大改动 / S4 退守 |

**最常见情境 (3+4+5 = 80%)**: M38 持平·**5/16 主路径 = 评 M33 LB + cross-seed 校准** (不起新 stack)

---

## 5/16 早 操作模板

### Step 1 · CEST 06:00 quota refresh

**立即提交 4 quota** (基于今晚数据决具体 ckpt):
- 优先 1: M33 (5-cluster · 必评 · LB 决 5/16 主路径)
- 优先 2: M22-rerun (anchor 确认)
- 优先 3: M38 (基于 dev 决 · A1/A2/A3 都评)
- 优先 4: M30 / M37 / M17 (基于场景决)

### Step 2 · CEST 09:00~10:00 LB 数据

- 4 LB 联合 audit · update EXP-055
- 决 5/16 4 slot 训练设计

### Step 3 · CEST 10:00 起 4 training jobs

- 基于 5/16 LB 数据·**信号驱动**
- **不起"未消化必做"焦虑驱动 jobs**
- **EXP-053/EXP-055 铁律严格遵守**:
  - Stack 必带单变量 baseline
  - 每方法 ≤ 2 seed
  - 替代 vs stack 严格区分

---

## 落 EXP-055 §"5/16 路线图" 更新

5/15 22:00 数据全回流后 · update EXP-055 §5 (5/16 路线) 用真实场景替代占位符。

---

## References

- EXP-055 §修正建议 (5/16 起 cross-seed + Multi-Emb K=2)
- EXP-035 5/10 Kaggle 视角 audit (每假设 ≤ 2 seed 铁律)
- EXP-053 5/15 Stack 必带单变量 baseline
- team-memory line 9 · evaluation quota 规则
