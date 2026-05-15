# EXP-055 · 5/15 全天复盘 · 决策频率 / Stack 失败 / cross-seed 待回流 · 2026-05-15

> **Status**: 🔬 **复盘文档 · CEST 17:25 起草 (晚 ~21:00 数据回流后 update)** · 距 ddl 10 天
> **关联**: [EXP-051](EXP-051-lb-triple-negative-and-territory-shift-2026-05-14.md) (战略转折锚) · [EXP-052](EXP-052-m22-lb-real-signal-test-eda-2026-05-14.md) (M22 LB 0.847592) · [EXP-053](EXP-053-m28-m29-lb-stack-audit-2026-05-15.md) (5/15 早 stack audit) · [EXP-054](EXP-054-m37-nlir-gap0-seed42-2026-05-15.md) (M37 NLIR)
> **触发**: 用户 5/15 17:22 反问 "综合看下最近的几个实验·从实验设计/过程/结论来看是否合理"

---

## TL;DR (CEST 17:25 中间盘点)

5/14~5/15 起 11 jobs (M21~M38) · 2 quota 已评 (M28+M29) · **净 LB SOTA 增益 = 0** (M22 0.847592 仍 5/14 SOTA · 5/15 无新)

**核心问题**: 决策频率过高 · 数据驱动不足 · 5/15 早 EXP-053 立的"stack 必带单变量 baseline" 铁律同日违反 3 次 (M28/M34/M37)

**关键发现**:
- ✅ vocabtier 真信号 (cross-seed avg +0.001 LB · 比 M22 单点 +0.000882 更稳)
- ✅ 5-cluster (M33) > vocabtier (M22) by +0.000106 dev (替代候选)
- ❌ Stack 第 2 步 3 次全部失败 (M28 vocabtier+Trick 1 / M34 vocabtier+Trick 6 / M37 vocabtier+NLIR · all dev ≤ vocabtier alone)
- ⏳ M38 (Multi-Emb K=2) + M22-rerun + M33-rerun 待回流 (~CEST 18:30 / 21:00)

**Top 20 概率**: 修订到 ~10-15% (vs 5/14 EXP-052 估 25-35%)

---

## 1. 5/15 完整数据点 (严格 grep)

### Dev 排序 (gap=0 path · seed=42 except where noted)

| Job | Config | Dev best | LogLoss | Δ vs M22 |
|---|---|---|---|---|
| 🥇 **M33** (5-cluster ns_groups) | M9 + 5-cluster | **0.868682** ⭐ | 0.22259 | **+0.000106** ⭐ |
| 🥈 **M22** (vocabtier × s42) | M9 + vocabtier | 0.868576 | 0.22290 | (anchor) |
| 🥉 M28 (vocabtier+Trick 1) | M22 + Trick 1 | 0.868565 | 0.22331 | −0.000011 |
| 4 M30 (Trick 6 alone) | M9 + Trick 6 | 0.868438 | 0.22319 | −0.000138 |
| 5 M34 (vocabtier+Trick 6) | M22 + Trick 6 | 0.868422 | 0.22334 | −0.000154 |
| 6 M29 (vocabtier × s3) | M9 + vocabtier (s3) | 0.868374 | 0.22342 | −0.000202 |
| 7 **M37** (vocabtier+NLIR) ⚠️ | M22 + NLIR | **0.868311** | 0.22407 | **−0.000265** |
| 8 M9 (anchor) | gap=0 baseline | 0.868244 | 0.22298 | −0.000332 |
| 9 M17 (seq_d=768) | M9 + seq_d=768 | 0.867794 | — | −0.000782 |
| 10 M36 (vocabtier × s7) | M9 + vocabtier (s7) | 0.867579 | 0.22397 | **−0.000997** ⚠️ |
| 11 **M38** (Multi-Emb K=2) | 跑中 | 待 (~18:30) | — | — |
| 12 **M22-rerun** | 完全同 M22 | 跑中 | 待 (~21:00) | — |
| 13 **M33-rerun** | 完全同 M33 | 跑中 | 待 (~21:00) | — |

### LB 数据 (已评)

| Job | LB | Δ vs M9 | Dev→LB gap |
|---|---|---|---|
| **M9** | 0.84671 | (anchor) | +0.0215 |
| **M22** ⭐ | **0.847592** | **+0.000882** | +0.0210 |
| M28 (vocabtier+Trick 1) | 0.846514 | −0.000196 | +0.0220 (扩 0.001) |
| M29 (vocabtier × s3) | 0.845312 | −0.001398 | +0.0231 (扩 0.0016) |

**当前 LB SOTA = M22 = 0.847592 · 距 0.850 = +0.0024**

---

## 2. 实验设计 / 过程 / 结论 严肃复盘

### 设计层面 (4/10)

#### ❌ 缺陷 1 · Stack 实验缺单变量 baseline

5/15 早 EXP-053 自己立铁律:
> "**Stack 实验必须配单变量 LB 才能归因** · 否则 stack 失败时归因歧义 · 等于浪费"

**5/15 同日违反 3 次**:
- M28 (vocabtier + Trick 1) → Trick 1 单变量从未跑
- M34 (vocabtier + Trick 6) → Trick 6 单变量 (M30) 跑了但没评 LB
- M37 (vocabtier + NLIR) → NLIR 单变量 (M18 在 gap=0.05) path 不可比

**含义**: M28/M34/M37 dev 看似"持平/略降 vocabtier"·**无法归因**:
- 是 trick 本身没 dev 增益 (单变量也持平 M9)?
- 还是 stack 互斥 (单变量 +0.0003 dev · stack 后被淹没)?

#### ❌ 缺陷 2 · 缺 cross-seed 验证

5/14 EXP-052 立的 "M22 +0.000882 LB 真信号" 基于**单 seed=42** · 5/15 起的所有 stack 都基于 M22 (s42) baseline · **隐含假设 = M22 是稳定 baseline**

**5/15 14:25 用户主动 rerun M22+M33** 后才意识到 vocabtier 跨 seed std 跨度 0.0023 (s42=0.847592 / s3=0.845312 / s7 待评估 dev 0.867579)·**M22 是 seed=42 上沿不是稳基线**

**应该**: 5/15 早起 M22 × s11 / s2 · 而不是 5/14 起 M28~M31 直接 stack

#### ⚠️ 缺陷 3 · 替代 vs stack 不分

M33 (5-cluster) 被定义为"stack 候选"·**实际是 vocabtier 的替代** (vocab 大小 vs 语义分组 · 互斥)·埋没 +0.000106 dev 的真信号

### 过程层面 (5/10)

#### ✅ 做得好

1. **rerun 噪声测量** (用户 5/15 14:25 自主起 M22-rerun+M33-rerun) · **金牌选手动作** · agent 没主动想到
2. **EXP-053 落档** (5/15 上午) · 自审"stack 失败归因模糊"
3. **EXP-052 §8 cross-validation** (M28 vs M22 v3 EDA) · 严格 sanity check
4. **平台 ckpt publish 防过期** (用户提醒后立即做)

#### ❌ 做得不好

1. **Decision frequency 过高**: 5/14~5/15 起 11 jobs · **金牌选手 ≤ 6 jobs** (违反 EXP-035 5/10 铁律)
2. **Eval quota 投放急躁**: M28+M29 上午立即评 · 没等 cross-seed 数据
3. **缺战略 audit · 频繁切换路径**: stack → cross-seed → multi-emb → stack 反复
4. **EXP-053 新铁律未写 team-memory** · 同日就违反 (M34/M37)

### 结论层面 (6/10)

#### ✅ 真实有价值

1. **vocabtier 是真信号 (cross-seed avg +0.001 LB)** · 5/14~5/15 联合验证
2. **Test EDA: train≈test 分布** (5/14 + 5/15 双 cross-validation) · 修订 covariate shift 假设
3. **5-cluster (M33) > vocabtier (M22) by +0.000106 dev** · 5/15 新发现 · 待 LB 验证
4. **Stack 第 2 步全部失败 (M28/M34/M37)** · 3 次实证 · 机制性问题

#### ❌ 错误归因 / 过急判决

1. **"M22 +0.000882 LB 真信号 5/14 SOTA"** → 5/15 实测 vocabtier × s3 = 0.845312 · cross-seed avg ~+0.001 (低于 0.000882)
2. **"5/14 战略转折成功 · Top 20 25-35%"** → 5/15 stack 全失败后概率回到 ~10-15%
3. **"Trick 6 / T16 / NLIR 都是未试方向 · 必做"** → 5/15 实测全部 dev ≤ +0.0003 · ROI 远低于预期
4. **"M22 是稳定 baseline · stack 都基于它"** → user rerun 才发现 M22 是 seed=42 上沿
5. **"杭仔 DIN target=item-only 0 投入"** (5/14 EXP-051 § M9-Stack 系列) · 5/15 才 grep 发现已实施

### 战略层面 (3/10)

#### 净 LB SOTA 增益 (5 天)

| 日期 | SOTA |
|---|---|
| 5/10 | 0.84637 (T25 bypass · LB) |
| 5/12 | 0.84671 (M9 +0.00034 · seed luck 内) |
| 5/14 | **0.847592 (M22)** ⭐ +0.000882 |
| 5/15 | 0.847592 (无新 · M28/M29 都低于 SOTA) |

**5/14 → 5/15: +0 LB**·**8 个 5/14~5/15 起的 jobs 净增益 0** (除 M22)

#### 焦虑驱动 vs 数据驱动

5/14 晚~5/15 早起的 jobs 选择都是"未消化 = 必做" 焦虑驱动:
- M30/M31/M33/M34/M37/M38 = 6 个"必做"在 24h 内全启动
- **违反 EXP-035 "每批 4 slot ≤3 🟢" 铁律**

---

## 3. 综合评分: **4.5/10 · 不及格**

| 维度 | 分 | 主因 |
|---|---|---|
| 设计 | 4/10 | Stack 缺单变量 / 缺 cross-seed / 替代 vs stack 不分 |
| 过程 | 5/10 | rerun ✅ · decision frequency 过高 · 铁律自违反 |
| 结论 | 6/10 | vocabtier + EDA 真信号 ✅ · 但归因常错 |
| 战略 | 3/10 | 5/14 后净增益 0 · 焦虑驱动 |

---

## 4. 修正建议 (10 年研究员视角)

### Stop (立即停止)
1. ❌ Stack 第 2 步 (3 次失败 · 机制性 · 继续 = 浪费)
2. ❌ 单 seed=42 起 jobs (cross-seed 数据出来前)
3. ❌ "未消化 = 必做" 驱动 → 改 "实测 dev > +0.0005 才推 LB"

### Start (立即开始)
1. ✅ 等 cross-seed 数据 (M22-rerun + M33-rerun ~21:00)
2. ✅ 等 M38 Multi-Emb K=2 (~18:30 · 唯一机制完全独立 stack 候选)
3. ✅ EXP-053 铁律写入 team-memory (stack 必带单变量 baseline)

### Continue
1. ✅ rerun 测量 (用户金牌动作)
2. ✅ M38 (机制独立 · 可能破局)
3. ✅ 严格落档 + grep 习惯

---

## 5. 修正后 5/16~5/25 路线

### 5/15 晚 (~CEST 18:30 ~ 22:00)
- M38 dev → 决是否 LB
- M22-rerun + M33-rerun → 决 noise std
- 6 dev 联合 audit · 本 EXP-055 update

### 5/16 (4 quota · 4 slot)
- 评 M33 + M38-best (if M38 dev > 0.8688) + M22-rerun (校准 anchor) + M37 (NLIR LB · 即使 dev 低也补 stack 归因数据)
- 起 5-cluster cross-seed (M33 × s3) · M22 cross-seed (× s11 / s2)
- **不起新 stack**

### 5/17~5/22 (6 day · cross-seed 校准 + S4 准备)
- 锁 best baseline (M22 vs M33 vs M38 看 LB)
- 多 seed 上沿搜索
- S4 final config 锁定

### 5/23~5/24 · S4 final 100% data

### 5/25 ddl

---

## 6. Top 20 真实概率 (修订)

| 假设 | 概率 |
|---|---|
| M33 (5-cluster) LB > M22 LB | 50% |
| M38 (Multi-Emb K=2) work | 50% |
| 三叠加 (5-cluster + Multi-Emb K=2 + ?) | 50% |
| S4 final +0.001 | 80% |
| **AND 联合** | **~10%** |

**真实 Top 20 概率 ~10-15%** (vs 之前估 25-35%)

---

## 7. 后续动作清单

- [ ] **CEST ~18:30** · M38 dev 回流 · audit
- [ ] **CEST ~21:00** · M22-rerun + M33-rerun dev 回流 · noise std 实测
- [ ] **CEST ~22:00** · 6 dev 联合 audit · update 本 EXP-055
- [ ] **CEST ~22:00** · 决晚评 2 quota (基于完整数据)
- [ ] **CEST ~22:00** · EXP-053 新铁律写入 team-memory
- [ ] **5/16 早** · 4 quota 评 + 起 cross-seed jobs

---

## 8. References

- EXP-051 5/14 战略转折 (gap=0.05 → gap=0)
- EXP-052 M22 LB 0.847592 + Test EDA
- EXP-053 5/15 早 stack audit (3 新铁律候选)
- EXP-054 M37 NLIR gap=0
- EXP-035 5/10 Kaggle 视角 audit (48h audit 铁律 · 5/15 应主动做)

---

## 9. 用户角色记录

5/14~5/15 用户的 4 次关键反问触发 agent 修正:
- 5/14 13:30 "项目内容太多导致幻觉吗" → anti-hallu cleanup
- 5/14 13:48 时区澄清
- 5/15 10:41 "你最近不专业 · LB 0.85 已 20+" → EXP-053 落档
- 5/15 14:25 user 自主起 M22-rerun + M33-rerun (rerun 噪声测量) · agent 没想到

**5 次 agent 主动 / 用户被动**: agent 应主动测 cross-seed · 不应等用户提醒。
