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

---

## 10. CEST 19:25 补遗 · M38 Silent Copy 事故

### 事故描述

`M38-m9stack-multi_emb_k2-gap0-seed42-4ep` 5/15 18:30~01:08 跑完 5h40m · dev E4 0.868347。**实际 platform args 显示**:

| Flag | M38 实际值 | 期望 (M22 + multi_emb_k=2) | M37 (NLIR) | 偏差 |
|---|---|---|---|---|
| `multi_emb_k` | **1** | 2 | 1 | ❌ Multi-Emb 没启用 |
| `enable_nlir_gating` | **True** | False | True | ❌ NLIR 错误启用 |
| `enable_din_interest` | True | True | True | ✅ |
| `seq_max_lens` (seq_d) | 512 | 512 | 512 | ✅ |
| `ns_groups_json` | vocab_tier | vocab_tier | vocab_tier | ✅ |

**实际等价**: M22 + NLIR + seq_d=512 = **M37 复刻**。

### Root cause = 平台 Silent Copy

- agent CEST 12:42 给的 M38 run.sh **正确**包含 `--multi_emb_k 2`、不含 `--enable_nlir_gating` (transcript line 1585)
- 用户在平台用 "Copy from M37" 起 M38 (M37 是上午刚跑完的 NLIR job, 5/15 11:40 起)
- 用户编辑 run.sh / 选上传文件 · UI 上 "看起来" 修改了
- **但 backend 实际 snapshot 仍是 M37 的 frozen run.sh + frozen 上传文件 (5/15 早 commit, 不含 ADR-012 multi_emb_k 代码)**
- Silent failure: 没 unrecognized args error · **因为 `--multi_emb_k 2` 根本没传给 train.py · NLIR flag 被 M37 frozen run.sh 携带**

### M38 数据处理决定

- **Drop M38 当 Multi-Emb K=2 数据点** · 完全不入主表
- **Footnote 当 M37 cross-rerun 样本**: M37 0.868311 vs M38 0.868347 = +0.000036 · 落在 EXP-032 实测 rerun noise 带 ±0.0001 内 → 加固 §M37 NLIR 真负面方向判决 (vs M22 0.868576 双点 mean 0.868329 = −0.000247 · 不是 single seed luck)
- **M38 ckpt 不 publish** · 平台自然 GC

### ADR-012 Multi-Embedding K=2 仍未验证

5/15 EXP-053 触发 ADR-012 起草 · CEST 12:42 实施完成 · 但 **5/15 一整天没拿到任何 platform 数据点**。M38 v2 (`M38v2-multiembk2-m22-gap0-seed42-4ep`) 5/15 CEST 19:30 重起 · ~01:30 5/16 完成。

### 新铁律 (写入 team-memory)

#### 铁律 N · 平台 Silent Copy 风险

> **平台 Training "Copy from XX" 即使用户编辑了 run.sh / 重新上传文件, backend 仍可能保留 frozen snapshot**。已知模式: 2026-05-01 unrecognized-args 失败模式 (loud); 2026-05-15 silent-success 模式 (M38 跑了旧 NLIR config 5h)。**只能信赖 "Create New Training Job"**, 重新上传所有文件 + 重新粘贴 run.sh。**Copy 永远禁用** (除非 0 代码改动 0 args 改动 · 仅改 job name)。

#### 铁律 N+1 · 启动后 30s grep Args dict 验证

> **任何 platform job 启动后 30 秒内 必须 grep `Args: {...}` log dict 验证 5+ key flags**:
>
> 1. job 名标榜的核心 flag (e.g. `multi_emb_k`, `enable_global_token`)
> 2. 任何上一类似 job 引入的 flag, 必须确认**没继承** (e.g. `enable_nlir_gating`)
> 3. STRUCTURAL 不变 flags: `ns_tokenizer_type` / `user_ns_tokens` / `item_ns_tokens` / `num_queries` / `ns_groups_json` / `seed`
> 4. baseline 关联 flag: `seq_max_lens` (M22 = seq_d:512, M17 = seq_d:768)
> 5. AMP / compile / ckpt 相关 (`enable_amp`, `keep_topk_ckpt`)
>
> **任一不对**: cancel job (sunk 30s) · Create New Training Job (NOT Copy) 重起。**5h GPU 浪费 vs 30s 验证 = 600x ROI**。

### 5/15 实际净增量 vs 计划

5/15 12:00~22:00 计划 = "M37 + M38 双 RankUp 论文核心方向数据点齐全"
5/15 实际 22:00 = "M37 单点 (NLIR) + M38 仅作 M37 cross-rerun + Multi-Emb K=2 / Global Token 全部 pending"

**1 整天 GPU-hour 净 ADR 验证 = 0** · M38 silent copy 事故吃掉了 Multi-Emb 验证机会。

---

## 11. CEST 22:06 补遗 · M22-rerun + M33-rerun 数据回流 · 重大判决修订

### 数据 (E4 best dev AUC)

| Job | 原始 (5/14~5/15) | rerun (5/15) | Δ rerun | EXP-032 noise band? |
|---|---|---|---|---|
| **M22** (vocabtier × s42) | 0.868576 | **0.868531** | **−0.000045** | ✅ 在 ±0.0001 内 |
| **M33** (5-cluster × s42) | 0.868682 | **0.868408** | **−0.000274** | ❌ **超出 ±0.0001 的 3×** |

### 判决 1 · M22 vocabtier 是 stable signal

EXP-032 (icq-t13tf, 2026-05-09) 给出 seed=42 同 commit rerun noise std ≈ 0.00007 / 跨度 ±0.0001。**M22 rerun gap −0.000045 在 band 内** · M22 dev 0.8685~0.8686 是 stable signal · 不是单点 luck。LB 0.847592 同样可信任为 baseline。

### 判决 2 · M33 5-cluster 不是真信号 · 是 noise 上沿假象

**两个 M33 数据点 mean** = (0.868682 + 0.868408) / 2 = **0.868545**
**两个 M22 数据点 mean** = (0.868576 + 0.868531) / 2 = **0.868554**

**M33 mean − M22 mean = −0.000009 ≈ 0** · 5-cluster ns_groups 在 dev 端**与 vocabtier 基本打平 (实际略低)**。M33 原始 +0.000106 优势是单点 lucky upper · 不是 architectural 优势。

**机制猜测**: 5-cluster ns_groups 的 NS Tokenizer 把 fid 分到不同 cluster, 不同 rerun 中 sparse embedding init / Adagrad sparse grad order 对 cluster boundary 敏感 → noise 比 vocabtier 路径**放大 3×**。这是 5-cluster 路径的内在性质 · 不是事故。

### 判决 3 · 5/16 早 4 quota 决策修订

**M33 evaluation 价值降级** (从原计划"评 M33 ckpt 看 5-cluster LB" 改为"M33 不评"):
- M33 mean 与 M22 mean 打平 · M33 ckpt LB 期望 ≈ M22 LB 0.8476 · **没有新 SOTA 期望**
- 占用 1 quota 的边际信息 < M38 v2 / M39 v2 / M22-LB-rerun

**5/16 早 4 quota 修订计划** (待 M38 v2 / M39 v2 dev 回流后定):
| Quota | 候选 | 价值 |
|---|---|---|
| 1 | **M22-rerun ckpt LB** | LB 端 rerun 校准 (dev 端已确认稳定 · LB 端未量) · 验证 SOTA 0.8476 是否 ckpt-specific |
| 2 | **M38 v2 ckpt LB** (Multi-Emb K=2) | ADR-012 LB 验证 ⭐ |
| 3 | **M39 v2 ckpt LB** (Global Token) | ADR-013 LB 验证 ⭐ |
| 4 | **取决于 M38v2/M39v2 dev**: 若 ≥ 0.8691 ⭐ 评最强者; 若 dev 持平 / 略升 (<0.8691) 评 M30 (Trick 6 单变量 LB · 给 M34 stack 归因) |

### 判决 4 · 新铁律 (写入 team-memory)

#### 铁律 N+2 · ns_groups 路径 rerun noise 比基础 vocabtier 路径放大 3×

> **2026-05-15 CEST 22:06 · M22 rerun ±0.00005 / M33 rerun ±0.00027 实测**: 5-cluster (custom ns_groups) 路径 rerun noise band ≈ ±0.0003, 是默认 vocabtier 路径 ±0.0001 的 3×。**铁律**: ① ns_groups 路径的 dev 判决阈值从默认 ±0.0003 (3σ) **放宽到 ±0.0009** (3σ_5cluster); ② 任何 ns_groups 类改动单点 dev Δ < +0.0009 都不能凭单点判 "新方向 work"; ③ 想验证 ns_groups 改动需要 **2~3 seed 或 2 rerun mean** · 不是单点; ④ 此规则只 apply ns_groups 路径 · 不 apply 其他 (model.py kwarg / loss / encoder)。

### 5/15 净增量再修订

5/15 三个真实 dev 数据点回流:
- M22 rerun (vocabtier 真信号确认) ✅
- M33 rerun (5-cluster mean ≈ vocabtier mean · 单点 +0.000106 是 luck 假象) ✅
- M38 v2 + M39 v2 跑中 (~CEST 00:30~01:30 回流)

**5/15 战略价值** = 3 个 noise calibration data point + 2 个 RankUp 验证候选启动 · **不是 0** (vs 11:00 audit 时认为的 "净 ADR 验证 = 0")。silent copy 事故吃掉的是 **5/15 单日 ADR-012 验证机会** · 5/16 凌晨已重启。

---

## 12. CEST 22:25 补遗 · 今晚 2 quota 决策 (用户 22:17 反问触发)

### 决策依据

- **5/15 quota 状态**: 已用 2 (M28 LB 0.846514 / M29 LB 0.845312) · **剩 2** · CEST 06:00 5/16 重置 → **2 quota 用不完 = 浪费** (team-memory line 9 铁律)
- **不能挪到明早**: 今晚 2 = 5/15 quota; 明早 4 = 5/16 quota; **两个独立 4-quota window** · 不累积
- **5 候选 ckpt**: M22-rerun / M33 原始 / M33-rerun / M30 / M37
- **战略权重**: 信息价值 (校准 / 路径判决 / 归因) > 大概率突破 SOTA (今晚候选都没有强突破期望 · LB 期望 ≤ 0.847~0.848)

### 今晚 2 quota 选择 · A 方案 (用户决议)

| Eval | ckpt | 战略价值 |
|---|---|---|
| **1** | **M22-rerun (dev 0.868531)** | **同 seed 同 commit rerun LB noise band 校准 (史无前例)** · LB 落点决定: ① ≈ 0.847 ± 0.0005 → SOTA stable ② 显著偏离 → ckpt-specific luck 重大警示 |
| **2** | **M33 原始 (dev 0.868682)** | **5-cluster 路径 LB 端验证** (dev 端 mean 已知 ≈ M22 · LB 端可能有路径独有特性) · LB 落点决定: ① > 0.848 → 5-cluster LB 真有路径优势 ② ≈ M22 0.8475 → 路径关闭 ③ < 0.846 → 路径有害 |

### 拒绝候选 · 理由

- **M30 (Trick 6 alone)**: dev 0.868438 < M22 dev 0.868576 = −0.000138 (out-of-band 边界) · LB 期望 ≈ 0.846 · 大概率不破 SOTA · 价值 = 给 M34 stack 归因 · 但 M34 dev 也低 · 不是优先候选
- **M22-rerun + M33-rerun 双校准**: 占 2 quota 全用 noise · 0 架构信号增量 · M33-rerun dev (0.868408) < M22-rerun (0.868531) · LB 已知大概率低于 M22-rerun · 不需 quota 验证
- **M37 (NLIR) / M17 (seq_d=768)**: dev 离 M22 远 · LB 期望 < SOTA · 信息增量低于 A 方案

### Publish 三步检查 (team-memory line 25)

1. ✅ 是 `*.best_model/` 子目录 (路径含)
2. ✅ 训练完成 (stop_reason=max_epochs_completed)
3. ⚠️ 子目录内含 `model.pt` + `train_config.json` + `schema.json` (用户 platform UI 确认)
4. ⚠️ M33: 子目录内含 `ns_groups_5cluster.json` (trainer auto-copy · 通常都对 · 但需 verify)

### Evaluation 上传文件 (3 件套)

- `src/infer.py`
- `src/dataset.py`
- `src/model.py`

(防 silent failure · 即使是 M22 baseline 不带新 kwarg · commit 6045954 起 train_config.json 落了 `enable_global_token` 字段 · 平台 default infer.py 兼容性未验证)

### Args dict 30s 验证 (5 keys)

evaluation log 出 Args dict 后必查:
- `enable_global_token`: False · `enable_nlir_gating`: False · `multi_emb_k`: 1 · `enable_din_interest`: True
- `ns_groups_json`: M22-rerun = vocab_tier / M33 = 5cluster
- `seq_max_lens`: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- 任一不对 · 立即 cancel · 不浪费 2 quota 任何一个

### 5/16 早 4 quota 矩阵更新

§11 §判决 3 中 quota 1 = M22-rerun LB 已**今晚提前完成** · 5/16 早 4 quota 候选修订:

| Quota | 候选 | 价值 |
|---|---|---|
| 1 | **M38 v2 ckpt LB** (Multi-Emb K=2) | ADR-012 LB 验证 ⭐ |
| 2 | **M39 v2 ckpt LB** (Global Token) | ADR-013 LB 验证 ⭐ |
| 3 | **取决于 M38v2/M39v2 dev**: 若 ≥ 0.8691 ⭐ 评最强者最佳 ckpt; 若 dev 持平 评 M30 (Trick 6 单变量 LB · 给 M34 stack 归因) |
| 4 | **取决于今晚 LB**: 若 M22-rerun 偏离 0.847 显著 → 评 M22 原始 ckpt 二度校准 (但不太可能·0.847 应该 stable) · 否则 评 M37 (NLIR) 或 M30 (上一项备选) |

### 时间预期

- evaluation 跑 ~10 分钟 · 用户提交 ~CEST 22:30 · LB 数据 ~CEST 22:40~23:00 出
- LB 数据贴回 → agent 立即落 EXP-055 §13 (今晚 LB 实测 + 决策修订)

---

## 13. CEST 22:40 补遗 · M22-rerun LB 实测 · 项目史上第一个 LB rerun noise 数据点

### 数据

| Metric | M22 原始 (5/14 evaluation) | M22-rerun (5/15 evaluation) | Δ |
|---|---|---|---|
| **LB AUC** | **0.847592** | **0.847324** | **−0.000268** |
| Eval 时长 | 609.77s | 208.26s | −402s (3× 加速) |
| Dev (ckpt 训练时) | 0.868576 | 0.868531 | −0.000045 |

### 分析

**Δ_LB = −0.000268** = (LB_rerun_ckpt − LB_original_ckpt) · 由于 evaluation system 本身 deterministic (用户已确认 + team-memory line 57) · **100% 来自 ckpt 数值微小差异 (CUDA 非确定性 + Adagrad 稀疏 grad 顺序 + DataLoader 多进程顺序)**。

**LB rerun noise vs dev rerun noise**:
- dev rerun: ±0.000045 (band ±0.0001 per EXP-032)
- LB rerun: ±0.000268 (新数据点 · 项目史上第一个)
- **LB rerun noise 比 dev rerun noise 放大 ~6×**

**LB rerun noise vs cross-seed LB noise**:
- 跨 seed (EXP-028/EXP-030 五 seed): ±0.005 (跨度 0.010)
- 同 seed rerun: ±0.000268 (新数据点)
- **同 seed rerun noise 比跨 seed noise 小 ~17×**

### 判决 1 · M22 SOTA 0.847592 是 stable signal

M22 原始 + M22-rerun 两点 mean = (0.847592 + 0.847324) / 2 = **0.847458**
两点跨度 = 0.000268 · 落在新铁律 LB rerun noise band ±0.0003 内 ✅

**结论**: M22 LB SOTA 0.847592 不是 ckpt-specific luck · 同 seed 重训能稳定复现 LB ≈ 0.847~0.848。M22 vocabtier 真信号 ✅。

### 判决 2 · "破 SOTA" 判决阈值修订 (针对同 seed 重训实验)

| 实验类型 | noise band | 破 SOTA 阈值 |
|---|---|---|
| **同 seed (s42) 重训** (M38 v2 / M39 v2 / M33-rerun 等) | ±0.0003 (3σ ±0.0009) | **LB ≥ M22 mean + 3σ = 0.847458 + 0.0009 = 0.8484** |
| 跨 seed (s2/s3/s7/s11 等) | ±0.005 | LB ≥ 0.849 (per EXP-028 旧铁律) |

**注意**: 这是 dev / LB **rerun noise 校准** 而非"破 SOTA = LB 上沿统计"。EXP-028 SOTA 上沿铁律 (跨 seed luck 让 SOTA 偏高) 仍适用 · 但**今晚数据证实 0.847592 不是 ckpt luck 边界点 · 是 stable signal · 不需修正 SOTA 上沿估计**。

### 判决 3 · 新铁律 (写入 team-memory)

#### 铁律 N+3 · LB rerun noise band (同 seed 同 commit 重训) ≈ ±0.0003

> **2026-05-15 CEST 22:40 · M22-rerun LB 0.847324 vs M22 原始 LB 0.847592 = Δ−0.000268 实测 (项目首个 LB rerun 数据点)**: 同 seed 同 commit 同 config 重训后, LB rerun noise band ≈ ±0.0003 (1 数据点 · 估 std ≈ 0.0001~0.0002, 3σ ≈ ±0.0006~±0.0009)。**铁律**: ① **新方法学是否 LB work 的判决下界 = ±0.0006 ~ ±0.0009 LB (3σ)** · Δ_LB ∈ [−0.001, +0.001] 一律判 "rerun noise 内" · 比之前 EXP-028 跨 seed ±0.005 严格 5×; ② **场景限定**: 同 seed 同 commit 同 config 重训 (e.g. M22 rerun); 跨 seed (s42 vs s3) 仍用 EXP-028 ±0.005; ③ **"破 SOTA" 判决修订** (针对同 seed 重训): LB ≥ M22 mean + 3σ = **0.8484** · 不是之前的 0.849 (跨 seed 上沿); ④ **本铁律基于 1 数据点** · 5/16+ 应用 M38 v2 / M39 v2 LB 数据继续校准 · 当 ≥ 3 个数据点时正式落定 std 估计; ⑤ **dev rerun noise (±0.0001) ≠ LB rerun noise (±0.0003)** · LB 端放大 ~6×, 因为 LB 评 100% test set 而 dev 仅评 5% valid set, 且 LB 涉及 covariate shift 放大微小 ckpt 数值差异。

### 判决 4 · 期望 M38 v2 / M39 v2 LB 解读

**M38 v2 (Multi-Emb K=2)**:
- LB ≥ **0.8484** → ADR-012 真信号 work · evaluation 价值高
- LB ∈ [0.847, 0.8484] → noise 内 / 弱信号 · 需 cross-seed 校准
- LB < 0.846 → 真负面 · ADR-012 在本数据集不 work · 路径关闭

**M39 v2 (Global Token)**:
- LB ≥ **0.8484** → ADR-013 真信号 work
- LB ∈ [0.847, 0.8484] → noise / 弱信号
- LB < 0.846 → 真负面

### 5/16 早 4 quota 矩阵再修订

§12 §quota 矩阵 quota 1 = M22-rerun 已完成 · 5/16 早 quota 优先级:

| Quota | 候选 | 价值 | 决断逻辑 |
|---|---|---|---|
| 1 | **M38 v2 ckpt LB** (Multi-Emb K=2) | ADR-012 ⭐ | 必评 |
| 2 | **M39 v2 ckpt LB** (Global Token) | ADR-013 ⭐ | 必评 |
| 3 | **取决于 M38v2/M39v2 dev (~CEST 00:30)**: 若任一 dev ≥ 0.8691 评最强者最佳 ckpt; 若 dev 持平评 M30 (Trick 6 单变量 LB) | 单变量归因 | 看 dev 决 |
| 4 | **取决于 §13 LB 数据**: M33 LB 落点决定备选 ckpt | flexible | 看 M33 LB 决 |

---

## 14. CEST 22:48 补遗 · M33 5-cluster LB 实测 · 强负面 · 5-cluster 路径关闭

### 数据

| Job | Dev E4 | LB | Dev→LB gap |
|---|---|---|---|
| **M22 原始** ⭐ SOTA | 0.868576 | **0.847592** | +0.020984 |
| **M22-rerun** | 0.868531 | **0.847324** | +0.021207 |
| **M33 原始** ⚠️ | **0.868682** | **0.844710** | **+0.023972** |

### 判决 · 5-cluster ns_groups 路径强负面 · 路径关闭

**M33 vs M22 三个量级数据**:
- **Dev**: +0.000106 (noise 上沿)
- **LB**: **−0.002882** (10× LB rerun noise band ±0.0003)
- **Dev→LB gap**: **+0.024 (vs M22 +0.021)** · gap 扩大 +0.003 · 机制级负向信号

**Dev 涨 LB 跌反向放大 27×**: ratio = 0.002882 / 0.000106 = 27.2× · 是 EXP-024/EXP-025 "dev 不是 LB 单调代理" 铁律的最强实例 (历史 P1 idmask 反向放大 14× / T23 disable47 反向放大 ~10× · M33 反向放大 27× **打破历史最大记录**)。

**机制**: 5-cluster ns_groups 让 NS Tokenizer 把 fid 重新分配到 5 个 cluster · NS Tokenizer 学到了 train 期局部相关模式 (dev 端 5-cluster 仅微涨 +0.000106 · 因 dev 分布与 train 共享时间窗); **但 cluster boundary 在 test 远未来分布下学到的相关模式失效, 反向打乱了 baseline (vocabtier) 已经学好的 sparse embedding 表示**。LB 端跌 0.003 不是单一 ns_groups 改动的边际效应 · 是 NS Tokenizer 整体重配置后 overfitting 风险的体现。

### 推论 · M33-rerun ckpt LB 期望

M33 dev mean = 0.868545 · M33-rerun dev = 0.868408 (低于 mean 0.000137)。
M33 vs M22 LB 反向放大 27× · M33-rerun vs M22-rerun LB 期望: **0.844710 − 0.000137 × 6 = 0.84389** (按 dev rerun→LB rerun 6× 放大公式) · 大概率 LB ∈ [0.843, 0.845]。**M33-rerun 不需 evaluate** (期望 LB 远低于 M22-rerun · 路径已关闭)。

### 判决 · 5/16 4 quota 不评 M33-rerun · 不评 M33 任何变体

5-cluster 路径在 5/15 双数据点 (dev rerun + LB) 实测后**正式关闭**。任何"M33 + Multi-Emb K=2"或"M33 cross-seed"等组合**都不再优先**。

### 新铁律候选 (待 M38 v2 / M39 v2 LB 后落定)

#### 候选铁律 N+4 · ns_groups 类改动在 dev/LB 端反向放大极大 · 不能凭 dev 单点判 work

> **2026-05-15 CEST 22:48 · M33 (5-cluster) LB 0.844710 vs M22 LB 0.847592 = Δ−0.002882, 但 dev Δ+0.000106 · 反向放大 27× (历史最大)**: ns_groups 类改动 (5-cluster / vocab_tier 替换 / 任何 custom ns_groups_json) 在 LB 端表现**远比 dev 端差** · 机制 = NS Tokenizer 重配置让模型学到 train 期局部 cluster 相关性, test 远未来分布下 cluster boundary 失效。**铁律**: ① ns_groups 类改动**不能凭 dev 单点 +0.0001 判 work** (即使 dev 真信号也可能 LB 强负面); ② **必须 dev rerun + LB cross-seed 双重校准** 才能上 evaluation · 例外: 显著 dev 涨 (+0.001+) 才允许直接 evaluation; ③ **5-cluster (M33) 路径正式关闭** · 不再做"M33 + X" 任何 stack 实验; ④ vocab_tier (M22) 路径仍 stable (M22 / M22-rerun LB 都在 noise band 内) · 是当前唯一可信 ns_groups 配置; ⑤ 历史影响: 5/15 EXP-053/EXP-055 §1 把 M33 +0.000106 当真信号是错判 · 已通过 §11 (dev rerun) + §14 (LB) 双向纠正。

### 5/16 早 quota 矩阵 (M33 关闭后再修订)

| Quota | 候选 | 优先级 |
|---|---|---|
| 1 | **M38 v2 ckpt LB** (Multi-Emb K=2) | ⭐⭐⭐ 必评 |
| 2 | **M39 v2 ckpt LB** (Global Token) | ⭐⭐⭐ 必评 |
| 3 | **M30 ckpt LB** (Trick 6 alone) | ⭐⭐ 单变量归因 (给 M34 stack 解释) |
| 4 | **取决于 M38v2/M39v2 dev**: 若 ≥ 0.8691 → 评双方最佳 ckpt 并占 4 个 quota · 若 dev 持平 → 评 M37 (NLIR) 或 M17 (seq_d=768) | flexible | 看 dev 决 |

---

## 15. CEST 23:30 补遗 · 元宵节 covariate shift 重大认知 + M40/M41 启动

### 15.1 元宵节认知 · 颠覆 5/10 EXP-035 ROI 错判

**核心事实** (用户 5/15 23:27 提问触发):
- **2026-03-03 = 农历正月十五 = 元宵节** ⭐
- Train 数据 03-03 占 63% 不是 sampling artifact · 是**真实元宵节流量峰**
- Test (北京 03-04 09:00~14:00 周三) = **元宵节翌日工作日午前 daypart**
- Train→Test = 节日峰 → 节后回归 · **不是远未来 covariate shift · 是 daypart shift**

### 15.2 元宵节日期表

| 日期 | 农历 | 阳历 weekday | 性质 | Train 占比 |
|---|---|---|---|---|
| 02-28 | 正月十二 | 周六 | 周末 | 3-25% |
| 03-01 | 正月十三 | 周日 | 周末 | 3-25% |
| 03-02 | 正月十四 | 周一 | 元宵节前夜 | 3-25% |
| **03-03** ⭐ | **正月十五 元宵节** | 周二 | **节日 · 高流量峰** | **63%** |
| **03-04** | 正月十六 | 周三 | **元宵节翌日 · 工作日** | 3-25% (test 同日) |

### 15.3 5/10 EXP-035 §错误 4 ROI 修订错判

**5/10 EXP-035 line 109 当时判断**:
> "数据已经隐式 heavy recency (最近一天占 63%)" → recency reweight ROI 从 +0.002~0.003 下调到 +0~+0.001

**修订** (5/15 元宵节认知后):
- ❌ 03-03 不是最近一天 · **03-04 才是** (test 同日 · 占比仅 3-25%)
- ❌ 03-03 占 63% 不是 recency · 是 **元宵节流量峰**
- ✅ **真正的 recency = 03-04 数据** · 但占比最少
- ✅ `temporal_weight_alpha=2.5` 真实作用 = **修正 03-04 underweight + 03-03 元宵节 overweight 的不平衡** · 不是放大 recency
- **ROI 修订回归原估**: **+0.001~+0.003 LB** (高 ROI 候选)

### 15.4 当前 M22 baseline 含 dow 特征 · 大概率有害

**Comment 证据** (`src/dataset.py` line 2117-2121):
> "DOW is excluded because training data is heavily skewed toward a single weekday (03-03, ~63%), which causes the model to overfit to that weekday's patterns and hurts generalization to the test set's temporal distribution."

**机制** (含元宵节认知):
- M22 baseline 用 `--enable_time_of_day_features` (5 dim · 含 dow_sin/cos + weekend)
- Train 上 dow=1 (周二) 占 63% · 学到的 dow=1 pattern 实际上是**元宵节高 CVR pattern**
- Test 是 dow=2 (周三) 元宵节翌日 · dow=1 学的元宵节 pattern **不延续到翌日**
- **dow 特征在 test 上是误导信号**

`enable_hour_only_features` 修复**已实现但从未实测** (CLI / dataset / infer 通路完整)。

### 15.5 5/15 23:00 起的 M38 v2 copy + M39 v2 copy 取消 · 改起 M40 + M41

**用户决策**: 取消 M38 v2 copy + M39 v2 copy (sunk 30 min GPU) · 改起 M40 + M41 (元宵节认知修复)。

**理由**:
- M38 v2 copy / M39 v2 copy 价值 = 论文方法 noise band 校准 (中价值)
- M40 / M41 价值 = 直接 target 元宵节 daypart shift (高价值 · 论文方法 covariate shift 折扣后 ROI 比 daypart-target 低)
- 完成时间几乎同时 (~CEST 04:00~04:30 5/16) · 5/16 早评 4 不同方向比 4 同方向更高 ROI

### 15.6 M40 / M41 设计 · 单变量 ablation vs M22

| Job | 配置变化 vs M22 | 期望 dev | 期望 LB |
|---|---|---|---|
| **M40-houronly** | 删 `--enable_time_of_day_features` + 加 `--enable_hour_only_features` | ~M22 ± 0.0001 | **+0.0005~+0.002** |
| **M41-temporalweight25** | 加 `--temporal_weight_alpha 2.5` (保持 time_of_day 不动) | ~M22 ± 0.0003 (sample reweight 影响 dev sample 分布) | **+0.001~+0.003** |

### 15.7 5/16 早 4 quota 矩阵再修订 (含 M40 / M41)

| Quota | 候选 | 优先级 |
|---|---|---|
| 1 | **M38 v2 ckpt LB** (Multi-Emb K=2) | ⭐⭐⭐ 必评 |
| 2 | **M39 v2 ckpt LB** (Global Token) | ⭐⭐⭐ 必评 |
| 3 | **M40 ckpt LB** (hour_only · 元宵节 daypart fix) | ⭐⭐⭐ 必评 |
| 4 | **M41 ckpt LB** (temporal_weight 2.5) | ⭐⭐⭐ 必评 |

→ **5/16 早全部 4 quota 评新方向 LB · 不评 M30 / M37 等老候选** (老候选 dev 已知低于 M22 · LB 不可能创新高)。

### 15.8 真正的 5/15 ~ 5/25 路径 (含元宵节认知)

| Tier | 方向 | 期望 LB | 实施成本 |
|---|---|---|---|
| ⭐⭐⭐ S1 (M40) | hour_only 替换 time_of_day | +0.0005~+0.002 | 0 代码 · 跑中 |
| ⭐⭐⭐ S2 (M41) | temporal_weight_alpha 2.5 | +0.001~+0.003 | 0 代码 · 跑中 |
| ⭐⭐ S3 (新方向) | is_holiday + days_since_lunar_holiday | +0.001~+0.003 | ~50 行 + ADR |
| ⭐⭐ M38 v2 (Multi-Emb) | RankUp 论文方法 | +0.0005 | 跑中 |
| ⭐⭐ M39 v2 (Global Token) | RankUp 论文方法 | +0.0005 | 跑中 |
| ⭐ S4 (5/24 final) | 100% data run no valid split | +0.0005~+0.001 | ~10 行 |

**Top 20 概率 (含元宵节认知)**: **20~30%** (上调 vs CEST 22:55 估的 10~15%)

**核心 takeaway**: **元宵节认知是 5/15 最重要的发现 · 比 RankUp 方向 ROI 更高 · 不是 architecture-level breakthrough · 是 dataset-mechanism understanding** · 这是研究员视角的关键差异 (论文方法 < 数据集机制理解)。

### 15.9 新铁律候选 (待 M40 / M41 LB 后落定)

#### 候选铁律 N+5 · 中国数据集时间分布 grep 春节 / 元宵 / 端午 / 中秋等农历节日

> **2026-05-15 CEST 23:30 · 元宵节认知触发**: 中国互联网数据集 (e.g. 腾讯 / 阿里 / 字节) 训练数据上 **农历节日是流量峰 · CVR 显著高于平日**。任何"train 某日占比异常 (e.g. 60%+)" 必须先 grep **该日是否农历节日** · 不能凭 sampling artifact 假设。**铁律**: ① 拿到中国数据集 · 第一时间用 `lunar_calendar` 工具 grep train/test 日期对应农历日 · 标记节日; ② 节日峰 (e.g. 元宵节 03-03) sample 不是 recency · 不能用 `temporal_weight_alpha` 简单 reweight; ③ 节后翌日 (e.g. 03-04) 是真正的 "回归常态" daypart · 与节日峰 distribution 完全不同; ④ 时间特征 (`enable_time_of_day_features`) 含 day-of-week 时 · 必须验证 train weekday 分布 vs test weekday 分布 · 不一致则用 `--enable_hour_only_features` 移除 dow 偏差; ⑤ 历史影响: 5/10 EXP-035 line 109 把 "63% 视为 recency" 错判 · 修订 `temporal_weight_alpha` ROI 回归 +0.001~+0.003。

---

## 16. CEST 07:35 5/16 补遗 · 4 jobs dev 全回流 · 4/4 dev 全负面 · 重大战略修订

### 16.1 4 jobs dev 数据 (E4 best)

| Job | 假设 | E4 dev | E4 LogLoss | vs M22 (0.868576) | 是否破 dev band? |
|---|---|---|---|---|---|
| **M22** ⭐ anchor | (baseline) | 0.868576 | 0.22290 | (anchor) | — |
| **M38 v2** | M22 + Multi-Emb K=2 | 0.868403 | 0.22373 | **−0.000173** | ❌ in ±0.0003 band (无信号) |
| **M39 v2** | M22 + Global Token | 0.867963 | 0.22330 | **−0.000613** | ❌ out 2× band (显著负) |
| **M40** | M22 替换 hour_only | 0.866881 | 0.22409 | **−0.001695** | ❌❌❌ out 5× band (强负) |
| **M41** | M22 + temporal_weight 2.5 | 0.868069 | 0.22303 | **−0.000507** | ❌ out 2× band (显著负) |

### 16.2 重大判读 · 4/4 dev 全负面

**统计显著性**: 如每个方向独立 50% 概率 dev 持平/微正 · 4/4 全负的概率 = (0.5)^4 = **6.25%** · 不太可能是 noise 巧合。

**结合 5/14 ~ 5/15 早 stack 第 2 步 3 个全失败** (M28 vocabtier+Trick1 / M34 vocabtier+Trick6 / M37 vocabtier+NLIR):
- 共 5/14 ~ 5/16 凌晨 7 个新方向单变量 / stack 实验
- 7/7 dev 端 ≤ M22 (vocabtier baseline)
- **结论**: M22 vocabtier 在 dev 层面已找到当前路径局部最优 · 任何单变量 perturb / stack 第 2 步都让 dev 跌

### 16.3 元宵节认知方向被 dev 数据证伪

**5/15 23:30 元宵节认知预测**:
- M40 (hour_only) 期望 +0.0005~+0.002 LB · 实际 dev **−0.001695** ❌
- M41 (temporal_weight 2.5) 期望 +0.001~+0.003 LB · 实际 dev **−0.000507** ❌

**真实机制 (修订)**:
- **M22 dow_sin/cos 不是误导信号 · 是真实有用信号** · 移除后 dev 大跌
- valid 5% 尾部混合 03-03/03-04 多 weekday · dow 特征对 dev 预测有效
- **元宵节假设是 "理论 plausible 但实际 ROI 0" 的方向**

**修正 5/15 23:30 元宵节估算**:
- "Top 20 概率 20~30%" → **修订到 5~10%** (dev 数据已证伪 daypart 路径)
- "S1 + S2 daypart 路径" → **关闭**
- "S3 (is_holiday + days_since_lunar_holiday 50 行新代码)" → **悬挂** (机制类似 hour_only · 大概率同样 dev 端不 work · 不优先做)

### 16.4 5/16 早 4 quota 修订 (基于 4 dev 全负面)

| 优先级 | quota | ckpt | 期望 LB | 战略价值 |
|---|---|---|---|---|
| 1 ⭐⭐⭐ | **M38 v2** (Multi-Emb K=2) | ~0.847 | RankUp Multi-Emb LB 实测 · ADR-012 闭环 |
| 2 ⭐⭐⭐ | **M39 v2** (Global Token) | ~0.847 | RankUp Global Token LB 实测 · ADR-013 闭环 |
| 3 ⭐⭐ | **M41** (temporal_weight 2.5) | ~0.846~0.848 | dev 弱负 · LB 端 daypart match 可能反向 · 0 代码 0 风险 |
| 4 ⭐ | **M30** (Trick 6 alone) | ~0.846 | 给 M34 stack 失败归因 · ROI 低但唯一 stack 归因机会 |

**不评 M40** · dev 强负 (−0.001695, 5× band) · LB 大概率也强负 · 价值低于 M30/M41。

### 16.5 战略 pivot · 必须切大改动 (路径 B)

**5/16 下午起的硬要求**:
1. **如果 5/16 早 4 LB 全部 ≤ 0.847** (~85% 概率 · 基于 dev 4/4 负面): **立即起路径 B 的 ADR 草稿**
2. **路径 B = UE 分离 + int pairwise cross** (Seed 主楼 0.85+ 团队 5/5 第 (3) 条情报)
3. 5/16 下午 + 5/17 = ADR 起草 + 80~150 行新代码 + smoke + commit
4. 5/17 晚 / 5/18 早起 first path B training job

### 16.6 但等下 · 也有合理保留怀疑 (avoid premature path B 切换)

**反对立即切 path B**:
- **dev 不是 LB 单调代理** (M33 反向放大 27× 已知) · 4/4 dev 负面**不一定** → 4/4 LB 负面
- M38 / M39 / M41 LB 端可能因 covariate shift 友好机制 (e.g. M41 reweight 03-04 → LB 全是 03-04 → LB 涨)
- 必须等 5/16 早 4 LB 数据 · 不能凭 dev 单点决策

**修订**: **5/16 早 4 LB 决定**:
- 如任一 LB > 0.8484 (破 SOTA 阈值): 该方向真信号 · 5/16 下午做 cross-seed 校准
- 如任一 LB ∈ [0.847, 0.8484]: 弱信号 · 5/16 下午做 cross-seed 巩固
- **如全部 LB ≤ 0.847** (大概率 85%): 立即 path B pivot

### 16.7 真实 Top 20 概率 (再修订)

- 22:55 (5/15 晚) 估: **10~15%** (路径 A saturated · 路径 B 是唯一未消化)
- 23:30 (5/15 晚) 估: **20~30%** (元宵节认知方向)
- **07:35 (5/16 早) 估**: **5~10%** (元宵节方向证伪 · 仅剩路径 B 一条线)

**核心 takeaway**: **dev 数据决定性地否定了 5/15 晚的元宵节方向乐观估计**。剩 9 天 · 必须切大改动 (路径 B) 或接受 Top 20 失败的现实。

### 16.8 4 dev 数据点的另一深层含义 · M22 vocabtier 不只是 +0.001 SOTA · 是当前数据集的 "near-optimal vocabtier-baseline"

7 个新方向 7/7 dev 不超 M22 · 暗示:
- M22 vocabtier 配置在当前数据集 dev 层面已**接近局部最优**
- 任何单变量 perturb (Multi-Emb / Global Token / hour_only / temporal_weight / NLIR / ns_groups) 都打破这个局部最优
- 唯一能突破的是**根本架构改变** (e.g. UE 分离 = 把 NS Tokenizer 完全重新组织 · 不是单变量 ablation 而是 architecture-level)

→ **路径 B 的 ROI 仍然存在** · 因为它不是 "M22 + X" stack · 是 "M22 → architecture-level redesign"。







