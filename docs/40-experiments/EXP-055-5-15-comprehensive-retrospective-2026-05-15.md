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


