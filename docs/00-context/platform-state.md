# Platform 实时状态 (Single Source of Truth)

> **每次 agent 推荐 platform 操作前必须 grep 此文件**。任何资源状态陈述 (slot / quota / in-flight) 凭此文件 · 不凭印象。
>
> **Last updated**: 2026-05-17 CEST 18:35 (5/17 晚 4 jobs in-flight: M73/M74/M75/M50-rerun · M76 等 M50 重跑完接力)

---

## Maintenance Protocol (硬流程)

**用户每次以下事件后 · agent 必须先 update 本文件再做后续推荐**:
1. 贴 platform job 启动 log (Args dict) → update In-Flight Jobs
2. 贴 platform job 完成 log (Training Summary) → update Last Job Status + free slot count
3. 贴 evaluation LB 结果 → update quota used + LB results
4. 反问 "现在还有几个 slot / quota" → agent 必须 grep 本文件回答 · 不凭印象

**如本文件 last updated > 30 min 或 agent 不确定准确性**:
- agent 必须 ask 用户确认 · **不能凭印象推荐 platform 操作**
- 用户回答后 agent 立即 update 本文件 + 改动 commit

---

## Training Slots (4 total · 完成即释放 · 任意时刻可起)

> 来源: team-memory line 175 ("训练 slot 与 evaluation quota 是两个完全独立的资源池")

### Current In-Flight (5/17 CEST 18:35 起)

| Slot | Status | Job | 起 (CEST) | 完成预估 (CEST) | 配置 |
|---|---|---|---|---|---|
| 1 | 🟡 **in-flight** | **M73** (path B × seed=2) | ~17:00 5/17 | ~22:00 5/17 | path B cross-seed 第 3 数据点 |
| 2 | 🟡 **in-flight** | **M74** (path B × seed=5) | ~17:00 5/17 | ~22:00 5/17 | path B cross-seed 第 4 数据点 |
| 3 | 🟡 **in-flight** | **M75** (path B + per-domain T38 v2) | ~17:00 5/17 | ~22:00 5/17 | T38 v2 fix + UE stack |
| 4 | 🟡 **in-flight** | **M50 重跑** (path B + weekend filter) | ~17:30 5/17 | ~22:30 5/17 | M50 ckpt 5/17 15:53 已过期 · 重训保 LB 评估 |

**Pending (M50 重跑完接力)**:
- ⏸️ **M76** (path B + DCN cross · T34 ready) · 期望启 ~CEST 22:30 · 完成 ~03:30 5/18 · ckpt 24h 过期 5/19 03:30 OK

**Note**: M73/M74/M75 用户已 self-verify Args 5 keys all OK (CEST 18:35 用户确认)。

### Last Completed Jobs (按时间倒序 · 5/17 14:15~21:26)

5/17 早起 4 jobs 全部完成:

| Job | Config | dev (E4 best) | E3 dev | LB | Notes |
|---|---|---|---|---|---|
| **M68** | path B + Q_init + DIN bypass (3 stack) | 0.870376 | 0.869659 | **0.846468** ❌ | Stack net −0.001694 LB · Q_init in path B 失败 |
| **M65** | path A + per-domain T38 v2 + DIN bypass | 0.868973 | 0.868623 | 待评 | T38 v2 fix 平台首验 ✅ · 弱真信号 |
| **M68 v2** | M68 same config rerun | 0.870355 | 0.869836 | 待评 | M68 dev rerun band ±0.0001 内 |
| **M71** | path B + DIN × s11 (cross-seed) | 0.870344 | 0.869952 | **0.846641** ⚠️ | path B cross-seed 第 2 数据点 · 下沿 |

### Training Summary 关键 ckpt (`keep_topk_ckpt=4` · top4 全保留)

5/17 早 4 jobs 各 4 ckpt (E1/E2/E3/E4) 都保留 · ckpt selection 战略 ready (但首测未做)。

5/16 晚 4 jobs (M51/M52/M53/M54): training_summary 没在 docs 详细记录 · 但 ckpt 5/18 04:22 24h 过期红线 · 需 5/18 早 quota refresh 后立即 evaluate (4 quota 全用 + 部分 ckpt selection)。

---

## Evaluation Quota (4/day · 北京 12:00 = CEST 06:00 重置 · 不累积)

> 来源: team-memory line 9

### 5/17 (today)

| Used | Remaining | Used by |
|---|---|---|
| **3** | **1** | M55 LB 0.844343 + M71 LB 0.846641 + M68 LB 0.846468 |

**M50 LB · 今晚剩 1 quota** (待 M50 重跑完成 ~22:30 后立即 evaluate · 用户已决)。

**🟢 失败 evaluation 不消耗 quota** (5/17 用户告知铁律): M68 第 1 次 evaluation state_dict mismatch 失败 (q_init_item key bug · commit `cfd15f7` 修复) · 重提成功。

### 5/18 (next day) · 4 quota refresh @ CEST 06:00

**5/18 早 4 quota 候选** (优先级修订 · 5/17 18:35):

| 优先级 | 候选 | 触发条件 | 期望 LB |
|---|---|---|---|
| ⭐⭐⭐ | **M53 LB** (5/16 晚 path B + NLIR · ckpt 5/18 04:22 过期 风险!) | publish-ready · 必须明早评 | 0.846~0.848 |
| ⭐⭐⭐ | **M51 LB** (5/16 晚 path B + temporal_weight · ckpt 5/18 04:22 过期 风险!) | publish-ready · 必须明早评 | 0.846~0.848 |
| ⭐⭐⭐ | **M75 LB** (今晚 path B + per-domain T38 v2 stack) | training_summary 出 + dev verify | 0.847~0.849 |
| ⭐⭐ | **M65 LB** (path A + T38 v2) | training_summary 已出 dev 0.868973 | 0.846~0.847 |
| ⭐⭐ | **M68 v2 LB** (M68 rerun · LB rerun band confirm) | dev 已出 | 0.846~0.847 |
| ⭐⭐ | **M68 E3 ckpt LB** (ckpt selection 战略首测 · E3 dev 0.869659) | E3 ckpt verify | 0.846~0.848 |
| ⭐ | **M73/M74 LB** (path B cross-seed s2/s5) | dev 出 | 0.845~0.848 |
| ⭐ | **M76 LB** (path B + DCN) | training 完成 ~03:30 5/18 | 0.847~0.848 |

**5/18 早 4 quota 推荐**: M53 + M51 (ckpt 24h 过期) + M75 (T38 v2 + UE 真测) + M68 E3 ckpt (ckpt selection 首测) **或** M76 (T34 DCN 首测)。

### 历史 quota 使用

| Day | Used | Remaining | Used by |
|---|---|---|---|
| 5/17 (today) | 3 | 1 | M55 0.844343 + M71 0.846641 + M68 0.846468 |
| 5/16 (closed) | 4 | 0 | M40 LB 0.846931 + M41 LB 0.847846 + M23 retry LB 0.836631 + **M49 LB 0.848162 ⭐ SOTA** |
| 5/15 (closed) | 4 | 0 | M28 LB 0.846514 + M29 LB 0.845312 + M22-rerun LB 0.847324 + M33 LB 0.844710 |
| 5/14 (closed) | 1 | (overflow ❌) | M22 LB 0.847592 (前 SOTA · 现 #2) |

**Next reset**: CEST 06:00 5/18 = 北京 12:00 5/18

---

## Publish-Ready Ckpt 状态 (5/17 CEST 18:35 audit)

| Job | Dev | Publish | 24h 过期 (CEST) | LB |
|---|---|---|---|---|
| **M22** (vocabtier × s42) | 0.868576 | ✅ historical | safe | **0.847592** ⭐ #2 |
| **M22-rerun** | 0.868582 | ✅ | safe | 0.847324 |
| **M49** ⭐⭐⭐ (UE split path B) | 0.870741 | ✅ | safe | **0.848162** ⭐ SOTA |
| **M51** (path B + temporal_weight) | TBD | ⚠️ user 已决今晚 publish | **5/18 04:22 ⚠️** | 待评 |
| **M52** (path B + per-domain · M52' silent skip) | TBD | TBD | 5/18 04:22 ⚠️ | 待评 |
| **M53** (path B + NLIR) | TBD | ⚠️ user 已决今晚 publish | **5/18 04:22 ⚠️** | 待评 |
| **M54** (per-domain alone · M54 silent skip) | TBD | TBD | 5/18 04:22 ⚠️ | 待评 |
| **M50 (5/16 训)** | 0.861974 | ❌ **已过期** (5/17 15:53) | gone | LB 永远评不到 |
| **M50 重跑** (今晚 ~22:30 完) | TBD | TBD | 5/18 22:30 OK | 今晚剩 1 quota 评 |
| **M68 / M65 / M68 v2 / M71** (今早) | (见 above) | 待 publish | 5/18 19:00~21:30 | 5/18+ 评 |
| **M30** (Trick 6) | 0.868438 | ✅ | safe | 待评 (LB 决 Q_init 路径) |
| **M37** (NLIR) | 0.868311 | ✅ | safe | 待评 |
| **M38 v2** (Multi-Emb K=2) | 0.868403 | ✅ | safe | 待评 |
| **M39 v2** (Global Token) | 0.867963 | ✅ | safe | 待评 |
| **M27 v2** (T31+user_15) | 0.868166 | ✅ | safe | 待评 |
| **M17** (seq_d=768) | 0.867794 | ✅ | safe | 待评 |
| ~~M34~~ / ~~M45~~ / ~~M48~~ | — | ❌ 已丢失 / dev 灾难 | gone | 不评 |

---

## Strategic Status (5/17 CEST 18:35 修订 · 5/20 ddl 战略约束下)

**LB SOTA**: **M49 (UE split path B · seed=42) = 0.848162** ⭐ (5/16)
**前 SOTA**: M22 (vocabtier × s42) = 0.847592 (5/14)
**距 0.852 (用户硬 deadline)**: 还差 +0.003838
**距 0.85 (Top 20)**: 还差 +0.0018

**用户硬 deadline (5/17 CEST 16:14 立)**: **0.852 by 5/20** (推前 5/22 → 5/20)

**5/17~5/20 战略 (team-memory line 195 新铁律)**:
- ✅ **cross-seed 主线开放** (5/22 lock 已 ARCHIVED)
- ✅ **path B cross-seed 优先** (M22 已 5 数据点 · path B 仅 2 + 今晚 +2 = 4 接近饱和)
- ✅ **stack accumulation** (T38 v2 + UE / DCN + UE 今晚)
- ✅ **ckpt selection 战略激活** (M68 E3 ckpt 首测 待 5/18 早)

**5/20 前到 0.852 概率**: ~20~30% (M68 stack 失败后下调 · 仍可行)

**5/17 全天 LB 数据点 (今天新)**:
- M55 LB **0.844343** (Bilinear gating · 闭环失败)
- M71 LB **0.846641** (path B × s11 cross-seed · 第 2 数据点)
- M68 LB **0.846468** (path B + Q_init stack · 净损害 −0.001694)
- → **path B cross-seed 跨度小** (s42 0.848 / s11 0.846 · 0.0015) · stack 加性失败 (Q_init 在 path B 上失败)

**今晚 4 jobs 设计逻辑** (EXP-068 落档):
- 2 cross-seed (M73 s2 / M74 s5) → path B 路径 4 数据点 · 触饱和阈值
- 1 stack (M75 T38 v2 + UE) → T38 v2 fix 真测 + path B
- 1 救活 (M50 重跑) → 防 ckpt 过期 + LB 评估
- M76 (DCN + UE) 接力 M50 重跑

**关键 deprecated 方向 (不再投资源)**:
- ❌ Q_init in path B (M68 LB 净损害 −0.001694 · stack 失败)
- ❌ Bilinear gating (M55 LB −0.004 · user-side gating 路径死)
- ❌ daypart filter (M45/M48 dev 灾难 · M50 双 stack 净损害)
- ❌ DCN cross alone (M23 LB −0.011 · path A DCN 死透 · 但 path B + DCN 是 M76 重新探索)
- ❌ 5-cluster ns_groups / idmask / NLIR alone / FaFE / per-token FFN
- ⚠️ Multi-Emb K=2 / Global Token (M38v2/M39v2 dev 弱负 · 待 LB 闭环 · 不优先 stack)

---

## 历史 Lessons (5+ 次资源混乱事故 · 5/9 ~ 5/17)

1. 5/9 22:10: agent 凭印象 "evaluation 截止 CEST 14:00" → 用户纠正
2. 5/10 08:40: num_epochs=3 凭印象 → 改 default 4
3. 5/10 16:21: agent 把 training slot 当 evaluation quota → line 175 铁律
4. 5/14 12:39: agent 又把 slot 当 quota · "等凌晨" 伪时序 → 用户反问
5. 5/16 08:35: agent 凭印象 "2 free slot" 实际 4 free → 用户反问 "彻底解决问题"
6. **5/17 16:42**: agent 凭印象 "5/16 晚 4 jobs (M60/M55/M54'/M52')" 实际 5/17 早起 4 jobs (M68/M65/M68 v2/M71) · 用户贴 log 后才修正 (4 jobs 全带 DIN bypass · 5/16 晚 EXP-065 4 jobs 实际不在今天 log 里)

**Root cause** (累计):
- 静态规则铁律已立 (4/day quota / 4 slots)
- **动态状态 (当前 free slot 数 / 剩余 quota / 哪些 jobs 在跑)** 由本文件单一 source of truth 维护
- agent 任何时候陈述资源状态前**必须 grep 本文件** · 不凭印象

---

## Update Log

- **2026-05-17 CEST 18:35**: 用户启动 M73/M74/M75 + M50 重跑 (M76 接力 M50 重跑) · M50 ckpt 5/17 15:53 已过期 · 重训挽救 LB 评估机会 · 4 slots 全占 · today quota used 3/4 (M55+M71+M68 LB) · 今晚剩 1 quota = M50 LB · 5/20 ddl 战略下 cross-seed heavy + stack accumulation + ckpt selection 三轨并进
- **2026-05-17 CEST 16:14**: 用户立新硬 deadline = 0.852 by 5/20 (推前 5/22 → 5/20) · 5/22 cross-seed lock ARCHIVED · 立新铁律 team-memory line 195 (cross-seed 主线开放) · commit `541ef12`
- **2026-05-17 CEST 14:15~15:38**: 用户启 M68 / M65 / M68 v2 / M71 4 jobs 5/17 早 (overnight train) · 全部 19:00~21:30 完成 dev
- **2026-05-17 CEST 15:30~16:30**: M55/M71/M68 LB 回流 · M55 LB 0.844343 (Bilinear 死) / M71 LB 0.846641 (path B s11) / M68 LB 0.846468 (path B + Q_init stack 净损害)
- **2026-05-17 CEST 13:50~14:30**: EDA-67 v2 evaluation submitted (G2/A2/F3 train-side) · 22:26 完成 · 18.9MB blob 写到平台 storage · 用户决今晚不下载 · 5/18 后再分析
- **2026-05-17 CEST 13:44**: M68 第 1 次 evaluation state_dict mismatch 失败 (q_init_item key bug) · commit `cfd15f7` 修复 + 用户重提成功 · **失败 evaluation 不消耗 quota 铁律告知** (team-memory line 109)
- **2026-05-16 CEST 21:15**: 🎉 **M49 LB 0.848162** = 项目首个超 SOTA · UE split (T37 ADR-014) 弱真信号 · Δ+0.000570 vs M22
- **2026-05-16 CEST 16:58**: M51-M54 4 jobs 启动 (T38 NameError fix commit `2638e1b` 后) · 全部 in-flight 预计 ~CEST 22:00 完成 dev
- **2026-05-16 CEST 16:25**: publish 状态全 audit · 5/17 早 quota matrix 修订
- **2026-05-16 CEST 14:43**: M23 retry LB 0.836631 + M45/M48/M49 dev 回流 · M49 dev 0.870741 (项目首个超 M22 真信号)
- **2026-05-16 CEST 08:43**: 文件创建 (用户反问 "彻底解决问题" 触发) · single source of truth
