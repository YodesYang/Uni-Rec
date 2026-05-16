# Platform 实时状态 (Single Source of Truth)

> **每次 agent 推荐 platform 操作前必须 grep 此文件**。任何资源状态陈述 (slot / quota / in-flight) 凭此文件 · 不凭印象。
>
> **Last updated**: 2026-05-16 CEST 16:58 (M51-M54 4 jobs 启动成功 · T38 NameError 已修复 · in-flight overnight)

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

| Slot | Status | Job | 起 (CEST) | 完成预估 (CEST) | dev (E4 best) |
|---|---|---|---|---|---|
| 1 | 🟡 **in-flight** | **M51** (M22 + UE split + temporal_weight 2.5 · path B + M41 stack) | ~16:55 5/16 (T38 fix 后重启) | ~22:00 5/16 | TBD |
| 2 | 🟡 **in-flight** | **M52** (M22 + UE split + per-domain time bucket · 修复 M28) | ~16:55 5/16 | ~22:00 5/16 | TBD |
| 3 | 🟡 **in-flight** | **M53** (M22 + UE split + NLIR · path B + Linear gating stack) | ~16:55 5/16 | ~22:00 5/16 | TBD |
| 4 | 🟡 **in-flight** | **M54** (M22 + per-domain bucket alone · M28 修复单变量) | ~16:55 5/16 | ~22:00 5/16 | TBD |

**Last completed jobs** (按时间倒序):
- **M50** (UE split + weekend filter 双 stack): completed CEST ~15:53 5/16 · dev 0.861974 · 双 stack 损害 net (−0.0066 vs M22) · LB 待评 (5/17 早 ⭐⭐ 候选)
- **M49** ⭐⭐⭐ (UE split T37 ADR-014): completed CEST ~14:00 5/16 · dev **0.870741** / LB **0.848162** ⭐ NEW SOTA (+0.000570 vs M22)
- **M45** (weekend filter): completed CEST ~14:15 5/16 · dev 0.858907 · daypart filter 路径死 (−0.0097)
- **M48** (03-04 only): completed CEST ~12:30 5/16 · dev 0.802487 · data 不足灾难 (−0.066)
- **M41** (temporal_weight 2.5): completed CEST ~04:30 5/16 · dev 0.868069 / LB 0.847846 ⭐
- **M40** (hour_only): completed CEST ~04:30 5/16 · dev 0.866881 / LB 0.846931
- **M39 v2** (Global Token): completed CEST ~01:30 5/16 · dev 0.867963 · LB 待评 ✅ publish-ready
- **M38 v2** (Multi-Emb K=2): completed CEST ~00:30 5/16 · dev 0.868403 · LB 待评 ✅ publish-ready

---

## Evaluation Quota (4/day · 北京 12:00 = CEST 06:00 重置 · 不累积)

> 来源: team-memory line 9

| Day | Used | Remaining | Used by |
|---|---|---|---|
| **5/16 (today)** | **4** | **0** | M40 LB 0.846931 + M41 LB 0.847846 + M23 retry LB 0.836631 + **M49 LB 0.848162 ⭐ 新 SOTA** |
| 5/15 (closed) | 4 | 0 | M28 LB 0.846514 + M29 LB 0.845312 + M22-rerun LB 0.847324 + M33 LB 0.844710 |
| 5/14 (closed) | 1 | (overflow to 5/15 ❌) | M22 LB 0.847592 (前 SOTA · 现 #2) |

**Next reset**: CEST 06:00 5/17 = 北京 12:00 5/17

**M49 evaluation 状态 (CEST 14:50)**: 用户提交 · 等 LB 结果 (~CEST 15:30 平台 inference + ~CEST 19:00 leaderboard 上榜延迟)

**5/17 早 4 quota 预案 (CEST 16:25 修订 · 2 老 + 2 新策略)**:
- Slot 1 ⭐⭐⭐ **M37** (NLIR vocabtier) · 类 M49 architecture-level · EXP-058 §Error E 反推可能误判
- Slot 2 ⭐⭐⭐ **M38 v2** (Multi-Emb K=2) · RankUp 闭环
- Slot 3 ⭐⭐⭐ **M51 / M52 / M53 dev 最高的** · 今晚训完 dev 出 · 选最强
- Slot 4 ⭐⭐ **M50** (UE split + weekend filter 双 stack 反例验证) · 或备用 M30

**5/17 早 决断红线**:
- 不破 cross-seed 铁律 (M49 LB 0.848162 < 0.8484 严格阈值 · 不触发例外)
- 不评 M45/M48 (dev 灾难)
- 不评 M39 v2 (Global Token 与 UE split slot 冲突 · 大概率持平)
- M34 ckpt **已永久丢失** (5/14 训 · 24h 过期 · 用户 5/16 16:25 确认)

**今晚 4 slot overnight 训练计划**:
- M51 = M22 + UE split + temporal_weight 2.5 (path B + M41 stack)
- M52 = M22 + UE split + per-domain time bucket (path B + EXP-058 §Error C 修复)
- M53 = M22 + UE split + NLIR (path B + Linear gating stack)
- M54 = M22 + per-domain time bucket alone (M28 修复单变量 · 不依赖 path B)
- 起 ~CEST 20:30 5/16 · 完成 ~CEST 01:30 5/17 · 5/17 早 quota refresh 时 dev 已出

---

## Publish-Ready Ckpt 待评 LB (按 dev 排序 · 5/16 16:25 publish 状态全 audit)

> 来源: 5/16 CEST 16:25 用户确认 publish 状态

| Job | Dev | Publish | 24h 过期 (CEST) | 期望 LB | 5/17 早评 Priority |
|---|---|---|---|---|---|
| **M30** (Trick 6) | 0.868438 | ✅ | safe | ~0.847 | ⭐⭐⭐ |
| **M38 v2** (Multi-Emb K=2) | 0.868403 | ✅ (5/16 早 publish) | safe | ~0.847 | ⭐⭐⭐ |
| **M37** (NLIR) | 0.868311 | ✅ | safe | ~0.847 | ⭐⭐⭐ |
| **M27 v2** (T31+user_15) | 0.868166 | ✅ | safe | ~0.846 | ⭐⭐ |
| **M39 v2** (Global Token) | 0.867963 | ✅ (5/16 早 publish) | safe | ~0.844 | ⭐ |
| **M17** (seq_d=768) | 0.867794 | ✅ | safe | ~0.845 | ⭐ |
| **M50** (UE split + weekend) | 0.861974 | publish 待 verify | 5/17 15:53 | 0.840~0.848 | ⭐⭐ 反例验证 |
| ~~M34 (vocabtier+Trick 6)~~ | ~~0.868422~~ | ❌ **已丢失 (24h 过期)** | (gone) | — | ❌ deprecated |
| ~~M45/M48~~ | < 0.86 | dev 灾难 | — | — | ❌ 不评 |

---

## Strategic Status (5/16 CEST 08:43)

**LB SOTA**: **M49 (UE split T37 ADR-014) = 0.848162** ⭐ NEW (5/16)
**前 SOTA**: M22 (vocabtier × s42) = 0.847592 (5/14)
**M49 vs M22 Δ**: +0.000570 (超 same-seed rerun band ±0.0003 · 未破 0.8484 严格阈值)
**LB rerun band**: ±0.0003 (M22 / M22-rerun)
**LB cross-seed band**: ±0.005 (EXP-028)
**破 SOTA 阈值 (same-seed rerun)**: ≥ 0.8484
**距 0.85 (Top 20)**: 还差 +0.0018

**🔴 5/22 前 seed 策略 (用户 5/16 CEST 08:48 战略决断)**:
- **默认锁 `seed=42`** (实证 LB 上沿 · EXP-028 + M22 路径双重确认)
- **不做 cross-seed 实验** (M22 × s11/s2 / 任何方向 × s3/s7 都不起)
- **唯一例外**: 强真信号 (LB ≥ 0.8484) 出现时才做 cross-seed (但 9 天内几乎不可能出现)
- **5/22~5/24 final 阶段** 才做 final config × s11/s2 上沿搜索

**今日新发现**:
- ⭐ M41 LB +0.000254 = 项目首个 dev↓ LB↑ 反例 (但弱信号 · band 内 · 不 cross-seed)
- ⚠️ M40 LB −0.000661 = dow 半证伪 (移除 dow 在 LB 上不全错也不全对)
- 7/7 新方向 (M37/M38/M39/M40/M41/Multi-Emb/Global Token) dev 都不超 M22 · vocabtier 路径**接近局部最优**
- **path B (UE 分离 + int pairwise cross)** 是当前唯一未尝试的破局方向 · ADR-014 待起草

**5/16 下午 14:00 计划 (CEST 09:30 audit · path B 暂缓)** (4 free slot):
- **4 slot 全留空** (path B / M44 都 suspended · 见 ADR-014 §Suspension Note)
- 等晚 2 evaluation (M23 + M38 v2) LB 出 (~CEST 22:30)
- 5/17 早根据数据决方向

**Path B 暂缓原因** (用户 5/16 09:30 决断):
- "UE 分离" 解读不确定 · 3 种 plausible interpretation · community 无二次澄清
- 7/7 历史新方向 dev 弱负 · 在解读不确定时投入资源高风险
- 重启条件: M23 LB 出后看 cross 路径是否 work · 或社区追问 Seed 帖细节

**M44 撤销原因**:
- M22 + temporal_weight 2.5 + Multi-Emb K=2 stack
- M41 弱信号 (LB +0.000254) + Multi-Emb 弱负 (M38 v2 dev −0.000173)
- Stack 期望 ~+0.0001 · 落 noise band 内 · ROI 极低

**关键 deprecated 方向** (不再投资源):
- ❌ 5-cluster ns_groups (M33 LB 反向放大 27× · 路径关闭)
- ❌ idmask (EXP-025 LB 反向 14×)
- ❌ NLIR / FaFE / per-token FFN (5/15 ~ 5/9 各种证伪)
- ⚠️ Multi-Emb K=2 / Global Token (M38v2/M39v2 dev 弱负 · 待 LB 闭环 · 不优先 stack)

---

## 历史 Lessons (5 次资源混乱事故 · 5/9 ~ 5/16)

1. **5/9 22:10**: agent 凭印象 "evaluation 截止 CEST 14:00" (实际 quota 重置 = 北京 12:00) → 用户反问纠正
2. **5/10 08:40**: num_epochs=3 凭印象 (EXP-019 明文 E4 best) → 改 default 4
3. **5/10 16:21**: agent 把 training slot 当 evaluation quota (4/day) → 用户主动澄清 · 立 line 175 铁律
4. **5/14 12:39**: agent 又把 training slot 当 quota · 提"等凌晨"伪时序 → 用户反问 "为什么要等凌晨" · line 175 铁律重申
5. **5/16 08:35**: agent 凭印象说 "5/16 下午 2 free slot" · 实际 4 free → 用户反问 "彻底解决问题"

**Root cause** (用户 5/16 提问触发):
- 静态规则 (4/day quota / 4 slots) 已立铁律
- **动态状态 (当前 free slot 数 / 剩余 quota) 没有 single source of truth**
- agent 凭对话历史拼凑 mental model · stateless 重启时容易污染

**Fix (5/16 实施)**:
- 创建本文件作 single source of truth
- 删除 active.md "Work in flight" 段 (stale info 源)
- team-memory 顶部加 "agent 推荐前 4 步硬流程"

---

## Update Log

- **2026-05-16 CEST 16:58**: M51-M54 4 jobs 启动成功 (T38 NameError fix commit `2638e1b` 后) · 全部 in-flight 预计 ~CEST 22:00 完成 dev · 5/17 早 quota refresh 时数据齐全 · 4 jobs Args verify 全通过 (M51 path B+M41 / M52 path B+per-domain / M53 path B+NLIR / M54 per-domain alone)
- **2026-05-16 CEST 16:25**: publish 状态全 audit (用户确认 M37/M38v2/M39v2/M30/M27v2/M17/M40/M41 都已 publish · M34 已丢失 · M45/M48/M50 publish 状态待 verify) · 5/17 早 quota matrix 修订为 "2 老 (M37 + M38v2) + 2 新 (M51-M54 dev 最高 + M50/M30)" · 4 free slot 启 M51-M54 stack overnight train (CEST ~20:30 起 ~01:30 5/17 完成) · 见 EXP-061
- **2026-05-16 CEST 16:00**: M50 dev 0.861974 (双 stack 损害 net −0.0066) · 用户反问 "train/valid 是否一致" 触发 src/dataset.py source-of-truth audit · 确认 M22/M49/M50 valid set 完全相同 · dev 严格可对比 · M50 LB 加入 5/17 早 quota 候选 (优先级 ⭐⭐) · 5/17 矩阵终版定稿 · 见 EXP-060
- **2026-05-16 CEST 21:15**: 🎉 **M49 LB 0.848162** = 项目首个超 SOTA · UE split (T37 ADR-014) 弱真信号 · Δ+0.000570 vs M22 · 超 rerun band 但未破 0.8484 严格阈值 · dev→LB 衰减 73% · 5/17 早 4 quota 优先评 M50 / M37 / M38v2 · 不破 cross-seed 铁律 · 见 EXP-059
- **2026-05-16 CEST 14:50**: M49 evaluation 已提交 · today quota 用满 4/4 · 等 LB 结果 (~CEST 19:00 上榜) · 期望区间 0.847~0.850 (path B 首测 · 决 5/17 4 quota 方向)
- **2026-05-16 CEST 14:43**: M23 retry LB 0.836631 (−0.011 强负 · path A DCN cross 死透闭环) + M45/M48/M49 dev 回流 · M49 dev 0.870741 (+0.002122 项目首个超 M22 真信号 · UE split ADR-014 实证) · 5/16 today quota 修订为 used=3 / remaining=1 · 晚 1 quota 必评 M49 LB
- **2026-05-16 CEST 08:43**: 文件创建 (用户反问 "彻底解决问题" 触发) · 状态镜像 5/16 早 4 jobs 完成 + 早 2 evaluation 完成
