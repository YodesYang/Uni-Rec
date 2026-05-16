# Platform 实时状态 (Single Source of Truth)

> **每次 agent 推荐 platform 操作前必须 grep 此文件**。任何资源状态陈述 (slot / quota / in-flight) 凭此文件 · 不凭印象。
>
> **Last updated**: 2026-05-16 CEST 08:43 (用户反问 "为什么 slot 混乱" 触发 root cause fix)

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
| 1 | **free** | — | — | — | — |
| 2 | **free** | — | — | — | — |
| 3 | **free** | — | — | — | — |
| 4 | **free** | — | — | — | — |

**Last completed jobs** (按时间倒序):
- **M41** (temporal_weight 2.5): completed CEST ~04:30 5/16 · dev 0.868069 / LB 0.847846 ⭐
- **M40** (hour_only): completed CEST ~04:30 5/16 · dev 0.866881 / LB 0.846931
- **M39 v2** (Global Token): completed CEST ~01:30 5/16 · dev 0.867963 · LB 待评
- **M38 v2** (Multi-Emb K=2): completed CEST ~00:30 5/16 · dev 0.868403 · LB 待评

---

## Evaluation Quota (4/day · 北京 12:00 = CEST 06:00 重置 · 不累积)

> 来源: team-memory line 9

| Day | Used | Remaining | Used by |
|---|---|---|---|
| **5/16 (today)** | **2** | **2** | M40 LB 0.846931 (CEST ~09:30) + M41 LB 0.847846 (CEST ~09:30) |
| 5/15 (closed) | 4 | 0 | M28 LB 0.846514 + M29 LB 0.845312 + M22-rerun LB 0.847324 + M33 LB 0.844710 |
| 5/14 (closed) | 1 | (overflow to 5/15 ❌) | M22 LB 0.847592 ⭐ SOTA |

**Next reset**: CEST 06:00 5/17 = 北京 12:00 5/17

**Today (5/16) remaining 2 quota usage plan**:
- 晚 2 (CEST ~21:00) **修订 (CEST 09:00 5/16 audit)**:
  - ⭐⭐⭐ **M23** (T34 DCN gap=0 · dev 0.868311 · LB 待评) · 决 cross 路径 work 不 work · 期望 LB +0.001~+0.003
  - ⭐⭐ **M38 v2** (Multi-Emb K=2) · RankUp 闭环 · 期望 LB ~0.847

**5/16 10:00 incident**: M23 evaluation 第 1 次失败 (`'<=' int vs str` · src/infer.py 没解析 dcn_cross_user_fids raw csv str) · quota refund · 修复 commit `0a16a98` (`_resolve_dcn_fids` helper) · M23 retry 待用户重新提交。

---

## Publish-Ready Ckpt 待评 LB (按 dev 排序)

> 来源: 5/15 5-16-decision-matrix.md + 5/16 早 4 jobs

| Job | Dev | Publish 状态 | 期望 LB | Priority |
|---|---|---|---|---|
| **M38 v2** (Multi-Emb K=2) | 0.868403 | 已 publish 待评 | ~0.846~0.847 | ⭐⭐⭐ 晚 2 候选 |
| **M39 v2** (Global Token) | 0.867963 | 已 publish 待评 | ~0.844~0.847 | ⭐⭐⭐ 晚 2 候选 |
| M30 (Trick 6) | 0.868438 | 已 publish 待评 | ~0.846 | ⭐⭐ 备选 |
| M34 (vocabtier+Trick 6) | 0.868422 | 已 publish 待评 | ~0.846 | ⭐ 低优 (stack 第 2 步 5/14~5/15 失败 3 次) |
| M37 (vocabtier+NLIR) | 0.868311 | 已 publish 待评 | ~0.846 | ⭐ 低优 |
| M27-v2 (T31+user_15) | 0.868166 | 已 publish 待评 | ~0.846 | ⭐ 低优 |
| M23 (T34 DCN gap=0) | 0.868311 | 已 publish 待评 | ~0.846 | ⭐ 低优 |
| M17 (seq_d=768) | 0.867794 | 已 publish 待评 | ~0.845 | ⭐ 低优 |

---

## Strategic Status (5/16 CEST 08:43)

**LB SOTA**: M22 (vocabtier × s42) = **0.847592** (5/14)
**LB rerun band**: ±0.0003 (M22 / M22-rerun)
**LB cross-seed band**: ±0.005 (EXP-028)
**破 SOTA 阈值 (same-seed rerun)**: ≥ 0.8484
**距 0.85 (Top 20)**: 还差 +0.0024

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

- **2026-05-16 CEST 08:43**: 文件创建 (用户反问 "彻底解决问题" 触发) · 状态镜像 5/16 早 4 jobs 完成 + 早 2 evaluation 完成
