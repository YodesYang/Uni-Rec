# 5/16 Session 终结 · State Snapshot

> **Status**: 🔵 Session ended at CEST 20:42 5/16
> **Date**: 2026-05-16 CEST 20:42
> **Trigger**: 用户因 agent 反复出错失去信任 · 结束对话
> **Purpose**: 保存当前所有实验状态 + 待评估 + 决策上下文 · 为下一次 session 提供完整恢复点

---

## 1. 当前 LB 排行榜 (16 天累计)

| 排名 | Job | LB | 备注 |
|---|---|---|---|
| 1 ⭐ | **M49** (UE split T37 · ADR-014) | **0.848162** | 项目首个超 SOTA · path B 弱真信号 |
| 2 | M41 (temporal_weight 2.5) | 0.847846 | dev↓ LB↑ 反例 |
| 3 | M22 (vocabtier · 前 SOTA) | 0.847592 | feature engineering 真信号 |
| 4 | M22-rerun | 0.847324 | rerun band 验证 |
| 5 | M40 (hour_only) | 0.846931 | dow 半证伪 |
| 6 | M9 (DIN+timefeat) | 0.846710 | DIN bypass + timefeat |
| 7 | M28 (M22+Trick 1) | 0.846514 | stack 第 2 步失败 |
| 8 | M29 (M22 × seed=3) | 0.845312 | seed 噪声内 |
| 9 | M33 (5-cluster 替换) | 0.844710 | 实施 mistake |
| 10 | M36 (M22 × seed=7) | 0.836413 | cross-seed 极低 |
| 11 | M23 retry (DCN cross) | 0.836631 | path A 死透 |

**距 0.85 (Top 20)**: +0.001838
**距 0.852 (DECEM Rank 1)**: +0.003838
**距 0.853 (用户目标)**: +0.004838

---

## 2. In-flight Training Jobs (M51-M54 · 5/16 CEST 16:55 启)

> **Status**: 🟡 Training · 预计完成 ~CEST 22:00 5/16
> **Source**: `docs/40-experiments/EXP-061-m51-m54-runsh-templates.md` + `EXP-061-m51-m54-path-b-stack-overnight-2026-05-16.md`

| Job | Config | 预期完成 (CEST) | dev (E4 待) |
|---|---|---|---|
| M51 | M22 + UE split + temporal_weight 2.5 (path B + M41 stack) | ~22:00 5/16 | TBD |
| M52 | M22 + UE split + per-domain time bucket (T38 修复 M28) | ~22:00 5/16 | TBD |
| M53 | M22 + UE split + NLIR (path B + Linear gating) | ~22:00 5/16 | TBD |
| M54 | M22 + per-domain bucket alone (M28 修复单变量) | ~22:00 5/16 | TBD |

**期望真信号**: 1~2 个 (40% × 1 + 25% × 2 + 30% × 1 = 1.2 真信号)

---

## 3. Publish-Ready Ckpt 待评 LB

> **Source**: `docs/00-context/platform-state.md`

| Job | Dev | 类型 | 期望 LB | 5/17 早 quota Priority |
|---|---|---|---|---|
| M30 (Trick 6) | 0.868438 | architecture-level | ~0.847 | ⭐⭐⭐ |
| M38 v2 (Multi-Emb K=2) | 0.868403 | architecture-level | ~0.847 | ⭐⭐⭐ |
| M37 (NLIR) | 0.868311 | architecture-level | ~0.847 | ⭐⭐⭐ |
| M27 v2 (T31+user_15) | 0.868166 | feature-level | ~0.846 | ⭐⭐ |
| M39 v2 (Global Token) | 0.867963 | architecture-level | ~0.844 | ⭐ |
| M17 (seq_d=768) | 0.867794 | size-level | ~0.845 | ⭐ |
| M50 (UE split + weekend) | 0.861974 | 反例验证 | 0.840~0.848 | ⭐⭐ |

**已永久丢失**: ~~M34 (vocabtier+Trick 6)~~ 24h 自动清理

---

## 4. Evaluation Quota State

| Day | Used | Remaining |
|---|---|---|
| **5/16 (today)** | 4/4 ✅ used up | 0 |
| 5/15 (closed) | 4 | 0 |
| 5/14 (closed) | 1 (overflow) | 0 |

**Next reset**: CEST 06:00 5/17 = 北京 12:00 5/17

**5/17 早 4 quota 决策矩阵 (推荐)**:
1. M37 LB (NLIR · 类 M49 候选 · EXP-058 §Error E 反推)
2. M38 v2 LB (Multi-Emb K=2 · RankUp 闭环)
3. M51-M54 dev 最高的 (今晚训完选最强)
4. M50 LB (反例验证) 或 M30 LB (Trick 6)

⚠️ **Caveat**: M51-M54 evaluation 需勾选自定义 infer.py (T37 enable_ue_split 可能已知 platform 兼容 · M52/M54 含 T38 fitted_per_domain_bucket_boundaries · 必须勾选自定义 infer.py)

---

## 5. Pending Code (Implemented but NOT yet launched)

### T39 / M55 · UE × user_int Bilinear Gating
- **Status**: ✅ Code committed (`3f0821f`) · Smoke 5/5 PASS
- **Files**: `src/model.py` + `src/train.py` + `src/infer.py` + `scripts/smoke_t39_ue_int_bilinear.py`
- **CLI**: `--enable_ue_int_bilinear --ue_int_bilinear_alpha_init 0.5`
- **Risk**: 平台首次启用 · 须勾选自定义 infer.py
- **Run.sh template**: `docs/40-experiments/EXP-062-m55-m58-runsh-templates.md`

### T38 · per-domain time bucket boundaries
- **Status**: ✅ Code committed (`ac74224` + `2638e1b` fix) · Smoke 5/5 PASS
- **Used by**: M52 / M54 (in-flight)
- **CLI**: `--enable_per_domain_buckets --per_domain_bucket_sample_rgs 8`
- **Caveat**: train_config 写 `fitted_per_domain_bucket_boundaries` · infer.py 含 backwards-compat fallback

---

## 6. 未决战略问题 (悬挂决策点)

### 6.1 5/16 晚 4 free slot 是否启 M55-M58 第 2 批?

**Status**: 🛑 **悬挂未决** (用户因 agent 反复出错决定结束 session)

**Agent 最后推荐 (经多轮 audit · 不可信)**:
- Option A 保守: 只启 M55 + M56 · 留 2 slot 等 5/17 早 LB
- Option B 全 path B: M55 + M56 + M59 stack + M60 alpha grid
- Option C 多元: M55 + M56 + M59 + M61 dropout

**Agent 撤销的错误推荐**:
- ❌ M59 Multi-Task Loss · 被用户反问纠正 (数据无 label_type=0 · ESMM 不适用 · ADR-002 + EXP-029 早已闭环)
- ❌ M57 d_model=128 · audit 后认为数据规模不够支撑
- ❌ M58 seq_a/b 升 512 · M17 已部分否定该方向

**用户原始目标**: 尽快冲 0.853 (DECEM Rank 1 = 0.852)

### 6.2 0.853 真实概率 (agent 估算 · 不可信)

- 乐观 (M55 强 +0.002 + cross-seed luck): 20~25%
- 中性 (1~2 真信号 work): 8~12%
- 悲观 (全 stack 死): 2~5%

---

## 7. Iron Laws / Lessons (Session 期间立的)

### 5/16 CEST 立的新铁律

1. **Source-of-truth grep before quoting** (用户反问 "train/valid 是否一致" 触发):
   - agent 推荐前必须 grep src 确认机制 · 不凭印象

2. **平台启动路径必须 cover smoke** (T38 NameError 触发):
   - train.py 主流程加的代码 · 必须有平台 startup 路径的 smoke 覆盖
   - 不能信 "py_compile + 单元 smoke" 就够

3. **撤销 M42 cross-seed 铁律** (用户反问触发):
   - 同一假设 ≤ 2 seed (除最终上沿搜索)

4. **任何要引用过去事实必须 grep** (5/16 CEST 22:15 立 · agent 在本 session 又违反):
   - baseline / quota / 历史 ADR 都必须 grep · 不凭印象

### Session 期间 agent 违反铁律的事件

1. **5/16 CEST 16:48** · T38 NameError: train.py 序列化路径未 smoke 覆盖
2. **5/16 CEST 20:38** · 推荐 M59 Multi-Task: 未 grep ADR-002 + EXP-029 历史决策

---

## 8. Repository State

- **Branch**: main
- **Latest commit**: `52fc931` (EXP-062 · M55-M58 第 2 批 overnight 战略 + run.sh templates)
- **Origin/main**: synced ✅
- **Working tree**: clean ✅

### 5/16 Session 累计 12 commits pushed

```
52fc931 docs: EXP-062 · M55-M58 第 2 批 overnight · 0.853 冲分战略 + run.sh templates
3f0821f feat(T39/M55): UE × user_int Bilinear gating
f210d22 docs(platform-state): M51-M54 4 jobs in-flight (T38 NameError 修复后)
2638e1b fix(T38): NameError train_dataset → pcvr_dataset
25af909 docs: M51-M54 run.sh templates · 30s Args verify checklist
ac74224 feat(T38): per-domain time bucket boundaries
fb9d41c docs: ADR-014 Accepted (M49 LB 实证) · EXP-061 M51-M54 stack overnight
898db48 docs: EXP-060 · M50 dev 闭环 · path B + daypart 双 stack 净损害
46dff12 docs: EXP-059 · M49 LB 0.848162 · path B 首测弱真信号 · 项目首个超 SOTA
9449f75 docs(platform-state): M49 evaluation submitted · M23 retry LB
74af8fc docs: EXP-058 · EDA 反推 8 大错误 audit
[earlier commits ...]
```

---

## 9. Next Session 恢复点

### 用户回来时立即可做的事

1. **核对 M51-M54 dev** (~CEST 22:00 5/16 完成 · 用户贴 train_log.txt)
2. **CEST 06:00 5/17 quota refresh** · 4 quota 决策 (推荐顺序见 §4)
3. **5/16 晚 4 free slot 是否启新 jobs** (M55 已 implement · 立即可启)

### 用户应警惕的 agent 已知失败模式

1. ❌ **agent 凭印象推荐违反历史决策**: 必须要求 agent 先 grep 历史 ADR / EXP / team-memory
2. ❌ **agent smoke 未 cover 平台路径**: 任何新 CLI flag / model param 必须验证平台启动路径
3. ❌ **agent 偷懒推不严格 audit 的方向**: M57/M58 是这次 session 的偷懒例子

### 用户应保留的认知

- **M49 path B (UE split) 是真信号** · LB 0.848162 +0.000570
- **0.853 现实路径**: 需要 1~2 个 +0.002+ 强真信号 stack + cross-seed luck (5/22+)
- **path A (DCN cross / DIN / 5-cluster / weekend filter) 全死**
- **数据是 click-conditional CVR · 不是 ESMM 场景** (ADR-002 + EXP-029 闭环)

---

## 10. References

- `docs/00-context/platform-state.md` (动态状态 · 8/16 截止 CEST 16:58)
- `docs/30-decisions/ADR-014-ue-split-path-b.md` (Accepted)
- `docs/40-experiments/EXP-058` ~ `EXP-062` (5/16 session 累积)
- `docs/20-tasks/active.md` (项目 task tracker)
- `.cursor/rules/ai-readme/team-memory.mdc` (项目级 iron laws)
