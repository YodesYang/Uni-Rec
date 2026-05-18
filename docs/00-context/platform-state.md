# Platform 实时状态 (Single Source of Truth)

> **每次 agent 推荐 platform 操作前必须 grep 此文件**。任何资源状态陈述 (slot / quota / in-flight) 凭此文件 · 不凭印象。
>
> **Last updated**: 2026-05-18 platform update (M93-M96 queued · platform auto-schedules queued jobs)

> **Hotfix Update · M75 Result**: M75 evaluation completed with **public AUC 0.848788** (731.79s), replacing M49 0.848162 as LB SOTA. Attribution caveat: M75 should **not** be counted as true T38/per-domain bucket success because its training log showed per-domain fit `n_samples=0`; treat it as a stronger path B / UE split family realization. Full record: [EXP-071](../40-experiments/EXP-071-m75-lb-new-sota-attribution-2026-05-18.md).

> **Morning Batch Update**: 用户回报 **M76 / M82 / M83 / M84 / M75(E6)** 均训练完成. Best dev: M75(E6)=0.870651, M84=0.870490, M76=0.870247, M82=0.869709, M83=0.867486. Full record: [EXP-072](../40-experiments/EXP-072-may18-morning-batch-results.md).

> **M84/M82 Eval Update**: M84 public **0.848854** (695.58s) is a tiny new SOTA over M75 by +0.000066 and validates true T38 as small public-positive. M82 public **0.846197** (684.43s) rejects the alpha+hour daypart reweight path.

> **XHS Intel Update**: external post says baseline + rewritten **time** features can reach ~0.849 while TokenFormer is not clearly superior. We added opt-in `--enable_beijing_time_features` and prepared M85-M88. Full record: [EXP-073](../40-experiments/EXP-073-xhs-time-intel-m85-m88-plan-2026-05-18.md).

> **M85/M86 Train Update**: M85 best dev **0.870887** and M86 best dev **0.871415**. Both used corrected T38 nonzero per-domain samples. M86 was the strongest dev candidate, but its later public result rejected naive UTC+Beijing time stacking.

> **M89/M90 Training Update**: 用户回报 M89 and M90 both started. Treat training pool as **2 in-flight / 2 free** until completion logs arrive.

> **M91/M92 Training Update**: 用户回报 M91 and M92 both submitted. Platform has a queue mechanism; submitted jobs may wait and then auto-start when a slot frees. Treat M91/M92 as **submitted / platform-scheduled**, and wait for completion logs to know actual run status. They are clean-window diagnostics, not direct LB submissions: use M91 vs M92 pseudo-public dev delta to judge whether time features transfer.

> **Breakthrough Protocol Update**: 用户回报 M93-M96 have been submitted to the platform training queue. They will auto-start as earlier queued/running jobs finish. M93 = clean-window delayed-feedback diagnostic, M94 = full delayed-feedback candidate, M95 = seq-stats candidate, M96 = delayed-feedback + seq-stats interaction. Full gates are in [EXP-075](../40-experiments/EXP-075-breakthrough-protocol-m93-m96-2026-05-18.md).

> **M87 Update**: M87 (`M84 + temporal_weight_alpha=2.5`) completed with best dev **0.869053**. It used corrected T38 nonzero samples, but is **-0.001438 vs M84** and far below M85/M86. Do **not** evaluate unless quota becomes abundant.

> **M88 Update**: M88 (`M84 6ep`) completed with best dev **0.870910** at E6. Longer training helps corrected T38 by **+0.000420 vs M84**, but remains below M86 by **-0.000505**. It is a score candidate, not a new mechanism.

> **M86 Eval Update**: M86 public **0.847106** (472.2s) is **-0.001748 vs M84** despite best dev 0.871415. Eval log confirms the intended `time_of_day_features=True` + `beijing_time_features=True` config and correct checkpoint, so treat this as a dev-public mismatch / time-stack overfit signal, not an infer config failure.

> **M86 Eval EDA Update**: Decoded the reversed Logs-UI base64/gzip blob from `eval_log.txt`. Test set has **523,244 rows**, public window **BJ 2026-03-04 09:49:51~14:49:07**, median **12:57:28**, and user IDs are singleton-like in the 200k cap. This supports EDA-71's time-window backtest focus.

> **EDA-71 Update**: `EDA71-time-window-backtest-audit` completed. Train ends at **BJ 2026-03-04 09:49:50** and public starts **1 second later** at **09:49:51**; exact public-window train rows = **0**. Current RG time split has **341,086s (~3.95d)** train/valid overlap, so normal dev is contaminated for time-feature decisions. Next time branch should use clean fixed-window backtests, not more absolute-time dense features by dev score.

> **Staleness Warning**: Training completion status and M84/M82/M86 eval results are synced. Publish-ready status for remaining candidates still needs platform confirmation before submit.

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

## Training Slots / Queue (4 workers · platform queue auto-schedules)

> 来源: team-memory line 175 ("训练 slot 与 evaluation quota 是两个完全独立的资源池")
>
> **Queue correction (2026-05-18)**: 平台训练池有队列机制。提交到队列的 job 不一定已经占用 worker；当前面 job 训练完成释放 slot 时，平台会自动从队列取下一个 job 跑。因此后续资源判断要区分 **running/in-flight** 和 **queued/submitted**；不要把 queued job 误当成已完成或手动 free slot。

### Current Running / Platform-Scheduled (latest sync · 2026-05-18)

| Worker/Order | Status | Job | Notes |
|---|---|---|---|
| running/earliest | 🟡 started | **M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep** | started by user 2026-05-18 14:13 CEST sync; Beijing time v1+v2 |
| running/earliest | 🟡 started | **M90-m86-utc-bjtime-6ep-gap0-seed42** | started by user 2026-05-18 14:13 CEST sync; M86 6ep |
| queued/scheduled | 🟠 submitted | **M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep** | clean-window M84 diagnostic; auto-starts when slot frees if not already running |
| queued/scheduled | 🟠 submitted | **M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep** | clean-window M86 diagnostic; auto-starts when slot frees if not already running |
| queued/scheduled | 🟠 submitted | **M93-cleanwin0303-m84-mnw6h-w03-gap0-seed42-4ep** | delayed-feedback clean-window diagnostic |
| queued/scheduled | 🟠 submitted | **M94-m84-mnw6h-w03-gap0-seed42-4ep** | full-data delayed-feedback candidate |
| queued/scheduled | 🟠 submitted | **M95-m84-seqstats-gap0-seed42-4ep** | sequence stats candidate |
| queued/scheduled | 🟠 submitted | **M96-m84-mnw6h-seqstats-gap0-seed42-4ep** | delayed-feedback + seq-stats interaction |

### Ready To Launch (M85-M88 · XHS time-feature response)

| Priority | Job | Run script | Purpose |
|---|---|---|---|
| 1 | `M85-m84-t38fix-bjtime-gap0-seed42-4ep` | `src/run_m85_t38_fix_bjtime.sh` | M84 stack; replace UTC time with Beijing-rich time |
| 2 | `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep` | `src/run_m86_t38_fix_utc_bjtime.sh` | M84 stack; append Beijing-rich time to current UTC time |
| 3 | `M87-m84-t38fix-tempweight25-gap0-seed42-4ep` | `src/run_m87_t38_fix_tempweight25.sh` | M84 stack + M41-style soft temporal reweight |
| 4 | `M88-m84-t38fix-6ep-gap0-seed42` | `src/run_m88_t38_fix_6ep.sh` | M84 stack longer training |

Started after M85/M86回流:

| Job | Run script | Purpose |
|---|---|---|
| `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep` | `src/run_m89_t38_fix_bjtime_v2.sh` | Beijing time v1 + distribution time v2; no sample reweight |
| `M90-m86-utc-bjtime-6ep-gap0-seed42` | `src/run_m90_t38_fix_utc_bjtime_6ep.sh` | M86 exact config extended to 6 epochs |

EDA / audit completed:

| Job | Platform entry | Purpose |
|---|---|---|
| `EDA71-time-window-backtest-audit` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Scan train same-clock windows for the public BJ 09:49~14:49 window and quantify RG time-split overlap |

Clean-window backtest submitted:

| Job | Platform entry | Purpose |
|---|---|---|
| `M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M84 stack, but train rows `< 2026-03-03 09:49:51 BJ` and validate on `2026-03-03 09:49:51~14:49:07 BJ` |
| `M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M86 UTC+Beijing time stack under the same clean pseudo-public window |

Next breakthrough batch submitted to queue:

| Job | Platform entry | Purpose |
|---|---|---|
| `M93-cleanwin0303-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Clean-window delayed-feedback diagnostic; compare vs M91 |
| `M94-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Full-data delayed-feedback public candidate if M93 passes |
| `M95-m84-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Full-data sequence length / timestamp-span feature candidate |
| `M96-m84-mnw6h-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Interaction candidate: delayed feedback + seq stats |

### Last Completed Jobs (按时间倒序)

EDA / diagnostics:

| Job | Result | Decision |
|---|---|---|
| **EDA71** | exact public-window train rows **0**; train max to public start **1s**; RG time split overlap **341,086s (~3.95d)** | `recommend_clean_window_backtest=true`; `recommend_more_absolute_time_dense=false` |

5/18 XHS time-feature batch:

| Job | Config | Best dev | Best epoch | LB | Notes |
|---|---|---:|---:|---|---|
| **M86** | M84 + UTC time + Beijing time | **0.871415** | E4 | **0.847106** ❌ | Dev leader but public -0.001748 vs M84; time-stack did not transfer |
| **M85** | M84 + Beijing time only | **0.870887** | E4 | 待评 | Slightly above M75 dev; mechanism-confirm candidate |
| **M87** | M84 + temporal_weight_alpha=2.5 | 0.869053 | E4 | 不优先 | Temporal reweight does not stack with corrected T38/path B |
| **M88** | M84 extended to 6 epochs | 0.870910 | E6 | 待评 | Score candidate; slight dev edge vs M85, below M86 |

5/18 morning batch:

| Job | Config | Best dev | Best epoch | LB | Notes |
|---|---|---:|---:|---|---|
| **M75(E6)** | M75 family extended to 6 epochs | **0.870651** | E6 | 待评 | E4/E5/E6 几乎持平；仍是 T38 `n_samples=0` default path |
| **M84** | path B + corrected per-domain T38 | 0.870490 | E4 | **0.848854** ⭐ SOTA | 有 schema timestamp resolve log；true T38 小正向 |
| **M76** | path B + DCN cross | 0.870247 | E4 | 待评 | DCN stack dev 未超过 M75 |
| **M82** | path B + alpha2.5 + hour 9-14 x5 | 0.869709 | E4 | **0.846197** ❌ | hour-aware reweight public 负向 |
| **M83** | target-day/hour x25 global weighting | 0.867486 | E4 | 不优先 | 明确负向；reject |

5/17 早起 4 jobs:

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

### 5/18 (today · refreshed @ CEST 06:00 / 北京 12:00)

| Used | Remaining | Used by |
|---|---|---|
| **3** | **1** | M84 LB 0.848854 + M82 LB 0.846197 + M86 LB 0.847106 |

**Future allocation policy (user decision)**:

- Spain morning: use **2-3** eval quotas.
- Spain evening: use **1-2** eval quotas.
- Keep the last quota flexible when training pool has unresolved high-upside candidates.

**User decision (2026-05-18 morning)**:

- Evaluate **M84 E4** now.
- Evaluate **M82 E4** now.
- Reserve the other **2 quotas** for evening decision after more information.
- Expected remaining after these two submissions: **2/4**.

**Evaluation failure note (2026-05-18 14:31 Beijing log)**:

- **M82 E4 evaluation failed before inference** with `dcn_cross_user_fids contains unknown fids [0, 5]`.
- Root cause: `infer.py` parsed residual DCN fields even when `enable_dcn_cross=False`; M82 should ignore those fields.
- Fix applied in `src/infer.py`: skip DCN fid parsing unless DCN is enabled; for true DCN ckpts, accept both raw-fid strings and already-mapped index lists.
- Retry M82 (and M84 if not completed) with patched custom `infer.py`. Failure should not consume quota under the known platform rule, but confirm UI remaining count before evening decisions.

**Evaluation results (2026-05-18 later update)**:

| Candidate | Eval job | Public AUC | Time | Decision |
|---|---|---:|---:|---|
| **M84 E4** | `M84-m22-uesplit-perdomain-t38fix-gap0-seed42-4ep_eval_1779085777310` | **0.848854** | 695.58s | Tiny new SOTA; true T38 small positive |
| **M82 E4** | `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep_eval_1779085835759_copy_1779086114784` | **0.846197** | 684.43s | Reject alpha+hour reweight |
| **M86 E4** | `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep_eval_1779111177317` | **0.847106** | 472.2s | Reject naive UTC+Beijing time stacking for public score |

**Current evening recommendation after M86 public result**:

| Priority | Candidate | Why |
|---|---|---|
| 1 | **M88 E6** | Safest remaining score candidate: M84 family without new Beijing stack, dev 0.870910 and tests longer corrected-T38 training |
| 2 | **M85 E4** | Only if using the last quota for mechanism learning; Beijing-only may be less redundant than M86 but M86 public miss lowers confidence |
| hold | **M89/M90** | Running; after M86 public miss, require training result plus stronger justification before spending eval quota |
| skip | **M87 E4** | 0.869053; temporal_weight stack is below M84 by -0.001438 |

**Reference 4-quota recommendation (post morning-batch)**:

| Priority | Candidate | Why | Mode |
|---|---|---|---|
| 1 | **M75(E6) E6** | Closest same-family score candidate; best dev 0.870651 | score-max |
| 2 | **M84 E4** | Corrected true-T38 path; highest information among new jobs | mechanism + possible score |
| 3 | **M82 E4** | Direct test daypart/reweight hypothesis; dev negative but public-shift question worth one quota | distribution-shift calibration |
| 4 | **M51** if publish-ready; otherwise **M76 E4** | M51 is softer temporal reweight with prior M41 public-positive hint; M76 only if limiting to new-batch jobs | fallback / closure |

Do **not** evaluate M83 with scarce quota.

### 5/17 (closed)

| Used | Remaining | Used by |
|---|---|---|
| **3** | **1** | M55 LB 0.844343 + M71 LB 0.846641 + M68 LB 0.846468 |

**M50 LB · 今晚剩 1 quota** (待 M50 重跑完成 ~22:30 后立即 evaluate · 用户已决)。

**🟢 失败 evaluation 不消耗 quota** (5/17 用户告知铁律): M68 第 1 次 evaluation state_dict mismatch 失败 (q_init_item key bug · commit `cfd15f7` 修复) · 重提成功。

### 5/18 (historical pre-refresh plan) · 4 quota refresh @ CEST 06:00

**Latest note (2026-05-18 morning)**: superseded by current 4-quota recommendation above.

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
| **M49** ⭐⭐⭐ (UE split path B) | 0.870741 | ✅ | safe | **0.848162** · previous SOTA |
| **M75** ⭐⭐⭐ (path B family · nominal per-domain) | 0.870856 | ✅ evaluated | safe | **0.848788** · previous SOTA · T38 not validated (`n_samples=0`) |
| **M75(E6)** (same family extended) | 0.870651 | TBD | 5/19 ~11:22 | 待评 · score-max candidate only if quota allows |
| **M84** (corrected T38 + path B) | 0.870490 | ✅ evaluated | safe | **0.848854** ⭐ SOTA · true T38 small positive |
| **M86** (M84 + UTC time + Beijing time) | 0.871415 | ✅ evaluated | safe | **0.847106** ❌ · dev-public mismatch |
| **M88** (M84 6ep) | 0.870910 | TBD | TBD | 待评 · best remaining score candidate after M86 miss |
| **M85** (M84 + Beijing time only) | 0.870887 | TBD | TBD | 待评低优先 · mechanism only after M86 miss |
| **M76** (path B + DCN) | 0.870247 | TBD | 5/19 ~07:37 | 待评低优先 |
| **M82** (path B + alpha/hour reweight) | 0.869709 | ✅ evaluated | safe | **0.846197** ❌ |
| **M83** (exact day/hour global weighting) | 0.867486 | TBD | 5/19 ~10:22 | 不评 |
| **M51** (path B + temporal_weight) | TBD | ✅ **published 5/17 19:51** | safe (publish 锁) | 待评 |
| **M52** (path B + per-domain · M52' silent skip) | TBD | ⚠️ TBD | 5/18 04:22 ⚠️ | 不评 (silent skip) |
| **M53** (path B + NLIR) | TBD | ✅ **published 5/17 19:51** | safe (publish 锁) | 待评 |
| **M54** (per-domain alone · M54 silent skip) | TBD | ⚠️ TBD | 5/18 04:22 ⚠️ | 不评 (silent skip) |
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

**LB SOTA**: **M84 (path B + corrected true T38) = 0.848854** ⭐
**前 SOTA**: M75 (path B family · nominal per-domain) = 0.848788
**距 0.852 (用户硬 deadline)**: 还差 +0.003146
**距 0.85 (Top 20)**: 还差 +0.001146

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

**Post-update caveat**:
- M75 LB 已回流为 **0.848788 SOTA**。
- 但 M75 训练日志里的 T38 fit `n_samples=0`，所以它不是 T38 真测成功；后续 T38 仍需依赖 corrected nonzero-sample run (如 M84b) 判断。
- 5/18 morning batch 未产生 dev 超越 M75-4ep 的新模型：M75(E6) 0.870651 / M84 0.870490 / M76 0.870247 / M82 0.869709 / M83 0.867486。
- 单 quota 下：若冲分优先，M75(E6) 是最接近 SOTA family 的候选；若机制学习优先，M84 是 true-T38 高信息候选。M83 不评。
- M84 public 0.848854 证明 true T38 小幅 public-positive，但增益只有 **+0.000066 vs M75**，不是足以单独冲 0.852 的大机制。
- M82 public 0.846197 证明 alpha2.5 + hour9~14 x5 reweight 负向；暂停强 daypart reweight variants。
- M86 public 0.847106 证明 UTC+Beijing dense time stacking 出现明显 dev-public mismatch；虽然 dev 比 M84 高 **+0.000925**，public 反而低 **-0.001748**。时间方向仍可研究，但不能再按 dev 绝对值直接消耗 quota。
- EDA71 证明 public 窗口是 train max 后 **1 秒**的未来窗口，且当前 RG time split valid 与 train 时间范围重叠 **~3.95 天**；因此时间特征下一步必须走 clean-window backtest (M91/M92)，不能继续堆 dense absolute time 后直接看 normal dev。
- M93-M96 实践协议已准备: 以 M91 clean-window 为尺子，优先检验 delayed feedback 和 sequence activity，而不是继续堆日历特征。Eval gate: M93-M91 >= +0.0004 才考虑 M94；M95 dev 至少 M84 +0.0003 且 logloss 不坏才考虑。
- 重新检查旧证据后，**不要启动 M91 hour_only**：M40 已测 hour_only，LB **0.846931**，比当前 SOTA 明显低；EDA-056 也指出 DOW/weekend 不是纯噪声。时间方向下一步应先做 EDA-71 / clean window backtest。

**关键 deprecated 方向 (不再投资源)**:
- ❌ Q_init in path B (M68 LB 净损害 −0.001694 · stack 失败)
- ❌ Bilinear gating (M55 LB −0.004 · user-side gating 路径死)
- ❌ daypart filter (M45/M48 dev 灾难 · M50 双 stack 净损害)
- ❌ hour_only replacing time_of_day (M40 LB 0.846931; DOW/weekend removal semi-rejected)
- ❌ naive UTC+Beijing time-feature stacking by dev score alone (M86 dev leader but LB -0.001748 vs M84)
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

- **2026-05-18 later**: 用户补充平台训练池有 **queue mechanism**: jobs submitted to queue auto-start when earlier jobs finish and a worker frees. 用户回报 **M93-M96 已提交平台训练队列**。State file corrected to separate **started/running** from **queued/submitted**. Current queued/scheduled set includes M91-M96 after M89/M90; do not infer manual free slots while queue is non-empty.
- **2026-05-18 later**: Prepared **M93-M96 breakthrough protocol**. M93 = clean-window mature-negative weighting 6h/w0.3 diagnostic; M94 = full-data mature-negative candidate; M95 = sequence stats candidate; M96 = mature-negative + sequence-stats interaction. Evaluation is gate-controlled; see EXP-075.
- **2026-05-18 later**: 用户回报 **M91 / M92 均已提交**。Initially treated as 4 in-flight / 0 free, later corrected after queue-mechanism clarification: M91/M92 are submitted/platform-scheduled clean-window diagnostics on BJ **2026-03-03 09:49:51~14:49:07**; use M91 vs M92 delta to decide whether M89/M90 deserve eval quota despite M86 public miss.
- **2026-05-18 later**: 用户回报 **EDA71-time-window-backtest-audit completed**。日志显示 train range BJ **2026-02-28 08:35:09~2026-03-04 09:49:50**，public window BJ **2026-03-04 09:49:51~14:49:07**，`TRAIN_MAX_TO_PUBLIC_START_SEC=1`，`EXACT_PUBLIC_WINDOW_ROWS_IN_TRAIN=0`，normal RG time split overlap **341086s (~3.95d)**。结论: `recommend_clean_window_backtest=true` / `recommend_more_absolute_time_dense=false`。训练池状态更新为 **2 in-flight / 2 free** (M89, M90 running). Added fixed-window backtest scripts **M91/M92**.
- **2026-05-18 later**: 用户已启动 **EDA71-time-window-backtest-audit**。训练池状态改为 3 in-flight / 1 free (M89, M90, EDA71)。Also decoded M86 eval EDA blob: test rows **523,244**, public BJ window **2026-03-04 09:49:51~14:49:07**, median **12:57:28**, user IDs singleton-like under 200k cap.
- **2026-05-18 later**: 用户同意继续深入时间方向。Re-audit found old **M40 hour_only** evidence: LB **0.846931**, and EDA-056 argued DOW/weekend is not pure noise. Therefore do **not** launch M91 hour_only. Added `src/eda_time_window_backtest.py` for log-only clean time-window audit around public BJ 2026-03-04 09:49:51~14:49:07. Platform entry was uploaded as `run.sh`; local `run_*.sh` templates are intentionally untracked.
- **2026-05-18 later**: 用户回报 **M86 eval completed**: job `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep_eval_1779111177317`, public **0.847106** (472.2s). Eval log confirms intended checkpoint `auc=0.871415.best_model` and dataset flags `time_of_day_features=True`, `beijing_time_features=True`, `beijing_time_v2_features=False`. Conclusion: M86 is a real dev-public mismatch, not an infer config miss. 今日 quota 用 **3/4**，剩 **1/4**。Next eval priority shifts to **M88 E6** if using the last quota.
- **2026-05-18 later**: 用户回报 **M88** completed. Config confirmed `M84 6ep`: `num_epochs=6`, `enable_time_of_day_features=True`, `enable_beijing_time_features=False`, corrected T38 nonzero sample fit. Best dev **0.870910** at E6, history E1 0.865492 / E2 0.868755 / E3 0.869584 / E4 0.870313 / E5 0.870022 / E6 0.870910. Conclusion: longer training helps M84 by +0.000420 dev; after M86 public miss, M88 is the best remaining score candidate.
- **2026-05-18 later**: 用户回报 **M87** completed. Config confirmed `temporal_weight_alpha=2.5`, `enable_time_of_day_features=True`, `enable_beijing_time_features=False`, corrected T38 nonzero sample fit. Best dev **0.869053** (E4), history E1 0.864877 / E2 0.867883 / E3 0.868288 / E4 0.869053. Conclusion: M41-style temporal reweight does **not** stack with corrected T38/path B; do not evaluate with scarce quota.
- **2026-05-18 14:13 CEST**: 用户回报 **M89 / M90 均已启动训练任务**。训练池状态改为 2 in-flight / 2 free。M89 = Beijing time v1+v2 feature-only; M90 = M86 exact config extended to 6 epochs.
- **2026-05-18 CEST later**: M84/M82 eval 回流。M84 public **0.848854** (695.58s) 成为 tiny new SOTA, vs M75 +0.000066；M82 public **0.846197** (684.43s), vs M75 -0.002591。结论: true T38 小正向，hour-aware reweight 负向。今日 quota 用 2/4，剩 2/4。
- **2026-05-18 CEST morning**: 用户回报 M76/M82/M83/M84/M75(E6) 全部训练完成并贴 `train_log.txt`。Best dev: M75(E6)=0.870651, M84=0.870490, M76=0.870247, M82=0.869709, M83=0.867486。结论: 无新 dev SOTA；M75(E6) 是弱冲分候选，M84 是高信息候选，M83 reject。详见 EXP-072。
- **2026-05-17 CEST 19:51**: 用户 publish M51 / M53 ckpt · 5/18 04:22 24h 过期风险解除 · 5/18 早 quota 可正常评 (无 ckpt 时间窗约束) · M52 / M54 未 publish (silent skip 失败 job · 不值得评)
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
