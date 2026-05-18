# EXP-072 · 5/18 Morning Batch Results · M76/M82/M83/M84/M75E6 · 2026-05-18

> **Status**: Completed training-log analysis
> **Trigger**: 用户回报 `M76 / M82 / M83 / M84 / M75(E6)` 均训练完成, 日志已贴 `~/Downloads/train_log.txt`
> **Related**: EXP-068 (5/17 evening batch), EXP-069 (M82 hour-aware reweight), EXP-071 (M75 public SOTA attribution), ADR-014 (UE split path B)

---

## TL;DR

This batch did not show a new dev-side model stronger than M75-4ep, but public evaluation later showed **M84 is a tiny new SOTA**.

Operational conclusions:

- **M84 public AUC = 0.848854** (695.58s), tiny new SOTA over M75 by **+0.000066**.
- M84 confirms corrected true T38 has a small public-positive effect, despite lower dev.
- **M82 public AUC = 0.846197** (684.43s), rejecting alpha2.5 + hour9~14 x5 reweight.
- **Current public anchor is M84 = 0.848854**.
- **Remaining gap to 0.852 = +0.003146**.

---

## 0. Public Results

| Job | Dev AUC | Public AUC | Dev-Public Gap | Delta vs M75 public | Decision |
|---|---:|---:|---:|---:|---|
| **M84** | 0.870490 | **0.848854** | **0.021636** | **+0.000066** | Tiny new SOTA; true T38 small positive |
| **M82** | 0.869709 | **0.846197** | 0.023512 | -0.002591 | Reject alpha+hour reweight |
| M75 | 0.870856 | 0.848788 | 0.022068 | — | Previous SOTA; T38 unvalidated |

Important shape:

- M84 dev is **-0.000366 vs M75**, but public is **+0.000066 vs M75**.
- M84 improves the dev-public gap by **0.000432** vs M75.
- This is consistent with "true T38 improves public generalization slightly, and local dev underestimates it."
- M82 has neither local nor public support; it is a clear negative result.

---

## 1. Training Results

| Job | Intended mechanism | Best dev AUC | Best epoch | Logloss | Δ vs M75-4ep dev 0.870856 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| **M75(E6)** | M75 family extended to 6 epochs | **0.870651** | E6 | **0.221541** | -0.000205 | Closest candidate, but no meaningful epoch gain |
| **M84** | path B + corrected per-domain T38 | 0.870490 | E4 | 0.222478 | -0.000366 | High information, not best score signal |
| **M76** | path B + DCN cross | 0.870247 | E4 | 0.222662 | -0.000609 | DCN stack not locally positive |
| **M82** | path B + alpha2.5 + hour 9-14 x5 | 0.869709 | E4 | 0.222851 | -0.001147 | Hour-aware reweight hurts dev materially |
| **M83** | alpha2.5 + exact 03-04 08-09 x25 global | 0.867486 | E4 | 0.224829 | -0.003370 | Over-specialized / reject |

Reference:

- **M75-4ep**: dev 0.870856, public 0.848788.
- **M49**: dev 0.870741, public 0.848162.
- **M22**: dev ~0.86858, public 0.847592.

---

## 2. Per-Job Notes

### 2.1 M75(E6)

Job: `M75-m22-uesplit-perdomain-gap0-seed42-4ep（6个epoch）`

Dev curve:

| Epoch | Step | Dev AUC |
|---:|---:|---:|
| E1 | 15941 | 0.865343 |
| E2 | 31882 | 0.868543 |
| E3 | 47823 | 0.869873 |
| E4 | 63764 | 0.870645 |
| E5 | 79705 | 0.870636 |
| E6 | 95646 | **0.870651** |

Key reading:

- E4/E5/E6 are almost tied; E6 beats E4 by only **+0.000006**.
- The best dev is **below** the previously evaluated M75-4ep best `0.870856`.
- Args confirm `num_epochs=6`, `enable_ue_split=True`, `enable_per_domain_buckets=True`, `per_domain_bucket_sample_rgs=8`.
- T38 logs again show all domains `n_samples=0`, so this remains the old nominal-per-domain/default-bucket path, not true T38.

Decision:

- If the only goal is immediate score maximization, this is the only plausible one-shot public candidate from this batch.
- It is still a weak bet because it does not improve dev over the known M75-4ep public SOTA.

### 2.2 M84

Job: `M84-m22-uesplit-perdomain-t38fix-gap0-seed42-4ep`

Dev curve:

| Epoch | Step | Dev AUC |
|---:|---:|---:|
| E1 | 15941 | 0.865526 |
| E2 | 31882 | 0.868882 |
| E3 | 47823 | 0.869638 |
| E4 | 63764 | **0.870490** |

Confirmed Args:

- `enable_ue_split=True`
- `enable_per_domain_buckets=True`
- `per_domain_bucket_sample_rgs=64`
- `enable_dcn_cross=False`
- no temporal/hour/target-day weighting

T38 evidence:

- Log contains `[T38] resolved timestamp columns from schema: {'seq_a': 'domain_a_seq_39', 'seq_b': 'domain_b_seq_67', 'seq_c': 'domain_c_seq_27', 'seq_d': 'domain_d_seq_26'}`.
- Unlike old M75/M75(E6), this run entered the corrected schema-resolve path.
- The copied log does not include the compact per-domain fit rows, but the corrected code path should fail-fast if all samples are zero; this makes M84 the first credible true-T38 candidate in the M75 family.

Decision:

- Evaluation priority is **medium**: high information, moderate score probability.
- It should not be treated as stronger than M75 unless public proves a dev-down/public-up behavior.

### 2.3 M76

Job: `M76-m22-uesplit-dcn-gap0-seed42-4ep`

Dev curve:

| Epoch | Step | Dev AUC |
|---:|---:|---:|
| E1 | 15941 | 0.866600 |
| E2 | 31882 | 0.867678 |
| E3 | 47823 | 0.869277 |
| E4 | 63764 | **0.870247** |

Decision:

- Below M75 by **-0.000609** dev.
- DCN cross stack is not locally positive in the path B family.
- Evaluation priority low unless quota is abundant.

### 2.4 M82

Job: `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep`

Dev curve:

| Epoch | Step | Dev AUC |
|---:|---:|---:|
| E1 | 15941 | 0.866622 |
| E2 | 31882 | 0.867595 |
| E3 | 47823 | 0.869076 |
| E4 | 63764 | **0.869709** |

Decision:

- Below M75 by **-0.001147** dev.
- The D-3 hour-aware reweight did not improve local validation.
- Public could theoretically improve if validation is misaligned with test, but this is now a lower-priority bet than M75(E6) / M84.

### 2.5 M83

Job: `M83-m22-uesplit-alpha25-day0304-h8to9x25-global-gap0-seed42-4ep`

Dev curve:

| Epoch | Step | Dev AUC |
|---:|---:|---:|
| E1 | 15941 | 0.862860 |
| E2 | 31882 | 0.866355 |
| E3 | 47823 | 0.866439 |
| E4 | 63764 | **0.867486** |

Decision:

- Below M75 by **-0.003370** dev.
- This is a clear negative result.
- Reject exact-day/hour x25/global weighting in this form.

---

## 3. Evaluation Priority

If only **1 evaluation quota** is available:

1. **Score-maximization mode**: evaluate `M75(E6)` best model (`global_step95646...auc=0.870651.best_model`).
2. **Information-gain mode**: evaluate `M84` E4 (`global_step63764...auc=0.870490.best_model`) to answer whether true T38 helps public.

My default recommendation under 5/20 score pressure:

> Use the next single quota on **M75(E6)** only if we are trying to squeeze immediate public score. Use it on **M84** only if we decide today is a mechanism-learning quota.

If **2 quotas** are available:

1. `M75(E6)` E6
2. `M84` E4

If **4 quotas** are available:

1. `M75(E6)` E6
2. `M84` E4
3. `M82` E4
4. `M51` if publish-ready; otherwise `M76` E4 if we limit strictly to the new morning batch

Rationale for slot 4:

- M51 is the softer temporal reweight stack and has a better prior than DCN because M41 had a small public-positive hint.
- M76 is useful only as DCN closure; it is below M75 by -0.000609 dev and path-A DCN already had a poor public history.

Do **not** spend scarce quota on M83. M82/M76 only become reasonable after the top two or if we specifically want to test dev-down/public-up behavior for the hour/DCN paths.

2026-05-18 morning user decision:

- Submit **M84 E4** and **M82 E4** now.
- Reserve the other **2 quotas** for evening, after the M84/M82 public results and any new training signals.
- This is an information-first allocation: true T38 + daypart/reweight get public calibration before committing the remaining quota.

2026-05-18 evaluation failure addendum:

- M82 E4 evaluation failed immediately with `dcn_cross_user_fids contains unknown fids [0, 5]`.
- The loaded model dir confirms it was M82: `...M82...global_step63764...auc=0.869709.best_model`.
- Root cause was inference-side compatibility, not the M82 model: `infer.py` tried to resolve residual DCN fields even though `enable_dcn_cross=False`.
- Patch applied in `src/infer.py`: skip DCN fid resolution when DCN is disabled; for DCN-enabled ckpts, support both raw fid strings and already-mapped index lists.
- Recommended action: retry M82 with patched custom `infer.py`; also use the patched infer for M84 to avoid the same residual-field issue.

2026-05-18 public result addendum:

- M84 completed: `M84-m22-uesplit-perdomain-t38fix-gap0-seed42-4ep_eval_1779085777310`, public **0.848854**, 695.58s.
- M82 completed after retry: `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep_eval_1779085835759_copy_1779086114784`, public **0.846197**, 684.43s.
- M84 replaces M75 as operational SOTA, but only by +0.000066.
- M82 closes the strong daypart reweight path as negative.

---

## 4. Strategic Implications

1. **Longer M75 training did not materially lift AUC.**
   - E4/E5/E6 are flat.
   - Do not spend more jobs on naive longer training of this exact config.

2. **True T38 is public-positive but small.**
   - M84 is below M75-4ep and M75(E6) on dev.
   - Public nevertheless improves to 0.848854, a +0.000066 SOTA.
   - Treat T38 as a small generalization-gap shrinker, not a large standalone mechanism.

3. **Hour reweight needs to be softened or paused.**
   - M82 already drops dev by >0.001.
   - M82 public is 0.846197, -0.002591 vs M75.
   - M83 proves target-day/hour over-weighting can easily overfit.

4. **DCN cross path is weak in this stack.**
   - M76 does not beat path B.
   - Do not launch DCN variants before stronger evidence.

5. **M84 public SOTA is the operational anchor.**
   - The batch has not found a clear additive mechanism to bridge the remaining +0.003146 to 0.852.
   - Next training should focus on either new signal families or public-driven checkpoint/seed exploration, not heavier versions of the failed reweight/DCN settings.

---

## 5. Caveats

- The copied log contains full Args for M84 and M75(E6), but not full Args rows for M76/M82/M83. Their config interpretation therefore combines job name, prior run.sh design, and training summary.
- The M84 copied log contains the schema-resolve T38 line but not the compact per-domain fit rows. Because the corrected implementation is expected to fail fast if all domains are zero, this is still substantially stronger evidence than old M75's explicit `n_samples=0` fallback.
