# EXP-071 · M75 LB 0.848788 新 SOTA · 归因修正与 Path B 上沿 · 2026-05-18

> **Status**: Completed
> **Trigger**: 用户回报 M75 evaluation 完成: `auc: 0.848788`, inference time `731.79s`
> **Related**: EXP-059 (M49 UE split path B), EXP-068 (5/17 晚 batch), EXP-069 (hour-aware reweight), ADR-014 (UE split path B)

---

## TL;DR

M75 成为新 public SOTA:

- **M75 public AUC = 0.848788**
- 旧 SOTA M49 = 0.848162
- **净增 = +0.000626**
- 相比 M22 = 0.847592, **净增 = +0.001196**
- 距 0.852 目标仍差 **+0.003212**

但最关键的结论是归因修正:

> **M75 不能归因为真实 T38 / per-domain time bucket 成功。**
>
> M75 训练日志显示 per-domain fit `n_samples=0`，且未出现后续 T38 修复版应有的 schema timestamp column resolve 日志。因此它更接近 **M49-like path B rerun / 更好泛化落点**，而不是 per-domain bucket 的正例。

**2026-05-18 update**: M84 later evaluated at **0.848854**, +0.000066 over M75. M84 is the corrected true-T38 run, so M75 is no longer the public SOTA, but this document's attribution caveat remains correct: old M75 itself was not true T38.

---

## 1. Ground Truth

### 1.1 M75 job

| Field | Value |
|---|---|
| Job Name | `M75-m22-uesplit-perdomain-gap0-seed42-4ep` |
| Nominal Config | M22 + `--enable_ue_split --ue_split_fid 61` + `--enable_per_domain_buckets` |
| Seed | 42 |
| Best checkpoint | `global_step63764.layer=2.head=4.hidden=64.auc=0.870856.best_model` |
| Eval AUC | **0.848788** |
| Inference time | 731.79s |

### 1.2 Checkpoints

| Epoch / ckpt | Dev AUC | Rank |
|---|---:|---:|
| `global_step63764` | **0.870856** | topk_1 / best |
| `global_step47823` | 0.870032 | topk_2 |
| `global_step31882` | 0.868695 | topk_3 |
| `global_step15941` | 0.865343 | topk_4 |

### 1.3 M75(E6) rerun addendum

On 2026-05-18, user provided a 6-epoch M75-family rerun log:

| Epoch / ckpt | Dev AUC |
|---|---:|
| E1 / `global_step15941` | 0.865343 |
| E2 / `global_step31882` | 0.868543 |
| E3 / `global_step47823` | 0.869873 |
| E4 / `global_step63764` | 0.870645 |
| E5 / `global_step79705` | 0.870636 |
| E6 / `global_step95646` | **0.870651** |

Key update:

- Longer training is essentially flat after E4: E6 beats E4 by only **+0.000006**.
- The rerun's best dev **0.870651** is below the original evaluated M75-4ep dev **0.870856**.
- Logs again show per-domain bucket fit `n_samples=0` for seq_a/b/c/d, so the attribution caveat remains unchanged.

---

## 2. Comparison

| Model | Mechanism | Dev AUC | Public AUC | Notes |
|---|---|---:|---:|---|
| M22 | vocabtier baseline | ~0.86858 | 0.847592 | prior stable anchor |
| M49 | M22 + UE split | 0.870741 | 0.848162 | old SOTA; path B validated |
| **M75** | M22 + UE split + nominal per-domain | **0.870856** | **0.848788** | new SOTA; attribution caveat |

Key deltas:

| Delta | Dev | Public |
|---|---:|---:|
| M49 - M22 | +0.002122 | +0.000570 |
| M75 - M49 | **+0.000115** | **+0.000626** |
| M75 - M22 | ~+0.002237 | **+0.001196** |

The shape matters: M75 barely improves dev over M49, but improves public by +0.000626.

---

## 3. Attribution

### 3.1 What is confirmed

**Path B / UE split remains the strongest validated mainline.**

M75 sits in the same family as M49: split the 256-dim UE slice from mixed user_dense, keep it as a dedicated user token, and route other dense features through bypass. This remains aligned with known test facts:

- Test user_id is 100% cold-start.
- Direct user-id memorization is useless.
- User-side pretrained representation / group features can still transfer.
- M49 already proved this mechanism is public-positive.

### 3.2 What is not confirmed

**True T38 per-domain bucket is not validated by M75.**

The nominal job name/config says per-domain bucket, but the observed M75 training logs showed:

- per-domain bucket fit `n_samples=0`
- missing expected schema resolve log like `[T38] resolved timestamp columns from schema: ...`
- therefore likely fallback/default time bucket behavior

So the correct statement is not:

> "M75 proves per-domain time bucket works."

The correct statement is:

> "M75 proves the path B / UE split family has a better public generalization point than M49, but T38 still needs a valid run such as M84b to be judged."

### 3.3 Why public improved more than dev

Dev delta from M49 is only +0.000115, which is near rerun/noise scale. Public delta is +0.000626, well above the dev delta. The public gain mostly comes from a smaller dev-to-LB gap:

| Model | Dev - Public Gap |
|---|---:|
| M49 | 0.022579 |
| M75 | 0.022068 |
| Shrink | **0.000511** |

Interpretation:

- M75 is likely a better generalization basin / checkpoint realization within the path B family.
- The gain may include training nondeterminism, data-order details, or tiny code-path perturbations.
- It should be treated as a new anchor, not a mechanism proof for T38.

---

## 4. Strategic Implications

1. **New anchor is M75 = 0.848788.**
2. Remaining gap to 0.852 is **+0.003212**.
3. M75 strengthens confidence in path B, but does not reduce uncertainty around true T38.
4. Future T38 decisions must use a valid fit log with nonzero samples across seq domains.
5. M84b / later T38 reruns remain high-information because they answer the still-open question M75 could not answer.
6. M82 / M83 / M76 should be compared against M75, not M49.

---

## 5. Decision Record

| Question | Decision |
|---|---|
| Is M75 new SOTA? | Yes, public AUC 0.848788 |
| Is M75 a true per-domain bucket success? | No |
| Should M75 replace M49 as operational anchor? | Yes |
| Should we evaluate other M75 ckpts? | Low priority; E4 is dev best and public confirmed |
| Should T38 still be pursued? | Yes, but only via corrected nonzero-sample run |

---

## 6. Follow-up

- Update any strategy docs still saying "current SOTA = M49 0.848162".
- Treat M75 as the comparison baseline for M82/M83/M76/M84b.
- In all future notes, label old M75 as **"path B better basin; T38 unproven"**.
