# EXP-092 · TIN-lite Temporal Interest · M143-M144 · 2026-05-21

> **Status**: Implemented locally; user submitted/queued M143 and M144. M144
> has returned with correct Args: TIN-lite improved aux/logloss but weakened the
> primary tiny-valid selector versus M138. M138 later evaluated at only
> **0.849451**, so M144 is now a high-information TIN/soup calibration candidate
> rather than a score-max favorite. M143 also returned with correct Args but is
> weaker than the seed7/freshness public-positive anchors. M144 dense-only soup
> later evaluated at public **0.848183**, closing TIN-lite/vr015-soup in this
> form for score-max.

## Trigger

The M123-family has the only confirmed public-positive evidence, but the gap
from M123 public **0.851011** to the current target **0.853** is still about
**+0.001989**. M137/M138/M141/M142 are refinement and checkpoint-stability
tests; if they plateau below 0.852, we need a mechanism-level improvement that
still stays close to the proven backbone.

## Mechanism

TIN-lite is a conservative temporal upgrade to the existing classifier-time DIN
interest extractor. It keeps the same classifier width as DIN, and therefore
does not add another large bypass vector.

For each sequence domain:

1. Keep the existing target-item query representation.
2. Embed the sequence time bucket.
3. Build a target-conditioned time modulation:
   `[time_emb, target_repr, time_emb * target_repr] -> d_model`.
4. Add it to the sequence tokens before target-item attention:
   `seq'_t = seq_t + alpha_i * time_mod_t`.
5. Run the same per-domain target attention and compact/per-domain merge as
   DIN.

The final temporal MLP layer is zero-initialized, so the extractor starts close
to ordinary DIN and learns temporal modulation only if useful.

## Guardrails

- `--enable_tin_interest` requires `--enable_din_interest`.
- It currently requires `--din_interest_source raw`, because raw sequence
  tokens align exactly with `seq_time_buckets`.
- It requires `use_time_buckets=True`.
- It does not change sparse embeddings, sequence encoders, HyFormer blocks,
  MTL heads, top-k checkpointing, aux selection, or dense-only soup.

## Jobs

| Priority | Job | Helper | Purpose |
|---:|---|---|---|
| 1 | `M143-m106-vr02-mtl-seed7-tin-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m143_m106_vr02_mtl_seed7_tin_auxselect_soup.sh` | M137 recipe with TIN-lite + aux selector + dense-only soup. |
| 2 | `M144-m123-vr015-mtl-seed5-tin-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m144_m123_vr015_mtl_seed5_tin_auxselect_soup.sh` | M138 recipe with TIN-lite + aux selector + dense-only soup. |

## Readout

Compare M143 against M137/M141 and M144 against M138/M142:

- If TIN improves primary validation without aux-only weirdness, it becomes
  the next public-eval candidate.
- If primary is flat but soup/aux is stronger, evaluate only after checking
  that the mechanism did not reproduce M136's high-aux/low-public pattern.
- If both M143 and M144 weaken primary materially, close TIN-lite in this form
  and consider a separate-token UE revisit or another temporal objective.

## Verification

Local static checks passed:

- `python3 -m py_compile src/model.py src/train.py src/infer.py src/trainer.py`
- `bash -n platform_run_templates/run_m143_m106_vr02_mtl_seed7_tin_auxselect_soup.sh`
- `bash -n platform_run_templates/run_m144_m123_vr015_mtl_seed5_tin_auxselect_soup.sh`

## M144 Completion

User reported M144 completed and copied logs to `~/Downloads/train_log.txt`.
Config is correct:

- `valid_ratio=0.015`
- seed **5**
- `multi_task_loss=True`
- legacy UE61-only: `ue_split_fids=''`, `ue_split_fid=61`
- `enable_tin_interest=True`, `tin_time_alpha_init=1.0`
- `enable_aux_ckpt_select=True`
- `enable_weight_soup=True`, `weight_soup_include_sparse=False`

Primary curve:

| Epoch | Step | Primary AUC | LogLoss | Aux m91 AUC | Blend score |
|---:|---:|---:|---:|---:|---:|
| 1 | 16564 | 0.868968 | 0.223768 | 0.844862 | 0.856915 |
| 2 | 33128 | 0.871021 | 0.221163 | 0.850133 | 0.860577 |
| 3 | 49692 | 0.871833 | 0.220840 | 0.854089 | 0.862961 |
| 4 | 66256 | **0.871954** | **0.220568** | **0.855139** | **0.863547** |

Primary checkpoint:

`global_step66256.layer=2.head=4.hidden=64.auc=0.871954.best_model`

Aux/blend side checkpoints:

- `global_step66256.layer=2.head=4.hidden=64.score=0.855139.auc=0.871954.aux_m91=0.855139.aux_m91_best`
- `global_step66256.layer=2.head=4.hidden=64.score=0.863547.auc=0.871954.aux_m91=0.855139.blend_m91_best`

Dense-only soup:

- Sources: E4 best_model **0.871954**, E3 topk **0.871833**, E2 topk
  **0.871021**
- Sparse embeddings preserved from E4 best source
- Output:
  `global_step66256.layer=2.head=4.hidden=64.auc=0.871954.members=3.weight_soup`

Comparison:

| Anchor | Primary | LogLoss | Aux m91 | M144 delta |
|---|---:|---:|---:|---:|
| M138 vr015 baseline | 0.872871 | 0.221286 | 0.854026 | primary **-0.000917**, logloss **-0.000718**, aux **+0.001113** |
| M123 public SOTA | 0.870584 | 0.221884 | 0.853945 | primary **+0.001370**, logloss **-0.001316**, aux **+0.001194** |
| M135 UE61+xitem | 0.871377 | 0.221546 | 0.853113 | primary **+0.000577**, logloss **-0.000978**, aux **+0.002026** |

Interpretation:

- TIN-lite is not a clean primary-valid win over M138. It lowers primary by
  nearly 0.001 on the same 1.5% split.
- It does improve aux and logloss, so it may be more calibrated for the public
  window than primary alone suggests.
- After M138 public **0.849451**, the `vr015` selector itself is suspect.
  Therefore M144 should not be framed as the highest expected score path.
- If the final quota before refresh would otherwise idle and no corrected
  seed7 `vr02+MTL` result returns, prefer **M144 dense-only soup** over plain
  M144 E4 as a high-information calibration, because soup is the intended
  stability mechanism and was built from close E2/E3/E4 snapshots.

## M144 Public Eval

User reported:

`M144-m123-vr015-mtl-seed5-tin-auxselect-soup-gap0-4ep_eval_1779394373414`

Public result: **0.848183** (431.87s).

Public deltas:

| Anchor | Public AUC | M144 delta |
|---|---:|---:|
| M123 `vr02+MTL` seed5 | 0.851011 | **-0.002828** |
| M119 `vr02` seed7 | 0.850754 | **-0.002571** |
| M106 standard seed5 | 0.849364 | **-0.001181** |
| M138 `vr015+MTL` seed5 | 0.849451 | **-0.001268** |

Interpretation:

- Dense-only soup did **not** stabilize the `vr015` failure exposed by M138.
- TIN-lite did **not** transfer, despite better aux m91 and logloss on M144.
- Close TIN-lite in this form for score-max. M143 was weak internally and M144
  is now public-negative.
- Do not evaluate M142 by default: it shares the rejected `vr015+soup` premise
  and lacks the public-negative TIN component only as a smaller ablation, not a
  plausible 0.853 path.

## M143 Completion

User reported M143 completed and copied logs to `~/Downloads/train_log.txt`.
Config is correct:

- `valid_ratio=0.02`
- seed **7**
- `multi_task_loss=True`
- legacy UE61-only: `ue_split_fids=''`, `ue_split_fid=61`
- `enable_tin_interest=True`, `tin_time_alpha_init=1.0`
- `enable_aux_ckpt_select=True`
- `enable_weight_soup=True`, `weight_soup_include_sparse=False`

Primary curve:

| Epoch | Step | Primary AUC | LogLoss | Aux m91 AUC | Blend score |
|---:|---:|---:|---:|---:|---:|
| 1 | 16480 | 0.866202 | 0.226203 | 0.844940 | 0.855571 |
| 2 | 32960 | 0.867509 | 0.223484 | 0.844590 | 0.856050 |
| 3 | 49440 | 0.868428 | 0.223589 | 0.849679 | 0.859053 |
| 4 | 65920 | **0.869295** | **0.222664** | **0.853440** | **0.861368** |

Primary checkpoint:

`global_step65920.layer=2.head=4.hidden=64.auc=0.869295.best_model`

Aux/blend side checkpoints:

- `global_step65920.layer=2.head=4.hidden=64.score=0.853440.auc=0.869295.aux_m91=0.853440.aux_m91_best`
- `global_step65920.layer=2.head=4.hidden=64.score=0.861368.auc=0.869295.aux_m91=0.853440.blend_m91_best`

Dense-only soup:

- Sources: E4 best_model **0.869295**, E3 topk **0.868428**
- Sparse embeddings preserved from E4 best source
- Output:
  `global_step65920.layer=2.head=4.hidden=64.auc=0.869295.members=2.weight_soup`

Comparison:

| Anchor | Primary | LogLoss | Aux m91 | M143 delta |
|---|---:|---:|---:|---:|
| M119 `vr02` seed7 public-positive | 0.871196 | 0.221458 | 0.855080 | primary **-0.001901**, logloss **+0.001206**, aux **-0.001640** |
| M123 `vr02+MTL` seed5 public SOTA | 0.870584 | 0.221884 | 0.853945 | primary **-0.001289**, logloss **+0.000780**, aux **-0.000505** |
| M143 sibling M144 `vr015+TIN` seed5 | 0.871954 | 0.220568 | 0.855139 | primary **-0.002659**, logloss **+0.002096**, aux **-0.001699** |

Interpretation:

- M143 is a correct run, but TIN-lite hurts the seed7 `vr02` backbone on every
  relevant internal metric versus M119/M123.
- This does **not** answer the pure M137/M141 question, because TIN is a
  confound. Corrected M141 remains the cleaner test of seed7 `vr02+MTL+soup`.
- Do **not** spend public quota on M143 by default. If the final quota must be
  used before refresh and M141 does not return, M144 dense-only soup is still
  the more informative calibration than M143.
