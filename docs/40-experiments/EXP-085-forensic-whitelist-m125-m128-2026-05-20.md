# EXP-085 · Forensic whitelist retry · M125-M128 · 2026-05-20

> **Status**: M125 and M127 clean diagnostics both red; M126/M128 completed logs-only and are not public-eval candidates by default.
> **Trigger**: The user questioned whether plausible directions have failed
> because of implementation flaws, earlier lack of clean-window validation, or
> missed old candidates. This audit separates confirmed public-negative branches
> from genuinely unresolved or suspicious branches.

## Audit Read

Do not reopen these by default:

| Branch | Evidence | Decision |
|---|---|---|
| Q-init item | M68 initially failed eval because `infer.py` missed the `enable_q_init_item` key, but after the fix it evaluated at public **0.846468** | closed |
| UE-int bilinear | M55 evaluated at public **0.844343** | closed |
| DCN cross | M23 public was severely negative; M76 path-B+DCN did not beat the M84/M106 family on dev | closed unless buying calibration |
| Dense time stacks | M86/M89 public misses plus clean-window failures M92/M99/M100 | closed for score-max |
| History-CVR | M104 public **0.847560** despite strong normal dev | closed for score-max |
| Seqstats / mature negative | M97/M98 clean gates failed hard | closed |

The unresolved whitelist is narrower:

1. **T31 is-missing**: code was implemented and smoke-tested, but no reliable
   full platform validation was run. EDA fullscan showed strong missingness
   signals, including user fids `99/109` and item fids `7/12/83/84/85`.
2. **T34 multi-embedding K=2**: the original M38 attempt suffered a platform
   silent-copy accident (`multi_emb_k=1`, `enable_nlir_gating=True`). Later
   evidence was weak, but the implementation-risk angle remains legitimate
   enough for a low-priority forensic retry on the current M106 backbone.
3. **Global Token** is not selected because it conflicts with `enable_ue_split`,
   and M106/M84 rely on the UE-split path.

## Jobs

All jobs use the M84/M106 corrected-T38 backbone:

- `--enable_ue_split --ue_split_fid 61`
- `--enable_din_interest`
- `--enable_per_domain_buckets --per_domain_bucket_sample_rgs 64`
- `--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"`

| Priority | Job | Local helper template | Purpose |
|---:|---|---|---|
| 1 | `M125-cleanwin0303-m84-ismiss-useritem-gap0-seed42-4ep` | `platform_run_templates/run_m125_cleanwin0303_m84_ismiss_useritem.sh` | Clean-window diagnostic: M84 + T31 missingness, compare vs M91 |
| 2 | `M126-m106-ismiss-useritem-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m126_m106_ismiss_useritem_seed5_auxm91.sh` | Full M106 candidate if M125 is not red |
| 3 | `M127-cleanwin0303-m84-multiembk2-b96-gap0-seed42-4ep` | `platform_run_templates/run_m127_cleanwin0303_m84_multiembk2_b96.sh` | Clean-window diagnostic for the old silent-copy Multi-Emb branch |
| 4 | `M128-m106-multiembk2-b96-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m128_m106_multiembk2_b96_seed5_auxm91.sh` | Full M106 Multi-Emb retry; heavier and lower priority |

Upload the selected helper as platform `run.sh`.

## Readout Gates

M91 clean baseline is **0.769513**.

| Pair | Green | Yellow | Red |
|---|---|---|---|
| M125 is-missing clean | `M125 - M91 >= +0.0004` | `[-0.0003, +0.0004)` plus stable logloss | `< -0.0003` |
| M127 multi-emb clean | `M127 - M91 >= +0.0004` | `[-0.0002, +0.0004)` | `< -0.0002` or OOM/instability |

Public eval policy:

1. Do not evaluate M126 by default: M125 is red, even though M126 full-data
   training is sane against the M106 family.
2. M128 is an information hedge only. Because Multi-Emb is capacity expansion,
   a weak clean result should veto public eval.
3. For any eval involving M125-M128, upload the current custom `infer.py`,
   `dataset.py`, and `model.py`; do not rely on platform defaults.

Startup checks from `Args` logs:

- M125/M126: `enable_is_missing=True`, expected missing-fid lists present.
- M127/M128: `multi_emb_k=2`, `batch_size=96`, `enable_nlir_gating=False`.
- All: `enable_ue_split=True`, `enable_per_domain_buckets=True`,
  `enable_din_interest=True`, correct seed.

## M125 Result

Source: user copied `~/Downloads/train_log.txt` after M120 and M125 completed.

M125 config was correct:

- `enable_is_missing=True`
- `is_missing_user_int_fids=86,94,99,100,101,102,103,104,108,109`
- `is_missing_item_int_fids=7,12,83,84,85`
- fixed clean window `[1772502591, 1772520547)`

Result:

| Job | Best clean AUC | Best ckpt | Logloss | Delta vs M91 | Decision |
|---|---:|---|---:|---:|---|
| M125 | **0.765837** | E4 `global_step10136...auc=0.765837.best_model` | 0.217792 | **-0.003676** | Red; veto M126 public eval by default |

Interpretation:

- This is not an implementation-silent-skip: the `Args` line shows the intended
  missingness flags and fid lists.
- Clean-window performance is far below the M91 baseline **0.769513** and below
  the red threshold.
- If M126 is already running, let it finish for logging, but do not spend public
  eval quota on the is-missing branch by default.

## M126 Result

Source: user copied `~/Downloads/train_log.txt` after M126 completed.

M126 config was correct:

- `enable_is_missing=True`
- `is_missing_user_int_fids=86,94,99,100,101,102,103,104,108,109`
- `is_missing_item_int_fids=7,12,83,84,85`
- standard 5% time validation with aux `m91:1772502591:1772520547`

Result:

| Job | Best main valid | Best ckpt | Logloss | Aux m91 | Decision |
|---|---:|---|---:|---:|---|
| M126 | **0.870945** | E4 `global_step63764...auc=0.870945.best_model` | 0.221435 | **0.852607** | Sane full training, but M125 clean gate is red; do not evaluate by default |

Interpretation:

- This branch is not broken at startup; the full-data run learns normally.
- The full-data score is below M106 E4 standard valid **0.871175**, and the
  clean-window diagnostic is materially negative.
- Public quota should remain reserved for stronger candidates such as M119 or
  any future clean-green forensic result.

## M127/M128 Result

Source: user copied `~/Downloads/train_log.txt` after all jobs except M133/M135
returned on 2026-05-21.

M127/M128 config was correct for the forensic MultiEmbK2 retry:

- `multi_emb_k=2`
- `batch_size=96`
- `enable_nlir_gating=False`
- corrected-T38 M84/M106 backbone

Result:

| Job | Best AUC | Best ckpt | Aux m91 | Delta vs anchor | Decision |
|---|---:|---|---:|---:|---|
| M127 clean | **0.763918** | E4 `global_step13448...auc=0.763918.best_model` | n/a | vs M91 **-0.005595** | Red; clean gate fails hard |
| M128 full | **0.870263** | E4 `global_step84796...auc=0.870263.best_model` | **0.851802** | vs M106 **-0.000912** | Hold; no public eval by default |

Interpretation:

- The historical M38 silent-copy suspicion is now resolved enough for score-max:
  a correctly launched MultiEmbK2 retry does not pass the clean diagnostic.
- M128 learns normally, but it is below the accepted M106/M119/M123 anchors on
  primary validation and does not have a strong auxiliary signal.
- Close the MultiEmbK2 forensic branch unless a future architecture change gives
  it a different reason to exist.
