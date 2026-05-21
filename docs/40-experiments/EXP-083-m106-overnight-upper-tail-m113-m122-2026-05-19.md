# EXP-083 · M106-family overnight upper-tail and MTL batch · 2026-05-19

> **Status**: M113-M122 all completed. M119 evaluated at public **0.850754**, validating the freshness / reduced-valid-ratio path. M116 later evaluated at public **0.847886**, rejecting standard-valid seed upper-tail as a score-max path.
> **Trigger**: M106 public returned **0.849364**, a real transferable gain over M84. M106 E3 later evaluated at **0.848760**, below E4, so checkpoint selection is not yet a validated lever. The remaining gap to 0.852 is **+0.002636**, so the overnight queue should search for transferable upper-tail variants on the proven M106/M84 backbone, plus one low-risk mechanism that uses click supervision.

## Read

M106 changes the posture:

- The M84 corrected-T38 family is the only current public-positive backbone.
- M106 seed5 beat M84 by **+0.000510 public** and preserved much of its normal-dev edge.
- M106 E3 scored **0.848760**, **-0.000604 vs E4**, so the public-positive signal currently belongs to the E4 checkpoint rather than the full checkpoint trajectory.
- This supports controlled upper-tail search, but does **not** by itself explain the missing +0.002636 to 0.852.

Important negative constraints:

- Do **not** reopen AUC/BPR loss by default. EXP-022 had a severe public-negative result: `loss_type=auc` contributed to a **-0.00993** public drop in the older stack.
- Do **not** promote M110/M112 by default. Their clean companions M109/M111 both failed.
- Do **not** reopen history-CVR, dense Beijing/UTC time stacks, seqstats, or longer training without a new reason.

## Jobs

All jobs keep the M106/M84 corrected-T38 backbone:

- `--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2`
- `--num_queries 2`
- `--ns_groups_json ns_groups_vocab_tier.json`
- `--seq_encoder_type swiglu --seq_hash_vocab 500000`
- `--enable_time_of_day_features`
- `--item_conditioned_query`
- `--enable_din_interest`
- `--enable_ue_split --ue_split_fid 61`
- `--enable_per_domain_buckets --per_domain_bucket_sample_rgs 64`
- `--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"`
- `--aux_valid_windows "m91:1772502591:1772520547"`

| Priority | Job | Local helper template | Single variable / purpose |
|---:|---|---|---|
| 1 | `M113-m84-t38fix-seed7-auxm91-gap0-4ep` | `platform_run_templates/run_m113_m84_t38fix_seed7_auxm91.sh` | seed upper-tail, standard 5% valid |
| 2 | `M114-m84-t38fix-seed11-auxm91-gap0-4ep` | `platform_run_templates/run_m114_m84_t38fix_seed11_auxm91.sh` | seed upper-tail, standard 5% valid |
| 3 | `M115-m84-t38fix-seed13-auxm91-gap0-4ep` | `platform_run_templates/run_m115_m84_t38fix_seed13_auxm91.sh` | seed upper-tail, standard 5% valid |
| 4 | `M116-m84-t38fix-seed17-auxm91-gap0-4ep` | `platform_run_templates/run_m116_m84_t38fix_seed17_auxm91.sh` | seed upper-tail, standard 5% valid |
| 5 | `M117-m106-vr02-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m117_m106_vr02_seed5_auxm91.sh` | reduce valid holdout to 2%, train on fresher pre-public rows |
| 6 | `M118-m106-vr01-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m118_m106_vr01_seed5_auxm91.sh` | reduce valid holdout to 1%, more aggressive freshness |
| 7 | `M119-m106-vr02-seed7-auxm91-gap0-4ep` | `platform_run_templates/run_m119_m106_vr02_seed7_auxm91.sh` | seed7 + fresher 2% valid |
| 8 | `M120-m106-vr02-seed13-auxm91-gap0-4ep` | `platform_run_templates/run_m120_m106_vr02_seed13_auxm91.sh` | seed13 + fresher 2% valid |
| 9 | `M121-m106-mtl-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m121_m106_mtl_seed5_auxm91.sh` | multi-task click + conversion, seed5 |
| 10 | `M122-m106-mtl-seed7-auxm91-gap0-4ep` | `platform_run_templates/run_m122_m106_mtl_seed7_auxm91.sh` | multi-task click + conversion, seed7 |

User later confirmed all ten helpers have been submitted and are either running
or queued. Upload entry was platform `run.sh` for each helper.

## M113-M122 Results

Source: user copied `~/Downloads/train_log.txt` after M113, M114, M121, and
M122 completed; later copied the M115/M116/M117/M118/M120 logs after
completion; later copied the M119 log after completion.

| Job | Branch | Best main valid | Best ckpt | Best/paired aux m91 | Readout |
|---|---|---:|---|---:|---|
| M113 | seed7, 5% valid | **0.870344** / logloss 0.221712 | E4 `global_step63764...auc=0.870344.best_model` | E4 **0.853309** | Aux is interesting, but main dev is **-0.000831 vs M106 E4** |
| M114 | seed11, 5% valid | **0.870128** / logloss 0.222434 | E3 `global_step47823...auc=0.870128.best_model` | E3 0.850630; E4 0.853194 | E4 aux rises while main dev drops; no default eval |
| M115 | seed13, 5% valid | **0.870309** / logloss 0.223152 | E4 `global_step63764...auc=0.870309.best_model` | E4 **0.853551** | Aux is strong, but main dev is **-0.000866 vs M106 E4** and **-0.000382 vs M121** |
| M116 | seed17, 5% valid | **0.870697** / logloss 0.222083 | E4 `global_step63764...auc=0.870697.best_model` | E4 **0.853761** | Best returned main valid by a hair and strongest aux, but still **-0.000478 vs M106 E4** |
| M117 | seed5, 2% valid freshness | **0.870034** / logloss 0.222221 | E4 `global_step65920...auc=0.870034.best_model` | E4 **0.852345** | Weaker than M106 on non-comparable 2% valid; freshness alone is not enough |
| M118 | seed5, 1% valid freshness | **0.869640** / logloss 0.223958 | E4 `global_step66592...auc=0.869640.best_model` | E1-E3 peak **0.849728**; E4 aux missing | Tiny-valid score is not comparable to 5% valid, but sibling aux is weak; no default eval |
| M119 | seed7, 2% valid freshness | **0.871196** / logloss 0.221458 | E4 `global_step65920...auc=0.871196.best_model` | E4 **0.855080** | Strongest 2% freshness signal and highest returned aux; top upside public calibration candidate |
| M120 | seed13, 2% valid freshness | **0.870606** / logloss 0.221836 | E4 `global_step65920...auc=0.870606.best_model` | E4 0.852176; E3 peak **0.852588** | Better than M117/M118, but still below M116 main/aux; no default eval |
| M121 | MTL seed5, 5% valid | **0.870691** / logloss 0.221751 | E4 `global_step63764...auc=0.870691.best_model` | E4 0.852123 | Essentially tied with M116 on main valid, but lower aux; still **-0.000484 vs M106 E4** |
| M122 | MTL seed7, 5% valid | **0.870176** / logloss 0.222393 | E3 `global_step47823...auc=0.870176.best_model` | E3 0.852629; E4 0.853627 | High E4 aux, but best main checkpoint is weaker |

Interpretation:

- The seed upper-tail sweep has a new best returned standard-valid job, M116, but it is still
  below M106 on primary validation. Its aux m91 is the strongest among the
  comparable standard-valid seed jobs,
  so it is the least-bad forced-choice public calibration candidate if we must
  stay within comparable 5% validation jobs.
- MTL is not broken: M121 is competitive and has healthy logloss. However it
  does not beat M106 on the main selection metric, and M116 now narrowly edges
  it on both main valid and aux m91.
- M117/M118/M120 did not rescue freshness-only, but M119 changes the read:
  seed7 + 2% valid reaches **0.871196** and aux m91 **0.855080**. Because the
  validation window changes, this is not a clean score estimate, but it is the
  strongest returned public-adjacent upside signal.
- Aux m91 is leaky for these full-data jobs. It can rank sibling jobs, but
  should not override the primary valid AUC without public calibration.

Decision: if a public quota must be used for upside / calibration from this
batch, prefer **M119 E4**. If the user wants a safer comparable-split seed
calibration instead, use **M116 E4**. M121 E4 is the second standard-valid
choice; M115/M113 are aux-interest holds, not score-max defaults.

## M119 Public Eval

User reported:

| Job | Public AUC | Time | Interpretation |
|---|---:|---:|---|
| `M119-m106-vr02-seed7-auxm91-gap0-4ep_eval_1779306323511` | **0.850754** | 703.23s | New SOTA; +0.001390 vs M106 E4 |

Takeaway: M119's `valid_ratio=0.02` / fresher-training setup transferred much better than prior standard-valid seed variants. It does not reach 0.852, but it changes the active score-max baseline from M106 **0.849364** to M119 **0.850754** and makes freshness-family followups (M123/M124 and future M129-M132 stacks) more important than older seed-only attempts.

## M116 Public Eval

User reported:

| Job | Public AUC | Time | Interpretation |
|---|---:|---:|---|
| `M116-m84-t38fix-seed17-auxm91-gap0-4ep_eval_1779309435749` | **0.847886** | 532.77s | Rejected; below M84/M106/M119/M123 |

Takeaway: M116 looked like the safest comparable 5% valid fallback, but public
fell sharply. Close standard-valid seed upper-tail as a score-max route; future
quota should prefer public-adjacent freshness/MTL stacks or new mechanisms.

## Why These Ten

### Layer 1: seed upper-tail

M106 proves same-family seed search can transfer. We should not pretend seed is a mechanism, but at `0.849364` even +0.0005 matters.

### Layer 2: fresher training rows

The standard `valid_ratio=0.05` with `valid_split_strategy=time` withholds the latest roughly 5% of training row groups from the model. Public starts immediately after train max, so these rows are likely the most public-adjacent supervised data we have. Reducing valid to 2% or 1% is a controlled way to train on more recent rows while retaining checkpoint selection.

Normal dev from these jobs is not directly comparable to M106 because the validation window changes. Use it mainly for sanity and checkpoint selection, not as a clean score estimate.

### Layer 3: multi-task learning

`--multi_task_loss --action_num 2` uses click supervision as an auxiliary head while keeping conversion as the final evaluated head. This is a plausible ads-system mechanism: click labels are denser and can regularize representation, while inference still uses the conversion logit.

This is more defensible than reopening BPR/AUC loss, because it does not directly perturb score calibration with pairwise ranking terms.

## Evaluation Gate

Next public quota should be spent in this order:

1. For M119, use public-adjacent rationale plus sanity checks because
   normal dev is not apples-to-apples. Prefer a job only if it is stable and
   not materially weaker than M106-family dev.
2. M119 E4 is now the top completed upside / information candidate from this
   batch: best 2% valid **0.871196**, aux m91 **0.855080**.
3. M116 E4 is the current best comparable standard-valid fallback candidate, but
   not a green score-max candidate because it is below M106 main dev.
4. Do not spend more quota on M106 internal checkpoint mining unless a new log
   signal appears; E3 lost to E4 publicly.

Default skips:

- M110/M112 even if full-data dev looks tempting.
- Any job with logloss rebound or obvious collapse.
- Any job whose only positive sign is contaminated normal-dev gain from a previously rejected branch.
