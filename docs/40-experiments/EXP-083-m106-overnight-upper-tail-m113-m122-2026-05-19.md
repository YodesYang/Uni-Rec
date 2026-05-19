# EXP-083 · M106-family overnight upper-tail and MTL batch · 2026-05-19

> **Status**: all ten M113-M122 jobs have been started or entered the platform queue.
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

Tomorrow's public quota should be spent in this order:

1. Among M113-M116, compare to M106 E4 by normal dev, logloss, and aux m91. Promote only if the candidate is clearly at or above M106.
2. For M117-M120, use public-adjacent rationale plus sanity checks because normal dev is not apples-to-apples. Prefer a job only if it is stable and not materially weaker than M106-family dev.
3. Promote M121/M122 only if conversion dev is competitive and no click-head pathology appears in logs.
4. Do not spend more quota on M106 internal checkpoint mining unless a new log signal appears; E3 lost to E4 publicly.

Default skips:

- M110/M112 even if full-data dev looks tempting.
- Any job with logloss rebound or obvious collapse.
- Any job whose only positive sign is contaminated normal-dev gain from a previously rejected branch.
