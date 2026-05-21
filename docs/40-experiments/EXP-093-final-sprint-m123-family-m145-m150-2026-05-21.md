# EXP-093 · Final sprint on public-positive M123/M119 family · M145-M150 · 2026-05-21

> **Status**: Prepared queue candidates after M144 public negative and M142 logs-only.
> The training candidate queue was empty while corrected M137 and M141 were
> still running. These jobs deliberately avoid rejected `vr015`, TIN-lite,
> UE61/87 concat, and broad feature branches.

## Trigger

Current team public SOTA remains **M123 = 0.851011**. The new target is
**0.853**, and user reports public Top1/SOTA around **0.85349**.

Recent negative calibration is strong:

- M138 `vr015 + MTL + auxselect`: public **0.849451**.
- M144 `vr015 + MTL + TIN-lite + soup`: public **0.848183**.
- M136 `UE61/87+xitem`: public **0.849015** despite very high aux.

Therefore the sprint queue should exploit only public-positive evidence:

- **M119**: `valid_ratio=0.02`, seed7 freshness-only, public **0.850754**.
- **M123**: `valid_ratio=0.02 + MTL`, seed5, public **0.851011**.
- **M124**: `valid_ratio=0.01 + MTL`, seed5, public **0.850718**.

## Prepared Jobs

| Priority | Job name | Helper | Purpose |
|---:|---|---|---|
| 1 | `M145-m123-vr02-mtl-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m145_m123_vr02_mtl_seed5_auxselect_soup.sh` | Exact M123 public SOTA recipe plus aux side checkpoints and dense-only same-run soup. |
| 2 | `M146-m119-vr02-seed7-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m146_m119_vr02_seed7_auxselect_soup.sh` | Exact M119 public-positive recipe plus aux side checkpoints and soup; no MTL. |
| 3 | `M147-m123-vr02-mtl-seed5-auxselect-soup-gap0-5ep` | `platform_run_templates/run_m147_m123_vr02_mtl_seed5_auxselect_soup_5ep.sh` | Test whether the public-positive M123 curve benefits from one more epoch while preserving E4/top-k/soup choices. |
| 4 | `M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep` | `platform_run_templates/run_m148_m119_vr02_mtl_seed7_auxselect_soup_5ep.sh` | Five-epoch extension of the corrected M141 idea; cancel/deprioritize if M141 returns clearly unhealthy before this starts. |
| 5 | `M149-m123-vr02-mtl-cw15-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m149_m123_vr02_mtl_cw15_seed5_auxselect_soup.sh` | MTL objective shape: lower conversion weight from 2.0 to 1.5 to lean more on click regularization. |
| 6 | `M150-m123-vr02-mtl-cw30-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m150_m123_vr02_mtl_cw30_seed5_auxselect_soup.sh` | MTL objective shape: raise conversion weight from 2.0 to 3.0 to favor the public conversion head. |

## Travel Buffer Jobs

User expects limited decision time on **2026-05-22** because of Spain -> San
Francisco travel. Prepared six lower-priority queue fillers so training slots
do not idle. These remain inside the public-positive family.

| Priority | Job name | Helper | Purpose |
|---:|---|---|---|
| 7 | `M151-m123-vr02-mtl-seed13-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m151_m123_vr02_mtl_seed13_auxselect_soup.sh` | Seed13 with MTL on the public-positive 2% freshness recipe; M120 seed13 freshness-only was not enough, but MTL may regularize it. |
| 8 | `M152-m123-vr01-mtl-seed7-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m152_m123_vr01_mtl_seed7_auxselect_soup.sh` | Aggressive 1% freshness with the public-positive seed7 axis; compare to M124 seed5 vr01. |
| 9 | `M153-m119-vr02-seed7-auxselect-soup-gap0-5ep` | `platform_run_templates/run_m153_m119_vr02_seed7_auxselect_soup_5ep.sh` | Five-epoch M119 no-MTL extension; keeps E4/top-k/soup available. |
| 10 | `M154-m123-vr02-mtl-seed5-lr7e5-auxselect-soup-gap0-5ep` | `platform_run_templates/run_m154_m123_vr02_mtl_seed5_lr7e5_5ep_auxselect_soup.sh` | Lower dense LR with one extra epoch, a smoother final-training variant of M123. |
| 11 | `M155-m123-vr02-mtl-seed5-sparse003-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m155_m123_vr02_mtl_seed5_sparse003_auxselect_soup.sh` | Sparse optimizer lower LR 0.03 to reduce embedding overfit. |
| 12 | `M156-m123-vr02-mtl-seed5-sparse007-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m156_m123_vr02_mtl_seed5_sparse007_auxselect_soup.sh` | Sparse optimizer higher LR 0.07 to test whether embeddings are under-adapting in the fresh tail. |

## Last-Chance UE Separate-Token Jobs

User asked whether this is the last good window to try the previously discussed
"true separate token" interpretation of the external `feat 61/87` clue. It is:
the old M133/M136 jobs concatenated fid61+fid87 into **one** UE token, so they
did not actually test two independent dense UE tokens.

Implementation note: directly adding one extra UE token would make
`T = num_queries * 4 seqs + num_ns = 17`, which violates the RankMixer
`d_model=64` divisibility constraint. Therefore these jobs use
`--user_ns_tokens 4` as a token-budget swap:

`4 user-int tokens + 2 UE dense tokens + 2 item-int tokens = 8 NS tokens`,
so `T = 8 query tokens + 8 NS tokens = 16` remains valid.

| Priority | Job name | Helper | Purpose |
|---:|---|---|---|
| 7A | `M157-m123-vr02-mtl-ue6187-septok-u4-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m157_m123_vr02_mtl_ue6187_septok_u4_seed5_auxselect_soup.sh` | Pure test of true separate fid61/fid87 UE tokens on the M123 SOTA recipe; no explicit xitem. |
| 7B | `M158-m123-vr02-mtl-ue6187-septok-xitem-a03-u4-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m158_m123_vr02_mtl_ue6187_septok_xitem_a03_u4_seed5_auxselect_soup.sh` | Same true separate tokens plus mild explicit UE x item (`alpha=0.3`) to test the complete external clue with reduced xitem risk. |

Upload each selected helper as platform `run.sh`. Include current `train.py`,
`trainer.py`, `model.py`, `infer.py`, `dataset.py`, `utils.py`,
`infer_schema.py`, and `ns_groups_vocab_tier.json`.

## Readout Rules

Default eval candidates after completion:

1. If M141 is healthy, compare M141 vs M148 first. If M141 is weak, do not let
   M148 consume an eval quota unless its 5-epoch curve strongly recovers.
2. M145 is the cleanest direct M123 upgrade; if its soup or aux/blend side
   checkpoint is not worse than M123 on primary/logloss and has stronger aux,
   it becomes a morning eval candidate.
3. M146 decides whether M119's no-MTL public strength was checkpoint-selection
   limited. It is useful even if M141 is mediocre.
4. M149/M150 are evaluated only if they improve the M123-family primary and do
   not damage aux/logloss. They are objective-shape bets, not blind quota burns.
5. M151-M156 are travel-buffer candidates. They should not displace M145-M150,
   but they are preferable to leaving the training pool idle during the flight.
6. M157/M158 are last-chance mechanism bets. Queue them after M145-M150 and
   before the lower-priority travel-buffer jobs if capacity allows. Evaluate
   only if their primary/logloss are at least M123-family healthy; do not chase
   a high aux-only result.

Stop conditions:

- Do not evaluate any `vr015` offspring by default.
- Do not evaluate TIN-lite in the current form.
- Do not evaluate UE61/87 concat or xitem offspring based only on high aux.
- Do not trust high tiny-window primary alone; require consistency with the
  public-positive M119/M123 family.
