# EXP-093 · Final sprint on public-positive M123/M119 family · M145-M150 · 2026-05-21

> **Status**: M148 evaluated at public **0.851365**, a new team SOTA but not a
> 0.853-sized jump. M151 public **0.850129** rejected the seed13+MTL high-aux
> signal, and M147 public **0.849379** rejected the M123 seed5 5ep signal.
> M148 dense-only `weight_soup` later evaluated at public **0.850563**, rejecting
> soup as a default score-max side checkpoint for this branch.
> M154/M155/M157 later returned; M154 is numerically interesting but it is a
> close M147/M123-seed5-5ep sibling and should not displace M148-lineage
> candidates by default. M159 later returned as a healthy M148-lineage lower-LR
> run but public eval was **0.851279**, slightly below M148. M165 then evaluated
> at **0.851162**, lower than both M148 and M159. M161/M163/M167/M168 later
> returned but are no-eval by default.

## Trigger

Current team public SOTA remains **M148 = 0.851365**. The target is to clear
**0.852** first, then chase **0.853+**; user reports public Top1/SOTA around
**0.85401**.

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

## Returned Results · M145-M152/M158

User reported that all latest jobs except **M153-M157** returned, then later
reported **M153/M156** completed and copied logs to `~/Downloads/train_log.txt`.
Parsed summaries:

| Job | Best epoch | Primary AUC | LogLoss | Aux m91 | Soup | Readout |
|---|---:|---:|---:|---:|---|---|
| M145 `M123+selector+soup` | E4 | 0.870540 | 0.221967 | 0.853691 | 2 members | Healthy rerun, but slightly below M123; hold. |
| M146 `M119+selector` | E4 | 0.871047 | 0.221636 | 0.855842 | skipped | Low-risk eval candidate; direct M119 public-positive branch. |
| M147 `M123 5ep` | E5 | 0.870843 | 0.221623 | 0.857534 | 2 members | Strong candidate; longer training helped this freshness+MTL branch. |
| M148 `M119+MTL 5ep` | E5 | **0.871378** | 0.221600 | 0.857345 | 2 members | Best first eval candidate from the returned batch. |
| M149 `M123 cw1.5` | E4 | 0.870833 | 0.221880 | 0.853163 | 2 members | Primary ok, aux weaker; hold. |
| M150 `M123 cw3.0` | E4 | 0.870717 | 0.222292 | 0.854515 | 2 members | Not ahead of M148/M147/M146 after M151 public miss. |
| M151 `M123 seed13+MTL` | E4 | 0.870969 | **0.221344** | 0.857198 | 2 members | Public 0.850129; high-aux/logloss false positive. |
| M152 `vr01 seed7+MTL` | E4 | 0.868962 | 0.226037 | 0.853201 | 2 members | Too weak; no default eval. |
| M153 `M119 no-MTL 5ep` | E4 | 0.870980 | 0.221421 | 0.854488 | 3 members | E5 raises aux to 0.856338 but primary drops; backup only. |
| M154 `M123 lr=7e-5 5ep` | E5 | 0.870880 | 0.221613 | **0.858290** | 3 members | Highest aux, but close M147 sibling after M147 public miss; risky information candidate only. |
| M155 `M123 sparse_lr=0.03` | E4 | 0.870171 | 0.222290 | 0.854241 | 2 members | Weaker than M123/M148; no default eval. |
| M156 `M123 sparse_lr=0.07` | E4 | 0.870717 | 0.221962 | 0.854368 | 2 members | Healthy but lower than top candidates; hold. |
| M157 `septok 61/87 no xitem` | E4 | 0.869781 | 0.222716 | 0.851257 | 2 members | True separate-token pure test is weak; no default eval. |
| M158 `septok 61/87+xitem a0.3` | E4 | 0.870462 | 0.222107 | 0.853288 | skipped | Separate-token+xitem did not beat M123-family anchors; wait for pure M157. |
| M159 `M148 lr=7e-5 seed7 5ep` | E5 | 0.871099 | 0.221668 | **0.859701** | disabled | Public 0.851279; near miss below M148, so low-LR/high-aux does not become SOTA. |
| M161 `M148 sparse_lr=0.07` | E5 | 0.870414 | 0.221893 | 0.858332 | disabled | Aux ok but primary too low; no default eval. |
| M163 `M148 cw=3.0` | E5 | 0.870217 | 0.221840 | 0.855860 | disabled | Conversion-weight increase is weak; no eval. |
| M165 `M148 exact 6ep` | E5 | 0.871193 | **0.221476** | 0.857794 | disabled | Public 0.851162; below M148/M159, so exact/continued M148-lineage does not solve the gap. |
| M167 `septok 61/87+xitem a0.3 seed7` | E5 | 0.870154 | 0.221767 | 0.856182 | disabled | M148-lineage separate-token+xitem remains weak; no eval. |
| M168 `M148 seed42 5ep` | E5 | 0.871011 | **0.221326** | 0.852665 | disabled | Healthy seed42 rerun with strong logloss, but primary trails M148/M165/M159 and aux is far below M159; no default eval. |

Recommended public-eval order if quota is available:

1. After M159 public **0.851279** and M165 public **0.851162**, do not spend
   more quota on returned local M148-lineage variants unless no stronger
   candidate returns before quota expiry.
2. Prefer later M148-lineage optimizer/objective candidates only if they return
   clearly stronger than M159/M165/M148 (M160/M162/M164/M166 or M170-M173 when
   healthy).
3. Do not evaluate M168 by default; its logloss is attractive, but M151/M159/M165
   show that single-metric wins without a balanced public-positive signature can
   be public-negative.
4. If quota would otherwise expire and only this batch is available, M154 normal
   `best_model` is the only new information candidate, but expected score-max
   value is limited by M147's public failure.
5. Otherwise, `M146` normal `best_model` at `global_step65920` remains the
   low-risk no-MTL calibration candidate; M153 normal `best_model` at
   `global_step65920` is lower priority.

M153 does not enter the top four. Its normal best-model checkpoint is E4
`global_step65920`, but that is only a slightly lower-primary/lower-aux rerun of
the M146 no-MTL branch; its E5 aux/blend side checkpoint is not preferred
because primary/logloss move backward.

Use normal best-model checkpoints before dense-only soup for the first public
pass. The soup checkpoints are useful side artifacts, but no post-soup internal
validation is computed; after M144's public miss, they should not outrank the
matching best-model checkpoint unless a quota would otherwise expire or a
matching best-model public result is already known.

Interpretation:

- The current viable lane remains **freshness + MTL**, not UE/TIN/history/time.
- Five epochs are not globally safe, but in the fresh `vr02` M123/M119 family
  both M147 and M148 improved primary, logloss, and aux at E5.
- High aux alone is insufficient after M136, but high aux plus healthy primary
  and logloss inside the proven M119/M123 family is the best current evidence.
- M158 suggests the explicit UE61/87 separate-token + xitem clue is still not
  ready to spend quota on. M157 later confirms the pure separate-token version
  is also weak in this M123 seed5 form.

## M148 Public Eval

User reported:

`M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep_eval_1779482611683`

Public AUC: **0.851365**.

This is a new team SOTA:

| Anchor | Public AUC | M148 delta |
|---|---:|---:|
| M123 `vr02+MTL` seed5 | 0.851011 | **+0.000354** |
| M119 `vr02` seed7 | 0.850754 | **+0.000611** |
| M124 `vr01+MTL` seed5 | 0.850718 | **+0.000647** |
| M106 corrected-T38 seed5 | 0.849364 | **+0.002001** |

Interpretation:

- The fresh `valid_ratio=0.02` + seed7 + MTL + 5ep branch transfers to public.
- The gain is real but small; it reduces the 0.853 gap to **+0.001635**.
- M148 validates continuing the public-positive freshness/MTL family, but does
  not validate spending scarce quota on soup or aux side checkpoints first.
- M151 and M147 later rejected the high-aux/high-logloss and M123-5ep selection
  signals. M148 remains the only positive result in this sprint batch so far.

## M148 Weight-Soup Public Eval

User reported:

`M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep-weight_soup_eval_1779486427228`

Public AUC: **0.850563** (690.98s).

Public deltas:

| Anchor | Public AUC | M148 weight_soup delta |
|---|---:|---:|
| M148 best_model | 0.851365 | **-0.000802** |
| M123 `vr02+MTL` seed5 | 0.851011 | **-0.000448** |
| M119 `vr02` seed7 | 0.850754 | **-0.000191** |

Interpretation:

- Dense-only same-run averaging did not stabilize this branch; it diluted the
  best public checkpoint.
- For final-day M148-lineage jobs, keep soup as a cheap side artifact only.
- Default public eval target becomes the normal `best_model`; evaluate soup only
  after a matching normal checkpoint validates the branch and quota is abundant.
- Later final-day M148-lineage helper templates disable `--enable_weight_soup`
  to save post-train time and keep the checkpoint list cleaner.
- An exact M148 rerun is reasonable only as a low-risk queue filler: prior
  exact/rerun evidence shows small variance, so it should not displace
  first-tier M148-lineage perturbations, but it can be useful if overnight
  capacity would otherwise idle.

## M151 Public Eval

User reported:

`M151-m123-vr02-mtl-seed13-auxselect-soup-gap0-4ep_eval_1779483206670`

Public AUC: **0.850129** (308.56s).

Public deltas:

| Anchor | Public AUC | M151 delta |
|---|---:|---:|
| M148 `vr02+MTL` seed7 5ep | 0.851365 | **-0.001236** |
| M123 `vr02+MTL` seed5 | 0.851011 | **-0.000882** |
| M119 `vr02` seed7 | 0.850754 | **-0.000625** |
| M124 `vr01+MTL` seed5 | 0.850718 | **-0.000589** |
| M106 corrected-T38 seed5 | 0.849364 | **+0.000765** |

Interpretation:

- Seed13+MTL was a validation false positive.
- Strong aux m91 (**0.857198**) and strong logloss (**0.221344**) are not
  sufficient selection signals even inside the public-positive freshness family.
- This downgrades seed rerolls and high-aux-only reasoning.
- M147 later showed that the same seed5 M123 recipe with one extra epoch is
  public-negative. Use **M146 best_model** only as no-MTL calibration if quota
  would otherwise expire. After later readouts, **M159 best_model** is the
  preferred successor candidate.
- Do not evaluate M151 soup or aux side checkpoints.

## M147 Public Eval

User reported:

`M147-m123-vr02-mtl-seed5-auxselect-soup-gap0-5ep_eval_1779484630724`

Public AUC: **0.849379** (683.46s).

Public deltas:

| Anchor | Public AUC | M147 delta |
|---|---:|---:|
| M148 `vr02+MTL` seed7 5ep | 0.851365 | **-0.001986** |
| M123 `vr02+MTL` seed5 | 0.851011 | **-0.001632** |
| M119 `vr02` seed7 | 0.850754 | **-0.001375** |
| M151 `vr02+MTL` seed13 | 0.850129 | **-0.000750** |
| M106 corrected-T38 seed5 | 0.849364 | **+0.000015** |

Interpretation:

- M123 seed5 + MTL + 5ep is public-negative.
- The E5 primary/aux/logloss gains were false positives for this branch.
- Close the general "5ep helps freshness+MTL" assumption. M148's positive result
  is specific to its seed7/M119+MTL trajectory, not a reusable longer-training
  rule.
- Do not evaluate M147 soup or aux side checkpoints.
- Remaining returned candidate M146 is now mostly a calibration/fallback, not a
  strong 0.853 path. Later M159 supersedes M154/M155/M157 as the preferred
  current eval candidate.
