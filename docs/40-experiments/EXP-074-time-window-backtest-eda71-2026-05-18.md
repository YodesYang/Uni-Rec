# EXP-074 · Time-Window Backtest Audit · EDA-71 · 2026-05-18

## Trigger

After M86 public came back at **0.847106** despite dev **0.871415**, we revisited whether the time direction is exhausted.

Key correction: do **not** launch a new hour-only training job. Old M40 already tested `--enable_hour_only_features` and got LB **0.846931**; EDA-056 also argued DOW/weekend carries real signal rather than pure noise.

## Current Read

Time work is not exhausted, but the naive variants are now low-confidence:

- Hard time reweight: M82 public **0.846197**, M83 dev bad, M87 dev bad.
- Rich absolute time dense stack: M86 dev good but public **-0.001748 vs M84**.
- Hour-only removal: M40 LB **0.846931**, not worth relaunching.

The unresolved question is not "add more hour/date flags"; it is whether we can build a validation/backtest view that matches the public test window.

## M86 Eval EDA Decode

Source: `~/Downloads/eval_log.txt` from M86 eval. The platform Logs UI printed the blob in reverse chronological order, so the base64/gzip chunks were reversed before decoding.

Decoded blob:

- `eda_format=test_eda_compact_v1`
- `total_samples=523244`
- `seq_item_id_top_k_limit=200`
- `user_id.counter_cap=200000`
- `user_id.distinct_capped=200000`
- `user_id.freq_hist_top20={"1": 200000}`

Interpretation: within the 200k cap, all observed test users are singleton-like. This makes user memorization / user-frequency features low-value for public; item/sequence and time-window generalization matter more.

Test absolute timestamp quantiles:

| Quantile | UTC epoch | Beijing time |
|---|---:|---|
| min | 1772588991 | 2026-03-04 09:49:51 |
| p1 | 1772591239 | 2026-03-04 10:27:19 |
| p5 | 1772594958 | 2026-03-04 11:29:18 |
| p25 | 1772597699 | 2026-03-04 12:14:59 |
| p50 | 1772600248 | 2026-03-04 12:57:28 |
| p75 | 1772603021 | 2026-03-04 13:43:41 |
| p95 | 1772605696 | 2026-03-04 14:28:16 |
| p99 | 1772606499 | 2026-03-04 14:41:39 |
| max | 1772606947 | 2026-03-04 14:49:07 |

Sequence length summary on test:

| Domain | mean | p1 | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| seq_a | 218.2 | 13 | 214 | 256 | 256 | 256 | 256 |
| seq_b | 199.0 | 2 | 136 | 256 | 256 | 256 | 256 |
| seq_c | 348.7 | 34 | 210 | 373 | 512 | 512 | 512 |
| seq_d | 455.8 | 0 | 512 | 512 | 512 | 512 | 512 |

Readout:

- Test is a narrow, afternoon-weighted slice of the public day, not a broad 09~14 uniform distribution.
- It is not enough to learn "03-04" or "09~14"; the model must generalize from pre-09:49 train rows or prior-day same-clock windows into **03-04 09:49~14:49**.
- Seq saturation is high, especially `seq_d`; if EDA-71 shows time window mismatch, a next branch could target time-window validation plus sequence recency / truncation diagnostics rather than more dense calendar flags.

## EDA-71 Design

Public test window observed from eval logs:

- Beijing: **2026-03-04 09:49:51 -> 14:49:07**
- UTC: **2026-03-04 01:49:51 -> 06:49:07**

Script:

- `src/eda_time_window_backtest.py`
- platform entry uploaded as `run.sh`; local `run_*.sh` templates are not
  committed

It prints log-only tables:

- train timestamp range and exact public-window overlap rows
- RG-level `valid_split_strategy=time` overlap diagnostic
- date/hour/daypart CVR summaries
- same-clock public-window rows and CVR by train date
- final flags:
  - `target_clock_window_train_rows`
  - `pseudo_public_windows_ge_10000_rows`
  - `rg_time_split_overlap_sec`
  - `recommend_clean_window_backtest`
  - `recommend_more_absolute_time_dense=false`

## Platform Job

Job name:

`EDA71-time-window-backtest-audit`

Files to upload:

- platform entry file as `run.sh`
- `src/eda_time_window_backtest.py`

No model files are needed. This task does not consume evaluation quota, only one training/EDA slot.

## EDA-71 Result

Platform log source: `~/Downloads/train_log.txt`, completed with `user exit code: 0`.

Overall train scan:

- files/RGs/rows: **1000 files / 1044 RGs / 2,099,956 rows**
- train CVR: **213,518 / 2,099,956 = 0.101677**
- train range BJ: **2026-02-28 08:35:09 -> 2026-03-04 09:49:50**
- public test window BJ: **2026-03-04 09:49:51 -> 14:49:07**
- `TRAIN_MAX_TO_PUBLIC_START_SEC=1`
- `EXACT_PUBLIC_WINDOW_ROWS_IN_TRAIN=0`

RG split diagnostic:

- `valid_ratio=0.05`
- train rows **1,990,981**, valid rows **108,975**
- RG train BJ range: **2026-02-28 08:35:09 -> 2026-03-04 09:49:18**
- RG valid BJ range: **2026-02-28 11:04:32 -> 2026-03-04 09:49:50**
- `rg_overlap_sec=341086` (~**3.95 days**)

The current `valid_split_strategy=time` is therefore not a clean future-window
split. RGs span multiple days, so the validation tail still overlaps most of
the train time range. This explains why M86 can improve dev and lose public.

Date distribution:

| BJ date | rows | row pct | CVR |
|---|---:|---:|---:|
| 2026-02-28 | 2,367 | 0.113% | 0.036755 |
| 2026-03-01 | 133,201 | 6.343% | 0.025150 |
| 2026-03-02 | 121,791 | 5.800% | 0.116577 |
| 2026-03-03 | 290,757 | 13.846% | 0.082048 |
| 2026-03-04 | 1,551,840 | 73.899% | 0.110854 |

Public-clock window by train date:

| BJ date | rows | row pct | window CVR | day CVR | pre-window CVR |
|---|---:|---:|---:|---:|---:|
| 2026-02-28 | 6 | 0.000% | 0.000000 | 0.036755 | 0.000000 |
| 2026-03-01 | 2,156 | 0.103% | 0.022263 | 0.025150 | 0.024918 |
| 2026-03-02 | 26,404 | 1.257% | 0.148803 | 0.116577 | 0.043554 |
| 2026-03-03 | 29,964 | 1.427% | 0.069083 | 0.082048 | 0.070922 |

Final flags:

- `target_clock_window_train_rows=0`
- `pseudo_public_windows_ge_10000_rows=2`
- `recommend_clean_window_backtest=true`
- `recommend_more_absolute_time_dense=false`

## Decision

EDA-71 rejects launching more naive absolute-time dense variants by dev score.
The public window is literally one second after train max, and the supervised
train set has no target-day rows in that window. Time features must be judged
by clean pseudo-public windows before they deserve evaluation quota.

Concrete follow-up prepared:

- add fixed timestamp validation-window support to `train.py` / `dataset.py`
- M91: M84 stack with clean 2026-03-03 09:49:51~14:49:07 valid window
- M92: M86 stack with the same clean valid window

Use M91 vs M92 clean-window dev delta as the next time-feature decision gate.
Do **not** evaluate M89/M90 solely because normal dev improves; M86 showed the
normal dev split is contaminated for this question.

## Follow-Up Submitted

User submitted both clean-window diagnostics:

- `M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep`
- `M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep`

Decision readout when logs return:

- If M92 > M91 by a clear margin on the fixed window, time features may still
  transfer when judged on a pseudo-public split; M89/M90 can receive eval
  consideration.
- If M92 <= M91, the M86 public miss is likely structural; stop spending LB
  quota on UTC+Beijing dense time stacks and move to non-time/ensemble paths.

## M91/M92 Result

Source: user pasted `~/Downloads/train_log.txt`; M92 final checkpoint table was
supplemented manually because the log copy was incomplete.

| Job | Clean-window config | Best AUC | Best ckpt |
|---|---|---:|---|
| `M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep` | M84 stack on BJ 2026-03-03 09:49:51~14:49:07 | **0.769513** | `global_step10136...auc=0.769513.best_model` |
| `M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep` | M86 UTC+Beijing time stack on the same window | **0.766748** | `global_step10136...auc=0.766748.best_model` |

Delta:

- M92 - M91 = **-0.002765**

Decision:

- Gate 1 fails hard.
- This pseudo-public split reproduces the M86 public failure mode, so the issue
  is not just one unlucky public evaluation.
- Do **not** spend scarce eval quota on M89/M90 solely because normal dev is
  high; normal dev is contaminated for dense time-feature decisions.
- Time work should shift away from adding more absolute-time encodings and
  toward delayed feedback, cleaner validation, sequence/activity features, or
  ensembling.

## M99/M100 Time-Feature Closure

Source: user pasted `~/Downloads/train_log.txt` after M93-M100 completed.

Both jobs use the same clean pseudo-public BJ window as M91/M92:

| Job | Clean-window config | Best AUC | Delta vs M91 | Decision |
|---|---|---:|---:|---|
| `M99-cleanwin0303-m84-bjtime-v2-gap0-seed42-4ep` | M89-style Beijing v1+v2, no UTC time | **0.764915** | **-0.004598** | fail |
| `M100-cleanwin0303-m84-bjtime-gap0-seed42-4ep` | M85-style Beijing-only | **0.763198** | **-0.006315** | fail |

Conclusion: the cleaner time-feature variants also fail the pseudo-public gate.
M89 and M85 should not receive default eval quota despite their normal-dev
signals; any eval would be a deliberate high-risk calibration probe.

## M89 Calibration Result

User spent one 5/19 quota to test whether the clean-window gate was too
pessimistic for M89.

| Full candidate | Normal dev | Clean proxy | Public AUC | Decision |
|---|---:|---:|---:|---|
| M89 Beijing v1+v2, no UTC | **0.871864** | M99 **0.764915** (**-0.004598 vs M91**) | **0.847650** | reject |

M89 is **-0.001204 vs M84 public 0.848854**. This confirms the clean-window
diagnostic direction for the time-feature branch: the normal-dev gain did not
transfer. The clean window remains imperfect, but after M86 and M89 it should
be trusted strongly for dense time-feature allocation decisions.
