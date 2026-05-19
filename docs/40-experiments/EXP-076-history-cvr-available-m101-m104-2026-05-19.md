# EXP-076 · History-CVR Available-Time Revisit · M101-M104 · 2026-05-19

## Context

Current public SOTA is **M84 = 0.848854**. Remaining gap to 0.852 is
**+0.003146**.

After M93-M100, all clean-window gates failed:

| Branch | Clean result | Decision |
|---|---:|---|
| delayed-feedback loss weighting | M93 **0.762562**, M91 baseline **0.769513** | reject |
| seqstats | M97 **0.760092**, M91 baseline **0.769513** | reject |
| Beijing time closure | M99 **0.764915**, M100 **0.763198**, M91 **0.769513** | reject |

Training pool is now empty and 5/19 has **4/4** evaluation quotas remaining.
Because ordinary RG dev is contaminated by time overlap, new score-max attempts
should have a clean-window companion whenever possible.

## Hypothesis

Old P0-C history CVR failed in EXP-009 on an earlier backbone and the default
timestamp-cutoff semantics. That evidence still matters; this is a **high-risk
revisit**, not a clean green-field direction.

The new reason to revisit is that current code supports
`history_cvr_time_mode=available`: positives enter history at `label_time`, and
negatives enter after `timestamp + negative_maturity_sec`. This tests delayed
feedback as **feature availability** rather than sample weighting. M93 showed
loss weighting is harmful; available-time history may still capture item-side
future-window priors without using validation/test labels.

## Jobs

Use current source bundle. Upload each local script as platform `run.sh`; do not
submit the generated `platform_run_templates/run_*.sh` templates to git.

| Job | Local template | Purpose |
|---|---|---|
| `M101-cleanwin0303-m84-hcvr-avail-gap0-seed42-4ep` | `platform_run_templates/run_m101_cleanwin0303_m84_hcvr_avail.sh` | Clean-window diagnostic for available-time history CVR |
| `M102-m84-hcvr-avail-gap0-seed42-4ep` | `platform_run_templates/run_m102_m84_hcvr_avail.sh` | Full-data candidate if M101 passes |
| `M103-cleanwin0303-m84-hcvr-ts24h-gap0-seed42-4ep` | `platform_run_templates/run_m103_cleanwin0303_m84_hcvr_ts24h.sh` | Timestamp-cutoff control on clean window |
| `M104-m84-hcvr-ts24h-gap0-seed42-4ep` | `platform_run_templates/run_m104_m84_hcvr_ts24h.sh` | Full-data candidate if M103 passes |

Shared base: M84 stack, corrected T38 per-domain buckets, path-B UE split,
`batch_size=128`, `num_epochs=4`, `gap0`, `seed42`.

Key flags:

| Branch | History semantics | Flags |
|---|---|---|
| M101/M102 | available-time | `--enable_history_cvr_features --history_cvr_time_mode available --history_cvr_available_lag_sec 0 --negative_maturity_sec 86400` |
| M103/M104 | 24h timestamp cutoff | `--enable_history_cvr_features --history_cvr_time_mode timestamp_cutoff --history_cvr_cutoff_sec 86400 --negative_maturity_sec 86400` |

Clean-window diagnostics use the same pseudo-public window as M91:

- `--fixed_valid_timestamp_min 1772502591`
- `--fixed_valid_timestamp_max 1772520547`

## Gates

| Gate | Promote condition | Action |
|---|---|---|
| M101 vs M91 | M101 - M91 >= **+0.0004** | consider evaluating M102 if normal dev/logloss are sane |
| M103 vs M91 | M103 - M91 >= **+0.0004** | consider evaluating M104 if normal dev/logloss are sane |
| both clean <= M91 | close history-CVR revisit | no public eval |
| clean slightly negative but full dev huge | only one deliberate information quota | prefer stronger clean/full consistency |

M91 clean baseline is **0.769513**. Promotion threshold is therefore
**0.769913**.

## Eval Policy

Do **not** spend the refreshed 5/19 quota on M93-M100 by default. Wait for
M101/M103 clean gates before using score-max quota on M102/M104.

If we need one calibration eval before this batch returns, M95 is the only
normal-dev temptation, but M97 failed by **-0.009421** vs M91. Treat that as a
calibration burn, not a rational score-max attempt.

## Risks

- EXP-009 already rejected old P0-C, so prior probability is modest.
- History sidecar building may increase startup time and checkpoint size.
- If both clean diagnostics fail, this branch should be closed quickly.

## Submission Status

User reported that **M101-M104 have all been submitted** to the platform on
2026-05-19. Await completion logs.

## M101/M103 Clean Results

Source: user copied `~/Downloads/train_log.txt` after M101/M103 completed.

| Job | Branch | Best clean AUC | LogLoss | Best epoch | Delta vs M91 | Gate |
|---|---|---:|---:|---:|---:|---|
| `M101-cleanwin0303-m84-hcvr-avail-gap0-seed42-4ep` | available-time | **0.769522861** | 0.215036 | E4 | **+0.000010** | yellow |
| `M103-cleanwin0303-m84-hcvr-ts24h-gap0-seed42-4ep` | timestamp-cutoff 24h | **0.769340533** | 0.214394 | E4 | **-0.000172** | yellow |

Reference:

- M91 clean baseline: **0.769513**.
- Green promotion threshold: **0.769913**.
- Red threshold from the practical rule: clean delta < **-0.0003**.

Readout:

- Neither branch clears the green threshold.
- Neither branch collapses like time/seqstats/delayed-feedback diagnostics.
- Available-time is slightly better than timestamp-cutoff on AUC, while
  timestamp-cutoff has slightly lower logloss.
- Classification: **neutral/yellow**, not eval-ready by default.

Decision:

- Do **not** evaluate M102/M104 yet.
- Wait for full-data M102/M104 training results.
- If M102 or M104 full-data dev is very strong (for example, at least around
  M84 +0.001) and logloss is sane, spend at most one information quota.
- If both full-data candidates are ordinary, close the history-CVR revisit.

## Clean-Window Caveat

The M91-style clean window should be used as a guardrail, not a perfect oracle.
It has already reproduced the M86 public miss and exposed normal-dev illusions
for time and seqstats branches, so it is much better than raw RG dev for
shift-sensitive mechanisms.

But it can still miss a good model when:

- the mechanism depends strongly on the latest full training day;
- the previous-day 09:49-14:49 slice is anomalous;
- the clean job's smaller training horizon changes model capacity use or
  sidecar statistics;
- full-data dev gain is large and clean-window loss is only marginal.

Practical decision rule for M101-M104:

| Zone | Condition | Action |
|---|---|---|
| green | clean delta >= +0.0004 | evaluate matching full candidate if normal dev/logloss sane |
| yellow | clean delta in [-0.0003, +0.0004) and full dev >= M84 +0.001 | consider one information quota |
| red | clean delta < -0.0003 or full dev weak | no eval unless explicitly buying calibration |

## M102 Full-Data Result

Source: user copied `~/Downloads/train_log.txt` after M102 completed.

Job: `M102-m84-hcvr-avail-gap0-seed42-4ep`.

Configuration confirmed:

- M84 corrected-T38 stack.
- `--enable_history_cvr_features`.
- `--history_cvr_time_mode available`.
- `--history_cvr_available_lag_sec 0`.
- `--negative_maturity_sec 86400`.
- `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:512`.
- No auxiliary validation windows were logged in this run.

Training and validation:

| Epoch | Step | Dev AUC | LogLoss | Train avg loss |
|---:|---:|---:|---:|---:|
| 1 | 15941 | 0.866281 | 0.224713 | 0.233133 |
| 2 | 31882 | 0.869616 | 0.222154 | 0.228691 |
| 3 | 47823 | 0.871107 | 0.221147 | 0.227328 |
| 4 | 63764 | **0.871458** | 0.221240 | 0.226203 |

Best checkpoint:

- `global_step63764.layer=2.head=4.hidden=64.auc=0.871458.best_model`
- `best_val_AUC=0.871457943702728`
- `best_val_logloss=0.22124020755290985`
- `stop_reason=max_epochs_completed`

Readout:

- M102 normal dev is about **+0.000968 vs M84** (`0.871458 - 0.870490`),
  essentially at the prior yellow-zone threshold of "around M84 +0.001".
- M101 clean-window paired diagnostic was neutral, not green:
  **0.769523**, only **+0.000010 vs M91**.
- The combination is therefore **information-eval worthy but not
  score-max certain**.
- Because EDA73 showed M91 should not be the sole oracle, M101's neutral result
  should not automatically veto M102.

Decision:

- If there are at least three eval quotas left, M102 is a rational candidate
  for **one information quota**, preferably after checking whether M104 or
  M105/M106 returns a cleaner signal before the next planned eval window.
- If quota pressure rises, prefer holding until M104/M105/M106 are known.
- If evaluated, use the E4 best checkpoint
  `global_step63764.layer=2.head=4.hidden=64.auc=0.871458.best_model`.

## M104 Full-Data Result

Source: user copied `~/Downloads/train_log.txt` after M104 completed.

Job: `M104-m84-hcvr-ts24h-gap0-seed42-4ep`.

Configuration confirmed:

- M84 corrected-T38 stack.
- `--enable_history_cvr_features`.
- `--history_cvr_time_mode timestamp_cutoff`.
- `--history_cvr_cutoff_sec 86400`.
- `--negative_maturity_sec 86400`.
- `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:512`.
- No auxiliary validation windows were logged in this run.

Training and validation:

| Epoch | Step | Dev AUC | LogLoss | Train avg loss |
|---:|---:|---:|---:|---:|
| 1 | 15941 | 0.866468 | 0.224555 | 0.233092 |
| 2 | 31882 | 0.869871 | 0.222080 | 0.228656 |
| 3 | 47823 | 0.871127 | 0.221192 | 0.227434 |
| 4 | 63764 | **0.871796** | **0.221065** | 0.226275 |

Best checkpoint:

- `global_step63764.layer=2.head=4.hidden=64.auc=0.871796.best_model`
- `best_val_AUC=0.8717960820806095`
- `best_val_logloss=0.2210647016763687`
- `stop_reason=max_epochs_completed`

Readout:

- M104 normal dev is about **+0.001306 vs M84**
  (`0.871796 - 0.870490`).
- It is **+0.000338 vs M102** and has slightly better logloss.
- The paired clean-window diagnostic M103 was not green:
  **0.769341**, **-0.000172 vs M91**.
- This is therefore stronger than M102 on full-data dev, but weaker than M102
  on the paired clean-window signal.

Decision:

- If spending one history-CVR information quota now, prefer **M104 over M102**
  because it has the stronger full-data AUC and logloss.
- This is still **yellow-plus**, not green: the clean-window companion did not
  validate the gain.
- If M105/M106 return soon, compare against them before spending quota.
- If evaluated, use the E4 best checkpoint
  `global_step63764.layer=2.head=4.hidden=64.auc=0.871796.best_model`.

## M104 Public Evaluation Result

Source: user reported the leaderboard result after evaluation.

Evaluation job:

- `M104-m84-hcvr-ts24h-gap0-seed42-4ep_eval_1779199262757`

Result:

| Candidate | Public AUC | Time | Delta vs M84 | Delta vs M89 |
|---|---:|---:|---:|---:|
| M104 | **0.847560** | 644.81s | **-0.001294** | -0.000090 |

Readout:

- M104 did **not** transfer despite normal dev **0.871796**
  (**+0.001306 vs M84**).
- The paired M103 clean-window warning was directionally correct:
  M103 was **-0.000172 vs M91**, not a green pass.
- This result rejects timestamp-cutoff history CVR as a score-max branch under
  the current implementation.
- Because M102 has weaker full-data dev than M104 and only neutral clean
  support, M102 should not receive scarce public quota by default.

Decision:

- Close M104 for score-max.
- Do **not** evaluate M102 unless all stronger candidates fail and the quota
  would otherwise expire.
- If M107/M108 `clk0302` diagnostics were already submitted, let them finish for
  validation-system learning. If not submitted yet, downgrade them to optional:
  useful but no longer urgent.
