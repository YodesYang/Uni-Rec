# EXP-075 · 5/20 Breakthrough Protocol · M93-M96 · 2026-05-18

## Goal

Reach **0.852 by 2026-05-20** from current SOTA **M84 public 0.848854**.
Remaining gap: **+0.003146**.

The working belief after EDA-71:

- ordinary dev is contaminated for time-feature decisions;
- public is a future window starting **1 second** after train max;
- the next jump must come from either delayed-feedback correction, better
  public-window validation, or low-correlated sequence/activity features;
- evaluation quota must be spent only after a candidate passes an explicit
  gate.

## Protocol

Use three evidence layers:

| Layer | Purpose | Gate |
|---|---|---|
| normal dev | catches obvious failures and ckpt quality | must not collapse vs M84 family |
| clean-window dev | tests future-window transfer | compare against M91 clean M84 baseline |
| public LB | scarce final truth | only spend when a gate passes |

## Gate 1 · Time Feature Transfer

Existing jobs:

- M91 = M84 stack on clean 2026-03-03 09:49:51~14:49:07 valid window.
- M92 = M86 UTC+Beijing time stack on the same clean window.

Decision:

- M92 - M91 **>= +0.0005**: time features may transfer; M89/M90 can be
  considered for evaluation if their normal dev is not suspicious.
- M92 - M91 in **[-0.0002, +0.0005)**: time features are weak/ambiguous; do
  not prioritize scarce quota.
- M92 - M91 **< -0.0002**: kill UTC+Beijing dense time stack for public score.

Result:

| Job | Best clean-window AUC |
|---|---:|
| M91 | **0.769513** |
| M92 | **0.766748** |

Gate 1 result: **failed hard**. M92-M91 = **-0.002765**, far below -0.0002.
Hold M89/M90 by default; M89 remains a high-risk information probe only because
its feature stack is not identical to M92.

M90 follow-up: M86 extended to 6 epochs also failed to rescue the mechanism.
Best normal dev is **0.871206** at E4, below M86's **0.871415**, with E5
**0.870696** and E6 **0.870880**. This removes the "maybe the time stack simply
needed longer training" explanation.

Two closure diagnostics are prepared for the overnight queue:

| Job | Role | Gate |
|---|---|---|
| `M99-cleanwin0303-m84-bjtime-v2-gap0-seed42-4ep` | M89-style Beijing v1+v2 without UTC on the clean window | compare vs M91 |
| `M100-cleanwin0303-m84-bjtime-gap0-seed42-4ep` | M85-style Beijing-only on the clean window | compare vs M91 |

Use them as information gates only:

- M99 - M91 **>= +0.0004**: M89 can be reconsidered as a high-risk information
  eval if stronger M93/M95 gates are absent.
- M100 - M91 **>= +0.0004**: M85 can be considered for mechanism learning, not
  score-max.
- If both are <= M91, close the XHS time-feature path for quota allocation.

Final closure results from `~/Downloads/train_log.txt`:

| Job | Best clean-window AUC | Delta vs M91 | Decision |
|---|---:|---:|---|
| M99 | **0.764915** | **-0.004598** | fail; do not evaluate M89 by default |
| M100 | **0.763198** | **-0.006315** | fail; do not evaluate M85 by default |

Gate 1 closure result: the time-feature path is now rejected for default quota
allocation. M89 remains a possible high-risk LB calibration only because its
normal dev is strong, not because the clean-window evidence supports it.

## Gate 2 · Delayed Feedback

Hypothesis: labels near train max contain immature negatives. Because public
starts one second after train max, the model may be trained to underpredict the
same near-future traffic it must rank in public.

Use a narrow, conservative correction:

- `negative_maturity_sec=21600` (6 hours)
- `immature_negative_weight=0.3`

Prepared jobs:

| Job | Platform entry | Role |
|---|---|---|
| `M93-cleanwin0303-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template untracked | diagnostic: compare vs M91 on clean window |
| `M94-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template untracked | full-data public candidate if M93 passes |

Decision:

- M93 - M91 **>= +0.0004**: delayed feedback is live; evaluate M94 if normal
  dev is not much worse than the M84 family.
- M93 - M91 **< 0**: do not evaluate M94; consider a softer weight only if all
  other lines fail.

### Final Readout · 2026-05-19 Morning

Source: user copied `~/Downloads/train_log.txt` after M93-M100 completed. The
M93 log now includes the final top-k summary, so the previous E3 checkpoint
label ambiguity is resolved in practice.

M93 clean-window diagnostic:

| Epoch | Clean-window AUC | LogLoss | Note |
|---:|---:|---:|---|
| 1 | 0.752438 | 0.219396 | below M91 |
| 2 | 0.762562 | 0.220036 | below M91 by -0.006951 |
| 3 | 0.760782 | missing in copied log | below M91 by -0.008731 |
| 4 | 0.760019 | 0.224426 | below M91 by -0.009494 |

Decision: M93 failed Gate 2. Best is **0.762562**, still **-0.006951** below M91
and **-0.007351** below the planned promote threshold M91 + 0.0004 =
**0.769913**. The current 6h/w0.3 mature-negative weighting formulation should
not promote a public candidate.

M94 full-data candidate:

| Epoch | Normal-dev AUC | LogLoss |
|---:|---:|---:|
| 1 | 0.854777 | 0.263282 |
| 2 | **0.857370** | 0.263853 |
| 3 | 0.857065 | 0.258410 |

Readout: the copied log does not include M94's final E4 summary, but E1-E3 are
already far below M84 normal dev **0.870490**, and M93 failed the gate. M94 is
therefore not an evaluation candidate.

## Gate 3 · Sequence Activity Features

Hypothesis: public users are singleton-like, so ranking depends more on item
and sequence state than user memorization. Eval EDA shows saturated sequences,
especially `seq_d`, so explicit sequence length / timestamp-span features may
help the dense tower calibrate activity intensity.

Prepared jobs:

| Job | Platform entry | Role |
|---|---|---|
| `M95-m84-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template untracked | full-data public candidate |
| `M96-m84-mnw6h-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template untracked | interaction candidate: delayed feedback + seq stats |
| `M97-cleanwin0303-m84-seqstats-gap0-seed42-4ep` | upload local `platform_run_templates/run_m97_cleanwin0303_m84_seqstats.sh` as `run.sh` | diagnostic: compare seqstats vs M91 on clean window |
| `M98-cleanwin0303-m84-mnw6h-seqstats-gap0-seed42-4ep` | upload local `platform_run_templates/run_m98_cleanwin0303_m84_mnw6h_seqstats.sh` as `run.sh` | diagnostic: compare interaction vs M91/M93/M97 on clean window |

Decision:

- M97 - M91 **>= +0.0004**: seqstats transfer is live; evaluate M95 if normal
  dev/logloss are not bad.
- M97 - M91 in **[-0.0002, +0.0004)**: seqstats is ambiguous; only evaluate M95
  if full-data M95 has an unusually strong normal-dev/logloss profile and quota
  is not competing with a stronger gate.
- M97 - M91 **< -0.0002**: do not evaluate M95 by default.
- M98 should only promote M96 if it beats the best relevant clean-window parent
  (M93 or M97) or if both M93 and M97 are individually positive. Otherwise the
  interaction is likely just a noisy combination.

Final results:

| Job | Split | Best AUC | Delta vs M91 when applicable | Decision |
|---|---|---:|---:|---|
| M95 | normal dev | **0.871160** | n/a | tempting but gated |
| M96 | normal dev | **0.859312** | n/a | reject |
| M97 | clean-window | **0.760092** | **-0.009421** | seqstats transfer gate failed |
| M98 | clean-window | **0.756595** | **-0.012918** | interaction rejected |

Readout: M95 is the only ordinary-dev positive candidate in this batch
(+0.000670 vs M84), but M97 fails the pseudo-public window by a large margin.
This is the exact failure mode EDA-71 warned about: the normal RG split likes
features that do not transfer to the future window. M95 should not receive
default score-max quota.

## Evaluation Allocation

5/18 final quota:

- **M88 E6 evaluated** as the final 5/18 quota.
- Result: public **0.846209**, which rejects longer M84 training / E6
  checkpoint selection despite normal-dev **0.870910**.

5/19 morning quotas:

1. No M93-M100 candidate passed its gate.
2. Hold M94/M95/M96/M89/M85 by default; do not spend score-max quota on them.
3. If one quota is intentionally reserved for LB calibration rather than score
   max, the least irrational probes are M95 (normal-dev seqstats signal) or M89
   (strongest normal-dev time signal), but both are high-risk because their
   clean-window evidence failed.
4. Preserve quota for a new, cleaner mechanism unless the user explicitly wants
   an information probe.

## Upload Notes

For M93-M96 upload the same current source bundle:

- platform entry script uploaded as `run.sh`
- `src/train.py`
- `src/dataset.py`
- `src/model.py`
- `src/trainer.py`
- `src/infer.py`
- `src/utils.py`
- `src/ns_groups_vocab_tier.json`

M95/M96 change the model input dimension via `enable_seq_stats_features`, so
the patched `infer.py` and `dataset.py` must be uploaded for evaluation.

Local `run_*.sh` templates are transient helpers and intentionally not
committed; the platform only uses `run.sh`.

## Queue Status

User submitted M93-M96 to the platform training queue, and later submitted
M97-M100 behind them. All eight jobs have now completed according to the copied
log, with partial recovery gaps for M94 only.

Implication:

- M93-M100 are completed from copied logs.
- The only missing detail is M94's final E4 line, which is not decision-relevant
  because both M93 and visible M94 evidence fail.
- The next platform action should be a new mechanism or a deliberately chosen
  high-risk information eval, not automatic promotion of this batch.
