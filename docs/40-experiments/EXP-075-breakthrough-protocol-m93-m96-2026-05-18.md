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
  dev is not much worse than M84/M88.
- M93 - M91 **< 0**: do not evaluate M94; consider a softer weight only if all
  other lines fail.

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

Decision:

- M95 normal dev **>= M84 + 0.0003** and logloss is not worse: evaluate when
  quota allows.
- M96 should only be evaluated if at least one of M93/M95 is positive; otherwise
  it is likely just a noisy combination.

## Evaluation Allocation

5/18 final quota:

- default score move: **M88 E6**;
- override only if M91/M92 return immediately and give an unusually strong
  clean-window time signal.

5/19 morning quotas:

1. M88 E6 if not used on 5/18.
2. Best of M89/M90 only if Gate 1 passes.
3. M94 only if Gate 2 passes.
4. M95/M96 only if Gate 3 passes and the candidate is not redundant.

Keep at least one 5/19 evening quota for the best new mechanism from M93-M96.

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

User submitted M93-M96 to the platform training queue. The platform training
pool auto-schedules queued jobs when earlier jobs complete and a worker frees.

Implication:

- M93-M96 are **submitted / queued**, not necessarily already running.
- Completion logs, not submission order alone, determine which candidates are
  ready for gate decisions.
- While the queue is non-empty, a freed worker is not manually free; it is
  expected to pull the next queued job automatically.
