# EXP-082 · DIN 2.0 Late Encoded Bypass · M109-M112 · 2026-05-19

## Context

Current public SOTA is **M84 = 0.848854**. The remaining gap to 0.852 is
**+0.003146**.

The recent negative evidence is now strong:

- Time-feature variants improved normal dev but failed public.
- Longer M84 training failed public.
- History-CVR improved normal dev but failed public.
- Clean-window diagnostics rejected delayed-feedback weighting, seqstats, and
  time-feature closure.

Seed search can still help at the margin, but it is unlikely to supply the
whole +0.003 gap. The highest-upside validated mechanism in this project
history remains target-aware interest / DIN. M84 already includes the original
T25 DIN bypass, so the next mainline should improve the DIN representation
without changing the high-risk parts of the backbone.

## Hypothesis

The original T25 DIN bypass is intentionally conservative:

- target = `mean(item_ns_tokens)`;
- DIN attends over raw sequence embeddings before HyFormer encoding;
- the four per-domain interests are merged to one `d_model` vector before the
  classifier.

This may now be the bottleneck. DIN 2.0 keeps the successful late classifier
bypass shape but upgrades two pieces:

1. **encoded DIN source**: run the classifier-time DIN over final
   HyFormer-encoded sequence states instead of raw sequence embeddings.
2. **per-domain merge**: optionally preserve the four per-domain interest
   vectors as `S*d_model` classifier input instead of compressing them to one
   vector.

This avoids the deprecated T28 failure mode: DIN is not injected inside every
block and does not repeatedly perturb RankMixer token fusion.

## Code Changes

Added two opt-in model/training config fields:

| Flag | Default | Meaning |
|---|---|---|
| `--din_interest_source {raw,encoded}` | `raw` | `raw` preserves T25; `encoded` uses final HyFormer sequence states for late DIN |
| `--din_interest_merge {compact,per_domain}` | `compact` | `compact` merges to `D`; `per_domain` keeps `S*D` into the classifier |

Backward compatibility:

- Existing T25/M84 configs remain `raw + compact`.
- Old checkpoints without these fields load through inference defaults.
- `encoded` source is supported only on the HyFormer path.

## Jobs

Upload each local helper as platform `run.sh`; do not submit generated
`run_*.sh` helpers to git.

| Job | Local helper | Purpose |
|---|---|---|
| `M109-cleanwin0303-m84-edindin-compact-gap0-seed42-4ep` | `platform_run_templates/run_m109_cleanwin0303_m84_encoded_din_compact.sh` | Clean-window diagnostic: M84 + encoded DIN, compact merge |
| `M110-m84-edindin-compact-auxm91-gap0-seed42-4ep` | `platform_run_templates/run_m110_m84_encoded_din_compact_auxm91.sh` | Full-data candidate with M91 aux logging |
| `M111-cleanwin0303-m84-edindin-perdomain-gap0-seed42-4ep` | `platform_run_templates/run_m111_cleanwin0303_m84_encoded_din_perdomain.sh` | Clean-window diagnostic: encoded DIN, per-domain interests |
| `M112-m84-edindin-perdomain-auxm91-gap0-seed42-4ep` | `platform_run_templates/run_m112_m84_encoded_din_perdomain_auxm91.sh` | Full-data per-domain candidate with M91 aux logging |

Shared base:

- M84 corrected-T38 stack.
- `--enable_ue_split --ue_split_fid 61`.
- `--enable_per_domain_buckets --per_domain_bucket_sample_rgs 64`.
- `--enable_din_interest`.
- `batch_size=128`, `num_epochs=4`, `gap0`, `seed42`.

Clean-window diagnostics use the M91 pseudo-public window:

- `--fixed_valid_timestamp_min 1772502591`
- `--fixed_valid_timestamp_max 1772520547`

Full-data jobs also log:

- `--aux_valid_windows "m91:1772502591:1772520547"`

## Launch Failure And Hotfix

User reported M109-M112 newly launched but repeatedly failed before training.
Sample M109 log at platform time `2026-05-20 00:23:54`:

```text
ValueError: not enough values to unpack (expected 5, got 4)
```

The failure happens after fixed-window dataset construction and before model
initialization. Diagnosis: this is an aux-validation framework interface
mismatch, not an encoded-DIN model failure. Updated `train.py` expected
`get_pcvr_data` to return `(train_loader, valid_loader, holdout_loader,
aux_valid_loaders, dataset)`, while the platform-side source bundle returned
the older four-value tuple.

Local hotfix:

- `src/train.py` now accepts both four- and five-value `get_pcvr_data`
  returns.
- If four values are returned, it sets `aux_valid_loaders={}` and logs a
  warning when aux windows were requested.
- Retry should still upload the matching updated `src/dataset.py` so
  M110/M112 keep their M91 aux metrics.

This launch failure invalidates no modeling hypothesis. M109-M112 should be
retried with the updated source bundle.

## Hotfix Restart

User reported that **M109-M112 have been restarted and are running** after the
compatibility hotfix. Treat the previous failure as a launch/source-bundle
issue unless the new logs repeat the same 4/5 return mismatch.

Current interpretation while running:

- M109 and M111 are the clean-window gatekeepers.
- M110 and M112 are the full-data candidates, but should not receive public
  eval promotion until their clean companions return, unless quota would
  otherwise expire and the user explicitly chooses a high-risk information
  eval.
- If M110/M112 logs show missing aux metrics, check whether the restarted
  bundle included the updated `dataset.py`; training can still be meaningful,
  but aux readout would be incomplete.

## Gates

Reference: M91 clean baseline **0.769513**.

| Zone | Condition | Action |
|---|---|---|
| green | clean delta >= +0.0004 | evaluate matching full candidate if normal dev/logloss are sane |
| yellow | clean delta in [-0.0003, +0.0004) and full dev is strong | consider one information quota |
| red | clean delta < -0.0003 or full dev weak | no public eval by default |

Evaluation priority:

1. If both compact and per-domain pass, prefer the cleaner full-data signal.
2. If only one clean companion is green/yellow-plus, evaluate only that branch.
3. If both clean companions are red, close DIN 2.0 in this form and avoid
   spending scarce public quota.

## M109 Result

Source: user copied `~/Downloads/train_log.txt` after M109 completed and
manually supplied the platform checkpoint table for the incomplete final log
tail.

Job: `M109-cleanwin0303-m84-edindin-compact-gap0-seed42-4ep`.

Confirmed config:

- Clean fixed window: `fixed_valid_timestamp_min=1772502591`,
  `fixed_valid_timestamp_max=1772520547`.
- M84 corrected-T38 stack.
- `enable_din_interest=True`.
- `din_interest_source=encoded`.
- `din_interest_merge=compact`.
- No aux window on this diagnostic job.

| Epoch | Step | AUC | LogLoss | Source |
|---:|---:|---:|---:|---|
| 1 | 2534 | 0.750343 | 0.221521 | log |
| 2 | 5068 | 0.764080 | 0.221435 | log |
| 3 | 7602 | **0.767038** | **0.218418** | log + checkpoint |
| 4 | 10136 | 0.765129 | n/a | user checkpoint table |

Best checkpoint:

```text
global_step7602.layer=2.head=4.hidden=64.auc=0.767038.best_model
```

Gate readout:

- M91 clean baseline: **0.769513**.
- Green threshold: **0.769913**.
- M109 best: **0.767038**.
- Delta vs M91: **-0.002475**.

Decision: **compact encoded-DIN fails the clean-window gate**. Do not evaluate
M110 by default, even if its ordinary full-data dev looks tempting. M111 later
returned red as well, so the per-domain branch is closed in this
encoded-late-bypass form.

## M111 Result

Source: user copied `~/Downloads/train_log.txt` after M111 completed.

Job: `M111-cleanwin0303-m84-edindin-perdomain-gap0-seed42-4ep`.

Confirmed config:

- Clean fixed window: `fixed_valid_timestamp_min=1772502591`,
  `fixed_valid_timestamp_max=1772520547`.
- M84 corrected-T38 stack.
- `enable_din_interest=True`.
- `din_interest_source=encoded`.
- `din_interest_merge=per_domain`.
- No aux window on this diagnostic job.

| Epoch | Step | AUC | LogLoss |
|---:|---:|---:|---:|
| 1 | 2534 | 0.750970 | 0.222518 |
| 2 | 5068 | **0.767948** | **0.220206** |
| 3 | 7602 | 0.756221 | 0.222447 |
| 4 | 10136 | 0.762607 | 0.221952 |

Best checkpoint:

```text
global_step5068.layer=2.head=4.hidden=64.auc=0.767948.best_model
```

Gate readout:

- M91 clean baseline: **0.769513**.
- Green threshold: **0.769913**.
- M111 best: **0.767948**.
- Delta vs M91: **-0.001565**.

Decision: **per-domain encoded-DIN also fails the clean-window gate**. Do not
evaluate M112 by default. Since M109 and M111 are both red, close DIN 2.0 in
this encoded-late-bypass form unless the user deliberately wants a public
calibration burn.

## M110/M112 Partial Running Readout

Source: user reported platform checkpoint tables while M110 and M112 were still
running.

M110 compact full-data candidate:

```text
global_step15941.layer=2.head=4.hidden=64.auc=0.865480.topk_2
global_step31882.layer=2.head=4.hidden=64.auc=0.868925.topk_1
global_step31882.layer=2.head=4.hidden=64.auc=0.868925.best_model
```

M112 per-domain full-data candidate:

```text
global_step15941.layer=2.head=4.hidden=64.auc=0.865514.topk_2
global_step31882.layer=2.head=4.hidden=64.auc=0.868436.topk_1
global_step31882.layer=2.head=4.hidden=64.auc=0.868436.best_model
```

Interpretation:

- M110/M112 are both below M106 E4 normal dev **0.871175** at the same rough
  stage family.
- Their paired clean diagnostics are already red: M109 **0.767038** and M111
  **0.767948** versus M91 **0.769513**.
- Therefore the running full-data companions no longer have enough expected
  value to occupy scarce overnight slots.

Decision: **stop/cancel M110 and M112 if doing so reclaims platform slots**. If
the platform cannot stop them cleanly, letting them finish is acceptable, but
the default action remains no public eval and no promotion.

## Risk

- `encoded` source adds a second attention read over encoded sequence states,
  so wall time may rise modestly.
- `per_domain` expands the classifier input by `+3*d_model` compared with
  compact DIN; this is still small and classifier-local, not per-fid/per-token
  parameter growth.
- Normal dev may overstate gains; clean companions remain required.

## Status

M109 and M111 completed; both clean-window diagnostics are red. M110/M112 are
still running by latest user report, but should be stopped if platform slots can
be reclaimed. If they finish anyway, neither should receive scarce public eval
quota by default.
