# EXP-077 · Auxiliary Validation Framework · 2026-05-19

## Motivation

M86 and M89 both showed the same failure mode:

| Candidate | Normal dev | Clean proxy | Public | Readout |
|---|---:|---:|---:|---|
| M86 UTC+Beijing | 0.871415 | M92 **0.766748**, M91 **0.769513** | 0.847106 | normal-dev illusion |
| M89 Beijing v1+v2 | 0.871864 | M99 **0.764915**, M91 **0.769513** | 0.847650 | normal-dev illusion |

The fixed clean-window diagnostics were useful, but running a separate
diagnostic job for every candidate doubles training demand. We need each future
training job to emit richer validation signals without changing how the model
is trained or selected.

## Implementation

Added optional CLI:

```bash
--aux_valid_windows "m91:1772502591:1772520547"
```

Format:

- comma-separated entries;
- each entry is `name:start:end`;
- `start` is inclusive, `end` is exclusive;
- requires `--valid_split_strategy time`;
- default empty string keeps old behavior unchanged.

Code paths:

| File | Change |
|---|---|
| `src/train.py` | parses `--aux_valid_windows` and passes specs to data/trainer |
| `src/dataset.py` | builds read-only aux DataLoaders from Row Groups touching each timestamp window |
| `src/trainer.py` | evaluates aux loaders after primary valid each epoch and logs metrics |

## Semantics

Aux validation is **diagnostic only**:

- does not update weights;
- does not select `best_model`;
- does not affect `topk`;
- does not affect early stopping;
- does not affect the logloss rebound guard.

As of 2026-05-20, there is one explicit opt-in exception:
`--enable_aux_ckpt_select` can save **extra side checkpoints** ranked by an aux
window and by a primary/aux blend. This still does not replace the canonical
primary-valid `best_model`, `topk`, early stopping, or training updates.

Each aux window is logged with:

- timestamp range;
- train max timestamp;
- `status=clean` if train max <= aux start;
- `status=leaky` otherwise;
- AUC / LogLoss per epoch;
- `gap_auc=valid-aux`.

This distinction matters. In a normal full-data run, an M91-style aux window is
usually **leaky**, because the model has trained on rows after that window. It
is still useful as a ranking/debug signal, but it is not equivalent to a
separate clean-window diagnostic job.

## Usage

For future full-data jobs:

```bash
--aux_valid_windows "m91:1772502591:1772520547"
```

For future dedicated clean diagnostics, continue to use:

```bash
--fixed_valid_timestamp_min 1772502591
--fixed_valid_timestamp_max 1772520547
```

Decision policy:

| Signal | Meaning |
|---|---|
| primary valid up, aux/leaky down | dev-public mismatch risk |
| primary valid up, aux also up | stronger candidate |
| clean fixed-window up | eval-worthy if full dev/logloss are sane |
| leaky aux up | useful, but not enough alone for scarce quota |

## Aux Checkpoint Side Selector

New optional CLI:

```bash
--enable_aux_ckpt_select \
--aux_ckpt_name m91 \
--aux_ckpt_primary_tolerance 0.0008 \
--aux_ckpt_blend_weight 0.5
```

When enabled, the trainer keeps the original primary-valid `*.best_model` and
`*.topk_N` directories, and additionally saves:

- `*.aux_m91_best`: highest `aux_valid/m91` AUC among checkpoints whose primary
  AUC is within `best_primary_auc - tolerance`;
- `*.blend_m91_best`: highest `(1-w)*primary_auc + w*aux_auc` under the same
  primary guard.

This is intended for late-stage manual public-eval selection. The primary AUC
guard prevents an aux spike from choosing a severely underfit checkpoint.

## Status

Implemented locally and syntax-checked:

```bash
python3 -m py_compile src/train.py src/trainer.py src/dataset.py
git diff --check
```

Both passed.
