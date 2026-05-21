# EXP-088 · Aux checkpoint side selector · 2026-05-20

> **Status**: Implemented locally. Default behavior remains unchanged.

## Trigger

After M119/M123/M124/M116 public results, the validation hierarchy changed:

- standard 5% primary valid is no longer a reliable public ranker;
- `aux_valid/m91` is more public-like, but full-data aux is leaky;
- M124 proves aux alone can over-rank a model.

So the right mechanism is not to switch `best_model` to aux. Instead, keep the
canonical primary-valid checkpoint and save aux/blend **side candidates** for
manual evaluation.

## CLI

```bash
--enable_aux_ckpt_select \
--aux_ckpt_name m91 \
--aux_ckpt_primary_tolerance 0.0008 \
--aux_ckpt_blend_weight 0.5
```

Defaults are safe:

- disabled unless explicitly requested;
- `aux_ckpt_name=m91`;
- primary AUC guard: `current_primary_auc >= best_primary_auc - 0.0008`;
- blend score: `(1-w)*primary_auc + w*aux_auc`, default `w=0.5`.

## Output

When enabled and the named aux window is present, training may save:

| Directory suffix | Selection score | Role |
|---|---|---|
| `*.aux_m91_best` | aux AUC | public-like checkpoint candidate |
| `*.blend_m91_best` | primary/aux blend | conservative checkpoint candidate |

The original `*.best_model` and `*.topk_N` outputs are untouched.

## Use On Current Queue

M133-M136 were already submitted to the platform queue before the selector
change was adopted operationally. They will not be paused/restarted. Treat
those queued jobs as primary-valid/topk-selected unless the copied platform
logs explicitly show `enable_aux_ckpt_select=True`.

For new jobs after M133-M136, enable this selector by default when an
`aux_valid/m91` window is present.
