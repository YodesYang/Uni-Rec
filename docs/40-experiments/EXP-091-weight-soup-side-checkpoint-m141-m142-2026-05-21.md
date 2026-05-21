# EXP-091 · Weight-Soup Side Checkpoint · M141-M142 · 2026-05-21

> **Status**: Implemented locally; user canceled the first M141/M142 submission
> and resubmitted after the dense-only soup hotfix. Later M138 and M144 public
> evals rejected the `vr015` premise, so M142 is downgraded to logs-only by
> default. M142 has now returned and is weaker than M138 on all comparable
> internal metrics; M141 remains the only score-relevant member of this pair.

## Trigger

After M136 public **0.849015**, the high-aux UE61/87+xitem branch is no longer
a good use of overnight training slots. The remaining gap from M123 public
**0.851011** to the new target **0.853** is about **+0.001989**, too large for
ordinary seed-only tuning.

The public-positive evidence is concentrated in the M106/M119/M123 family:

- corrected T38 per-domain time buckets,
- UE61 split + item-conditioned query + DIN,
- fresher validation/training selection (`valid_ratio=0.02` / nearby),
- MTL as mild regularization.

## Mechanism

Add an opt-in post-train checkpoint averaging path:

- `--enable_weight_soup`
- `--weight_soup_topk`
- `--weight_soup_primary_tolerance`
- `--weight_soup_min_members`

The trainer still saves normal `*.best_model`, `*.topk_N`, and optional
`aux/blend` side checkpoints exactly as before. At training-summary time, if
enabled, it gathers the same-run `best_model + topk` checkpoints, deduplicates
duplicate steps, filters to checkpoints within `best_auc - tolerance`, averages
eligible dense floating tensors on CPU fp32, preserves sparse embedding tensors
from the best source by default, copies non-floating tensors from the first
source, and writes one extra checkpoint directory:

`global_step*.auc=*.members=*.weight_soup/`

This is intentionally a side artifact. It does **not** affect:

- optimizer updates,
- early stopping,
- primary `best_model`,
- top-k retention,
- aux/blend checkpoint selection.

## Jobs

| Priority | Job | Helper | Purpose |
|---:|---|---|---|
| 1 | `M141-m106-vr02-mtl-seed7-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m141_m106_vr02_mtl_seed7_auxselect_soup.sh` | M137 recipe plus weight soup. Tests the cleanest public-positive missing cell with a smoother final checkpoint. |
| 2 | `M142-m123-vr015-mtl-seed5-auxselect-soup-gap0-4ep` | `platform_run_templates/run_m142_m123_vr015_mtl_seed5_auxselect_soup.sh` | M138 interpolation plus weight soup. Downgraded after M138 public 0.849451 and M144 soup public 0.848183; logs-only by default. |

Both use:

- `--keep_topk_ckpt 4`
- `--enable_weight_soup`
- `--weight_soup_topk 4`
- `--weight_soup_primary_tolerance 0.0012`
- `--weight_soup_min_members 2`

Sparse embeddings are **not** averaged by default. This is deliberate: current
training uses epoch-boundary sparse embedding cold restart, so cross-epoch
embedding tables are not guaranteed to live in an aligned coordinate system.
The soup therefore behaves like a dense-head / transformer-body averaging
mechanism while keeping the best checkpoint's sparse tables intact. The escape
hatch is `--weight_soup_include_sparse`, but do not enable it for the current
M123-family jobs unless logs prove reinit is disabled.

## Readout

Training log must contain one of:

- `[weight_soup] saved checkpoint to .../model.pt`
- `[weight_soup] skipped: ...`
- `[weight_soup] failed: ...`

If saved, the checkpoint directory is directly publish/eval-able like other
platform checkpoints. After the M138/M144 misses, public-eval priority is:

1. M141 `*.weight_soup` if saved and primary/aux are healthy,
2. M141 normal `best_model` if soup is skipped or suspicious,
3. M142 only for an explicit diagnostic, not for score-max.

## Caveat

Local machine does not have `torch` installed, so only syntax/static checks were
run locally. The real checkpoint load/average path will first execute on the
platform. The implementation is defensive: soup errors are logged and stored in
`training_summary.json`, but they do not fail the main training job.

## M142 Completion

User reported M142 completed and copied logs to `~/Downloads/train_log.txt`.
Config is correct:

- `valid_ratio=0.015`
- seed **5**
- `multi_task_loss=True`
- legacy UE61-only: `ue_split_fids=''`, `ue_split_fid=61`
- `enable_aux_ckpt_select=True`
- `enable_weight_soup=True`, `weight_soup_include_sparse=False`
- no TIN-lite

Primary curve:

| Epoch | Step | Primary AUC | LogLoss | Aux m91 AUC | Blend score |
|---:|---:|---:|---:|---:|---:|
| 1 | 16564 | 0.868212 | 0.224400 | 0.845488 | 0.856850 |
| 2 | 33128 | 0.871022 | 0.221139 | 0.849474 | 0.860248 |
| 3 | 49692 | 0.871737 | 0.221040 | 0.851972 | 0.861855 |
| 4 | 66256 | **0.872461** | **0.221440** | **0.853713** | **0.863087** |

Primary checkpoint:

`global_step66256.layer=2.head=4.hidden=64.auc=0.872461.best_model`

Aux/blend side checkpoints:

- `global_step66256.layer=2.head=4.hidden=64.score=0.853713.auc=0.872461.aux_m91=0.853713.aux_m91_best`
- `global_step66256.layer=2.head=4.hidden=64.score=0.863087.auc=0.872461.aux_m91=0.853713.blend_m91_best`

Dense-only soup:

- Sources: E4 best_model **0.872461**, E3 topk **0.871737**
- Sparse embeddings preserved from E4 best source
- Output:
  `global_step66256.layer=2.head=4.hidden=64.auc=0.872461.members=2.weight_soup`

Comparison:

| Anchor | Primary | LogLoss | Aux m91 | M142 delta |
|---|---:|---:|---:|---:|
| M138 `vr015+MTL` | 0.872871 | 0.221286 | 0.854026 | primary **-0.000410**, logloss **+0.000154**, aux **-0.000313** |
| M144 `vr015+MTL+TIN+soup` | 0.871954 | 0.220568 | 0.855139 | primary **+0.000507**, logloss **+0.000872**, aux **-0.001426** |
| M123 public SOTA | 0.870584 | 0.221884 | 0.853945 | primary **+0.001877**, logloss **-0.000444**, aux **-0.000232** |

Interpretation:

- M142 is a correct run and soup saved successfully, but it is weaker than M138
  on primary, logloss, and aux.
- M138 already evaluated at public **0.849451**, and M144 soup evaluated at
  public **0.848183**. Therefore M142 has no score-max public-eval case.
- Treat M142 as a diagnostic confirming the `vr015+soup` premise is not worth
  further quota. Do not evaluate by default.
