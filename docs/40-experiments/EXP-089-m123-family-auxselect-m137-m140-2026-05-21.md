# EXP-089 · M123-family aux selector and low-risk xitem · M137-M140 · 2026-05-21

> **Status**: Prepared because the user reported the training queue was empty.
> After M136 public eval returned negative, keep M137/M138 and cancel or
> deprioritize M139/M140 if platform slots can be reclaimed. M137 returned,
> but its copied Args do **not** match the intended recipe, so it is an invalid
> seed7+MTL readout. M138 returned with a very strong, correctly configured
> `valid_ratio=0.015` training result, but public eval later rejected it at
> **0.849451**.

## Trigger

M123 is the current public SOTA at **0.851011**. The reliable public-positive
pattern is now:

- M106/M119/M123 family, not standard-valid seed upper-tail alone.
- `valid_ratio=0.02` freshness transferred strongly.
- MTL improved seed5 (`M123`) over freshness-only seed5.

The external UE61/87+xitem branch produced very high aux m91 but failed the
clean-window diagnostics. M136 later returned public **0.849015**, confirming
that high aux alone is not enough for this mechanism.

## Design

This batch fills two missing cells:

1. Pure seed7 freshness+MTL without the UE61/87+xitem confound.
2. M123-family runs with the new aux/blend checkpoint selector enabled.
3. Lower-strength UE61 x item variants that avoid `fid87` and start the
   explicit interaction at `alpha=0.3` instead of the current `1.0`. These are
   now lower priority after M136's miss unless already queued.

All four jobs use the corrected-T38 M106/M123 backbone:

- `--enable_ue_split --ue_split_fid 61`
- `--enable_din_interest`
- `--enable_per_domain_buckets --per_domain_bucket_sample_rgs 64`
- `--aux_valid_windows "m91:1772502591:1772520547"`
- `--enable_aux_ckpt_select`

The aux selector is a side selector only: it preserves primary `best_model`,
top-k, early stopping, and training, while saving extra `aux_m91_best` and
`blend_m91_best` checkpoint directories if eligible.

## Jobs

| Priority | Job | Helper | Purpose |
|---:|---|---|---|
| 1 | `M137-m106-vr02-mtl-seed7-auxselect-gap0-4ep` | `platform_run_templates/run_m137_m106_vr02_mtl_seed7_auxselect.sh` | Missing core ablation: M119 seed7 freshness + MTL, no UE61/87/xitem confound. |
| 2 | `M138-m123-vr015-mtl-seed5-auxselect-gap0-4ep` | `platform_run_templates/run_m138_m123_vr015_mtl_seed5_auxselect.sh` | Freshness interpolation between public-positive M123 (`vr02`) and M124 (`vr01`). |
| 3 | `M139-m123-vr02-mtl-ue61-xitem-a03-seed5-auxselect-gap0-4ep` | `platform_run_templates/run_m139_m123_vr02_mtl_ue61_xitem_a03_seed5_auxselect.sh` | Lower-strength M135: UE61 only, explicit xitem, alpha init 0.3; lower priority after M136 miss. |
| 4 | `M140-m106-vr02-mtl-ue61-xitem-a03-seed7-auxselect-gap0-4ep` | `platform_run_templates/run_m140_m106_vr02_mtl_ue61_xitem_a03_seed7_auxselect.sh` | Seed7 counterpart to M139; lower priority after M136 miss. |

Upload the selected helper as platform `run.sh`. Include updated `train.py`,
`trainer.py`, `model.py`, `infer.py`, `dataset.py`, `utils.py`,
`infer_schema.py`, and `ns_groups_vocab_tier.json`.

## Readout

Primary anchor:

- M123 public SOTA: **0.851011**
- M119 public: **0.850754**
- M136 public: **0.849015**

Training-readout priority:

1. If M137 beats M123/M119 on primary or aux, it becomes the cleanest next eval.
2. If M138 is close to M123 primary and saves a stronger aux/blend side
   checkpoint, it is a low-risk selector candidate.
3. If M139/M140 are cancellable, reclaim the slots for the weight-soup branch
   (M141/M142). If they cannot be cancelled cheaply, keep them as lower-risk
   xitem diagnostics only.
4. For M139/M140 to become public-eval candidates after M136's miss, they need
   a much healthier primary curve than M134/M136. High aux alone is insufficient.

## M137 Completion · Config Mismatch

User reported M137 completed and copied logs to `~/Downloads/train_log.txt`.
The log header says:

`M137-m106-vr02-mtl-seed7-auxselect-gap0-4ep`

But the actual `Args` show:

- `seed=5`, not intended `seed=7`
- `enable_aux_ckpt_select=False`, not intended `True`
- `ue_split_fids='61,87'`, with model init confirming `fids=[61, 87]`
- no weight soup

The local helper
`platform_run_templates/run_m137_m106_vr02_mtl_seed7_auxselect.sh` is correct:
it contains `--seed 7`, `--enable_aux_ckpt_select`, and legacy UE61-only
`--ue_split_fid 61`. Therefore this platform run likely used a stale or wrong
uploaded `run.sh`.

Actual curve:

| Epoch | Step | Primary AUC | LogLoss | Aux m91 AUC |
|---:|---:|---:|---:|---:|
| 1 | 16480 | 0.865254 | 0.226464 | 0.845970 |
| 2 | 32960 | 0.867555 | 0.223864 | 0.846393 |
| 3 | 49440 | 0.868320 | 0.224448 | 0.852058 |
| 4 | 65920 | **0.868682** | **0.223456** | **0.852891** |

Checkpoint:

`global_step65920.layer=2.head=4.hidden=64.auc=0.868682.best_model`

Interpretation:

- This is **not** a valid test of M119 seed7 freshness + MTL.
- The result is below M123, M135, and the real M133/M134/M136 stack, so it has
  no public-eval value.
- Treat true M137 as still unanswered unless M141/M143 return with correct
  seed7 Args. Because M141 is the same core recipe plus auxselect/soup, a
  correctly uploaded M141 can effectively supersede a rerun of plain M137.
- For M141/M143 start logs, explicitly check `seed=7`,
  `enable_aux_ckpt_select=True`, `enable_weight_soup=True`, and no
  `ue_split_fids='61,87'`.

## M138 Completion

User reported M138 completed and copied logs to `~/Downloads/train_log.txt`.
Config is correct:

- `valid_ratio=0.015`
- seed **5**
- `multi_task_loss=True`
- legacy UE61-only: `ue_split_fids=''`, `ue_split_fid=61`
- `aux_valid_windows=m91:1772502591:1772520547`
- `enable_aux_ckpt_select=True`

Primary curve:

| Epoch | Step | Primary AUC | LogLoss | Aux m91 AUC | Blend score |
|---:|---:|---:|---:|---:|---:|
| 1 | 16564 | 0.868153 | 0.224440 | 0.845495 | 0.856824 |
| 2 | 33128 | 0.870821 | 0.221210 | 0.849034 | 0.859927 |
| 3 | 49692 | 0.871673 | 0.221079 | 0.852904 | 0.862288 |
| 4 | 66256 | **0.872871** | **0.221286** | **0.854026** | **0.863449** |

Primary checkpoint:

`global_step66256.layer=2.head=4.hidden=64.auc=0.872871.best_model`

Aux/blend side checkpoints:

- `global_step66256.layer=2.head=4.hidden=64.score=0.854026.auc=0.872871.aux_m91=0.854026.aux_m91_best`
- `global_step66256.layer=2.head=4.hidden=64.score=0.863449.auc=0.872871.aux_m91=0.854026.blend_m91_best`

Because primary, aux, and blend all peak at E4, these three checkpoint choices
are effectively the same model snapshot for public-eval purposes.

Comparison:

| Anchor | Primary | Aux m91 | M138 delta |
|---|---:|---:|---:|
| M123 public SOTA (`vr02+MTL` seed5) | 0.870584 | 0.853945 | primary **+0.002287**, aux **+0.000081** |
| M124 (`vr01+MTL` seed5) | 0.868723 | 0.854380 | primary **+0.004148**, aux **-0.000354** |
| M135 UE61+xitem | 0.871377 | 0.853113 | primary **+0.001494**, aux **+0.000913** |

Pre-eval interpretation:

- M138 was the strongest returned M123-family primary readout.
- It seemed to support the hypothesis that `valid_ratio=0.015` might be a
  better freshness sweet spot than either M123's 0.02 or M124's 0.01.
- Since 1.5% validation has only **31,439** rows, optimism risk was explicitly
  flagged before public calibration.

## M138 Public Eval

User reported:

`M138-m123-vr015-mtl-seed5-auxselect-gap0-4ep_eval_1779391358487`

Public result: **0.849451** (698.57s).

Public deltas:

| Anchor | Public AUC | M138 delta |
|---|---:|---:|
| M123 `vr02+MTL` seed5 | 0.851011 | **-0.001560** |
| M119 `vr02` seed7 | 0.850754 | **-0.001303** |
| M124 `vr01+MTL` seed5 | 0.850718 | **-0.001267** |
| M106 standard seed5 | 0.849364 | **+0.000087** |

Interpretation:

- The `valid_ratio=0.015` primary selector was over-optimistic. It can produce
  a very high tiny-window AUC without public transfer.
- This downgrades M142, which is the same `vr015` family plus dense-only soup,
  unless soup needs a direct calibration read.
- It also weakens M144 as a score-max candidate, because M144 uses the same
  `vr015` selector, although M144 remains high-information for TIN/soup.
- The public-positive spine is still `vr02` freshness + MTL, especially
  M119/M123. Corrected seed7 `vr02+MTL` jobs should be preferred over more
  `vr015` public evals if they return in time.
