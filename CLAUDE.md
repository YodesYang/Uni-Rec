# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TAAC 2026 competition baseline — post-click conversion rate prediction (pCVR) for Tencent advertising. Single-model architecture (`PCVRHyFormer`) evaluated by AUC on held-out test data.

- **Preliminary deadline**: May 25, 2026 (25 days from 2026-05-02)
- **Platform**: Tencent Angel (太极 Taiji), **4 training slots (always available, frees on completion · NOT a daily quota)**, 4 evaluation submissions/day (北京 12:00 / CEST 06:00 reset)
- **Remote**: `git@github.com:YodesYang/Tencent_Uni_Rec.git` (main branch)
- **🔴 Agent platform 操作前必读**: [`docs/00-context/platform-state.md`](docs/00-context/platform-state.md) (single source of truth · 每次操作前 grep)

---

## Environment

```bash
conda env create -f environment.yml
conda activate tencent-unirec
```

Dependencies: Python 3.10, PyTorch ≥ 2.7, numpy ≥ 2.2, pyarrow ≥ 23, scikit-learn, tensorboard. The `environment.yml` uses a CPU wheel by default; swap the torch index URL for `cu126` when on a CUDA machine.

---

## Running

### Local smoke / fast-proxy run

```bash
python src/train.py \
 --data_dir ./data/demo_1000 \
 --ckpt_dir ./runs/smoke_ckpt \
 --log_dir ./runs/smoke_log \
 --num_epochs 1 --batch_size 256 \
 --valid_split_strategy time \
 --seq_encoder_type swiglu \
 --limit_train_rgs 8 --limit_valid_rgs 2 --max_train_steps 500 \
 --device cpu
```

### Full training

```bash
python src/train.py \
 --data_dir /path/to/parquet \
 --ckpt_dir ./runs/full_ckpt \
 --num_epochs 10 --batch_size 256 \
 --valid_split_strategy time \
 --seq_encoder_type swiglu \
 --enable_count_features --enable_seq_stats_features \
 --d_model 64 --num_queries 1
```

### Inference / evaluation

```bash
MODEL_OUTPUT_PATH=./runs/full_ckpt/global_stepXXX.best_model \
EVAL_RESULT_PATH=./eval_results \
python src/infer.py
```

`infer.py` reads `train_config.json` + `schema.json` from the checkpoint directory to reconstruct the exact model. The platform's default `infer.py` does the same — custom upload is only required if `infer.py` itself is modified.

### Syntax / compile check (no test suite)

```bash
python -m py_compile src/train.py src/dataset.py src/model.py src/trainer.py src/infer.py
```

There is no test suite and no CI. Verification is manual: local smoke run → platform submission.

---

## Architecture

### `src/model.py` — PCVRHyFormer

The model unifies multi-domain behavior sequences and non-sequence features:

1. **NS Tokenizers** (`GroupNSTokenizer` / `RankMixerNSTokenizer`) — embed user + item discrete features into a fixed set of "non-sequence" (NS) tokens.
2. **MultiSeqQueryGenerator** — for each of the 4 sequence domains (a/b/c/d), generates Q query tokens from NS tokens + mean-pooled sequence.
3. **Sequence encoders** — per-domain encoder (choice: `swiglu`, `transformer`, `longer`) compresses variable-length behavior sequences.
4. **MultiSeqHyFormerBlock** — cross-attends each encoder's output with the domain's query tokens, then runs a joint `RankMixerBlock` (parameter-free token mixing + per-token FFN) across all Q tokens + NS tokens.
5. **Output projection** — final token representation → binary classifier.

Key constraint: when `rank_mixer_mode='full'`, `d_model % (num_queries * num_sequences + num_ns) == 0` must hold.

Empirically: **SwiGLU is the default** — 3.5× faster inference than transformer, best AUC in current experiments.

### `src/dataset.py` — data pipeline

- Streams Parquet row groups with 16 workers, pre-allocated numpy buffers (no per-row allocation).
- `FeatureSchema` maps column names → (offset, length) for efficient tensor slicing.
- **Time bucketing**: sequence timestamps → log-scale deltas → 64 log-boundaries (65 total buckets + padding token).
- Synthetic dense features (all opt-in via CLI flags):
  - `--enable_count_features` (P0-A): log-scaled count of non-zero user_int / item_int fields.
  - `--enable_seq_stats_features` (P0-B): per-domain sequence lengths + timestamp ranges.
  - `--enable_history_cvr_features` (P0-C): leak-safe historical item CVR from `.npz` sidecar.
- `--valid_split_strategy time` sorts row groups by `max(timestamp)` and holds out the tail (ADR-001). Use this instead of `filename` to avoid label-leakage from overlapping time windows.
- `_diag_*` fields flow through to trainer for slice diagnostics (ADR-002).

### `src/trainer.py` — training loop

- **Dual optimizer**: Adagrad for sparse embedding tables, AdamW for all dense parameters.
- Saves `model.pt` + `train_config.json` + `schema.json` + `ns_groups.json` into every `*.best_model/` checkpoint automatically.
- `training_summary.json` in the checkpoint root — grep-friendly, records best val AUC, step, epoch.
- Slice diagnostics: per-quantile AUC breakdown on `user_int_nonzero`, `item_int_nonzero`, `seq_total_len`, `label_type`.

### `src/train.py` — CLI entry point

Key flags beyond the obvious:

| Flag | Purpose |
|------|---------|
| `--valid_split_strategy {filename,time}` | Time-aware split (ADR-001) |
| `--limit_train_rgs N` / `--limit_valid_rgs N` | Fast proxy: cap row groups |
| `--max_train_steps N` | Fast proxy: stop early |
| `--enable_count_features` | P0-A synthetic features |
| `--enable_seq_stats_features` | P0-B synthetic features |
| `--enable_history_cvr_features` | P0-C synthetic features |
| `--enable_row_time_cutoff` | T8.5: row-level timestamp filter |
| `--seq_encoder_type {swiglu,transformer,longer}` | Encoder selection |
| `--rank_mixer_mode {full,ns_only}` | Token mixing scope |
| `--loss_type {bce,focal}` | Loss function |

---

## Experiment Discipline

**Always use the layered funnel before consuming platform quota:**

```
① py_compile + git diff --check
② local smoke (limit_train_rgs=8, max_train_steps=200)
③ fast proxy on platform (limit_train_rgs~50, max_train_steps=500, 1 epoch)
④ 2-3 epoch half-train confirmation
⑤ full train only for high-confidence candidates
⑥ leaderboard eval only when valid AUC signal is credible
```

Full train takes ~10 hours on platform. Do not skip the funnel.

### Naming conventions

- Tasks: `T<N>` tracked in `docs/20-tasks/active.md`
- Architecture decisions: `docs/30-decisions/ADR-NNN-descriptive-name.md`
- Experiments: `docs/40-experiments/EXP-NNN-theme-YYYY-MM-DD.md`
- Keep inline task notes minimal — push reasoning to ADR, numbers to EXP.

When touching `docs/20-tasks/active.md`, follow the 7-step audit in `.cursor/rules/todo-review.mdc`: status validation → timeline pressure → priority reorder → recipe timeliness → red-line check → concretize next steps → overwrite "last audit" header (do not accumulate).

---

## Platform Gotchas

- **"Copy" trap**: Platform Training → Copy clones a frozen snapshot. After any code change, create a **New Training Job** and re-upload `src/`. Using Copy after code changes causes `unrecognized arguments` failures (confirmed 2026-05-01).
- **Publish checklist** before publishing a checkpoint for evaluation:
  1. It is a `*.best_model/` subdirectory (not a mid-step checkpoint).
  2. It is the global latest best after training completes (not epoch-1).
  3. The directory contains `model.pt` + `train_config.json` + `schema.json`.
- **Leaderboard delay**: Evaluations submitted before 16:00 AOE update the leaderboard at ~19:00 AOE same day. "Evaluation Success" ≠ score posted immediately.
- **Inference time budget**: SwiGLU ≈ 82 s, LongerEncoder ≈ 203 s, Transformer ≈ 514 s on 1.9M train + 523k test. Hard limit is 30 min.
- **predictions.json must cover all test user_ids** — missing IDs cause score penalty (523,244 users in current test set).

---

## Competition Red Lines

- **No ensemble**: multi-seed averaging, stacking, blending, TTA are all forbidden. MoE trained end-to-end is permitted. SWA is a grey area — confirm with organizers before using.
- **No external data**: only official Tencent-provided features.
- **Single architecture only** for final submission.
- Before recommending any strategy (ensemble-adjacent, pretraining, external data), grep `docs/00-context/competition.md` §3 and `docs/10-research/community-intel.md` for red lines.

---

## Python Conventions

All functions must have complete type annotations (Python 3.10+ syntax: `X | None` not `Optional[X]`). Catch specific exception types, never bare `except`. Chain exceptions with `raise ... from e`. Use `None`-sentinel defaults instead of mutable default arguments.

---

## Commit / Push Convention

Follow conventional commits: `feat:`, `fix(scope):`, `docs:`, `chore:`. Commit only when a task reaches a stable milestone and the working tree is self-contained. **Do not push to `main` without explicit user instruction** — present the commit message draft for approval first.
