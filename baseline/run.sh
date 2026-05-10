#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t27-compile-amp-halftrain-gap05
#
# T27 · torch.compile + AMP bf16 first REAL half-train (not proxy).
#
# Rationale (switched from smoke to half-train on 2026-05-09):
#   A proxy smoke (500 steps) only answers "does it crash?". A
#   half-train answers three questions in the SAME quota:
#     (1) Numerical correctness — bf16 autocast must keep Dev AUC
#         within ±0.001 of the fp32 baseline `icq-half-gap05`
#         (EXP-030: Dev E3 0.842443, LB 0.839223, Dev→LB gap +0.0032).
#     (2) Wall-clock speedup — compare epoch times against historical
#         `icq-half-seed42-rerun-2` / `icq-half-gap05` runs. Target
#         ≥1.5× speedup; <1.2× means compile hit too many recompiles
#         and we should fall back to AMP-only.
#     (3) Capacity headroom — read peak GPU mem from platform logs;
#         target ≤70% of 19GB (≤13GB). If true, the next job can
#         bump d_model=96 or batch_size=192 inside the same budget.
#
# Config = `icq-half-gap05` EXP-030 baseline (ICQ + swiglu +
# seq_hash_vocab=500k + time_of_day + valid_split=time + row_valid_gap
# 0.05 + seed=42 + 3 epochs) with three flags added:
#   --enable_torch_compile --enable_amp --amp_dtype bfloat16
#
# Go/No-go criteria (for promoting to "default training config"):
#   ✅ Dev AUC at E3 within [0.8414, 0.8434] (±0.001 of baseline 0.8424)
#   ✅ No dynamo traceback, no CUDA OOM
#   ✅ Epoch wall time ≤ 0.67× historical baseline (= ≥1.5× speedup)
#   ⚠️ If Dev drifts >0.001 low: investigate bf16 numerical issue on a
#      specific layer (likely the score head or MultiSeqHyFormerBlock
#      cross-attention softmax).
#
# Next day plan (if this passes): launch 2 architecture-bet jobs
# reusing the compile+AMP defaults — (a) DIN-block-integrated (move
# target attention from bypass to inside every HyFormer block),
# (b) multi-task click+conversion joint loss (Seed 主楼 vague hint).
# These two are the only paths with +0.005 ceiling; everything else
# is ±0.001 noise.
# ═══════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --enable_row_valid_gap \
    --valid_gap_ratio 0.05 \
    --item_conditioned_query \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 4 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
