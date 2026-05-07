#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: fafe-gap05-icq-halftrain-swiglu-vr005-bs128
#
# H2 · Feature-aware Feature Embedding (FaFE) in RankMixer NS
# tokenizer. Each fid's raw embedding passes through a per-fid
# LayerNorm+Linear(emb_dim, emb_dim) before chunked concatenation.
#
# Baseline: icq-half-gap05 (Dev 0.842443, LB 0.839223).
# Change:   +--enable_fafe
#
# Rationale (Seed 主楼方案 2 + InterFormer CIKM 2025):
#   * Default tokenizer shares chunk projection across all fids, so
#     fid-specific structure can get diluted.
#   * FaFE gives each fid a dedicated transform BEFORE concatenation,
#     letting the model learn feature-specific representations.
#   * InterFormer paper reports +0.14% AUC on same-distribution test.
#     Per EDA EXP-029: model is known to overfit info-rich samples,
#     so per-fid transform should help redistribute representation
#     capacity.
#
# Extra params: ~257k dense (60 fids × 4288 params per-fid-transform
# for emb_dim=64). Training cost: +~3-5%.
#
# Decision thresholds (ΔLB vs icq-half-gap05 LB 0.839223):
#   ≥ +0.005  🎯 Real signal, FaFE Accept.
#   +0.002..+0.005  Suspected, need multi-seed.
#   -0.002..+0.002  Flat.
#   < -0.002  Negative.
#
# ⚠️ Platform note: --enable_fafe is a PCVRHyFormer.__init__ kwarg.
# Must upload baseline/infer.py so eval-time model is built the same.
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
    --valid_gap_ratio 0.05 \
    --enable_row_valid_gap \
    --item_conditioned_query \
    --enable_fafe \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
