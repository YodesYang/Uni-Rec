#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: hash100k-gap05-icq-halftrain-swiglu-vr005-bs128
#
# T24 F1 · Aggressive seq hash vocab (500k → 100k) under the new
# gap=0.05 funnel. Paired with icq-half-gap05-icq-halftrain as the
# baseline; the pure methodological signal is
#   LB(hash100k-gap05) − LB(icq-half-gap05)
#
# Rationale (EDA EXP-029 finding 3):
#   seq_c fid=47 has vocab=278M (item_id). Current hash500k already
#   maps 278M IDs into 500k buckets (~560 IDs/bucket collision). EDA
#   also found: model AUC drops 0.04-0.05 on information-rich
#   samples (sparse user AUC 0.89 vs rich 0.85), indicating the
#   model overfits to specific train-period IDs. More aggressive
#   hashing (100k buckets, ~2780 IDs/bucket) should force additional
#   implicit regularisation via collision smoothing.
#
# Expected outcomes:
#   * Dev under gap=0.05 ≈ [0.843, 0.851] (compared to
#     icq-half-gap05's expected ~0.848)
#   * Dev→LB gap should remain ~0.010 (same funnel as baseline)
#   * LB target ≥ icq-half-gap05 LB + 0.003 for "real signal"
#
# ⚠️ Platform note: 100k hash vocab is a sparse embedding size
# reduction. This may speed up training (less embedding lookup) by
# ~10-15%. Quota budget ~5h.
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
    --seq_hash_vocab 100000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --valid_gap_ratio 0.05 \
    --enable_row_valid_gap \
    --item_conditioned_query \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
