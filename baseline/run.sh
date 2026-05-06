#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: icq_seed7-icq-halftrain-swiglu-vr005-bs128
#
# Multi-seed verification · fourth data point for the "SOTA seed luck"
# hypothesis (EXP-028). With seed=42/2/3/7 we'll have 4 LB points on
# bit-identical config, enough to characterize the seed noise band.
#
# Change from M1: --seed 7 (instead of 3). Everything else identical
# to icq-half SOTA config.
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
    --item_conditioned_query \
    --seed 7 \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
