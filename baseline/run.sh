#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t16-add-a01-ho005-lgrp1-full-swiglu-vr005-bs128
#
# T16-only full train (10 epoch) as "best known path" fallback. Reason:
#   * icq-t13tf-halftrain E4 → LB 0.84637 (current SOTA, 2026-05-04)
#   * T16-only half-train E3 dev=0.868061 is already +0.00036 vs t13tf E3,
#     likely higher local ceiling than ICQ when extended to full train
#   * T18 (BPR) + T19 (per-token FFN) halftrain both lagging (see
#     t16-add-a01-t18-auc-w03-... E4 dev 0.865516 / holdout 0.867196)
#     — T19 is confirmed negative, T18 is at best neutral on holdout
#   * Strategy: commit to the single bullet that demonstrably works
#     (T16 additive target injection), protect with stricter logloss guard
#     (patience=1, vs halftrain's patience=2) to avoid icq-full's
#     E5→E6 overfitting trap that cost LB −0.00277
#
# Hitch-ride: holdout_ratio=0.05 for continued ADR-005 calibration
#
# Early-stop budget: logloss_rebound_patience=1 + standard AUC patience=5.
# Based on icq-full's E5 rebound pattern, expected stop at E5-E7 with
# best_model = E4 or E5 ckpt.
#
# Evaluation target: best ckpt LB ≥ 0.848 (current SOTA 0.84637 + 0.0015)
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
    --holdout_ratio 0.05 \
    --target_item_seq_injection additive \
    --target_inject_alpha_init 0.1 \
    --logloss_rebound_patience 1 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 10 \
    "$@"

# ═══════════════════════════════════════════════════════════════════
# Fallback · 4-in-1 halftrain (T16+T18+T19+T17) — archived for comparison
# ═══════════════════════════════════════════════════════════════════
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --seq_encoder_type swiglu \
#     --seq_hash_vocab 500000 \
#     --enable_time_of_day_features \
#     --valid_split_strategy time \
#     --valid_ratio 0.05 \
#     --holdout_ratio 0.05 \
#     --target_item_seq_injection additive \
#     --target_inject_alpha_init 0.1 \
#     --loss_type auc \
#     --auc_loss_weight 0.2 \
#     --auc_margin 0.0 \
#     --auc_max_pairs 256 \
#     --rank_mixer_ffn_mode per_token \
#     --logloss_rebound_patience 2 \
#     --logloss_rebound_delta 0.0 \
#     --batch_size 128 \
#     --num_epochs 4 \
#     "$@"
#
# ═══════════════════════════════════════════════════════════════════
# Fallback · baseline-only (no new bullets, reproduces t13tf-full EXP-018)
# ═══════════════════════════════════════════════════════════════════
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
