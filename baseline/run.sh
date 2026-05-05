#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: disablefid47-icq-halftrain-swiglu-vr005-bs128
#
# T23 · Structural disable of seq_c feat47 (the 278M-vocab item_id).
#
# Single-variable ablation vs current LB SOTA:
#   Baseline: icq-t13tf-halftrain-swiglu-vr005-bs128 (E4 LB 0.84637)
#   Change:   +--disable_seq_fids 47
#             (every row zeros out the seq_c/fid=47 slot at train /
#              valid / inference; model sees no item_id sequence)
#
# Motivation (community Seed LB 0.85+ main thread, problem 5):
#   "长序列: seqc feat47 跟 itemid 是一样的, 做不做 match 特征都掉分"
# And our P1 ablation (EXP-025, idmaskc15 LB 0.842176 vs SOTA 0.84637)
# proved input-level random mask is mechanism-wrong: 85% unmasked
# tokens still let the model overfit to id embedding. The next
# logical step is a full structural removal of the id channel.
#
# Hypothesis: If item_id in seq_c is indeed noise (Seed's claim) and
# the dev→LB gap is driven by id OOV on test, then removing the id
# column entirely should:
#   * leave dev AUC roughly flat or slightly down (~-0.0005 to 0.000)
#   * shrink dev→LB gap (model no longer relies on fragile id signal)
#   * produce LB ≥ 0.846 (parity) or > 0.847 (SOTA break)
#
# Decision thresholds (LB vs SOTA 0.84637):
#   LB ≥ 0.847   → Accept, next round widen disable (add fid 29, 36)
#   LB ∈ [0.845, 0.847) → direction good, continue widening
#   LB ∈ [0.843, 0.845) → mixed signal, try seq_c full disable
#   LB < 0.843   → direction wrong, escalate T22-fix (row-level gap)
#
# Not stacked (single-variable discipline):
#   * --id_mask_prob         (P1 closed)
#   * --target_item_seq_injection  (T16 additive closed)
#   * --holdout_ratio        (independent cost unresolved)
#   * --valid_gap_ratio      (P3 blocked)
#
# Early-stop: 4 epoch + patience=2 + LogLoss rebound guard p=2.
# Expected wall-clock: ~5.5 hours (seq_c slot removed saves small
# amount of embedding lookup).
#
# ⚠️ Platform submission note:
# --disable_seq_fids is written to train_config.json. Platform's
# default infer.py will NOT parse this key, causing train/inference
# mismatch (inference sees full seq_c while training zero'd fid=47).
# **Must upload baseline/infer.py** (this repo's version, which
# contains _parse_disable_seq_fids).
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
    --disable_seq_fids 47 \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"

# ═══════════════════════════════════════════════════════════════════
# Fallback · P3 gap15-icq-halftrain (Blocked, RG-level gap failed)
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
#     --valid_gap_ratio 0.15 \
#     --item_conditioned_query \
#     --logloss_rebound_patience 2 \
#     --logloss_rebound_delta 0.0 \
#     --batch_size 128 \
#     --num_epochs 4 \
#     "$@"
#
# ═══════════════════════════════════════════════════════════════════
# Fallback · P1 id_mask c=15 (EXP-025, closed, LB 0.842176)
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
#     --item_conditioned_query \
#     --id_mask_prob 0.15 \
#     --id_mask_seq_domains c \
#     --logloss_rebound_patience 2 \
#     --logloss_rebound_delta 0.0 \
#     --batch_size 128 \
#     --num_epochs 4 \
#     "$@"
#
# ═══════════════════════════════════════════════════════════════════
# Fallback · original icq-half (LB 0.84637 SOTA)
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
#     --item_conditioned_query \
#     --batch_size 128 \
#     --num_epochs 4 \
#     "$@"
