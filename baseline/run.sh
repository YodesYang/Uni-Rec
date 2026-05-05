#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: gap15-icq-halftrain-swiglu-vr005-bs128
#
# Single-variable ablation vs current LB SOTA:
#   Baseline: icq-t13tf-halftrain-swiglu-vr005-bs128 (E4 LB 0.84637,
#             current SOTA as of 2026-05-04)
#   Change:   +--valid_gap_ratio 0.15
#             (Funnel Redesign P3 / ADR-005 alt C: drop 15% of Row
#              Groups between train tail and dev head so dev no longer
#              sits immediately after train. Goal: make dev→LB
#              correlation tighter by forcing dev into a "mini future"
#              regime that mimics the real future-dated test set.)
#
# Hypothesis (EXP-024 diagnostic): the persistent +0.021~0.030 dev→LB
# gap is partly because current time-split puts dev "adjacent" to
# train tail, so dev reflects "near-future continuation of train"
# rather than "far-future extrapolation to test". Leaving a 15% gap
# region (dropped entirely) pushes dev away by ~half a day to a day
# of data, which should shrink dev→LB gap even if dev absolute AUC
# drops.
#
# Expected signals:
#   * E4 dev AUC ~0.861~0.866 (LOWER than icq-half's 0.868619 — this
#     is normal; train lost 15% data + dev moved further out)
#   * E4 LB ≥ 0.846 (parity with SOTA is a win — proves P3 works)
#   * dev→LB gap shrinks from +0.022 (icq-half) to < +0.018 (P3
#     success signal). If gap ≥ +0.020 → P3 failed regardless of
#     LB absolute value.
#   * If E4 dev drops below 0.858 → train data loss hurts too much,
#     scale gap down to 0.10 on next iteration.
#
# Early-stop budget: 4 epoch + patience=2, same as icq-half. LogLoss
# rebound guard on with patience=2.
#
# Not stacked on this run (single-variable discipline):
#   * --id_mask_prob               (P1, own ablation in progress)
#   * --target_item_seq_injection  (T16, full train in progress)
#   * --holdout_ratio              (independent cost not yet verified)
#
# Evaluation target (if quota allows): compare E4 LB and dev→LB gap
# directly against icq-half E4 (LB 0.84637 / gap +0.022). Even if
# E4 LB < 0.847 but gap < 0.018, P3 is validated as a funnel
# improvement and should become default for future ablations.
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
    --valid_gap_ratio 0.15 \
    --item_conditioned_query \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"

# ═══════════════════════════════════════════════════════════════════
# Fallback · P1 id_mask c=15 (previous submission, currently running)
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
# Fallback · original icq-half (LB 0.84637 SOTA, pure baseline)
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
