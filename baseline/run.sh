#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t16-add-a01-t18-auc-w02-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128
#
# Four orthogonal bullets + holdout hitchhike, one half-train (4 epoch):
#   T16 / ADR-004 · additive target-item injection into seq tokens (α init 0.1)
#   T18 / ADR-006 · CombinedAUCLoss = 0.8·BCE + 0.2·PairwiseBPR (margin=0)
#   T19 / ADR-007 · RankMixer per-token FFN (D=64, T=16, +496k dense/block)
#   T17 / ADR-005 · holdout split (5%) for dev→LB gap calibration
#
# Baseline inherited from t13tf-full (EXP-018): swiglu + hash500k + timefeat
#   + rankmixer NS tokenizer (user_ns=5, item_ns=2, num_queries=2) + vr=0.05
#
# Guards:
#   logloss_rebound_patience=2 — per-token FFN (+15× capacity) + BPR
#     gradient noise can perturb LogLoss earlier; relax from default 1.
#
# Notes:
#   * New PCVRHyFormer kwargs (target_item_seq_injection, rank_mixer_ffn_mode)
#     → MUST upload the full baseline/*.py folder AND custom infer.py.
#     Do NOT rely on platform default infer.py (team-memory rule).
#   * Start conservative: auc_margin=0.0 + auc_weight=0.2 (T18 vanilla BPR,
#     smallest ranking perturbation). Escalate to margin=0.5 / w=0.3 only
#     after baseline convergence confirmed.
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
    --loss_type auc \
    --auc_loss_weight 0.2 \
    --auc_margin 0.0 \
    --auc_max_pairs 256 \
    --rank_mixer_ffn_mode per_token \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"

# ═══════════════════════════════════════════════════════════════════
# Fallback · previous single-bullet config (T16 only, preserved for ablation)
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
#     --target_item_seq_injection additive \
#     --target_inject_alpha_init 0.1 \
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
