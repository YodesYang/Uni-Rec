#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: rowgap15-icq-halftrain-swiglu-vr005-bs128
#
# C6 / T22-fix · row-level valid gap cutoff (fix EXP-026 failure).
#
# Single-variable ablation vs icq-half SOTA:
#   Baseline: icq-t13tf-halftrain-swiglu-vr005-bs128 (E4 LB 0.84637)
#   Change:   +--enable_row_valid_gap --valid_gap_ratio 0.15
#             (row-level double cutoff: [train | gap dropped | dev]
#              split by row-wise timestamp quantiles, NOT RG-level)
#
# EXP-026 found RG-level gap ineffective because each RG spans ~4
# days (gap and train time ranges completely overlap). This C6 fix
# sorts every row by timestamp and picks the two boundary timestamps
# by exact row quantiles, guaranteeing a clean time gap even when RG
# internal time ordering is chaotic.
#
# Goal (funnel fix, not direct LB):
#   * E4 dev AUC likely drops to [0.861, 0.866] (train lost 15% rows
#     AND dev is now temporally further from train — both effects)
#   * dev→LB gap should shrink to < +0.018 if the covariate-shift
#     hypothesis is right; current gap is +0.022 (icq-half) to +0.030
#     (various ablations)
#   * LB itself may stay near 0.846 or drop; the value of this job
#     is the gap change, not the absolute LB
#
# Decision: if dev→LB gap drops below +0.018, adopt
# enable_row_valid_gap as default for subsequent ablations — future
# dev AUC will become a more reliable proxy for LB. If gap stays
# >= +0.020, the covariate-shift hypothesis is refuted; try other
# funnel fixes (triple-split, larger gap, etc).
#
# Not stacked:
#   * --id_mask_prob         (T21 closed)
#   * --disable_seq_fids     (T23 separate ablation running)
#   * --target_item_seq_injection  (T16 closed)
#   * --holdout_ratio        (mutually exclusive with enable_row_valid_gap)
#
# ⚠️ Platform submission note:
# --enable_row_valid_gap / --valid_gap_ratio are dataset-level flags
# (not model __init__ kwargs), so platform default infer.py is fully
# compatible. However, we still recommend uploading baseline/infer.py
# to keep the evaluation-side code path identical to train (avoid
# version drift with the platform's infer.py).
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
    --enable_row_valid_gap \
    --item_conditioned_query \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
