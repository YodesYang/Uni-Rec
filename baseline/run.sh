#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t27-compile-amp-proxy-smoke
#
# T27 · torch.compile + AMP bf16 platform smoke.
#
# Goal: verify on GPU that --enable_torch_compile + --enable_amp
# (bf16) don't crash on a fast-proxy configuration before launching a
# full training run. Target the exact ICQ+swiglu+time-split config
# that holds our LB SOTA (0.84637), so we can compare step times
# apples-to-apples against historical proxy runs.
#
# Expected: same loss trajectory as non-AMP proxy (bf16 is numerically
# equivalent to fp32 for BCE), step time drop of 1.5~2.5x on A100,
# no OOM, no dynamo tracebacks, saved best_model can be loaded by
# platform default infer.py (raw_model.state_dict() has no
# _orig_mod. prefix -- verified in local smoke).
#
# If this passes: promote to a full 4~8 epoch training job with
# --enable_torch_compile --enable_amp as the default acceleration
# baseline going forward.
#
# Change vs icq-half-seed3-rerun-2: + --enable_torch_compile
# + --enable_amp + fast-proxy limits (limit_train_rgs / max_train_steps).
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
    --seed 42 \
    --batch_size 128 \
    --num_epochs 1 \
    --limit_train_rgs 50 \
    --max_train_steps 500 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
