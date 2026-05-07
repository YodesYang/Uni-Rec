#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: din-gap05-icq-halftrain-swiglu-vr005-bs128
#
# T25 / G1 · DIN-style target-aware interest pooling.
#
# Baseline: icq-half-gap05-icq-halftrain (Dev 0.842443, LB 0.839223)
# Change:   +--enable_din_interest
#
# Mechanism:
#   DINInterestExtractor: 4 cross-attentions (one per seq domain), each
#   uses mean(item_ns_tokens) as single-token query and attends to the
#   full seq_i tokens to extract a domain-specific interest vector.
#   The 4 interest vectors are concatenated and projected to (B, D),
#   then concatenated to the backbone output before the classifier
#   (bypass mode, identical pattern to --enable_dense_bypass).
#
# Why this is different from T16 additive (falsified):
#   T16 adds a single projected target vector to every seq token with
#   scalar α. Low expressiveness, uniform weighting.
#   DIN performs attention pooling: target query selectively weights
#   which seq positions are relevant. Matches Alibaba DIN (KDD 2018).
#
# Extra params: ~100k dense params (4 cross-attentions @ d_model=64 +
# merge MLP). Training adds ~5-10% compute; no effect on inference.
#
# Decision thresholds (ΔLB vs icq-half-gap05 LB 0.839223):
#   ≥ +0.005   🎯 Real signal (beats seed noise ±0.005), DIN Accept.
#   +0.002..+0.005  Suspected positive, need multi-seed confirmation.
#   -0.002..+0.002  Flat (inside noise), no signal.
#   < -0.002   Negative, DIN direction closed.
#
# ⚠️ Platform note: --enable_din_interest is a PCVRHyFormer.__init__
# kwarg. Platform default infer.py must recognize it or the model
# would build without the din_extractor at inference (train/infer
# mismatch). Upload baseline/infer.py (this version reads
# cfg.get('enable_din_interest', False)).
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
    --enable_din_interest \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
