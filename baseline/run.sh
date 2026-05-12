#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: M5-t25bypass-gap0-seed3-4ep
#
# ⭐ M5 · 今晚最高 ROI bet (最可能破 SOTA 0.84637 → Top 20)
#
# T25 DIN bypass × gap=0 × seed=3 · 首次组合。从未被 eval 过:
# - EXP-036 B (t25-din-seed3-halftrain-gap0-4ep) Dev 0.867661 但 ckpt
#   5/11 CEST 10:00 过期清理，LB 永久丢失 → 现在重跑重建
# - 历史 seed=3 × gap=0 × no-DIN LB = 0.845329 (第 2 高)
# - DIN bypass 在 gap=0.05 证 +0.004 LB (T25 B 0.843433 vs baseline 0.839223)
# - 叠加预期: LB ≈ 0.846~0.850 (Top 20 阈值 0.850)
#
# 和今天 M1/M2 (T28 integrated) 的差异:
# - M1/M2 用 --enable_din_integrated (DIN 进 block 内部 × 2 次)
# - M5 用 --enable_din_interest (DIN bypass, classifier 前 1 次)
# - 今日 M1 LB 0.841807 < T25 bypass LB 0.843069 → T28 integrated 证伪
# - M5 = 回归 T25 bypass (已证 work) + 切换到 gap=0 SOTA 领地
#
# Go/No-go:
#   Dev ≥ 0.869 (超过 EXP-019 seed=42 gap=0 no-DIN E4 0.868619) → 🎯 eval
#   Dev ∈ [0.867, 0.869) → 🟡 eval, 预期 LB 0.844~0.847
#   Dev < 0.867 → ⚪ 持平 baseline, 不评
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
    --enable_din_interest \
    --seed 3 \
    --batch_size 128 \
    --num_epochs 4 \
    --logloss_rebound_delta 0.003 \
    --logloss_rebound_patience 2 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
