#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# 🛡️ AGENT NOTICE · 平台 run.sh 提交前 ABSOLUTELY NEEDS:
#   1. Read this file FULLY before drafting any new run.sh template.
#   2. NEVER reconstruct from memory — fork from this file as the
#      canonical baseline, then change AT MOST 1~2 flags as the
#      single experimental variable.
#   3. STRUCTURAL flags (group A below) are immutable across all
#      jobs. Removing any of them = OOM / silent regression / lost
#      M9 SOTA base. See team-memory 2026-05-14 CEST 14:25 ironlaw.
#   4. Manually verify d_model % T == 0 where
#      T = num_queries × num_sequences (=4) + num_ns
#        = num_queries × 4 + (user_ns_tokens + has_user_dense
#                              + item_ns_tokens + has_item_dense).
#      Default config: T = 2×4 + (5+1+2+0) = 16, 64 % 16 == 0 ✅
#   5. ⚠️ 2026-05-15 CEST 19:15 ironlaw: M22 baseline uses
#      seq_d:512 (NOT 768). M17 was seq_d=768 era and is deprecated.
#      Forking from this file MUST verify Group B seq_max_lens
#      matches the intended baseline.
#
# Job context (current = M22 baseline · seq_d=512 · gap=0 · seed=42)
# ═══════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    `# ───── Group A · STRUCTURAL · DO NOT REMOVE ─────` \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups_vocab_tier.json" \
    --emb_skip_threshold 1000000 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --item_conditioned_query \
    --enable_din_interest \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 4 \
    --logloss_rebound_delta 0.003 \
    --logloss_rebound_patience 2 \
    --keep_topk_ckpt 4 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    --num_workers 8 \
    `# ───── Group B · EXPERIMENTAL · CHANGE PER JOB ─────` \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    `# ───── Group C · OPTIONAL · ENABLE PER EXPERIMENT ─────` \
    `#   --enable_row_valid_gap --valid_gap_ratio 0.05    # gap=0.05 path` \
    `#   --enable_dcn_cross                               # T34 DCN-V2 cross` \
    `#   --enable_dense_bypass                            # T25 dense bypass` \
    `#   --enable_din_integrated                          # T28 (deprecated)` \
    `#   --enable_nlir_gating                             # T33 (deprecated)` \
    `#   --enable_fafe                                    # FaFE (deprecated)` \
    `#   --enable_is_missing --is_missing_item_int_fids "..."  # T31` \
    "$@"
