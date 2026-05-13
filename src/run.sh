#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: M17-t25bp-gap0-seed42-seqd768-4ep
#
# 🟢 M17 · DECEM Trick 2 · seq_max_lens 盲调 · seq_d 512→768
#
# 基于 M9 (seed=42 · gap=0 · T25 DIN bypass · LB 0.84671 已验证 SOTA 之一),
# 改单一变量: --seq_max_lens 中 seq_d 从 512 加长到 768。
#
# 单变量设计:
# - 基座 = M9 完全相同 config (保证 LB 0.84671 基线可归因)
# - 只动 seq_d (512→768), seq_a/b/c 保持默认
# - 任何 ΔLB 归因到 "seq_d 加长" 单一变量
#
# 方向选择理由:
# - DECEM Trick 2 原话: "按 p99 分析后域特定 seq_max_lens" · +0~0.002 LB
# - 社区情报 "牛马艺术家" 亲测 "从 1500 回到 1024 涨点" → 加长并非越多越好
# - 选 768 (512→768, 1.5×) 而非翻倍 (512→1024) · 避开"过长反降"陷阱
# - 选 seq_d 而非 seq_c: seq_c = item_id 序列 (Seed 帖子证实信号稀薄, 加长 ROI 低)
# - seq_d 是 cross-domain 补充信号, 加长更可能抓长期行为
#
# M16 互补:
# - M16 正在跑 EDA (会给出真 p99 数据) · 晚上回流后做精调版 M18
# - M17 = 先用 "保守加长" 盲调占位, 并行验证方向对不对
#
# Go/No-go:
#   Dev ≥ 0.870 (明显超 M9 基线 0.868619) → 🎯 eval · 预期 LB 0.847~0.850
#   Dev ∈ [0.868, 0.870) → 🟡 eval · 预期 LB 0.845~0.848 (和 M9 同级)
#   Dev < 0.868 → ⚪ seq_d 加长无收益, Trick 2 方向值得保留到 M16 EDA 后再精调
#
# 资源预估 (vs M9):
# - 训练时间: +10~15% (seq_d 从 512→768, 长度 50%↑ 但 attn 复杂度 ~2.25×↑, batch 内摊薄)
# - GPU 显存: +5~10% (仅 seq_d 变长, 其他域不变)
# - 预计总时长: ~9.5~10.5h (vs M9 ~9h)
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
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:768" \
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
    "$@"
