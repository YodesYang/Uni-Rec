#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
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
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_per_domain_buckets \
    --per_domain_bucket_sample_rgs 64 \
    --temporal_weight_alpha 2.5 \
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
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    "$@"
