# M51-M54 Run.sh Templates · 5/16 Overnight Stack Batch

> **Date**: 2026-05-16 CEST 16:50 起草
> **Trigger**: EXP-061 path B stack overnight 实施
> **使用**: 平台 Create New Training Job · run.sh 编辑框粘贴对应模板 · 上传文件清单见每个 job

---

## ⚠️ 通用注意事项 (4 个 job 都适用)

1. **Create New Training Job** (不要 Copy · 因为 Args 改动 · 防 silent copy bug)
2. **30s Args verify**: 启动后立即贴 Args dict log · agent 检查关键 keys
3. **上传文件清单**:
   - `src/dataset.py` (含 T38 fit_log_scale_boundaries + _scan_per_domain_time_diffs · per-domain bucket)
   - `src/train.py` (含 --enable_per_domain_buckets CLI)
   - `src/infer.py` (含 _resolve_bucket_boundaries helper)
   - `src/model.py` (含 enable_ue_split T37 · 已 publish ckpt 用过)
   - `src/trainer.py`
   - `ns_groups_vocab_tier.json`
4. **不上传 run.sh** · run.sh 仅在平台编辑框粘贴
5. **不勾选自定义 infer.py** · 平台 default infer.py 已识别 enable_ue_split · per-domain bucket 通过 train_config.json 自动恢复 (T38 设计 backwards compatible)
   - ⚠️ 但 **如果 5/17 早 evaluate M52/M54 (用 T38 per-domain bucket 的 ckpt)** · **必须勾选自定义 infer.py** 因为 default infer.py 不识别 fitted_per_domain_bucket_boundaries config key

---

## M51 · M22 + UE split + temporal_weight 2.5 (path B + M41 stack)

**Job Name**: `M51-m22-uesplit-tempweight2.5-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `temporal_weight_alpha: 2.5`
- ✅ `enable_per_domain_buckets: False` (M51 不用 per-domain bucket)

```bash
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
    --enable_ue_split \
    --ue_split_fid 61 \
    --temporal_weight_alpha 2.5 \
    "$@"
```

---

## M52 · M22 + UE split + per-domain time bucket (path B + T38 修复 M28)

**Job Name**: `M52-m22-uesplit-perdomain-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `enable_per_domain_buckets: True` ⭐
- ✅ `per_domain_bucket_sample_rgs: 8`
- ⚠️ **Log 必看**: `[T38] per-domain bucket fit · domain=seq_X n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...] last_5_boundaries=[...]` (4 行 · 4 domain) · 验证 fit 成功

```bash
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
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_per_domain_buckets \
    --per_domain_bucket_sample_rgs 8 \
    "$@"
```

---

## M53 · M22 + UE split + NLIR (path B + Linear gating stack)

**Job Name**: `M53-m22-uesplit-nlir-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `enable_nlir_gating: True` ⭐
- ✅ `enable_per_domain_buckets: False`

```bash
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
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_nlir_gating \
    "$@"
```

---

## M54 · M22 + per-domain time bucket alone (M28 修复单变量 · 无 path B)

**Job Name**: `M54-m22-perdomain-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: False` (M54 不开 UE split · 单独验 per-domain bucket)
- ✅ `enable_per_domain_buckets: True` ⭐
- ✅ `per_domain_bucket_sample_rgs: 8`
- ⚠️ **Log 必看**: `[T38] per-domain bucket fit ...` (4 行 · 4 domain)

```bash
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
    --enable_per_domain_buckets \
    --per_domain_bucket_sample_rgs 8 \
    "$@"
```

---

## 启动顺序建议

**用户操作顺序** (按 4 free slot 启 4 jobs):

1. **Slot 1: M51** (无新代码 · 最稳) → 先启
2. **Slot 2: M53** (无新代码) → 紧接启
3. **Slot 3: M54** (有 T38 新代码 · 但单变量风险低) → 启 · 30s 看 Args + T38 fit log
4. **Slot 4: M52** (有 T38 新代码 + path B stack) → 启 · 30s 看 Args + T38 fit log

→ 4 jobs 启动间隔 ~30s · 总启动时间 ~2 min

**完成时序预估** (CEST):
- M51/M53 (无 T38 fit) · 完成 ~CEST 21:30 5/16 (5h)
- M52/M54 (含 T38 fit · ~30s 启动期延长 · 可忽略) · 完成 ~CEST 21:30 5/16

→ **5/17 早 quota refresh (CEST 06:00) 时 4 jobs 全部完成 · dev 数据齐全**

---

## 30s Args verify checklist 模板 (用户启每个 job 后立即贴 log)

agent 必须检查的 keys:

| Key | M51 | M52 | M53 | M54 |
|---|---|---|---|---|
| `enable_ue_split` | True | True | True | False |
| `ue_split_fid` | 61 | 61 | 61 | (N/A) |
| `temporal_weight_alpha` | 2.5 | 0.0 | 0.0 | 0.0 |
| `enable_per_domain_buckets` | False | True ⭐ | False | True ⭐ |
| `per_domain_bucket_sample_rgs` | (default 8) | 8 | (default 8) | 8 |
| `enable_nlir_gating` | False | False | True ⭐ | False |
| `seed` | 42 | 42 | 42 | 42 |
| `valid_ratio` | 0.05 | 0.05 | 0.05 | 0.05 |
| `valid_split_strategy` | time | time | time | time |

任何 key 错 → cancel + restart (silent copy bug 教训)

---

## 启动后 (CEST ~20:30 5/16) agent 监控脚本

agent 在用户启 4 jobs 后:
1. 用户贴 4 jobs Args dict log (30s 后)
2. agent verify 上述 checklist
3. 用户贴 4 jobs 早期 progress log (~CEST 22:00 · E1 dev) 给 agent
4. 5/17 早 ~CEST 02:00 用户贴 4 jobs final summary
5. agent 选 dev 最高 1~2 个 + M37/M38v2 形成 5/17 早 4 quota matrix
