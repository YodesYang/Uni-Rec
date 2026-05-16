# M55-M58 Run.sh Templates · 5/16 第 2 批 Overnight (CEST ~22:00 起)

> **Date**: 2026-05-16 CEST 19:30 起草
> **Trigger**: 用户深度思考目标 0.853 · 4 个新候选基于 path B 真信号体系探索
> **使用**: M51-M54 完成后 (~CEST 22:00) · 启动第 2 批 4 jobs · 5/17 早 dev 数据齐全 · 选最强 evaluate

---

## 战略 Context

**M49 LB 0.848162** = path B 单变量真信号 · 距 0.853 还差 +0.0048

**M51-M54 (in-flight)**: path B + M41 stack / per-domain bucket / NLIR / per-domain alone
**M55-M58 (本批)**: 4 个**未触碰过**的真信号方向

| Job | Config | 类型 | 期望 LB Δ | 真信号概率 |
|---|---|---|---|---|
| **M55** | M22 + UE split + UE × user_int Bilinear | Seed 主楼后半句 (T39 新代码) | +0.001~+0.002 | 35% |
| **M56** | M22 + UE split + num_hyformer_blocks=3 | 深度增加 (0 code) | +0.0005~+0.001 | 20% |
| **M57** | M22 + UE split + d_model=128 | embedding 维度加倍 (0 code · 显存增) | +0.001~+0.002 | 25% |
| **M58** | M22 + UE split + seq_a/b 升 512 | 序列长度增 (0 code) | +0.0003~+0.001 | 15% |

**期望产出**: 1~2 个真信号 (期望值 = 0.95)

**累加上限估算 (假设 M55-M58 + M51 全 work)**:
- M49 base = 0.848162
- + M51 (+0.0008) = 0.849
- + M55 (+0.0015) = 0.8505
- + cross-seed 上沿 (5/22+ +0.001) = **0.8515**
- 仍**不到 0.853** · 但接近

→ **0.853 现实路径**: M55-M58 至少 1 个 +0.002+ 强真信号 + cross-seed luck → 0.852~0.853

---

## ⚠️ 通用注意事项 (4 个 job 都适用)

1. **Create New Training Job** (不要 Copy)
2. **30s Args verify** 后立即贴 log 给 agent
3. **上传文件清单** (每个 job 都上传):
   - `src/dataset.py` (含 T38 per-domain bucket)
   - `src/train.py` (含 T39 --enable_ue_int_bilinear CLI)
   - `src/infer.py` (含 T39 cfg fallback)
   - `src/model.py` (含 T39 ue_int_bilinear_proj · 已 smoke 5/5 PASS)
   - `src/trainer.py`
   - `ns_groups_vocab_tier.json`
4. **不上传 run.sh** · 仅在平台编辑框粘贴
5. **M55 evaluation 时必须勾选自定义 infer.py** (T39 新 kwarg · 平台 default infer.py 不识别)
6. **M56-M58 evaluation 不需要勾选自定义 infer.py** (无新 kwarg · 仅改超参)

---

## M55 · M22 + UE split + UE × user_int Bilinear (T39 · Seed 主楼后半句)

**Job Name**: `M55-m22-uesplit-bilinear-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `enable_ue_int_bilinear: True` ⭐⭐
- ✅ `ue_int_bilinear_alpha_init: 0.5`

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
    --enable_ue_int_bilinear \
    --ue_int_bilinear_alpha_init 0.5 \
    "$@"
```

---

## M56 · M22 + UE split + num_hyformer_blocks=3 (深度增加)

**Job Name**: `M56-m22-uesplit-blocks3-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `num_hyformer_blocks: 3` ⭐
- ⚠️ Param 增加 ~50% · 显存可能增 · 注意 OOM (但 batch_size=128 应该 OK)

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
    --num_hyformer_blocks 3 \
    "$@"
```

---

## M57 · M22 + UE split + d_model=128 (embedding dim 加倍)

**Job Name**: `M57-m22-uesplit-dmodel128-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `d_model: 128` ⭐
- ✅ `emb_dim: 128` ⭐
- ⚠️ T-divisibility: T = 2×4 + (5+1+2) = 16 · 128 % 16 = 0 ✅
- ⚠️ 显存大幅增加 · 必须降低 batch_size 到 64
- ⚠️ Param 增加 ~3× · wall-time 可能 +50%

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
    --batch_size 64 \
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
    --d_model 128 \
    --emb_dim 128 \
    "$@"
```

⚠️ **如果 OOM** (显存不足) → 降 batch_size 到 32 · 或换 seq_c=256 / seq_d=256。

---

## M58 · M22 + UE split + seq_a/b 升 512 (序列长度增)

**Job Name**: `M58-m22-uesplit-seqab512-gap0-seed42-4ep`

**Args verify (30s 后必看)**:
- ✅ `enable_ue_split: True`
- ✅ `seq_max_lens: "seq_a:512,seq_b:512,seq_c:512,seq_d:512"` ⭐
- ⚠️ 训练时间 +30% · 完成时间晚一些

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
    --seq_max_lens "seq_a:512,seq_b:512,seq_c:512,seq_d:512" \
    --enable_ue_split \
    --ue_split_fid 61 \
    "$@"
```

---

## 启动顺序建议 (CEST ~22:00 5/16)

1. **Slot 1: M55** (path B + Bilinear · ROI 最高 · T39 新代码) → 先启
2. **Slot 2: M56** (blocks=3 · 0 code · 风险低) → 紧接启
3. **Slot 3: M58** (seq_a/b 升 512 · 0 code · 中风险) → 第三启
4. **Slot 4: M57** (d_model=128 · 0 code · OOM 风险) → 最后启 · 如 OOM 降 batch_size

→ 4 jobs 启动间隔 ~30s · 总启动时间 ~2 min

**完成时序预估** (CEST):
- M55/M56/M58: ~CEST 03:00 5/17 (5h)
- M57 (d_model=128 + bs=64): ~CEST 04:00~05:00 5/17 (6~7h · 2× wall-time)

→ **5/17 早 quota refresh (CEST 06:00) 时 4 jobs 全部完成 · dev 齐全**

---

## 30s Args verify checklist 模板

| Key | M55 | M56 | M57 | M58 |
|---|---|---|---|---|
| `enable_ue_split` | True | True | True | True |
| `ue_split_fid` | 61 | 61 | 61 | 61 |
| `enable_ue_int_bilinear` | **True ⭐** | False | False | False |
| `ue_int_bilinear_alpha_init` | 0.5 | (default) | (default) | (default) |
| `num_hyformer_blocks` | 2 | **3 ⭐** | 2 | 2 |
| `d_model` | 64 | 64 | **128 ⭐** | 64 |
| `emb_dim` | 64 | 64 | **128 ⭐** | 64 |
| `batch_size` | 128 | 128 | **64 ⭐** | 128 |
| `seq_max_lens` | seq_d:512 | seq_d:512 | seq_d:512 | **seq_a/b/c/d:512 ⭐** |
| `seed` | 42 | 42 | 42 | 42 |

任何 key 错 → cancel + restart

---

## 5/17 早 quota matrix (终版终版 · 含 M55-M58)

5/17 早 4 quota 候选 (按 dev 待出后选 4 强):

**已知 dev** (今晚 in-flight):
- M51 / M52 / M53 / M54: ~CEST 22:00 完成
- M55 / M56 / M57 / M58: ~CEST 03:00~05:00 完成
- 总 8 个新 candidates dev 数据 + 老候选 (M37/M38v2/M30/M27v2/M50)

**5/17 早 4 quota 决策**:
- 选 8 个新候选 dev 最高的 2~3 个 evaluate
- 留 1~2 quota 给老候选 (M37/M38v2)
- 不评 dev 显著负的候选 (大概率 LB 同向负)

**LB 期望**: 1~2 个真信号 + 老候选闭环 = 期望最强 LB 0.848~0.851

---

## 关键 Caveat (5/17 早)

- ⚠️ **M55 evaluation 必须勾选自定义 infer.py** (T39 新 kwarg · 平台 default 不识别)
- ⚠️ **M56-M58 evaluation 不需要勾选** (无新 kwarg)
- ⚠️ **ckpt 24h 过期防护**: M55-M58 完成后 (~CEST 03:00 5/17) · 5/17 中午前 publish 即可 (24h = 5/17 ~03:00 5/18)
