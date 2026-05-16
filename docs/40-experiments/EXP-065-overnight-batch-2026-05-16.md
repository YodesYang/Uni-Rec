# EXP-065 · 5/16 晚 4 free slots Overnight Batch · M60/M55/M54'/M52' · 2026-05-16

> **Status**: 🟡 4 jobs to launch ~CEST 23:30 5/16 · 用户 23:00 决断路径 A
> **Date**: 2026-05-16 CEST 23:35 起草
> **Trigger**: M51-M54 全部 dev 回流 (M52/M54 因 T38 silent skip bug 实际 = M49/M22 rerun · commit `3b06e1d` 已修复) · 4 free slots overnight 战略
> **Related**: EXP-061 (M51-M54 第 1 批 path B stack) · EXP-062 (M55-M58 战略起草) · EXP-064 (三滤镜框架) · `3b06e1d` (T38 fix commit)
> **Decision lineage**: 用户 5/16 22:48 选 "现在修复 T38" → 23:30 commit `3b06e1d` push → 23:00 用户决路径 A + fork_all → 本 EXP

---

## TL;DR · 4 jobs 战略

**目标**: 多样化 + 修复 5/16 早事故 + 三滤镜框架 ≥ 3/4 全过

| Job | Config | 类型 | 三滤镜 | 期望 dev | 期望 LB | 真信号概率 |
|---|---|---|---|---|---|---|
| **M60** | M22 + UE split + Q_init=MLP(item_emb) | path B + DECEM Trick 6 | **4/4 全过** ⭐ | 0.870~0.872 | +0.0005~+0.001 | 35% |
| **M55** | M22 + UE split + UE × user_int Bilinear (T39) | path B + Seed 主楼后半句 | 3.5/4 (gating 引入新交互) | 0.869~0.871 | +0.0003~+0.0008 | 25% |
| **M54'** | M22 + per-domain time bucket alone (T38 fix 后) | M28 修复单变量 | 3.5/4 (T38 fix 平台首验) | 0.868~0.869 | +0~+0.0005 | 25% |
| **M52'** | M22 + UE split + per-domain time bucket (T38 fix 后) | path B + per-domain stack | 3.5/4 | 0.870~0.872 | +0.0003~+0.001 | 30% |

**累计真信号期望值** = 35% + 25% + 25% + 30% ≈ **1.15 真信号 / 4 jobs**

**5/16 晚 4 jobs 选择理由 (基于 EXP-064 三滤镜框架 + 三条铁律)**:
- ✅ **多样性** (EXP-035 5/10 铁律): 1 个修复事故 (M54') + 1 path B + per-domain (M52') + 1 path B + Q_init (M60) + 1 path B + Bilinear (M55)
- ✅ **DECEM "不要大改动"** (5/12 铁律): 排除 M56 (num_blocks=3) / M57 (d_model=128) / M58 (seq_a/b 升 512)
- ✅ **三滤镜降级 M59**: Multi-Emb K=2 + Global Token 论文锚 +0.21% online · 但滤镜 1+2 触发 · 修订期望 +0~+0.0008 · 不优先

**5/22 前 cross-seed 锁** (5/16 23:00 用户严守): M22/M49 × s11 不起。

---

## 1. 通用注意事项 (4 jobs 共用)

### 1.1 启动方式

1. **Create New Training Job** · 不 Copy (5/15 silent copy bug 铁律)
2. **30s Args verify**: 启动后用户立即贴 Args dict log → agent 检查 §6 关键 keys
3. **不上传 run.sh** (5/13 18:09 铁律 · 平台编辑框粘贴)
4. **平台默认 infer.py 兼容性**:
 - **M52'/M54'**: 因含 T38 `fitted_per_domain_bucket_boundaries` config key · **必须勾选自定义 infer.py** (上传 `src/infer.py`)
 - **M55**: T39 `enable_ue_int_bilinear` 平台首启 · **必须勾选自定义 infer.py**
 - **M60**: `q_init_item` flag M30 已验证平台默认 infer.py 兼容 · 可不勾选自定义 (但建议勾选稳妥)

### 1.2 上传文件清单 (4 jobs 共用)

```
src/dataset.py     # T38 silent skip fix 在 train.py · dataset.py 已 supports per-domain
src/train.py       # 含 T38 fix (commit 3b06e1d 已 push origin/main)
src/infer.py       # T38 _resolve_bucket_boundaries · T39 enable_ue_int_bilinear
src/model.py       # T37 enable_ue_split / T39 enable_ue_int_bilinear / q_init_item 全 ready
src/trainer.py     # 不变
src/utils.py       # 不变
src/infer_schema.py # 不变
ns_groups_vocab_tier.json  # M22 baseline 默认
```

⚠️ **必须确保上传的是 commit `3b06e1d` (含 T38 fix) 的 train.py** · 否则 M52'/M54' 又会触发 silent skip。
⚠️ **检查方式**: 平台 Job 启动后 30s · grep Args dict 是否含 `'enable_per_domain_buckets': True` (M52'/M54') + 后续 log 是否出现 `[T38] per-domain bucket fit · domain=seq_X n_samples=Y` 4 行 (而非 `Skipping per-domain fit` warning)

### 1.3 起动顺序建议 (按 risk 升序)

按 platform 占 slot 顺序启:

1. **Slot 1: M60** (CEST ~23:35) · 0 新代码 · 0 平台 risk · 启动最稳
2. **Slot 2: M55** (M60 启后 ~30s) · T39 平台首启 · 必勾自定义 infer.py · 30s 看 Args
3. **Slot 3: M54'** (M55 后 ~30s) · T38 fix 平台首验 · 单变量 risk 较低 · 30s 看 Args + 等 ~5 min 看 `[T38] per-domain bucket fit` 4 行 log
4. **Slot 4: M52'** (M54' Args + T38 fit log 都 OK 后 ~5 min) · T38 fix + path B stack · 双重新代码 + 修复 risk · 最后启

**预估完成时序 (CEST)**:
- M60 (无新代码): ~CEST 04:30 5/17 (5h)
- M55 (T39): ~CEST 04:30 5/17 (5h)
- M54'/M52' (T38 fit ~30s 启动期): ~CEST 04:30~05:00 5/17

→ **5/17 早 quota refresh CEST 06:00** 时 4 jobs 全部完成 dev · 5/18 早 quota 决策依据齐全。

---

## 2. 期望产出 + 5/18 早 quota 矩阵

### 2.1 4 jobs 期望产出 + 5/17 已锁 quota

5/17 早 quota (用户 22:40 决): M53 / M51 / M30 / M37 (M27v2/M50 备用)

5/18 早 quota 候选 (4 jobs LB + 5/17 LB 结果联合 audit):
- M60 LB (如 dev 强 · 5/18 必评)
- M52' LB (T38 fix 后真测 path B + per-domain bucket)
- M55 LB (T39 首测)
- M54' LB (per-domain bucket 单变量) 或 M50 备用

---

## 3. M60 · M22 + UE split + Q_init=MLP(item_emb) (DECEM Trick 6)

**Job Name**: `M60-m22-uesplit-qinit-gap0-seed42-4ep`

**机制**: T37 path B (UE split · M49 已 verified) + T32/DECEM Trick 6 (Q_init = MLP(item_emb) · query 端 item-aware) · 两机制完全正交 (T37 改 NS Tokenizer / T32 改 MultiSeqQueryGenerator)。

**与 M30 区别**: M30 = M22 + Q_init alone (无 UE split) dev 0.868438 · M60 = M30 + UE split = M30 改架构 base 加 path B。

**Args verify (启动后 30s 必查 · 5 keys)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `q_init_item: True` ⭐ (注意 Args dict 字段名是 `q_init_item` 不是 `enable_q_init_item`)
- ✅ `enable_ue_int_bilinear: False`
- ✅ `enable_per_domain_buckets: False`
- ✅ `multi_emb_k: 1`
- ✅ `temporal_weight_alpha: 0.0`
- ✅ `enable_global_token: False`
- ✅ `enable_nlir_gating: False`
- ✅ `seed: 42`
- ✅ `valid_ratio: 0.05`
- ✅ `valid_split_strategy: time`
- ✅ `enable_row_valid_gap: False`

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

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
    `# ───── Group B · EXPERIMENTAL ─────` \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    `# ───── Group C · M60 path B + Trick 6 stack ─────` \
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_q_init_item \
    "$@"
```

---

## 4. M55 · M22 + UE split + UE × user_int Bilinear (T39 · Seed 主楼后半句)

**Job Name**: `M55-m22-uesplit-bilinear-gap0-seed42-4ep`

**机制**: T37 path B (UE split) + T39 (UE 与 user_int pair Bilinear gating · Seed 主楼第 (1) 条后半句"其他跟 int pair 加权")。code commit `3f0821f` (5/16 17:00 · smoke 5/5 PASS · 含 zero-init bit-identical 验证)。

**Args verify (启动后 30s 必查 · 5 keys)**:
- ✅ `enable_ue_split: True`
- ✅ `ue_split_fid: 61`
- ✅ `enable_ue_int_bilinear: True` ⭐
- ✅ `ue_int_bilinear_alpha_init: 0.5`
- ✅ `q_init_item: False`
- ✅ `enable_per_domain_buckets: False`
- ✅ `multi_emb_k: 1`
- ✅ `temporal_weight_alpha: 0.0`
- ✅ `seed: 42`

⚠️ **必勾选自定义 infer.py** · T39 平台首启 (team-memory 5/4 T16 兼容性铁律)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

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
    `# ───── Group B · EXPERIMENTAL ─────` \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    `# ───── Group C · M55 path B + UE×user_int Bilinear ─────` \
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_ue_int_bilinear \
    --ue_int_bilinear_alpha_init 0.5 \
    "$@"
```

---

## 5. M54' · M22 + per-domain time bucket alone (T38 fix 后重起)

**Job Name**: `M54p-m22-perdomain-gap0-seed42-4ep` (注意 p = "prime · 修复后")

**机制**: M28 修复单变量 (无 UE split) · 验证 per-domain time bucket 是否真比全局 default ladder 好。EDA-056 §Q5 实证 4 序列时间跨度差 11× (seq_d p99=56天 vs seq_c p99=629天) · 全局 64-bucket 是中庸方案。

**vs 5/16 早失败的 M54**: M54 因 train.py line 832 silent skip bug · 实际跑 M22 rerun · dev 0.868830 落在 M22 noise band 内。**commit `3b06e1d` 已修复**。

**Args verify (启动后 30s + ~5 min · 6 keys + Plan B 验证 4 行 fit log)**:
- ✅ `enable_ue_split: False` (M54' 不开 path B · 单独验 per-domain)
- ✅ `enable_per_domain_buckets: True` ⭐
- ✅ `per_domain_bucket_sample_rgs: 8`
- ✅ `time_bucket_boundaries: ''` (不显式 supply · 否则触发 dataset.py line 3445 elif warning)
- ✅ `q_init_item: False`
- ✅ `enable_ue_int_bilinear: False`
- ✅ `temporal_weight_alpha: 0.0`
- ✅ `enable_nlir_gating: False`
- ✅ `seed: 42`

⚠️ **Plan B 必查 (T38 fix 平台首验)**: 启动后 ~5 min log 必须包含 4 行:
```
[T38] per-domain bucket fit · domain=seq_a n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...] last_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_b n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...] last_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_c n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...] last_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_d n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...] last_5_boundaries=[...]
```

⚠️ **如出现 `[T38] enable_per_domain_buckets=True but bucket_boundaries already supplied (likely from --time_bucket_boundaries CLI). Skipping per-domain fit; using supplied boundaries.` warning** → **立即 cancel** · 大概率上传的 train.py 不是 commit `3b06e1d` 版本。

⚠️ **必勾选自定义 infer.py** · T38 fitted_per_domain_bucket_boundaries config key 平台 default infer.py 不识别。

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

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
    `# ───── Group B · EXPERIMENTAL ─────` \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    `# ───── Group C · M54' per-domain bucket alone (T38 fix 后) ─────` \
    --enable_per_domain_buckets \
    --per_domain_bucket_sample_rgs 8 \
    "$@"
```

---

## 6. M52' · M22 + UE split + per-domain time bucket (T38 fix 后重起)

**Job Name**: `M52p-m22-uesplit-perdomain-gap0-seed42-4ep`

**机制**: path B (UE split) + per-domain time bucket 双 stack · 修复 5/16 早 M52 silent skip 事故 (M52 实际 = M49 rerun · dev 0.870718 落在 M49 noise band 内)。

**EXP-061 §3 信息价值**: M52' vs M54' dev 比较揭示 path B 与 per-domain bucket 互补性 (互补/无冲突/冲突 三种 scenario 都决断方向)。

**Args verify (启动后 30s + ~5 min · 7 keys + Plan B 验证 4 行 fit log)**:
- ✅ `enable_ue_split: True` ⭐
- ✅ `ue_split_fid: 61`
- ✅ `enable_per_domain_buckets: True` ⭐
- ✅ `per_domain_bucket_sample_rgs: 8`
- ✅ `time_bucket_boundaries: ''`
- ✅ `q_init_item: False`
- ✅ `enable_ue_int_bilinear: False`
- ✅ `temporal_weight_alpha: 0.0`
- ✅ `enable_nlir_gating: False`
- ✅ `seed: 42`

⚠️ **Plan B 必查**: 同 M54' · 4 行 `[T38] per-domain bucket fit` log 必须出现 · `Skipping per-domain fit` warning **不能出现**。
⚠️ **必勾选自定义 infer.py** · 同 M54'。

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

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
    `# ───── Group B · EXPERIMENTAL ─────` \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    `# ───── Group C · M52' path B + per-domain bucket (T38 fix 后) ─────` \
    --enable_ue_split \
    --ue_split_fid 61 \
    --enable_per_domain_buckets \
    --per_domain_bucket_sample_rgs 8 \
    "$@"
```

---

## 7. 4 jobs Args verify checklist (用户启动 4 jobs 后立即贴 log · agent 严格 grep)

| Key | M60 | M55 | M54' | M52' |
|---|---|---|---|---|
| `enable_ue_split` | True | True | False | True |
| `ue_split_fid` | 61 | 61 | (N/A · False) | 61 |
| `q_init_item` | **True ⭐** | False | False | False |
| `enable_ue_int_bilinear` | False | **True ⭐** | False | False |
| `enable_per_domain_buckets` | False | False | **True ⭐** | **True ⭐** |
| `per_domain_bucket_sample_rgs` | (default 8) | (default 8) | 8 | 8 |
| `time_bucket_boundaries` | `''` | `''` | `''` | `''` |
| `multi_emb_k` | 1 | 1 | 1 | 1 |
| `enable_global_token` | False | False | False | False |
| `enable_nlir_gating` | False | False | False | False |
| `enable_dcn_cross` | False | False | False | False |
| `enable_dense_bypass` | (auto · ue_split=True forces) | (auto) | False | (auto) |
| `temporal_weight_alpha` | 0.0 | 0.0 | 0.0 | 0.0 |
| `seed` | 42 | 42 | 42 | 42 |
| `valid_ratio` | 0.05 | 0.05 | 0.05 | 0.05 |
| `valid_split_strategy` | time | time | time | time |
| `enable_row_valid_gap` | False | False | False | False |
| `valid_gap_ratio` | 0.0 | 0.0 | 0.0 | 0.0 |
| `num_epochs` | 4 | 4 | 4 | 4 |
| `batch_size` | 128 | 128 | 128 | 128 |
| `keep_topk_ckpt` | 4 | 4 | 4 | 4 |
| `seq_max_lens` | seq_d:512 | seq_d:512 | seq_d:512 | seq_d:512 |

**任一 key 错** → cancel + restart (5/15 silent copy bug 教训)

---

## 8. T38 fix 平台首验 Plan B checklist (M54'/M52' 专属)

启动后 ~5 min · `[T38]` log 必查:

✅ **fit 路径触发** (期望 4 行):
```
[T38] per-domain bucket fit · domain=seq_a n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_b n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_c n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...]
[T38] per-domain bucket fit · domain=seq_d n_samples=Y p50=Zs p99=Ws first_5_boundaries=[...]
```

❌ **silent skip warning · 立即 cancel**:
```
[T38] enable_per_domain_buckets=True but bucket_boundaries already supplied (likely from --time_bucket_boundaries CLI). Skipping per-domain fit; using supplied boundaries.
```

❌ **遇 silent skip 时排查清单**:
1. 上传的 `src/train.py` 是否 commit `3b06e1d` (含 fix)?
2. Args dict `time_bucket_boundaries` 是否 `''`?
3. Args dict `enable_per_domain_buckets` 是否 `True`?
4. 如全 yes 但仍 skip · 大概率平台用了缓存的旧 train.py · 删除 platform job + 重新 Create New Training Job (不 Copy)

---

## 9. 风险评估

### Risk 1 · T38 fix 平台首验失败 (概率 ~10%)

- **影响**: M52'/M54' 再次 silent skip · 损失 10 GPU-hour (累计 20)
- **Mitigation**: §8 Plan B checklist + agent 启动后实时监控 · 启动 5 min 内若 silent skip 立即 cancel
- **后果**: 5/16 晚 4 slots 减为 2 (M60 + M55) · 仍可继续

### Risk 2 · M60 / M55 stack sub-additive (概率 ~30%)

- **历史**: 5/16 实测 M51/M53 path B + 已 verified 真信号 stack 都 sub-additive (M51 dev < M49 / M53 dev < M49)
- **Mitigation**: 三滤镜框架 (§EXP-064 §3.3) 已 acknowledge stack 第 2 步互斥风险
- **后果**: 即使 dev 不超 M49 · LB 可能仍微正 (类 M41 dev↓ LB↑ 反例) · 5/18 早 evaluate 决断

### Risk 3 · 24h ckpt 过期 (M34 教训)

- **完成时间**: ~CEST 04:30~05:00 5/17
- **24h 过期**: ~CEST 04:30~05:00 5/18
- **Mitigation**: 用户 ~CEST 17:00 5/17 之前必须 publish 4 个 ckpt 到 Model Management

### Risk 4 · Args silent copy bug (5/15 教训)

- **Mitigation**: §7 checklist · 用户启动后 30s 贴 Args dict log 给 agent · agent 严格 grep 比对

---

## 10. 5/18 早 quota 决策预案 (供 5/17 早后参考)

5/18 早 4 quota 候选池 = 5/16 晚 4 jobs (M60/M55/M54'/M52') + publish-ready 老候选剩余:

| 优先级 | 候选 | 触发条件 |
|---|---|---|
| ⭐⭐⭐ | M60 LB | dev > M49 (≥ 0.871) |
| ⭐⭐⭐ | M52' LB (T38 fix 后) | dev > M22 mean (≥ 0.869) · T38 fit log 出现 |
| ⭐⭐ | M55 LB | dev > M22 mean · 但 T39 OOV 滤镜 1 部分敏感 · 优先级取决 5/17 早 LB |
| ⭐⭐ | M54' LB | dev > M22 mean (排除 silent skip 复现) |
| ⭐⭐ | M27 v2 LB (老候选) | dev 0.868166 publish-ready · 5/17 早未评 |
| ⭐ | M50 LB (老候选) | dev 0.861974 反例验证 · 5/17 早未评 |

**5/18 早默认推荐 (待 5/17 早 LB 后修订)**: M60 + M52' + M55 + (M54' 或 M27v2)

---

## 11. References

- EXP-061 (M51-M54 path B stack overnight · 含 T38 silent skip bug 触发)
- EXP-062 (M55-M58 战略 · M55 已实施 commit `3f0821f`)
- EXP-063 (5/16 session end snapshot)
- EXP-064 (三滤镜分析框架 · M59 降级证据)
- ADR-014 (UE split path B Accepted · M49 LB 实证)
- ADR-012 (Multi-Embedding K=2 · M59 论文锚)
- ADR-013 (Global Token · M59 论文锚)
- src/run.sh (canonical fork base · commit `3b06e1d`)
- src/train.py line 832~847 (T38 fix · commit `3b06e1d`)
- scripts/smoke_t38_bug_fix.py (8/8 PASS · commit `3b06e1d`)
- team-memory: 5/13 18:09 (run.sh 不上传 · 平台编辑框) · 5/14 14:25 (Args 30s verify) · 5/15 silent copy bug · 5/4 T16 兼容性

---

## 12. Update Log

- **2026-05-16 CEST 23:35**: 文件创建 (用户 23:00 决路径 A + fork_all + new_exp_065 后) · 4 jobs run.sh templates fork from src/run.sh commit `3b06e1d` · 各 template 仅改 Group C 1~3 flag · Group A 全 immutable · Args verify checklist + Plan B fit log 验证要求齐全

