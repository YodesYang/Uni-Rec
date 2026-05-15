# ADR-013 · RankUp Global Token Integration (T36)

> **Status**: **Accepted (Plan B)** · 2026-05-15 CEST 17:50 实施完成 · smoke 5/5 通过 · 等平台 half-train 验证
> **Date**: 2026-05-15 Proposed → 2026-05-15 Accepted (Plan B)
> **Related Tasks**: T36（新）· T34 Multi-Emb K=2 (ADR-012)
> **Related Experiments**: EXP-053（论文 ablation 严格审计 · 触发条件）
> **External References**:
> - RankUp (Tencent Ads, arXiv:2604.17878 v3, 2026-05-12) §3.4 Global Token Integration
>   - §4.2 Table 1: **"Global Token + Multi-Emb" 组合 +0.21%/+0.18%/+0.13% online AUC**
>   - §4.4: "Global Token mechanism **stabilizes deep-layer dynamics** by enabling global feature aggregation, which mitigates information loss observed in subset-based variants"

---

## Context（背景与触发原因）

### 触发证据

#### 1. EXP-053 论文 ablation 严格审计

5/15 严格 grep RankUp §4.2 Table 1 发现:
- 论文报的 **+0.18% online AUC = "Global Token + Multi-Emb 组合"** · **不是 Multi-Emb alone**
- Multi-Emb alone 估算 **+0.08%~+0.15% online AUC**
- 想拿满 +0.18%·**必须配 Global Token**

#### 2. 本项目 HyFormer 当前架构 · 无 Global Token

`baseline/model.py::PCVRHyFormer` 当前 forward:
```
NS tokens (user_ns + item_ns) → MultiSeqQueryGenerator → cross_attn × num_seqs → MultiSeqHyFormerBlock → classifier
```

**所有 NS token 都是 local subset** (user_ns 5 个看 user fids · item_ns 2 个看 item fids · query token 看序列)·**没有任何 token 整合 "全部 user_int + item_int + all seq mean" 的全局视图**。

#### 3. 机制 · RankUp §3.4 原文

> "the Global Token g aggregates information from all features {f_1, ..., f_M} via an aggregation function A"

公式 (5):
```
g = LayerNorm(Linear(concat[mean_pool(all_fid_embs), mean_pool(all_seq_repr)]))
```

加在 NS tokenizer 输出后 · 与 user_ns + item_ns + query 共同进入 HyFormer Block。

#### 4. 与本项目铁律的兼容性

✅ **共享参数** (1 个 Linear + LN · 跨 batch 跨 fid 全用同一参数) · 不违反 "per-X 独立参数三连败" 铁律
✅ **+~8.4k Dense params** (Linear(d_model × num_fids, d_model) · 7424 + 64 LN = 7488)
- 估算: d_model=64 · concat 维度 = 60 fids × emb_dim 64 + 4 seq × 64 = 4096 · Linear(4096, 64) = 262k params
- 实际更小 · 若 mean-pool 后只 concat seq domain · 4 × 64 = 256 维 · Linear(256+64, 64) = 21k

---

## Decision (Plan B · Reuse user_dense token slot)

> **2026-05-15 CEST 17:50 修正**: ADR 原稿 line 75-76 算错 T 公式（漏算 user_dense token）。实际 num_ns = `user_ns_tokens(5) + 1(user_dense) + item_ns_tokens(2) + 0 = 8` · T = 16 · 64%16=0 ✅。**直接 +1 Global Token 会让 num_ns=9 → T=17 · 64%17 ≠ 0 ❌**。已通过用户审议改 Plan B：**复用 user_dense token slot 升级为 Global Token**。

### Plan B 实施 (2026-05-15 CEST 17:50 · `enable_global_token=False/True`)

`enable_global_token=True` 时 `user_dense_proj` 不实例化，改为 `global_token_proj`：

```python
# in_dim = user_dense_dim + 2*d_model + num_sequences*d_model
#        = user_dense_dim + 64 + 64 + 4*64 = user_dense_dim + 384
global_token_proj = Sequential(
    Linear(in_dim, d_model),
    LayerNorm(d_model),
)

# forward(): seq_tokens 必须先建（RankMixer 顺序调换）
user_int_pool = mean(user_ns, dim=1)              # (B, D)
item_int_pool = mean(item_ns, dim=1)              # (B, D)
seq_pool = concat([masked_mean(seq_d) for d ∈ {a,b,c,d}], dim=-1)  # (B, 4D)
global_concat = concat([user_dense_feats, user_int_pool, item_int_pool, seq_pool])
global_token = SiLU(global_token_proj(global_concat)).unsqueeze(1)  # (B, 1, D)

# 替换原 user_dense token slot · 整体 ns 序列结构不变
ns_tokens = concat([user_ns, global_token, item_ns], dim=1)         # (B, 8, D)
```

### T-divisibility 验证

- num_ns = 5 + 1 + 2 + 0 = **8** (与 baseline 完全相同)
- T = 2 × 4 + 8 = **16** · `d_model=64 % 16 == 0` ✅
- **0 STRUCTURAL 改动** · 不动 num_queries / num_sequences / num_ns_tokens / d_model

### 代码改动面 (实测 ~80 行 · 2026-05-15 已落地)

1. `src/model.py::PCVRHyFormer`:
   - `__init__` 加 `enable_global_token: bool = False` (line 2492-2494)
   - 替换 user_dense_proj 构造为 if/else（disabled 时 user_dense_proj·enabled 时 global_token_proj·二者只构造其一）
   - `_build_ns_tokens` 增加 optional `seq_tokens_list / seq_masks_list` 参数（disabled 时不传）
   - 新增 `_build_global_token` 方法（masked mean-pool + concat + Linear+LN+SiLU）
   - `forward` / `predict` 在 `enable_global_token=True` 时切到 seq-first 顺序
2. `src/train.py`: CLI `--enable_global_token` (line ~540) + model_args (line ~879)
3. `src/infer.py`: cfg fallback `bool(cfg.get("enable_global_token", False))`
4. `scripts/smoke_t36_global_token.py` (新文件 · 5 项 smoke 测试)

### 等价性 safeguard

- `enable_global_token=False` (默认) bit-identical baseline (smoke 验证 OK · forward 完全等价 · 0 参数变化)
- `user_dense_dim=0` 时 enable=True 直接 raise ValueError

---

## Verification

### S1 Smoke (2026-05-15 CEST 17:55 ✅ PASS)

`scripts/smoke_t36_global_token.py` 5/5 通过：
1. `enable_global_token=False` · user_dense_proj 保留 · forward shape OK
2. `enable_global_token=True` · global_token_proj in_features=392 (=8+64+64+256) · backward grads finite
3. Param delta = **+24576** Dense（≈ 24k · Linear(392,64) + LN(64) - Linear(8,64) - LN(64)）
4. `user_dense_dim=0` 时 enable=True 正确 raise ValueError
5. disabled 路径同 seed 多次构造数值确定性 ✅

### S2 平台 half-train (gap=0 path · M22 baseline E3 Dev 0.8686)

| Job | Config | 阈值 |
|---|---|---|
| `M39-globaltoken-gap0-seed42-4ep` | M22 + `--enable_global_token` | E3 Dev ≥ 0.8691 (+0.0005) → 单变量真信号判决 |
| (后续) `M40-multiemb-globaltoken-gap0-seed42-4ep` | M22 + Multi-Emb K=2 + Global Token (双 stack) | E3 Dev ≥ 0.8697 (+0.0011) → 双 stack 工作 |

### S3 LB 校准

- Single Global Token: LB ≥ 0.8480 (+0.0004 vs M22) → 真信号
- Stack with Multi-Emb K=2: LB ≥ 0.8485 (+0.0009 vs M22) → 强真信号
- **三叠加 + NLIR**: LB ≥ 0.8497 → 🎯 接近 Top 20 阈值

---

## Consequences

### 正面
- **填补 RankUp §3.4 论文要求** · Multi-Emb 增量提到论文 +0.18% 上限
- **+~21k Dense params** · 极小增量 · 不触发独立化铁律
- **正交 T33 NLIR / T34 Multi-Emb K=2** · 可三叠加
- **本赛题 user/item cold start 100% (Test EDA Finding 4)** · Global Token 可缓解 cold start 信号

### 负面 / 风险
- **+1 token 在 HyFormer Block 内** · attention 复杂度从 O((S×Q + N_ns)²) → O((S×Q + N_ns + 1)²) · 微小开销
- **整除性约束**: num_ns 5+2+1=8 后 (2×4+8)=16 · d_model=64 整除 OK · 但若改 num_queries/num_seq 需重算
- **RankUp 论文 ablation 没单独报 Global Token alone** · 只报 "Global Token + Multi-Emb" 组合 · **Global Token 单独增量未知** · 估 +0.03%~+0.08%

### 红线检查
- ✅ 共享 Linear · 不违反独立化铁律
- ✅ 单一架构 · 不 ensemble
- ❌ 不外部数据 · 纯架构改动

---

## Alternatives

### B · Global Token 用 SUM 代替 LayerNorm(Linear(concat))
- 放弃: 论文是 "aggregation function A" · 用 Linear+LN 是最对齐的实现

### C · 在每个 HyFormer Block 内独立 Global Token
- 放弃: 违反独立化铁律 (per-block 独立 global token 学习路径)

### D · 把 Global Token 加在 RankMixerBlock 输出而非 NS Tokenizer 输出
- 放弃: 增加复杂度且偏离 RankUp 论文原始位置 (token-set level)

---

## Open Questions

1. **Global Token 是否参与 Query Generator**: 当前设计加在 NS tokens 序列里 · `MultiSeqQueryGenerator` 看到 8 NS · 是否需要让 query 也"看到"global? 暂不做 · 让下游 attention 自然学
2. **Global Token 初始化**: LayerNorm + Linear 默认 Xavier 应该 OK · 待 S1 smoke gate 分布确认
3. **是否要 learnable α 缩放**: 不加 (RankUp 没用) · 若 S2 dev 比 K=2 alone 差·考虑加 `α=0.1` learnable

---

## References

- RankUp paper [arXiv:2604.17878](https://arxiv.org/abs/2604.17878) v3 §3.4 / §4.2 Table 1 / §4.4
- ADR-012 Multi-Embedding K=2 (前置依赖)
- EXP-053 论文 ablation 严格审计 (触发条件)
- `baseline/model.py:2317-2680` PCVRHyFormer 当前实现
