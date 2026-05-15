# ADR-013 · RankUp Global Token Integration (T36)

> **Status**: **Proposed**（2026-05-15 CEST 11:35 起草 · 触发 = EXP-053 论文 ablation 严格审计揭示 Multi-Emb alone 增量不足 · 必须叠加 Global Token 才能到 RankUp §4.2 +0.18% 报告值）
> **Date**: 2026-05-15 Proposed
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

## Decision

在 `RankMixerNSTokenizer` 或 `PCVRHyFormer.__init__` 加 `--enable_global_token`:

1. **生成 Global Token**: forward 时计算
   - `user_pool = mean(user_fid_embs)` (对 user_int fid 做 mean-pool 跨 fid)
   - `item_pool = mean(item_fid_embs)` (同)
   - `seq_pool = concat([mean_pool(seq_a), ..., mean_pool(seq_d)])` (4 seq domain)
   - `global_token = LayerNorm(Linear(concat[user_pool, item_pool, seq_pool], d_model))`
2. **加到 NS token 序列**: `ns_tokens = concat([user_ns, item_ns, global_token])` · `num_ns_tokens = 5 + 2 + 1 = 8`
3. **下游不变**: HyFormer Block / MultiSeqQueryGenerator / classifier 自动处理 num_ns 增加

### 代码改动面 (~50 行)

1. `baseline/model.py::PCVRHyFormer`:
   - `__init__` 加 `enable_global_token: bool = False`
   - 新增 `self.global_token_layer: nn.Sequential | None`
   - `forward` 计算 global_token 并 concat 到 ns_tokens (~25 行)
2. `baseline/train.py`: CLI `--enable_global_token` (~3 行)
3. `baseline/infer.py`: cfg fallback (~1 行)
4. **整除性约束**: `d_model % (num_queries × num_sequences + num_ns) == 0`
   - 当前 num_ns = 5 + 2 = 7 · num_queries=2 · num_sequences=4 → 8+7=15
   - Global Token 后 num_ns = 8 · 8+8=16 · `d_model=64 % 16 = 0` ✅

### 等价性 safeguard

K=1 / disabled 时 bit-identical 老路径 (默认 off)。

---

## Verification

### S1 Smoke
- py_compile + lint 0 errors
- params: Δ = 1 × Linear(D_concat, d_model) + LN(d_model)
- enabled=False bit-identical
- enabled=True · train 5 steps + infer 1000 predictions OK

### S2 平台 half-train (gap=0 path · M22 baseline E3 Dev 0.8686)

| Job | Config | 阈值 |
|---|---|---|
| `t36-globaltoken-halftrain-gap0-seed42` | M22 + `--enable_global_token` | E3 Dev ≥ 0.8691 (+0.0005 = 50% RankUp Global Token alone 增量 +0.05%) |
| `t36-multiemb-globaltoken-halftrain-gap0-seed42` | M22 + Multi-Emb K=2 + Global Token (双 stack) | E3 Dev ≥ 0.8697 (+0.0011 = 50% RankUp +0.18% 组合迁移) |

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
