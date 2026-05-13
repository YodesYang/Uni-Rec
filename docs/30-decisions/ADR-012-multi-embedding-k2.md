# ADR-012 · RankUp Multi-Embedding K=2 for NS Tokenizer（T34）

> **Status**: **Proposed**（2026-05-13 CEST 15:00 起草）
> **Date**: 2026-05-13 Proposed
> **Related Tasks**: T34（见 `docs/20-tasks/active.md`）
> **Related Experiments**: EXP-046（`t34-multiemb-k2-halftrain-gap05-*`，待启动）
> **External References**:
> - RankUp (Tencent Ads, arXiv:2604.17878 v3, 2026-05-12) §3.3 Multi-embedding Representation Paradigm
>   - §4.2 Ablation Table 1：**w/ Global Token + Multi-Emb** 组合 +0.21%/+0.18%/+0.13% 线上 AUC 跨 3 任务（**是 5 组件里最大单项**）
>   - §4.4：Multi-embedding 主要作用于 **Block 0 的 early-stage effective rank**，显著提升稀疏场景的 rare signal 捕捉
> - Pan et al., 2024 "Ads recommendation in a collapsed and entangled world" (SIGKDD 2024) — Multi-Embedding 专治 embedding collapse
> - Guo et al., 2023 "On the Embedding Collapse when Scaling up Recommendation Models" (arXiv:2310.04400) — 理论基础
> - 本项目 EXP-029 §EDA：大 vocab 全在 seq 域，Seed 主楼问题-1 "id 特征 OOV 线下虚高线上掉"

---

## Context（背景与触发原因）

### 触发证据

#### 1. 机制层：本项目 NS tokenizer 是单 embedding table

`baseline/model.py::RankMixerNSTokenizer.__init__`（line 1898-1907）：

```python
for vs, offset, length in feature_specs:
    skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
    if skip:
        embs.append(None)
    else:
        embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
```

**每个 fid 只有一个 `nn.Embedding`**。对一个高基数但长尾的 fid，单 embedding table 被迫把"头部 item 的判别信号"和"尾部 item 的稀疏信号"都压到同一个 d=emb_dim 子空间，**embedding collapse 在高 vocab fid 上尤其严重**（Guo et al. 2023 + Pan et al. 2024 两篇 Tencent 论文的核心发现）。

#### 2. 外部证据：RankUp §4.2 ablation 表格 · 同团队

RankUp 作者 = Tencent Ads 19 人（含 Junwei Pan, Jie Jiang, Haijie Gu），与本赛题 HyFormer baseline 作者**完全同团队**。§4.2 Table 1 Online Ablation：

| 变体（单独加到 RankMixer baseline） | Order | Book | Add Service |
|---|---|---|---|
| Randomized Permutation Split | +0.06% | +0.06% | +0.08% |
| **w/ Global Token + Multi-Emb** | **+0.21%** | **+0.18%** | **+0.13%** |
| w/ Cross Embedding (Pre-trained) | +0.22% | +0.10% | +0.03% |
| w/ Task Token | +0.09% | +0.02% | +0.02% |
| **RankUp full** | **+0.41%** | **+0.23%** | **+0.25%** |

- **Multi-Emb + Global Token 组合是 5 组件里最大单项**（+0.13%~+0.21% AUC）
- Cross Pre-trained Embedding 对本赛题 **不适用**（禁外部数据）
- **单独 Multi-Emb 的贡献**估计 +0.10%~+0.18%（减去 Global Token ~0.03~0.05 份额），迁移到 LB 约 **+0.001~+0.003**

RankUp §4.4 效果分析明确：**"The Multi-embedding paradigm primarily improves early-stage representation diversity by mapping identical raw features into independent embedding subspaces"** —— 精确对口本赛题 EXP-029 观察到的 "信息丰富样本 AUC 更差"（模型在 head item 上过拟合、tail item 欠拟合）问题。

#### 3. 数据层支持：本赛题 EDA 结论直接对口

EXP-029 `docs/40-experiments/EXP-029-eda-report-2026-05-05.md` 三大发现：
- 结论 1：顶层 user_int / item_int 最大 vocab 只有 25k，**这些 fid 不是 Multi-Emb 主要目标**
- 结论 2：**"信息丰富"样本 AUC 低 0.04~0.05**（稀疏 user AUC 0.89 vs 丰富 0.85）→ Seed 主楼问题-1 "id 特征线下虚高线上掉" 的机制证实
- 结论 3：**大 vocab 全部在 seq 域**（seq_c/47=2.8亿, seq_b/69=8600万, seq_c/29=820万）

Multi-Emb **精确对口结论 2/3**：对"信息丰富 + 大 vocab" 的 fid（主要在 seq 域、item_int 头几个 fid），K=2 份独立 embedding 让模型**从两个正交子空间**学习相同 id，**缓解 head/tail 单子空间撕裂**。

#### 4. 和已证伪方向的关键差异（关键铁律核对）

**team-memory 铁律**："per-X 独立参数在 1.9M 数据下必死"（T19 / T28 / FaFE 三连败）。**Multi-Emb 会不会触发这条铁律？**

**答案：不会。关键区分如下：**

| 维度 | per-X 独立参数（已证伪） | Multi-Embedding（本 ADR） |
|---|---|---|
| 独立参数位置 | **下游模块**（FFN / LayerNorm / DIN） | **Embedding table**（每 fid 的输入侧） |
| 每参数的梯度信号 | 每 X 只见 1/X 的样本 · 严重 starve | 每 table 见**所有样本**（同 fid 的所有 batch 都更新两个 table） |
| 梯度耦合 | 解耦 · 信号稀释 | 两 table 对同 id 的 sum/concat 下游共享，梯度流过所有下游参数 |
| 论文证据 | RankMixer §3.2 per-token FFN（本项目 T19 证伪） | RankUp §3.3 + Pan 2024 SIGKDD（工业 proven） |
| 本项目历史 | T19/T28/FaFE 三连败 | **未曾尝试** |

**机制层：Multi-Emb 的本质是"同一样本在同一 fid 上产生 2× embedding lookup，对下游贡献是 concat/sum"**，这和 per-X 独立 FFN **完全不同**。每个 embedding table 的参数在训练时都是**所有 batch 都更新的**，不存在梯度稀释问题。

### 问题域

| 维度 | 现状 | 目标 |
|---|---|---|
| 每 fid 的 embedding table 数量 | 1 | **K=2** |
| 每 fid 的 embedding 输出维度 | `emb_dim=64` | concat 后 `2×emb_dim=128`，再经一个 `Linear(128→64)` 投影回原维度 |
| 下游 RankMixerNSTokenizer 修改 | — | `forward` 里 `fid_emb = concat(emb_A(x), emb_B(x))` 后投影 |
| Embedding 参数量增量 | 当前约 ~300M（大部分在 sparse embedding） | **+~300M**（K=2 翻倍，且 vocab 没变 · 只是 table 个数翻倍）|
| **等价性 safeguard**：K=1 时 bit-identical 老路径 | — | `--multi_emb_k 1`（默认）严格同现在 |
| GPU VRAM 风险 | ~19 GB（batch=128） | **+~3~5 GB** for embedding table（可控） |

### Trade-offs（放弃了什么）

- **放弃 K=3 / K=4**：ROI 递减 + VRAM 压力大。K=2 是论文同款 + 工程稳妥。
- **放弃"只对大 vocab fid 加 Multi-Emb"**：选择性应用需要额外 config（"哪些 fid 配 2 table"），一方面需要 EDA 数据（见 M16），另一方面会让判决更复杂。**首版对所有 non-skipped fid 统一 K=2**，后续可细化。
- **放弃"sum 代替 concat+projection"**：sum 更省参数但 RankUp §3.3 原文是 "a tuple of embeddings drawn from different tables"，**concat+projection 是论文原味**。projection `Linear(2d→d)` 多 +8k params per fid（60 fid × 8k = 480k dense params），可接受。
- **放弃"共享 vocab mapping + 独立 embedding initial"**：这样参数增量最小但论文机制丢失。K=2 的价值就在"两个**完全独立**的参数空间"，不共享。

---

## Decision（最终决定做什么）

在 `RankMixerNSTokenizer` 新增 `--multi_emb_k: int` 参数（默认 1 = backward-compat），K≥2 时：

1. **Embedding**：每个 non-skipped fid 分配 K 个 `nn.Embedding(vocab+1, emb_dim, padding_idx=0)`
   - 每个 table 独立 Xavier 初始化（保证初始分布不同）
2. **Forward**：同一 fid 的 K 个 embedding 在 forward 时 concat → shape `(B, K*emb_dim)`
3. **Projection**：通过一个共享的 `Linear(K*emb_dim, emb_dim)` 投影回 `emb_dim`，**其余下游完全不变**（chunk / split / token projection 都看到的是 `emb_dim` 维输入）
4. **Skipped fid**（vocab > emb_skip_threshold）：不改，仍然产出全零向量

**关键等价性**：K=1 时必须 bit-identical 老路径。实现上：
- K=1 走旧分支（单 `nn.Embedding` · 无 projection）
- K≥2 走新分支（K 个 `nn.Embedding` + 1 个 `Linear`）

### 代码改动面

1. `baseline/model.py::RankMixerNSTokenizer.__init__`:
   - 加参数 `multi_emb_k: int = 1`
   - K=1 路径：当前代码（line 1900-1907 不变）
   - K≥2 路径：`self.embs = nn.ModuleList([nn.ModuleList([Emb_1..Emb_K]) for fid in non_skipped])`
   - 新增 `self.multi_emb_proj: nn.Linear(K*emb_dim, emb_dim) | None`（K=1 为 None）
2. `baseline/model.py::RankMixerNSTokenizer.forward` (line 1958-1992):
   - 在 `fid_emb = emb_layer(...)` 位置分支：K=1 原样；K≥2 循环 K 次 concat 后过 `multi_emb_proj`
3. `baseline/model.py::PCVRHyFormer.__init__`：透传 `multi_emb_k` 到 tokenizer
4. `baseline/train.py`：CLI `--multi_emb_k` (int, default 1, choices=[1, 2])
5. `baseline/infer.py`：`cfg.get('multi_emb_k', 1)` fallback

**总改动：~60 行**。默认 off（K=1 bit-identical）。

---

## Verification（验证计划 / Go / No-go 阈值）

### S1 本地 smoke

1. `py_compile` 全过
2. K=1 时参数量 **bit-identical** 当前 M9 SOTA config（当前 Total ~353M）
3. K=2 时参数量增量：期望 +~300M embedding params + (K-1) 个 `Linear(128→64)` = 约 +300M（接近 2×）
4. `embs[fid][0].weight` 和 `embs[fid][1].weight` 初始化**不同**（Xavier 独立采样）
5. Forward: K=2 下同 fid 两个 table 对同样输入应产生不同 embedding，经 projection 后得到 (B, emb_dim) shape
6. train 5 steps（demo 1k）+ infer 1000 predictions 无 state_dict mismatch
7. 本地 VRAM/RAM 估算：demo 下从当前 ~2GB 增到 ~2.5-3GB（因为 demo vocab 小）
8. **关键 safeguard**：训推 K 必须一致，否则 state_dict load 会失败

### S2 平台 half-train（Go/No-go 阈值）

**⚠️ 首先需要验证平台 VRAM 是否够**：
- 当前 M9 SOTA config（K=1）在平台 GPU 19.12 GiB 下 batch=128 大约用 **~18 GB**
- K=2 后 embedding 参数 +300M（float32 = +1.2 GB），**加上 optimizer state (Adagrad for sparse, AdamW for dense)**，估算 **+3~5 GB**
- **可能需要降 batch 到 96 或 64**（需要 S1 本地模拟估算）

**Job 设计**：

| Job Name | Config | 阈值（vs `icq-half-gap05` E3 Dev 0.842443） |
|---|---|---|
| `t34-multiemb-k2-halftrain-gap05-seed42` | gap=0.05 + `--multi_emb_k 2` + `--batch_size 96` + M9 非 DIN config | E3 Dev ≥ 0.8437 · Go |
| `t34-multiemb-k2-halftrain-gap05-seed3` | 同上 seed=3 · 松 guard | 双 seed 确认 |

**判决矩阵（gap=0.05 配置下）**：
- E3 Dev ≥ 0.846（+0.0036）→ 🎯🎯 **强真信号**（对标论文 +0.18% AUC 迁移到 dev）→ 立即 LB 校准
- E3 Dev ∈ [0.8437, 0.846) → 🎯 真信号 · eval
- E3 Dev ∈ [0.8421, 0.8437) → ⚪ 持平 · Deprecated
- E3 Dev < 0.8421 → ❌ 负向 + 回顾是不是 VRAM 降 batch 造成的收敛问题

### S3 Leaderboard 校准（仅 S2 通过）

- 1 eval quota / seed
- LB ≥ 0.842 → 真信号 · 进 S4
- LB ≥ 0.846（破 T25 bypass B）→ 🎯🎯 强真信号 · 优先叠加
- LB < 0.841 → 疑似 noise · seed=3 rerun

### S4 叠加

- 与 T25 DIN bypass：`t34-multiemb-k2-t25-halftrain-gap05-seed42`
- 与 T33 NLIR：`t34-multiemb-k2-t33-nlir-halftrain-gap05-seed42`
- 叠加 S4 最终 LB 阈值：**≥ 0.848**（破 M9 SOTA + Δ+0.0013）

---

## Consequences（后果）

### 正面

- **参数增量大但机制层无风险**（embedding 不违反独立化铁律）
- **对口 Seed 主楼问题-1 的机制**（id OOV），是所有候选里与本赛题观察到的痛点**最对齐**的方向
- **RankUp ablation 最大单项 +0.18% AUC**，ROI 在 Top 10 路径里最高（除 T33 NLIR）
- **理论上和 T33 NLIR 完全正交**（一个在输入侧补 rank，一个在 attention output 补 rank），S4 叠加期望最大

### 负面 / 待观察

- **VRAM 压力**：+300M params + optimizer state ≈ +3-5 GB，**必须降 batch size**（预计从 128 降到 96 或 64）。降 batch 对 dev 的影响未知（可能 +0.0005 负向 · per EXP-032 batch 影响分析），需要对照实验时加 batch=128 vs batch=96 的 K=1 对照。
- **训练时间**：embedding lookup 2× + projection overhead，预计 per-epoch 时间 +20%（4 epoch 半程从 ~9h 到 ~11h）。**DDL 内勉强过关**。
- **sparse reinit 交互未验证**：本项目 sparse embedding 有 `reinit_sparse_after_epoch` 机制，K=2 下 reinit 是否要同步对 K 个 table 一起做？**保守策略：同步 reinit 所有 K 个 table**（保持老语义）。
- **平台默认 infer.py 不支持**：新 kwarg `multi_emb_k` 必上传自定义 infer.py（team-memory 铁律）。

### 红线检查

- ❌ 不 ensemble — 训练时 K 个 embedding 是端到端联合训练的单模型，**不是** inference-time multi-seed/multi-encoder averaging
- ❌ 不 SWA — 不涉及
- ✅ 单一架构最终提交 — 若 S3 通过，K=2 作为 tokenizer 标准配置
- ✅ **per-X 独立参数铁律豁免**（embedding-level K≥2 和下游 per-X FFN 不是一回事 · 见上表对比）
- ❌ 不外部数据 — Multi-Emb 不需要预训练（RankUp §3.5 Cross Pretrained 才需要，那条本项目不做）

---

## Alternatives Considered（被放弃的方案）

### B · Sum 代替 Concat+Projection

- **放弃理由**：sum 让两 table 学同一子空间（梯度相同方向累加），失去 "两个独立子空间" 的机制价值。Concat+Proj 是论文原味。

### C · K=2 只对大 vocab fid

- **放弃理由**：选择性应用的 config 复杂度高，首版追求机制清晰。**复赛 / 5/16 后若首版 work，可做 C 变体**（EDA 驱动选 fid）。

### D · Embedding dim 翻倍（128 vs 2× 64）替代 Multi-Emb

- **放弃理由**：emb_dim 翻倍破坏整除性约束（`d_model % n_total == 0`），要连带改 num_ns / num_queries。Multi-Emb 保持 emb_dim=64 不变，**对下游零影响**，是最小侵入方案。

### E · Tie K=2 tables' initialization（initial weights 相同）

- **放弃理由**：tied init 会让训练初期两 table 同步学习 → 失去 diversity。论文明确是 "independent"。

---

## Open Questions

1. **K=2 后 warmup 长度**：两 table 独立收敛可能需要更长 warmup。当前 ADR 不改 warmup，若 S2 Dev 明显拖后（E1 vs K=1 E1 差 -0.002+），考虑 warmup +20%。
2. **与 T29-C Randomized Split 的相互作用**：T29-C 打乱 fid concat 顺序，T34 翻倍 embedding table。两个都在 tokenizer 层，但独立 axis（T29 改顺序，T34 改 table 数），理论正交。S4 可叠加试一次。
3. **batch size 是否要 seed sweep**：如果 batch=96 成立，是否要再跑 batch=128 with gradient accumulation 的对照版本？**暂不做**，单 batch=96 判决为准。

---

## References

- RankUp paper: [arXiv:2604.17878](https://arxiv.org/abs/2604.17878) v3 §3.3 / §4.2 Table 1 / §4.4
- Pan et al., 2024 "Ads recommendation in a collapsed and entangled world" SIGKDD 2024
- Guo et al., 2023 "On the Embedding Collapse when Scaling up Recommendation Models" arXiv:2310.04400
- `baseline/model.py:1852-2020` RankMixerNSTokenizer 当前实现
- EXP-029 §EDA 结论 2/3（信息丰富样本 AUC 差 + 大 vocab 全在 seq）
- ADR-007 Per-token FFN (反例 · 注意 Multi-Emb 是 embedding-level 不违反)
- ADR-010 DIN Integrated (反例 · 注意 Multi-Emb 非 per-block)
- EXP-046（T34 实验计划 + 回流数据）
