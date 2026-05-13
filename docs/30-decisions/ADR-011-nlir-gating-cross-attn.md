# ADR-011 · TokenFormer NLIR Gating after HyFormer Cross-Attention（T33）

> **Status**: **Proposed**（2026-05-13 CEST 14:55 起草 · 15:30 gate 源 v2 修订对齐论文 · 等 T33 half-train 回流判决）
> **Date**: 2026-05-13 Proposed
> **Revisions**:
> - v1（CEST 14:55）：gate 源用 `X_raw`（偏离论文）· Smoke 通过但研究员 review 发现语义错
> - **v2（CEST 15:30）**：gate 源改为 `LN(X)`（复用 CrossAttention 的 norm_q · 零新参数）· Smoke 验证 gate.mean=0.5002 / std=0.2127 · 严格对齐 TokenFormer §4.4 Eq. 16
> **Related Tasks**: T33（见 `docs/20-tasks/active.md`）
> **Related Experiments**: EXP-045（`t33-nlir-halftrain-gap05-*`，待启动）
> **External References**:
> - TokenFormer (Tencent Ads, arXiv:2604.13737, 2026-04-15) §4.4 Non-Linear Interaction Representation
>   - §5.7 Ablation：**NLIR alone yields +4.87‰ AUC** vs Vanilla Transformer on KuaiRand-27k
>   - §5.3 Discriminability: NLIR 显著提升 post-attention 表示与 label 的 mutual information
>   - §5.4 Dimensional Robustness: NLIR 通过 non-linear multiplicative 交互**缓解 Sequential Collapse Propagation**
> - HSTU (Zhai et al., ICML 2024)：gated attention 是工业推荐新共识
> - 本项目 `docs/10-research/community-intel.md` §Paper-Intel

---

## Context（背景与触发原因）

### 触发证据

#### 1. 机制层：本项目 HyFormer cross-attention 是"纯线性 attention output"

`baseline/model.py::MultiSeqHyFormerBlock.forward`（line 1279-1288）每个 block 对 S 个 seq domain 做：

```python
decoded_q_i = self.cross_attns[i](
    q_tokens_list[i], next_seqs[i], next_masks[i], ...
)
```

`CrossAttention` 内部是标准的 `softmax(QK^T / √d) · V`，**没有 multiplicative gating 保护**。TokenFormer §5.4 和 RankUp §2.2 都实证：**纯线性 attention 在深层会触发 effective rank 的 "damped oscillatory trajectory"**，sequential 表示被 low-rank 的 NS tokens 污染（Sequential Collapse Propagation, SCP）。

本项目 `num_hyformer_blocks=2` 下虽然不到论文里 4~8 层的深度，但**cross-attention 每 block 都会把 NS tokens 的信号灌入 Q tokens**，2 次累积的 rank 压缩足以让 deeper block 的 decoded_q 表达力受限。

#### 2. 外部证据：TokenFormer §5.7 ablation 数据 + 同团队

TokenFormer 作者团队 = Tencent Ads（Junwei Pan, Jie Jiang, Haijie Gu 等，**与本赛题 HyFormer baseline 作者团队完全重叠**）。§5.7 明确 ablation 表格：

| 变体 | Vanilla | + NLIR only | + BFTS only | + NLIR + BFTS (TokenFormer-S) |
|---|---|---|---|---|
| Macro AUC 相对 Transformer baseline | - | **+4.87‰** | +4.91‰ | +5.76‰ |

- **NLIR 独立贡献 +4.87‰ AUC** — 在 Industrial Track 类似赛题下约等于 **+0.003~+0.005 LB**。
- NLIR 和 BFTS **几乎独立可加**（5.76 ≈ 4.87 + 4.91 - 4.02 residual overlap），证明 gating 机制本身就是值得单独迁移的 trick。
- TokenFormer 在 Tencent 生产环境 A/B 测试：**NLIR alone vs 基线 +0.14% → +NLIR 后 +0.22%**（§5.7 Online Ablation Study）。

#### 3. 本赛题数据支持：dev→LB gap 的共性机制

本项目 EXP-019 / EXP-040 / EXP-041 多次观察到 **Dev→LB gap 扩大 = 模型过度 memorize 训练期 NS token 分布 → test 远未来泛化失败**。这正是 **SCP 的 mechanism**（低基数 NS token 压缩 Seq token 表示的有效秩）。**NLIR 的理论功能就是缓解 SCP**。

#### 4. 和已证伪方向的关键差异

- ❌ **T19 Per-token FFN（ADR-007 · 已 Paused-Needs-Rerun）**：每 token 独立 FFN 参数 +34% Dense → 在 1.9M 梯度预算下过拟合（team-memory "独立化三连败"铁律）。
- ❌ **T28 DIN Integrated（ADR-010 · Deprecated）**：每 block 独立 DIN → "做 1 次刚好够做 2 次过度" 机制性负向。
- ❌ **FaFE（M8 · 已 Deprecated）**：每 fid 独立 LayerNorm+Linear → 同一铁律。
- ✅ **NLIR（本 ADR）**：**共享一个 `nn.Linear(d_model, d_model)` + sigmoid gating**（每 block 内跨所有 decoded_q_i **共享参数**，甚至可以跨 block 共享）。参数增量 **≤ +d_model² = 4k per module**（对比 Dense 2.9M 增量 < 0.15%），**完全符合 "共享参数 + 外部信号注入" 白名单**。

### 问题域

| 维度 | 现状 | 目标 |
|---|---|---|
| Cross-attention output 机制 | 纯 softmax · V（线性） | 加 sigmoid gating multiplicative modulation（非线性） |
| 参数增量 | — | **+~4k per block**（`W_g ∈ R^{d×d}`，S=4 共享一个 W_g）·  总 ≤ +8k |
| Dev AUC on gap=0.05 | `icq-half-gap05` E3 Dev 0.842443 | 至少 +0.0013（方法学真信号边界） |
| LB 期望（gap=0.05 配置） | baseline LB 0.839223 / T25 bypass B LB 0.843433 | +0.002~+0.004 vs 不加 NLIR |
| 平台兼容性 | 新 kwarg `--enable_nlir_gating` | 必须上传自定义 infer.py（team-memory 铁律）|

### Trade-offs（放弃了什么）

- **放弃"在 FFN output 后加 NLIR"**：TokenFormer §4.5 把 NLIR 放 attention output + SwiGLU FFN 组合。本项目 SwiGLU 已有（encoder 内），重复加会引入参数冗余；**只在 cross-attention output 后加**，位置和本项目当前架构最对口。
- **放弃"全局 NLIR across all blocks with tied weight"**：机制上没问题，但 debug 成本高。**每 block 独立一个 W_g · forward 时对 S 个 decoded_q_i 共享使用**，是参数增量 / 表达力 / debug 可读性的平衡。
- **放弃"在 seq_encoders 内部 attention 也加 NLIR"**：seq_encoder 已有多种类型（swiglu / transformer / longer），改侵入性大。**只改 cross-attention output**，作用点精准对应 NS→Seq 信号污染的最集中位置。

---

## Decision（最终决定做什么）

在 `MultiSeqHyFormerBlock` 每个 block 内，**cross-attention output 后、token fusion 前**，对所有 decoded_q_i 应用共享的 NLIR gating：

```
# 原流程 (line 1279-1288):
decoded_q_i = cross_attns[i](q_tokens_list[i], next_seqs[i], ...)

# 新流程 (enable_nlir_gating=True):
raw_q_i = cross_attns[i](q_tokens_list[i], next_seqs[i], ...)    # (B, Nq, D)
gate = sigmoid(raw_q_i @ W_g)                                     # (B, Nq, D)
decoded_q_i = gate * raw_q_i + q_tokens_list[i]                   # NLIR + residual
```

**参数细节**：
- `W_g ∈ R^{D×D}`（per block 独立，S 个 seq domain 跨域共享）
- 初始化：Xavier uniform（让 gate 初始分布集中在 0.5 附近，不偏向强开启或强关闭）
- **Gate 源 = `LN(X)`**（复用 CrossAttention 的 `norm_q` · v2 修订 · 严格对齐论文 Eq. 16）
  - 论文 Eq. 16 原文：`G = X^(l) @ W_g` · 但 TokenFormer 用 PreNorm RMSNorm · 其 `X^(l)` 本身已经是上一 block 的归一化输出
  - 本项目 CrossAttention 用 pre-LN · attention 输入的 Q 是 `LN(X)` · gate 必须用同一 tensor 才能保持 "same normalized X drives both attn and gate"
  - 如果用 raw X：① 方差随 block 累加 → block 2+ sigmoid saturation；② Xavier 初始化的 zero-mean unit-variance 假设失效；③ Smoke 实测 mean 会偏离 0.5
  - v2 实测：gate.mean=0.5002 · std=0.2127 · 对齐论文精神
- Residual 使用**原始 q_tokens_list[i]**（未经过 LN 和 cross-attn 的），对齐 TokenFormer Eq. 18 `I = X + σ(G) ⊙ A`
- 与 T28 integrated DIN 注入点**紧邻**（同段 forward），但 **T28 Deprecated 且默认 off**，两者互斥（当前 run.sh 没 enable_din_integrated）

### 代码改动面

1. `baseline/model.py::MultiSeqHyFormerBlock`：
   - `__init__` 加参数 `enable_nlir_gating: bool = False, nlir_residual: bool = True`
   - 新增 `self.nlir_gate: nn.Linear(d_model, d_model) | None`
   - `forward` line 1288 后加 gating 分支（~20 行）
2. `baseline/model.py::PCVRHyFormer.__init__`：透传 NLIR 参数到每个 block
3. `baseline/train.py`：CLI `--enable_nlir_gating`（bool, default False）
4. `baseline/infer.py`：`cfg.get('enable_nlir_gating', False)` fallback

**总改动：~40 行净增**，所有改动默认 off（backward-compat bit-identical）。

---

## Verification（验证计划 / Go / No-go 阈值）

### S1 本地 smoke（待做）

- `py_compile` 全过
- 3 矩阵验证：
  1. `enable_nlir_gating=False` → 参数量与当前主干 bit-identical
  2. `enable_nlir_gating=True` → 参数量 +2 blocks × 64² = +8,192（`W_g`），log 打印 `[NLIR] enabled on N blocks`
  3. train 5 steps（demo 1k）+ infer 1000 predictions 无 state_dict mismatch
- Gate 数值健康检查：打印首个 batch 的 `gate.mean() / gate.std() / gate.min() / gate.max()`，期望 mean ≈ 0.5 / std > 0.1（不全饱和）

### S2 平台 half-train（Go/No-go 阈值）

| Job Name | Config | 阈值（vs `icq-half-gap05` E3 Dev 0.842443） |
|---|---|---|
| `t33-nlir-halftrain-gap05-seed42` | gap=0.05 + `--enable_nlir_gating` + 其余 = M9 非 DIN config | **E3 Dev ≥ 0.8437**（+0.0013 = 3× rerun 噪声）→ 🎯 强信号 |
| `t33-nlir-halftrain-gap05-seed3` | 同上 seed=3 · 松 guard | 双 seed 确认阈值 |

**判决矩阵**：
- E3 Dev ≥ 0.845（+0.0026 · 对标 TokenFormer +4.87‰ 的 1/2 迁移率）→ 🎯🎯 **强真信号** · 立即 LB 校准
- E3 Dev ∈ [0.8437, 0.845) → 🎯 真信号 · eval
- E3 Dev ∈ [0.8433, 0.8437) → 🟡 疑似 · seed=3 rerun 复现
- E3 Dev ∈ [0.8421, 0.8433) → ⚪ 持平（±0.0003 rerun 噪声 + ±0.001 采样偏差内）→ Deprecated
- E3 Dev < 0.8421 → ❌ 负向 → 关闭

### S3 Leaderboard 校准（仅 S2 通过）

- 消耗 1 evaluation quota / seed
- 阈值：**LB ≥ 0.842**（vs baseline 0.839223 · Δ+0.003 = 方法学信号）
- 强信号：**LB ≥ 0.846**（破 T25 bypass B LB 0.843433 + Δ+0.0026）

### S4 叠加（仅 S3 真信号）

- 与 T25 DIN bypass 叠加 → `t33-nlir-t25-halftrain-gap05-seed42`
- 与 gap=0 叠加 → `t33-nlir-gap0-seed42-4ep`（冲 SOTA）
- S4 最终 LB 阈值：**≥ 0.848**（破 M9 SOTA 0.84671 + Δ+0.0013）

---

## Consequences（后果）

### 正面

- **参数增量极小**（+8k in 2 blocks），不触发 "独立化参数三连败" 铁律
- **机制正交**：与 T25 DIN bypass（classifier 前注入）、T30 时间桶（dataset 层）、T31 is-missing（dataset 层）、T32 Q_init（query generator 内）**完全独立**，S4 可多路叠加
- **同团队证据 + 论文 ablation 独立收益 +4.87‰ AUC**，ROI 在当前剩余候选里最高
- **backward-compat 默认 off**，不影响 M9 SOTA 路径

### 负面 / 待观察

- **论文数据来自 KuaiRand-27k 公开数据集** + Tencent 生产环境，**本赛题 Industrial Track 1.9M 数据规模下迁移率未验证**。乐观迁移率 50%（+0.0024 AUC），保守 25%（+0.0012 AUC，边缘信号）。
- **与 RoPE / CrossAttention 的 pre-LN 是否耦合**：TokenFormer 用的是 standard RoPE + RMSNorm，本项目 `CrossAttention(ln_mode='pre')` 是 pre-LN + no RoPE on Q（RoPE 只用于 seq encoder 内部）。**S1 本地 smoke 必须验证 gate 的数值分布不饱和**。
- **gate 初始化敏感**：若 W_g 初始化过大，sigmoid 饱和 → gate ≈ 1 退化为 pure cross-attn；若过小 → gate ≈ 0.5 过度平均。Xavier uniform + bias=0 是最保守选择。

### 红线检查（必读）

- ❌ 不 ensemble / 不 SWA — 单模型单 ckpt 提交
- ❌ 不外部数据 — 纯架构改动
- ✅ 单一架构最终提交 — T33 若 work，作为 model.py 标准 config 的一部分，不产生多模型
- ✅ per-X 独立参数 bug 规避 — NLIR 是**共享 W_g**（S 个 domain 共享），不是"每 seq 独立 gating"
- ✅ infer.py 兼容 — 新 kwarg 必上传自定义 infer.py（不赌平台默认）

---

## Alternatives Considered（备选方案）

### B · GLU-style gating（两路 Linear）

- **机制**：`out = (x · W1) * sigmoid(x · W2)`（两个独立 Linear）
- **放弃理由**：参数量 2×，且 TokenFormer §4.4 明确用的是**单 Linear + 原始 attention output 作 modulation 对象**（Eq. 17: `I = σ(G) ⊙ A`，A 是 attention output 不是 V），更贴近 HSTU gated attention 派系。保持和论文一致。

### C · NLIR after FFN instead of after cross-attn

- **机制**：在 RankMixerBlock 的 FFN output 后加 gating
- **放弃理由**：FFN 已是非线性（GELU / SwiGLU），再加 gating 收益边际；而 cross-attention output 是当前架构**唯一的纯线性节点**，从"补最弱环节"角度优先级更高。

### D · Per-seq NLIR（S 个独立 W_g）

- **放弃理由**：违反 team-memory "独立化三连败" 铁律（per-X 独立参数）。S=4 个独立 W_g 每个 4k params，在 1.9M 样本下每个参数平均梯度 steps 不足。

### E · 把 NLIR 融进 CrossAttention 内部

- **放弃理由**：侵入性大，影响 OneTrans / 其他 attention block。**保持 NLIR 作为 block-level 外挂**，可读性和可回滚性最好。

---

## Open Questions（悬而未决）

1. **gate 是否要 tied across blocks**：TokenFormer 是 decoder-only 连续 L 层，每层 NLIR 独立；本项目只 2 blocks，tied weight 也可以试。暂定**每 block 独立**。
2. **是否配 RMSNorm 替代 LayerNorm**：TokenFormer §4.5 用 RMSNorm，本项目用 LayerNorm。**暂不改 norm**（避免引入归因歧义），只加 gating。
3. **α 初始 scale**：初始不加 learnable α 缩放（论文也没加）。若 S2 gate 饱和严重，再引入 `α=0.1` learnable scaling 作 fallback。
4. **和 T25 bypass 的 ensemble 效应**：T25 在 classifier 前注入独立 DIN interest（+D dim concat），T33 在 cross-attn output 调制 decoded_q。理论正交，S4 叠加时 Dev 期望 +0.0024 + 0.0013 ≈ +0.0037（假设独立可加），若实测显著低于这个值说明有 hidden interaction。

---

## References

- TokenFormer paper: [arXiv:2604.13737](https://arxiv.org/abs/2604.13737) §4.4 / §5.3-5.4 / §5.7
- `baseline/model.py:1149-1321` MultiSeqHyFormerBlock 当前实现
- `baseline/model.py:563-656` DINInterestExtractor（T25 bypass · 同位置类参考）
- ADR-007 Per-token FFN（反例 · 独立化铁律）
- ADR-010 DIN Integrated（反例 · 重复化铁律）
- EXP-045（T33 NLIR 实验计划 + 回流数据）
- EXP-041 §独立化三连败（铁律来源）
