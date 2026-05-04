# ADR-004 · Target-Item 硬交互注入序列侧（seq-token 级 target-item conditioning）

> **Status**: Accepted（2026-05-04 晚 · S1 本地 smoke 通过后升 Accepted）
> **Date**: 2026-05-04
> **Related Tasks**: T16（见 active.md）
> **Related Experiments**: 待 EXP-020（fast proxy / half-train 验证）
> **Supersedes / Extends**: 扩展 EXP-019 ICQ 方案（query 侧 item conditioning）

---

## Context（背景与触发原因）

### 触发证据

- **leaderboard gap 仍显著**：本地 best 约 0.868（t13tf-full E4）,leaderboard 0.84496,与 SOTA 0.85060 差 **≈ 0.006**。在单次 evaluation 配额「严格 4 次/天」+ 剩余 21 天的约束下,继续微调 hash vocab / dropout / 3 blocks 等**只能取 ~0.001 级别的收益**,ROI 不足。
- **ICQ 已验证但仍有上限**：EXP-019 `icq-t13tf-halftrain` E4 相对 t13tf E4 仅 Δ+0.000124 AUC。ICQ 在 **query 侧**把 target item 融入 global query——每条序列拿到的 query 会"知道 target item 是谁"——但**序列 token 自身**（per step key/value）仍然不含 target item 信号,交叉注意力在高碰撞 hash embedding 下退化成 soft average pooling,细粒度 target-aware matching 被稀释。
- **DIN/DIEN/CAN 文献共识**：广告推荐场景下,target-item 与 user behavior sequence 的**硬交互**（不是只在 query 端 condition,而是在 key/value 侧显式注入 target item 表征）是核心增益来源。DIN 用 attention score 做 target-specific pooling;CAN 则更进一步把 target 与 seq item 做 co-action 网络。我们的 MultiSeqHyFormer 当前是 **target-agnostic** 的 seq encoder,target 只通过 query 进入——这是 **SOTA-breaking 可能性最高的单点改造**。
- **与当前红线兼容**：端到端单一模型,不涉及 ensemble / pretrain / 外部数据;只是模型结构 incremental change,完全符合 `docs/00-context/competition.md` §3 红线。

### 问题域

- **不能只在 query 端做 target conditioning**：ICQ 已经验证只做 query 侧的收益上限约 +0.000124 (half-train E4)。
- **不能无限加深模型**：3-blocks proxy 已显示负向（EXP-016），RankMixer 加深不是有效路径。
- **不能依赖更多 sparse vocab**：hash vocab 500k→2M 已验证几乎持平（EXP-018 修正版）,不是破局点。
- **必须保持推理时长 ≤ 30min**：当前 SwiGLU ≈ 82s，有足够余量加 side-channel（估算新增 <10% 推理时间）。

### 时间压力

- 距初赛 ddl 21 天（5/25）。
- evaluation 配额严格 4 次/天,且每次需等榜单更新（实际 22h+ cycle），能用在"方法学验证"上的 slot 极少（估计最多 20 个 slot 用于非微调类 evaluation）。
- **本 ADR 必须先过 half-train（2~3 epoch + 128/32 RG 或 full RG 半程）才能花 leaderboard slot**。

---

## Decision（最终决定做什么）

**在 `MultiSeqHyFormerBlock` 的 Cross-Attention 之前，把 target-item 表征显式注入到每条 sequence 的 token 序列里（seq-token 级 target-item conditioning）**，让 SwiGLUEncoder 从第一层起就看到 target item 信号,从根源上把 target-agnostic seq encoder 改造为 target-aware。

### 具体设计（三种候选实现，按代码侵入度 / 预期增益综合评估）

#### 方案 A · Additive Side-info（推荐首发,改动最小）

```text
target_repr = mean(item_ns_tokens)              # (B, D), 与 ICQ 同源
target_proj_seq_i = W_target_i(target_repr)     # (B, D), per-sequence 线性投影
seq_tokens_i_new = seq_tokens_i + α_i * target_proj_seq_i.unsqueeze(1)
                                                 # (B, L_i, D), 把 target 加到每个 seq step
```

- **位置**：在 `PCVRHyFormer.forward` 的 `seq_encoders` 调用**之前**（或之后都可,先选之前,让 encoder 内部的 attention 能直接利用 target-aware key/value）。
- **参数量**：每个 sequence 一个 `Linear(D, D)` + 一个可学 scalar `α_i` → S × (D² + 1) ≈ 4 × 4097 = 16k,忽略不计。
- **冷启动友好**：`α_i` 初始化为 0（或很小值）,训练早期近似等于关闭,避免破坏已训练好的基线 encoder,让模型自己决定注入强度。
- **与 ICQ 正交**：本方案改 key/value 侧,ICQ 改 query 侧,理论上两者收益可叠加（需实验验证）。

#### 方案 B · Multiplicative Gating（中等改动,若方案 A 正向后再试）

```text
gate_i = σ(W_gate_i([seq_tokens_i, target_repr.broadcast]))  # (B, L_i, D)
seq_tokens_i_new = (1 - gate_i) * seq_tokens_i + gate_i * W_target_i(target_repr)
```

- **相对方案 A**:加 per-step gate,每个 seq step 独立决定"让 target 覆盖自己多少"。
- **参数量**: S × (2D × D + D²) ≈ S × 12k = 48k,仍然小。
- **风险**：gate 可能被优化器推到极端（全 0 或全 1）破坏信号;训练早期需要 warmup 或 gate init bias = -2（倾向关闭）。

#### 方案 C · Co-Action Unit（CAN 风格,最大改动,仅在 A/B 都正向后上）

```text
for each sequence token t_j in seq_i:
    co_repr_j = MLP_i([t_j, target_repr, t_j ⊙ target_repr, t_j - target_repr])
    seq_tokens_i_new[:, j, :] = co_repr_j
```

- **文献背书**：CAN (Bian et al. 2022) 在 Alibaba 广告 CVR 上相对 DIN +0.003~0.005 AUC。
- **参数量**：每个 sequence 一个 MLP（input 4D → hidden 4D → D） ≈ S × 20k = 80k,仍可接受。
- **风险**：MLP 参数较多,proxy 可能"大参数假负向",需直接 half-train 验证（参考 hash500k / ICQ 教训,已入 team-memory 铁律）。

### 推荐路线

1. **EXP-020 step 1**：方案 A additive + `α_i` init=0.1 → fast proxy (128/32 RG + 1500 steps) 对比 t13tf-only proxy 基线;若 proxy 负向也别放弃（参考 ICQ 规律）,**继续 step 2**。
2. **EXP-020 step 2**：方案 A half-train（2~3 epoch 全 RG）→ 判断是否升 full train。阈值 `E3 best AUC ≥ t13tf E3 + 0.0002`。
3. **EXP-021**（若 EXP-020 正向）：方案 A + ICQ 正交叠加 half-train,验证两条 conditioning 路径是否独立加和。
4. **EXP-022**（方案 A 饱和后）：尝试方案 C Co-Action Unit,直接 half-train 跳过 proxy。
5. **方案 B 仅在 A 负向且 C 未启动时作为备选**（gate 风险较高,不优先）。

---

## Alternatives Considered（备选方案）

- **A: 加更多 NS token（扩 user_ns_tokens / item_ns_tokens）**
    - Pros: 改动更小（只改参数）
    - Cons: 已验证 user_ns=5 / item_ns=2 是当前 sweet spot;继续加 NS tokens 会让 `d_model % (num_queries * num_sequences + num_ns) == 0` 约束更紧,且 ICQ 已在消费 item_ns 信号,边际收益递减。**不推荐**。

- **B: 换 DIN-style target attention 替换 SwiGLUEncoder**
    - Pros: 文献背书最强
    - Cons: 代码量大（需重写 seq encoder）,且 SwiGLU 已是当前最优 encoder（EXP-002 实测 3.5× 推理加速 + 最高 AUC）,替换风险高。**保留为后备,仅当方案 A/B/C 全部失败再考虑**。

- **C: 把 target concat 到 seq 输入（而非加到 token embedding 里）**
    - Pros: 信息融入更早（token embedding 层）
    - Cons: 需改 dataset.py,与 hash500k 的 seq_hash_vocab 路径冲突（hash 后再 concat 无物理意义）;**不推荐**。

- **D: 多 target（同一 batch 内多个 target item 做 MoCo 对比学习）**
    - Pros: self-supervised,可能增强 target 表征质量
    - Cons: 超出 end-to-end supervised baseline 的复杂度,ddl 21 天下风险过高;**不推荐**。

---

## Consequences（影响与外溢）

### 正向
- **解锁 target-aware seq encoding**：当前 MultiSeqHyFormer 是 target-agnostic 的,本方案从根源上修复这一点,预期是**目前 GAP 最大的改造空间**。
- **与 ICQ 正交**：若方案 A 正向,可与 ICQ 叠加,总增益可能 +0.0005 ~ +0.002 AUC（保守估计）。
- **推理时长无显著影响**：方案 A 只加一次 per-sequence Linear + 一次 broadcast 加法,推理时长增量 < 1s。

### 负向 / 待观察
- **过拟合风险**：target-item 直接注入会放大模型对 target-item id 的记忆倾向;需观察 valid AUC 曲线是否在中后期掉头（类似 sparse reinit 问题）。
- **与 sparse reinit 交互未知**：当前 `reinit_sparse_after_epoch=1` 每 epoch 重置 hash 表;target_repr 来自 item_ns,可能也受 reinit 影响,需在 half-train 时监控 E1→E2 是否能正常跨越 reinit barrier。
- **必须 AUC-select EarlyStopping**（team-memory 铁律）：配合本实验必须严格走 `--early_stop_metric auc` 默认值,不能因"本地 LogLoss 更优" 误用 LogLoss-select 导致 AUC 最优 ckpt 被覆盖。

### 长期影响
- 若方案 A 验证成功,该 seq-token 级 target conditioning 可以进一步扩展到 **query + key/value + co-action** 三层联合,形成论文级 unified target-aware module,对 KDD 报告（T6）也是一个可写成 Section 的贡献点。

---

## Verification（如何验证决策正确）

### 已验证（静态分析 / 本地推演 / S1 smoke · 2026-05-04 晚）
- 参数量估算已完成,方案 A 增量 16k 远小于现有 MultiSeqQueryGenerator（~500k）。
- 位置分析已完成: `PCVRHyFormer.forward` 的 seq_encoders 调用路径允许 non-invasive 插入。
- 与 ICQ 的正交性在代码层面成立（query 侧 vs key/value 侧）,可独立启用。
- **S1 本地 smoke 通过**（demo_1000 + 10 step train + infer）:
    - `py_compile` 5 文件全部通过;ReadLints 无错误
    - `train.py` + `--target_item_seq_injection additive --target_inject_alpha_init 0.1` 正常完成训练,最终 Early Summary / best_model / training_summary.json 正常写出
    - `train_config.json` 正确包含 `target_item_seq_injection: "additive"` + `target_inject_alpha_init: 0.1`
    - `infer.py` 动态读取 train_config 重建模型成功,输出 1000 predictions from 2000 rows,score 分布 0.27~0.66（mean 0.44）,无崩坏
    - **可复现性验证**:同 seed 下 `additive α=0` 两次跑结果 bit-identical（loss 0.376609776701246 完全一致）
    - **注入生效验证**:`additive α=0.1` 相对 `additive α=0.0` 10 step 后 loss 略有差异（0.37673 vs 0.37661）, AUC 0.963 vs 0.958,说明注入机制真正参与 forward 计算
    - **与 off 基线轻微不一致**: `off` 模式与 `additive α=0` 结果不 bit-identical（源于 4 个 `Linear(D,D)` init 消耗 torch RNG 流）。Forward 仍是数学 identity, 生产上直接用 `additive α=0` 作为起点, 无需回退 off。

### 待验证（proxy → half-train → full train 漏斗）

| 阶段 | 环境 | 期望信号 | 决策阈值 |
|---|---|---|---|
| S1 本地 smoke | demo_1000 | 2 step train 无 shape error + predictions.json 正常输出 | shape 错误直接回滚 |
| S2 fast proxy | 128/32 RG + 1500 steps | 相对 t13tf-only proxy 基线 AUC 不退超过 0.002（参考"大参数假负向"规律） | 退 < 0.002 继续 S3；退 > 0.002 也进 S3（ICQ 规律） |
| S3 half-train | 2~3 epoch 全 RG | E3 best AUC ≥ t13tf E3 best + 0.0002 | 达标进 S4；不达标回滚 |
| S4 full train | 10 epoch + batch 128 | best AUC ≥ 0.869（现 t13tf 0.8685） | 达标进 S5 |
| S5 leaderboard | evaluation slot | LB ≥ 0.845（现 0.844956） | 达标则 commit 方案 A;否则分析 local → LB gap |

### 回滚触发条件
- S3 half-train E3 best AUC < t13tf E3 best AUC - 0.0005 → 方案 A 回滚,尝试方案 C。
- S4 full train 曲线出现 E5+ 掉头 > 0.001 → 过拟合风险,加 dropout 或 reduce α_i init 值重试。
- S5 LB < 0.844 → local/LB 系统性背离,方案 A 回滚,重新评估是否是 train/test 分布 shift 问题。

---

## 附: 实际代码改动（2026-05-04 晚落地）

- `baseline/model.py`:
    - 新增 `TargetItemSeqInjector` 类（~85 行, 含文档）—— 支持 `off` / `additive` 两种 mode,`off` 是零参 identity
    - `PCVRHyFormer.__init__`: 新增参数 `target_item_seq_injection` + `target_inject_alpha_init`, 保存 `num_user_ns` / `num_item_ns` 到 self, 实例化 `self.target_item_seq_injector`
    - `PCVRHyFormer.forward` / `predict`: 在 `_build_seq_tokens` 后、路径分叉前调用 `_maybe_inject_target_into_seq`
    - 新增 helper `_maybe_inject_target_into_seq`: 从 ns_tokens 按**显式 offset** 切出 item_ns 段（不依赖"末尾"假设,未来安全）,mean pooling 后传给 injector
- `baseline/train.py`:
    - CLI 参数注册（`--target_item_seq_injection {off,additive}`, `--target_inject_alpha_init float`）
    - `model_args` dict 加入新字段,自动落盘到 `train_config.json`
- `baseline/infer.py`:
    - `model_args` dict 同步从 `cfg.get()` 读取新字段, 默认值 `off` / `0.0`
- **总计 ~120 行代码改动**(略超 ADR-004 原估计 60 行,主要是 injector 类的文档注释 + 两处 helper)。所有改动都在 `baseline/` 下, `dataset.py` / `trainer.py` 零改动。

### 平台 evaluation 兼容性注意事项

- **开启 `additive` 时,evaluation 必须上传本仓库自定义 `infer.py`**:平台默认 infer.py 不一定认识 `target_item_seq_injection` kwarg,可能 pass 到 `PCVRHyFormer.__init__` 时报 unexpected kwarg 错误;我们的 infer.py 已显式读取该字段
- 如果只用 `off` 模式 fallback:平台默认 infer.py 在 `cfg.get()` 模式下会取不到该字段 → 走 `PCVRHyFormer` 默认 `off` → **可能仍然兼容**,但**不要赌**,第一次 evaluation 时务必用自定义 infer.py。已落 team-memory 铁律。

---

## 附: 初版代码改动预估（保留作对比）

原 ADR Draft 预估 ~60 行,实际 ~120 行(主要偏差来自 injector 类的文档注释和 `_maybe_inject_target_into_seq` helper 的安全性检查,核心逻辑仍然 ~30 行)。

---

## 最终提醒

本 ADR 属于**方法学破局**类,不是常规参数微调。在 ddl 21 天 + 评估额度 4 次/天 的约束下,此类改造只允许 **≤ 2 次** 全流程实验（S3→S4→S5 一整轮,每轮约需 3~4 天）。必须把 S1~S3 做扎实再消耗 S4 / S5 配额,严禁跳过 half-train 直接 full train。
