# ADR-010 · DIN Integrated Inside HyFormer Block（T28）

> **Status**: **Proposed**（2026-05-11 CEST 05:15 起草 · 等 T29 或 T28 half-train 回流判决）
> **Date**: 2026-05-11 Proposed
> **Related Tasks**: T28（待 open，本 ADR 作为依据）· T25（T28 是 T25 的"block-integrated 升级"）
> **Related Experiments**: EXP-039（`t28-din-integrated-halftrain-gap05`，4 epoch half-train · 待启动）
> **External References**:
> - DIN 原论文 (Alibaba KDD 2018)：target-aware attention pooling
> - UniRec（hojiahao/TAAC2026 41 stars）方案 1：DIN + MLP + `interest_token` 生成机制
> - 小红书 DECEM（Industrial Track Rank 1, LB 0.852181, 2026-05-08 帖子）："我在前期就做了的 DIN + MLP"
> - EXP-033（T25 DIN bypass 实装 · smoke + ADR-009 前身）
> - EXP-036（T25 B LB 0.843433 · Δ+0.00421 vs baseline · 真信号）

---

## Context（背景与触发原因）

### 触发证据

#### 1. T25 bypass 已证 DIN target-aware attention 在本赛题 work

| 指标 | icq-half-gap05（baseline, gap=0.05）| t25-din-halftrain-gap05-seed42（T25 bypass）|
|---|---|---|
| E3 Dev AUC | 0.842443 | 0.843696（Δ+0.00125）|
| LB AUC | 0.839223 | **0.843433**（Δ+0.00421, 9× rerun 噪声）|
| Dev→LB gap | +0.0032 | +0.000263（最小！）|

**T25 是过去 48 天唯一 LB 真信号级方法学（+0.004 LB）**。但 T25 只是"bypass 版"—— DIN 的 target-aware interest vector 只在 classifier 前拼接一次，**不参与 HyFormer block 内部的 token mixing**。

#### 2. T25 bypass 的 3 个局限

- **信号只进入 classifier 1 次**：interest vector 没机会影响 HyFormer block 里的 Query-Decoding 或 RankMixer token mixing。模型通过 classifier 一次看到 target-aware 信号太晚。
- **输入是"未编码的 seq tokens"**：`_apply_din_interest_bypass` 用的是 `_build_seq_tokens` 原始 embedding，没经过 `MultiSeqHyFormerBlock.seq_encoders` 编码。原始 seq 在"interest pooling"阶段信号质量更低（充满 noisy 原始 id embedding）。
- **边际递减**：EXP-036 显示 gap=0 下 DIN 仅 Δ+0.000305 Dev（而 gap=0.05 下是 +0.00125）。机制假说：gap=0 近未来本来就容易凭 id 记忆，DIN 的 selective attention 作用小。更深入介入 block 内部才能放大增益。

#### 3. DECEM 原话 + UniRec 架构暗示 "integrated" 才是 DIN 真身

- DECEM 原话 "我在前期就做了的 **DIN + MLP**"（方案里和"MLP"一起提 → 暗示是 block 内部的 DIN + FFN，而非 classifier-level bypass）
- UniRec 的 `DIN target attention` 生成 `interest_token` **作为独立 NS-token 进入 backbone**（早于 classifier）
- 本项目 T25 bypass 只是"最小风险实现"；**block-integrated 才是 DIN 方法的原始形态**

#### 4. 从方法学剩余增量看：T28 是 T25 升级版，不是独立 bet

- DECEM 方法学峰值估算 ~0.847，本项目 SOTA 0.84637（seed 上沿）距 ~0.001
- DECEM 说 work 的 4 个方向：compile+AMP（已做）/ 时间特征（已做）/ Tokenization（T29 在做）/ DIN+MLP（T25 bypass 已做 +0.004 LB）
- 要继续推高 LB，需要**同类方向深挖** —— T28 就是 DIN 方向的深化

### 问题域

| 维度 | T25 bypass | T28 integrated |
|---|---|---|
| DIN 作用位置 | classifier 前（1 次）| 每个 HyFormer block 内部（num_blocks=2 次）|
| 输入 seq | 原始 embedding | 每 block encoded seq（渐进细化）|
| 作用对象 | output 拼接 | decoded_q_i（影响 RankMixer token mixing）|
| Dense 参数增量 | +104,896（+3.6%） | +201,602（+7.0%） |
| RankMixer token 数约束 | 不影响 | 不影响 |
| 与 T25 的关系 | — | 可叠加但首次 ablation 纯 T28 |

### Trade-offs（放弃了什么）

- **放弃"每 block 独立 interest token 作为新 NS token"**（候选 b）：会让 token 数变为 `num_queries*S + num_ns + 1`，与 `d_model % T == 0` 约束冲突（T 从 16 变 17，64 不整除 17）。放弃。
- **放弃"DIN 子模块跨 block 共享"**：虽然省 ~50% dense 参数，但每 block 的输入 seq 已被 encoder 修改（渐进细化），让 DIN 学"对不同精细度的 seq 用同一套参数"压制了表达力。选"每 block 独立 DIN"。
- **放弃 alpha 初始化 0.0**（完全 identity 起点）：0.1 是 T16 已验证的"安全但非零"起点，让梯度能流向 DIN 子模块又不至于一步扰动太大。

---

## Decision（最终决定做什么）

**在 `MultiSeqHyFormerBlock.__init__` 新增 2 个可选 kwarg**（向后兼容默认 False / 0.1）：
- `enable_din_integrated: bool = False`
- `din_integrated_alpha_init: float = 0.1`

**在 `PCVRHyFormer.__init__` 新增 2 个对应 kwarg**，在构造 `self.blocks` 时透传到每个 block。

**`MultiSeqHyFormerBlock.forward` 新增 optional `target_repr: Optional[Tensor]` 参数**：
- 仅在 `enable_din_integrated=True` 时必需，否则忽略
- 注入点：Step 2.5（Query Decoding 后、Token Fusion 前）
  ```python
  if self.enable_din_integrated:
      interest = self.din_extractor(target_repr, next_seqs, next_masks)  # (B, D)
      decoded_qs = [q_i + self.din_alpha * interest.unsqueeze(1) for q_i in decoded_qs]
  ```

**`PCVRHyFormer.forward/predict` 调用链**：
1. `_compute_target_repr(ns_tokens)` 计算一次（仅当 `enable_din_integrated=True`）
2. 传给 `_run_multi_seq_blocks(..., target_repr=target_repr)`
3. `_run_multi_seq_blocks` forward 每 block 时传 `target_repr=target_repr`

**关键实现细节**：
- **`_compute_target_repr` helper dedupe**：T25 bypass 和 T28 integrated 共用 "target = mean(item_ns)" 规约，不重复实现
- **每 block 独立 `DINInterestExtractor` 实例**：复用已有类（T25 G1 落地 2026-05-09），参数独立可学
- **每 block 独立 `din_alpha` 标量**：`nn.Parameter(torch.full((1,), 0.1))`，让不同 block 学不同的 injection 强度
- **渐进细化**：block 0 见"原始 seq → encoder0 编码后"的 interest；block 1 见"block 0 输出 → encoder1 编码后"的 interest（因为每 block 有独立 encoder）—— 这是与 T25 bypass 的根本差异

---

## Verification（验证计划 / Go / No-Go 阈值）

### S1 本地 smoke（✅ 2026-05-11 CEST 05:00 完成）

- `py_compile baseline/model.py baseline/train.py baseline/trainer.py baseline/dataset.py baseline/infer.py baseline/utils.py` 全过
- `ReadLints` 无 linter 错误
- **S1 T28 off smoke**：`enable_din_integrated: False`, Total params **353,334,977**（与 pre-T28 baseline **bit-identical**，backward-compat 完美）
- **S2 T28 on smoke**：`enable_din_integrated: True`, Total params **353,536,579**（Δ+201,602 = 2 blocks × ~100,801，符合理论估算）
- **S3 T28 on infer smoke**：train_config.json 正确落盘 `enable_din_integrated: True`，infer.py `cfg.get("enable_din_integrated", False)` fallback 正确加载，无 state_dict missing/unexpected keys，`Saved 1000 predictions from 2000 rows`

### S2 平台 half-train（待启动 · 1 个 training slot）

- **Job Name**: `t28-din-integrated-halftrain-gap05-seed42`
- **Config**：复用 `icq-half-gap05` EXP-030 baseline 所有参数 + 加 `--enable_din_integrated --din_integrated_alpha_init 0.1`
- **注意**：**不叠加 T25 bypass**（`--enable_din_interest` 不加），首次 ablation 纯 T28
- **注意**：**新 kwarg 在 run.sh 里显式出现 + 首次 platform evaluation 必须上传自定义 infer.py**（team-memory 铁律：新 kwarg 首次平台 evaluation 必须勾选自定义 infer.py 上传）

### 判决阈值（vs `icq-half-gap05` E3 Dev 0.842443）

T28 的 benchmark 不是 baseline，**是 T25 bypass**（0.843696 E3 Dev / 0.843433 LB）。要证明 T28 值得替换 T25，需要：

| E3 Dev AUC | 判决 | 动作 |
|---|---|---|
| ≥ **0.847**（Δ+0.005 vs baseline / +0.003 vs T25）| 🎯 **强真信号** | T28 替换 T25 成为主路径，立即 evaluation |
| [0.8452, 0.847) | 🟡 T28 好过 T25 但需 LB 验证 | 消耗 1 eval quota 验证 LB |
| [0.8437, 0.8452) | ⚪ 持平 T25 | 保留作备选，不优先投入 |
| < 0.8437 | ❌ T28 无增益 | Deprecated，T25 bypass 保持主路径 |

**LB 验证阈值**（若 Dev ≥ 0.8452 触发 evaluation）：
- LB ≥ **0.847** → SOTA 大突破 + Top 20 概率飙升
- LB ≥ **0.844** → 优于 T25 bypass LB 0.843433（真信号）
- LB < 0.843 → T28 无增益 vs T25

### S3 叠加（若 T28 S2 通过）

- `t28-din-integrated-t29-tokfix-halftrain-gap05-seed42`（T28 + T29 Tokenization 组合）
- `t28-din-integrated-gap0-seed42`（T28 切 gap=0 领地冲 SOTA 上沿）

S3 目标：**LB ≥ 0.850**（Top 20）

---

## Consequences（后果）

### 正面

- **T25 bypass 的自然升级**：复用已有 `DINInterestExtractor` + 共享 target_repr helper，零新组件设计
- **严格单变量 ablation**：T28 on → T25 off，纯新方向判决
- **完全 backward-compat**：T28 off 时 Total params / forward 路径与 pre-T28 **bit-identical**（S1 smoke 验证）
- **正交可叠加**：与 T29 Tokenization fix、T27 compile+AMP、gap=0 / gap=0.05 领地切换、seed sweep 都正交
- **平台兼容性保持**：仅新增 PCVRHyFormer kwarg + train_config.json 字段；infer.py 只是 cfg.get fallback，新 ckpt 在旧 infer.py 下会用 default False → 安全降级（但为避免歧义，首次 evaluation 上传自定义 infer.py）

### 负面 / 待观察

- **Dense 参数 +7%**（353.3M → 353.5M total，但主要是 sparse embedding，Dense 部分增量是 +7% 相对 Dense）—— 在 1.9M 数据下每参数平均梯度 steps 可能略紧张。team-memory 铁律"容量扩张 +34% Dense 必死（T19 教训）"的警告带是 20%+，T28 的 +7% 在安全区内。
- **compile wrapper 兼容性**：`din_alpha` 是 `nn.Parameter`，torch.compile 处理 parameter 没问题；但 `DINInterestExtractor` 里 per-seq for-loop 触发 dynamic shape（不同 seq_len）可能触发 recompile。**建议首次 T28 + compile 组合前先 T28 off compile 做基线**，避免 compile 出问题被误归因为 T28。
- **首次 platform evaluation 必须上传自定义 infer.py**（team-memory 铁律）。

### 红线检查

- ❌ 不 ensemble / 不 SWA — T28 是单一架构改动，单 ckpt 评估
- ❌ 不外部数据 — DIN target_repr 来自 item_ns embedding，完全内部
- ✅ 单一架构最终提交 — T28 作为 HyFormer block 的增强组件，整体仍是 PCVRHyFormer
- ✅ 容量扩张在安全带内（+7% Dense < +20% 警戒线）

---

## Alternatives Considered（被放弃的方案）

### B · 每 seq 生成独立 interest token 作为新 NS-token

- **机制**：不注入 decoded_q，而是把 S 个 `interest_i` 作为额外 token 拼到 concat 里
- **放弃理由**：token 数变为 `Nq*S + Nns + S`（这里 S=4），`T=20`，**64 % 20 ≠ 0** 破坏 RankMixer d_model 约束。要修需降 `num_queries` 或扩 `d_model`，双刃剑

### C · 把 T25 bypass 改成"bypass + integrated" 双路径

- **放弃理由**：首次验证必须纯 T28 vs 纯 T25 对照，叠加会产生归因歧义（T18 组合实验教训）。验证 T28 成功后再考虑

### D · T28 配合 `target_item_seq_injection=additive` 变成 seq 侧注入

- **放弃理由**：T16 additive 已证伪（2026-05-04 full-train LB 负向）。T28 的 mechanism 是 query 侧 attention pooling，与 T16 additive 正交但机制不同，首次不混用

### E · 跨 block 共享 DINInterestExtractor（参数省一半）

- **放弃理由**：每 block 的输入 seq 是不同"精细度"（encoder 渐进编码），共享 DIN 压制表达力。增量 +3.6%（和 T25 同）虽然省，但与 T28 "integrated 深挖" 的初心不符

### F · `din_alpha` 初始化 0.0（完全 identity 起点）

- **放弃理由**：0 初始化让梯度完全依赖 din_extractor 的随机初始化 + 数据传递，可能学不到信号。0.1 是 T16 已验证的"安全但非零"偏好

---

## Open Questions（悬而未决）

1. **T28 与 compile+AMP 叠加的 dynamo 兼容性**：`DINInterestExtractor.forward` 有 per-seq for-loop，可能触发多次 dynamo recompile。首次 T28 platform job **优先不开 torch.compile**（即用 AMP bf16 但不 compile），确认 T28 本身信号后再补 compile
2. **alpha 初值 0.1 是否最优**：可能 0.05 或 0.2 更好。本次固定 0.1，若 T28 signal 正向但边缘，未来可扫 alpha ∈ {0.05, 0.2}
3. **T28 block 数 ≥ 3 时的稳定性**：当前 `num_hyformer_blocks=2`，若 scaling 到 3+ 块，每块独立 DIN 的梯度和总容量是否稳定未知。当前不探索此方向
4. **T28 和 T29 叠加顺序**：T29 Tokenization fix 改 fid 拼接 → 间接改变 ns_tokens 内容 → 改变 `_compute_target_repr` 的 item_ns mean → 改变 T28 的 target_repr。两者**不是严格正交**，但交互方向应该正向（更合理的 tokenization 产出更高质量的 target_repr）。S3 叠加实验将验证

---

## References

- [EXP-039](../40-experiments/EXP-039-t28-din-integrated-halftrain-2026-05-11.md)（T28 实验计划 + smoke 数据）
- [ADR-004](./ADR-004-target-item-seq-conditioning.md)（T16 additive seq 侧注入，已证伪）
- [EXP-033](../40-experiments/EXP-033-t25-din-halftrain-2026-05-09.md)（T25 DIN bypass 实装）
- [EXP-036](../40-experiments/EXP-036-gap0-t25-seed-sweep-tokenization-2026-05-10.md)（T25 B LB 0.843433 真信号）
- [ADR-009](./ADR-009-tokenization-fix-ns-groups.md)（T29 Tokenization fix · 与 T28 正交可叠加）
- `baseline/model.py:1149-1344`（MultiSeqHyFormerBlock · T28 注入点）
- `baseline/model.py:563-655`（DINInterestExtractor · 复用 T25 类）
- `baseline/model.py:2780-2805`（`_compute_target_repr` helper · T25/T28 共用）
- Community Intel `docs/10-research/community-intel.md` §工业赛道 Rank 1 DECEM "DIN + MLP"
