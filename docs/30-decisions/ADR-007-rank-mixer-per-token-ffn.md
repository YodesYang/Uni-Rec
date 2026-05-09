# ADR-007 · RankMixerBlock Per-token FFN（修正共享 FFN 偏差）

> **Status**: **❌ Deprecated**（2026-05-09 CEST 09:43 · EXP-034 单独 half-train 证伪后降级 · 见下方 §Deprecation Note after EXP-034）
> **Date**: 2026-05-04（首次 Accepted）· 2026-05-09（Deprecated）
> **Related Tasks**: T19（见 `docs/20-tasks/active.md`）
> **Related Experiments**: EXP-023（本地 smoke）· **EXP-034（单独 half-train 证伪）**
> **External References**:
> - RankMixer: Scaling Up Ranking Models in Industrial Recommenders（arXiv 2507.15551, 2025）
> - `docs/00-context/model-py-section-map.md` §RankMixer：已明确标注 "仓库 vs 论文的细节差异 —— 仓库用 shared FFN，论文是 per-token FFN"

---

## Context（背景与触发原因）

### 触发证据

- **飞机伴读地图已明确标注这个差异**：`docs/00-context/model-py-section-map.md` line 160-164：
  > 论文 RankMixer 原版是 **"Per-token FFN"**：T 个 token 各自有独立 FFN 参数（总参数量 T × FFN size）
  > **仓库实现用的是 shared FFN**（line 352-353 `self.fc1 = nn.Linear(d_model, d_model * hidden_mult)`）→ 所有 token 共享一套 FFN 参数
  > **这是一个潜在的改进点**：如果论文里 Per-token FFN 是核心，而仓库用的 shared，可能你做一个真正的 Per-token FFN 版本就是 +0.002~0.005 AUC 的子弹
- **RankMixer 原论文动机**（arXiv 2507.15551）：Per-token FFN 的设计理由是"handling feature subspace heterogeneity ... allocating independent parameters to different feature subspaces, rather than allowing domination by a single shared model"。共享 FFN 会让少数高频主导特征（dominant feature）淹没长尾 token 的表达。
- **本赛题长尾极端**：`domain_c_seq_47` vocab=278M（已用 hash500k），item 长尾分布典型，per-token FFN 的独立容量**直接对口**该场景。
- **与 T16 / T17 / T18 完全正交**：本 ADR 只改 `RankMixerBlock` 内部的 FFN 参数化，不动 seq encoder (T16)、valid split (T17)、loss (T18)，可叠加。

### 问题域

- **共享 FFN 的潜在问题**：所有 T 个 token（`num_queries * num_sequences + num_ns` = 我们实际配置 15~16）共用一套 (D × 4D) → (4D × D) 的两层 MLP。Q tokens 来自 4 个不同 domain，NS tokens 来自 user/item 特征，**语义完全异质**，但被迫用同一组 FFN 权重。
- **Per-token FFN 代价可控**：D=64 / hidden_mult=4 / T≈16 下，每个 block 新增 dense 参数量约 **496k**（shared 是 33k）。`num_hyformer_blocks=2` 全模型增量 **~993k dense**。跟 ICQ（+429k）、hash500k 的 sparse embedding 比，是同量级增量，推理时长增量 < 5%。
- **向后兼容**：`--rank_mixer_ffn_mode shared`（默认）必须与现在的旧行为 **bit-identical**。新路径 `per_token` 通过 opt-in。

### 时间压力

- 5/25 初赛 ddl，21 天。
- 代码改动最小（~60 行 `RankMixerBlock` + 参数串联），与 T16/T17/T18 并行，可搭车 half-train。
- ROI 估算（按外部 +0.002~0.005 AUC 论文 anchor × 30% LB 转化率）：**LB +0.0006 ~ +0.0015**。

---

## Decision（最终决定做什么）

**在 `RankMixerBlock.__init__` 加 `ffn_mode: Literal['shared', 'per_token']` 参数（默认 `shared` 保持向后兼容），`per_token` 路径下把 FFN 参数化为每个 token 独立的权重矩阵，forward 通过 `torch.einsum` 向量化实现，避免 Python for-loop。**

### 设计细节

#### 1. 参数化

- **Shared（legacy，默认）**：`self.fc1 = nn.Linear(D, D*hm)`，`self.fc2 = nn.Linear(D*hm, D)`
  - 参数量：`2 * D² * hm + D*hm + D` = 33,344（D=64, hm=4, 加 2 个 LayerNorm）

- **Per-token（新）**：
  ```python
  self.fc1_weight = nn.Parameter(torch.empty(T, D, D*hm))    # (T, D, H)
  self.fc1_bias   = nn.Parameter(torch.zeros(T, D*hm))       # (T, H)
  self.fc2_weight = nn.Parameter(torch.empty(T, D*hm, D))    # (T, H, D)
  self.fc2_bias   = nn.Parameter(torch.zeros(T, D))          # (T, D)
  ```
  - 参数量：`T * (2 * D² * hm + D*hm + D)` + 2 LN = T× shared FFN + LN = **529,664**（T=16, 是 shared 的 15.88×）

#### 2. Forward 向量化

```python
# per_token path
h = einsum('btd,tdh->bth', x, fc1_weight) + fc1_bias   # (B, T, D) -> (B, T, H)
h = F.gelu(h); h = dropout(h)
y = einsum('bth,thd->btd', h, fc2_weight) + fc2_bias   # (B, T, H) -> (B, T, D)
```

- 用 `einsum` 而非 `nn.ModuleList[Linear]*T + for-loop`，保持 GPU 并行与 `torch.compile` 兼容性。
- 无中间 `(B, T, D, H)` 张量分配，内存 overhead 可忽略。

#### 3. 初始化

- **Shared 保持原样**（PyTorch 默认 `nn.Linear` Kaiming uniform）。
- **Per-token** 对每个 token 独立 Kaiming uniform + 独立 bias uniform `[-1/√fan_in, +1/√fan_in]`，等价于 T 个独立的 `nn.Linear`。
  - 数值验证：token 5 的 per-token 输出与独立 `nn.Linear(D, H)` + `nn.Linear(H, D)`（权重 copy 对齐）**bit-identical**（max|Δ| = 0.00e+00）。

#### 4. 与 `rank_mixer_mode`（full / ffn_only / none）的交互

| `rank_mixer_mode` | 是否生效 |
|---|---|
| `full` | ✅ token_mixing + per-token FFN |
| `ffn_only` | ✅ 仅 per-token FFN（跳过 token_mixing） |
| `none` | ❌ identity passthrough，不创建任何 FFN 参数（两种 `ffn_mode` 都是 0 params） |

#### 5. CLI 接入

```
--rank_mixer_ffn_mode {shared, per_token}   # 默认 shared
```

自动通过 `vars(args)` 落入 `train_config.json`。`infer.py` 的 `model_args` 加 `cfg.get("rank_mixer_ffn_mode", "shared")` 做向后兼容（平台默认 infer.py 已存在的 ckpt 不会崩）。

#### 6. 跟 `PCVRHyFormer.__init__` 的 kwarg 传递链

```
train.py CLI → model_args dict → PCVRHyFormer.__init__(rank_mixer_ffn_mode=...)
                                      ↓ stored self.rank_mixer_ffn_mode
                                      ↓ 传给 MultiSeqHyFormerBlock(rank_mixer_ffn_mode=...)
                                            ↓ 传给 RankMixerBlock(ffn_mode=...)
```

---

## Alternatives Considered（其他备选）

- **A: `nn.ModuleList[Linear]*T` + for-loop**
  - Pros: 代码最直观
  - Cons: Python-level for-loop 在 GPU 上慢；`torch.compile` 不能 fuse；PyTorch JIT trace 体积大。**不推荐**。

- **B: 用 grouped-linear kernel (torch.bmm)**
  - Pros: 底层 cuBLAS 直接支持
  - Cons: 需要 `(T, B, D)` 转置，shape manipulation overhead 比 einsum 更高。选了 einsum。

- **C: 保持 shared，但增加 `T`（扩 NS tokens）**
  - Pros: 不改架构
  - Cons: T 变大约束 d_model 整除更难满足（当前 d=64 / T=15 已是 sweet spot）；且论文 anchor 明确说 per-token FFN 的动机是"subspace heterogeneity"而非 T 大小。

- **D: Sparse-MoE（RankMixer 论文 §5.2 Sparse-MoE 变体）**
  - Pros: 参数量级可扩到 B+ 级
  - Cons: MoE routing 代码量巨大，且本赛题数据 1.9M 行用 MoE 大材小用；竞赛红线又禁 ensemble，MoE 属"端到端 MoE" OK 的灰色地带，但风险高。**不在本次范围**。

- **E: 不做，只做 T16 / T18 的叠加**
  - Pros: 零风险
  - Cons: 外部 anchor +0.002~0.005 AUC 的子弹放弃，ddl 21 天下每一颗子弹都不能白放过。**不选**。

---

## Consequences（后果与副作用）

### 正向

- **容量解锁**：T 个 token 各自独立 FFN，消除"dominant feature 淹没长尾"机制
- **与 T16/T17/T18 完全正交**，可叠加至四合一 half-train
- **向后兼容严格**：不传 `--rank_mixer_ffn_mode` 默认 `shared`，与旧行为 bit-identical（10 step smoke 验证 AUC=0.8737376434 与昨天 T18 backward-compat case 完全一致）

### 负向 / 待观察

- **参数量 +993k dense**（`num_hyformer_blocks=2`），与 ICQ 同量级 → 遵循 team-memory "大参数分支 proxy 假负向" 铁律，**S2 proxy 负向也别放弃，直接上 half-train**
- **过拟合风险升高**：容量增加 ~15×，demo 10-step smoke 已观察到 `train_loss` 明显更低（0.31 vs 0.37）。需要配合：
  - AUC-select EarlyStopping（已默认）
  - LogLoss rebound guard `patience=2`（比 shared 路径更容易反弹）
  - 如 full train 仍见过拟合，下调 `hidden_mult 4→3` 或加 `dropout`
- **推理时长增加**：预计 +5~10%（精确值待 full train 测量），远低于 30min 硬上限（当前 SwiGLU 82s）
- **Init 流程 per-token 循环**：`__init__` 里有个 `for t in range(T)` 做初始化，T=16 量级可忽略，不影响训练启动时间

### 不在本 ADR 覆盖范围

- 不动 `RankMixerBlock.token_mixing`（参数无关的 reshape，per-token FFN 不改这块）
- 不做 Sparse-MoE 变体（超出最小改动红线，且与 ensemble 灰色地带接近）
- 不调整 `hidden_mult`（默认 4，若 per-token full train 见过拟合再考虑降到 3）

---

## Verification（验证计划）

### 已验证（S0 Static / S1 Local Smoke · 2026-05-04 晚）

**S0 Static** ✅
- `python3 -m py_compile baseline/{utils,trainer,train,infer,dataset,model}.py` exit 0
- `ReadLints` 无新错误（仅环境 basedpyright 解析 warning）

**S1 单元级数值验证** ✅（10 项检查全过）

| # | 检查 | 结果 |
|---|---|---|
| 1 | shared FFN 参数量（D=64, T=16, hm=4） | 33,344 ✓ |
| 2 | per-token 参数量 | 529,664 = **15.88× shared**（≈T） ✓ |
| 3 | forward shape 两路径 | (2,16,64) → (2,16,64) ✓ |
| 4 | per-token 独立性：扰动 token 3 fc1_weight ×1000 | 只 token 3 变（188），其他严格 0 ✓ |
| 5 | backward 流通两路径 | grad 非零，不 NaN ✓ |
| 6 | 非法 `ffn_mode` 抛 ValueError | ✓ |
| 7 | per-token token 5 输出 == 独立 `nn.Linear(D,H)+nn.Linear(H,D)`（weight copy） | max\|Δ\| = **0.00e+00** (bit-identical) ✓ |
| 8 | `mode='ffn_only' + ffn_mode='per_token'` 兼容 | ✓ |
| 9 | `mode='none'` 两种 `ffn_mode` 都无参 | 0 params ✓ |
| 10 | `MultiSeqHyFormerBlock` 透传 `ffn_mode` + 参数量增量验证 | per-token block +496k dense params ✓ |

**S1 集成级 smoke**（demo_1000, 10 step train + max_train_steps 早停）✅

| 配置 | best AUC | best LogLoss | train loss @ step 9 | 说明 |
|---|---|---|---|---|
| `--rank_mixer_ffn_mode shared`（默认） | **0.8737376434** | 0.3066 | 0.3740 | **与 T18 backward-compat case bit-identical**（2 轮都是 0.8737376434） |
| `--rank_mixer_ffn_mode per_token` | 0.8758085184 | 0.3097 | **0.3147** | train loss 显著更低（容量增加），10 步 AUC +0.0021；LogLoss +0.0030（过拟合迹象符合预期） |

- `train_config.json` 正确写入 `rank_mixer_ffn_mode` 字段
- `best_model/` 正常落盘，`training_summary.json` 正确记录

### 待验证（proxy → half-train → full-train 漏斗）

| 阶段 | 环境 | 期望信号 | 决策阈值 |
|---|---|---|---|
| S1 本地 smoke | demo, 10 step | shape + 向后兼容 + 独立性 | **已通过** |
| S2 fast proxy | 128/32 RG + 1500 steps + `--rank_mixer_ffn_mode per_token` | 相对 t13tf proxy 不退 > 0.003 | ≥ −0.003 进 S3；< −0.003 仍进 S3（大参数分支 proxy 假负向铁律，ICQ / hash500k 先例） |
| S3 half-train | 2~4 epoch 全 RG + `--holdout_ratio 0.05` | E3 dev AUC ≥ t13tf E3 + 0.0005 | 达标进 S4；差距大回滚或叠加 dropout |
| S4 full train | 10 epoch + batch 128 + `--logloss_rebound_patience 2` | best dev AUC ≥ 0.870 | 达标进 S5 |
| S5 leaderboard | 1 evaluation slot | LB ≥ 0.847 | 达标 commit ADR-007 |

### 回滚触发条件

- S3 half-train E3 dev AUC 比 t13tf E3 低 > 0.001 AND 训练 loss 持续高（说明 per-token 没 learn 出差异化）→ 回滚，考虑 `hidden_mult 4→3` 降容量
- S4 曲线 E5+ 出现严重 LogLoss 反弹（ΔLogLoss > 0.005）但 AUC 停滞 → 容量过大，加 `--dropout_rate 0.1` 重试
- S5 LB < t13tf LB → local-LB 转化率不一致，per-token 容量优势被 valid→LB gap 吞掉

---

## 实际代码改动（2026-05-04 晚落地）

- `baseline/model.py`（+~85 行）
  - `RankMixerBlock.__init__` 加 `ffn_mode: str = 'shared'` 参数，per-token 分支创建 `fc1_weight / fc1_bias / fc2_weight / fc2_bias` 四个 `nn.Parameter`
  - 新增 `_apply_ffn(x)` helper，dispatch shared / per_token
  - `forward` 改为调用 `_apply_ffn`，不再硬编码 `self.fc1/fc2`
  - `MultiSeqHyFormerBlock.__init__` 加 `rank_mixer_ffn_mode` 参数，透传给 `RankMixerBlock`
  - `PCVRHyFormer.__init__` 加 `rank_mixer_ffn_mode: str = 'shared'` 参数，store + 透传
- `baseline/train.py`（+~15 行）
  - CLI 注册 `--rank_mixer_ffn_mode {shared, per_token}`
  - `model_args` dict 加字段
  - 启动日志追加 `rank_mixer_ffn_mode=...`
- `baseline/infer.py`（+1 行）
  - `model_args` 的 `cfg.get("rank_mixer_ffn_mode", "shared")` 向后兼容
- `baseline/trainer.py`（+1 行）
  - `[summary]` 白名单 `key_train_config` 追加 `rank_mixer_mode` 与 `rank_mixer_ffn_mode`
- `baseline/utils.py` / `baseline/dataset.py`：**零改动**

**总计 ~100 行代码改动**。

### 平台 evaluation 兼容性

- `rank_mixer_ffn_mode` 是 `PCVRHyFormer.__init__` 的**新 kwarg** → 按 team-memory 铁律，**首次平台 evaluation 必须上传自定义 `infer.py`**
- 已在 `infer.py` 加 `cfg.get(..., "shared")` 默认值兜底，若平台默认 infer.py 不识别该字段，模型走 shared 路径（与存 shared 的 ckpt 兼容；但存 per_token 的 ckpt 用平台默认 infer.py 加载会 shape mismatch）
- **重要**：per_token 的 ckpt **必须**用自定义 infer.py 推理，否则 `model_args` 会 fallback 到 `shared`，`fc1_weight (T, D, H)` 无法 load 到 `self.fc1.weight (H, D)` 上，推理会报 state_dict mismatch

---

## 最终提醒

本 ADR 遵守 team-memory 两条铁律：

1. **大参数 dense 分支 proxy 假负向** → S2 proxy 负向也别回滚，必须直接上 half-train
2. **新 `PCVRHyFormer.__init__` kwarg 首次 evaluation 必上传自定义 infer.py**

与 T16 / T17 / T18 的关系：**完全正交**。可搭车"四合一 half-train"（T16 additive + T18 auc-loss + T17 holdout + T19 per-token FFN）一次验证。

---

## §Deprecation Note after EXP-034（2026-05-09 CEST 09:43）

### 判决

`rank_mixer_ffn_mode=per_token` 在本赛题（1.9M train rows）**Deprecated**。默认路径保持 `shared`，代码分支保留但不再推荐启用。

### 证据

EXP-034 `t19-perffn-halftrain-gap05-noamp`（严格单变量 vs `icq-half-gap05` baseline）：

| 指标 | Baseline E2 | T19 E2 | Δ |
|---|---|---|---|
| Dev AUC | 0.841124 | 0.840142 | **−0.000982**（10× rerun 噪声） |
| Val LogLoss | 0.22468 | 0.28208 | **+0.0574**（25% 绝对值异常） |

- `stop_reason=logloss_rebound_epoch_2` — LogLoss guard 在 E2 就触发，**baseline 要到 E5+ 才反弹**，说明 per-token FFN 严重提前过拟合
- 没跑到 E3（baseline 的 best），按 EXP-034 Go/No-go 直接落入"负向"区间

### 机制

**+34% Dense params（+992,640）在 1.9M × 30k steps/epoch 的数据规模下**严重 warmup 不足。每参数平均只摊到约 30 optimizer steps 的梯度信号（baseline shared FFN 约 120 steps，4× 更充分）。E2 刚勉强收敛 → E3 立刻 overfit → LogLoss guard 拦截。

### 与外部情报的印证

Seed 主楼 2026-05-04 情报（community-intel.md §问题-4）：

> "pure model 反向信号 ⚠️: 主流 rankmixer/onetrans/hyformer/mixformer 都基本为了提 MFU 改成 pure model, 没有太多精细的传统派小结构建模, 但试了下都掉分, 可能数据量不够"

Seed 的 "pure model 掉分" 和 T19 "per-token FFN 欠拟合"**是同一机制**：**1.9M 数据规模不支持容量扩张型架构 bet**。RankMixer 论文验证场景是**抖音日活级（十亿样本）**，与本赛题差 3 个量级。

### 保留代码的理由

- `rank_mixer_ffn_mode=per_token` 分支不回滚，保留历史 / 未来复赛 10× 数据时可重试
- 任何希望"在本赛题 1.9M 数据下启用" 的未来实验，**必须先解决数据量 warmup 问题**（例如：预训练 + 微调、更小 hidden_mult、更少 epoch 长度）

### 规则推广

把 "**容量扩张型 bet 在本赛题几乎必死**" 升级为 team-memory 铁律：
- 每参数平均 ≥ 60 steps 梯度 = 安全
- 每参数平均 30~60 steps = 高风险
- 每参数平均 ≤ 30 steps = 几乎必过拟合（T19 命中）

未来候选方向如需 Dense +20%+，**直接跳过，不投 quota**。

详见 [`EXP-034 §Result + §Lessons`](../40-experiments/EXP-034-t19-perffn-halftrain-2026-05-09.md)。
