# ADR-006 · CombinedAUCLoss = WeightedBCE + PairwiseBPR

> **Status**: ❌ **Deprecated**（2026-05-04 晚 10 · 三合一 E4 LB evaluation 0.836445 实锐证伪,vs SOTA 0.84637 **Δ LB −0.00993**）
> **Original Status**: Accepted（2026-05-04 晚 · S1 本地 smoke 通过后升 Accepted；晚 10 被 LB 证伪后降级为 Deprecated）
> **Date**: 2026-05-04
> **Related Tasks**: T18（见 `docs/20-tasks/active.md`，已关闭）
> **Related Experiments**: EXP-022（三合一 half-train LB evaluation 实锐）
> **External References**:
> - Twitter / 腾讯: "Understanding the Ranking Loss for Recommendation with Sparse User Feedback", arXiv 2403.14144
> - PWiseR: "Pairwise Ranking Loss for Multi-Task Learning in Recommender Systems", arXiv 2406.02163
> - UniRec (hojiahao/TAAC2026, 同赛题开源参考): `model/heads.py::CombinedAUCLoss`

---

## ⚠️ Deprecation Notice（2026-05-04 晚 10）

**T18 CombinedAUCLoss 路线已 Deprecated**。实验证据：

| Job | Config | Local dev | Local holdout | **LB** | Gap |
|---|---|---|---|---|---|
| t13tf-full E4（参考基线，纯 BCE） | T13+TF | 0.868495 | — | 0.844956 | +0.0235 |
| icq-half E4（T16 基线等价，纯 BCE） | T16 additive | 0.868619 | — | **0.84637** | +0.0222 |
| **三合一 E4（T16+T18 w=0.3）** | T16 + T18 | **0.865516** | **0.867196** | **0.836445** ❌ | **+0.0291** |

**关键数字**：
- **LB Δ vs SOTA = −0.00993**（比 icq-half E4 LB 低 ~1%）
- dev-LB gap **0.0291** / holdout-LB gap **0.0308** 都是历史最大
- dev Δ vs icq-half = −0.003, LB Δ = −0.010 → **LB 负向放大 3.3×**

**机制解释（2026-05-05 修订 · 原版基于 "calibration 破坏" 的论证有逻辑漏洞）**：

### 🔍 原版论证的 bug

原 Deprecation Notice 写："BPR 破坏 sigmoid 绝对值 calibration → LB AUC 下降 −0.010"。

**这条逻辑在严格意义下站不住**：
- AUC = `P(score_pos > score_neg)`，是**排序指标**
- 对 score 做任意**单调变换**（例如整体 ×10、+5、logit transform）不改变 AUC
- 所以**在同分布 test 下，calibration 破坏不会直接降 AUC**

### ✅ 修订后的机制解释（更严谨）

BPR 在本赛题下 LB 崩盘的真正机制，是**"学到的排序依赖于 train 绝对分数分布，在 test 分布偏移下不稳定"**：

1. **BPR 的训练目标是推开 pair 间的分数差**：`-log σ(s_pos - s_neg - margin)`，梯度驱使模型把 positive 和 negative 的 score 推向极端（例如 pos → +∞, neg → −∞）
2. **这种"推极端"的方式依赖于 train batch 的局部分布**——哪些是 hard positive、哪些是 hard negative，由 batch 采样决定
3. 当 **train/test 分布发生 shift**（本赛题 test 是远未来 + 时间分布漂移）：
   - test 上"曾经是 easy positive 的样本"可能 score 不再极端
   - test 上"新出现的 hard negative"可能落在模型从未学过的 score region
   - **排序关系整体变得不稳定**，部分 pair 的相对排序翻转 → AUC 下降
4. **BCE 的排序依赖更"平滑"**：BCE 梯度是 `σ(s) − y`，在整个 score range 都有梯度，不会推向极端 → 排序对分布漂移**更 robust**
5. 这也解释了为什么**三合一 dev LogLoss 比 icq-half 高 +0.003**：BPR 让模型的概率分布推得更极端，dev 上 LogLoss 校准度变差 → 暗示 test 上泛化会更差（LogLoss 比 AUC 对分布 shift 更敏感）

### ⚠️ 归因歧义（重要！）

**必须承认：三合一 E4 LB −0.010 的归因存在歧义**，因为三合一是**组合实验**同时改了 3 个变量：
- T16 additive target injection
- T18 CombinedAUCLoss (w=0.3, m=0.5)
- `holdout_ratio=0.05` → **训练数据从 95% 减到 90%**（损失 ~100k 样本）
- `logloss_rebound_patience` 从 1 改到 2

**三合一 vs icq-half 的 LB 差距 −0.00993 无法干净归因到单一变量**。可能的分解：

| 归因候选 | 可能贡献 | 证据强度 |
|---|---|---|
| T18 BPR（排序稳定性） | -0.003 ~ -0.007 | 中（机制合理但未独立验证） |
| Holdout 切分（训练数据 −5%） | -0.002 ~ -0.005 | 中（1.9M × 5% ≈ 100k sparse embedding 更新减少） |
| T16 vs ICQ 架构差异 | ±0.002 | 低（两者都是 target conditioning，强度相当） |
| logloss_patience 放宽 | 0 ~ -0.001 | 低（未到反弹拐点前都一样） |

**结论**：T18 Deprecated 的判决仍然成立（排序稳定性论证 + dev LogLoss 显著抬高），但**贡献幅度** −0.003 ~ −0.007（而非之前暗示的全部 −0.010）。**完整干净归因需要独立 T18-only half-train vs BCE-only half-train 对照**（目前无此数据）。

### ⚠️ 后续 holdout_ratio 的独立嫌疑

若 T16-only full train（当前 run.sh 配置 `--holdout_ratio 0.05`）LB 结果显著低于 icq-half E4 LB 0.84637，则**反向证明 holdout 切分本身有 −0.003 ~ −0.005 的 LB 成本**，独立于 T18。届时需要:
1. 把 ADR-005 T17 `holdout_ratio` 默认值回归到 0.0
2. 重新设计 "holdout as LB proxy" 的切分策略（例如 ADR-005 方案 C：留时间空窗而非切走 train 数据）

**为什么 external anchor（Tencent +0.70~1.26% GMV）在这里不适用**：
- 腾讯论文的 ablation 在**同分布 test** 下做
- 本赛题 test 是**远未来 + 0.023 baseline gap** 的"远外推"场景
- calibration 破坏在远外推场景下被放大，超过 ranking 增益

**Deprecation 影响**：
- 🚫 `--loss_type auc` 不再推荐使用（代码保留但默认 `bce`）
- 🚫 后续所有 `--auc_loss_weight / --auc_margin / --auc_max_pairs` 调参实验全部关闭
- ⚠️ Holdout→LB 经验公式 "LB ≈ holdout − 0.022" **仅在纯 BCE 路径下有效**，BPR 路径下 gap 不可预测
- ✅ 代码本身不回滚（`combined_auc_loss` 纯函数 + trainer 分支保留），便于未来有"calibration-aware ranking loss"出现时快速验证

---

## Context（背景与触发原因，写于 Accepted 时期）

### 触发证据

- **LB 差距仍显著**：当前最佳 `t13tf-full E4` local AUC 0.868495 / LB 0.844956，距社区 SOTA ~0.858 约 **−0.013**。21 天 ddl + 评估配额严格 4/day。
- **常规特征工程已验证饱和**：ADR-003 所列 P0-A/B/C 在 full train 上全部负向（EXP-009），hash2m vs hash500k 几乎持平（EXP-018 修正版）。
- **Loss 层面是唯一未被消化的工业级杠杆**：本 repo `trainer.py` 目前仅支持 `bce` / `focal`，**完全没有 ranking 成分**。而腾讯自己公开的生产数据（arXiv 2403.14144 §5.2）显示在 BCE 基础上加 ranking loss 带来 **GMV +0.70%~+1.26%**；多任务 CTR+CVR 场景下 PWiseR AUC 全面超 BCE（arXiv 2406.02163 Table 2）。
- **与本届赛题的对口性**：
  - 评估指标就是 AUC（不是 LogLoss，不是 GMV）
  - 正样本极稀疏（label_type=2 约 4% / 点击后样本约 10%），BCE 在稀疏正样本下梯度会在负样本上消失，而 pairwise 在负样本上仍有强梯度
  - 单模型红线允许（loss 改造是**同一 forward pass 内**的事，不构成 ensemble）
- **同赛题开源参考**：hojiahao/TAAC2026（UniRec, 41 stars）已将 `CombinedAUCLoss = WeightedBCE + PairwiseBPR` 作为默认 loss，是目前唯一可直接借鉴的同赛题 loss 设计。

### 问题域

- BCE 是 pointwise calibration loss，它的梯度 `∂L/∂s = σ(s) − y` 在负样本（y=0, σ(s) 极小）时趋零，正样本极稀疏场景下**负样本的 learning signal 被严重削弱**。
- PairwiseBPR `-log σ(s_pos - s_neg)` 直接优化 `P(score_pos > score_neg)`，这**正是 AUC 的定义**；对"难负样本"（s_neg 高）的梯度远大于 BCE。
- 但纯 BPR 会**破坏概率校准**（训出来的 sigmoid 输出不再是 P(y=1|x)），因此工业实践都是 `α·BCE + (1-α)·BPR` 的加权。

### 时间压力

- 5/25 初赛 ddl，21 天。
- Loss 改造是**与 T16 / T17 正交**的独立破局候选，可并行推进，不抢 training quota（写代码 + smoke ≤ 3h）。
- 主目标：把第一个 half-train 数据点收进来，validate 假设。

---

## Decision（最终决定做什么）

**在 `baseline/utils.py` 新增 `pairwise_bpr_loss` + `combined_auc_loss`；在 `baseline/trainer.py` 把 `loss_type` 选项扩为 `{bce, focal, auc}`，`auc` 路径等价 `(1-w)·BCE + w·PairwiseBPR`；CLI 新增 `--auc_loss_weight` / `--auc_margin` / `--auc_max_pairs` 三个旋钮；`--loss_type` 默认仍是 `bce`，不改任何已有 job 的行为。**

### 设计细节

#### 1. PairwiseBPR 定义

```
pairs = (s_pos_i - s_neg_j - margin)  ∀ i ∈ pos, j ∈ neg
loss  = -mean(log sigmoid(pairs))
```

- `margin` 是超参；`margin=0` 为 vanilla BPR；`margin=0.5` 为 UniRec 默认（论文 arXiv 2403.14144 §4.3 Table 5 实测 margin ∈ [0.2, 0.8] 收益稳定）。
- **空正 / 空负保护**：若 batch 内仅有 pos 或 neg，返回 `scores.sum() * 0.0`（保持 autograd graph），调用方 `.backward()` 不会报错。
- **P × N 爆炸保护**：`max_pairs` 对 pos 和 neg 各自采样上限（默认 256），内存 O(max_pairs²) = 65k 项 × fp32 ≈ 260KB，单 batch 可忽略。

#### 2. CombinedAUCLoss 聚合

```
total = (1 - auc_weight) * BCE + auc_weight * BPR
```

- `auc_weight ∈ [0, 1]`；超出抛 `ValueError`（早期失败）。
- `auc_weight == 0` 时走**短路分支**直接返回 BCE → 与旧 `loss_type='bce'` 路径**数学恒等**（bit-identical，已本地验证）。
- `sample_weight` 仅作用于 BCE 分量（BPR 本身是 pair-wise，无单样本 weight 语义）；若调用方希望重赋类权重，应继续走 BCE 组件。

#### 3. 与已有 loss 路径的关系

| loss_type | `_binary_loss` 实现 | 备注 |
|---|---|---|
| `bce` | `F.binary_cross_entropy_with_logits(..., reduction='none').mean()` | 默认，行为不变 |
| `focal` | `sigmoid_focal_loss(...)` | 已有路径，不动 |
| `auc`（新） | `combined_auc_loss(...)` 直接返回 scalar | **新增** |

#### 4. 与 `multi_task_loss` 分支的交互

- `multi_task_loss=True` 时，**click head 保持 BCE**（固定，不随 `loss_type` 改），**conversion head 走 `_binary_loss` → 可以被 `auc` 覆盖**。
- 设计理由：click 是 multi_task 里的辅助监督，用 BCE 确保校准；conversion 才是评估目标（AUC），该由 AUC loss 优化。
- 本届赛题数据不含 label_type=0 样本，`multi_task_loss` 已判定不可行（T15 Archived），本分支代码仅为兼容保留。

#### 5. CLI 参数面

```
--loss_type auc                  # trigger
--auc_loss_weight 0.3            # α ∈ [0,1], Tencent paper sweet spot 0.1~0.5
--auc_margin 0.5                 # UniRec default, vanilla BPR is 0.0
--auc_max_pairs 256              # P×N cap
```

全部字段通过 `vars(args)` 自动落入 `train_config.json`；平台默认 infer.py 不消费 loss 字段（推理不走 loss），**无需上传自定义 infer.py**。

---

## Alternatives Considered（其他备选）

- **A: 纯 BPR 替代 BCE**
  - Pros: 最直接优化 AUC
  - Cons: 破坏 probability calibration，LogLoss 会飘；团队已落地 LogLoss rebound guard，会错误触发早停。**不推荐**。

- **B: Triplet loss / Contrastive loss**
  - Pros: ML 文献中 loss 家族最丰富
  - Cons: 需要 hard negative mining，代码复杂度 +3×；赛题时间不够。

- **C: PDAOM (Personalized Differentiable AUC)，美团 2023**
  - Pros: 论文报告 +1.4% click
  - Cons: 需要 user-wise 分组（gAUC），但本赛题评估是 global AUC，不对口。

- **D: 继续迭代 focal loss 参数**
  - Pros: 零代码改动
  - Cons: Focal 本质仍是 pointwise；focal(alpha=0.25, gamma=2) 已是标准值，continued tuning ROI 极低。已实验证伪。

- **E: Label smoothing + BCE**
  - Pros: 工程最简
  - Cons: 对 AUC 基本无收益（label smoothing 是防过信的，不是防 ranking error 的）。

---

## Consequences（后果与副作用）

### 正向

- **与比赛指标 AUC 直接对齐**，比 BCE 的间接代理更有针对性。
- **与 T16（seq-token 侧 target 注入）、T17（dev+holdout valid 重设计）正交**，可叠加。
- **推理时长零影响**：loss 只在训练时用，推理不走。
- **向后兼容保证** —— 不传新 flag 完全等价旧行为，bit-identical 已实测。

### 负向 / 待观察

- **过大的 `auc_weight` 会破坏 LogLoss 校准** → LogLoss rebound guard 可能被早期误触发。首次 half-train 建议 `auc_weight ∈ [0.1, 0.3]` + `logloss_rebound_patience ≥ 2` 放宽。
- **BPR pair 采样引入随机性** → 同 seed 结果仍可复现（用了 `torch.randperm` 在默认 generator），但 CPU/GPU 之间小浮点漂移可能比 pure BCE 大一点点。
- **`auc_max_pairs` 过小** 会让 BPR 梯度方差变大；过大浪费显存 / 计算。默认 256 是保守值，full train 可以试到 512 看是否稳定。
- **valid→LB gap 未必减小**：BPR 直接优化 AUC，可能在 local valid 上涨更快，但 LB 涨幅受 valid→LB covariate shift 限制。需与 T17 holdout 搭车一起监控。

### 不在本 ADR 覆盖范围

- **不动** `trainer.py::_save_step_checkpoint`、**不动** EarlyStopping 策略 —— 仍按 AUC 选 best_model，LogLoss guard 仍生效。
- **不动** ADR-004 (T16) 实现 —— T18 与 T16 是两条正交战线。
- **暂不做 gAUC / per-user ranking loss** —— 评估指标就是 global AUC，不需要 gAUC 对齐。

---

## Verification（验证计划）

### 已验证（S0 Static / S1 Local Smoke · 2026-05-04 晚）

**S0 Static** ✅
- `python3 -m py_compile baseline/{utils,trainer,train,infer,dataset,model}.py` exit 0
- `ReadLints` 无新错误（仅环境 basedpyright 解析 warning）

**S1 单元级数值验证** ✅（8 个检查全过）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | BPR 在 `all pos > neg` 时 loss → 0 | `4.5e-5` ✓ |
| 2 | BPR 在 `all pos < neg` 时 loss ≈ 10（对应 diff=-10） | `10.0000` ✓ |
| 3 | 空正 / 空负 batch → 返回零 grad tensor，不破坏 autograd | `grad=[0,0,0]` ✓ |
| 4 | `combined_auc_loss = (1-w)·BCE + w·BPR` 加权恒等 | Δ=0.0 @ w∈{0,0.3,0.5,1.0} ✓ |
| 5 | `auc_weight=0` 等价纯 BCE（向后兼容） | bit-identical ✓ |
| 6 | 同 seed 可复现 + gradient 正常流动 | Δ=0.0, grad_norm=0.13 ✓ |
| 7 | `auc_weight=1.5` 抛 ValueError | 已抛 ✓ |
| 8 | `sample_weight` 仅影响 BCE 分量 | 0.772 → 0.751 ✓ |

**S1 集成级 smoke（demo_1000, 10 step train + max_train_steps 早停）** ✅

| 配置 | best AUC | best LogLoss | 说明 |
|---|---|---|---|
| `--loss_type bce`（baseline） | 0.8737376434 | 0.3066050708 | reference |
| `--loss_type auc --auc_loss_weight 0.0` | 0.8737376434 | 0.3066050708 | **bit-identical** → 向后兼容验证通过 |
| `--loss_type auc --auc_loss_weight 0.3 --auc_margin 0.5` | 0.8940688308 | 0.2798988521 | 10 步上 +0.020 AUC（样本太少无统计意义，但机制生效） |

- `train_config.json` 正确写入 `loss_type/auc_loss_weight/auc_margin/auc_max_pairs` 四字段
- `best_model/` 正常落盘，目录名含 auc 后缀
- `training_summary.json` 正确记录，`[summary]` 行全出

### 待验证（proxy → half-train → full-train 漏斗）

| 阶段 | 环境 | 期望信号 | 决策阈值 |
|---|---|---|---|
| S1 本地 smoke | demo_1000, 10 step | 无 shape error + 可复现性 + 向后兼容 | 失败回滚 —— **已通过** |
| S2 fast proxy | 128/32 RG + 1500 steps + `--loss_type auc --auc_loss_weight 0.3 --auc_margin 0.5` | dev AUC 相对 t13tf 基线不退超过 0.002 | < −0.002 回滚；≥ −0.002 进 S3 |
| S3 half-train | 2~4 epoch 全 RG + `--holdout_ratio 0.05`（ADR-005 搭车） | E3 dev AUC ≥ t13tf E3 + 0.0003；同时观察 dev vs holdout gap | dev AUC 达标 + gap 未失控 → S4 |
| S4 full train | 10 epoch + batch 128 + logloss_rebound_patience=2 | best dev AUC ≥ 0.870 | 达标 → S5 |
| S5 leaderboard | 1 evaluation slot | LB ≥ 0.847 | 达标 commit ADR-006；否则分析 dev→LB 传导 |

### 回滚触发条件

- S2 proxy dev AUC < t13tf proxy − 0.003 → 先尝试 `auc_weight=0.1, margin=0.0` 再跑一轮，仍负向 → 回滚，默认 loss 保持 `bce`
- S3 half-train E3 dev AUC 反而低于 t13tf E3 → 说明 BPR 与 sparse reinit / LogLoss guard 有负面交互，触发"BPR warmup"子探索（前 K step 只用 BCE）
- S4 full train 曲线 E5+ 出现 LogLoss 大幅反弹(> 0.005)而 AUC 停滞 → `auc_weight` 过高，下调到 0.1 重试

---

## 实际代码改动（2026-05-04 晚落地）

- `baseline/utils.py`（+~130 行）
  - 新增 `pairwise_bpr_loss(logits, targets, margin, max_pairs, generator)` —— pure function
  - 新增 `combined_auc_loss(logits, targets, auc_weight, margin, max_pairs, sample_weight, generator)` —— pure function
  - 两者均为纯函数式实现，便于单元测试；`sigmoid_focal_loss` 不动
- `baseline/trainer.py`（+~25 行）
  - `from utils import ... combined_auc_loss ...`
  - `__init__` 新增 `auc_loss_weight / auc_margin / auc_max_pairs` 三个参数（均有默认值，向后兼容）
  - `_binary_loss` 加 `elif self.loss_type == 'auc'` 分支，直接 return `combined_auc_loss(...)`
  - `[summary]` 的 `key_train_config` 白名单追加三个新字段
  - `logging.info` 启动 banner 追加三字段，方便平台日志 grep
- `baseline/train.py`（+~25 行）
  - `--loss_type` choices 追加 `'auc'`
  - 新增 `--auc_loss_weight`（默认 0.5）/ `--auc_margin`（默认 0.0）/ `--auc_max_pairs`（默认 256）三个 CLI
  - trainer 构造处 pass through
- `baseline/infer.py`: **零改动**（推理不走 loss）
- `baseline/dataset.py`, `baseline/model.py`: **零改动**

**总计 ~180 行代码改动**，全部在 `baseline/` 下，纯函数式 loss 模块可独立单测。

### 平台 evaluation 兼容性注意事项

- 开启 `--loss_type auc` 时，平台默认 `infer.py` 通过 `cfg.get('loss_type', 'bce')` 读取，但**推理路径完全不调用 loss 函数**，所以新字段的存在对推理零影响
- **首次平台 evaluation 仍建议上传自定义 infer.py**（按 team-memory 铁律），但本次 ADR 的代码改动理论上**与默认 infer.py 完全兼容**
- 不涉及 `PCVRHyFormer.__init__` 新 kwarg，不走"kwarg 兼容性"陷阱

---

## 最终提醒

本 ADR 属于**方法学级别改造**，不是参数微调。在 ddl 21 天 + 评估配额 4 次/24h 的约束下，建议路线：
1. **先走 S2 proxy（1 training quota，< 2h）** 确认 loss 在 full data 上不崩盘
2. **S3 half-train 与 T16 full train / T17 holdout 搭车**（zero extra quota）
3. **S4 full train 只在 S3 达标后启动**（10h + 1 quota）
4. **S5 evaluation 仅当 local AUC > t13tf LB 对应阈值 0.870** 时消耗

不要因"看起来很有道理"就直接跳 S4，必须先看 S2 数据。
