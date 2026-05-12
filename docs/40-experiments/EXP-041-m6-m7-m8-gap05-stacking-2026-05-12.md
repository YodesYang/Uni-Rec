# EXP-041 · M6/M7/M8 gap=0.05 叠加 Half-train · 2026-05-12 晚

> **Status**: 🔄 **部分回流** · M8 FaFE 已证伪（⚠️ 方向关闭）· M6/M7 还在跑
> **日期**: 2026-05-12 CEST 12:53 启动 · 5/13 CEST 01:35 M8 回流 · M6/M7 waiting
> **关联**：[EXP-040](EXP-040-m5-m9-gap0-din-seed-sweep-2026-05-12.md)（M5/M9 gap=0 系列）· [EXP-030](../40-experiments/)（gap=0.05 baseline）· [EXP-039](EXP-039-t28-din-integrated-halftrain-2026-05-11.md)（T28 证伪的类似机制）

---

## TL;DR

今晚 5/12 CEST 12:53 启动 4 个 gap=0.05 领地 training job（M5~M8），其中：

- **M5** gap=0 × DIN × seed=3 → LB **0.841812** · 已回流 · 记录在 [EXP-040](EXP-040-m5-m9-gap0-din-seed-sweep-2026-05-12.md)
- **M6** T25+T29 叠加 × seed=42 × gap=0.05 → waiting
- **M7** T25+T29 叠加 × seed=3 × gap=0.05（松 guard） → waiting
- **M8** FaFE × T25 bypass × seed=42 × gap=0.05 → **已回流 · E3 Dev 0.842646 / 方向证伪**

本 EXP 记录 M6/M7/M8，M5 另立 EXP-040。

---

## M8 · FaFE + T25 bypass × gap=0.05 × seed=42

### Config (`M8-fafe-t25bp-gap05-seed42-4ep`)

```bash
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2
--num_queries 2 --ns_groups_json ""
--seq_encoder_type swiglu --seq_hash_vocab 500000
--enable_time_of_day_features
--valid_split_strategy time --valid_ratio 0.05
--enable_row_valid_gap --valid_gap_ratio 0.05    # gap=0.05 领地
--item_conditioned_query --enable_din_interest   # T25 DIN bypass
--enable_fafe                                     # ← M8 新增: FaFE per-fid transform
--seed 42
--batch_size 128 --num_epochs 4
--enable_torch_compile --enable_amp --amp_dtype bfloat16
```

**改动**：`--enable_fafe`（T25/H2 已实装 2026-05-09，per-fid LayerNorm+Linear(emb_dim, emb_dim) · +130k Dense params）

### 训练轨迹

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15168 | 0.838068 | — |
| 2 | 30336 | 0.839165 | — |
| **3** ⭐ | **45504** | **0.842646** | **0.28098** |
| 4 | 60672 | **0.834809** ⚠️ | 0.28340 |

**stop_reason**: `logloss_rebound_epoch_4`（LogLoss 反弹 0.28098→0.28340 触发 guard）
**best_checkpoint**: global_step45504 E3
**Wall-time**: 6h41m（比 T29 的 3h 慢 2× · +130k FaFE params 拖慢）

**🔥 E3→E4 Dev 断崖 −0.00784** —— FaFE 极度容易过拟合

### 严格判决对照

**对标锚点**（同 gap=0.05 领地 × seed=42）:

| 方向 | E3 Dev | E4 Dev | LB | gap | Dev Δvs T25 bypass |
|---|---|---|---|---|---|
| Baseline icq-half-gap05 | 0.842443 | 0.842 | 0.839223 | +0.0032 | −0.00125 |
| **T25 bypass** (T25 B) | **0.843696** | 0.844 | **0.843433** ⭐ | +0.000263 | — |
| M3 T29 vocabtier | 0.841629 | 0.840309 | 0.840603 | +0.001026 | −0.00207 |
| **M8 T25 + FaFE** | **0.842646** | **0.834809** ⚠️ | 未评 | 未知 | **−0.00105** |

### 核心发现：FaFE 方向证伪

#### Finding 1: M8 Dev 比 T25 bypass 低 0.00105 (3.5× rerun 噪声)

在已经 work 的 T25 bypass 上加 FaFE = **净负向**。FaFE 非但没带来 +0.001 LB（Seed 预期），反而让 Dev 降 0.001。

#### Finding 2: E3→E4 Dev 断崖 −0.00784 = 严重过拟合

FaFE 的 +130k per-fid params **放大了每个 fid 单独记忆训练期分布的能力**，在远未来 test 上必然过拟合。
与 T28 integrated 的 "DIN 重复注入放大训练期依赖" 机制**完全同类**。

#### Finding 3: 机制归因 —— "独立性放大过拟合"

FaFE 让每个 fid 有自己的 LayerNorm+Linear transform（理论上 + 表达力）。但：
- 在 1.9M 训练数据规模下，每 fid 增量的 +2.8k params 摊到每个样本只有 ~1.5 梯度 step
- 这些 params 学到的是 **"训练期该 fid 值的细分布"**，test 远未来分布偏移下无迁移价值
- 与 T28 integrated "DIN 每 block 独立注入" / T19 "Per-token FFN" 是同一类过拟合模式：**"独立化"方向在当前数据规模下全面失败**

#### Finding 4: Seed 主楼 "fafe+din+序列 merge" 方案的 LB 证伪可能

Seed 在 "TAAC Industry Track 0.85+ 分享" 里提到方案 "raw embedding + 序列 merge + fafe + din"。但 Seed 本人也说 "问题好多 没招了"。
**M8 实证 FaFE 不 work**，至少在我们这条 baseline 下：
- 我们 T25 bypass 已吃到 +0.004 LB
- 再加 FaFE = 净 −0.001 Dev（未评 LB，但 Dev 机制性负向）
- Seed 方案 "fafe+din" 在他自己 baseline 下可能 work，但**与我们的 baseline 不对口**

---

## ❌ 方向关闭决定

- **FaFE 方向关闭**（M8 单 seed 证 Dev 负向 + E4 断崖 · 不浪费 eval quota）
- 代码保留（`--enable_fafe` CLI + `RankMixerNSTokenizer` FaFE 分支默认 off，backward-compat）
- **不做 FaFE 其他 variant**（不同 emb_dim 内部结构 / FaFE 只用在 user / FaFE 只用在高基数 fid 等）—— 机制性负向细调无意义

## Lessons（归到 team-memory 铁律）

### 新铁律：**"独立化" 方向三连败 → 本数据规模下禁止"per-X 独立参数"类探索**

| 方向 | 机制 | 结果 |
|---|---|---|
| T19 Per-token FFN | 每 token 独立 FFN 参数 (+34% Dense) | ❌ 过拟合必死（EXP-034） |
| T28 DIN integrated (per-block DIN) | 每 block 独立 DIN 参数 (+7% Dense) | ❌ LB 负向（EXP-039） |
| **T25/H2 FaFE** | 每 fid 独立 LayerNorm+Linear (+4.5% Dense) | ❌ Dev 负向 + 过拟合（本 EXP） |

**机制共性**：**"per-X 独立参数" 在 1.9M 样本的有限梯度预算下，必然分到每个独立参数的 signal 不足，被迫记忆训练期分布 → test 远未来分布偏移下必然过拟合**。

**未来禁止探索的"独立化"方向**：
- ❌ Per-domain 独立 loss 权重（如 per seq domain loss weight）
- ❌ Per-fid 独立 sparse embedding size（有意义但无数据支持）
- ❌ Per-sample dynamic routing（如 soft MoE）
- ❌ Per-feature attention head（如 MultiHead per fid）

**允许探索的"共享参数"方向**：
- ✅ Shared architecture 扩容（d_model 64→96 · 但 DECEM 证 scaling up 也无效，间接关闭）
- ✅ Shared mechanism 重新组织（如 T29 fid 拼接顺序 · 零参数）
- ✅ **外部信号注入** 而非内部参数分化（如 T30 时间桶边界 / T31 is-missing / T32 Q_init）← **这是未来路径**

---

## M6 · T25+T29 叠加 × seed=42 × gap=0.05

### Config (`M6-t25bp-t29vt-gap05-seed42-4ep`)

```bash
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2
--num_queries 2 --ns_groups_json "${SCRIPT_DIR}/ns_groups_vocab_tier.json"  # T29-B
--seq_encoder_type swiglu --seq_hash_vocab 500000
--enable_time_of_day_features
--valid_split_strategy time --valid_ratio 0.05
--enable_row_valid_gap --valid_gap_ratio 0.05  # gap=0.05 领地
--item_conditioned_query --enable_din_interest  # T25 DIN bypass
--seed 42
--batch_size 128 --num_epochs 4
--enable_torch_compile --enable_amp --amp_dtype bfloat16
```

### 训练轨迹

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15168 | 0.834019 | — |
| 2 | 30336 | **0.833226** ⚠️ | — |
| **3** ⭐ | **45504** | **0.841291** | **0.27927** |
| 4 | 60672 | 0.840541 | 0.27938 |

**stop_reason**: `logloss_rebound_epoch_4`（E3→E4 LogLoss 0.27927→0.27938 触发 guard）
**best_checkpoint**: global_step45504 E3
**Wall-time**: 7h14m

**⚠️ E2 Dev 0.833226 < E1 0.834019**（非单调收敛 · 异常）

### 判决：M6 叠加失败

| 对比 | E3 Dev | ΔDev |
|---|---|---|
| T25 bypass s42 only (T25 B) | 0.843696 | — |
| T29 vocabtier s42 only (M3) | 0.841629 | −0.00207 |
| **M6 T25+T29 s42** | **0.841291** | **−0.00241 vs T25** · **比 T29 alone 还低 0.000338** |

**叠加 = 净负向**。推测机制：
- T29 vocabtier 重排了 user_int / item_int 的 fid 拼接顺序 → 改变了 NS tokens 的表达
- T25 bypass 的 `DINInterestExtractor` 从 item_ns 计算 target_repr，**target_repr 依赖 NS tokens 的稳定性**
- 当 NS tokens 因 T29 改变时，T25 学到的"哪些 fid 构成有效 target" 失效 → DIN 激活模式偏移 → Dev 反而下降

**M6 不 evaluate**（Dev 明显负向，LB 预期 0.838~0.840 无价值）。

---

## M7 · T25+T29 叠加 × seed=3 × gap=0.05 (松 guard)

### Config (`M7-t25bp-t29vt-gap05-seed3-4ep`)

```bash
# 同 M6，只改：
--seed 3
--logloss_rebound_delta 0.003 --logloss_rebound_patience 2   # 松 guard
```

### 训练轨迹

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15168 | 0.830810 | — |
| 2 | 30336 | 0.838435 | — |
| **3** ⭐ | **45504** | **0.843524** | **0.27642** ⭐ 项目史上最低 gap=0.05 LogLoss |
| 4 | 60672 | 0.843006 | 0.27639 |

**stop_reason**: `max_epochs_completed`（松 guard 未触发，E4 LogLoss 0.27639 < E3 0.27642 还在下降）
**best_checkpoint**: global_step45504 E3
**Wall-time**: 7h23m

### 判决：M7 是今晚最有价值的 ckpt 🎯

**双重亮点**:
1. **Dev 0.843524** = 项目史上 gap=0.05 × seed=3 × DIN 相关配置第 1 或 2 高（排除 M8 FaFE 那个被过拟合拉高的 E3）
2. **LogLoss 0.27642** = **项目史上最低 gap=0.05 LogLoss**
   - baseline 0.27917 · T25 B 0.27822 · T29 vocabtier 0.27813 · T25+T29 s3 (M7) **0.27642**
   - LogLoss −0.0018 vs T25 bypass = 显著 **calibration 改善**

### 核心对比

| 对比 | E3 Dev | LogLoss | Δvs baseline |
|---|---|---|---|
| baseline s42 | 0.842443 | 0.27917 | 0 |
| T25 bypass s42 (T25 B) | 0.843696 | 0.27822 | +0.00125 |
| T29 vocabtier s42 (M3) | 0.841629 | 0.27813 | −0.00081 |
| **M7 T25+T29 s3** | **0.843524** | **0.27642** | **+0.00108** |

**与 T25 B s42 比较**:
- M7 s3 Dev 0.843524 vs T25 B s42 0.843696 = Δ−0.00017（几乎持平）
- M7 s3 LogLoss 0.27642 vs T25 B s42 0.27822 = Δ**−0.00180** ⭐
- **Dev 持平但 LogLoss 更低 = test 泛化可能更好**

### LB 预期（3 角度推理）

1. **Dev→LB gap 推**：T25 B s42 Dev→LB gap +0.000263（黄金三元组） → M7 s3 gap 预期类似 → **LB ≈ 0.843**
2. **seed=3 基线推**：T25 bypass s3 LB 0.843069（历史）+ T29 vocabtier +0.00138 = **0.8444**
3. **LogLoss 改善推**：M7 LogLoss 比 T25 B 低 0.0018 = 更好 calibration → 可能 **LB 比 T25 B (0.843433) 高 0.001+ = 0.844+**

**综合预期 M7 LB: [0.843, 0.845]**，有**小概率破 SOTA 0.84637** 或接近（0.845~0.847）如果 LogLoss 改善真传导到 LB。

---

## 🔥 核心发现：**T25+T29 叠加的 seed 敏感性**

**M6 vs M7 的跨 seed 对比**:

| 维度 | M6 (s42) | M7 (s3) | Δ |
|---|---|---|---|
| E3 Dev | 0.841291 | **0.843524** | **+0.00223** |
| E3 LogLoss | 0.27927 | **0.27642** | −0.00285 |
| E2 Dev | 0.833226 ❗ | 0.838435 | **+0.00521** |
| 训练稳定性 | 非单调 | 平稳上升 | — |

**Δ Dev 0.00223 = 4× rerun 噪声 ± 大 seed 噪声**，远超"纯 seed 波动"可以解释的幅度。

### 机制假说

- **T29 vocabtier 改变 NS token fid 顺序 → 改变 target_repr 的语义**
- seed=42 在原 fid 顺序下已经训到 "T25 bypass 最优激活模式"，T29 重排**破坏了学到的对应关系**
- seed=3 的随机初始化 + 训练路径**刚好适应新的 fid 顺序**（或起始得到更兼容的激活模式）
- 类似 M5 seed=3 × gap=0 × DIN 反向情况（seed=3 × 新配置有时好有时差）

### 新铁律候选

**"方向叠加 seed 敏感性" 铁律**:
- 两个单独 work 的方向叠加**不是总 work**
- 叠加效应跨 seed 可能正负翻转（M6 s42 −0.002 / M7 s3 +0.001）
- **判决**: 未来方向叠加实验 **必须双 seed 验证**（seed=42 + seed=3 各一），不能单 seed 下判决
- **若两 seed 方向一致正向** → 真叠加 work
- **若两 seed 一正一负** → 叠加机制脆弱，谨慎使用

---

## ❌ 方向结论

- **M6 叠加失败 · s42 Dev 明显负向**，**不 evaluate**
- **M7 叠加成功 · s3 Dev + LogLoss 都改善**，**必须 evaluate**（过期风险最大）
- **T25+T29 叠加方向部分 work**（1/2 seed 正向），需要更多实验验证稳定性

## ⚠️ Ckpt 过期时间表（严格追踪）

| Job | 完成时间 (CEST) | 24h 过期时间 | 最晚 publish 时间 |
|---|---|---|---|
| **M7** | 5/12 23:15 | **5/13 23:15** | 5/13 ~20:00（留 3h buffer）|
| M8 FaFE (不评) | 5/13 01:35 | 5/14 01:35 | N/A |
| M6 (不评) | 5/13 02:05 | 5/14 02:05 | N/A |
| M9 (等完成) | ~5/13 04:00 | ~5/14 04:00 | 5/13 ~20:00 |

**M7 必须在明早 CEST 06:00 quota 重置后立刻 evaluate**（与 EXP-036 A 过期丢失教训一致）。

---

## 下一步（等 M6/M7 回流）

### 情景 A：M6 或 M7 LB ≥ 0.846
**T25+T29 叠加 work** · 证明 "小增益叠加 > 单方向深挖" 战略有效。起 M10 = T25+T29+**时间桶 T30** 三重叠加冲 0.847~0.849。

### 情景 B：M6/M7 LB ∈ [0.843, 0.846]
**T25+T29 叠加微弱正向** · 保留 M6/M7 为 gap=0.05 seed 上沿锚点。继续 T30/T31/T32 各自单变量验证。

### 情景 C：M6/M7 LB < 0.842
**T25+T29 叠加未达预期** · FaFE + T28 + T25+T29 同批全负向 → 战略复盘，怀疑 gap=0.05 领地已近 ceiling。

### 并行 M9 情景

M9 (T25 bypass × gap=0 × seed=42) 明早 CEST ~04:00 回流：
- 如果 M9 LB ≥ 0.846 → gap=0 × DIN 仍是上沿候选
- 如果 M9 LB < 0.844 → gap=0 × DIN 方向**彻底关闭**

---

## 附录：eval 决策逻辑

今晚 M6/M7/M8 的 eval 优先级（按 ROI 排序）:

1. **M8 FaFE → 不评**（Dev 负向明确，eval 浪费 quota）
2. **M6 T25+T29 s42 → 必评**（Dev 若 ≥ 0.844 就立即 eval）
3. **M7 T25+T29 s3 → 视 M6 结果**：若 M6 LB ≥ 0.845 → 评 M7 交叉验证 seed；若 M6 LB ≤ 0.842 → 不评 M7

明早 CEST 06:00 quota 重置 4，M9 加入后 4 候选配 4 quota 够用。
