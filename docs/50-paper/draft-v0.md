# TAAC 2026 Industrial Track · 技术方案草稿（v0）

> **状态**：v0 骨架 · 由 T6 启动，2026-04-30 PDT 21:50
> **赛题**：腾讯广告算法大赛 2026 · Industrial Track（PCVR 预估）
> **当前最佳**：Leaderboard AUC **0.84381**（swiglu encoder） · 距社区 SOTA 0.85809 差 0.0143
> **本草稿覆盖范围**：仅记录已落地实验数据 + 架构选型推理；**所有 "TODO" 标签项需后续实验回填**
> **迭代规则**：覆盖式更新（小改）；做出方向性结论时 → 升 `draft-v1.md` 并把当前版本标记为 archived

---

## 1. Motivation

### 1.1 任务背景

TAAC 2026 Industrial Track 给出一个**单标签二分类点击转化率（PCVR）预估**任务。每条样本：

- **特征**：46 个 user int features（含 sparse id + 部分 list-typed）+ 14 个 item int features + 4 个 dense user features + **4 个域的用户行为序列**（`domain_a/b/c/d_seq_*`，每域含独立 ts_fid + 多个 sparse id features）
- **标签**：raw `label_type ∈ {0, 1, 2}`（曝光 / 点击 / 转化）→ 二值化为 `label = (label_type == 2)` 作为转化标签
- **规模**：训练集 1.9M 行 × 1000 parquet 分片；测试集 523,244 user_id × 200 parquet 分片
- **评估**：单一 AUC 指标，平台 leaderboard 自动打分

### 1.2 三大设计挑战

| # | 挑战 | 落地难度根源 |
|---|---|---|
| C1 | **多域异构序列** | 4 个序列域共存，每域独立 ts_fid + 多个 side-info fid；attention 在 L=256~512 长度下 O(L²) 成本极高 |
| C2 | **高基数 sparse 特征** | 单个 embedding table vocab 接近 1e6，198M 总参数中 99% 是 sparse；常规 AdamW 不可行 |
| C3 | **训练 / 测试分布偏移** | 测试集是真实未来时段，训练集时间分布严重不均（社区线索：03-03 单日占 63%）；本届评估 Local Val AUC ≈ 0.864 而 Leaderboard AUC ≈ 0.844，**系统性 gap ≈ 0.02** |

### 1.3 设计原则（贯穿全文）

1. **同构 backbone + 编码器可插拔**：用 HyFormer 风格的 Query Token 抽取，把序列编码器抽象成一个可替换组件 → 一次实验切换 SwiGLU / Transformer / LONGER 三种 encoder
2. **稀疏-稠密分离优化**：Embedding 走 Adagrad（稀疏更新），dense 走 AdamW；高基数 Embedding 在每 epoch 末做 cold-restart（KuaiShou MultiEpoch trick）
3. **平台兼容性优先**：所有改动严格保留 `train_config.json` + `schema.json` 的 sidecar 写出 → 平台默认 `infer.py` 能动态反序列化模型构建（实测验证，§4.2）
4. **最小变更红线**：所有改动只改现有 baseline 文件（`baseline/dataset.py / model.py / trainer.py`），不新建 test 文件，不引入新依赖

---

## 2. Method

### 2.1 总体架构（PCVRHyFormer）

借鉴 HyFormer（抖音搜索全量部署），骨架是一个**双向 Query Decoding + Query Boosting** 的 Transformer 块栈：

```
[user_int + user_dense + item_int + item_dense] ──▶ NS Tokenizer ──▶ NS-tokens (T_ns × D)
                                                                          │
[seq_a, seq_b, seq_c, seq_d (with ts_fid + side-info)]                   │
                            │                                             │
                            ▼                                             │
                    Per-domain Sequence Tokenizer                         │
                            │                                             │
                            ▼                                             │
              [seq_token, mask, time_bucket_emb] × 4 domains              │
                                       │                                  │
                                       ▼                                  │
                          ┌────────────────────────────┐                  │
                          │ Per-domain Seq Encoder     │ ← 三选一        │
                          │ (SwiGLU / Trans / LONGER)  │                  │
                          └────────────────────────────┘                  │
                                       │                                  │
                                       ▼                                  │
                          MultiSeqHyFormerBlock × 2                       │
                                       │                                  │
                          ┌────────────┴────────────┐                     │
                          ▼                         ▼                     │
                  Query Decoding             Query Boosting               │
                  (CrossAttn Q←K_l, V_l)    (RankMixer-style)             │
                          │                         │                     │
                          └────────────┬────────────┘                     │
                                       ▼                                  ▼
                                  Q tokens (T_q × D)        →  concat with NS-tokens
                                                                          │
                                                                          ▼
                                                                  Final RankMixer
                                                                          │
                                                                          ▼
                                                                  Binary classifier
```

**关键超参**（默认配置，三组 encoder 共享）：

```text
d_model=64, emb_dim=64, num_queries=2, num_hyformer_blocks=2,
num_heads=4, hidden_mult=4, dropout_rate=0.01,
ns_tokenizer_type=rankmixer, user_ns_tokens=5, item_ns_tokens=2,
emb_skip_threshold=1e6, seq_id_threshold=1e4,
seq_max_lens={a:256, b:256, c:512, d:512}
```

### 2.2 NS Tokenizer 设计（取自 RankMixer）

**rankmixer 模式**（不同于 baseline 默认的 `group` 模式）：

1. 每个 user / item 的全部 sparse fid 各自查 embedding，拼成一个长向量
   - user 侧：46 fids × emb_dim 64 = `total_emb_dim 2944`，分成 5 个 NS tokens，`chunk_dim ≈ 589`
   - item 侧：14 fids × 64 = `896`，分成 2 个 NS tokens，`chunk_dim 448`
2. 每个 chunk 经 RankMixer block（参数无关 token mixing + per-token FFN）后输出固定大小的 token
3. **优点**：参数量与 fid 数解耦，比 group mode 更灵活；**约束**：`d_model` 必须整除 token 数

### 2.3 三种序列编码器对比

| Encoder | 思想 | 计算复杂度 | 本届实现 |
|---|---|---|---|
| **SwiGLU** | 完全无 self-attention，纯 SwiGLU FFN 堆叠 | O(L · D²) | `SwiGLUEncoder` |
| **Transformer** | 标准 multi-head self-attention + FFN | O(L² · D + L · D²) | `TransformerEncoder` |
| **LONGER** | Top-K cross-attention 替代 self-attention，Q 长度 ≪ K/V 长度 | O(L_H · L · D + L · D²) | `LongerEncoder`，top_k=50 |

每个 encoder 内部**与上层 HyFormer 块解耦**，靠统一的 `(B, L, D)` 输出接口对接。

### 2.4 训练侧关键技巧

1. **Sparse / Dense 双优化器**：196M sparse params 走 `Adagrad(lr=0.05)`，2.5M dense params 走 `AdamW(lr=1e-4, betas=(0.9,0.98))`
2. **High-cardinality cold-restart**（KuaiShou MultiEpoch）：epoch 末 reinit `vocab > threshold` 的 Embedding，重建 Adagrad state；保留低基数表的优化器历史
3. **Sequence id-feature dropout**：序列内 vocab > 1e4 的 id features 在 forward 时额外应用 `dropout(rate × 2)`，对抗高基数过拟合
4. **Embedding skip**：vocab > 1e6 的 features 不创建 Embedding 表（用 zero vector 占位），节省显存（本届数据未实际触发，所有 fid vocab 均 < 1e6）

---

## 3. Experiments

### 3.1 实验环境

- **Platform**：Tencent Angel · 单 GPU 19 GB · 9 CPU · 55 GB RAM · CUDA 12.6 / PyTorch 2.7.1
- **Data**：1.9M training rows / 1000 parquet shards · 523,244 test users / 200 shards · 4 个序列域（a/b/c/d）
- **Reproducibility**：seed=42 · `valid_ratio=0.1`（按 filename 排序取最后 10% RG）

### 3.2 编码器消融（T4 / EXP-002）

**主结论**：本数据上 SwiGLU 既最快又最高分，与 HyFormer 论文 Table 1（"Transformer ≥ LONGER ≥ SwiGLU"）期望反向。

| Encoder | Local Best Val AUC (Step) | Leaderboard AUC | Inference Time | Avg Epoch Time |
|---|---|---|---|---|
| Transformer | 0.8641 (38050) ⚠️ | 0.834015 ⚠️ | 513.9s | ~2 h |
| **SwiGLU** | **0.8650 (30440)** | **0.84381** | **81.93s** | **~35 min** |
| LONGER | 0.8634 (53270, Early Stop @ E12) | 0.8422 | 203.1s | ~40 min |

> ⚠️ Transformer Leaderboard 用的是 step=7610（Val 0.8617），并非全局最佳 step=38050（Val 0.8641）。Re-publish 后预计 Leaderboard 跃升至 ~0.836，**待回填**。

**收敛轨迹**（Local Val AUC，每 epoch 一次 validation）：

```
SwiGLU:      E1 0.8612 → E2 0.8632 → E3 0.8641 → E4 0.8650* → E5~E8 plateau (0.8645~0.8647)
Transformer: E1 0.8617 → E2 0.8636 → E3 0.8640 → E4 0.8633 → E5 0.8641* → E6 0.8639
LONGER:      E1 0.8589 → E2 0.8612 → E3 0.8622 → E4 0.8625 → E5 0.8624 → E6 0.8630 → E7 0.8634*
             → E8~E11 衰减 → E12 Early Stop
```

> *标注全局最佳 step。

**为什么 SwiGLU 反超 Transformer？**（草稿假说，待 ADR 沉淀）

1. **过拟合假说**：dropout=0.01 偏低，Transformer 的 O(L²) 注意力在长序列下捕捉了 train 上的伪关联；SwiGLU 无 attention，反而正则化更强
2. **序列长度假说**：当 max_len ∈ [256, 512] 时，Transformer 注意力的 noise vs signal 比偏向 noise；缩短序列长度或加 RoPE 可能让 Transformer 重新领先（**TODO**：T4 收口后是否做这组 follow-up 取决于是否决定冲 SOTA）
3. **数据假说**：本届 PCVR 任务上「最近行为」比「跨时段交互」更重要；SwiGLU 的 token-wise 处理刚好抓重点

### 3.3 推理时长 profile（T9 / EXP-002）

| Encoder | Inference Time | Throughput | T9 余量（30min 上限） |
|---|---|---|---|
| Transformer | 513.9s | ~1018 samples/s | 71% |
| LONGER | 203.1s | ~2576 samples/s | 89% |
| SwiGLU | **81.93s** | **~6390 samples/s** | **95%** |

**结论**：编码器选型在推理时长上有 6.3× 差异，但所有方案都远低于 30min 硬上限。即使把 d_model 扩 3× 或加 attention head，SwiGLU 仍是最安全的延迟选项。

### 3.4 训练 / 测试分布偏移（T8 信号）

| Encoder | Local Best Val AUC | Leaderboard AUC | **Gap (Local − Board)** |
|---|---|---|---|
| Transformer | 0.8641 | 0.834015 | +0.030 |
| SwiGLU | 0.8650 | 0.84381 | +0.0212 |
| LONGER | 0.8634 | 0.8422 | +0.0212 |

**解读**：

- 三个模型 gap 一致在 **0.021~0.030** → 不是模型问题，是 **valid 集与 test 集分布不一致**
- 当前 `valid_split_strategy=filename` 取 sorted glob 的最后 10% RG → valid 与 train 时间高度重叠（社区证据：03-03 单日占 train 63%）
- **TODO**（T8 后续）：跑一次 `--valid_split_strategy time` 对比 `overlap_sec`，验证按时间切是否能把 gap 缩到 ≤ 0.015

### 3.5 切片诊断（T10 信号）

**TODO**（T10 后续）：本轮训练 jobs 启动于 T10 探针落地之前，等下一轮 platform job 收 4 张切片表（user_int_nonzero / item_int_nonzero / seq_total_len / label_type）后，回填到本节作为 P0 特征工程方向的依据。

---

## 4. Reproducibility & Engineering Notes

> 替代 HyFormer / OneTrans / RankMixer 论文的 "Online A/B" 节（本届赛事无 A/B）。

### 4.1 Checkpoint 自包含约定

`baseline/trainer.py::_save_step_checkpoint` 把以下三个 sidecar 文件复制到每个 `*.best_model/` 目录：

- `model.pt` — 模型权重
- `schema.json` — 特征布局元数据（fid → vocab_size → embed_dim 映射）
- `train_config.json` — 训练时全部超参快照
- `ns_groups.json`（可选）— NS token 分组配置

**好处**：每个 ckpt 是一个**自包含的 evaluation 单元**，跨主机迁移时无路径依赖。

### 4.2 平台默认 `infer.py` 兼容性（T7 验证）

**结论**：平台 Evaluation 页预置的 `infer.py`（15.6K）**完全动态构建模型**，无需自定义上传。

实测铁证（来自 inference log）：

```text
INFO - Loaded train_config from .../best_model/train_config.json
INFO - Building PCVRHyFormer with cfg: {... seq_encoder_type=swiglu, ns_tokenizer_type=rankmixer, ...}
INFO - Using schema: .../best_model/schema.json
```

三组配置各异的 encoder + rankmixer NS tokenizer 全部 Eval Success，确认默认 infer.py 走 `train_config.json` 反序列化模型构建。**实践含义**：参赛队伍只要保证模型类签名（`PCVRHyFormer.__init__`）稳定，就可以自由切换超参，无需上传任何脚本。

### 4.3 评估配额管理

| 资源类型 | 配额 | 备注 |
|---|---|---|
| 评估提交 / 天 | **4** | 榜单重置补偿前为 3；Failed/Stopped 不计入 |
| Resource quota / 队 | **4** | train + inference 共享，串行 |
| Inference 单次时长 | ≤ **30 min** | 超时强制 kill |
| 上传脚本总大小 | ≤ 100 MB | 本方案上传脚本为 0（用平台默认 infer.py） |

**经验**：把每次评估视为「投资决策」，提交前先做本地 sanity check（schema / train_config 一致性 + ckpt 大小 + best_model 子目录命名规范），避免 4 配额浪费在格式问题上。

### 4.4 已知踩坑

1. **transformer 错 publish ckpt**：本届首批评估时把 epoch 1 ckpt 当作 best_model 发布（漏掉后续 4 个更优 epoch）→ Leaderboard 损失 ~0.002 AUC；下次必须等训练完整收敛再 publish
2. **`PyTorch _foreach_norm` CUDA kernel bug**：在 `clip_grad_norm_` 时遇到，加 `foreach=False` workaround
3. **NumExpr 多线程警告**：平台 384 核被 NumExpr 误识别 → 设 `NUMEXPR_MAX_THREADS=16` 即可，不影响推理

---

## 5. Discussion

### 5.1 当前差距与后续动作

距 SOTA 0.85809 差 +0.0143。可见提升路径（按 ROI 排序）：

| ROI | 动作 | 预期增益 | 代价 |
|---|---|---|---|
| 高 | Re-publish transformer step=38050 | +0.002 | 1 评估配额 / 30 秒 |
| 高 | T8 time-aware split 上线 + 重训 | +0.005~0.015（缩 train/test gap） | 1 训练 + 1 评估 |
| 中 | T10 切片表 → P0 特征工程 | +0.003~0.010 | 取决于发现的弱片段 |
| 中 | SwiGLU + RoPE 或 d_model=128 | +0.002~0.005 | 1 训练 + 1 评估 |
| 低-中 | T5 OneTrans Mixed-Param 复现 | +0.001~0.005 | ~600 行新代码 |

### 5.2 与 prior art 的差异

- vs **HyFormer**：本方案沿用其 Query Decoding + Query Boosting 模式，但不做 Mixed-Param Attention（OneTrans 的核心 idea），保持参数共享以收敛速度优先
- vs **OneTrans**：暂未引入 Mixed Causal Attention 与 KV Caching；T5 决议暂缓
- vs **RankMixer**：完整复用其 NS Tokenizer 思想，但未引入 Sparse-MoE 变体（本届 198M 参数已塞满 19GB GPU，扩容空间有限）
- vs **LONGER**：本届实测 LONGER 不如 SwiGLU，可能与序列长度（max 512）和 top_k 选取（默认 50）有关

### 5.3 Future Work（赛后或复赛）

1. **OneTrans-style Mixed Causal Attention**：本届若进复赛，可复用本仓库 PCVRHyFormer 骨架替换 attention pattern，预计 +0.001~0.005 AUC
2. **跨域序列联合编码**：本届 4 个序列域被独立编码，可探索共享 KV cache 或全局 cross-attention
3. **辅助任务**：`label_type ∈ {0, 1, 2}` 隐含点击 / 转化两个动作，可设 multi-task loss（CTR + CVR）
4. **生成式推荐架构**：HSTU / ULTRA-HSTU 路线（本届未尝试）

---

## 6. Reproducibility Checklist

- [ ] 代码：本仓库 `baseline/` 目录（HEAD commit hash 待 v1 填）
- [x] 数据：平台 `/data_ams/industrial_training_data` + `EVAL_DATA_PATH`
- [x] 训练命令：`bash baseline/run.sh --seq_encoder_type {swiglu, transformer, longer}`
- [x] Seed：`--seed 42`
- [x] Checkpoint 自含 `schema.json` + `train_config.json`，可平台默认 infer.py 直接评估
- [ ] Final hyperparameters：当前以 SwiGLU + 默认共享配置为基线，**TODO** 等 T8 后给出最终生产配置

---

## 草稿元信息

**已用素材源**：
- [`docs/40-experiments/EXP-002-platform-baseline-job.md`](../40-experiments/EXP-002-platform-baseline-job.md) — 三组评估数据 + Local Val AUC 曲线
- [`docs/30-decisions/ADR-001-time-aware-valid-split.md`](../30-decisions/ADR-001-time-aware-valid-split.md) — T8 决策推理
- [`docs/30-decisions/ADR-002-slice-diagnostics.md`](../30-decisions/ADR-002-slice-diagnostics.md) — T10 决策推理
- [`docs/10-research/prior-art.md`](../10-research/prior-art.md) — HyFormer / OneTrans / RankMixer / LONGER
- [`docs/00-context/competition.md`](../00-context/competition.md) — 赛题规则 / 时间表
- [`docs/00-context/platform-angel.md`](../00-context/platform-angel.md) — 平台手册
- [`baseline/{train,trainer,model,dataset}.py`](../../baseline/) — 实现细节

**v1 升级触发条件**（满足任一）：
1. T8 time-aware split 上 leaderboard 数字到位 → 替换 §3.4 的 TODO 段
2. T10 4 张切片表回流 → 写实 §3.5
3. Re-publish transformer step=38050 后 leaderboard 数字到位 → 替换 §3.2 的 ⚠️ 注释
4. 完成 P0 特征工程改动并跑出新 leaderboard → 升 §3.6 新章节
