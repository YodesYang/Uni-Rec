# EXP-002 · 平台首个 baseline job（GPU / batch=256 / 1.9M 行 / 198.7M params）

> **Date**: 2026-04-30（晨训练 + 夜评估）
> **Status**: Completed（训练 + 三组编码器评估全部 Success；T7 / T9 一并解除阻塞）
> **Related Tasks**: T3 / T4 / T7 / T9（均已 archived，本 EXP 即结案证据）
> **Platform**: Tencent Angel
> **Commit**: T8/T10 之前的 baseline tip

---

## Setup

- **数据**：平台真实初赛数据 / 1,900,000 行 / 1,000 个 parquet 分片
- **配置**：默认 `run.sh`（即 `--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 --emb_skip_threshold 1000000 --num_workers 8 --batch_size 256`）
- **资源**：GPU（19GB） / 9 CPU / 55GB RAM（平台标配）

---

## Result

### 关键指标对比（demo vs 平台真实）

| 指标 | demo (1k) | 平台 (1.9M) | 一致？ |
|---|---|---|---|
| user_int fid 数 | 46 | 46 | ✓ |
| item_int fid 数 | 14 | 14 | ✓ |
| domain seq 数 | 4（a/b/c/d） | 4（a/b/c/d） | ✓ |
| 总参数量 | 160.9 M | **198.7 M** | sparse vocab 真值更大（demo 偏小），但**字段集**一致 |
| 训练 AUC | 0.6761~0.6799（CPU 2 epoch） | 待回填（用户报告公榜 0.843371） | — |
| 推理时长 | — | 待 T9 profile 回填 | — |

---

## 编码器消融评估（2026-04-30 21:09~21:21 PDT，全部在 16:00 AOE 重置后跑）

T4 编码器消融三组评估，**ckpt 经平台默认 `infer.py` 全部跑通 Success**。

| Eval ID | 模型 | seq_encoder_type | best_ckpt step | Leaderboard AUC | Inference Time | T9 余量（30min 上限） |
|---|---|---|---|---|---|---|
| 44897 | official baseline | `transformer` | global_step**7610** | **0.834015** | 513.9s (~8.6 min) | 71% |
| 44891 | swiglu_eval | **`swiglu`** | global_step**30440** | **0.84381**（最佳）| **81.93s (~1.4 min)** | **95%** |
| 44851 | longer_eval | **`longer`** | global_step**53270** | 0.8422 | 203.1s (~3.4 min) | 89% |

**共享配置**（`Building PCVRHyFormer with cfg` 日志确认全部一致）：
`d_model=64 / num_heads=4 / num_hyformer_blocks=2 / num_queries=2 / ns_tokenizer_type=rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / emb_skip_threshold=1e6`。

### 训练侧 Local Val AUC 收敛曲线（来自 `Validation | AUC:` 行）

| Encoder | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 | **Best (Step)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer | 0.8617 | 0.8636 | 0.8640 | 0.8633 | **0.8641** | 0.8639 | — | — | — | — | — | — | **0.8641 (38050)** |
| swiglu | 0.8612 | 0.8632 | 0.8641 | **0.8650** | 0.8647 | 0.8645 | 0.8647 | 0.8647 | — | — | — | — | **0.8650 (30440)** |
| longer | 0.8589 | 0.8612 | 0.8622 | 0.8625 | 0.8624 | 0.8630 | **0.8634** | 0.8623 | 0.8619 | 0.8615 | 0.8601 | 0.8585 | **0.8634 (53270)** Early Stop E12 |

### 模型规模 & 训练时长

| Encoder | Total Params | Sparse / Dense | 平均 Epoch 时长 | 收敛 Epoch |
|---|---|---|---|---|
| transformer | 198,682,945 | 196.18 M / 2.50 M | **~2:00 / epoch** | E5（步 38050） |
| swiglu | 198,648,641 | 196.18 M / 2.47 M | **~0:35 / epoch**（最快 3.5×）| E4（步 30440） |
| longer | 198,683,969 | 196.18 M / 2.50 M | **~0:40 / epoch** | E7（步 53270） |

> Sparse 占 99% 总参数（embedding tables），dense 部分 ~2.5M 几乎相同 → 编码器选型对推理时长的差异不来自参数量，而来自**前向 FLOPs**（transformer 的 O(L²) attention 是元凶）。

### ⚠️ 关键发现：Transformer 发布的不是最佳 ckpt

**transformer 训练已到 epoch 6，最佳 step=38050（Val AUC 0.8641），但平台评估 44897 加载的是 step=7610（epoch 1，Val AUC 0.8617）→ Local Val AUC 损失 0.0024**。

可能原因：用户在 epoch 1 完成（约 2 小时）后就触发了 Publish，后续训练完成的更优 ckpt 未替换发布版本。

**改善动作**：重新 Publish step=38050 ckpt 后再提交一次评估，预计 leaderboard AUC 从 0.834 → **~0.836** 区间（按 Local Val 提升 0.0024 的等比例外推 + 少量 train/test 衰减）。占用 1 次评估额度，零代码改动，**ROI 极高**。

**验证 swiglu / longer 已对**：swiglu 30440 = 训练全程最高 Val ✓；longer 53270 = 训练全程最高 Val（之后 Early Stop） ✓。

### ⚠️ Local Val AUC vs Leaderboard AUC 系统性 gap ≈ 0.02

| Encoder | Local Best Val AUC | Leaderboard AUC | **Gap (Local − Board)** |
|---|---|---|---|
| transformer | 0.8617*（已发布的 epoch 1） | 0.834015 | **+0.0277** |
| swiglu | 0.8650 | 0.84381 | +0.0212 |
| longer | 0.8634 | 0.8422 | +0.0212 |

> *transformer 行用的是已发布的 epoch 1 val，非全局最佳。三组真实 gap 在 0.021~0.028 之间。

**解读**：
- 这个 gap 远超随机噪声（AUC 0.02 ≈ 几个百分点的 RankNet pair 翻转）
- 当前 `valid_split_strategy=filename`（取 sorted glob 后最后 10% RG）下，验证集与训练集**时间高度重叠**（baseline_train ts 跨 2/28~3/4，03-03 单日占 63%）→ Val 上等于"用近未来的样本验证"，实际 leaderboard test 集应该是更远未来 → 时间外推损失了 ~0.02 AUC
- **强化 T8 重要性**：下一次平台 job 务必用 `--valid_split_strategy time` + 打印 `overlap_sec`，验证 time 模式能否把 gap 缩到 ≤ 0.015

---

## Findings（关键发现）

1. **fid 集合稳定**：平台 job 启动日志确认 46 user_int + 14 item_int + 4 domain seq 与 [EXP-001](./EXP-001-baseline-cpu-demo.md) demo 完全一致 → `docs/00-context/data-schema.md §3` 字段集**不需要扩展**

2. **`emb_skip_threshold=1e6` 在真实数据上未触发**：demo 里 `seq_a` 跳了 1/8 个 fid（demo 在该 fid 上 vocab 估值偏大），平台真实数据里**未跳**（实际 vocab < 1e6）。说明真实 sparse 参数比 demo 多 1 个 tensor / +37.7M 参数。**不影响正确性**，只是 sparse table 数量不同

3. **真实数据 1.9M 行 / 1000 parquet 分片** → 本地为 demo 写的 `data/demo_rg100/` 重写步骤仅 demo 需要，平台无此问题

4. **平台 Evaluation 页已预置 `infer.py` (15.6K)** → 原 T7「新建 infer.py」计划作废，改为「Publish 后下载默认 infer.py 审查兼容性」

5. **【T7 解除证据】平台默认 `infer.py` 完全动态加载**：三次评估的 inference log 一致打印
   ```text
   INFO - Loaded train_config from .../best_model/train_config.json
   INFO - Building PCVRHyFormer with cfg: {... 三组各异的 seq_encoder_type ...}
   INFO - Using schema: .../best_model/schema.json
   ```
   说明默认 infer.py 走 ckpt 目录里的 `train_config.json` + `schema.json` 反序列化构建模型，**对偏离 baseline 默认的所有自定义配置（rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2 / 三种 encoder）原生兼容**，无需自定义上传 → T7 阻塞条件 #1（动态构建分支）成立 → T7 直接结案

6. **【T9 解除证据】推理时长安全余量**：三组实测 81.93s ~ 513.9s，最长不到 30min 硬上限的 30%，T9 设定的 1500s 安全阈值远未触及 → 即使后续模型规模 × 3 也仍有余量；编码器选型对推理时长影响 6.3×（swiglu vs transformer）

7. **【T4 进展】swiglu 既最快又最高 AUC**：与 HyFormer 论文 Table 1 的 "Transformer ≥ LONGER ≥ SwiGLU (AUC)" 期望**不一致**，本数据上 self-attention 看起来反而是负担。可能原因：当前 `seq_max_lens={a:256,b:256,c:512,d:512}` 下序列足够长 + transformer 过拟合（dropout=0.01 偏低）；后续可单独验证 swiglu + 更大模型/序列的天花板

8. **预测覆盖**：`Saved 523244 predictions` ≈ 完整覆盖测试集 user_id（无 missing），符合官方 leaderboard "全量覆盖" 要求

---

## 后续阻塞与决策（影响后续 EXP）

- ~~**T7 必须先解除阻塞**~~ → 已用三组评估 Success 结案（见 Findings #5），后续直接 publish ckpt 即可上榜
- ~~**T9 推理 30min 硬上限**~~ → 已实测安全（见 Findings #6）
- **T8 + T10 工具仍就绪**：本次平台 job 的训练日志里应有 `[T8] train ts:...` 和 `[diag] slice by ...` 输出，可回看历史训练 log 一次性回填两批信号（详见 [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md) + [ADR-002](../30-decisions/ADR-002-slice-diagnostics.md)）

---

## Reproduce Recipe

平台上直接 `bash baseline/run.sh`（环境变量平台自动注入）。

如需在本地用 demo 重现 198.7M 参数量级（不可能，受限于 demo 数据 vocab） → 不可重现，本 EXP 是平台真实数据**专属**结论。

---

## 待回填字段

- [x] ~~**训练 AUC（首次 platform job 收敛）**~~ — 已替换为 leaderboard AUC + Local Val AUC 收敛表
- [x] ~~**Inference Time**~~ — 见编码器消融评估表
- [x] ~~**Local Val AUC vs Leaderboard AUC 差距**~~ — 已回填，gap ≈ 0.021~0.028，强化 T8 必要性
- [ ] ~~**`overlap_sec`（filename 模式）**~~ → **本轮训练在 T8 探针落地前启动，log 里没有 `[T8]` 行**；待**下一次** platform job（带 T8 instrumentation 的 baseline tip）回填
- [ ] ~~**4 张切片表**~~ → 同上，**本轮训练在 T10 探针落地前启动**；待下一次 job 回填
- [ ] **Re-publish transformer step=38050 后的 leaderboard AUC** — 验证「最佳 ckpt 错配」修复后的实际增益
