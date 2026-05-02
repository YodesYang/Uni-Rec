# Baseline 代码地图（Codebase Map）

> **稳定文档**。来源：原 `knowledge_base.txt §4`。
> 任何 `baseline/` 下的代码改动，**必须**在对应模块下方追加一行 inline 注释（保留行号或 commit）。
> 大改动（新增类 / 改函数签名）→ 同时落盘 ADR（参考 `docs/30-decisions/`）。

---

## 项目结构

```
baseline/
├── dataset.py           # PCVRParquetDataset：流式 IterableDataset
├── model.py             # PCVRHyFormer 主模型（HyFormer 风格）
├── trainer.py           # PCVRHyFormerRankingTrainer
├── train.py             # 训练入口
├── ns_groups.json       # NS 分组配置（GroupNSTokenizer 才需要）
├── infer_schema.py      # 自动生成 schema.json 工具
├── utils.py             # logging / EarlyStopping / set_seed / sigmoid_focal_loss
└── run.sh               # 启动脚本（当前激活 RankMixerNSTokenizer）
```

---

## `dataset.py` — `PCVRParquetDataset`

- 直接读取扁平列 Parquet（不依赖嵌套结构）
- 预分配 numpy buffer，融合 padding loop
- 时间桶 `BUCKET_BOUNDARIES`（65 桶，含 padding=0）
- 序列截断：`seq_max_lens` dict（默认 256，可逐 domain 配）
- 处理脏数据：null / -1 → 0；OOB id 可 clip 或 raise

**新增功能**（2026-04-30 ADR-001 + ADR-002）：
- `_rg_ts_range()` helper + `get_pcvr_data(valid_split_strategy={'filename','time'})` 参数 + sanity log（详见 ADR-001）
- `_convert_batch()` 末尾在 `is_training=True` 分支加 4 个 `_diag_*` 字段（详见 ADR-002）
- 2026-05-02 修复 `time` split 只排序日志、不影响 dataset 实际 RG 列表的问题：`PCVRParquetDataset` 支持显式 `row_groups`，`get_pcvr_data` 直接传入排序后的 train/valid RG 列表。
- 2026-05-01 T11：`get_pcvr_data(limit_train_rgs, limit_valid_rgs)` 支持 fast proxy 数据切片；默认 0，不影响 full train。

---

## `model.py` — `PCVRHyFormer` 主模型

**主要模块**：
- `RotaryEmbedding (RoPE)` + `RoPEMultiheadAttention`
- `SwiGLU`、`CrossAttention`、`TransformerEncoder`、`SwiGLUEncoder`、`LongerEncoder`
- `RankMixerBlock`（Token Mixing + Per-token FFN）
- `GroupNSTokenizer` / `RankMixerNSTokenizer`（两种 NS Tokenization）
- `MultiSeqQueryGenerator`（独立 Query token / 序列）
- `MultiSeqHyFormerBlock`（Seq Evolve + Query Decoding + Query Boosting）

**关键约束**：
- `rank_mixer_mode='full'` 时，`d_model` 必须能整除 `T = num_queries*S + num_ns`

---

## `trainer.py` — `PCVRHyFormerRankingTrainer`

- **Dual optimizer**：
  - Adagrad → 稀疏 Embedding 参数
  - AdamW → 稠密 Dense 参数
- **Loss**：BCE / Focal（`focal_alpha`, `focal_gamma` 可调）
- **高基数特征 cold-restart**：`reinit_sparse_after_epoch` + `reinit_cardinality_threshold`（参考 KuaiShou MultiEpoch: https://arxiv.org/pdf/2305.19531）
- **评估指标**：BinaryAUROC（sklearn）+ binary logloss；NaN 过滤
- **EarlyStopping** + step/epoch 双粒度 evaluate
- **checkpoint 与 schema/ns_groups/train_config 一并落盘**

**新增功能**（2026-04-30 ADR-002）：
- `evaluate()` 收集 `_diag_*` 字段 + `_slice_diagnose()` 方法跑 4 张切片表（详见 ADR-002）
- 2026-05-01 T11：`max_train_steps` 支持达到指定 optimizer step 后立即跑 final validation 并停止；默认 0，不影响 full train。
- 2026-05-02：训练结束 / Early Stop / fast proxy 停止时自动打印 `[summary]` 摘要，并写出 `training_summary.json` 到 ckpt 与 log 目录，便于平台训练完成后快速读取 best AUC / best step / best ckpt。
- 2026-05-02：best checkpoint 目录名追加 local valid AUC（如 `global_step48060.layer=2.head=4.hidden=64.auc=0.865884.best_model`）；仍满足平台 `global_step` 前缀、≤300 字符、合法字符约束。

---

## `train.py` — 训练入口

- **环境变量优先**：`TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_LOG_PATH` / `TRAIN_TF_EVENTS_PATH`
- **默认超参**：`lr=1e-4`，`sparse_lr=0.05`，`patience=5`，`emb_skip_threshold=0`
- `seq_max_lens` 默认 `'seq_a:256,seq_b:256,seq_c:512,seq_d:512'`

**新增 CLI flag**（ADR-001）：
- `--valid_split_strategy {filename,time}`，默认 `filename`（行为不变）

**新增 CLI flag**（T11 fast proxy）：
- `--limit_train_rgs N`：常规切分后只取靠近 valid cut 的最后 N 个 train RG
- `--limit_valid_rgs N`：常规切分后只取前 N 个 valid RG
- `--max_train_steps N`：训练 N 个 optimizer step 后跑一次 final validation 并退出

---

## `ns_groups.json` — NS 分组配置（GroupNSTokenizer 才需要）

- `user_ns_groups` U1..U7 + `item_ns_groups` I1..I4（共 11 组）
- 注释强调：是「示例」，需按自己的特征语义重新分组

---

## `utils.py` 与 `run.sh`

- `utils.py`：`create_logger` / `EarlyStopping` / `set_seed` / `sigmoid_focal_loss`
- `run.sh`：当前激活 RankMixerNSTokenizer，不依赖 `ns_groups.json`，`user_ns_tokens=5`、`item_ns_tokens=2`、`num_queries=2`

---

## 命名约束（重要）

- baseline 内部使用 `seq_a / seq_b / seq_c / seq_d` 这些 logical 名
- 通过 `schema.json` 的 `prefix` 字段映射到原始列名 `domain_a_seq` / `domain_b_seq` / ...
- 用户自行制作 `schema.json` 时务必保证 `prefix` 与 parquet 列名一致
