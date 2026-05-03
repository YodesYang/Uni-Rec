# EXP-009 · P0 full train 首轮回流

> **Date**: 2026-05-02 PDT
> **Status**: Completed（全部 P0 方向已收口，路线关闭）
> **Related Tasks**: [T10](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Related ADR**: [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: efbf595 + uncommitted P0 platform snapshot
> **Job Names**: `p0seq-full-swiglu-vr005-bs128`, `p0count-full-swiglu-vr005-bs128`, `p0only-full-swiglu-vr005`, `p0c-full-swiglu-vr005-bs128`, `p0bp0c-full-swiglu-vr005-bs128`, `mnw-full-swiglu-vr005-bs128`

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time` + `valid_ratio=0.05`
- 目标：验证 EXP-008 中 P0-A/P0-B larger proxy 信号是否能转化为完整训练收益
- 资源：Tencent Angel GPU job

共同参数：

```text
valid_ratio=0.05
limit_train_rgs=0
limit_valid_rgs=0
max_train_steps=0
num_workers=4
seq_encoder_type=swiglu
ns_tokenizer_type=rankmixer
user_ns_tokens=5
item_ns_tokens=2
num_queries=2
emb_skip_threshold=1000000
```

## Result

### Current Summary

| Job | Config | Actual batch_size | Status | Train Rows | Valid Rows | Best AUC | Best LogLoss | Best Epoch | Best Step | Current Read |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `swiglu-split_strategy_time-valid_ratio_0.05` | no P0 reference | 256 | Evaluated | - | - | 0.865884 | - | 6 | 48060 | leaderboard AUC 0.844059 |
| `p0seq-full-swiglu-vr005-bs128` | P0-B seq-stats-only | **256** | Completed / Early Stop | 1,990,981 | 108,975 | 0.865128 | 0.225103 | 6 | 48012 | local 未超过 reference |
| `p0count-full-swiglu-vr005-bs128` | P0-A count-only | **128** | Running / epoch 8 seen | 1,990,981 | 108,975 | 0.865044 | 0.225021 | 6 | 95634 | local 未超过 reference |
| `p0only-full-swiglu-vr005` | P0-A + P0-B | **256** | Completed / Early Stop | 1,990,981 | 108,975 | 0.865100 | 0.224833 | 7 | 56014 | local 未超过 reference |

关键日志：

```text
p0seq-full-swiglu-vr005-bs128:
batch_size=256
enable_count_features=False
enable_seq_stats_features=True
Parquet train: 1990981 rows, valid: 108975 rows
[T8] train ts: [1772238909, 1772588958]
     valid ts: [1772247872, 1772588990]
     overlap_sec: 341086

Epoch 1 Validation | AUC: 0.8609510981707817, LogLoss: 0.23072266578674316
Epoch 2 Validation | AUC: 0.8626065939311081, LogLoss: 0.22693900763988495
Epoch 3 Validation | AUC: 0.8647351987544482, LogLoss: 0.22535914182662964
Epoch 4 Validation | AUC: 0.8641141503706087, LogLoss: 0.22588662803173065
Epoch 5 Validation | AUC: 0.8642940753014898, LogLoss: 0.2263154834508896
Epoch 6 Validation | AUC: 0.8651277721803491, LogLoss: 0.22510284185409546
Epoch 7 Validation | AUC: 0.8644780426945876, LogLoss: 0.22544455528259277
Epoch 8 Validation | AUC: 0.8644229036588456, LogLoss: 0.22578351199626923
Epoch 9 Validation | AUC: 0.8639269436103433, LogLoss: 0.22583746910095215
Epoch 10 Validation | AUC: 0.8635956664373247, LogLoss: 0.23013392090797424
Epoch 11 Validation | AUC: 0.8625704625638759, LogLoss: 0.22810472548007965

[summary] stop_reason=early_stop_epoch_11
[summary] best_val_AUC=0.8651277721803491 best_val_logloss=0.22510284185409546 best_epoch=6 best_global_step=48012
[summary] best_checkpoint_dir=.../global_step48012.layer=2.head=4.hidden=64.auc=0.865128.best_model

p0only-full-swiglu-vr005:
batch_size=256
enable_count_features=True
enable_seq_stats_features=True
Parquet train: 1990981 rows, valid: 108975 rows
[T8] train ts: [1772238909, 1772588958]
     valid ts: [1772247872, 1772588990]
     overlap_sec: 341086

Epoch 1 Validation | AUC: 0.86178977833352, LogLoss: 0.22783203423023224
Epoch 2 Validation | AUC: 0.8627991273612003, LogLoss: 0.2279462218284607
Epoch 3 Validation | AUC: 0.8639918063310996, LogLoss: 0.22588273882865906
Epoch 4 Validation | AUC: 0.8644152682070758, LogLoss: 0.22523453831672668
Epoch 5 Validation | AUC: 0.8639065977602801, LogLoss: 0.2259741723537445
Epoch 6 Validation | AUC: 0.8650921875319364, LogLoss: 0.2263605296611786
Epoch 7 Validation | AUC: 0.8651001096039556, LogLoss: 0.22483305633068085
Epoch 8 Validation | AUC: 0.8641540254822002, LogLoss: 0.22565290331840515
Epoch 9 Validation | AUC: 0.8636070084100449, LogLoss: 0.22662557661533356
Epoch 10 Validation | AUC: 0.8618982496737899, LogLoss: 0.22716158628463745
Epoch 11 Validation | AUC: 0.8615926352495165, LogLoss: 0.22783777117729187
Epoch 12 Validation | AUC: 0.8590648935826846, LogLoss: 0.2299295961856842

[summary] stop_reason=early_stop_epoch_12
[summary] best_val_AUC=0.8651001096039556 best_val_logloss=0.22483305633068085 best_epoch=7 best_global_step=56014
[summary] best_checkpoint_dir=.../global_step56014.layer=2.head=4.hidden=64.auc=0.865100.best_model

p0count-full-swiglu-vr005-bs128 (partial log, still running):
batch_size=128
enable_count_features=True
enable_seq_stats_features=False
Parquet train: 1990981 rows, valid: 108975 rows
[T8] train ts: [1772238909, 1772588958]
     valid ts: [1772247872, 1772588990]
     overlap_sec: 341086

Epoch 1 Validation | AUC: 0.8620749617591823, LogLoss: 0.22785428166389465
Epoch 2 Validation | AUC: 0.8640718031593673, LogLoss: 0.22645996510982513
Epoch 3 Validation | AUC: 0.8644226961383404, LogLoss: 0.2256266176700592
Epoch 4 Validation | AUC: 0.8643572150437023, LogLoss: 0.22568213939666748
Epoch 5 Validation | AUC: 0.8643597997441642, LogLoss: 0.226292222738266
Epoch 6 Validation | AUC: 0.8650442153983038, LogLoss: 0.22502146661281586
Epoch 7 Validation | AUC: 0.864575160429822, LogLoss: 0.2251657247543335
Epoch 8 Validation | AUC: 0.8640457993511371, LogLoss: 0.22561076283454895

current best_checkpoint_dir=.../global_step95634.layer=2.head=4.hidden=64.auc=0.865044.best_model
```

### Best-Epoch Slice Diagnostics (`p0seq`, epoch 6)

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [0,58] | 26,375 | 0.1118 | 0.8930 | 0.1145 |
| user_int_nonzero | [58,71] | 27,266 | 0.1039 | 0.8675 | 0.1066 |
| user_int_nonzero | [71,86] | 28,064 | 0.0960 | 0.8474 | 0.1027 |
| user_int_nonzero | [86,283] | 27,270 | 0.0913 | 0.8480 | 0.1004 |
| item_int_nonzero | [0,8] | 15,937 | 0.1554 | 0.9317 | 0.1669 |
| item_int_nonzero | [8,9] | 18,720 | 0.0998 | 0.8405 | 0.1074 |
| item_int_nonzero | [9,12] | 39,088 | 0.0758 | 0.8318 | 0.0789 |
| item_int_nonzero | [12,33] | 35,230 | 0.1038 | 0.8556 | 0.1076 |
| seq_total_len | [0,1114] | 27,210 | 0.0949 | 0.8988 | 0.0978 |
| seq_total_len | [1114,1322] | 27,184 | 0.1040 | 0.8629 | 0.1065 |
| seq_total_len | [1322,1536] | 54,581 | 0.1018 | 0.8475 | 0.1097 |
| label_type | 1 | 98,011 | 0.0000 | nan | 0.0770 |
| label_type | 2 | 10,964 | 1.0000 | nan | 0.3641 |

### Best-Epoch Slice Diagnostics (`p0only`, epoch 7)

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [0,58] | 26,375 | 0.1118 | 0.8947 | 0.1116 |
| user_int_nonzero | [58,71] | 27,266 | 0.1039 | 0.8666 | 0.1030 |
| user_int_nonzero | [71,86] | 28,064 | 0.0960 | 0.8468 | 0.0983 |
| user_int_nonzero | [86,283] | 27,270 | 0.0913 | 0.8472 | 0.0955 |
| item_int_nonzero | [0,8] | 15,937 | 0.1554 | 0.9329 | 0.1632 |
| item_int_nonzero | [8,9] | 18,720 | 0.0998 | 0.8407 | 0.1013 |
| item_int_nonzero | [9,12] | 39,088 | 0.0758 | 0.8308 | 0.0759 |
| item_int_nonzero | [12,33] | 35,230 | 0.1038 | 0.8552 | 0.1038 |
| seq_total_len | [0,1114] | 27,210 | 0.0949 | 0.8984 | 0.0953 |
| seq_total_len | [1114,1322] | 27,184 | 0.1040 | 0.8648 | 0.1025 |
| seq_total_len | [1322,1536] | 54,581 | 0.1018 | 0.8464 | 0.1051 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 98,011 | 0.0000 | nan | 0.0733 |
| label_type | 2 | 10,964 | 1.0000 | nan | 0.3590 |

## Findings

1. **`p0seq` / `p0only` full 均没有转化为 local gain**：`p0seq` best local AUC 0.865128，`p0only` best local AUC 0.865100，均低于 reference `swiglu + time + valid_ratio=0.05` 的 0.865884；按既有 local -> leaderboard gap，当前不应优先 evaluation。
2. **Job Name 与实际 batch size 不一致**：`p0seq-full-swiglu-vr005-bs128` 和 `p0only-full-swiglu-vr005` 的日志都显示 `batch_size=256`，因此这两组不是原计划的 bs128 复刻。后续必须以日志 Args 为准，不以 Job Name 为准。
3. **P0 larger proxy 的排序没有稳定外推到 full train**：EXP-008 中 `p0seq` larger proxy 最强，但 full 训练峰值仍在旧 reference 附近；说明 proxy 适合排除坏方向，但对 full leaderboard 提升的外推需要更保守。
4. **`p0only` 组合仍无叠加收益**：larger proxy 已显示 P0-A + P0-B 不如单分支；full train 也确认组合低于 `p0seq` 且低于 no-P0 reference，短期不继续扩大组合分支。
5. **`p0count` 是真实 bs128，但目前也没有转化为 local gain**：partial log 到 epoch 8，best 仍是 epoch 6 的 0.865044，之后连续两轮回落；若最终不超过 0.8665，当前不建议消耗 evaluation。
6. **full train 每个 epoch 后都会重置 sparse embedding**：日志反复出现 `Re-initialized 96 high-cardinality Embeddings (vocab>0), kept 1`，而 fast proxy 在 `max_train_steps` 下常常不跨 epoch；这可能是 proxy -> full 外推失真的一个高优先级排查点。

## 第二轮回流（2026-05-03）

| Job | 关键配置 | Best AUC | Best Epoch | Early Stop | vs reference | 评估建议 |
|---|---|---|---|---|---|---|
| `p0c-full-swiglu-vr005-bs128` | P0-C history CVR | **0.865097** | E4 @ step 63756 | E9 | -0.000787 | ❌ 不评估 |
| `p0bp0c-full-swiglu-vr005-bs128` | P0-B + P0-C | 未跑完（已停） | — | — | — | ❌ 已停 |
| `mnw-full-swiglu-vr005-bs128` | MNW immature_weight=0.5 | **0.864357** | E11 | E13+ | -0.001527 | ❌ 不评估 |

**P0-C 收敛曲线**（E1→E9 Early Stop）：
```
E1: 0.8624 → E2: 0.8636 → E3: 0.8643 → E4: 0.8651* → E5: 0.8647 → E6: 0.8650 → E7: 0.8642 → E8: 0.8643 → E9: 0.8635 → ES
```

## P0 路线总收口

所有 P0 特征工程全量训练结果：

| 特征 | Best AUC | vs reference | 评估 |
|---|---|---|---|
| P0-B（seq-stats） | 0.865128 | -0.000756 | ❌ |
| P0-A+B（组合） | 0.865100 | -0.000784 | ❌ |
| P0-A（count） | 0.865044 | -0.000840 | ❌ |
| **P0-C（history CVR）** | **0.865097** | -0.000787 | ❌ |
| MNW（延迟反馈降权） | 0.864357 | -0.001527 | ❌ |

**结论（2026-05-03）**：所有 P0 特征全量训练均无法超过 no-P0 reference 0.865884，差值系统性聚集在 -0.0007~-0.0015，当前 HyFormer 架构对这类 dense 特征信号已饱和。**P0 特征工程路线正式关闭，不再消耗任何训练 / 评估额度。**
3. 若 `p0count` 后续跳到 `>=0.8665`，可作为弱候选；`>=0.8685` 才是强 evaluation 候选。
4. sparse embedding no-reinit 已由 [EXP-010](./EXP-010-sparse-reinit-proxy-control-2026-05-02.md) 验证为负向，不启动 no-reinit full train；后续不继续扩 P0 组合分支。

日志来源：用户下载的平台 `train_log.txt`，本地路径 `/Users/80233614/Downloads/train_log.txt`。
