# EXP-008 · P0 larger proxy 回流

> **Date**: 2026-05-02
> **Status**: Completed
> **Related Tasks**: [T10](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Related ADR**: [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: efbf595 + uncommitted P0 platform snapshot
> **Job Names**: `p0only-proxy-swiglu-vr005-rg128x32-s1500`, `p0seq-proxy-swiglu-vr005-rg128x32-s1500`, `p0count-proxy-swiglu-vr005-rg128x32-s1500`

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time`
- 目标：用更大 proxy 窗口验证 EXP-007 中 P0-only 正向信号，并拆分 P0-A count features 与 P0-B sequence stats features
- 资源：Tencent Angel GPU job

共同参数：

```text
valid_ratio=0.05
limit_train_rgs=128
limit_valid_rgs=32
max_train_steps=1500
num_workers=4
seq_encoder_type=swiglu
ns_tokenizer_type=rankmixer
user_ns_tokens=5
item_ns_tokens=2
num_queries=2
emb_skip_threshold=1000000
```

## Result

### Summary

| Experiment | Config | batch_size | Train Rows | Valid Rows | Best AUC | Best LogLoss | Best Step | Last AUC | Last LogLoss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-004 | `time`, no P0, 64x16, s500 | 256 | 134,541 | 33,712 | 0.824108 | 0.252679 | 500 | 0.824108 | 0.252679 |
| EXP-007 | P0-only, 64x16, s500 | 256 | 134,541 | 33,712 | 0.826286 | 0.251651 | 500 | 0.826286 | 0.251651 |
| EXP-008 | P0-only, 128x32, s1500 | 256 | 266,116 | 67,135 | 0.831841 | 0.244238 | 1,070 | 0.831411 | 0.245546 |
| EXP-008 | P0-B seq-stats-only, 128x32, s1500 | 128 | 266,116 | 67,135 | 0.836734 | 0.242303 | 1,500 | 0.836734 | 0.242303 |
| EXP-008 | P0-A count-only, 128x32, s1500 | 128 | 266,116 | 67,135 | 0.836497 | 0.243201 | 1,500 | 0.836497 | 0.243201 |

关键日志：

```text
p0only-proxy-swiglu-vr005-rg128x32-s1500:
enable_count_features=True
enable_seq_stats_features=True
Epoch 1 Validation | AUC: 0.8318411397608052, LogLoss: 0.24423840641975403
Max-step Validation | AUC: 0.8314110468344668, LogLoss: 0.2455458790063858
[summary] best_val_AUC=0.8318411397608052 best_val_logloss=0.24423840641975403 best_global_step=1070

p0seq-proxy-swiglu-vr005-rg128x32-s1500:
enable_count_features=False
enable_seq_stats_features=True
Max-step Validation | AUC: 0.8367341875772585, LogLoss: 0.24230262637138367
[summary] best_val_AUC=0.8367341875772585 best_val_logloss=0.24230262637138367 best_global_step=1500

p0count-proxy-swiglu-vr005-rg128x32-s1500:
enable_count_features=True
enable_seq_stats_features=False
Max-step Validation | AUC: 0.8364966759396285, LogLoss: 0.24320079386234283
[summary] best_val_AUC=0.8364966759396285 best_val_logloss=0.24320079386234283 best_global_step=1500
```

### Slice Diagnostics

#### P0-only, larger proxy

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| label_type | 1 | 60,341 | 0.0000 | nan | 0.0671 |
| label_type | 2 | 6,794 | 1.0000 | nan | 0.3055 |

#### P0-B seq-stats-only, larger proxy

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [0,58] | 16,242 | 0.1141 | 0.8676 | 0.1087 |
| user_int_nonzero | [58,71] | 16,817 | 0.1038 | 0.8383 | 0.0966 |
| user_int_nonzero | [71,85] | 16,357 | 0.0970 | 0.8164 | 0.0912 |
| user_int_nonzero | [85,283] | 17,719 | 0.0908 | 0.8178 | 0.0914 |
| item_int_nonzero | [0,8] | 9,787 | 0.1578 | 0.9191 | 0.1557 |
| item_int_nonzero | [8,10] | 23,718 | 0.0846 | 0.8107 | 0.0810 |
| item_int_nonzero | [10,12] | 11,959 | 0.0809 | 0.7763 | 0.0724 |
| item_int_nonzero | [12,33] | 21,671 | 0.1050 | 0.8255 | 0.1011 |
| seq_total_len | [0,1114] | 16,774 | 0.0956 | 0.8769 | 0.0934 |
| seq_total_len | [1114,1321] | 16,770 | 0.1057 | 0.8306 | 0.0980 |
| seq_total_len | [1321,1536] | 33,591 | 0.1018 | 0.8159 | 0.0980 |
| label_type | 1 | 60,341 | 0.0000 | nan | 0.0726 |
| label_type | 2 | 6,794 | 1.0000 | nan | 0.3120 |

#### P0-A count-only, larger proxy

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [0,58] | 16,242 | 0.1141 | 0.8689 | 0.1189 |
| user_int_nonzero | [58,71] | 16,817 | 0.1038 | 0.8360 | 0.1087 |
| user_int_nonzero | [71,85] | 16,357 | 0.0970 | 0.8165 | 0.1039 |
| user_int_nonzero | [85,283] | 17,719 | 0.0908 | 0.8176 | 0.1045 |
| item_int_nonzero | [0,8] | 9,787 | 0.1578 | 0.9205 | 0.1681 |
| item_int_nonzero | [8,10] | 23,718 | 0.0846 | 0.8088 | 0.0950 |
| item_int_nonzero | [10,12] | 11,959 | 0.0809 | 0.7739 | 0.0826 |
| item_int_nonzero | [12,33] | 21,671 | 0.1050 | 0.8266 | 0.1118 |
| seq_total_len | [0,1114] | 16,774 | 0.0956 | 0.8778 | 0.1008 |
| seq_total_len | [1114,1321] | 16,770 | 0.1057 | 0.8290 | 0.1120 |
| seq_total_len | [1321,1536] | 33,591 | 0.1018 | 0.8159 | 0.1114 |
| label_type | 1 | 60,341 | 0.0000 | nan | 0.0843 |
| label_type | 2 | 6,794 | 1.0000 | nan | 0.3273 |

## Findings

1. **P0 larger proxy 信号继续增强**：P0-only 从 EXP-007 的 0.826286 提升到 0.831841，说明 64x16/s500 的正向不是偶然。
2. **P0-B sequence stats 是目前最强单项**：`p0seq` AUC 0.836734 / LogLoss 0.242303，略高于 `p0count`，且明显优于 P0-only 的 best AUC 0.831841 / LogLoss 0.244238。
3. **P0-A count feature 不是负向，但组合没有叠加收益**：`p0count` AUC 0.836497 / LogLoss 0.243201，和 `p0seq` 很接近；P0-only 反而低，可能是 count 与 seq stats 存在冗余 / 校准相互影响，也可能受 batch size 与 epoch boundary 差异影响。
4. **暂不 evaluation proxy checkpoint**：这些仍是 1,500-step proxy，不是 leaderboard 候选；但 `p0seq` 已足够支持启动 full train。

## Next

1. 立即启动 `p0seq-full-swiglu-vr005-bs128` 完整训练，优先级最高。
2. 若还有一个 full-train 资源，可并行启动 `p0count-full-swiglu-vr005-bs128`，作为与 P0-B 接近的独立候选。
3. 暂不启动 P0-only full；如果要公平复核组合，应先跑 `p0only-proxy-swiglu-vr005-rg128x32-s1500-bs128`，用同 batch size 对比。

## Reproduce Recipe

P0-B seq-stats-only larger proxy:

```bash
bash baseline/run.sh \
  --seq_encoder_type swiglu \
  --valid_split_strategy time \
  --valid_ratio 0.05 \
  --limit_train_rgs 128 \
  --limit_valid_rgs 32 \
  --max_train_steps 1500 \
  --batch_size 128 \
  --enable_seq_stats_features \
  --num_workers 4
```

日志来源：用户下载的平台 `train_log.txt`，本地路径 `/Users/80233614/Downloads/train_log.txt`。
