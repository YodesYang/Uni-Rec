# EXP-007 · T8.5-only / P0-only 平台归因 proxy

> **Date**: 2026-05-02
> **Status**: Completed
> **Related Tasks**: [T8](../20-tasks/active.md), [T10](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Related ADR**: [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md), [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: efbf595 + uncommitted T8.5/P0 platform snapshot
> **Job Names**: `p0only-proxy-swiglu-vr005-rg64x16-s500`, `t85only-proxy-swiglu-vr005-rg64x16-s500`

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time`
- 目标：拆分 EXP-006 组合 proxy 下降的来源，分别验证 P0-A/B synthetic dense features 与 T8.5 row-level cutoff
- 资源：Tencent Angel GPU job

共同参数：

```text
valid_ratio=0.05
limit_train_rgs=64
limit_valid_rgs=16
max_train_steps=500
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

| Experiment | Config | Train Rows | Valid Rows | AUC | LogLoss | vs EXP-004 time |
|---|---|---:|---:|---:|---:|---|
| EXP-004 | `time`, no T8.5/P0 | 134,541 | 33,712 | 0.824108 | 0.252679 | baseline |
| EXP-006 | `time` + T8.5 + P0-A/B | 159,839 | 8,414 | 0.819652 | 0.291509 | AUC -0.004456 / LogLoss +0.038830 |
| EXP-007 | `P0-only` | 134,541 | 33,712 | 0.826286 | 0.251651 | AUC +0.002178 / LogLoss -0.001028 |
| EXP-007 | `T8.5-only` | 159,839 | 8,414 | 0.821174 | 0.289528 | AUC -0.002934 / LogLoss +0.036849 |

关键日志：

```text
P0-only:
enable_row_time_cutoff=False
enable_count_features=True
enable_seq_stats_features=True
Max-step Validation | AUC: 0.826285703873776, LogLoss: 0.251651406288147
[summary] best_checkpoint_dir=.../global_step500.layer=2.head=4.hidden=64.auc=0.826286.best_model

T8.5-only:
enable_row_time_cutoff=True
enable_count_features=False
enable_seq_stats_features=False
[T8.5] exact row-level target after filtering: candidate_rows=168253, train_rows=159839, valid_rows=8414
Max-step Validation | AUC: 0.8211743758568328, LogLoss: 0.2895282506942749
[summary] best_checkpoint_dir=.../global_step500.layer=2.head=4.hidden=64.auc=0.821174.best_model
```

### Slice Diagnostics

#### P0-only

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [1,58] | 8,136 | 0.1122 | 0.8639 | 0.1214 |
| user_int_nonzero | [58,71] | 8,395 | 0.1016 | 0.8249 | 0.1059 |
| user_int_nonzero | [71,86] | 8,748 | 0.0996 | 0.8000 | 0.1020 |
| user_int_nonzero | [86,283] | 8,433 | 0.0946 | 0.8104 | 0.0977 |
| item_int_nonzero | [0,8] | 4,902 | 0.1605 | 0.9040 | 0.1714 |
| item_int_nonzero | [8,10] | 11,895 | 0.0833 | 0.7994 | 0.0936 |
| item_int_nonzero | [10,12] | 5,946 | 0.0832 | 0.7728 | 0.0800 |
| item_int_nonzero | [12,33] | 10,969 | 0.1059 | 0.8165 | 0.1060 |
| seq_total_len | [0,1114] | 8,425 | 0.0990 | 0.8734 | 0.1091 |
| seq_total_len | [1114,1320] | 8,408 | 0.1032 | 0.8188 | 0.1054 |
| seq_total_len | [1320,1536] | 16,879 | 0.1027 | 0.8019 | 0.1059 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 30,277 | 0.0000 | nan | 0.0844 |
| label_type | 2 | 3,435 | 1.0000 | nan | 0.3021 |

#### T8.5-only

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [2,58] | 2,039 | 0.1564 | 0.8676 | 0.1447 |
| user_int_nonzero | [58,71] | 2,077 | 0.1218 | 0.8222 | 0.1171 |
| user_int_nonzero | [71,85] | 2,097 | 0.1087 | 0.7959 | 0.1115 |
| user_int_nonzero | [85,199] | 2,201 | 0.1190 | 0.7803 | 0.1102 |
| item_int_nonzero | [0,8] | 1,008 | 0.2569 | 0.8939 | 0.2358 |
| item_int_nonzero | [8,10] | 2,915 | 0.1022 | 0.7884 | 0.0969 |
| item_int_nonzero | [10,12] | 1,715 | 0.0915 | 0.7478 | 0.0945 |
| item_int_nonzero | [12,33] | 2,776 | 0.1254 | 0.8257 | 0.1197 |
| seq_total_len | [17,1126] | 2,101 | 0.1318 | 0.8832 | 0.1240 |
| seq_total_len | [1126,1322] | 2,099 | 0.1305 | 0.8136 | 0.1218 |
| seq_total_len | [1322,1536] | 4,214 | 0.1213 | 0.7852 | 0.1182 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 7,352 | 0.0000 | nan | 0.0932 |
| label_type | 2 | 1,062 | 1.0000 | nan | 0.3098 |

## Findings

1. **P0-A/B 是当前正向主线**：在相同 RG-level `time` proxy valid 上，P0-only 相对 EXP-004 time proxy AUC +0.002178，LogLoss -0.001028，是目前最干净的正向信号。
2. **T8.5 是组合下降的主要来源**：T8.5-only 的 AUC / LogLoss 与 EXP-006 组合结果接近，说明 row-level future tail 显著更难，并且 valid rows 只有 8,414，波动更大。
3. **T8.5 当前不应作为 leaderboard 训练主线**：它适合继续作为 clean future validation / 报告诊断工具，但若目标是近期 leaderboard 提分，当前实现不应直接 full train。
4. **P0 在 T8.5 valid 下没有立刻转正**：EXP-006 比 T8.5-only 低约 0.0015 AUC，说明 P0 对 clean future tail 的收益还需要更大 proxy 或 full train 才能确认；短期不能把 T8.5+P0 作为候选。

## Next

优先推进 P0-only，而不是 T8.5+P0：

1. 启动 `p0only-proxy-swiglu-vr005-rg128x32-s1500`，用更大 RG 窗口和 1,500 step 确认 P0-only 信号是否稳定。
2. 若仍优于 EXP-004 time proxy，再启动 `p0only-full-swiglu-vr005` 完整训练。
3. P0-only 进入 evaluation 时必须上传同版 `infer.py` / `dataset.py` / `model.py`，保证 synthetic dense features 在推理端一致生成。

## Reproduce Recipe

P0-only:

```bash
bash baseline/run.sh \
  --seq_encoder_type swiglu \
  --valid_split_strategy time \
  --valid_ratio 0.05 \
  --limit_train_rgs 64 \
  --limit_valid_rgs 16 \
  --max_train_steps 500 \
  --enable_count_features \
  --enable_seq_stats_features \
  --num_workers 4
```

T8.5-only:

```bash
bash baseline/run.sh \
  --seq_encoder_type swiglu \
  --valid_split_strategy time \
  --valid_ratio 0.05 \
  --limit_train_rgs 64 \
  --limit_valid_rgs 16 \
  --max_train_steps 500 \
  --enable_row_time_cutoff \
  --num_workers 4
```

日志来源：用户下载的平台 `train_log.txt`，本地路径 `/Users/80233614/Downloads/train_log.txt`。
