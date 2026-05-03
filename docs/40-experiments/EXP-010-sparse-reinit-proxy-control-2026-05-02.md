# EXP-010 · sparse embedding reinit proxy 对照

> **Date**: 2026-05-02 PDT（platform log timestamp: 2026-05-03 UTC）
> **Status**: Completed
> **Related Tasks**: [T11](../20-tasks/active.md)
> **Platform**: Tencent Angel
> **Commit**: efbf595 + uncommitted platform snapshot
> **Job Names**: inferred from args/context: `reinit-proxy-swiglu-vr005-rg128x32-e4`, `noreinit-proxy-swiglu-vr005-rg128x32-e4`

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time` + `valid_ratio=0.05`
- 目标：验证 full train 每个 epoch 后 sparse embedding reinit 是否导致 proxy/full 外推失真
- 资源：Tencent Angel GPU job

共同参数：

```text
limit_train_rgs=128
limit_valid_rgs=32
num_epochs=4
patience=4
max_train_steps=0
batch_size=256
enable_count_features=False
enable_seq_stats_features=False
valid_split_strategy=time
valid_ratio=0.05
```

## Result

### Summary

| Job | `reinit_sparse_after_epoch` | Train Rows | Valid Rows | Best AUC | Best LogLoss | Best Epoch | Best Step | Last AUC | Last LogLoss | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `reinit-proxy-swiglu-vr005-rg128x32-e4` | 1 | 266,116 | 67,135 | **0.845306** | **0.238652** | 3 | 3210 | 0.840142 | 0.245032 | default reinit 明显更好 |
| `noreinit-proxy-swiglu-vr005-rg128x32-e4` | 999 | 266,116 | 67,135 | 0.837562 | 0.244176 | 1 | 1070 | 0.619751 | 0.568140 | no-reinit 快速崩盘 |

关键日志：

```text
default reinit:
reinit_sparse_after_epoch=1
Epoch 1 Validation | AUC: 0.8375593514432013, LogLoss: 0.24417687952518463
Epoch 2 Validation | AUC: 0.8418670094651008, LogLoss: 0.2418241649866104
Epoch 3 Validation | AUC: 0.8453061441695384, LogLoss: 0.23865197598934174
Epoch 4 Validation | AUC: 0.8401421214297154, LogLoss: 0.245032399892807
[summary] best_val_AUC=0.8453061441695384 best_val_logloss=0.23865197598934174 best_epoch=3 best_global_step=3210
[summary] best_checkpoint_dir=.../global_step3210.layer=2.head=4.hidden=64.auc=0.845306.best_model

no-reinit:
reinit_sparse_after_epoch=999
Epoch 1 Validation | AUC: 0.8375621346635991, LogLoss: 0.24417617917060852
Epoch 2 Validation | AUC: 0.8085284490763629, LogLoss: 0.2763201594352722
Epoch 3 Validation | AUC: 0.7288164229146961, LogLoss: 0.41671323776245117
Epoch 4 Validation | AUC: 0.6197505383702009, LogLoss: 0.5681397914886475
[summary] best_val_AUC=0.8375621346635991 best_val_logloss=0.24417617917060852 best_epoch=1 best_global_step=1070
[summary] best_checkpoint_dir=.../global_step1070.layer=2.head=4.hidden=64.auc=0.837562.best_model
```

## Findings

1. **no-reinit 假设被否决**：如果 sparse embedding reinit 是 full train 表现受限的主因，`reinit_sparse_after_epoch=999` 应该至少不差；实际 AUC 从 0.8376 连续崩到 0.6198。
2. **默认 reinit 像强正则**：默认每 epoch 重置 sparse embedding 后，AUC 从 0.8376 提升到 0.8453，明显优于 no-reinit。
3. **proxy/full mismatch 不能简单归因于 reinit**：EXP-008 的 1500-step proxy 与 full train 差异仍然存在，但本对照说明“直接关 reinit”不是解法。
4. **不应启动 no-reinit full train，也不应 evaluation 这两个 proxy**：它们都是 128/32 RG、4 epoch proxy，只用于诊断。

## Next

1. 关闭 no-reinit full train 路线。
2. 如果继续排查 sparse 侧，只能考虑更细的方向：降低 `sparse_lr`、提高 `reinit_cardinality_threshold` 只保留低基数字段、或用 shorter proxy 做小网格；优先级低于寻找更直接的 leaderboard 增益。
3. 当前 evaluation 仍只给完成的 full-train 高置信候选；这两个 proxy 不提交。

日志来源：用户下载的平台 `train_log.txt`，本地路径 `/Users/80233614/Downloads/train_log.txt`。
