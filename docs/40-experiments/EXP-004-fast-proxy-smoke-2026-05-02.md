# EXP-004 · fast proxy smoke jobs 验证 T8/T10/T11 链路

> **Date**: 2026-05-02
> **Status**: Completed
> **Related Tasks**: [T8](../20-tasks/active.md), [T10](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Related ADR**: [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md), [ADR-002](../30-decisions/ADR-002-slice-diagnostics.md), [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: latest platform snapshot containing `row_groups` time split fix + T11 fast proxy switches

---

## Setup

- 数据：`/data_ams/industrial_training_data`
- 配置：`swiglu` + `rankmixer`，分别跑 `valid_split_strategy=time` 与 `filename` proxy 对照
- 资源：Tencent Angel GPU job
- 任务性质：fast proxy smoke，只验证代码链路 / split 诊断 / 切片诊断，不作为 leaderboard 候选

关键参数：

```text
batch_size=256
num_workers=4
valid_ratio=0.05
limit_train_rgs=64
limit_valid_rgs=16
max_train_steps=500
seq_encoder_type=swiglu
num_queries=2
ns_tokenizer_type=rankmixer
user_ns_tokens=5
item_ns_tokens=2
```

## Result

### Time vs filename proxy 对照

| Split | train RG / rows | valid RG / rows | overlap_sec | AUC | LogLoss | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `time` | 64 / 134,541 | 16 / 33,712 | 302,765 | 0.824108 | 0.252679 | RG-level time sort 仍有 row-level 时间重叠 |
| `filename` | 64 / 130,922 | 16 / 32,480 | 304,064 | 0.832785 | 0.250541 | overlap 与 time 基本同量级，proxy AUC 略高 |

**解读**：在 64/16 RG proxy 规模下，`time` 没有显著降低 `overlap_sec`，反而 AUC 略低。该现象进一步说明当前 RG 粒度切分不足以构造 clean future validation，需要等待 full job 复核；若 full job 同向，应启动 T8.5 row-level timestamp cutoff。

### Fast proxy 是否生效

| 项 | 结果 |
|---|---|
| total RG | 1044 |
| base split | `time`: 992 train / 52 valid; `filename`: 992 train / 52 valid |
| proxy split | 64 train / 16 valid |
| proxy rows | `time`: 134,541 train / 33,712 valid; `filename`: 130,922 train / 32,480 valid |
| stop condition | reached `max_train_steps=500` |
| final AUC | `time`: 0.824108; `filename`: 0.832785 |
| final LogLoss | `time`: 0.252679; `filename`: 0.250541 |
| checkpoint | `global_step500.layer=2.head=4.hidden=64.best_model/model.pt` |
| exit code | 0 |

### T8 时间切 sanity

```text
[T8] valid_split_strategy=time: sorted 1044 RGs by max(timestamp) ASC
[T8] train ts: [1772254389, 1772588958]
     valid ts: [1772286193, 1772588967]
     overlap_sec (train_max - valid_min, <=0 means clean split): 302765

[T8] valid_split_strategy=filename
[T8] train ts: [1772292678, 1772588920]
     valid ts: [1772284856, 1772588939]
     overlap_sec (train_max - valid_min, <=0 means clean split): 304064
```

**结论**：RG-level `time` / `filename` split 在真实平台数据上都有明显时间重叠，且 proxy 对照中 `time` 只比 `filename` 少 1,299 秒 overlap。即使按 Row Group 的 `max(timestamp)` 排序，单个 RG 内部时间跨度仍会导致 train/valid row-level overlap。若 full job 复现该现象，应启动 T8.5 row-level timestamp cutoff。

### T10 切片诊断

#### `time` proxy

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [1,58] | 8,136 | 0.1122 | 0.8615 | 0.1281 |
| user_int_nonzero | [58,71] | 8,395 | 0.1016 | 0.8205 | 0.1150 |
| user_int_nonzero | [71,86] | 8,748 | 0.0996 | 0.7985 | 0.1121 |
| user_int_nonzero | [86,283] | 8,433 | 0.0946 | 0.8107 | 0.1095 |
| item_int_nonzero | [0,8] | 4,902 | 0.1605 | 0.9029 | 0.1750 |
| item_int_nonzero | [8,10] | 11,895 | 0.0833 | 0.7963 | 0.1042 |
| item_int_nonzero | [10,12] | 5,946 | 0.0832 | 0.7730 | 0.0906 |
| item_int_nonzero | [12,33] | 10,969 | 0.1059 | 0.8135 | 0.1162 |
| seq_total_len | [0,1114] | 8,425 | 0.0990 | 0.8707 | 0.1143 |
| seq_total_len | [1114,1320] | 8,408 | 0.1032 | 0.8149 | 0.1139 |
| seq_total_len | [1320,1536] | 16,879 | 0.1027 | 0.8017 | 0.1179 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 30,277 | 0.0000 | nan | 0.0934 |
| label_type | 2 | 3,435 | 1.0000 | nan | 0.3156 |

#### `filename` proxy

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [1,58] | 7,880 | 0.1189 | 0.8585 | 0.1141 |
| user_int_nonzero | [58,71] | 8,049 | 0.1019 | 0.8247 | 0.1021 |
| user_int_nonzero | [71,86] | 8,351 | 0.0964 | 0.8193 | 0.1004 |
| user_int_nonzero | [86,311] | 8,200 | 0.0943 | 0.8250 | 0.0973 |
| item_int_nonzero | [0,8] | 4,735 | 0.1569 | 0.9195 | 0.1588 |
| item_int_nonzero | [8,9] | 5,749 | 0.1047 | 0.8014 | 0.1030 |
| item_int_nonzero | [9,12] | 11,607 | 0.0746 | 0.7765 | 0.0805 |
| item_int_nonzero | [12,33] | 10,389 | 0.1082 | 0.8289 | 0.1038 |
| seq_total_len | [0,1119] | 8,087 | 0.0958 | 0.8667 | 0.0991 |
| seq_total_len | [1119,1322] | 8,151 | 0.1077 | 0.8280 | 0.1060 |
| seq_total_len | [1322,1536] | 16,242 | 0.1036 | 0.8160 | 0.1042 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 29,145 | 0.0000 | nan | 0.0812 |
| label_type | 2 | 3,335 | 1.0000 | nan | 0.2971 |

## Findings

1. **T11 fast proxy 链路有效**：`limit_train_rgs`、`limit_valid_rgs`、`max_train_steps` 都按预期生效，4 分半内完成 train + final validation + best checkpoint 保存。
2. **T8.5 被强烈触发，但需等 full job 复核**：proxy 数据上 `time overlap_sec=302765`，`filename overlap_sec=304064`，说明 RG-level split 无法保证 row-level clean split。
3. **P0-A 缺失计数 dense feature 被强烈触发**：`user_int_nonzero` / `item_int_nonzero` 桶间 AUC 差远超 0.02，尤其 item 侧从 0.9029 到 0.7730。
4. **P0-B 序列活跃度 dense feature 被强烈触发**：`seq_total_len` 从短序列桶 0.8707 降到长序列桶 0.8017。
5. **本 job 不应提交 leaderboard evaluation**：只训练 500 step 且只使用 64/16 RG，是 smoke/proxy 证据，不是线上候选模型。

## Next

- 等 `swiglu-valid_split_strategy_time` 与 `swiglu-split_strategy_time-valid_ratio_0.05` full jobs 完成后，复核 `[T8] overlap_sec` 与 `[diag]` 是否同向。
- 若 full job 仍 `overlap_sec > 0`，启动 T8.5 row-level timestamp cutoff。
- 在等待 evaluation 额度期间，可以先写 P0-A/P0-B feature flags，并先走 fast proxy，不消耗 evaluation quota。

## Reproduce Recipe

平台命令：

```bash
bash baseline/run.sh \
  --seq_encoder_type swiglu \
  --valid_split_strategy time \
  --valid_ratio 0.05 \
  --limit_train_rgs 64 \
  --limit_valid_rgs 16 \
  --max_train_steps 500 \
  --num_workers 4
```

filename proxy 对照：

```bash
bash baseline/run.sh \
  --seq_encoder_type swiglu \
  --valid_split_strategy filename \
  --valid_ratio 0.05 \
  --limit_train_rgs 64 \
  --limit_valid_rgs 16 \
  --max_train_steps 500 \
  --num_workers 4
```

日志来源：用户下载的平台 `train.log`，本地路径 `/Users/80233614/Downloads/train.log`。
