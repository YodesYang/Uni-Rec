# EXP-006 · T8.5 + P0-A/B 主线平台 fast proxy

> **Date**: 2026-05-02
> **Status**: Completed
> **Related Tasks**: [T8](../20-tasks/active.md), [T10](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Related ADR**: [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md), [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: efbf595 + uncommitted T8.5/P0 platform snapshot
> **Job Name**: `t8p0-proxy-swiglu-vr005-rg64x16-s500`

## Setup

- 数据：`/data_ams/industrial_training_data`
- 配置：`swiglu` + `rankmixer` + `valid_split_strategy=time`
- 目标：验证 full-scale 数据上 T8.5 row-level cutoff + P0-A/B synthetic dense features 是否能通过平台 fast proxy
- 资源：Tencent Angel GPU job

关键参数：

```text
valid_ratio=0.05
limit_train_rgs=64
limit_valid_rgs=16
max_train_steps=500
enable_row_time_cutoff=True
enable_count_features=True
enable_seq_stats_features=True
num_workers=4
seq_encoder_type=swiglu
ns_tokenizer_type=rankmixer
user_ns_tokens=5
item_ns_tokens=2
num_queries=2
emb_skip_threshold=1000000
```

## Result

### Fast proxy summary

| Item | Value |
|---|---:|
| base split | 992 train RG / 52 valid RG |
| proxy RG window | 64 train RG / 16 valid RG |
| pre-cutoff candidate rows | 168,253 |
| row-level cutoff | `1772587083` |
| post-cutoff train rows | 159,839 |
| post-cutoff valid rows | 8,414 |
| RG-level overlap before cutoff | 302,765 sec |
| stop condition | `max_train_steps=500` |
| best / last valid AUC | 0.819652 |
| best / last valid LogLoss | 0.291509 |
| best checkpoint | `global_step500.layer=2.head=4.hidden=64.auc=0.819652.best_model` |
| wall time | ~3m45s train + validation |
| exit code | 0 |

关键日志：

```text
[T8] train ts: [1772254389, 1772588958] valid ts: [1772286193, 1772588967] overlap_sec=302765
[T8.5] row-level timestamp cutoff enabled: cutoff=1772587083
[T8.5] exact row-level target after filtering: candidate_rows=168253, train_rows=159839, valid_rows=8414
Max-step Validation | AUC: 0.8196516724762238, LogLoss: 0.2915090322494507
[summary] best_val_AUC=0.8196516724762238 best_val_logloss=0.2915090322494507 best_epoch=1 best_global_step=500
```

### Proxy comparison

| Experiment | Config | Valid Rows | AUC | LogLoss | Interpretation |
|---|---|---:|---:|---:|---|
| EXP-004 | `time`, no T8.5/P0 | 33,712 | 0.824108 | 0.252679 | original proxy baseline |
| EXP-004 | `filename`, no T8.5/P0 | 32,480 | 0.832785 | 0.250541 | original proxy contrast |
| EXP-006 | `time` + T8.5 + P0-A/B | 8,414 | 0.819652 | 0.291509 | harder row-level future tail; combined change is not yet full-train candidate |

**注意**：EXP-006 的 valid 集定义已经改变，valid rows 从约 33.7k 缩到 8.4k，且更靠近 row-level future tail。因此它不能和 EXP-004 做严格 apples-to-apples 对比；但 AUC 低约 0.0045、LogLoss 高约 0.039，足以说明“不应直接升级 full train”，需要先拆分归因。

### Slice diagnostics

| Slice | Bucket | n | pos_rate | AUC | pred_mean |
|---|---:|---:|---:|---:|---:|
| user_int_nonzero | [2,58] | 2,039 | 0.1564 | 0.8667 | 0.1454 |
| user_int_nonzero | [58,71] | 2,077 | 0.1218 | 0.8182 | 0.1261 |
| user_int_nonzero | [71,85] | 2,097 | 0.1087 | 0.7947 | 0.1214 |
| user_int_nonzero | [85,199] | 2,201 | 0.1190 | 0.7817 | 0.1234 |
| item_int_nonzero | [0,8] | 1,008 | 0.2569 | 0.8892 | 0.2325 |
| item_int_nonzero | [8,10] | 2,915 | 0.1022 | 0.7825 | 0.1099 |
| item_int_nonzero | [10,12] | 1,715 | 0.0915 | 0.7464 | 0.1009 |
| item_int_nonzero | [12,33] | 2,776 | 0.1254 | 0.8246 | 0.1285 |
| seq_total_len | [17,1126] | 2,101 | 0.1318 | 0.8822 | 0.1261 |
| seq_total_len | [1126,1322] | 2,099 | 0.1305 | 0.8074 | 0.1292 |
| seq_total_len | [1322,1536] | 4,214 | 0.1213 | 0.7872 | 0.1301 |
| label_type | 0 | 0 | - | - | - |
| label_type | 1 | 7,352 | 0.0000 | nan | 0.1027 |
| label_type | 2 | 1,062 | 1.0000 | nan | 0.3102 |

## Findings

1. **T8.5 平台链路有效**：平台日志中确认 `enable_row_time_cutoff=True`，并成功生成 row-level future validation；训练 / 验证 / checkpoint / summary 全部完成。
2. **P0-A/B 平台链路有效**：`count_features=True`、`seq_stats_features=True` 在 train 与 valid dataset 都打开，说明 synthetic dense feature 的训练链路可用。
3. **组合 proxy 暂不升级 full train**：相对 EXP-004 的 `time` proxy，AUC 从 0.824108 降到 0.819652，LogLoss 从 0.252679 升到 0.291509；即使考虑 valid 定义变硬，也不应直接烧 10h full train。
4. **P0 信号仍存在**：`user_int_nonzero`、`item_int_nonzero`、`seq_total_len` 分桶 AUC 差仍大，说明这些特征方向仍可能有价值；当前下降更可能来自组合改动、valid 变小变硬、或 500 step underfit 的混合影响。
5. **本 job 不应提交 leaderboard evaluation**：只训练 500 step，且 local proxy 没有超过原 proxy，对线上分没有候选价值。

## Next

先做拆分归因 proxy，再决定 full train：

1. `t85only-proxy-swiglu-vr005-rg64x16-s500`：只开 `--enable_row_time_cutoff`，不加 P0 特征，判断下降是否主要来自 clean future validation。
2. `p0only-proxy-swiglu-vr005-rg64x16-s500`：只开 `--enable_count_features --enable_seq_stats_features`，不开 row cutoff，和 EXP-004 的 `time` proxy 做更接近的 apples-to-apples 对比。
3. 若两条都不差，再跑 `t8p0-proxy-swiglu-vr005-rg128x32-s1500`，用更大窗口和更多 step 降低 8.4k valid 的波动。

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
  --enable_row_time_cutoff \
  --enable_count_features \
  --enable_seq_stats_features \
  --num_workers 4
```

日志来源：用户下载的平台 `train_log.txt`，本地路径 `/Users/80233614/Downloads/train_log.txt`。
