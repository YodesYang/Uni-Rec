# EXP-005 · T8.5 + P0-A/B 主线本地 smoke

> **Date**: 2026-05-02
> **Status**: Completed
> **Related Tasks**: T8 / T10 / T11
> **Related ADR**: [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md), [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: 本地 macOS / CPU
> **Commit**: 待提交

## Setup

- 数据：`data/demo_rg100/`（1k demo，10 个 row groups）
- 目标：只验证 shape / cutoff / sidecar / inference contract，不用 AUC 做路线判断
- 配置：`swiglu` + `--enable_row_time_cutoff` + `--enable_count_features` + `--enable_seq_stats_features`，`max_train_steps=2`

训练命令：

```bash
TRAIN_DATA_PATH=data/demo_rg100 \
TRAIN_CKPT_PATH=runs/mainline_smoke/ckpt \
TRAIN_LOG_PATH=runs/mainline_smoke/log \
TRAIN_TF_EVENTS_PATH=runs/mainline_smoke/tb \
/opt/anaconda3/envs/tencent-unirec/bin/python -u baseline/train.py \
  --ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 \
  --num_queries 2 --ns_groups_json "" --emb_skip_threshold 1000000 \
  --batch_size 32 --num_epochs 1 --num_workers 0 \
  --valid_ratio 0.2 --limit_train_rgs 4 --limit_valid_rgs 2 \
  --max_train_steps 2 --valid_split_strategy time \
  --enable_row_time_cutoff \
  --enable_count_features \
  --enable_seq_stats_features \
  --seq_encoder_type swiglu \
  --seq_max_lens seq_a:32,seq_b:32,seq_c:32,seq_d:32 \
  --device cpu
```

自定义推理命令：

```bash
MODEL_OUTPUT_PATH=runs/mainline_smoke/ckpt \
EVAL_DATA_PATH=data/demo_rg100 \
EVAL_RESULT_PATH=runs/mainline_smoke/eval \
EVAL_NUM_WORKERS=0 \
/opt/anaconda3/envs/tencent-unirec/bin/python -u baseline/infer.py
```

## Result

| Check | Result |
|---|---|
| `py_compile` | PASS: `dataset.py`, `train.py`, `trainer.py`, `infer.py` |
| RG-level split | `[T8] overlap_sec=765`（demo 数据本身重叠） |
| Row-level cutoff | `cutoff=1772725451`, `candidate_rows=600`, `train_rows=480`, `valid_rows=120` |
| Synthetic dense dim | inference rebuilds `user_dense_dim=765`（原 dense + count 2 + seq stats 8） |
| Train smoke | PASS, checkpoint `global_step2.layer=2.head=4.hidden=64.auc=0.660317.best_model` |
| Summary | PASS, `training_summary.json` 写入 ckpt 与 log 目录 |
| Infer smoke | PASS, `runs/mainline_smoke/eval/predictions.json` 含 1000 条 predictions |

## Findings

1. **T8.5 行级 cutoff 生效**：RG 级仍有 overlap，但 dataset 实际通过 `timestamp < cutoff` / `>= cutoff` 过滤出 clean row-level split。
2. **P0-A/B dense 特征 shape 正常**：训练和推理都能用同一套 flag 重建 `user_dense_dim=765`，没有 state_dict mismatch。
3. **evaluation 必须使用自定义 inference bundle**：开启 P0-A/B 后，平台默认 infer 未必知道如何生成 synthetic dense features；提交该类模型时应上传 `baseline/infer.py`、`baseline/dataset.py`、`baseline/model.py` 及必要依赖文件，保证训练/推理代码一致。
4. **本地 AUC 不作路线判断**：本次只跑 2 step 且 demo 很小，AUC 0.660317 只说明链路通，不代表线上效果。

## Next

平台下一步走 fast proxy，而不是直接 full train：

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
  --enable_seq_stats_features
```

若 proxy 的 local AUC / LogLoss 不明显退化，再启动 full train；full train 完成后评估时使用本 EXP 里的自定义 inference bundle。
