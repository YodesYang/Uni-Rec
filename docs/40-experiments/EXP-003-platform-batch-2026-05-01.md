# EXP-003 · 平台批次实验（2026-05-01：swiglu-time / d_model=128 / dropout=0.05）

> **Date**: 2026-05-01
> **Status**: In Progress（2 个已评估 + 1 个训练完成待评估 + 1 个任务仍在跑）
> **Related Tasks**: T8 / T10 / T6
> **Related ADR**: [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md), [ADR-002](../30-decisions/ADR-002-slice-diagnostics.md), [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)
> **Platform**: Tencent Angel
> **Commit**: 7f05ea7（实验前文档状态）

---

## Setup

- 数据：`/data_ams/industrial_training_data`（1.9M / 1000 shards）
- 公共配置：`seq_encoder_type=swiglu`, `batch_size=256`, `seed=42`, `rankmixer`, `num_queries=2`
- 本批实验：
  1. `--valid_split_strategy time`
  2. `--d_model 128`
  3. `--dropout_rate 0.05`
  4. transformer re-train（修复错误 publish ckpt，仍在跑）

---

## Result

### 评估结果（当前可见）

| Eval ID | 任务名 | 关键配置 | Leaderboard AUC | Inference Time | 结论 |
|---|---|---|---|---|---|
| 46930 | swiglu-d_model128-eval | d_model=128 | **0.836944** | 161.96s | 明显低于 swiglu 基线 0.84381 |
| 46928 | swiglu-valid_split_strategy_time-eval | valid_split_strategy=time（用户说明：基于旧 swiglu 任务修改） | **0.842532** | 232.92s | 比基线低 0.0013，当前不构成 time split 有效证据 |
| - | swiglu-dropout=0.05 | dropout=0.05 | 待评估额度刷新 | - | 训练已完成，待提交评估 |
| - | transformer re-train | transformer | 仍在训练 | - | 待回收 best ckpt 后评估 |

### 训练侧 best Val AUC（来自 `train_log.txt`）

| 训练任务 | Local Best Val AUC | Best Step | 收敛形态 |
|---|---|---|---|
| swiglu + time | **0.865094** | 30440 | Early Stop @ epoch 9 |
| swiglu + d_model=128 | **0.864881** | 15220 | Early Stop @ epoch 7（早期见顶） |
| swiglu + dropout=0.05 | **0.864563** | 53270 | 训练到 epoch 11，后期缓慢回落 |

### 修复后 time split full jobs（2026-05-02 回流）

| 训练任务 | valid_ratio | 状态 | overlap_sec | Local Best Val AUC | Best Step | 当前结论 |
|---|---:|---|---:|---:|---:|---|
| swiglu-valid_split_strategy_time | 0.10 | Completed / Early Stop @ epoch 12 | 341,068 | 0.863248 | 53004 | RG-level time split 仍严重 overlap，local AUC 低于 swiglu filename 基线 |
| swiglu-split_strategy_time-valid_ratio_0.05 | 0.05 | Training（截至 epoch 9） | 341,086 | **0.865884** | 48060 | 当前 local 最强，但仍需等训练完成后再决定 evaluation |

关键日志：

```text
# valid_ratio=0.10
[T8] train ts: [1772238909, 1772588940]
     valid ts: [1772247872, 1772588990]
     overlap_sec: 341068
Epoch 7 Validation | AUC: 0.8632482256979388
Early stopping at epoch 12

# valid_ratio=0.05（仍在训练）
[T8] train ts: [1772238909, 1772588958]
     valid ts: [1772247872, 1772588990]
     overlap_sec: 341086
Epoch 6 Validation | AUC: 0.8658836865584829
Epoch 9 Validation | AUC: 0.8644454326618461
model earlyStopping counter: 3 / 5
```

---

## Findings（关键发现）

1. **d_model=128 路线当前判定为负收益**：
   - Local best 与 d_model=64 基线几乎持平，但 leaderboard 从 0.84381 降到 0.836944（-0.0069）。
   - 推断：当前数据规模下增 dense 容量引入了更强分布外过拟合，不是瓶颈方向。

2. **swiglu+time 结果暂不可用于验证 T8**：
   - 用户确认该任务是直接在旧 swiglu 任务上改参数提交，非全新 job 链路。
   - 当前缺少 `[T8]` / `[diag]` 关键日志片段，不足以证明 `overlap_sec` 与切片诊断是否生效。
   - 已有新建 job 重跑，等待新日志回收再做结论。

2b. **修复后 RG-level time split 仍不能形成 clean future validation**：
   - `valid_ratio=0.10` full job：`overlap_sec=341068`。
   - `valid_ratio=0.05` full job：截至 epoch 9，`overlap_sec=341086`。
   - 结论与 [EXP-004](./EXP-004-fast-proxy-smoke-2026-05-02.md) proxy 对照一致：Row Group 内部时间跨度太宽，单靠 RG 排序无法消除 row-level overlap；T8.5 row-level timestamp cutoff 应升为下一轮高优先级工程动作。

2c. **valid_ratio=0.05 当前比 0.10 更值得等完成后评估**：
   - `valid_ratio=0.10` best local AUC 0.863248，低于原 swiglu filename local 0.8650。
   - `valid_ratio=0.05` 截至 epoch 9 best local AUC 0.865884（step 48060），高于原 swiglu filename local，但尚未 Early Stop，需等待最终 best checkpoint。

3. **dropout=0.05 是当前最值得立刻补评估的候选**：
   - 训练已完成，Local best 0.864563，和基线差距不大。
   - 若线上 AUC 不降（或微升），可作为抗过拟合配置进入下一轮 P0 特征工程基线。

4. **Top5 策略取舍（本批后）**：
   - 关闭 `d_model=128` 路线（除非后续有强正则组合新证据）。
   - 保留 `swiglu` 主线，优先推进：dropout 评估 -> transformer-retrain 修复回收 -> T8/T10 真正有效日志回收 -> ADR-003 P0 特征工程。

---

## Reproduce Recipe

平台训练命令（示意）：

```bash
bash baseline/run.sh --seq_encoder_type swiglu --valid_split_strategy time
bash baseline/run.sh --seq_encoder_type swiglu --d_model 128
bash baseline/run.sh --seq_encoder_type swiglu --dropout_rate 0.05
```

已知踩坑：

- **不要从旧任务直接 Copy 改参数就当作“新代码版本验证”**：Copy 可能复用冻结脚本快照，导致 T8/T10 探针未真正生效。
- 评估提交前必须做 `best_model` 三步校验（见 `team-memory.mdc`）。

---

## 待回填

- [ ] transformer re-train 完成后的 best ckpt 评估结果（目标：修复 wrong publish 后回到 ~0.836 或更高）
- [x] swiglu+time **新建 job** 的 `[T8] overlap_sec` + `[diag]` 四张切片表（valid_ratio=0.10 已完成；0.05 仍在训练）
- [ ] swiglu-dropout=0.05 leaderboard AUC
- [ ] 若 T8/T10 数据齐全，创建 EXP-004（P0 特征工程首轮）
