# EXP-003 · 平台批次实验（2026-05-01：swiglu-time / d_model=128 / dropout=0.05）

> **Date**: 2026-05-01
> **Status**: Completed（dropout=0.05 评估已回流 2026-05-02；transformer re-train 已从历史 eval 列表中确认为 0.834015）
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
| 50709 | swiglu-dropout0.05-eval | dropout=0.05 | **0.83977** | 197.86s | ❌ 低于 swiglu 基线 0.84381（-0.004），dropout 调参方向关闭 |
| 50688 | swiglu-time-valid_ratio_0.05-eval | time, valid_ratio=0.05 | **0.844059** | 188.64s | 当前最佳，微升 +0.000249 |
| 48876 | swiglu-time-valid_ratio_0.05-eval（重复） | time, valid_ratio=0.05 | **0.844059** | 174.06s | 与上一条一致 |
| 46930 | swiglu-d_model128-eval | d_model=128 | **0.836944** | 161.96s | 明显低于 swiglu 基线 0.84381 |
| 46928 | swiglu-valid_split_strategy_time-eval | time（旧链路） | **0.842532** | 232.92s | 旧链路不采信 |
| 44897 | official_baseline-eval | transformer（错误 ckpt） | **0.834015** | 513.9s | 发布了 epoch 1 ckpt |
| 44891 | swiglu_eval | swiglu 基线 | **0.84381** | 81.93s | 历史第一个 swiglu |
| 44851 | longer_eval | longer encoder | **0.8422** | 203.1s | longer 低于 swiglu |

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
| swiglu-split_strategy_time-valid_ratio_0.05 | 0.05 | Completed / Evaluated | 341,086 | **0.865884** | 48060 | 线上 0.844059，小幅超过旧 swiglu 基线但不足以追当日头部 |

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
   - `valid_ratio=0.05` best local AUC 0.865884（step 48060），高于原 swiglu filename local；leaderboard AUC 0.844059，仅较旧 swiglu 基线 0.84381 微升 +0.000249。
   - 当日 leaderboard 最优 0.85035，当前差距 0.006291；split 参数不是破局点，后续应转向 T8.5 row-level cutoff 与 P0-A/P0-B 特征工程。

3. **dropout=0.05 已评估，结论负向（2026-05-02 回流）**：
   - Leaderboard AUC **0.83977**，比 swiglu 基线 0.84381 退步 **-0.004**，比当前最佳 0.844059 退步更多。
   - Inference Time 197.86s 远超 swiglu 通常的 ~82s，可能平台资源竞争导致偏慢，但 AUC 退步是确定性结论。
   - **dropout 调参方向完全关闭**，不再作为特征工程基础配置考虑；当前主线保持 dropout=0.01。

4. **Top5 策略取舍（本批后）**：
   - 关闭 `d_model=128`、`dropout=0.05` 两条路线。
   - 当前 leaderboard 全部评估结果最佳 **0.844059**（swiglu + time + valid_ratio=0.05）。
   - 下一轮高优先级：P0-C（历史 CVR）+ P1（multi-task）proxy，已在本地 smoke 通过，等平台 proxy 回流。

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

- [x] transformer re-train best ckpt 评估结果（确认为 0.834015，系 epoch 1 错误 ckpt）
- [x] swiglu+time **新建 job** 的 `[T8] overlap_sec` + `[diag]` 四张切片表 + valid_ratio=0.05 leaderboard AUC
- [x] swiglu-dropout=0.05 leaderboard AUC（**0.83977，负向，方向关闭**）
- [x] 若 T8/T10 数据齐全，创建 EXP-005（T8.5 + P0-A/B 主线本地 smoke）
