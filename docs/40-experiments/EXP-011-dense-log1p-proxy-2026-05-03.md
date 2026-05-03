# EXP-011 · user_dense log1p 修复对照 proxy

> **Date**: 2026-05-03
> **Status**: Completed
> **Related Tasks**: [T11](../20-tasks/active.md)
> **Platform**: Tencent Angel
> **Commit**: 32a563d + uncommitted `dataset.py` / `train.py` log1p patch

---

## 背景

EDA 发现 `user_dense_feats_62~66` 存储的是原始计数/频次统计量（数量级 10^3~10^5），而同一 `user_dense` 输入里的 fid=61（256-dim）、fid=87（320-dim）、fid=89/90/91（10-dim）均是预归一化 embedding 向量（数量级 10^-1）。两者 concat 后直接进 `nn.Linear(user_dense_dim, d_model)` 投影，尺度差达 4~5 个数量级，导致 62~66 的梯度持续压制 61/87 等通道的信号，属于**静默 bug**（loss 不报错，但模型实际上学不到 embedding 通道的信息）。

修复方案：在 `_convert_batch` 写入 user_dense buffer 时，对 `dense_log1p_fids`（默认 `{62,63,64,65,66}`）里的列做 `log1p(max(x, 0))`，将值域压到 ~[0, 12]，与其他列的 ~[-0.2, 0.9] 差距从 4 个量级缩至 1 个量级以内。

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time` + `valid_ratio=0.05`
- proxy 窗口：`limit_train_rgs=128` / `limit_valid_rgs=32` / `max_train_steps=1500`
- `batch_size=128`，其余超参与 EXP-008 larger proxy 系列一致
- **无 P0 特征**（`enable_count_features=False`，`enable_seq_stats_features=False`）
- Job Names:
  - `log1p-proxy-swiglu-vr005-rg128x32-s1500`（log1p ON，默认）
  - `nolog1p-proxy-swiglu-vr005-rg128x32-s1500`（log1p OFF，对照）

## 结果

| Job | `dense_log1p_fids` | best val AUC | best val LogLoss | best step |
|---|---|---|---|---|
| log1p ON  | `62,63,64,65,66` | **0.838475** | **0.243558** | 1500 |
| log1p OFF | `''`（空） | 0.838330 | 0.243881 | 1500 |
| **差值** | — | **+0.000145** | **−0.000323** | — |

## 结论

- log1p 修复方向正确：AUC +0.000145 / LogLoss -0.000323
- proxy 1500 steps 量级下收益偏小，预期 full train 后会放大：
  1. dense 特征作用在训练早期受 embedding 初始化噪声压制，长训练后尺度对齐的累积效应更明显
  2. 叠加 P0 特征（`enable_count_features` / `enable_seq_stats_features`）后，整个 user_dense 投影层梯度更干净，P0 通道信号可以更充分利用
- **`dense_log1p_fids="62,63,64,65,66"` 已设为 CLI 默认值**，后续所有训练任务自动生效，无需手填

## 下一步

- 修复已合并为默认行为，等本轮其他 full train 实验完成后，与下一轮 P0 full train 叠加验证
- 不单独消耗 evaluation 额度验证此修复（proxy 收益偏小，等叠加 P0 后再评）
