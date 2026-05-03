# EXP-012 · Impression 时刻时间特征对照 proxy（T12）

> **Date**: 2026-05-04
> **Status**: Completed
> **Related Tasks**: [T12](../20-tasks/active.md), [T11](../20-tasks/active.md)
> **Platform**: Tencent Angel
> **Commit**: 754196d（feat: add impression time-of-day features）

---

## 背景

社区情报 ray（Academic Track 第 27 名）报告"加时间特征 0.8135 → 0.824"（+0.0105），是其所有优化里最大单点。我们的 P0-B（`enable_seq_stats_features`：序列时间戳范围）在 full train 上未复现类似收益，推断 ray 的"时间特征"是 **impression 请求时刻的绝对时间**，而非序列内相对时间。

实施：从行级 `timestamp` 提取 5 个 synthetic dense 特征：
- `hour_sin`, `hour_cos`：小时（0~23）的圆形编码
- `dow_sin`, `dow_cos`：星期几（0~6）的圆形编码
- `is_weekend`：周六/周日为 1，其余为 0

FID=900004，dim=5，追加至 `user_dense` 通道末尾（`--enable_time_of_day_features` opt-in）。

## Setup

- 数据：`/data_ams/industrial_training_data`
- 基础配置：`swiglu` + `rankmixer` + `valid_split_strategy=time` + `valid_ratio=0.05`
- proxy 窗口：`limit_train_rgs=128` / `limit_valid_rgs=32` / `max_train_steps=1500`
- `batch_size=128`，`dense_log1p_fids=62,63,64,65,66`（默认 ON）
- Job Names:
  - `timefeat-proxy-swiglu-vr005-rg128x32-s1500`（timefeat ON）
  - `notimefeat-proxy-swiglu-vr005-rg128x32-s1500`（timefeat OFF，对照）

## 结果

| Job | `enable_time_of_day_features` | best val AUC | best val LogLoss | best step |
|---|---|---|---|---|
| timefeat ON  | `True`  | **0.839283** | **0.241525** | 1500 |
| timefeat OFF | `False` | 0.838475 | 0.243558 | 1500 |
| **差值** | — | **+0.000808** | **−0.002033** | — |

## 与历次 proxy 对比

| 修复/特征 | proxy AUC Δ | proxy LogLoss Δ | 备注 |
|---|---|---|---|
| log1p 修复（EXP-011） | +0.000145 | −0.000323 | 以 nolog1p 为基准 |
| **T12 time-of-day（本轮）** | **+0.000808** | **−0.002033** | 以 timefeat OFF 为基准 |

time-of-day 特征的 proxy 收益约为 log1p 修复的 **5~6 倍**，是目前单点增益最大的特征。

## 结论

- time-of-day 特征方向正确且信号明显，**升为 full train 候选**
- LogLoss −0.002 在 proxy 尺度已非随机波动（之前 log1p 的 LogLoss 仅 −0.0003）
- 圆形编码（sin/cos）避免了午夜 0→23 的不连续跳变，无需额外调参
- `--enable_time_of_day_features` 升为 full train 默认推荐参数

## 下一步

启动 full train：`timefeat-full-swiglu-vr005-bs128`，同时开启 `--enable_time_of_day_features`。
阈值：`best local AUC >= 0.8665` 考虑 evaluation，`>= 0.8685` 强推。
