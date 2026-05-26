# 2026 Season Review · 明年快速回忆入口

> 这是 TAAC x KDD Cup 2026 工业组赛季结束后的“压缩记忆”。明年重新打开仓库时，先读本目录，再回到 `00-context/`、`30-decisions/`、`40-experiments/` 做细节追溯。

## 最终结果

| 项目 | 结果 |
|---|---:|
| 赛道 | Industrial Track |
| 最终排名 | **35 / 689** |
| 百分位 | **Top 5.1%** |
| 最佳 public AUC | **0.851365** |
| 最佳公开提交 | `M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep` |
| 最佳 checkpoint | normal `best_model` |
| 被拒绝的 side checkpoint | M148 dense-only `weight_soup` public **0.850563** |

## 三份必读

1. [`final-recap-2026.md`](./final-recap-2026.md)
   赛季压缩复盘：最终结果、主线提升、失败路线、明年要继承什么。

2. [`next-year-playbook.md`](./next-year-playbook.md)
   明年比赛开局手册：第一天该读什么、先跑什么、不要重复踩什么坑。

3. [`../00-context/platform-state.md`](../00-context/platform-state.md)
   全量平台时间线和每个关键任务的原始判断。它很长，但查细节时最可靠。

## 最短记忆版

```text
最终最好：M148 public 0.851365，最终排名 35/689。
真正上分主线：corrected T38 -> M106 -> fresh-tail M119 -> MTL M123 -> M148。
核心认知：验证体系比模块花样更重要；普通 local AUC 经常误导。
最该继承：time-aware split、clean windows、aux windows、public-calibrated family reasoning。
最不该重复：盲信普通 dev、盲信高 aux、盲信 logloss/seed、盲评 time/history/UE/TIN 分支。
```

## 查细节路线

| 你想回忆 | 看这里 |
|---|---|
| 最终 public eval 和最后几天决策 | [`../00-context/platform-state.md`](../00-context/platform-state.md) 顶部 |
| M148/M159/M165/M162/M146 这一族 | [`../40-experiments/EXP-093-final-sprint-m123-family-m145-m150-2026-05-21.md`](../40-experiments/EXP-093-final-sprint-m123-family-m145-m150-2026-05-21.md) |
| M119/M123 如何起来 | [`../40-experiments/EXP-083-m106-overnight-upper-tail-m113-m122-2026-05-19.md`](../40-experiments/EXP-083-m106-overnight-upper-tail-m113-m122-2026-05-19.md), [`../40-experiments/EXP-084-m106-freshness-mtl-m123-m124-2026-05-20.md`](../40-experiments/EXP-084-m106-freshness-mtl-m123-m124-2026-05-20.md) |
| clean-window/validation oracle | [`../40-experiments/EXP-074-time-window-backtest-eda71-2026-05-18.md`](../40-experiments/EXP-074-time-window-backtest-eda71-2026-05-18.md), [`../40-experiments/EXP-079-eda72-validation-oracle-audit-2026-05-19.md`](../40-experiments/EXP-079-eda72-validation-oracle-audit-2026-05-19.md), [`../40-experiments/EXP-080-eda73-validation-oracle-cappedseq-2026-05-19.md`](../40-experiments/EXP-080-eda73-validation-oracle-cappedseq-2026-05-19.md) |
| 哪些方向被证伪 | [`final-recap-2026.md`](./final-recap-2026.md#被证伪或暂时不该继承的路线) |
