# EXP-016 · 3-Block HyFormer Proxy（模型加深探索）

**日期**：2026-05-04  
**作者**：Agent  
**状态**：✅ 完成（负向，不推进 full train）

---

## 目标

验证将 `num_hyformer_blocks` 从 2 增加到 3 是否能带来 AUC 提升，探索模型容量瓶颈。

---

## 实验配置

**Job Name**：`t14-3blocks-timefeat-proxy-swiglu-vr005-rg128x32-s1500`

**关键 args**（在 timefeat baseline 上叠加 +1 block）：

```
--num_hyformer_blocks 3   （原为 2）
--enable_time_of_day_features
--limit_train_rgs 128 --limit_valid_rgs 32
--max_train_steps 1500
--valid_split_strategy time --valid_ratio 0.05
--seq_encoder_type swiglu
--batch_size 128 --num_epochs 4
```

**模型规模变化**：
- Dense 参数：2,783,041（+13% vs 2-blocks 的 2,466,049）
- Sparse 参数：196,182,912（vs 356M，本次 proxy 因 emb_skip 差异略少）

---

## 结果

| 指标 | 3-blocks + timefeat（本轮） | 2-blocks + timefeat（EXP-012） | Δ |
|---|---|---|---|
| **best AUC** | 0.838003 | 0.839285 | **−0.001282** |
| **LogLoss** | 0.242790 | 0.241525 | +0.001265 |
| avg train loss | 0.2582 | 0.2572 | +0.001 |

**切片诊断（max-step eval）**：

```
[diag] slice by user_int_nonzero:
[0,58]    AUC=0.8691
[58,71]   AUC=0.8390
[71,85]   AUC=0.8179
[85,283]  AUC=0.8200

[diag] slice by item_int_nonzero:
[0,8]     AUC=0.9189
[8,10]    AUC=0.8104
[10,12]   AUC=0.7774
[12,33]   AUC=0.8298

[diag] slice by seq_total_len:
[0,1114]     AUC=0.8782
[1114,1321]  AUC=0.8308
[1321,1536]  AUC=0.8182

label_type=1  pred_mean=0.0866
label_type=2  pred_mean=0.3473
```

---

## 分析

1. **proxy 规模不足以区分更深模型**：1500 步对 3 块模型的预热时间与 2 块相同，但 3 块有更多待优化参数。proxy 结果为负不代表 full train 一定差，但按照漏斗规则，负向 proxy 不推进 full train。

2. **2-block 已经够用**：在当前 d_model=64、T=16 tokens 的组合下，2 块 HyFormer 已经足够表达力。加第 3 块的边际收益在 proxy 尺度上未体现正向信号。

3. **切片分布基本一致**：各桶 AUC 与 EXP-012 相比无结构性差异，说明是全局均匀的小幅退步而非某个片段特别差。

---

## 结论

- **不推进 3-blocks full train**：proxy Δ−0.001282，低于 timefeat-only 基线。
- **维持 `num_hyformer_blocks=2` 作为默认**。
- 目前所有 proxy 候选全部完成，valid 结果汇总：

| 优化项 | proxy AUC Δ | 状态 |
|---|---|---|
| dense log1p 修复 | +0.000145 | 默认 ON |
| T12 time-of-day features | **+0.000808** | full train 中 |
| focal loss (α=0.25, γ=2.0) | +0.000234 | 不推进 |
| ICQ 目标物品感知 query | −0.001417 | 放弃 |
| T13 hash embedding (500k) | −0.001601 | 放弃 |
| **3-blocks（本轮）** | **−0.001282** | 放弃 |

**唯一等待项**：`timefeat-full-swiglu-vr005-bs128` full train 结果。
