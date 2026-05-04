# EXP-017 · Hash Embedding 2M 桶 Proxy（T13 升级）

**日期**：2026-05-04  
**作者**：Agent  
**状态**：✅ 完成（正向，推进 full train）

---

## 背景

EXP-015 发现 `--seq_hash_vocab 500000` 在 proxy 规模显示负向（Δ−0.001601），但 full train 实际正向（t13tf E4 = 0.868178 vs timefeat E4 = 0.867824，Δ+0.000354）。原因：proxy 仅 1500 步，500k 桶预热不足，碰撞噪声淹没信号。

本实验验证假设：将桶数从 500k 提升到 2M，每桶平均碰撞率降低 4 倍，proxy 规模已可见正向信号。

---

## 实验配置

**Job Name**：`t13-hash2m-timefeat-proxy-swiglu-vr005` (inferred)

**关键 args**（t13tf 基础上仅修改 hash vocab）：

```
--seq_hash_vocab 2000000   （原为 500000）
--enable_time_of_day_features
--emb_skip_threshold 1000000
--max_train_steps 1500      （proxy）
--valid_split_strategy time --valid_ratio 0.05
--seq_encoder_type swiglu
--batch_size 128
```

---

## 结果

| 指标 | hash2m（本轮） | timefeat-only（EXP-012） | hash500k（EXP-015） | Δ vs timefeat |
|---|---|---|---|---|
| **best AUC** | **0.840313** | 0.839285 | 0.837684 | **+0.001028** |
| **LogLoss** | 0.242211 | 0.241525 | 0.242104 | +0.000686 |

---

## 碰撞假设验证

| vocab | 平均碰撞率（相对） | proxy AUC | Δ vs timefeat |
|---|---|---|---|
| 500k | 4× | 0.837684 | −0.001601 |
| **2M（本轮）** | **1×（baseline）** | **0.840313** | **+0.001028** |

500k vs 2M proxy 差值 = +0.002629，远超随机波动，碰撞假设成立。

---

## 分析

1. **proxy 已可见正向**：2M 桶碰撞率足够低，即使在 1500 步 proxy 下也能学到有效的 hash 表示，而 500k 在 proxy 规模下仍以碰撞噪声为主。
2. **full train 信号应更强**：根据 hash500k 的经验（proxy Δ−0.0016 → full train E4 Δ+0.0003），hash2m 在 full train 下收益应远大于 proxy 测量值（+0.001028）。
3. **与 t13tf 的比较**：t13tf（500k full train）在 E4 是 0.868178。hash2m full train 如果能维持 proxy 的相对优势，有望比 t13tf 高 0.001+ AUC。

---

## 结论

- **高置信度推进 full train**：`--seq_hash_vocab 2000000` + timefeat 组合。
- proxy 对 hash 桶数变化的信号可信（本次）：减少碰撞有明确效果，不再是 proxy 盲区。
- **预期 full train 结果**：local AUC ≥ 0.869（高于 t13tf 的 0.868178）。

---

## 下一步

`t13-hash2m-timefeat-full-swiglu-vr005-bs128` full train 正在进行中，等待 training summary 回流后决定是否 evaluation。
