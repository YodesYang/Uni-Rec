# EXP-019 · ICQ + T13TF Half-Train（icq-t13tf）结果汇总

**日期**：2026-05-04  
**作者**：Agent  
**状态**：✅ 完成 → full train 也已完成 → half-train E4 evaluation **成为 LB SOTA 0.84637**(2026-05-04 晚 9 补档)

---

## 背景

EXP-014 中 ICQ proxy（1500 steps）显示负向 Δ−0.001417，当时判断原因为参数量增加但 steps 不足（gate 未收敛）。  
EXP-018（t13tf full train E4）完成后，决定在 t13tf 基础上开启 ICQ 跑 half-train（4 epoch）验证 full-train 规模下的真实收益。

---

## 配置

**Job Name**：`icq-t13tf-halftrain-swiglu-vr005-bs128`

```
--item_conditioned_query True
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--num_epochs 4
```

---

## 结果

| Epoch | Step | Val AUC | Val LogLoss |
|---|---|---|---|
| E1 | 15939 | 0.864992 | 0.225920 |
| E2 | 31878 | 0.867557 | 0.224177 |
| E3 | 47817 | 0.867965 | 0.223990 |
| **E4** | **63756** | **0.868619** | **0.223356** |

**best = E4 @ step 63756，local AUC = 0.868619**

总训练时长：约 5h 19min（单 epoch ≈ 80min，与 t13tf 一致）

---

## 与对照比较

| Config | ICQ | hash | timefeat | E4 AUC | vs t13tf |
|---|---|---|---|---|---|
| t13tf（EXP-018 best）| ✗ | 500k | ✓ | 0.868495 | baseline |
| **icq-t13tf（本次）** | ✓ | 500k | ✓ | **0.868619** | **+0.000124** |

---

## 关键发现

**ICQ proxy（1500 steps）结论翻转**：proxy 尺度 Δ−0.001417 → half-train E4 Δ+0.000124。

与 EXP-015（hash500k proxy 负向 → full train 正向）模式完全一致，再次验证了：

> **含大型 embedding table 或新参数分支的变量，proxy 因冷启动不足会显示假负向；绕过 proxy 直接用 2~3 epoch half-train 才可靠。**

ICQ 增加 +429,824 dense 参数（gate 网络），需要足够的梯度信号让 gate 从随机初始化（sigmoid ≈ 0.5）收敛到有意义的 item-specific steering。

**收益方向**：E4 时 ICQ 收益较小（+0.000124），但训练曲线全程单调上升，E4 仍是 global best（无早停迹象），预计 full train（10 epoch）可累积更大增益。

---

## 决策

**推进 ICQ+t13tf full train（10 epoch）**

- Job Name 建议：`icq-t13tf-full-swiglu-vr005-bs128`
- 配置同上，去掉 `--num_epochs 4` 限制（或改为 `--num_epochs 10`）
- evaluation 阈值：final best local AUC ≥ **0.870**

不在 half-train E4 ckpt 做 evaluation，+0.000124 local 按 34% 转化率预计 LB 仅 +0.00004，不值得消耗配额。

---

## 下一步

1. 启动 `icq-t13tf-full-swiglu-vr005-bs128`（10 epoch full train）
2. 等待 final `[summary]`，阈值 ≥ 0.870 考虑 evaluation
3. 同期关注 `timefeat-full` 是否已跑完（T12），两者做 E10 对比确认 ICQ 增益

---

## 补档 · full train 回流 + half-train evaluation 翻转（2026-05-04 晚 9）

### ICQ full train 完整轨迹（job `icq-t13tf-full-swiglu-vr005-bs128`）

| Epoch | Step | Val AUC | Val LogLoss | ΔAUC | ΔLogLoss | EarlyStop 状态 |
|---|---|---|---|---|---|---|
| E1 | 14k | 0.864992 | 0.225920 | — | — | reset |
| E2 | 28k | 0.867575 | 0.224159 | +0.002583 | −0.001761 | reset |
| E3 | 42k | 0.868195 | 0.223949 | +0.000620 | −0.000209 | reset |
| E4 | 56k | 0.868669 | 0.223416 | +0.000474 | −0.000534 | reset |
| E5 | 70k | 0.868733 | 0.223572 | +0.000064 | **+0.000156 ← LogLoss 首次反弹** | reset |
| **E6** | 84k | **0.869138** | 0.222990 | +0.000405 | −0.000582 | reset ← AUC global best |
| E7 | 98k | 0.868604 | 0.223510 | −0.000534 | +0.000520 | +1 |
| E8 | 112k | 0.868837 | 0.222797 | +0.000233 | −0.000713 | +2 |
| E9 | 126k | 0.868452 | 0.223949 | −0.000386 | +0.001152 | +3 |

注：该 full train 的训练日志显示 `logloss_rebound_patience=None`（旧 commit），LogLoss rebound guard 尚未落地。若当时启用（patience=1），应在 E5 评估完停机，best_model=E4 ckpt。

### 关键 Evaluation 对照

| CKPT | Local AUC | **LB AUC** | Local−LB Gap |
|---|---|---|---|
| t13tf-full E4（EXP-018 参考基线） | 0.868495 | 0.844956 | +0.023539 |
| **icq-t13tf-halftrain E4** | **0.868619** | **0.84637** ✅ **LB SOTA** | **+0.022249** |
| icq-t13tf-full E6（global best） | 0.869138 | 0.843599 ❌ | +0.025539 |

**⭐ 核心发现**：
- **half-train E4 ckpt 的 LB (0.84637) 比 full-train best E6 ckpt 的 LB (0.843599) 高 +0.00277**。
- **Local 涨 +0.000519（E4→E6）,LB 反而跌 −0.00277,负向杠杆 ~5.3×**。
- LB 新天花板 **0.84637** 由 icq-half E4 建立，vs 前 SOTA t13tf-full E4 的 0.844956 提升 **+0.001414**。

### 机制解释

E5 起 LogLoss 首次反弹（+0.000156），标志过拟合开始：
- AUC 仍在涨 0.000064 → 0.000405 → 0.000233（E5~E8），但这部分涨幅来自 train/dev 共享的**局部时间模式记忆**
- test（远未来）的时间模式与 dev 不同 → 过拟合的 AUC 增益无法传递到 LB
- `valid→LB gap` 从 E4 的 +0.0222 扩大到 E6 的 +0.0255（两个 job 的 gap 都随训练长度单调扩大）

### 对后续实验的影响

1. **LB 新红线 0.84637**：任何新策略的 evaluation 阈值至少要在此之上。
2. **优先 evaluation half-train E4 ckpt**，不必等 full train 完成。
3. **确认 LogLoss rebound guard 的必要性**：trainer 里已实施（patience=1 默认），未来实验默认保护。
4. **valid→LB gap 不是常数，随训练长度扩大**：更新 team-memory 铁律（已落）。

### 三条新铁律（已写入 team-memory.mdc · 2026-05-04 晚 9）

1. Half-train E4 ckpt 的 LB 可以优于 full-train best ckpt
2. valid→LB gap 随训练长度扩大而非常数
3. Holdout AUC 在 half-train E1/E2 比 dev AUC 更稳定/更优

### 本次 evaluation 消耗情况

- 当日 evaluation quota 用量：2/4（`t13tf-loss-full` E9 LB 0.838707 + `icq-t13tf-halftrain` E4 LB 0.84637）
- 剩余 2 slots 至 PDT 21:00 窗口重置
- 下一个 evaluation 触发条件：四合一 / 三合一 half-train E4 的 holdout AUC ≥ 0.867（预测 LB ≥ 0.847，即超过当前 SOTA 0.84637 至少 +0.0003）
