# EXP-019 · ICQ + T13TF Half-Train（icq-t13tf）结果汇总

**日期**：2026-05-04  
**作者**：Agent  
**状态**：✅ 完成 → 推进 full train

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
