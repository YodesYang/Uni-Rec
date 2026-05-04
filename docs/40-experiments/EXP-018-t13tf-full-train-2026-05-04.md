# EXP-018 · T13+TF Full Train（t13tf）结果汇总

**日期**：2026-05-04  
**作者**：Agent  
**状态**：✅ 完成 + 已评估

---

## 配置

**Job Name**：`t13tf-full-swiglu-vr005-bs128`

```
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--num_workers 4
```

---

## 结果

| Epoch | local AUC | 备注 |
|---|---|---|
| E4 | **0.868495** | 当前最高 local AUC |

**Leaderboard 评估结果**：**0.844956**

---

## Local → Leaderboard 校准

| | 本次（t13tf）| 参考（time split baseline）|
|---|---|---|
| Local valid AUC | 0.868495 | 0.865884 |
| Leaderboard AUC | 0.844956 | 0.844059 |
| Local 提升 | +0.002611 | — |
| LB 提升 | +0.000897 | — |
| **转化率** | **34%** | — |
| Local-LB Gap | 0.023539 | 0.021825 |

**⚠️ 重要发现**：转化率只有 34%，Gap 相比上次扩大。本地 AUC 提升不能等比转化到 LB，后续优化需关注泛化性而非单纯堆本地 AUC。

---

## 三路 E4 归因分析

同期跑了三个对照 job，E4 快照对比：

| Config | hash | timefeat | E4 AUC | vs baseline |
|---|---|---|---|---|
| baseline（no-P0 ref） | ✗ | ✗ | 0.865884 | — |
| timefeat-only（full, E4/E6）| ✗ | ✓ | ~0.8678 | +0.001916 |
| t13only | 500k | ✗ | 0.864827 | **−0.001057** |
| **t13tf** | 500k | ✓ | **0.868495** | **+0.002611** |
| hash2m | 2M | ✓ | 0.867533 | +0.001649 |

**归因计算（基于 t13only 和 t13tf）**：

```
timefeat 独立贡献  ≈ +0.001916
hash500k 独立贡献  = 0.864827 - 0.865884 = -0.001057（有害！）
预期独立叠加        = -0.001057 + 0.001916 = +0.000859
t13tf 实际增益      = +0.002611
──────────────────────────────────────────
正向交互项          = +0.001752  ← 关键发现
```

**结论**：hash500k 单独有害（table 未充分预热），但与 timefeat 组合存在 **+0.001752 的强正向交互**——timefeat 提供的梯度信号显著加速了 hash embedding 的收敛。

---

## hash2m 意外表现

hash2m（2M 桶）在 proxy 显示 +0.001028（优于 500k 的 -0.001601），但 E4 full train 却低于 t13tf（0.867533 vs 0.868495，Δ−0.000962）。

假说："可控碰撞是隐式正则化"——500k 强迫相似 item 共享 embedding，在广告 CVR 这类噪声数据上可能起到平滑效果，优于 2M 的更独立表示。

---

## 下一步

- 等 hash2m 和 t13only full train 完成
- 若 hash2m final AUC ≥ 0.870，考虑第二次 evaluation
- 下一个优先探索：ICQ（item-conditioned query）half-train（4 epoch），DIN-style 注意力可能需要 full train 规模才能体现收益
