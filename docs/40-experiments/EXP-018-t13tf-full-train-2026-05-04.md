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

**初版归因（基于 E4 快照）**：

```
timefeat 独立贡献  ≈ +0.001916
hash500k 独立贡献  = 0.864827 - 0.865884 = -0.001057（有害！）
预期独立叠加        = -0.001057 + 0.001916 = +0.000859
t13tf 实际增益      = +0.002611
──────────────────────────────────────────
正向交互项          = +0.001752  ← 关键发现（初版）
```

**初版结论**：hash500k 单独有害（table 未充分预热），但与 timefeat 组合存在 +0.001752 的强正向交互。

---

## 归因修正（2026-05-04 晚 · E7 完整数据回流后）

**触发**：`t13only-full-swiglu-vr005-bs128` 继续训练到 E7，best local AUC 升到 **0.866571**（E7/E10），不再"有害"。E4 快照是 hash table 尚未充分预热的中间态,不能作为 hash500k 独立贡献的终值。

**修正后归因**（baseline ref 0.865884 · t13only E7 best 0.866571 · t13tf E4 best 0.868495 · t13tf E7 已相近）：

```
timefeat 独立贡献  ≈ +0.001916    （不变，timefeat-only E4 snapshot）
hash500k 独立贡献  = 0.866571 - 0.865884 = +0.000687    （由 −0.001057 翻正）
预期独立叠加        = +0.000687 + 0.001916 = +0.002603
t13tf 实际增益      = +0.002611
──────────────────────────────────────────
正向交互项          ≈ +0.000008   （初版 +0.001752 → 实际 ~0）
```

**修正后结论**：hash500k 与 timefeat **几乎是独立可加的**——初版看到的 "+0.001752 强交互" 本质是 E4 快照偏差带来的伪信号;hash500k 只是需要比 timefeat 更长训练步数才能让 embedding table 充分预热。

**意义**：
- "大参数稀疏分支 proxy 假负向 → full train 正向" 规律再次成立（EXP-015 hash500k proxy Δ−0.0016 → full train E7 Δ+0.0007）
- 未来所有"引入新大型 sparse embedding table"类 proxy 应**绕过 proxy，直接 half-train（2~3 epoch）**（已写入 team-memory）
- 不能用 E4 中间态作为任何稀疏分支的最终归因依据

---

## hash2m 意外表现

hash2m（2M 桶，E9 best 0.868483）在 proxy 显示 +0.001028（优于 500k 的 -0.001601），但 **full train E9 仍低于 t13tf E4 的 0.868495**（Δ−0.000012，几乎持平但没体现 2M 桶的低碰撞优势）。

假说："可控碰撞是隐式正则化"——500k 强迫相似 item 共享 embedding，在广告 CVR 这类噪声数据上可能起到平滑效果，优于 2M 的更独立表示。**在本项目数据规模（1.9M train rows）下，hash vocab size 并非越大越好**，该结论已写入 team-memory 铁律。

---

## `t13tf-loss-full` 的 LogLoss-select 副作用

同期启动的 `t13tf-loss-full-swiglu-vr005-bs128`（仅多一个 `--early_stop_metric logloss`）回流显示：
- **实际 best AUC E5 = 0.868932**（超过 t13tf E4 的 0.868495）
- 但因 EarlyStopping monitor=LogLoss，E8 ckpt（AUC=0.868571，LogLoss 更低）被 overwrite 保留,**E5 的 AUC 最高 ckpt 被丢弃**

**教训**：比赛指标是 AUC，EarlyStopping 必须也以 AUC 为 monitor;LogLoss-select 会静默丢失 AUC 最高 ckpt（已写入 team-memory 铁律）。

---

## 下一步

- ✅ t13only E7 数据已并入归因（2026-05-04 晚）
- 等 `icq-t13tf-full-swiglu-vr005-bs128` 完成，若 best AUC ≥ 0.870 考虑第二次 evaluation
- **不再单独 evaluate t13only / t13tf-loss** — 分数未超过首轮 t13tf（0.844956），信息增益低，不值一个 evaluation 配额（每日仅 4 次）
- 下一个优先探索见新 ADR 草稿：target-item 硬交互（把 target item concat 到 seq side-info）
