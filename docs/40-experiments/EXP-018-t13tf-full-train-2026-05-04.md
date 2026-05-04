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

## `t13tf-loss-full` 的 LogLoss-select 副作用 + LB 实测（2026-05-04 晚回流）

> 本节含**后续实测 LB 分数**，请读到底再做结论。

### 训练收敛（完整 epoch-wise 表）

| Epoch | Valid AUC | Valid LogLoss | Train Avg Loss | AUC Δ | LogLoss Δ | 信号 |
|---|---|---|---|---|---|---|
| E1 | 0.864577 | 0.226712 | 0.235514 | — | — | |
| E2 | 0.865825 | 0.225783 | 0.231683 | +0.001248 | −0.000929 | ✓ 同向改善 |
| E3 | 0.867679 | 0.224243 | 0.229686 | +0.001854 | −0.001540 | ✓ 同向改善 |
| **E4** | **0.868349** | **0.223789** | 0.227966 | +0.000670 | **−0.000454** | ✓ 最后同向改善 epoch |
| **E5** | **0.868932** | 0.224397 | 0.227220 | +0.000583 | **+0.000608** | ⚠️ **LogLoss 反弹，过拟合拐点** |
| E6 | 0.868684 | 0.223547 | 0.226324 | −0.000248 | −0.000850 | AUC 开始回落 |
| E7 | 0.868550 | 0.224680 | 0.226026 | −0.000134 | +0.001133 | 进一步劣化 |
| E8 | 0.868571 | **0.223020** | 0.225189 | +0.000021 | −0.001660 | LogLoss 局部最低（但 AUC 不是） |
| E9 | **0.868989** | —（未记录） | — | +0.000418 | — | AUC 新高（但已深度过拟合） |

**拐点判据**：E4→E5 是 **AUC 仍涨但 LogLoss 反弹** 的第一个 epoch。

### Leaderboard 实测

| Ckpt | Valid AUC | Valid LogLoss | LB AUC | Local-LB Gap |
|---|---|---|---|---|
| t13tf-full **E4**（首次提交的 best） | 0.868495 | — | **0.844956** | 0.023539 |
| t13tf-loss-full **E9**（新提交，10 epoch 过训练） | **0.868989** | — | **0.838707** | **0.030282** |

**关键对照**：valid AUC **+0.000494**，LB AUC **−0.006249**。local 涨 1 份，LB 跌 **10 份**的负向杠杆。

### 机制解释

E5 起，模型为了进一步记忆 train/valid 尾部共享的局部时间模式，把预测概率推向更极端（positive 更接近 1、negative 更接近 0）：
- **AUC**（排序指标）只看相对顺序，对这种"拉伸"不敏感甚至受益 → 继续涨
- **LogLoss**（校准指标）对过度自信的错误预测惩罚呈 log 增长 → 反弹
- **LB test 集**是**更远未来**的样本，协变量漂移更大，模型记忆的局部模式**失效** → AUC 掉得比 valid 还多

### 历史 7 次提交的 Gap 扫描

| 提交 | Local Valid AUC | LB AUC | Gap |
|---|---|---|---|
| transformer baseline (epoch 1 错 ckpt) | 0.8617 | 0.834015 | 0.0277 |
| swiglu baseline (filename) | 0.8650 | 0.84381 | 0.0212 |
| swiglu + time + vr=0.05 | 0.865884 | 0.844059 | 0.0218 |
| swiglu + d_model=128 | 0.864881 | 0.836944 | 0.0279 |
| swiglu + dropout=0.05 | 0.864563 | 0.83977 | 0.0248 |
| **t13tf-full E4** | 0.868495 | **0.844956** | **0.0235** |
| **t13tf-loss-full E9** | 0.868989 | **0.838707** | **0.0303** |

**Gap 稳定在 0.021~0.030**，不是单点事故，是 **valid split 方法论**（尾部 valid 与训练尾部共享局部时间模式）的系统性症状。

### 三重教训（已沉淀 team-memory 铁律）

1. **单 monitor EarlyStopping 不可靠**：AUC-only 容易训过头（本案 E4→E9 多训 5 epoch 反而 LB 掉 0.006），LogLoss-only 会静默丢 AUC 最高 ckpt（E5 AUC 0.868932 被 E8 0.868571 覆盖）。必须**双指标联合停机**（任一指标反弹即停）。
2. **local AUC 不是 LB 的等比例代理**：决策红线以 `t13tf-full E4 LB = 0.844956` 为准，不以 local AUC 为准。
3. **valid split 系统性缺陷**：gap 0.021~0.030 不会自行消失。后续需 valid split 重设计（ADR-005 Proposed）。

---

## 下一步

- ✅ t13only E7 数据已并入归因（2026-05-04 晚）
- ✅ t13tf-loss-full E9 已 evaluate，LB = 0.838707，**低于** t13tf-full E4 的 0.844956 → 过拟合诊断成立
- ❌ t13only / t13tf-loss 不再追加 evaluation（信息增益为负）
- ⏸ icq-t13tf-full 等训完再看（但 EarlyStopping 策略仍是 AUC-only，需警惕过训练）
- 🔥 **当前第一优先级已变为**：trainer 实施 AUC+LogLoss 双指标联合 EarlyStopping（LogLoss 反弹即停）→ T16 再做 full train
- 下一个战略 ADR 草稿：ADR-005 Valid Split Redesign（从 gap 根源下手）
