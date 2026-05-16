# EXP-060 · M50 双 Stack Dev 闭环 + Path B / Daypart 互补审视 · 2026-05-16

> **Status**: ✅ Closed (M50 dev 已出 · 双 stack 损害 net · 但 LB 仍待评)
> **Date**: 2026-05-16 CEST 16:00
> **Trigger**: M50 (UE split + weekend filter 双 stack) 训练完成 · dev 0.861974 · 用户反问 "train/valid 是否一致" 触发严格 source-of-truth audit
> **Related**: EXP-057 (daypart batch 死透) · EXP-058 (8 大错误 audit · §Error A weekend 真信号) · EXP-059 (M49 LB 0.848162 path B 弱真信号)

---

## TL;DR

**M50 dev 0.861974** vs M22 baseline 0.868619 = **−0.006645** ⚠️
**M50 dev** vs M49 (UE split alone) 0.870741 = **−0.008767** ⚠️

**判决修订** (5/16 CEST 16:00 用户反问触发):
1. **M50 train/valid 完全独立** (源码 verify · valid set 与 M22/M49 完全相同)
2. **dev 严格可对比** · M50 dev 下降是真实信号
3. **但 LB 不能仅由 dev 推断** · "train/test 同质性增益" hypothesis 仍开放
4. **修订决断**: M50 LB 加入 5/17 早 quota 候选 (从原 "不评" → 优先级 ⭐⭐ 中等)

---

## 1. M50 训练数据

### 1.1 Args 验证 (30s 流程)

```
enable_ue_split: True
ue_split_fid: 61                          # = user_dense_feats_61 (256-dim 预训练 UE)
user_train_timestamp_min: 1772409600      # UTC 03-02 00:00 · 剔除 02-28+03-01 周末
seed: 42
valid_split_strategy: time
valid_ratio: 0.05
gap_ratio: 0 · holdout_ratio: 0
num_epochs: 4
```

✅ 所有关键 keys 配置正确

### 1.2 Training Summary

| Field | Value |
|---|---|
| Job Name | M50-m22-ue-split-weekend-gap0-seed42-4ep |
| Train wall-time | 4h40m47s (vs M49 2h48m / M22 ~3h) |
| stop_reason | max_epochs_completed |
| best dev AUC | **0.861974** (E4) |
| best dev LogLoss | 0.229625 |
| val_auc_history | E1: 0.853858 → E2: 0.858502 → E3: 0.860125 → **E4: 0.861974** (单调上升) |
| topk_ckpts | E4 best · 全部 epoch 都好转 (无 overfit · 无 logloss rebound) |

---

## 2. Source-of-truth Audit (用户反问 "train/valid 是否一致" 触发)

### 2.1 严格 grep src/dataset.py

**train_dataset 构建** (src/dataset.py:3271-3274):
```python
row_groups=train_rgs,
timestamp_min=train_timestamp_min,    # = user_train_timestamp_min = 1772409600
timestamp_max=train_timestamp_max,    # = None (M50 没启 row_time_cutoff)
```

**valid_dataset 构建** (src/dataset.py:3327-3337):
```python
row_groups=valid_rgs,
timestamp_min=valid_timestamp_min,    # = None (line 3087 默认)
timestamp_max=valid_timestamp_max,    # = None
```

**timestamp_min 在 PCVRParquetDataset 里的用法** (src/dataset.py:1761-1762):
```python
if self.timestamp_min is not None:
    keep_mask &= timestamps >= self.timestamp_min
```

→ valid_dataset 的 `timestamp_min=None` · **不做过滤** · 完整保留 valid_rgs 所有行

### 2.2 valid_rgs 选择

`valid_rgs` = `rg_info[valid_start_idx:valid_end_idx]` = 时间排序后**尾部 5% RG**
- 与 `user_train_timestamp_min` 无关
- 与 M22 / M49 完全相同 (same valid_split_strategy='time' + valid_ratio=0.05)

### 2.3 结论

✅ **M22 / M49 / M50 valid set 完全相同** (相同 row groups · 相同行 · 不被 weekend filter 影响)

✅ **dev 数据严格可对比** · M50 dev 下降 −0.0066 是真实方法学差异 · 不是 valid set 偏移

✅ **train set 不同**:
- M22 / M49 train: 全 timestamps · 含周末 (~2.1M rows)
- M50 train: timestamp ≥ 1772409600 · 剔周末 (~1.95M rows · 减 7.2%)

---

## 3. M50 vs M22 vs M49 vs M45 数据矩阵

| Job | Config | Train rows | Train 含周末 | Valid set | Dev | Δ vs M22 |
|---|---|---|---|---|---|---|
| **M22** | vocabtier baseline | ~2.1M | ✅ | 尾部 5% | 0.868619 | (anchor) |
| **M49** | + UE split (T37) | ~2.1M | ✅ | 同 M22 | 0.870741 | **+0.002122** ✅ |
| **M45** | + weekend filter | ~1.95M | ❌ | 同 M22 | 0.858907 | **−0.009712** ⚠️ |
| **M50** | + UE split + weekend | ~1.95M | ❌ | 同 M22 | 0.861974 | **−0.006645** ⚠️ |

### 双 stack 增量分解 (假设 加性独立)

预测 (假设线性加性):
- M22 + UE split: +0.002122
- M22 + weekend filter: −0.009712
- M22 + 双 stack 预测: +0.002122 + (−0.009712) = **−0.007590**

实际:
- M50 dev = 0.861974
- Δ = M50 − M22 = **−0.006645**
- 预测差: −0.006645 − (−0.007590) = **+0.000945** (实际比预测好)

→ **轻微互补 ~+0.001 dev** (UE split 在 weekend filter 后增量 +0.003067 · 略大于 M49 的 +0.002122)
→ 但 weekend filter 主导损害 (−0.0097) 远大于互补 (+0.001) · 净损害

---

## 4. M50 LB 估算 · 决断

### 4.1 简单 dev→LB 映射 (悲观)

历史 gap = M22 dev (0.8686) − M22 LB (0.8476) = 0.0210

M50 LB 估算 (assuming gap ~0.021):
- M50 dev 0.861974 → LB ~**0.840974**

→ **远低于 M22 LB** · M22 SOTA 0.847592 · gap = −0.0066 LB

### 4.2 修订 hypothesis · "train/test 同质性增益"

**EXP-058 §Error A** 推论 weekend 是真信号 · 但**真实机制可能不是** "周末 = noise":
- weekend data 含 cross-day temporal pattern · 含信号
- 但 weekend data 与 test (03-04 周三) **分布显著不同** (CVR 5×)

**Hypothesis B (反 hypothesis A)**:
- M50 train (工作日 only) 与 test (03-04 周三 工作日) **更同质**
- 这种 "train/test 同质" 增益**只在 LB 上体现** · 在 dev 上反而被 data 减少损害掩盖
- LB 上可能反向**涨 +0.001~+0.002** vs M50 dev gap 推断

类比依据:
- M41 (temporal_weight 2.5) 同样让 train 偏向工作日 · 但**没减 data**
- M41 dev −0.000550 → LB **+0.000254** (5× 反向放大)
- 同方向偏移 + 数据完整 · 真信号机制 ✓

但 M50 的不同点:
- **真减 data 7.2%** (M41 是 reweight 不减)
- data 减少损害可能 dominant · LB 仍负

### 4.3 决断 · M50 LB 加入 5/17 早 quota 候选

**之前 (CEST 14:00 判断)**: 不评 M50 LB · 节省 quota
**修订 (CEST 16:00 用户反问后)**: M50 LB 加入候选 · 优先级 ⭐⭐

理由:
1. dev → LB 不一定线性 (M41 反例已证)
2. M50 是 "weekend filter 在 path B 加持下是否反向 work" 的唯一数据点
3. quota 不评 → 永远不知道答案 · 可能错过真信号

**风险**:
- M50 LB 大概率 < 0.847 (悲观估算 0.841 · 中性 0.844 · 乐观 0.848)
- 即使乐观 0.848 也仅持平 M22 · 不破 SOTA
- ROI: 中等 (信息价值 + 0.7 quota 成本)

---

## 5. 5/17 早 4 quota 修订矩阵 (CEST 16:00 终版)

| Slot | 候选 | 期望 LB | 优先级 | 理由 |
|---|---|---|---|---|
| 1 | **M37** (NLIR · vocabtier) | 0.846~0.849 | ⭐⭐⭐ | EXP-058 反推 architecture-level 不依赖 train stat · 类 M49 候选 |
| 2 | **M38 v2** (Multi-Emb K=2) | 0.846~0.847 | ⭐⭐⭐ | RankUp 闭环 · architecture-level |
| 3 | **M50** (UE split + weekend) | 0.840~0.848 | ⭐⭐ | 验 path B + daypart 互补 / "train/test 同质" hypothesis |
| 4 | **M30** (Trick 6 · Q_init) | 0.846~0.849 | ⭐⭐ | architecture-level · stack candidate |

**Backup**: M27-v2 (T31+user_15) / M34 (vocabtier+Trick 6) / M17 (seq_d=768)

**不评**:
- ❌ M45 LB (dev 灾难 −0.0097 · 单变量 weekend filter 已确证伤害)
- ❌ M48 LB (dev −0.066 · data 不足)
- ❌ M22 cross-seed (5/22 前铁律锁 · M49 LB 0.848 < 0.8484 不触发例外)
- ❌ M39 v2 (Global Token 与 UE split slot 冲突 · 大概率持平 M22)

---

## 6. 经验总结

### 6.1 用户反问触发的纪律强化

**铁律强化 · agent 必须 grep source-of-truth 才能下判决**:
- 用户反问 "train/valid 是否一致" → agent 立即 grep src/dataset.py timestamp_min 流向
- 不凭印象 (印象认为 weekend filter 不影响 valid · 必须 source-of-truth verify)
- 即使印象正确 · grep 也不亏 (~30s 时间 · 强化判读 confidence)

### 6.2 dev → LB 映射不可线性外推

| Job | dev Δ | LB Δ | 比率 |
|---|---|---|---|
| M22 | 0 (anchor) | 0 | 1× |
| M49 (UE split) | +0.002122 | +0.000570 | 27% retain |
| M41 (temporal_weight) | −0.000550 | +0.000254 | **−46% (反向)** |
| **M50 (双 stack)** | −0.006645 | ? | 待评 |

→ dev 大幅下降的方法 LB 大概率同向下降 · 但**反例存在** (M41) · quota 充足时应验证

### 6.3 EXP-057 期望 dev 估算错误 · 教训

EXP-057 line 69 我自己写过:
> "**期望 dev**: 持平 M22 (~0.8685 ± 0.0005) · 因 valid set 5% 尾部 (主要 03-04 末段) 不受影响"

**错在哪**: valid set 不受影响 ≠ dev 数据不变 · dev 反映的是 "model on valid 的 AUC"
- model 由 train data 训练而成 · train data 减少 → model 信息量减少 → dev AUC 降
- 我把 "valid set 不变" 错误外推为 "dev 不变"

**修正**: 未来对 train data 减少类实验的 dev 期望必须考虑 model 信息量损失 · 不能只看 valid set 是否变。

---

## 7. References

- ADR-014 (UE split path B · M49 实证后 Accepted)
- EXP-052 (M22 LB 0.847592 SOTA)
- EXP-057 (M45-M48 daypart batch · M45/M48 dev 灾难闭环 · 已死)
- EXP-058 (8 大错误 audit · §Error A weekend filter 真信号机制反思)
- EXP-059 (M49 LB 0.848162 · path B 弱真信号 · 项目首个超 SOTA)
- src/dataset.py:1761-1762 (PCVRParquetDataset.timestamp_min 行级过滤实施)
- src/dataset.py:3093-3094 (user_train_timestamp_min → train_timestamp_min mapping)
- src/dataset.py:3327-3337 (valid_dataset 构建 · timestamp_min=None 不过滤)
- platform-state.md (5/17 早 4 quota 候选矩阵)
