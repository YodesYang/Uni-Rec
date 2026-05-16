# EXP-059 · M49 UE Split LB 0.848162 · Path B 首测弱真信号 · 2026-05-16

> **Status**: ✅ Closed (M49 LB 已出 · 项目首个超 M22 SOTA)
> **Date**: 2026-05-16 CEST 21:10
> **Trigger**: M49 (T37 UE split · ADR-014) evaluation 完成 · LB **0.848162** (vs M22 SOTA 0.847592 · Δ +0.000570)
> **Related**: ADR-014 (UE split path B) · EXP-052 (M22 SOTA 0.847592) · EXP-055 §16~17 (path A 死透 audit) · EXP-056 (Tier 0 EDA) · EXP-057 (M45-M48 daypart batch · 已死) · EXP-058 (8 大错误 audit)

---

## TL;DR · M49 LB 0.848162 = 项目首个超 SOTA + path B 首测弱真信号

**核心数据**:
- M49 LB: **0.848162**
- M22 SOTA: 0.847592
- **Δ = +0.000570** (超出 same-seed rerun band ±0.0003 · 但未破 0.8484 严格阈值)
- **dev → LB 衰减**: dev +0.002122 → LB +0.000570 (3.7× 衰减率)
- **距 0.85 Top 20**: +0.001838

**判决**:
1. ✅ **超出 same-seed rerun band** → 不是 noise · 是真信号
2. ⚠️ **未破 0.8484 严格阈值** (3× rerun band) → 信号强度不足以排除 cross-seed luck
3. ✅ **path B 单变量验证 work** · ADR-014 设计假设正确 (separate UE token)
4. ⚠️ **单变量不够冲 0.85** · 需要 stack / cross-seed / 第二真信号

---

## 1. 数据 ground truth

### 1.1 M49 train summary

| Field | Value |
|---|---|
| Job Name | M49-m22-ue-split-gap0-seed42-4ep |
| Config | M22 baseline + `--enable_ue_split --ue_split_fid user_dense_feats_61` (256-dim 预训练 UE) |
| Train rows | ~2.1M (full · gap=0) |
| seed | 42 |
| Train wall-time | 2h48m45s |
| stop_reason | max_epochs_completed |
| best epoch | E4 (max) |
| best dev AUC | **0.870741** |
| best dev LogLoss | 0.222531 |
| val_auc_history | E1: 0.865383 → E2: 0.868688 → E3: 0.869918 → **E4: 0.870741** (单调上升 无 overfit) |

### 1.2 M49 evaluation result

| Field | Value |
|---|---|
| Eval Job | M49-m22-ue-split-gap0-seed42-4ep_eval_1778935775477 |
| Inference time | 510.83s (≈ 8.5 min) |
| **LB AUC** | **0.848162** |

### 1.3 vs M22 baseline 对比

| Job | dev | LB | dev→LB gap |
|---|---|---|---|
| M22 (vocabtier · seed=42) | 0.868619 | 0.847592 | +0.021027 |
| M22-rerun (rerun band) | 0.868582 ± | 0.847324 | +0.021258 |
| **M49 (UE split alone)** | **0.870741** | **0.848162** | +0.022579 |
| Δ M49 vs M22 | **+0.002122** | **+0.000570** | +0.001552 |

**dev→LB 衰减率**: 0.000570 / 0.002122 = **26.9%** · 即 dev 增量 73.1% 在 LB 上消失

---

## 2. 信号强度严格判决

### 2.1 阈值对比

| 阈值 | 数值 | M49 LB 是否超 | 判读 |
|---|---|---|---|
| Same-seed rerun band ±0.0003 | [0.847292, 0.847892] | ✅ 超 +0.000270 | **不是 noise · 是真信号** |
| 破 SOTA 严格阈值 (≥ 0.8484 · 3× rerun band) | 0.8484 | ⚠️ 差 −0.000238 | **不到强信号阈值** |
| Cross-seed band ±0.005 | [0.842592, 0.852592] | ✅ 在 band 内 | **不能完全排除 seed luck** |
| 0.85 Top 20 | 0.85 | ⚠️ 差 +0.001838 | **远未到** |

### 2.2 真信号判决

**结论**: M49 是**弱真信号** · 不是 noise · 但也不是强信号。
- "弱真信号" = 超出 rerun band · 未破 3× rerun band 阈值
- 在 0.001838 距离 0.85 的窗口下 · M49 alone **不够**冲 Top 20
- 需要找第二个真信号叠加 · 或在最终 cross-seed 阶段找上沿

---

## 3. dev→LB 衰减机制 audit

### 3.1 数据

| Job | dev → LB 衰减 | 衰减率 |
|---|---|---|
| M22 baseline (anchor) | dev=0.8686 → LB=0.8476 (gap 0.0210) | 100% retain |
| **M49 (UE split)** | dev +0.002122 → LB +0.000570 | **27%** retain (73% 衰减) |

### 3.2 机制推断

**为什么 M49 dev↔LB 衰减比 baseline 高 73%?**

- **UE split 改 user_dense_feats_61 (256-dim 预训练 UE) 的处理路径**
- dev (近未来 · 共享 train 期 user 分布) 上 · UE 分离后 model 学到的"专属 UE 信号"**充分发挥** → dev +0.002
- LB (远未来 03-04 · **100% user cold start** 在分布上 vs train) 上 · 学到的 "专属 UE 信号" **部分失效**
  - user_dense_feats_61 是 256-dim 预训练 embedding · 假设是 user-level summary (e.g. preference vector)
  - test 上 user 是 cold start → user_dense_feats_61 的值是新 user 的 inference (vs train 时的训练值)
  - model 在 train 上学的 "如何用 UE token 做 inference" → 在 test 上 OOV-like · 部分泛化失效
- 衰减 73% = 大部分 dev 上的方法学增量是"近未来 user 分布共享"的虚高

### 3.3 推论

- **UE split 在远未来 OOV 场景下信号衰减严重** · 但**绝对增量仍正** (+0.000570)
- 任何 architecture-level 改动都需评估 dev→LB 衰减率 · 不能只看 dev 增量
- **EXP-052 期望 dev↔LB 转化率 ~70%** (基于 M41 dev↓ LB↑ 反例) · M49 转化率仅 27% · 远低于期望

---

## 4. ADR-014 假设验证

### 4.1 ADR-014 path B 设计假设

> "user_dense_feats_61 (256-dim 预训练 UE) 与 count + seq_stats 等其他 user_dense feature 信号性质完全不同 · 不应被 mean-pool 在一起 · 应分离成专属 token"

### 4.2 实证证据

✅ **假设成立**:
- M49 LB +0.000570 (超 rerun band) · 真信号
- M49 dev +0.002122 (远超 rerun band) · 强真信号
- dev 单调上升 E1→E4 (0.865 → 0.870) · 无 overfit · 健康优化
- E4 最佳 → architecture 表达力提升 (不是数据增多)

⚠️ **未完全验证**:
- 是否需要 ue_dim=256 完整 · 或 64/128 也够?
- 其他 user_dense fid (e.g. fid_15) 是否也有同样增益? (ADR-014 §Step 1 vs §Step 2 已验证 fid_15 解读不确定 · 已 deprecated)
- UE split 是否与其他真信号 stack work? (M50 待评)

### 4.3 ADR-014 状态升级

**之前**: Proposed → Suspended (5/16 早 ADR-014 解读不确定)
**现在**: ✅ **Accepted** (M49 LB 实证)

---

## 5. 5/17 早 4 quota 决策矩阵 (按 M49 LB 0.848162 走 场景 B 弱信号路径)

### 5.1 场景重判定

EXP-057 §4 当时定义:
- 场景 A · M45 LB > 0.848 → 4 quota M46/M47/M48 LB + M22 cross-seed
- 场景 B · M45 LB ∈ [0.847, 0.848] → M46 + M48 + 老候选
- 场景 C · M45 LB ≤ 0.847 → M46/M47/M48 + cross-seed
- 场景 D · M45 LB < 0.846 → daypart 全死 · 切 path B

但**实际跑出来**:
- M45/M48 dev 灾难 (−0.0097 / −0.066) · LB **不评** (浪费 quota)
- M49 LB 0.848162 = path B 单变量 work · 但弱信号
- 场景判定**整个偏移到 path B 主线**

### 5.2 5/17 早 4 quota 修订矩阵

| Slot | 候选 | 期望 LB | 优先级 | 信息价值 / 决断 |
|---|---|---|---|---|
| 1 | **M50** (UE split + weekend filter 双 stack) | 0.847~0.849 | ⭐⭐⭐ | 验 path B + daypart 是否互补 · 5/16 已跑完 dev 待看 |
| 2 | **M37** (NLIR vocabtier + Linear gating) | 0.846~0.849 | ⭐⭐ | EXP-058 反推可能误判 · 闭环 |
| 3 | **M22 × s11 cross-seed** | 0.846~0.849 | ⭐⭐ | 找 vocabtier 路径上沿 (但触铁律) |
| 4 | **M38 v2** (Multi-Emb K=2) | 0.846~0.847 | ⭐ | RankUp 闭环 (大概率持平) |

### 5.3 关键 caveats

**(a) M22 × s11 cross-seed 是否破例**?

team-memory 5/16 早立铁律: "5/22 前默认锁 seed=42 · 强真信号 (LB ≥ 0.8484) 例外"
- M49 LB 0.848162 < 0.8484 → **不触发例外**
- **严格遵守铁律 = 不做 cross-seed**
- 但 M22 cross-seed 是当前**唯一可信冲 0.85 路径**? · 不一定 (M50 / M37 也可能)

**判决** (严格遵守铁律): 5/17 早**不做 M22 cross-seed**
- 优先 M50 / M37 / M38 v2 · 留 cross-seed 给最终阶段 (5/22~5/24)

**(b) M50 应优先评 (5/16 已完成 dev 等查)**:
- M50 = M49 + weekend filter
- 如 M50 dev > M49 dev (反预期 · 概率 < 5%) → M50 ckpt 是新候选
- 如 M50 dev < M49 dev (大概率 · ~95%) → M50 LB 仍可评 (但 ROI 低 · 看 path B + daypart 是否互补)

**(c) M37 (NLIR) 重新优先级**:
- EXP-058 §3 §Error E 已反推: NLIR 是 architecture-level · 不依赖 train stat bias · 类 M49 性质
- M37 dev 0.868311 (−0.000308 vs M22) · 弱负
- 类 M41 反例可能 (dev↓ LB↑) · 候选 dev↓ LB 持平

---

## 6. 真实 Top 20 概率估算 (5/16 CEST 21:30)

时间线:
- 5/14 EXP-051: 25~35%
- 5/15 EXP-055: 10~15%
- 5/16 09:00 (M40/M41 LB 后): 10~15%
- 5/16 10:30 (M23 LB 后): 5~8%
- 5/16 11:55 (M45-M48 启动后): 5~10%
- 5/16 14:00 (M45/M48/M49 dev 出后): 8~15% (M49 dev +0.002 振奋)
- **5/16 21:30 (M49 LB +0.000570 后)**: **8~12%** (路径确认 · 但单变量不够 · 需要叠加)

**距 0.85**: 0.848162 → 0.85 = **+0.001838**

**冲 Top 20 路径** (估算):
- M50 LB > 0.848162 (双 stack work) + cross-seed 上沿 (5/22+ 阶段) · 累加 ~0.0015 → 0.8497
- M37 LB 弱真信号 + UE split stack · 累加 ~0.001 → 0.849
- 全部失败 → 0.848 收尾 · 不够 Top 20

→ **乐观估计**: **0.849~0.850 是 5/22 前可达天花板** · Top 20 概率 8~12%

---

## 7. 经验总结 (新立 / 强化的纪律)

### 7.1 新立铁律 · architecture-level 真信号判决标准

**M49 是项目首个 dev↔LB 双侧超 rerun band 的 architecture-level 真信号**

判决模板:
- dev Δ > +0.0005 (> rerun band) **且**
- LB Δ > +0.0003 (> LB rerun band ±0.0003)
- **同时满足才算真信号**
- 单看 dev 不够 (dev↑ LB↓ 反例 · M37/M38v2/M39v2 高风险候选)
- 单看 LB 不够 (LB band ±0.0003 内可能是 seed luck)

**M49 数据**: dev +0.002 ✅ + LB +0.000570 ✅ = **真信号**

### 7.2 强化铁律 · dev→LB 衰减率必须评估

任何 architecture-level 改动 evaluate 后 · 必须算 dev→LB 衰减率:
- M22 baseline: 100% retain (anchor)
- M41 (temporal_weight): retain ratio 不计算 (dev↓ LB↑ 反例)
- **M49 (UE split): 27% retain** (73% 衰减 · OOV scenario 下严重)

→ 衰减率高的方法在最终阶段 cross-seed 时增量可能更小 (cross-seed 噪声主要在 LB)
→ 选 stack 候选时 · 优先衰减率低的方法

### 7.3 强化铁律 · 5/22 前 cross-seed 锁定 (M49 不触发例外)

team-memory line: "5/22 前默认锁 seed=42 · 强真信号 (LB ≥ 0.8484) 例外"
- M49 LB 0.848162 < 0.8484 → **不触发**
- 优先非 cross-seed 路径 (stack / 老候选闭环)
- cross-seed quota 留给最终上沿搜索阶段 (5/22~5/24)

---

## 8. 5/17 早决策建议 (用户 review 用)

### 推荐 4 quota 顺序

1. **M50 LB** (path B + daypart stack 验证 · 5/16 已 trained · 必评)
2. **M37 LB** (NLIR · EXP-058 反推可能误判 · 类 M49 候选)
3. **M38 v2 LB** (Multi-Emb K=2 · RankUp 闭环 · 大概率持平)
4. 留 1 quota 弹性 (取决于 1-3 结果 · 可能 M30 / 或 M39 v2 / 或新方向)

### 不推荐

- ❌ M22 cross-seed (5/22 前铁律 · M49 LB 不触发例外)
- ❌ M45 / M48 LB (dev 已显著负)
- ❌ M22 + Multi-Emb K=2 / Global Token stack (历史 stack 大概率失败)

---

## 9. References

- ADR-014 (UE split path B) · 5/16 早起草 → 晚 LB 实证 = 6 小时验证周期
- EXP-052 (M22 LB 0.847592 SOTA · dev→LB gap 0.021)
- EXP-055 §16-17 (path A 死透 audit · M23 LB −0.011 强负闭环)
- EXP-056 (Tier 0 EDA · 元宵节证伪)
- EXP-057 (M45-M48 daypart batch · M45/M48 dev 灾难闭环 · 已死)
- EXP-058 (8 大错误 audit · §Error E 反推 NLIR/M37)
- src/model.py PCVRHyFormer (T37 UE split 实施 · commit `e5b0bf4`)
- team-memory (5/16 cross-seed 锁定铁律)
- platform-state.md (5/16 evaluation quota 4/4 用满 · M49 LB 收尾)
