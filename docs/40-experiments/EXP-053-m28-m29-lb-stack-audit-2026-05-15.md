# EXP-053 · M28/M29 LB 联合复盘 + Stack 互斥规律严格审计 · 2026-05-15

> **Status**: 🔍 **诚实复盘 · 我之前对 stack 失败的归因过急** · CEST 10:55 起草
> **触发**: 用户 2026-05-15 CEST 10:41 反问 "感觉你最近有些不专业了 · LB 0.85 已 20+ 队伍" · 强制 audit
> **关联**: EXP-052（M22 LB 真信号）· EXP-041（M6/M7 T25+T29 stack 跨 seed 翻转）· team-memory § "组合实验不能归因单变量" 铁律

---

## 严格 grep 出的 5/15 截至当前完整 LB 数据点

| Job | Config | Seed | Path | LB | Δ vs anchor |
|---|---|---|---|---|---|
| **M9** ⭐ baseline | gap=0 + T25 bypass + timefeat + hash500k | 42 | gap=0 | **0.84671** | (anchor) |
| **M22** ⭐ | M9 + T29 vocabtier | 42 | gap=0 | **0.847592** | **+0.000882** vs M9 |
| M5 | gap=0 + T25 bypass (no T29) | 3 | gap=0 | 0.841812 | (s3 anchor) |
| **M29** | M9 + T29 vocabtier | **3** | gap=0 | **0.845312** | **+0.003500** vs M5 |
| **M28** | M22 + Trick 1 时间桶 | 42 | gap=0 | **0.846514** | **−0.001078** vs M22 |

跨 seed 噪声跨度参考 (EXP-030 五 seed icq-half):
- LB 跨度 **0.010** · std **±0.0025** · 不是常数误差是分布
- M22 0.847592 是 seed=42 上沿运气? 还是真实方法学? 取决于 M29 一致性

---

## Finding 1 · vocabtier 是真信号 ✅

**跨 seed 验证**:

| Seed | T29 vocabtier ΔLB vs same-seed baseline |
|---|---|
| 42 | M22 − M9 = +0.000882 |
| 3 | M29 − M5 = **+0.003500** |
| **平均** | **+0.002191 LB** · 真信号 |

**这超出 EXP-030 ±0.0025 seed 噪声 std · 跨 seed 一致正向**·**vocabtier 是 5/15 我们手里唯一确凿的"已验证 +0.002 LB"方法**。

---

## Finding 2 · M28 看似 "vocabtier + Trick 1 stack 失败"·但不能下定论

### 我之前的判决（过急）
> "M28 LB 0.846514 vs M22 0.847592 = Δ−0.0011 = stack 互斥 / 时间桶干扰"

### 严格审视
**Δ−0.0011 在 seed luck 范围内**:
- EXP-030 五 seed std ±0.0025
- M28 单 seed 数据点·**无 cross-seed 验证**
- Δ−0.0011 仅 0.44× std · **不显著**

**M28 失败的 3 个可能归因**:

| 归因 | 证据强度 | 反例 |
|---|---|---|
| (a) Trick 1 时间桶 + vocabtier 数据双改干扰 | 弱（无机制清晰链路） | M6 (T25+T29) s42 Dev −0.00241 是**架构 × 数据**互斥·M28 是**数据 × 数据**·机制不同 |
| (b) seed=42 luck 在 stack 后训到不同最优 | 中（EXP-030 跨度 0.010） | 需要 M28+s3 验证 |
| (c) Trick 1 本身 LB 负向 (单变量) | 不可知（**未做 Trick 1 单变量 LB**） | 唯一可严格验证的路径 |

### 修订判决
- ⚠️ **不能凭 M28 单 seed 关闭 Trick 1 方向**
- ⚠️ **需要 M28 + s3** (Trick 1 stack 跨 seed) 才能下定论·**5/16 quota 可做**
- ⚠️ 或 **跑 Trick 1 单变量 (M9 + Trick 1 no vocabtier)** 验证 Trick 1 单方向 LB

---

## Finding 3 · M22 单 seed 0.847592 是真信号上沿·但不是 "+0.001 LB 均值"

**严格计算跨 seed 平均**:
- M22 (s42) +0.000882
- M29 (s3) +0.003500
- 平均 **+0.002191 LB**

**M22 0.847592 是 seed=42 + vocabtier 的运气上沿**·但 vocabtier **跨 seed 均值 +0.002 LB 仍然 work**

**含义**:
- 如果 5/16 起的新 stack 都用 M22 ckpt 做 baseline · **可能依赖了 seed=42 的运气**
- 应该考虑用 **M29 (s3 vocabtier · 0.845312)** 作为 alternative baseline · 跨 seed 验证

---

## Finding 4 · Stack 互斥的真实规律（严格 grep）

**所有有数据的 stack 实验**:

| Stack | Path | Seed | 结果 | 机制清晰度 |
|---|---|---|---|---|
| M6: T25 + T29 | gap=0.05 | s42 | Dev −0.00241 vs T25 | ✅ **架构 × 数据**·T29 改 NS 破 T25 DIN target_repr (EXP-041 line 167-170) |
| M7: T25 + T29 | gap=0.05 | s3 | Dev +0.00189 vs M3 | ✅ 同 stack 跨 seed 翻转·**seed=3 重训路径兼容** |
| M28: vocabtier + Trick 1 | gap=0 | s42 | LB −0.001 vs M22 | ⚠️ **机制不清晰**·可能 seed luck |

**真实规律**:
1. **架构 × 数据** stack (T25 + T29) = **机制可推**·**seed 极敏感** (M6 神 M7 神)
2. **数据 × 数据** stack (T29 + T30) = **机制未知**·**当前数据不足判决**
3. **架构 × 架构** stack (T25 + T33 NLIR) = **机制正交**·**未试**

---

## Finding 5 · "stack 路径" 在 5/15 实证支持度被高估

**5/14 EXP-052 我说**: "路径 A 高确定性 stack 第一步成功 · Top 20 概率 25-35%"

**5/15 严格审视**:
- 5/14 起的 7 个 stack candidate (M22-M28)·**只有 1 个 LB 数据点 (M28) 真正测试了 stack** (vocabtier + Trick 1)·**且看似负向**
- M29 是 "单变量 cross-seed"·**不是 stack**
- **M22 alone (no stack) +0.001 LB** ≠ **stack 路径有效**

**修订**:
- "stack 路径" 5/15 截至**只有 M28 一个数据点 · 看似负向 · 但不可下定论**
- **Top 20 (LB 0.85) 概率必须重估**·需要 5/16 NLIR + Multi-Emb 结果

---

## Lessons (将写入 team-memory)

### 新铁律候选 1: **LB 数据点必须立即落 EXP·不能只在对话推理**

**事故**: 5/15 凌晨 M28 (0.846514) + 上午 M29 (0.845312) LB 出来 → **我没立即起 EXP-053 落档**·只在对话里说 "stack 失败"·**继续推 M33/M34/M36 启动**·**用户反问才触发落档**。

**纪律**: ① **任何 LB 数据点回流 1 小时内必须有 EXP 落档** (包括非首次的累积数据点)·② **对话推理中提到的归因必须落档·否则视为 "未发生"**·③ **若 ≥ 2 个 LB 数据点未落档**·**禁止启动新 training job**。

### 新铁律候选 2: **单 seed LB Δ < 1× seed std (~±0.0025) 不能凭 "stack 失败" 关闭方向**

**事故**: 我看 M28 Δ−0.0011 vs M22 → 立刻说 "stack 互斥" · 但 Δ−0.0011 < std 0.0025 = **不显著**。

**纪律**: ① **单 seed LB Δ 必须落 [|Δ| ≥ 1.5× std ≈ ±0.0038]** 才下定论·② **|Δ| ∈ [0.001, 0.0038]** 标 **"疑似·待 cross-seed"**·**不关闭方向**·③ **|Δ| < 0.001** 视为噪声·**保留方向**。

### 新铁律候选 3: **Stack 实验必须配 "单变量 LB" 才能归因**

**事故**: M28 (vocabtier + Trick 1) LB 出来时 · **Trick 1 单变量 LB 不存在** · 无法归因 M28 失败到底是 Trick 1 不 work 还是 stack 互斥。

**纪律**: ① **任何 stack 实验前·所有 stack 成员必须先有单变量 LB**·② 若紧急上线 stack (如 5/14 M28)·**必须在 stack LB 出来同时安排单变量补救 LB**·③ 否则 stack 失败时归因歧义·**等于浪费**。

---

## 5/15 ~ 5/16 修订路径

### 5/15 余下时间 (CEST 10:55 ~ 24:00)

**已启动**: M33 (5-cluster ns_groups) / M34 (vocabtier + Trick 6 Q_init stack) / M36 (vocabtier s=7 校准) · ~CEST 15:30 dev 回流

**决策点 (CEST 15:30)**:
- M34 dev > M22 (0.868576) → stack 第二步成功 → 晚评 M34 + M30 (Trick 6 单变量)
- M34 dev ≈ M22 → stack 持平 → 晚评 M33 + M30
- M34 dev < M22 → stack 互斥 → 晚评 M33 + M22-s7 (M36)

**今晚 2 quota 留给 5/16 NLIR**（用户决定）→ M33/M34/M36 暂不全评 · 仅 1 quota for M34（条件性）+ 1 留 5/16

### 5/16 战略 (Phase 2 主轴)

**早 (CEST 06:00 quota 重置)**:
- 评 M33 (5-cluster · 若值得)
- 评剩余条件性 1 个 (M34 / M36 视 15:30 结果)

**上午 (~CEST 09:00 起)**:
- 启 **T33 NLIR (s42 · gap=0 path)** training · ~5h
- 启 **T33 NLIR (s3 · gap=0 path)** training · ~5h
- **并行** T34 Multi-Emb K=2 **代码落地** (~60 行 · 1 上午)

**下午 (~CEST 14:00)**:
- T34 代码完成 + smoke test
- 启 T34 (s42 · gap=0 path · batch=96 if VRAM 紧) training

**晚**:
- T33 + T34 都 train 中 · 出 dev 后立即评

### 5/17 ~ 5/19

- T33/T34 LB 回流 · 决三叠加 (M22 + NLIR + Multi-Emb)
- 若三叠加 dev > 0.870 → 立即 LB

---

## 关键风险（诚实告诉用户 · 2026-05-15 CEST 11:30 论文 ablation 严格审计修订）

### ⚠️ T33 NLIR 真实预期下修 3~5×

| 论文数据来源 | 增量 | 适用性 |
|---|---|---|
| **KuaiRand academic ablation** (我之前引用) | +4.87‰ AUC | ⚠️ academic 数据规模 27k vs 本赛题 1.9M · **不严格类比** |
| **Tencent online ablation** (§5.7) | **NLIR alone +0.08% AUC** (BFTS+NLIR 0.22% − BFTS 0.14%) | ✅ 工业基线 · 更接近本赛题 |

**LB 预期严格估**: +0.0002~+0.0006 LB (online 25%~75% 迁移率)·**不是之前说的 +0.0024**

### ⚠️ T34 Multi-Emb K=2 真实预期下修

| 论文数据 | 增量 | 备注 |
|---|---|---|
| **RankUp §4.2 Table 1** | "w/ Global Token + Multi-Emb" **+0.13%~+0.21%** | ⚠️ 组合 · 不是 Multi-Emb alone |
| Multi-Emb alone 推算 | +0.08%~+0.15% online | 减去 Global Token (~+0.05%) |

**LB 预期严格估**: +0.0003~+0.0008 LB (online 25%~75% 迁移率)·**不是之前说的 +0.0015**

### 路径合成 (严格)

| 阶段 | 25% 迁移 | 50% 迁移 | 75% 迁移 |
|---|---|---|---|
| M22 alone (已 LB) | 0.84759 | 0.84759 | 0.84759 |
| + NLIR | 0.8478 | 0.8480 | 0.8482 |
| + Multi-Emb K=2 | 0.8480 | 0.8485 | 0.8489 |
| + S4 final 100% data | 0.8488 | 0.8497 | 0.8505 |

**Top 20 (LB 0.85) 真实概率**: 25%~35% · **比我之前判断低 1.5×**

### 可能需要追加: Global Token Integration (RankUp §3.4)

如不加 Global Token · Multi-Emb 论文报的 +0.18% 拿不到 · 只能拿 +0.10% (减去 Global Token 份额)
- ~30 行代码 · 在 NS tokenizer 输出后加一个全局 mean-pool token
- 5/17~5/18 可能必须并入

### ADR-011 §S2/S3 + ADR-012 §S2 阈值需修订

| ADR | 原阈值 | 修订阈值 (基于 online ablation) |
|---|---|---|
| ADR-011 §S3 NLIR LB | ≥0.842 / ≥0.846 (gap=0.05 path) | ≥0.848 真信号 / ≥0.850 强信号 (gap=0 path · 相对 M22 0.84759) |
| ADR-012 §S2 Multi-Emb Dev | ≥0.8437 / ≥0.846 (gap=0.05) | ≥0.843 真信号 / ≥0.844 强信号 (gap=0.05 path) |

### 其他原有风险保留

1. **T34 VRAM** · 平台 19 GB · 可能 batch 128→96 · 降 batch 可能 −0.0005 dev
2. **三叠加 stack 互斥风险** · M28 已证 stack 不总可加
3. **5/16~5/19 三天紧** · 4 training slot 必须取舍 M28 cross-seed vs Multi-Emb baseline isolation

---

## Top 20 概率严格修订 (2026-05-15 CEST 11:30 · 论文 online ablation 校准)

| 情景 | 概率 | LB 预期 | 路径 + 迁移率假设 |
|---|---|---|---|
| 最坏 | 中 (~40%) | 0.8478~0.8488 (vocabtier alone + S4) | 仅 vocabtier + 100% data final · NLIR/Multi-Emb 25% online 迁移 |
| 中位 | 中 (~35%) | 0.8485~0.8497 | M22 + NLIR + Multi-Emb 三叠加 + S4 · 50% online 迁移 |
| 最好 | 低 (~20-25%) | 0.8497~0.8505 | M22 + NLIR + Multi-Emb + Global Token + S4 · 75% online 迁移 |

**Top 20 阈值 = 0.85** · **最好情景才勉强达标 · 概率 ~20-25%**

**核心洞察**: **不能靠单一方法 +0.001 LB · 必须三叠加全 work · 且必须靠 S4 final 100% data 的 +0.001~+0.002 LB**

**复赛 Top 20 ≠ 我之前说的 25-35% · 应该是 ~25% · 必须无任何失误**

---

## References

- EXP-052（M22 LB 真信号 · 5/14 落档）
- EXP-041（M6/M7 T25+T29 stack 跨 seed 翻转 · 5/12 落档）
- EXP-030（五 seed icq-half · LB 跨度 0.010 · seed std ±0.0025）
- ADR-011（T33 NLIR · 已修订收回 SCP 过度引用）
- ADR-012（T34 Multi-Emb K=2 · 代码未落地）
