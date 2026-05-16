# EXP-061 · M51-M54 Path B Stack Overnight Batch · 2026-05-16

> **Status**: 🔵 Documented · 🟡 4 jobs to launch ~CEST 20:30 5/16
> **Date**: 2026-05-16 CEST 16:30 起草
> **Trigger**: M49 LB 0.848162 path B 单变量已验证真信号 + 4 free slot + 5/17 早 quota 不缺老候选 → 今晚 4 slot 全力做 path B + 第二真信号 stack 探索
> **Related**: EXP-059 (M49 LB · path B 弱真信号) · EXP-060 (M50 双 stack 闭环) · EXP-058 (8 大错误 audit · §Error C per-domain bucket 反推) · ADR-014 (UE split path B · upgrade pending)

---

## TL;DR · Path B 体系下首次 stack 探索 · 4 jobs overnight

**核心战略**:
- Path B (UE split alone · M49 LB +0.000570) 已验证真信号
- 距 0.85 Top 20 还差 +0.0018 · 单变量 path B 不够 · 必须找第二真信号叠加
- 5/17 早 quota 不缺老候选 (M37/M38v2/M30/M27v2 都 publish-ready) · 不与新候选竞争
- → 今晚 4 free slot **全力做 path B + 第二真信号 stack** · 5/17 早 quota matrix = "2 老 (M37/M38v2) + 2 新 (M51-M54 dev 最高 + M50)"

**4 jobs 配置**:

| Job | Config | 类型 | 期望 dev | 期望 LB | 真信号概率 |
|---|---|---|---|---|---|
| **M51** | M22 + UE split + temporal_weight 2.5 | path B + M41 stack | 0.871~0.873 | 0.848~0.850 | 40% (高) |
| **M52** | M22 + UE split + per-domain time bucket | path B + EXP-058 §Error C 修复 | 0.870~0.872 | 0.847~0.849 | 25% (中) |
| **M53** | M22 + UE split + NLIR | path B + Linear gating stack | 0.870~0.872 | 0.847~0.849 | 25% (中) |
| **M54** | M22 + per-domain time bucket alone | M28 修复单变量 (无 path B) | 0.868~0.869 | 0.846~0.848 | 30% (中) |

**期望产出** (1 ~ 2 真信号):
- 真信号期望值 = 40% × 1 + 25% × 2 + 30% × 1 = **1.2 真信号**

---

## 1. 候选选择 · 严格 audit

### 1.1 已知真信号 (5/14~5/16 LB 实证)

| Job | LB | Δ vs M22 | 类型 |
|---|---|---|---|
| **M22** (vocabtier) | 0.847592 | (anchor) | feature engineering |
| **M41** (M22 + temporal_weight 2.5) | 0.847846 | +0.000254 | dev↓ LB↑ 反例 |
| **M49** (M22 + UE split) | 0.848162 | **+0.000570** ⭐ | architecture (path B) |

→ 当前 LB 上沿 0.848162 · 距 0.85 = +0.001838

### 1.2 候选机制 (按真信号概率排序)

#### M51 = M49 + temporal_weight 2.5 (40% 真信号概率) ⭐⭐⭐⭐

**机制**: 两个**已验证真信号**的 stack
- M49: UE split (architecture) 改 user_dense slot
- M41: temporal_weight (loss reweight) 让 train 偏向工作日 · 不减 data
- 两者**机制完全正交**: 一个改架构 一个改 loss 权重

**叠加期望**:
- 加性预期 LB: M22 + (+0.000570) + (+0.000254) = **0.848416**
- 实际取决于互补程度 (5/14~5/16 6/6 stack 第 2 步只有 M41 work · 失败率 83%)
- 但本次 stack base = M49 (而非 M22) · base 已变 · 历史失败率不直接外推

**风险**:
- M41 dev↓ LB↑ 反例机制依赖 train 期 minor reweight · 与 UE split 改架构是否互不冲突 不确定
- 可能 dev 落在 ~0.870 (微负) · LB 涨 ~0.0003

#### M52 = M49 + per-domain time bucket (25% 真信号概率) ⭐⭐⭐

**机制**: path B + EXP-058 §Error C 修复
- M49 UE split alone work · path B 已验证
- per-domain time bucket: EXP-056 §Q5 实证 · seq_d (p99=56天) vs seq_c (p99=629天) **量级跨 11×** · 全局 64-bucket 浪费/不细
- per-domain: 每 domain 独立 fit 64-bucket boundaries

**实施**: 需新代码 (T38)
- dataset.py time_bucket_boundaries 改 List → Dict[domain, List]
- train.py CLI 加 `--per_domain_time_buckets`
- infer.py cfg fallback
- ~50 LOC + smoke

**风险**:
- M28 (M22 + 全局 time bucket) LB 0.846514 vs M22 0.847592 = **−0.001078**
- 但 M28 是**全局** bucket · per-domain 是修复版 · 机制不同
- 期望: per-domain 修复 M28 失败的部分 · LB 可能持平/微正

#### M53 = M49 + NLIR (25% 真信号概率) ⭐⭐⭐

**机制**: path B + Linear gating stack (EXP-058 §Error E 反推可能误判 NLIR)
- M37 NLIR alone dev 0.868311 (−0.000308 vs M22) · LB 待评
- NLIR (Linear gating) 是 architecture-level · 不依赖 train stat bias
- M49 + NLIR = **path B (token 分离) + transformer 内部 gating** 双架构正交 stack

**风险**:
- M37 LB 大概率 0.846~0.847 (dev 弱负 · 大概率 LB 弱负或持平)
- 但 EXP-058 §Error E 反推 NLIR 可能 dev↓ LB↑ 反例
- M53 = M49 + NLIR · 如 NLIR 单独反例 → M53 = M49 + (+0.0003) ≈ 0.8485 (接近破 0.8484 阈值)

#### M54 = M22 + per-domain time bucket (30% 真信号概率) ⭐⭐

**机制**: M28 修复单变量 · 不依赖 path B
- 与 M52 共用 T38 per-domain bucket 代码
- 单独测试 per-domain bucket 是否真比全局 bucket 好
- 双重验证: M52 (path B + per-domain) vs M54 (per-domain alone) · 两者 dev/LB 关系揭示 path B 与 per-domain bucket 的互补性

**风险**:
- 单变量 LB 大概率 0.846~0.848 (持平 M22)

---

## 2. 实施计划

### 2.1 T38 · per-domain time bucket (M52/M54 依赖)

**改动文件**:
- `src/dataset.py`: time_bucket_boundaries List → Dict[domain, List] · ~50 LOC
- `src/train.py`: CLI `--per_domain_time_buckets` flag · ~20 LOC
- `src/infer.py`: cfg fallback · ~10 LOC
- `scripts/smoke_t38_per_domain_bucket.py`: 5-test smoke · ~80 LOC

**实施时间**: 2~3h (CEST 16:30 → 19:00)

**Risk**:
- 现有 time_bucket_boundaries 是 64-bucket 全局 · 改 per-domain 后需要 4× 64 = 256 bucket
- 影响: emb table size 增加 (256 vs 64) · 但 emb_dim 仍 = d_model = 64 · table size 仍可控
- Smoke test 必须 cover: ① 全局 bucket 仍 work (regression) · ② per-domain bucket dev 可计算 · ③ ckpt save/load · ④ infer.py cfg fallback · ⑤ ns_groups 不冲突

### 2.2 4 jobs run.sh 模板 (T38 完成后)

**M51 (path B + M41 stack)**: 不需新代码 · 仅 run.sh
**M52 (path B + per-domain bucket)**: 需 T38
**M53 (path B + NLIR)**: 不需新代码 · 仅 run.sh
**M54 (per-domain bucket alone)**: 需 T38

→ 实施顺序: 先 T38 · 再 4 jobs run.sh

### 2.3 启动时序

CEST 16:30 现在
→ T38 实施 + smoke (~2.5h · 完成 CEST 19:00)
→ 4 run.sh 模板 (~30 min · 完成 CEST 19:30)
→ 用户启 4 jobs (~CEST 20:30 · 给 30~60 min buffer)
→ 4 jobs ~5h training (CEST 20:30 → 01:30 5/17)
→ 5/17 早 dev 出 · 06:00 quota refresh

---

## 3. M52/M54 共用 T38 代码 · ROI 计算

**T38 实施成本**: 2.5h
**M52 + M54 双 job 收益**: 2 个候选 dev/LB 数据点
- 如 M52 dev > M54 dev → path B 与 per-domain bucket **互补** (两机制叠加)
- 如 M52 dev ≈ M54 dev → path B 与 per-domain bucket **不冲突也不互补** (各自独立)
- 如 M52 dev < M54 dev → path B 与 per-domain bucket **冲突** (反向解读)

**信息价值**: 极高 (3 种结果都决断方向)

---

## 4. 风险评估

### Risk 1 · stack 第 2 步历史 6/6 失败 (除 M41)

5/14~5/16 stack 数据:
- M22 + Trick 1 (M28): −0.001078
- M22 + 5-cluster (M33): −0.002882 (实施错)
- M22 + DCN cross (M23): −0.010961 ❌
- M22 + temporal_weight (M41): **+0.000254** ✅ 唯一 work
- M22 + hour_only (M40): −0.000661
- M22 + UE split (M49): **+0.000570** ⭐ (path B 已验证)

→ stack 第 2 步成功率 2/6 = 33%
→ 但 M51-M53 都基于 M49 base · base 已是 work 的真信号 · 可能成功率 > 33%
→ **保守估计 4 jobs 中真信号 1~2 个**

### Risk 2 · M52 per-domain bucket 实施风险

- emb table size 256 vs 64 · 显存增加可控
- 但 ns_groups_vocab_tier.json 配置可能与 per-domain bucket 冲突 (vocabtier 对 high-vocab item id 单独 chunk · per-domain bucket 是 time embedding · 应不冲突)
- Smoke test 必须 cover

### Risk 3 · 4 jobs 同时 GPU 共享 → 训练时间延长

- 单 job ~5h · 4 jobs 同时跑可能 6~7h (GPU 共享 · 但平台一般 1 job/GPU · 如分配充足应正常)
- 完成时间从 ~CEST 01:30 5/17 推到 ~CEST 02:30~03:30 5/17 · 仍在 quota refresh 06:00 之前

### Risk 4 · M50 ckpt 24h 过期 (CEST 5/17 15:53)

- M50 ckpt publish 状态待 verify (用户 5/16 16:25 没明确说)
- **必须立即 publish** 否则 5/17 下午前过期
- M50 LB 评估留 5/17 早 quota slot 4

---

## 5. 5/17 早 4 quota 决策矩阵 (CEST 16:30 final)

按 dev 选 4 个最强候选评估:

| Slot | 候选 | dev (待 5/17 早) | 优先级 | 备注 |
|---|---|---|---|---|
| 1 | **M37** (NLIR vocabtier) | 0.868311 (已知) | ⭐⭐⭐ | 老候选 · architecture-level · 类 M49 |
| 2 | **M38 v2** (Multi-Emb K=2) | 0.868403 (已知) | ⭐⭐⭐ | 老候选 · RankUp 闭环 |
| 3 | **M51 / M52 / M53** dev 最高的 | TBD ~CEST 03:30 5/17 | ⭐⭐⭐ | 今晚训完 dev 出 · 选最强 stack |
| 4 | **M50** (UE split + weekend) | 0.861974 (已知) | ⭐⭐ | 反例验证 · 或备用 M30 / M54 |

**判决决断**:
- 如 M51/M52/M53 dev 最高 ≥ 0.871 (强真信号) → 替换 slot 3 + 4 评 2 个新候选 (slot 4 = M51 或 M52 dev 第 2 高)
- 如 M51-M53 dev 都 ≤ 0.870 (中庸) → 仍按上面矩阵 · slot 4 = M50 反例验证

---

## 6. References

- ADR-014 (UE split path B · M49 实证后 Accepted)
- EXP-058 (8 大错误 audit · §Error C per-domain bucket 反推 · §Error E NLIR 反推)
- EXP-059 (M49 LB 0.848162 · path B 弱真信号)
- EXP-060 (M50 dev 闭环 · daypart filter 路径死)
- src/dataset.py:1761-1762 (PCVRParquetDataset.timestamp_min · 行级过滤)
- src/dataset.py time_bucket_boundaries (T38 改动目标)
- platform-state.md (5/17 早 4 quota 矩阵 · 2 老 + 2 新策略)
- team-memory (cross-seed 5/22 前锁 · ckpt 24h 过期 · publish ≠ evaluation)
