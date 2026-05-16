# EXP-062 · M55-M58 第 2 批 Overnight · 0.853 冲分战略 · 2026-05-16

> **Status**: 🔵 Documented · 🟡 4 jobs to launch ~CEST 22:00 5/16 (M51-M54 完成后)
> **Date**: 2026-05-16 CEST 19:30 起草
> **Trigger**: 用户 CEST 18:30 提出 "尽快冲 0.853 (DECEM Rank 1)" + path B 已确立 + 16 天来未触碰的真信号方向探索
> **Related**: EXP-061 (M51-M54 第 1 批) · EXP-059 (M49 LB 实证) · EXP-035 (Kaggle 视角 audit · DECEM 4 方向消化)

---

## TL;DR · 0.853 冲分战略 · 第 2 批 4 jobs 探索未触碰真信号方向

**核心战略**:
- 5/16 LB SOTA = M49 0.848162 · 距 0.853 还差 +0.004838
- M51-M54 (第 1 批 in-flight) = path B 体系下 stack 探索 · 期望 +0.0005~+0.001
- M55-M58 (第 2 批 · 本 EXP) = **未触碰真信号方向探索** · 期望 +0.001~+0.002
- 累加上限 (假设 5/8 候选 work) = 0.851~0.852 · 接近 DECEM Rank 1 但**仍不够 0.853**
- 实现 0.853 需要: 至少 1 个 +0.002+ 强真信号 + cross-seed 上沿 luck

**4 jobs 配置**:

| Job | Config | 类型 | 期望 LB Δ | 真信号概率 | 实施成本 |
|---|---|---|---|---|---|
| **M55** | M22 + UE split + UE × user_int Bilinear | T39 新代码 (Seed 主楼后半句) | +0.001~+0.002 | 35% | 2h code + smoke 5/5 PASS ✅ |
| **M56** | M22 + UE split + num_hyformer_blocks=3 | 深度增加 (0 code) | +0.0005~+0.001 | 20% | 0 |
| **M57** | M22 + UE split + d_model=128 | embedding 维度加倍 (0 code) | +0.001~+0.002 | 25% | 0 (但 batch_size 降 64 防 OOM) |
| **M58** | M22 + UE split + seq_a/b 升 512 | 序列长度增 (0 code) | +0.0003~+0.001 | 15% | 0 |

---

## 1. 战略 Context · 为什么这 4 个方向

### 1.1 已知 LB 真信号 (16 天累计)

| Job | LB | Δ vs M22 | 类型 |
|---|---|---|---|
| M22 (vocabtier) | 0.847592 | (anchor) | feature engineering |
| M41 (temporal_weight) | 0.847846 | +0.000254 | dev↓ LB↑ 反例 |
| **M49 (UE split)** | **0.848162** ⭐ | **+0.000570** | architecture-level path B |

→ **3 个真信号** · M22 baseline 已含 vocabtier · 5/16 后续战略基于 M49 base

### 1.2 已死透方向 (8 个 path)

❌ DCN cross (M23 −0.011) · DIN (M5/M6 −0.004) · Multi-Emb K=2 (M38 v2 dev 弱负) · Global Token (M39 v2 dev 弱负) · NLIR (M37 dev 弱负 · 待 LB) · 5-cluster ns_groups (M33 −0.003) · idmask (EXP-025 −0.004) · weekend filter / daypart engineering

### 1.3 未触碰真信号方向 (本批探索)

EXP-058 § audit 8 大错误时未发现的真未触碰方向 (重新审视 16 天所有实验):

#### M55 · UE × user_int Bilinear (Seed 主楼 5/5 第 (1) 条**后半句**)

**Seed 原话**: "user dense feats 将 ue 分离出来 · 其他跟 int pair 加权"

- **前半句** (UE 分离) = M49 已做 ✅
- **后半句** (其他 dense feats 与 user_int pairwise 加权) = **0 投入** ⚠️

**实施 (T39)**:
- model.py 加 `enable_ue_int_bilinear` flag
- `_apply_dense_bypass` 内部: pooled user_ns → Linear → tanh → 加 1 → element-wise multiply other_dense
- Init: weights/bias zero · alpha=0.5 · → step 0 时 gating ≈ 1 (bit-identical to baseline)

**与 DCN-V2 (M23 死) 区别**:
- DCN-V2: layer-wise polynomial cross · learns train-internal user × item co-occurrence · OOV test 反向放大 noise
- M55 Bilinear: single-layer multiplicative gating · OOV-friendly · 仅 reweight 现有 dense feats · 不学共现

**风险**: stack 第 2 步 (M22 + UE split + Bilinear) · 6/6 历史 stack 1 work · 但 M55 base 是 M49 已 work
**期望 LB Δ**: +0.001~+0.002 · 35% 真信号概率

#### M56 · num_hyformer_blocks=3 (深度增加)

**机制**:
- 16 天 model 一直是 num_blocks=2 · 从未试过 3
- 数据量 1.9M × 30k steps/epoch × 4 epoch = 240M sample-step · 应该够支持 3 blocks (按 EXP-034 line 17 公式 · Dense params 增 50% 后仍 >30 sample-step/param)
- 多 1 层 cross-attention + token mixing · 增加 model capacity

**风险**:
- LogLoss rebound 早 (E2 就 overfit · 不到 E4)
- 但有 LogLoss guard 保护

**期望 LB Δ**: +0.0005~+0.001 · 20% 真信号概率

#### M57 · d_model=128 (embedding dim 加倍)

**机制**:
- 当前 d_model=64 · emb_dim=64 (从未改过)
- 加倍到 128 = embedding capacity 4× (numel = D × V)
- T-divisibility check: T = 2×4 + 8 = 16 · 128 % 16 = 0 ✅

**风险**:
- 显存大幅增加 · batch_size 必须降到 64 (从 128)
- wall-time +50% (~6~7h)
- 1.9M data × 4 epoch 可能不够支持 D=128 (按公式: Dense param 增 4× → sample-step/param 减 4×)

**期望 LB Δ**: +0.001~+0.002 · 25% 真信号概率

#### M58 · seq_a/b 升 512 (序列长度增)

**机制**:
- 当前 `seq_a:256, seq_b:256, seq_c:512, seq_d:512`
- M58 = `seq_a:512, seq_b:512, seq_c:512, seq_d:512`
- 短期序列 (a/b) 也看更多上下文 · 可能捕捉长尾用户行为 pattern

**风险**:
- 训练时间 +30% (短期序列 batch 内 padding 增多)
- 显存 OK (sparse seq · 实际 max_len 平均 << 512)

**期望 LB Δ**: +0.0003~+0.001 · 15% 真信号概率 (M17 seq_d=768 dev 0.867794 弱负 · 单边 seq 加长可能边际收益小)

---

## 2. 累加增量预测 · 0.853 现实路径

### 2.1 累加上限估算

假设全部 work (理论 ceiling):

| Step | LB |
|---|---|
| M22 baseline | 0.847592 |
| + UE split (M49) | 0.848162 (+0.000570) ✅ 已实证 |
| + temporal_weight (M51) | 0.849 (+0.0008) 假设 work |
| + Bilinear (M55) | 0.851 (+0.002) 假设强 work |
| + cross-seed s11 (5/22+) | 0.852 (+0.001) 假设上沿 |
| + 0.5 个剩余 boost | 0.8525 |

**总 ceiling = 0.8525** · 距 0.853 还差 +0.0005

### 2.2 现实路径

**保守估算 (5/8 候选 ≥ 1 个 +0.001+ work)**:
- 5/17 LB = 0.849~0.850
- 5/20 LB = 0.850~0.851
- 5/22 cross-seed 上沿 = 0.851~0.852
- **5/25 ddl 前 0.853 概率 ≈ 8~12%**

**乐观估算 (M55 强 work +0.002 + cross-seed luck)**:
- 5/17 LB = 0.850~0.851
- 5/20 LB = 0.851~0.852
- 5/22+ cross-seed = **0.852~0.853**
- **5/25 ddl 前 0.853 概率 ≈ 20~25%**

**悲观估算 (M51-M58 全失败 / 全 noise)**:
- 5/17 LB = 0.848 (持平 M49)
- 5/22 cross-seed = 0.849
- **0.853 概率 ≈ 2~5%**

---

## 3. 5/17 早 4 quota 决策 (含 M55-M58)

5/17 早 quota refresh (CEST 06:00):

**总候选**: 8 个新 + 5 个老 publish-ready = 13 个 candidates

| 候选 | dev (待 5/17 早) | 类型 | 优先级 |
|---|---|---|---|
| **M51-M54 dev 最高** | TBD ~CEST 22:00 5/16 | path B 第 1 批 | ⭐⭐⭐ |
| **M55-M58 dev 最高** | TBD ~CEST 05:00 5/17 | 未触碰方向第 2 批 | ⭐⭐⭐ |
| M37 (NLIR · 老) | 0.868311 | 老候选 | ⭐⭐ |
| M38 v2 (Multi-Emb K=2 · 老) | 0.868403 | 老候选 | ⭐⭐ |
| M30 (Trick 6 · 老) | 0.868438 | 老候选 | ⭐ |
| M50 (反例验证) | 0.861974 | dev 显著负 | ⭐ |

**5/17 早 4 quota 推荐**:
1. **新候选 dev 最高 1**: M51-M58 中 dev 最高 (大概率 path B 体系)
2. **新候选 dev 最高 2**: 第 2 高的新候选
3. **M37 (NLIR)**: EXP-058 §Error E 反推可能 dev↓ LB↑
4. **M38 v2 (Multi-Emb K=2)**: RankUp 闭环 · 大概率持平

**判决红线**:
- 不评 dev 显著负的候选 (M50/M45/M48 等)
- 不破 cross-seed 铁律 (5/22 前)
- M55 evaluation 必须勾选自定义 infer.py (T39 新 kwarg)

---

## 4. 后续 5/17 战略

**场景 A · M55 LB > 0.849** (强真信号):
- 5/17 下午起 4 slot:
  - M22 + UE split + Bilinear + temporal_weight (3 真信号 stack)
  - M22 + UE split + Bilinear + per-domain bucket
  - M22 + UE split + Bilinear (alpha grid 0.3 / 0.7)
  - M22 + UE split + Bilinear + NLIR
- 期望 5/18 早 LB 0.849~0.851 · 进一步逼近 0.853

**场景 B · M55 LB ∈ [0.848, 0.849]** (弱真信号):
- 5/17 下午: stack 探索其他方向 (M56/M57 dev 高的)
- 5/18 早 LB 0.848~0.849

**场景 C · M55 LB < 0.848** (反例 dev↑ LB↓):
- M55 path 关闭 · 探索 M56/M57/M58 谁 work
- 5/18 早 LB ≤ 0.848

**场景 D · 全部新候选 LB ≤ 0.848** (整批失败):
- path B 体系下 stack 整体死透
- 5/18+ 切回 multi-task multi-head / DECEM 未消化方向

---

## 5. 实施成本 (本 EXP)

- **T39 (M55) 实施**: 2h ✅ 已完成 (commit `3f0821f`)
  - model.py: ~50 LOC
  - train.py: ~30 LOC
  - infer.py: ~5 LOC
  - smoke 5/5 PASS (含 zero-init bit-identical · max abs diff = 0.00e+00)
- **M56/M57/M58 run.sh 模板**: ~30 min ✅
- **总实施时间**: ~2.5h (CEST 17:00 → 19:30)

---

## 6. References

- ADR-014 (UE split path B · M49 实证后 Accepted)
- EXP-035 (Kaggle 视角 audit · DECEM 4 方向消化清单)
- EXP-058 (8 大错误 audit · 反推未消化方向)
- EXP-059 (M49 LB 0.848162 · path B 弱真信号)
- EXP-061 (M51-M54 第 1 批 stack 战略)
- src/model.py PCVRHyFormer (T39 ue_int_bilinear · commit `3f0821f`)
- scripts/smoke_t39_ue_int_bilinear.py (5/5 PASS · 含 bit-identical 验证)
- docs/40-experiments/EXP-062-m55-m58-runsh-templates.md (4 个 run.sh 模板)
- 社区情报 Seed 主楼 5/5 第 (1) 条原话 · `docs/99-raw/community-snapshots/xhs-decem-0512/README.md` line 196
