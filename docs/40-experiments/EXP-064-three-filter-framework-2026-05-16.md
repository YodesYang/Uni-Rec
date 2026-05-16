# EXP-064 · 三滤镜分析框架 · 60+ Jobs 真信号 vs 失效机制总结 · 2026-05-16

> **Status**: 🔬 战略框架文档 (CEST 22:50 起草 · 仅本 EXP 落档 · 不动 ADR / team-memory / active.md)
> **Date**: 2026-05-16 CEST 22:50
> **Trigger**: 用户 CEST 20:51 反问 "我们目前有生效策略的原因是什么 · 期望较大策略但实际没生效的原因又是什么 · 不要炼丹"
> **Related**: EXP-035 (Kaggle audit) · EXP-051 (gap=0.05 saturation) · EXP-052 (M22 真信号) · EXP-058 (8 大错误 audit) · EXP-059 (M49 path B) · ADR-014 (UE split)

---

## TL;DR

基于 5/4~5/16 累计 60+ jobs 实证数据 (4 个 LB 真信号 + 7 个高期望失效案例),提取出**三滤镜分析框架**:任何新策略提议在执行前必须按机制级别评估 4 个字段,把"凭印象/论文数字直接外推" 转化为 "数据驱动期望"。

**核心发现** (3 句话):
1. **真信号都是"信息结构重排"** (T25 / T29 / T37 / M41) · 不增 per-X 独立参数 · 不学 train 内共现 · OOV friendly
2. **高期望失效都触犯 3 滤镜之一**: OOV 不可迁移 / 梯度预算饥饿 / 排序 loss 远未来分布失稳
3. **本赛题"信号"和"实际收益"之间隔着 3 个滤镜** · 论文 ablation 数字不可直接外推 · 必须乘 OOV/梯度/loss 三重衰减

**对未来策略的影响 (M59 降级证据基础)**:
- M59 (Multi-Emb K=2 + Global Token 论文最大单项 +0.21% online) 触犯滤镜 1+2 → 期望从 +0.001~+0.002 修订为 +0~+0.0008
- M30 / M52 / M54 / M51 / M53 (per-domain bucket / Q_init / path B stack) 滤镜全过 → 升为 5/17 早 quota 优先

---

## 1. 4 个 LB 真信号的共同机制

### 1.1 数据 ground truth (按 LB Δ 排序)

| 策略 | LB Δ vs 前一基线 | 实施位置 | 增加参数 | 数据出处 |
|---|---|---|---|---|
| **T25 DIN bypass** | **+0.004** (T25 base 0.843433 vs 前 baseline 0.839223) | classifier 前 bypass | ~50k Dense (跨 seq domain 共享 cross-attn) | EXP-040 |
| **T29 vocabtier ns_groups** | **+0.000882** (M22 0.847592 vs M9 0.846710) | NS Tokenizer chunk 边界重排 | **0** (改 chunk 划分) | EXP-052 |
| **T37 UE split** | **+0.000570** (M49 0.848162 vs M22) | NS Tokenizer 内 + dense bypass 路径 | ~4.2k Dense | EXP-059 |
| **M41 temporal_weight** | **+0.000254** (M41 0.847846 vs M22) | sample loss weight | **0** | EXP-057 衍生 |

**累计 LB SOTA 跃迁**: 0.839223 → 0.843433 (T25) → 0.846710 (M9 含 timefeat+hash) → 0.847592 (M22 + T29) → 0.848162 (M49 + T37) · **16 天累计 +0.008939 LB**。

### 1.2 共同机制 (4 个真信号交集)

#### 共同点 1 · **不增加 per-X 独立参数**
- T25 共享 cross-attention (跨 seq domain 同一模块)
- T29 改 chunk 边界 (0 新参数)
- T37 把已有 256-dim UE 分离 (~4.2k 共享 Linear)
- M41 改 sample weight (0 新参数)

vs **per-X 独立参数三连败 (T19 / T28 / FaFE)**: 新增独立参数与样本梯度不耦合 → 信号稀释。

#### 共同点 2 · **不学 train-internal co-occurrence pattern**
- T25 学的是 "target-aware" filter (target 与 seq 的相似度) · 不学 user × item 共现
- T29 / T37 / M41 完全不引入交叉模式

vs **DCN cross (M23 LB −0.011)**: 学 user_1 × item_5 共现 → test 100% cold start 下完全不可迁移。

#### 共同点 3 · **OOV-friendly (test 远未来 100% user cold start 场景)**
- 不依赖 train-test id 共享 · 即使 test user 全新,模型行为有 well-defined 路径
- T25 target attention: target item 端虽是 train 出现过的 id 集合,但**机制本身**对新 user-target pair 仍 well-defined
- T37 UE token: fid_61 (256-dim 预训练 embedding) 是 production 已学的 user-level summary · 新 user 仍有 inference 路径

#### 共同点 4 · **增量幅度 ≤ +0.001 (T25 异常值除外)**
- T29/T37/M41 都在 +0.0003~+0.0009
- T25 +0.004 是项目最大单点 · 16 天累计无任何策略接近

**机制级总结**: 真信号 = **对现有信息的更好结构化** (re-routing) · 不是新增 capacity 或新增模式。

---

## 2. 7 个高期望失效案例机制分析

### 2.1 数据 ground truth (按错判幅度排序)

| 策略 | 期望 LB Δ (论文/情报锚) | 实测 LB Δ | 错判幅度 | 出处 |
|---|---|---|---|---|
| **T34 DCN cross (M23)** | +0.001~+0.003 (DECEM Trick 5 + EXP-049 EDA top user×item) | **−0.011** | **~13×** 反向 | EXP-058 §Error B |
| **T18 CombinedAUCLoss** | 外部 anchor +0.005 | LB **−0.010** (三合一) | ~3× 反向 | team-memory 5/4 |
| **EXP-025 idmask** | 防 id OOV 过拟合 (Pung 评论建议) | LB **−0.004** | 2× 反向 | EXP-025 |
| **T28 DIN integrated** | T25 work +0.004 推 +0.005~+0.007 | LB **−0.00126**~**−0.005** | T25 → integrated 反向 | EXP-039 |
| **M33 5-cluster ns_groups** | EXP-049 EDA-correlation 4 cluster 实证 | LB **−0.003** | 实施 mistake | EXP-058 §Error D |
| **T19 per-token FFN** | RankMixer §3.2 (Tencent Ads 同团队) | Dev 负 + E2 guard 触发 | 论文 → 项目反向 | EXP-034 / EXP-039 |
| **M38 v2 Multi-Emb K=2** | RankUp §4.2 +0.10%~+0.15% online (论文 ablation 直推) | Dev −0.000173 (LB 待评) | 论文 → 项目几乎 0 | EXP-046 |

### 2.2 失效机制 3 大根本原因

#### **原因 A · OOV transferability 与 train 内信号被混淆** (代表: T34 DCN cross)

EXP-049 给的 EDA 数字 (top user×item Cramér V 0.27~0.28) **本身没错**,但**EDA 测的是 train 内共现强度**,不是 test 上的可迁移性。机制:
- DCN-V2 学 `user_1 × item_5` 共现 pattern
- Test 100% user cold start (EXP-052 Fact 4 实证) · `user_1=新值 × item_5=新值` 模型从未见过
- **DCN 给模型一个"我懂这个 pair"的虚假信心 → OOV 场景下系统性误导**

→ **任何 cross/interaction feature 必须先验证 test 端候选 cross 值的 OOV 率**,不能只看 train 内 Cramér V/MI。

#### **原因 B · 梯度预算不足让 per-X 独立参数饥饿** (代表: T19 / T28 / FaFE)

数学化 (EXP-034 公式):
- 1.9M data × 4 epoch × batch 128 ≈ **60k optimizer steps** total
- T19 per-token FFN ≈ 1000 个 token · 每 token 平均 60 steps gradient
- T28 DIN integrated 每 block 独立 = 2 块 · 每 block ~60k steps 看似够 · 但 DIN 内部 cross-attn 参数数量大 + 下游耦合弱
- FaFE per-fid LayerNorm+Linear ≈ 1000 fids · 每 fid 60 steps

**独立参数下游耦合弱 → gradient signal 不足以学 generalize 模式 → 被迫记忆 train 期局部分布 → test 远未来必过拟合**。

→ **工业论文 (RankMixer / RankUp 数十亿数据) 直接迁移 1.9M 学术赛题的最大坑**。

#### **原因 C · 稀疏 + 局部依赖 loss 在远未来分布偏移下排序失稳** (代表: T18 CombinedAUCLoss)

team-memory 5/4 修订原话:
> "BPR 学到的排序依赖于 train 绝对分数分布,在 test 分布偏移下不稳定 —— BPR 推极端 pair 分数的方式依赖 train batch 局部分布,test 上新的 hard negative 落在未训练过的 score region,排序关系整体不稳定部分 pair 翻转 → AUC 下降。BCE 梯度 σ(s)−y 在整个 score range 平滑,排序对分布漂移更 robust。"

→ **任何依赖 "train batch 内 ranking margin" 的 loss 在 test covariate shift 下系统性失效**。

### 2.3 1 个非机制性失效 · 实施 mistake (M33)

M33 5-cluster ns_groups **不是路径错,是把 vocabtier 整体替换了**。EXP-049 EDA 给的是"user × user CV 聚类" (user-side 维度);vocabtier 是 item-side 维度。**两者正交本应 stack 而非替换**。M33 替换后失去 high-vocab item id 单独 chunk (M22 真信号源) → LB −0.003。

→ EXP-058 §Error D 铁律: **LB 强负 first 检查 implementation diff,不要默认路径死**。

---

## 3. 三滤镜分析框架 (本 EXP 核心贡献)

### 3.1 框架结构

任何新策略的"实际 LB 收益"是"信号源给的期望"经过 3 个滤镜衰减后的产物:

```
论文 / EDA / 情报给的"期望信号"
   ↓ 滤镜 1 · OOV 可迁移性 (test 100% user cold start)
   ↓ 滤镜 2 · 梯度预算 (1.9M × 4 epoch ≈ 60k steps · 是否够新参数)
   ↓ 滤镜 3 · 远未来分布稳定性 (loss / 排序学习是否 fragile)
   ↓ 实施细节 (是否替换关键已 work 路径)
实际 LB 收益
```

### 3.2 4 字段 audit 表 (新策略提议必填)

| 字段 | 问题 | 答 Yes 衰减 | 答 No 衰减 |
|---|---|---|---|
| **滤镜 1 (OOV)** | 该策略是否依赖 train-test id 共享 / 学 train 内共现 pattern? | × 0.3 | × 1.0 |
| **滤镜 2 (梯度)** | 新增 per-X 独立参数,且每 X 平均梯度 < 100 steps? | × 0.5 | × 1.0 |
| **滤镜 3 (Loss)** | 引入 ranking-style loss / batch 内排序依赖? | × 0.3 | × 1.0 |
| **实施 risk** | 是否替换已 work 关键路径 (而非 stack)? | 强制要求单变量 baseline | 不强制 |

**修订期望 = 期望信号 × 滤镜 1 × 滤镜 2 × 滤镜 3** (如有实施 risk · 进一步要求 ablation 隔离)。

### 3.3 适用范围限制 (用户 5/16 22:48 反问触发的修正)

⚠️ **本框架仅适用 "新提议未起动 jobs"** · 不适用以下场景:
- ❌ **已 in-flight jobs** (M51-M54) · slot 已占用 · 重评只是 mental exercise · 应等数据回流
- ❌ **已 publish-ready ckpt 待评 LB** (M30/M37/M38v2/M39v2/M27v2/M50/M17) · LB 是 ground truth · 不需估算
- ❌ **已 LB 闭环数据** (T25/T29/T37/M22/M28/M29/M33/M40/M41/M23/M36/M49/M9 共 13 个) · 历史已实测
- ✅ **未来未起动新提议** (M55/M59/M60/M61/M62 / 任何"如果起 X 应该 +Y") · 必填 4 字段

### 3.4 历史回溯验证 (4 真信号 + 7 失效全部命中)

#### 4 真信号回溯
| 策略 | 滤镜 1 | 滤镜 2 | 滤镜 3 | 实施 | 框架预测 | 实测 | 命中 |
|---|---|---|---|---|---|---|---|
| T25 DIN bypass | ✅ (target attn) | ✅ (共享) | ✅ (BCE) | ✅ stack | + | +0.004 | ✅ |
| T29 vocabtier | ✅ (chunk 重排) | ✅ (0 新参数) | ✅ | ✅ stack | + | +0.000882 | ✅ |
| T37 UE split | ✅ (UE token 分离) | ✅ (4.2k 共享) | ✅ | ✅ stack | + | +0.000570 | ✅ |
| M41 temporal_weight | ✅ (sample reweight) | ✅ (0 新参数) | ✅ | ✅ | + | +0.000254 | ✅ |

#### 7 失效回溯
| 策略 | 滤镜 1 | 滤镜 2 | 滤镜 3 | 实施 | 框架预测 | 实测 | 命中 |
|---|---|---|---|---|---|---|---|
| T34 DCN cross | ❌ (学共现) | ✅ | ✅ | ✅ | × 0.3 衰减 | −0.011 | ✅ (滤镜 1) |
| T18 BPR loss | ✅ | ✅ | ❌ (ranking) | ✅ | × 0.3 衰减 | −0.010 | ✅ (滤镜 3) |
| EXP-025 idmask | ❌ (input mask 依赖 train id) | ✅ | ✅ | ✅ | × 0.3 衰减 | −0.004 | ✅ (滤镜 1) |
| T28 DIN integrated | ✅ | ❌ (per-block 独立) | ✅ | ⚠️ 重复 T25 | × 0.5 衰减 | −0.005 | ✅ (滤镜 2) |
| M33 5-cluster | ✅ (chunk 重排) | ✅ | ✅ | ❌ 替换 vocabtier | 实施 risk 触发 | −0.003 | ✅ (实施) |
| T19 per-token FFN | ✅ | ❌ (per-token 独立 ~1000 token) | ✅ | ✅ | × 0.5 衰减 | Dev 负 | ✅ (滤镜 2) |
| M38 v2 Multi-Emb K=2 | ✅ | ❌ (sparse param 翻倍 600M) | ✅ | ✅ | × 0.5 衰减 | Dev 弱负 | ✅ (滤镜 2) |

**11/11 命中** · 框架在历史 11 个数据点上无反例。

---

## 4. M59 降级证据 (基于本框架重评)

### 4.1 M59 配置

`M22 + Multi-Emb K=2 + Global Token` · 论文 RankUp §4.2 Table 1 报 +0.21%/+0.18%/+0.13% online AUC (3 任务) · "5 组件里最大单项"。

### 4.2 严格 4 字段 audit

| 字段 | M59 答案 | 衰减 | 依据 |
|---|---|---|---|
| **滤镜 1 (OOV)** | 部分 Yes · Multi-Emb 增 sparse table 容量 · 在 test 100% cold start 下,新 user 的 lookup 仍落到已学 embedding,但学到的"两个独立子空间"是否在新 user 上仍 well-defined? **不确定** (M38 v2 单独 Dev −0.000173 暗示风险) | × 0.7 (部分) | EXP-046 |
| **滤镜 2 (梯度)** | **Yes** · Multi-Emb K=2 直接翻倍 sparse param (300M → 600M) · 每参数梯度信号减半。本项目已实测 hash 500k > 2M 就是这个原因 (team-memory 2026-05-04) | × 0.5 | hash500k vs hash2M 实证 |
| **滤镜 3 (Loss)** | No · Multi-Emb 不直接影响 loss 路径 | × 1.0 | — |
| **实施** | No · stack 不替换 | 不触发 | — |

**修订期望**:
- 论文期望 = +0.21% online ≈ +0.001~+0.002 LB (50% 迁移率)
- 滤镜衰减 = × 0.7 × 0.5 × 1.0 = × 0.35
- **修订 M59 期望 LB Δ = +0.0004~+0.0007** (vs 之前估 +0.001~+0.002)

### 4.3 优先级修订

- 之前 (§2 我的初稿): M59 = ⭐⭐⭐ (论文最大单项 · 16 天唯一未测)
- **修订 (本 EXP)**: M59 = ⭐ (滤镜 1+2 触发 · ROI 不及 M30/M52/M54)

### 4.4 替代方向 (滤镜全过的高 ROI 候选)

| 候选 | 滤镜 1 | 滤镜 2 | 滤镜 3 | 实施 | 修订期望 LB Δ |
|---|---|---|---|---|---|
| **per-domain time bucket (T38, M52/M54)** | ✅ | ✅ (0 新参数 · 改 boundaries) | ✅ | ✅ | +0.0005~+0.001 (维持) |
| **Q_init=MLP(item_emb) (Trick 6, M30)** | ✅ | ✅ (~50 行共享 MLP) | ✅ | ✅ | +0.0005~+0.001 |
| **M27 v2 (T31 is-missing item-only)** | ⚠️ (引入新 vocab · 但 train≈test missing 分布 EXP-052 Fact 2) | ✅ | ✅ | ✅ | +0.0003~+0.0008 |
| **M55 (UE × user_int Bilinear T39)** | ⚠️ (gating 是 reweight 但引入新交互) | ✅ (共享) | ✅ | ✅ | +0.0003~+0.001 |
| **path B stack (M51/M53)** | ✅ (M49 base 已 work) | ✅ | ✅ | ⚠️ stack 第 2 步历史失败率高 | +0~+0.001 (依 stack 互斥风险) |

---

## 5. 3 天路线图修订 (基于本框架)

### 5.1 5/17 早 4 quota 优先级 (修订)

> **前提**: M51-M54 dev ~CEST 22:00 5/16 回流后 + 用户决 5/16 晚 4 free slots 用法

| Slot | 候选 | 滤镜状态 | 期望 LB | 优先级 |
|---|---|---|---|---|
| 1 | **M52 / M54 (per-domain bucket) dev 较高的** | 4/4 全过 | +0.0005~+0.001 | ⭐⭐⭐ |
| 2 | **M30 (Trick 6 / Q_init=MLP)** | 4/4 全过 | +0.0005~+0.001 | ⭐⭐⭐ |
| 3 | **M51 / M53 (path B stack) dev 最高的** | 3.5/4 (stack 第 2 步 risk) | +0~+0.001 | ⭐⭐ |
| 4 | 弹性: M37 (NLIR · 反例验证) / M27v2 (T31) / M50 (反例) | 视 1-3 LB 而定 | — | ⭐~⭐⭐ |

### 5.2 5/16 晚 4 free slots 决策原则 (等 M51-M54 dev 后定)

**可候选 jobs** (按本框架排序):
1. **M30 重训** (如平台 ckpt 风险高需 rerun) · 滤镜全过
2. **M55 (T39 UE × user_int Bilinear)** · 3.5/4 滤镜 · code ready (commit 3f0821f)
3. **M51 / M53 / M52 / M54 重训** (如 dev 不达预期需 cross-seed 校准 · 但触 5/22 lock)
4. **M59** · 1.5/4 滤镜 · 降级 · 不优先

**用户决策时机**: 5/16 ~CEST 22:00 (M51-M54 dev 回流) → 5/16 ~CEST 23:00 决 (1 小时 audit 窗)

### 5.3 5/17~5/19 总目标修订

- **Day 1 (5/17)**: 评 4 quota → 累积 +0.0005~+0.0015 LB → SOTA 0.8487~0.8497
- **Day 2 (5/18)**: 基于 Day 1 LB 起 stack 闭环 / M55 / 论文组合 (如 M59 不优先则 M55+其他)
- **Day 3 (5/19)**: cross-seed 上沿搜索 (5/22 lock 是否破例视 LB 进度而定)

**Top 20 (LB 0.851) 修订概率**: 三滤镜框架下,M59 不再是高 leverage,真实路径回到 stacking 多个 +0.0005 真信号 + cross-seed luck → **概率 8~12%** (vs §2 初稿 18~25% 的乐观估算)。

---

## 6. 框架使用纪律 (用户 22:48 反问后的范围限制)

### 6.1 何时使用本框架

✅ **必须使用**: 任何"未来想起 X jobs / X 应该 +Y" 类提议 · 写 EXP / ADR / active.md "下一步" 时
✅ **必须使用**: 给用户推荐"晚 4 slots 起什么"前 · 每个候选必填 4 字段
✅ **必须使用**: agent 给"乐观/悲观估算" 类陈述时 · 每个数字必须有滤镜出处

### 6.2 何时不用本框架

❌ **in-flight jobs**: 等数据 · 不重评 (M51-M54)
❌ **publish-ready ckpt 待评 LB**: LB 是真理 · 不估算 (M30/M37/M38v2/M39v2/M27v2/M50/M17)
❌ **已 LB 闭环**: 历史数据已实测 (13 个 LB jobs)
❌ **非方法学问题**: cross-seed / quota 安排 / 平台操作 · 无关方法学衰减

### 6.3 框架本身的局限

⚠️ **滤镜系数 (× 0.3 / × 0.5) 是经验估算** · 基于 11 个历史数据点拟合 · 非严格统计
⚠️ **滤镜独立性假设** · 实际可能耦合 (如 OOV + 梯度同时触发可能比 × 0.3 × 0.5 更狠)
⚠️ **新机制可能不在 3 滤镜内** · 未来实验可能暴露第 4 / 第 5 滤镜 · 本 EXP 仅基于 16 天数据

→ **不绝对化此框架** · 每用一次后看预测 vs 实际偏差 · 持续校准。

---

## 7. References

- EXP-035 (Kaggle 视角 audit · 5/10 立"48h audit"铁律)
- EXP-040 (T25 bypass +0.004 LB 实证)
- EXP-052 (M22 vocabtier +0.000882 + Test EDA train≈test 实证)
- EXP-058 (8 大错误 audit · §Error A~H)
- EXP-059 (M49 path B +0.000570 + ADR-014 Accepted)
- EXP-046 (M38 v2 Multi-Emb K=2 Dev 弱负)
- ADR-012 (Multi-Embedding 设计 · 论文 ablation 参考)
- ADR-013 (Global Token 设计 · 论文 ablation 参考)
- ADR-014 (UE split path B Accepted)
- team-memory 2026-05-04 (BPR/ranking loss 远未来失效铁律)
- team-memory 2026-05-12 (per-X 独立参数三连败铁律)
- team-memory 2026-05-13 (user cold start + EDA 实证)
- src/model.py · enable_dcn_cross / enable_dense_bypass / enable_din_interest / enable_global_token / enable_ue_split / enable_ue_int_bilinear / multi_emb_k

---

## 8. Update Log

- **2026-05-16 CEST 22:50**: 文件创建 (用户 20:51 反问"为什么生效/失效 · 不要炼丹" 触发 · 22:48 反问"活跃方向没复查"补丁 §3.3 适用范围限制)

