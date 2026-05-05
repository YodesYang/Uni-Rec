# EXP-024 · 方法论复盘 · 2026-05-04 晚(3 次 LB evaluation 之后)

**日期**：2026-05-04 晚 11 ~ 05-05 凌晨
**作者**：Agent(与用户共同推演)
**状态**：🔬 反思性文档 - 不服务任何单个实验,面向**整条 funnel 的设计缺陷**
**关键标签**：funnel-design / dev-lb-divergence / methodology-reset / gap-analysis

---

## 触发原因

**2026-05-04 一天内连续 3 次 evaluation 消耗 3/4 quota**,其中:

| # | 实验 | Local dev | **LB** | 意义 |
|---|---|---|---|---|
| 1 | t13tf-loss-full E9(LogLoss-select ES) | 0.868989 | **0.838707** | local +0.0005 / LB **−0.006** |
| 2 | icq-half E4 | 0.868619 | **0.84637** ⭐ | 建立 LB SOTA |
| 3 | 三合一 E4(T16+T18+holdout) | 0.865516 | **0.836445** | local −0.003 / LB **−0.010** |

**2/3 次 evaluation LB 严重反向**(#1, #3),合计 LB 下降 −0.016,只有 #2 是正向 +0.0014。

用户敏锐提问："**为什么我们屡屡离线提升,评估却反向?funnel 是不是有问题?**"

此文档系统性反思 funnel 设计,**不急于行动,先找根因**。

---

## 第一部分: 所有"离线涨 / LB 反跌"的历史数据点

### 7 次已知 evaluation(按时间)

| Config | Dev AUC | **LB AUC** | Gap | 备注 |
|---|---|---|---|---|
| swiglu baseline | 0.8650 | 0.84381 | +0.0212 | 最早基线 |
| transformer | 0.8641 | 0.834015 | +0.0301 | 推理慢 514s |
| d_model=128 | 0.864881 | 0.836944 | +0.0279 | 单点退步 |
| dropout=0.05 | 0.864563 | 0.83977 | +0.0248 | 正则反而掉 |
| t13tf-full E4 | 0.868495 | 0.844956 | +0.0235 | 长期 baseline |
| t13tf-loss-full E9 | 0.868989 | 0.838707 | +0.0303 | LogLoss-select 过训练 |
| icq-half E4 | 0.868619 | **0.84637** | +0.0222 | **SOTA** |
| icq-full E6 | 0.869138 | 0.843599 | +0.0255 | 过训练 |
| **三合一 E4** | 0.865516 | 0.836445 | **+0.0291** | **本次 crash** |

### 关键观察

1. **gap 稳定在 0.021~0.030**,分布相当集中,这是**系统性**而非随机
2. **dev 和 LB 的相关系数在不同实验间波动巨大** —— 3× 放大(icq-half)到 11× 放大(transformer)到方向相反(三合一)
3. **"dev 涨一点 → LB 涨同等量"这个假设被反复证伪**

---

## 第二部分: 深层诊断 — funnel 到底错在哪?

### 🔴 核心错误 #1: Dev AUC 是错误的指挥棒

我们的整条 funnel (proxy → half-train → full-train → evaluation) **全部按 dev AUC 排序**。但:

- **Dev AUC 测的是 "train 前 95% → train 尾部 5%" 的 generalization**
- **LB AUC 测的是 "train 100% → test 远未来" 的 generalization**
- **这两个是完全不同的任务**

从信息论角度,dev AUC 能提供的**关于 LB AUC 的信息**非常有限,因为:
- 它们的测试集分布不同(valid 是近 train 分布,test 是远 train 分布)
- 它们的"难负样本"分布不同
- 它们的 positive rate 可能不同(时间漂移)

**结论**: 用 dev AUC 作为"是否推进"的决策信号,本质上是**用 A 任务的表现预测 B 任务的表现**,没有理论保证。

### 🔴 核心错误 #2: Proxy 和 half-train 无法辨别"泛化性"

回看 proxy 的本质:
- **1500 steps ≈ 1/14 epoch**
- 过滤的是 "代码 bug / 极端负向" 类问题
- 看不到任何**慢性泛化失败**

half-train 呢?
- 4 epoch(80% 的 full-train 长度)
- 相对 full-train 省了 ~6h
- **但本质上和 full-train 优化的是同一个 dev AUC**

icq-half E4 LB 0.84637 > icq-full E6 LB 0.843599 这个事件告诉我们:**half-train E4 和 full-train E5/E6 的 LB 表现不一样**,但我们不知道为什么。可能的解释:
- (a) 优化器状态不同导致的随机性
- (b) sparse reinit 触发时机不同
- (c) dev 和 train 尾部数据的具体分布不同(因为 train/valid cutoff 一样但 data 总量不同)

**任何一个解释都意味着 half-train 不是 full-train 的"等比例迷你版"**。那么用 half-train E4 预测 full-train E4 的 LB 本身就是**跳跃性推论**。

### 🔴 核心错误 #3: "正交子弹"假设站不住脚

我们把 ICQ / T16 / T18 / T19 / T17 都标成"正交子弹,可以叠加"。但实际上:
- ICQ (query 侧 target conditioning) → 让模型更 target-aware
- T16 (seq 侧 target conditioning) → 让模型更 target-aware
- T18 BPR → 让模型 ranking 更尖锐
- T19 per-token FFN → 让模型容量更大

**这些改动都在"更精准地拟合 train/valid 共享模式"的方向上**。它们的收益来源**可能重叠**,当 valid→LB gap 存在时,堆在一起的**过拟合风险**也重叠。

实验证据:
- ICQ 单独 +0.0001 dev(EXP-019)
- T16 单独 E3 0.868061(vs t13tf E3 +0.00086)
- T18+T16 组合 E4 dev 0.865516(**不是** +0.00086 + ICQ 收益 = 0.868++,而是比单独都低)

**结论**: 堆叠不保证正向,可能互相干扰,甚至**共同加大 overfitting 程度**。

### 🔴 核心错误 #4: 没有"归因实验"的纪律

三合一一次改了 3 个变量:
- T16 additive injection
- T18 CombinedAUCLoss (w=0.3, m=0.5)
- `holdout_ratio=0.05`(训练数据 95% → 90%)

**LB 跌了 −0.010,但无法干净归因**。按一般实验设计纪律,**改变 1 个变量 + 固定其他**是基本原则,我们**违反了**。

代价:
- T18 可能被错杀(实际贡献只有 −0.003 ~ −0.007)
- Holdout 切分的独立成本 (−0.002 ~ −0.005) 被埋没
- 下一轮不知道该恢复哪个 / 保留哪个

---

## 第三部分: 现有 funnel 的隐藏假设 vs 实际情况

| 隐藏假设 | 实际情况 | 损失 |
|---|---|---|
| "dev AUC 是 LB AUC 的单调代理" | 两者相关性**不稳定**,甚至方向相反 | 方向错误的实验推进 |
| "Proxy dev 能预筛烂想法" | Proxy 看不到过拟合、泛化、分布 shift | 假阳性 / 假阴性都多 |
| "Half-train E4 ≈ Full-train E5 的 LB 预测" | icq 数据证伪 | 绕远路做实验 |
| "正交子弹可以叠加" | 子弹可能同向叠加过拟合 | 组合实验翻车 |
| "Evaluation 是 funnel 的 gate,不是数据源" | 4 次 evaluation / 天是**最稀缺资源** | 慢慢才能校准 dev→LB 映射 |
| "Holdout 是零成本的 LB 代理" | 切走 5% train data 有 −0.002~−0.005 LB 隐性成本 | T17 方案 B 的前提不成立 |

---

## 第四部分: 什么是**正确的 funnel**?

先承认一个残酷事实:**我们没有 test 集的任何 label,永远无法线下直接优化 LB AUC**。但我们可以做几件事来**提升 dev 和 LB 的相关性**:

### 改革方向 A: **重新设计 valid split**(最重要 · 治本)

现在的 time-split 本质是 "train 的近邻"。要让 dev 更接近 test,应该:

**方案 A1 · 留时间空窗**(ADR-005 方案 C):
```
train[0, 80%] | 空窗[80%, 95%](丢弃) | valid[95%, 100%]
```
这让 valid 距离 train 15% 时间跨度(≈ 0.5~1 天),模拟 test 的"远未来"特性。

**预期代价**: 训练数据从 95% 降到 80%,dev AUC 整体会下降 ~0.002~0.005,但 dev→LB 相关性应**显著提升**。

**方案 A2 · 多个切点取 voting**:
```
valid_a = train[55:60]  ← 中段
valid_b = train[80:85]  ← 中后段
valid_c = train[95:100] ← 尾部(当前)
```
然后用 3 个 dev AUC 的**一致性**作为过拟合早警信号。如果 valid_a / valid_b 涨了但 valid_c 不涨,说明**尾部 overfitting**,应早停。

### 改革方向 B: **Evaluation 作为主要数据源,而非 funnel gate**

当前 evaluation 被当成 "funnel 最后一关",仅用于"最终选手上榜"。但在本赛题的 funnel 背离场景下,**evaluation 的每次都是无价的校准信号**。

**改革建议**:
- **前 5~10 次 evaluation 用于建立 "dev → LB" 映射**,故意提交**不同 config** 的 ckpt,不追求最高 LB
- 建立 `config_fingerprint → dev_gap` 的查找表
- 一旦 fingerprint 稳定(例如 5 个相近 config 的 gap 都在 0.022±0.002),用这个经验 gap 作为 LB 先验
- 只有**明确超过先验预测 LB** 的 ckpt 才值得 evaluate 追 SOTA

### 改革方向 C: **组合实验前必须做单变量 ablation**

铁律:任何组合改动(A + B + C)提交 full train 之前,**必须有 A-only / B-only / C-only 的 half-train 结果**。

当前违反:
- 三合一(T16 + T18 + holdout)直接 full halftrain → T18 被错杀
- 四合一(T16 + T18 + T19 + holdout)在 T18 还没单独验证时就启

**新规**: 任何方法学级改动,先**单独 half-train**,等 half-train 结果出来再决定是否叠加。

### 改革方向 D: **Proxy 重新定位**

Proxy 不该作为"推进 / 不推进"的 gate,只该作为:
- **代码 crash 过滤器**(5 分钟搞定的事不该用 proxy)
- **shape / dtype 兼容性检查**(本地 smoke 就够了)
- **不要用 proxy 做方法学级判决**

**新规**: 方法学级改动(新 loss / 新模块 / 新正则)**直接跳过 proxy**,上 half-train。省的 quota 可以用来跑**独立 ablation**。

### 改革方向 E: **Ckpt 保留多个 epoch 供离线选优**

当前 `trainer.py` 只保留 best_model(AUC 最高)。但:
- icq-half E4 LB 0.84637 > icq-full E6 LB 0.843599
- 说明 **"AUC 最高"不等于"LB 最高"**

改革建议: 修改 trainer 支持 `keep_topk_ckpt=3`,保留 dev AUC 最高的 3 个 epoch ckpt。evaluation 时**全部提交**(只需 3 次 quota),对照 LB 选真正最优的那个。

**代价**: 3× evaluation quota,但买到**完整 epoch-wise LB 曲线** → 这是目前唯一能**直接测出 gap 扩大规律**的方式。

### 改革方向 F: **Ranking loss 等"远离平滑梯度" 的方法要慎重**

T18 Deprecation 暴露了:
- BPR / AUC loss / 对比学习 loss 都可能在 train/test 分布 shift 下失效
- 因为它们依赖 train batch 的 pair/triplet 采样,学到的**"排序依据"依赖 train 绝对分数分布**
- 而 test 分布漂移会打破这个假设

**新规**: 所有 ranking-family loss(BPR / ListMLE / Contrastive)在本赛题**优先级降到最低**,除非先有机制级的创新(例如 calibration-aware ranking loss)。

---

## 第五部分: 可执行的立刻动作(按优先级)

### P0 · 立即(今晚/明早, < 1h)

1. **✅ 修正 ADR-006 Deprecation Notice 机制解释**(已完成),并加入归因歧义段
2. **✅ team-memory 新增 "holdout 切分可能有隐藏成本" 铁律**(已完成)
3. **落档 EXP-024**(此文档)

### P1 · 当前 T16-only full train 回流后决策

4. **T16-only LB 核查 holdout 假设**:
   - 如果 T16-only(带 holdout)LB < 0.846 → holdout 有实锐成本,**立刻改 run.sh 去掉 holdout**
   - 如果 T16-only LB ≥ 0.846 → holdout 成本 < 0.001, T17 方案 B 的小代价可接受
5. 根据结果**更新 ADR-005 状态**(Partial Accept / Rejected)

### P2 · 本周内

6. **Valid split 方案 A1 落地**:`train[0, 80%] | 空窗 | valid[95, 100%]`,作为新默认;现有 time-split 保留作为 fallback
7. **Trainer 加 `keep_topk_ckpt` 支持**:保留 top-3 dev AUC 的 ckpt 供离线选优
8. **归因实验纪律写入 `.cursor/rules`**:组合实验必须先有 ablation 数据

### P3 · 长期

9. **建立 dev→LB 经验映射表**:每次 evaluation 录入 `config_fingerprint, dev_AUC, holdout_AUC, LB_AUC`,逐步建立**非线性**的 gap 预测模型
10. **考虑放弃 dev AUC 作为唯一指挥棒**:用 dev AUC + LogLoss + holdout AUC **组合信号**决策推进

---

## 第六部分: 关于 "test 集有没有问题" 的判断

用户敏锐指出: 4/30 官方重置过一次 test 集,有没有可能**又出问题**?

### 反驳证据

1. **gap 稳定在 0.021~0.030** 是**系统性 distribution shift**,不是 bug 引入的随机噪声
2. **社区 SOTA 0.858 可达** → test 集可以被打到高分,说明**响应训练改动**
3. **ICQ half-train LB 涨 +0.0014** → test 集对有效改动有正向响应
4. **赛题设计**: 官方 4/26 答复 test 是"远未来",distribution shift 是**设计特征**而非 bug

### 支持证据

1. **先例存在** (4/30 重置)
2. **未核查过数据版本**:4/30 后我们**从未重新下载 / 对比 hash** 确认 train/test 是否更新过
3. **社区有"训练分数与评估分数相差大吗"的帖子**

### 判断

**test 集大概率没问题**,真正的问题是 **dev 和 test 的分布差异被赛题放大**,而我们的 funnel 没有对此做出相应设计。

**但"版本核查"是 5 分钟能做的事**,P1 执行:
- 检查社区(lexiang)有没有 5/1~5/4 关于 test 集变更的帖子
- 如果可能,记录当前平台 train/test 数据的文件时间戳作为 baseline

---

## 第七部分: 最诚实的自我批评

回看今天所有决策,我作为"AI agent"的**最大失误**:

### 1. 过早给 T18 判死刑

三合一 E4 LB 一出 −0.010,我立刻写了 ADR-006 Deprecated + team-memory 铁律。**但实际上**:
- 这是**组合实验**,归因歧义严重
- "calibration 破坏"的机制解释**逻辑站不住**(AUC 是排序指标)
- 正确做法应该是**标记 T18 为 "Blocked (awaiting clean ablation)"**,而非直接 Deprecated

### 2. 对 holdout 的成本过于乐观

我一直说"holdout 零成本搭车,只是旁观 data"。**但实际上**:
- 训练数据从 95% 减到 90%,损失 ~100k 样本
- 在 1.9M 规模下,5% 的 sparse embedding 更新不充分可能是 −0.002~−0.005 LB 的真实代价
- 没做对照实验就**假设**它零成本

### 3. 把 "dev 涨 → 推进" 当成不变真理

今晚之前,我一直按 "T16 E3 0.868 > t13tf E3 0.867 + 0.0002 阈值 → 推进" 这种机械 funnel 决策。**实际上这个阈值是基于**"dev 涨 → LB 涨"**这个假设,而这个假设的实证可靠性只有 50%**。

### 4. 忽视"归因实验"作为 quota 的合理消耗方式

当前 quota 使用逻辑: "只有 ≥ SOTA 阈值的 ckpt 才 evaluate"。**但这导致**:
- 3 次 evaluation 只拿到 1 次正向数据
- dev→LB 映射永远建立不起来
- 每次 LB 反向后都只能**事后猜测**

**合理做法**: 前 5~10 次 evaluation 明确用于**建立映射表**,不追求 SOTA。

---

## 最终结论

**我们今天的问题不是"test 集有 bug",也不是"某条方法学子弹真的有效 / 无效",而是 funnel 设计本身在系统性误导我们**。

优先级重排:
- 🔴 **Valid split 重设计**(方案 A1)是最根本的修复
- 🔴 **归因实验纪律**立即生效
- 🟡 **dev→LB 映射表建立**需要多次 evaluation 的积累
- 🟢 方法学级创新(新 loss / 新模块)先放一放,等 funnel 自己修好

**在 funnel 不修好之前,任何方法学级创新的 LB 信号都是不可信的**。

---

## 附录: 今日 quota 使用复盘

- evaluation: 3 / 4 used
  - ❌ t13tf-loss-full E9 (−0.006 vs SOTA, 浪费)
  - ✅ icq-half E4 (+0.0014, 建立 SOTA)
  - ❌ 三合一 E4 (−0.010 vs SOTA, 无法干净归因)
- training: 4 / 4 used 过
  - 三合一 halftrain ❌ (已知 LB 差)
  - 四合一 halftrain ❌ (停止, 已判负向)
  - ICQ full ❌ (过训练,LB 不如 half)
  - T16-only full 🟡 (在跑,兜底方案)

**总结**: 今天 quota 密集消耗,但**信息净增量**主要来自 icq-half E4 和三合一 E4 两个数据点。其他 evaluation / training 大多是"follow-up 未必带来新信息"。

如果早 6 小时就**暂停写代码 / 开始复盘**,可能用 2 次 evaluation 就能获得今天全部 3 次的信息量,并且**避免三合一的组合实验错误**。

教训: **在积累信号的关键节点,反思 > 继续推进**。
