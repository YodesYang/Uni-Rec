# 7 篇论文的演进地图

> **用途**：在读任何一篇原论文前，先看这张地图，知道"这篇在解决什么问题、前面谁启发了它、后面谁继承了它"。
> **核心观点**：**不要孤立读 7 篇论文**。它们是一条时间线 + 两条技术主线的交织。

---

## 一眼看懂：7 篇论文的时间线

```
2017 ────── 2018 ────── 2021 ──── 2022 ─── 2023 ─── 2025 ─── 2026 (现在)
  │           │           │                           │        │
  ▼           ▼           ▼                           ▼        ▼
 [1]         [6]         [2]                        [4][7]   [3][5]
Attention   ESMM        RoFormer                   RankMixer HyFormer
Is All      (阿里)       (RoPE)                     LONGER    OneTrans
You Need                                           (抖音)    (赛题核心)
(Google)                                                    (WWW'26)
```

两条主线：
- **A. 架构演进线**：Attention [1] → RoPE [2] → RankMixer [4] / LONGER [7] → HyFormer [3] / OneTrans [5]
- **B. 广告/推荐应用线**：ESMM [6] → (其他 CVR 论文省略) → HyFormer [3] / OneTrans [5]

**交汇点**：HyFormer 和 OneTrans 同时吸收了 A 线的架构和 B 线的业务 insight，是两条线的"现代综合"。

---

## 每篇论文的"一句话定位"

| # | 论文 | 一句话 | 对赛题的重要性 |
|---|---|---|---|
| 1 | Attention Is All You Need (2017) | **地基**：定义了 Transformer / Self-Attn 的基本形式 | ⭐⭐⭐ 必读 §1-3 |
| 2 | RoFormer: RoPE (2021) | 用旋转矩阵替代 sin/cos 位置编码，相对位置感知 | ⭐⭐ 读 §3.4 |
| 3 | HyFormer (2026) | 赛题核心：多序列 + Query Decoding + RankMixer boost | ⭐⭐⭐ 精读 §3 |
| 4 | RankMixer (抖音, 2025) | 无参 token mixing 替代 attention，适合推荐 | ⭐⭐⭐ 读 §3 |
| 5 | OneTrans (WWW 2026) | S 和 NS token 混合参数化，把推荐做成统一 Transformer | ⭐⭐⭐ 精读 §3 |
| 6 | ESMM (阿里, 2018) | CVR 预估的样本选择偏差解法 | ⭐ 读中文博客 即可 |
| 7 | LONGER (2025) | 超长序列的 top-K cross-attn 压缩 | ⭐ 看 abstract + Fig 1 |

---

## 三者关系搞清楚：HyFormer / RankMixer / OneTrans

这是你 D 诊断（多篇关系不清）的核心痛点。一次讲清。

### 共同点：都是给"推荐/广告"场景的 Transformer 变种

三篇都在解决同一个问题：**如何把 Transformer 适配到"有序列有非序列、参数要高效、推理要快"的推荐/广告场景**。

三篇的答案：**不一样**。

### 技术路线对比

| 维度 | RankMixer (2025) | HyFormer (2026) | OneTrans (2026) |
|---|---|---|---|
| 核心创新 | Token Mixing + Per-token FFN | 多序列 + Query Decoding | Mixed Parameterization |
| 是否有 Attention | ❌ 完全无 attention | ✅ 有（cross-attn） | ✅ 有（mixed self-attn） |
| 参数效率 | 极高（token mixing 无参） | 中（Q decode 压缩） | 低（NS 每个独立参数） |
| 表达力 | 中（靠 FFN 交互） | 中高（Q-S cross-attn） | 高（S+NS 全连 attn） |
| 赛题 AUC | （未独立测） | 0.848712 (v2) | **0.859408 (v3) ✅** |

### 仓库为什么三个都用了？

本仓库的 baseline 不是"选一个"，而是"用 switch 控制选哪个"：

- 默认：**OneTrans + NSConditioner + RankMixer NS tokenizer**
- 这里 OneTrans 负责主 attention（主要表达力）
- HyFormer 的 MultiSeqQueryGenerator 被用作**初始化** NS tokens 和 Q tokens 的模式（思想被借鉴）
- RankMixer 的 **Token Mixing 思想** 被融合进 RankMixerBlock，用于顶层 boost

### 一个类比：菜系融合

这三个不是"三家打擂台选一家"，而是**三种技法**融合进同一道菜：
- **RankMixer = 刀工**（token reshape 的精确控制）
- **HyFormer = 调味**（query-based 信息凝练的层次感）
- **OneTrans = 主料**（mixed param attention 的主干表达力）

读代码时不要问"这是哪一篇"，而要问"这一块在做哪一种操作"。

---

## 飞机上的论文阅读指南（跳读策略）

### 📗 必读（16h 飞行）

#### 论文 1 - Attention Is All You Need（2h，共 15 页）

- **必读**：§1 Introduction（动机）, §2 Background, §3.1-3.2 Attention 机制, §3.3 Positional Encoding
- **跳过**：§4 Applications（机器翻译细节）, §5 Training, §6 Results, §7 Conclusion
- **关键图**：Figure 1（Transformer 结构图）, Figure 2（Multi-head 示意图）
- **阅读时问自己的问题**：
  - 为什么叫 "Attention"？和 "Attention" 在认知心理学中的含义一样吗？
  - `√d_k` 除数为什么是 d_k 的平方根？
  - Multi-head 为什么用多个小矩阵，不用一个大矩阵？

#### 论文 2 - RoFormer: RoPE（1h，共 15 页）

- **必读**：§3.4 "Proposed approach"（RoPE 的定义和实现）
- **跳过**：§3.1-3.3（历史 PE 方法对比，可以跳）, §4 Theoretical Analysis（除非你爱数学）, §5 Experiments
- **关键图**：Figure 1（旋转的几何解释）
- **阅读时问自己的问题**：
  - RoPE 的"相对位置感知"怎么从 `<R(m)·q, R(n)·k>` 推出来的？（12-code-qa §1 Q4 讲过）
  - 如果 base 从 10000 改成 100，会发生什么？（答：高频 channel 旋转更快，短程关系更敏感）

#### 论文 3 - HyFormer（2.5h，共 7 页，论文短但是核心）

- **必读全文**：论文只有 7 页，没有"跳过"的奢侈
- **重点**：§3 Method，特别是 §3.2-3.4（Multi-Sequence Query Decoding + RankMixer Boost）
- **关键图**：Figure 2（整体架构）, Figure 3（Query Decoding 细节）
- **阅读时问自己的问题**：
  - 为什么 Query Decoding 用 cross-attn 而不是 self-attn？
  - 为什么 RankMixer Boost 在最顶层？

#### 论文 4 - RankMixer（2h，共 14 页）

- **必读**：§3 Method（特别是 §3.2 Token Mixing, §3.3 Per-token FFN）
- **跳过**：§4 Experiments 的详细数字（看结论即可）, §5 Discussion
- **关键图**：Figure 2（Token Mixing 可视化！必看）
- **阅读时问自己的问题**：
  - Token Mixing 和 MLP-Mixer 2021 的差异在哪？（答：MLP-Mixer 用 Linear 做 token mixing 有参数，RankMixer 用 reshape 无参数）
  - Per-token FFN vs Shared FFN 的取舍？

#### 论文 5 - OneTrans（2.5h，共 12 页，核心但稍长）

- **必读**：§3 Method（Mixed Attention + Mixed FFN），§4 关键消融
- **跳过**：§5 实验的 Alibaba 内部 metric 细节
- **关键图**：Figure 2（整体架构）, Figure 3（Mixed Attention mask 结构）
- **阅读时问自己的问题**：
  - Mixed Parameterization 和普通 Transformer 的参数量差距？（答：NS 部分多 M× 参数）
  - 论文用 causal mask for S，本仓库改成 bidirectional，为什么？（答：推荐 domain 字母序无时序）

### 📘 选读（有时间 4-6h 再读）

#### 论文 6 - ESMM（1h 读中文博客，跳过英文论文）

- 直接读 [阿里技术博客中文版](https://zhuanlan.zhihu.com/p/37562283)
- 只需要理解核心图：两 tower 共享 embedding，分别建模 CTR 和 CTCVR
- 飞机上离线：出发前把博客页面用 Safari 的"加入阅读列表"保存

#### 论文 7 - LONGER（30min 看 abstract + Figure 1 + Figure 2）

- 本赛题序列长度 L ≈ 200-500，**LONGER 主要针对 L > 10000 的场景**，和赛题不匹配
- 只需要理解 "top-K cross-attn" 的思路即可
- 不用精读

### 📕 完全跳过（除非你已经读完上面所有）

- 论文 1 的 §5-7（Training, Results, Conclusion）
- 论文 2 的 §4（Theoretical Analysis）
- 论文 4, 5, 7 的 §4-5 Experiments 详细数字

---

## 超有用的"论文速读套路"（节省 50% 时间）

每篇论文都按这个顺序读：

1. **Abstract 1 遍**（2 分钟，知道论文在讲什么）
2. **Figure 1 和最后一个 Figure**（5 分钟，知道架构长什么样）
3. **Introduction 的最后一段**（3 分钟，作者自己总结的 contributions）
4. **Method 一节**（20-40 分钟，细读）
5. **Experiments 的 Table 1**（5 分钟，看最主要的对比数字）
6. **Conclusion**（2 分钟，作者总结）

### 跳读技巧

- **看到"Related Work"直接跳**：99% 的"前人工作综述"你不需要
- **看到公式先看上下文，再看公式**：90% 的公式都能从上下文推出，不需要硬看
- **不懂的符号先猜**：能从上下文推出含义的 80% 都对

---

## 读完后的"消化动作"

每读完一篇论文，写一段 **3 行总结** 到你的笔记：

```
论文 X - [标题]
- 核心创新：一句话
- 对赛题的启发：一句话
- 开放问题（我还不懂的）：一句话
```

这份 3 行笔记比任何读书笔记都有价值。**留下开放问题最重要**——它们是你下一次研究的方向。

---

## 飞机上如果时间不够——最短路径

如果飞行中睡了一半或者体力跟不上，只保证这 **3 件事**：

1. 读完 [`10-background-primer.md`](10-background-primer.md) 的 Part 2（LLM 基础）—— 30 分钟
2. 读完 [`12-code-qa.md`](12-code-qa.md) 的 §3, §5, §8, §9, §10（最核心的 5 个 class）—— 2 小时
3. 看完 3Blue1Brown 的 Attention 可视化视频 —— 27 分钟

**总共 3.5 小时**，你在 LLM 基础 + 赛题核心理解 + Attention 直觉 三件事上都达到 80% 水平。剩下的原论文可以下次再读。

---

**下一步**：带着本文建立的框架，选定你的阅读优先级，打开 [`12-code-qa.md`](12-code-qa.md) 开始代码反问式精读。
