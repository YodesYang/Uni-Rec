# 16 小时飞行阅读计划（blog + 视频 + 代码为主版）

> **与 [`flight-reading-list-2026-05-06.md`](flight-reading-list-2026-05-06.md) 并存**：原版以"读 7 篇英文论文"为主线；本版以"blog + 视频 + 仓库代码"为主线，论文为辅。
> **适用场景**：阅读自诊断显示 B（动机不清）/ C（论文-代码对不上）/ D（多篇关系不清）/ E（背景缺失）。对症下药。
> **预估总时长**：16 小时飞行，扣除吃饭/睡觉/休息 6 小时，有效学习时间 **~10 小时**。

---

## 核心原则（和原版的区别）

| 维度 | 原版 | 本版 |
|---|---|---|
| 主要素材 | 7 篇英文论文 PDF | **Blog（中文优先）+ 视频 + 仓库代码 + 本地 md 讲义** |
| 阅读顺序 | 按论文编号顺序 | **按认知难度阶梯**：背景 → 代码 → 论文 |
| 语言切换 | 英文为主 | **中文优先，英文仅在必要时** |
| 输出成果 | 读完 7 篇 | **读完 3 份本地 md + 看 2 个视频 + 跳读 3 篇论文 + 1 份笔记** |

---

## 时间分配（10 小时有效学习）

```
┌────────────────────────────────────────────────────────┐
│ Stage 1: 建骨架 (2h)                                   │
│   └─ 读 10-background-primer.md  +  看 3Blue1Brown 视频 │
│                                                         │
│ Stage 2: 精读代码 (3.5h) ⭐最核心                       │
│   └─ 读 12-code-qa.md (对照 08-model-py-key-sections)  │
│                                                         │
│ Stage 3: 建立论文地图 (1h)                              │
│   └─ 读 11-paper-roadmap.md + 决定哪些论文要读         │
│                                                         │
│ Stage 4: 选择性读论文 (2.5h)                           │
│   └─ 只读 HyFormer + RankMixer + OneTrans 的 Method 节 │
│                                                         │
│ Stage 5: 产出笔记 (1h)                                 │
│   └─ 写 3 行总结 × 3-5 篇 + 2-3 个开放问题             │
└────────────────────────────────────────────────────────┘
```

---

## Stage 1: 建骨架（起飞后前 2 小时，精神最好时）

### 1.1 读背景速通（45 分钟）

**文件**：[`10-background-primer.md`](10-background-primer.md)

- Part 1（广告 CVR 速通）：15 分钟 - 你有推荐底子，快速过
- Part 2（LLM 底层原理）：**30 分钟，重点**，反复看两遍
- Part 3（趋势）：10 分钟快速浏览
- Part 4（战术建议）：自测 5 分钟

**关键验收**：能答出 Part 2 结尾的 4 个概念题（Autoregressive / Pre-LN / Causal Mask / FlashAttention）。

### 1.2 看 3Blue1Brown 视频（55 分钟）

**视频**：Chapter 5-6 "Attention in transformers, visually explained"

- **必看** Chapter 6（Attention）：27 分钟
- **推荐** Chapter 5（What is a GPT）：27 分钟

**视频来源**（出发前必须下载好，飞机上离线看）：
- YouTube 链接：https://www.youtube.com/watch?v=eMlx5fFNoYc
- 如果有 YouTube Premium：直接 App 里下载
- 如果没有：Mac 上用 `yt-dlp -f 'best[height<=1080]' <url>` 下载 mp4

**看的时候做什么**：
- 暂停在 **Q, K, V 矩阵可视化** 的画面（约 7 分钟处），手画一下自己的版本
- 暂停在 **softmax 归一化** 的画面（约 12 分钟处），想清楚它是怎么把 attention score 变成权重的

### 1.3 短休息（15 分钟）

起来走两步、喝水、看窗外。大脑需要消化刚学的东西。

---

## Stage 2: 精读代码（起飞后 2-5.5 小时，最核心）⭐⭐⭐

### 核心资源三件套

1. **讲义**：[`12-code-qa.md`](12-code-qa.md) ← 主读物
2. **代码**：[`08-model-py-key-sections.md`](08-model-py-key-sections.md) ← 对照
3. **源码**：`baseline/model.py` ← Working Copy 里打开（同一仓库同步）

### 推荐读法：讲义 + 代码 + 手画 三合一

打开 iPad 分屏：
- **左半屏**：12-code-qa.md（讲义）
- **右半屏**：08-model-py-key-sections.md（代码）

**循环过程**：
1. 读讲义的一个 Q
2. 滚到代码对应段，看 3-5 行
3. 在 GoodNotes 里手画 tensor shape 变化
4. 回讲义读下一个 Q

### 精读顺序（按重要性排）

| 顺序 | 讲义节 | 代码位置 | 预计用时 |
|---|---|---|---|
| 1 | §1 RotaryEmbedding | L26-L92 | 20 min |
| 2 | §3 RoPEMultiheadAttention ⭐ | L117-L243 | **40 min**（手画 shape） |
| 3 | §5 RankMixerBlock ⭐ | L315-L414 | **40 min**（手画 token mixing） |
| 4 | §8 MultiSeqHyFormerBlock ⭐ | L1054-L1193 | 30 min |
| 5 | §9 OneTransMixedAttn ⭐ | L1194-L1367 | **40 min**（手画 mask 矩阵） |
| 6 | §10 OneTransNSConditioner ⭐⭐⭐ | L1431-L1497 | 30 min |
| 7 | §2 SwiGLU + §4 CrossAttention | L100-L314 | 15 min |
| 8 | §6 MultiSeqQueryGenerator | L563-L705 | 15 min |
| 9 | §7 三个 Encoder 对比 | L706-L1053 | 20 min |

**总计**：~3.5 小时，和 Stage 2 配额一致。

### 关键任务（不做就白读）

每精读完一个 ⭐ 节，必须做以下其中一项：
- **手画**：在 GoodNotes 里画 tensor shape 演化
- **改写**：用中文口语把代码功能复述一遍（写在讲义边上）
- **反问**：自己提一个"如果不这样会怎样"的问题并尝试回答

### 休息（15 分钟）

Stage 2 耗神，中间必须插一次休息。

---

## Stage 3: 建立论文地图（起飞后 5.5-6.5 小时）

### 3.1 读论文地图（45 分钟）

**文件**：[`11-paper-roadmap.md`](11-paper-roadmap.md)

- 看时间线图（5 分钟）
- 理解 HyFormer / RankMixer / OneTrans 三者关系（15 分钟，这是你 D 诊断的核心痛点）
- 读每篇论文的"跳读指南"（20 分钟）
- 决定你接下来 Stage 4 要读哪几篇（5 分钟）

### 3.2 决策时间（15 分钟）

**关键决策**：Stage 4 的 2.5 小时读哪几篇论文的哪些章节？

**推荐默认组合**（高优先级）：
- HyFormer 全文（2.5 小时 的 50%）
- OneTrans §3 Method（30%）
- RankMixer §3 Method（20%）

**替代组合 A**（如果 Stage 2 做得很顺，基础扎实）：
- HyFormer 全文
- OneTrans 全文
- Attention Is All You Need §1-3

**替代组合 B**（如果 Stage 2 做得吃力，时间紧）：
- 只读 HyFormer 的 §3（1h）
- 只读 OneTrans 的 Figure 2/3 + §3.2（1h）
- 剩下时间重读 12-code-qa §9 和 §10

---

## Stage 4: 选择性读论文（起飞后 6.5-9 小时）

### 关键心态

**不要试图读懂每个公式**。带着 Stage 2 的问题去论文里找答案：

- "RoPE 的复数解释怎么写"
- "RankMixer 的 token mixing 在论文里怎么可视化"
- "OneTrans 的 mixed mask 为什么这么设计"

论文是"查字典"，不是"从头到尾精读"。

### 读的时候做什么

1. 找到对应章节（不是从第一页开始读）
2. 读两段 → 合上论文 → 在讲义或笔记上写"这一段讲的是什么"
3. 如果写不出来，回去重读
4. 写得出来，翻下一章节

### 特别提示

**英文读累了就不要硬撑**。飞机上注意力有限，硬读只会低效。休息 10 分钟再继续。

---

## Stage 5: 产出笔记（起飞后 9-10 小时）

### 目标：产出一份 `EXP-XXX-flight-learning-notes.md`

用 Working Copy 在 iPad 上直接新建 md 文件（提前确认能写入 git）。

### 笔记模板

```markdown
# 飞行学习笔记 - 2026-05-06

## 论文 3 行总结

### HyFormer
- 核心创新：___
- 对赛题启发：___
- 开放问题：___

### OneTrans
- 核心创新：___
- 对赛题启发：___
- 开放问题：___

### RankMixer
- 核心创新：___
- 对赛题启发：___
- 开放问题：___

## 代码理解检查点

- [ ] 能口述 RoPE 的旋转原理（无公式）
- [ ] 能手画 Token Mixing 的 tensor shape 变化
- [ ] 能口述 HyFormer 三步曲
- [ ] 能口述 OneTrans mixed mask 的四个象限含义
- [ ] 能口述 NSConditioner 为什么带 +0.010 AUC

## 飞机上产生的 idea

（有 1 个就够，3 个是惊喜）

1. ___
2. ___
3. ___

## 落地下来后的行动

- 本周内：___
- 这两周内：___
- 比赛结束前：___
```

### 为什么这一步必须做

**学习 = 输入 + 输出**。只输入不输出的学习，3 天后会忘掉 80%。

**强制自己写笔记**，即使写得烂，也能把知识"固化"到你的认知结构里。

---

## 备忘：出发前必备物品

### iPad 端

- [ ] Working Copy 装好，仓库已 clone
- [ ] 本地确认 `docs/99-raw/flight-reading/` 所有 md 和 PDF 都能离线打开
- [ ] GoodNotes（或 Notability / PDF Expert）装好，用于手画
- [ ] 3Blue1Brown 视频已下载到本地（2 集约 1GB）
- [ ] Taio（可选）装好用于更好的 md 阅读体验

### 出发前 30 分钟要做的事

- [ ] Mac 上 `git push` 确保最新版在 GitHub
- [ ] iPad 的 Working Copy 里 fetch + pull 一次，确认同步
- [ ] 飞机座椅上测试：离线能打开 08 / 10 / 11 / 12 这 4 个 md 吗？
- [ ] 充满电的耳机（看视频必备）

---

## 如果飞行不顺利的 Plan B

飞机上可能遇到的意外：
- **乘客太吵**：只看文字不看视频，优先 Stage 2（代码）
- **座位没电源**：iPad 电量紧张，关闭视频 Stage，把省下的电用在读 md
- **头晕眼花**：只看视频不读 md。3Blue1Brown 看 3 遍也是大收获
- **完全睡过去 10 小时**：出舱后没学到任何东西也没关系，飞机本来就是补觉好地方。下次在家用碎片时间补

---

## 到达后的行动清单

### 第一天（抵达当天）

- [ ] commit + push 飞机上写的 EXP-XXX-flight-learning-notes.md
- [ ] 挑一个飞机上产生的 idea，写 3 行 ADR 草稿

### 第一周内

- [ ] 把 idea 变成具体的 fast proxy 实验
- [ ] 在平台上跑 proxy，1500 steps 看信号
- [ ] 根据信号决定是否升级到 full train

---

**重申核心理念**：**16 小时不是读完 7 篇论文的时间，是建立可持续学习框架的时间**。比赛还有 19 天，飞机只是起点，不是终点。飞机上学的每一点知识，都应该为你回来后的 18 天冲刺做铺垫。

**祝一切顺利。**
