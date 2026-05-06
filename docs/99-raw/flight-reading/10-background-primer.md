# 广告 pCVR + LLM 背景速通

> **用途**：用 30 分钟补齐广告/LLM 的"常识骨架"。阅读顺序建议在 12-code-qa 之前，作为它的背景铺垫。
> **假设的读者**：7 年推荐经验，熟悉 CTR / user-item 召回排序，但没做过广告，没研究 LLM 底层。
> **写作风格**：用推荐类比广告，用推荐类比 LLM。

---

## Part 1: 广告系统的 "K" 字——CTR / CVR / GMV / eCPM

### 推荐 vs 广告的本质差异

| 维度 | 推荐（你熟悉的） | 广告（赛题所在） |
|---|---|---|
| 谁出内容 | 平台/UGC 创作者 | **广告主付钱投的 creative** |
| 谁决定排序 | 用户兴趣匹配度 | 多方博弈：**用户兴趣 × 广告主出价 × 平台收入** |
| 排序目标 | CTR / 停留时长 / 多样性 | **eCPM = Bid × CTR × CVR**（或变体） |
| 转化定义 | 点击 / 播放完成 / 收藏 | 点击 + **后续转化**（下载、注册、购买） |
| 反馈信号 | 秒级 | **延迟反馈**（购买往往 D+7 才发生） |

### 核心概念（3 分钟速通）

- **CTR**（Click-Through Rate）= P(点击 | 曝光)
- **CVR**（Conversion Rate）= P(转化 | 点击)  ← **赛题就是预估这个**
- **pCVR** = predicted CVR
- **CPA**（Cost Per Action）= 广告主每次转化愿意支付的价格
- **eCPM**（effective Cost Per Mille）= 千次曝光的预期收入
  - 在 CPA 结算下：`eCPM = CPA × CTR × CVR × 1000`
  - **这就是为什么 pCTR/pCVR 预估准确度直接决定平台收入**

### 为什么 pCVR 比 pCTR 难？

| 问题 | CTR | CVR |
|---|---|---|
| 正样本比例 | 1%-10% | **0.1%-1%**（稀疏 10×） |
| 反馈延迟 | 秒级 | **小时到天级** |
| 样本选择偏差 | 小 | **大**（CVR 只在"点击了"的样本上有标签） |
| 特征可用性 | 曝光时全部可见 | 点击后的交互特征也重要 |

### 样本选择偏差（SSB）——ESMM 论文的核心问题

**问题**：训练 CVR 模型时，我们只有"被点击过的曝光"的 label。但线上推理时，要对"所有曝光"（没点击的也要）预估 pCVR。训练和推理分布不一致。

**ESMM 的解法**：同时建模 **CTR** 和 **CTCVR = CTR × CVR**，用全样本训练 CTR，用"点击过"训练 CTCVR，CVR 由 `CTCVR / CTR` 隐式得到。

**飞机上快速吸收**：读 [阿里技术博客中文版](https://zhuanlan.zhihu.com/p/37562283) 比读论文快 3×。核心图只有一张：两个 tower 共享底层 embedding，分别预测 CTR 和 CTCVR，loss = BCE(pCTR) + BCE(pCTCVR)。

### 延迟反馈（Delayed Feedback）

用户**点击广告 → 下载 App → D+3 注册 → D+7 付费**。训练时的 "conversion label" 在哪个时间点取？

这个赛题设计方已经简化了（label 在数据里直接给），但实际工业场景会用：
- Importance sampling
- Survival analysis
- 短窗期 proxy label + 长窗期 correction

### 本赛题的"广告"维度简化

看一眼 `baseline/dataset.py` 你会发现：
- **没有广告主 bid**（不涉及出价博弈）
- **没有费率**（不涉及 eCPM 排序）
- **只有 pCVR 预估**，评估用 AUC

所以本赛题**本质还是一个 CVR 二分类问题**，比真实广告系统简化得多。你 7 年推荐经验的 CVR 建模技巧 95% 都适用。差异仅在：
- 样本稀疏（正样本 ~1%）
- 序列建模权重更高（用户行为多 domain）

---

## Part 2: LLM 底层原理速通——只讲你需要的

### 为什么推荐算法工程师现在必须懂 LLM？

2024-2026 年的显著趋势：**推荐/广告领域开始大量借鉴 LLM 架构**。
- 序列推荐 → 全面 Transformer 化（HSTU, Pinterest, Meta FBG）
- 生成式推荐（用 LLM 做 item 生成）
- 架构复用：Attention / RoPE / RMSNorm / SwiGLU 进入推荐代码库

本赛题的 baseline 就是 Transformer + RoPE + SwiGLU，全是 LLM 来的东西。**不懂 LLM，读这个代码就是读"黑话"**。

### Transformer 三块积木（你只需要理解这三个）

#### 积木 1: Self-Attention

**核心公式**（就一行）：
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

**做什么**：每个 token 看其他所有 token，算一个加权平均。权重由 "我（Q）和它（K）的相似度" 决定。

**3 分钟直觉**：
- 类比 **Retrieval**：Q 是"查询向量"，K 是"所有候选的 key"，V 是"候选的内容"
- `softmax(QK^T/√d)` 产出一个 L×L 的注意力矩阵 A，A[i][j] = "token i 对 token j 的注意权重"
- 最终每个 token 的输出 = 对 V 按注意力加权求和

**和推荐里的什么像**：DIN（Deep Interest Network）——用候选 item 做 query，去 attention 历史行为。DIN 本质就是一个 cross-attention。你已经会了。

#### 积木 2: Multi-Head

把 Q, K, V 切成 H 份（每份 d_head = d_model / H 维），每个 head 独立做 attention，最后拼回来。

**为什么**：每个 head 学不同的"注意模式"。head_0 可能学"近的更重要"，head_1 学"相同 category 更重要"。类似 CNN 的多卷积核。

#### 积木 3: FFN（Position-wise Feed-Forward）

标准 Transformer Block 里 attention 之后接一个 FFN：
```
FFN(x) = W2 · ReLU(W1 · x)
```

**做什么**：每个 token 独立过一个 2 层 MLP。注意力负责 token 间交互，FFN 负责 token 内 refine。

**SwiGLU** = FFN 的升级版（上面 12-code-qa §2 讲过）。LLM 全用 SwiGLU。

### Transformer Block = Attention + FFN + 两个 LayerNorm + 残差

一个 Transformer 层的完整计算：
```
x = x + Attention(LN(x))   # Pre-LN 写法（现代主流）
x = x + FFN(LN(x))
```

堆 N 层就是 N 层 Transformer。GPT-3 有 96 层，LLaMA-2-7B 有 32 层，本赛题 baseline 通常 2-4 层。

### 位置编码（Position Encoding）—— RoPE vs 其他

Transformer 没有循环结构，不知道 token 的顺序。必须显式加位置信息。

历史演进：
1. **Sinusoidal**（原论文）：`x + sin/cos(pos)` 加法编码
2. **Learned**（GPT-2）：可学习的 position embedding
3. **RoPE**（LLaMA / 本赛题）：用旋转矩阵乘法编码相对位置
4. **ALiBi**：直接在 attention score 上加线性偏置

RoPE 在 2021 后成为 LLM 主流，**原因看 12-code-qa §1 Q2**。

### LLM 从 Transformer 的三步跃迁

懂了上面的 Transformer Block，你已经理解 **Encoder-only（BERT）** 和 **Decoder-only（GPT）** 的 90%。

| 演进阶段 | 关键差异 |
|---|---|
| Transformer 2017 | Encoder-Decoder 结构（翻译任务） |
| BERT 2018 | 只保留 Encoder，masked LM 预训练 |
| GPT 2018-2022 | **只保留 Decoder**，causal mask，自回归生成 |
| LLaMA / DeepSeek 2023+ | Decoder + RoPE + SwiGLU + RMSNorm + GQA |

**本赛题的 encoder**：类 BERT（双向 self-attn），因为序列是"已发生的行为"不是"待生成的内容"。

### 你需要知道的 4 个 LLM 概念（之后读论文用得到）

1. **Autoregressive（自回归）**：生成第 t 个 token 时只能看前 t-1 个，需要 causal mask。GPT 系列都是这样。
2. **Pre-LN vs Post-LN**：LayerNorm 放在 residual 之前（Pre-LN，稳定）或之后（Post-LN，原论文，训练难）。现代主流 Pre-LN。
3. **Causal Mask**：下三角矩阵，禁止看未来 token。本赛题 OneTrans 的 S→S 故意**不用** causal（见 12-code-qa §9 Q4）。
4. **FlashAttention**：一种 attention 的 CUDA 实现，把 O(L²) 显存降到 O(L)。PyTorch 2.0+ 的 `F.scaled_dot_product_attention` 底层可能调用它。

---

## Part 3: 为什么广告/推荐开始"LLM 化"——2024-2026 趋势

### 问题背景：传统推荐模型的天花板

传统 DNN 推荐（Wide&Deep, DeepFM, DIN, SIM）的瓶颈：
- **参数利用效率低**：embedding table 占 99% 参数，backbone 很薄
- **序列长度受限**：DIN/SIM 处理千级序列已经吃力
- **跨特征交互弱**：特征 cross 靠人工设计 FM / FiBiNET

### LLM 带来的三个"借鉴点"

1. **Attention 替代人工特征交叉**：Self-attention 自动学习 all-pairs 交互
2. **Transformer 堆叠替代 DNN 堆叠**：残差 + LN 让训练深网络变容易
3. **Scaling Law**：LLM 证明"模型越大越好"，推荐开始追求 100B+ 参数

### 但是——推荐/广告有 LLM 没有的约束

1. **推理时延**：LLM 生成一条回复可以 1 秒，广告竞价 30ms 上限
2. **特征异构**：LLM 全是 token，推荐有 sparse / dense / sequence 三种特征
3. **长尾分布**：item 分布极端长尾，LLM 的词表是均衡的

### 本赛题 baseline 的"LLM 化"程度

| 组件 | 来源 | 状态 |
|---|---|---|
| Self-attention | Transformer 2017 | ✅ 有（RoPEMultiheadAttention） |
| RoPE | LLaMA 2023 | ✅ 有 |
| SwiGLU | LLaMA 2023 | ✅ 有 |
| Pre-LN | GPT-2+ | ✅ 有 |
| RMSNorm | LLaMA | ❌ 用 LayerNorm（差异小） |
| Mixed Parameterization | OneTrans 2026 | ✅ 有（推荐特色） |
| Token Mixing (无 attn) | MLP-Mixer 2021 | ✅ 有（RankMixer） |

**重点理解**：本赛题 baseline 是 "LLM 组件 + 推荐特色设计" 的混合体，不是纯 LLM 也不是纯传统推荐。

---

## Part 4: 你的优势 vs 劣势 —— 打比赛的战术建议

### 你的优势（7 年推荐经验）

1. **特征工程直觉强**：知道什么 feature cross 容易涨点，知道 label leak 怎么防
2. **训练调参经验丰富**：lr / batch size / warmup / clip 都熟
3. **业务理解深**：看到 CVR 预估知道"延迟反馈"和"样本选择偏差"的坑
4. **Ad-hoc 分析能力强**：slice 诊断、分布对比这些你都会

### 你的劣势（本赛题语境）

1. **LLM 底层原理不熟**：读到 RoPE / SwiGLU 这些会卡
2. **读英文论文慢**：但不是大问题（看 blog/视频更快）
3. **广告特定机制不熟**：但本赛题简化成 CVR 分类，不是大问题

### 飞机上应该做什么（优先级）

1. **最高**：读 `12-code-qa.md`（本文的姊妹文件），10 个关键 class 搞懂 → **直接对准赛题**
2. **次高**：读 `11-paper-roadmap.md` 建立论文关联 → **帮你选择性读论文**
3. **必读**：本文 Part 2（LLM 基础）反复看两遍 → **补齐你的最大短板**
4. **选读**：3Blue1Brown 的 Attention 可视化视频 → **直觉升级**
5. **性价比中等**：7 篇原论文只读指定章节（见 11-paper-roadmap） → **不要全读**

### 飞机上**不应该**做什么

- ❌ 试图把 7 篇论文全部读完（时间不够，且边际收益递减）
- ❌ 抠 RoPE 公式的复数推导（除非你天生爱数学）
- ❌ 在纸上手写 attention 的所有 tensor shape（手画 1 次即可）

---

## 自测（5 分钟）

如果下面 5 个问题你都能脱口而出，本文达到目的：

1. **业务侧**：为什么 pCVR 比 pCTR 样本更稀疏？ESMM 如何解决样本选择偏差？
2. **LLM 侧**：Self-attention 的计算复杂度为什么是 O(L²)？为什么现代 LLM 用 Pre-LN 不用 Post-LN？
3. **位置编码**：RoPE 和 Sinusoidal PE 的核心差异？为什么 RoPE 是现代主流？
4. **架构趋势**：LLM 给推荐/广告带来的三个借鉴点？推荐/广告有哪些 LLM 不具备的约束？
5. **赛题理解**：本赛题 baseline 属于"LLM 化"到什么程度？哪些是 LLM 来的，哪些是推荐特色？

答不出的回到对应 Part 再读。

---

**下一步**：带着这些背景去读 [`11-paper-roadmap.md`](11-paper-roadmap.md)（论文地图）或 [`12-code-qa.md`](12-code-qa.md)（代码反问式解读）。
