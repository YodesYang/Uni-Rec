# 16 小时飞机论文精读清单（2026-05-06 出差）

> **场景**：16 小时飞行，无稳定网络；目标是从零建立 transformer + HyFormer/OneTrans/RankMixer 武器库。
> **离线装备**：本清单 + 5 篇 PDF + `baseline/model.py` 打印版 + A4 白纸 + 双色笔。
> **配套资料**：`docs/00-context/model-py-section-map.md`（论文 ↔ 代码对照地图）。

---

## 使用方法

1. **按顺序读**：论文 1 → 2 → 3 → 4 → 5 → 6（不要跳顺序，后面依赖前面）
2. **每篇先看"带着这 N 个问题读"**，读的时候把答案写在纸上，不是脑子里
3. **读完立刻翻对应仓库代码**（见 `model-py-section-map.md`），**论文 ↔ 代码 side-by-side** 是最高杠杆
4. **验收标准不过就回去重读**，不要自己骗自己"大概懂了"

---

## 时间分配建议

| 时段 | 内容 | 小时 | 累计 |
|---|---|---|---|
| 起飞后 1-3h | **论文 1** Attention Is All You Need（地基） | 2.5h | 2.5 |
| 3-5h | **论文 2** RoPE / RoFormer | 2h | 4.5 |
| 5-6h | 🍽 中场休息 + 吃饭（纸上画 MHA 数据流复盘） | 1h | 5.5 |
| 6-9h | **论文 3** HyFormer（对照 `MultiSeqHyFormerBlock`） | 3h | 8.5 |
| 9-11h | **论文 4** RankMixer（对照 `RankMixerBlock`） | 2h | 10.5 |
| 11-12h | 💤 小睡 | 1h | 11.5 |
| 12-14h | **论文 5** OneTrans（对照 `OneTransBlock`） | 2h | 13.5 |
| 14-15h | **论文 6** ESMM（补广告 CVR 业务） | 1h | 14.5 |
| 15-16h | 复盘：纸上画"如果让我做 T5/MoE/attention variant 分别改哪里" | 1h | 16 |

---

## 论文 1: Attention Is All You Need（Vaswani et al., 2017）

- **URL**: https://arxiv.org/abs/1706.03762
- **PDF 下载**：arxiv PDF 直链 + 打印

### 带着这 5 个问题读

1. Query / Key / Value 为什么要用**三个独立的线性投影**？能不能共享？共享后会有什么问题？
2. Scaled Dot-Product 里为什么除以 `sqrt(d_k)` 而不是 `d_k` 或不除？（提示：softmax 输入方差控制）
3. Multi-head 是"真的并行算 h 次 attention"还是"一次矩阵乘 + reshape"？**画出 `[B, L, D] → [B, h, L, d/h]` 的 reshape 过程**
4. Position encoding 为什么选 sin/cos？它能表示"相对位置"吗？（为论文 2 铺垫）
5. Decoder 的 causal mask 在矩阵里具体长什么样？**在纸上手画一个 4×4 的 mask 矩阵**（1 = allowed, 0 = blocked）

### 重点看

- §3.2.1 Scaled Dot-Product Attention（公式 1）
- §3.2.2 Multi-Head Attention（公式 2-3 + Figure 2 右图）
- §3.3 Position-wise FFN
- §3.5 Positional Encoding（公式 4-5）

### 可跳过

- §4 Why Self-Attention（比较表格，对本赛题不关键）
- §6 Training Details（超参表，用到再查）

### 验收标准

**合上论文**，在白纸上从零画出单层 transformer encoder 的数据流：
- 输入 shape `[B, L, D]` →
- 经过 Q/K/V 投影后 shape？→
- reshape 成 multi-head 后 shape？→
- attention 矩阵 shape？→
- 加 mask 的位置？→
- 输出 shape？→
- FFN 输入 / 输出 shape？→
- 残差连接在哪里？→
- LayerNorm 在哪里（Post-LN vs Pre-LN）？

**每一步标注维度**，画不出来就回去重读。

---

## 论文 2: RoFormer / RoPE（Su et al., 2021）

- **URL**: https://arxiv.org/abs/2104.09864
- **核心价值**：LLaMA / ChatGLM / Qwen 全系 + 你仓库 `RotaryEmbedding` 都是它

### 带着这 3 个问题读

1. **三种位置编码对比**（绝对位置编码 / 相对位置编码 / RoPE）：三者**在 attention 矩阵里各自怎么注入位置信息**？各自的缺点？
2. RoPE 的"旋转"到底旋转的是什么？
   - 答案：在 Q 和 K 的每对相邻维度 `(x_{2i}, x_{2i+1})` 上，把它视为 2D 向量，旋转角度 `θ_i × position`，其中 `θ_i = base^(-2i/d)`
3. 为什么 RoPE 同时满足"绝对位置感知"和"相对位置感知"？
   - 提示：`<RoPE(q, m), RoPE(k, n)>` 只依赖 `m - n`，但每个 token 的表示里又编码了绝对位置

### 重点看（别跳！）

- §3.4 旋转矩阵推导（公式 12-15）—— **整篇灵魂，数学是高中三角函数**
- §3.4.2 Long-term Decay（RoPE 天然带长距离衰减，不用显式加）
- Figure 1 的几何直觉图

### 可跳过

- §2 Related Work（了解历史背景而已）
- §4.3 的各种 GLUE 实验数字

### 验收标准

1. 在纸上手写公式（15）：`RoPE(x, m) = [x_0 cos(mθ_0) - x_1 sin(mθ_0), x_0 sin(mθ_0) + x_1 cos(mθ_0), ...]`
2. 解释清楚"为什么 `inv_freq = 1/base^(2i/d)`"里的 base=10000 是任意选择的
3. 在纸上画出 d=4 时 RoPE 把一个 4 维 Q 向量旋转后的结果

---

## 论文 3: HyFormer（抖音搜索）

- **URL**: https://arxiv.org/abs/2601.12681
- **核心价值**：**你仓库 `PCVRHyFormer` 主模型 + `MultiSeqHyFormerBlock` 就是它**。必须读穿。

### 带着这 7 个问题读

1. HyFormer 和 "Encode-then-Interaction"（DIN 那套）的根本区别是什么？
2. **Global Tokens (Query tokens)** 的作用是什么？为什么不直接把 NS feature 拼到序列最后？
3. **Query Decoding**（cross-attn）和 **Query Boosting**（RankMixer）的分工是什么？
   - 前者：Q 从序列里 "吸" 信息
   - 后者：Q 和 NS token 之间做跨特征交互
4. 为什么每个 domain 要**独立的 Q token 集**而不是共享？trade-off 是什么？
5. 论文 Table 1 里 HyFormer vs LONGER+RankMixer **只领先 0.17%**，作者凭什么说自己更好？
   - 提示：看 FLOPs / 推理延迟那一列
6. `num_queries * num_sequences + num_ns` 为什么必须整除 `d_model`？（看 RankMixer 的 token mixing）
7. HyFormer 在 **pCVR 任务上** 是否有特殊处理？（原论文是搜索场景，迁移到广告 CVR 有什么 gap？）

### 重点看

- Figure 2 整体架构图 —— **这张图飞机上反复看**
- §3.2 Query Decoding Block
- §3.3 Query Boosting Block
- §3.4 Multi-Sequence Extension
- Table 1 消融实验

### 对照仓库（边读边开）

- `baseline/model.py:906 MultiSeqHyFormerBlock` → 对应论文 §3.2-3.4
- `baseline/model.py:1689 PCVRHyFormer` → 整体搭建
- `baseline/model.py:315 RankMixerBlock` → 对应论文 Query Boosting
- `baseline/model.py:415 MultiSeqQueryGenerator` → 对应论文 Query tokens 初始化

### 验收标准

1. 在纸上画出完整的 HyFormer 一个 block 的数据流：
   - 输入：Q tokens (B, Nq, D) × S 个序列 + NS tokens (B, Nns, D) + Seq tokens (B, L_i, D) × S
   - Step 1 Sequence Evolve: 每个序列独立过 encoder
   - Step 2 Query Decode: 每个 Q 集合 cross-attend 到对应序列
   - Step 3 Query Boost: 所有 Q + NS 拼起来过 RankMixer
   - 输出 shape？
2. 解释清楚你仓库 `num_queries=2, num_sequences=4, user_ns_tokens=5, item_ns_tokens=2` 的 **T = 2×4 + 7 = 15**，所以 `d_model` 必须是 15 的倍数（`d_model=60` 满足，`d_model=64` 也要调整到 15 的倍数）

---

## 论文 4: RankMixer（抖音 feed）

- **URL**: https://arxiv.org/abs/2507.15551
- **备用（如找不到原文）**: https://www.huaxiaozhuan.com/applications/recommendation/ctr_prediction/chapters/2025_RankMixer.html

### 带着这 5 个问题读

1. **Multi-head Token Mixing** 为什么是"参数无关"的？它实际上是一个 reshape + transpose 操作 —— **手画 T=4, d_model=8, d_sub=2 的具体变换过程**
2. **Per-token FFN** vs 共享 FFN 的核心区别？为什么说"避免 dominant feature 淹没长尾"？
3. RankMixer 的 **MFU 4.5% → 45%** 为什么能提升 10 倍？（硬件感知设计的具体内容）
4. **Sparse-MoE 变体** 的 gating 怎么设计？每 token 激活几个 expert？（**这是你合规多模型路线的子弹**）
5. 为什么 RankMixer 在广告场景特别适合？（推理延迟敏感 + 特征数量多）

### 重点看

- Figure 1 / 2 架构图
- **Token Mixing 的数学形式**（reshape 公式）—— 本篇灵魂
- §4 MFU 分析（理解"硬件感知"是什么意思）
- MoE 变体那一节（如果原文有）

### 对照仓库

- `baseline/model.py:315-414 RankMixerBlock` → 对应论文核心
- **重点读 `RankMixerBlock.token_mixing` 方法**：
  - `(B, T, D) → (B, T, T, d_sub)` 切分子空间
  - `(B, T, T, d_sub) → (B, T, T, d_sub)` transpose 交换 token 和 subspace 维度
  - `(B, T, T, d_sub) → (B, T, D)` flatten 回去
  - 这就是"参数无关 token mixing"的全部实现

### 难点重点看

- Token mixing 的数学形式。本质是"沿 token 维度做一个固定的 permutation"，没有可学习参数
- **在纸上画出 T=4, d_model=8, d_sub=2 的具体 reshape 过程**，4 个 token 经过 mixing 后 channel 0 来自哪些原 token？

### 验收标准

1. 能 3 分钟讲清"RankMixer 凭什么比 attention 便宜又不差"
2. 能写出 token mixing 伪代码（就 3 行 reshape/transpose）
3. 能说出 `d_model % T != 0` 时 token mixing 为什么不成立

---

## 论文 5: OneTrans（Mixed Parameterization）

- **URL（解读）**: https://datawhalechina.github.io/fun-rec/chapter_6_scaling/5.one_trans.html
- **原论文**: 如果能在 arxiv 找到 "OneTrans Mixed Parameterization" 就下载；找不到用解读版
- **核心价值**：**你仓库已经有 `OneTransBlock` 实现了（1000+ 行），EXP-013 跑到 0.859（+0.010）**。这是当前最接近"结构性创新"的子弹

### 带着这 6 个问题读

1. OneTrans 与 HyFormer 在**NS token 处理上**的根本分歧是什么？
   - HyFormer: global query tokens 桥接 S/NS，cross-attention 解耦
   - OneTrans: S 和 NS 拼成同一 attention 序列，mixed mask + mixed param
2. **Mixed Parameterization** 的 "S-token 共享 W / 每个 NS-token 独立 W" 具体长什么样？参数量怎么算？
   - 若 S 共享 (Wq, Wk, Wv, Wo)，参数量 `4 × d²`
   - 若 每个 NS_j 独立 (Wq^j, Wk^j, Wv^j, Wo^j)，参数量 `4 × d² × M`（M 个 NS token）
3. **Mixed Causal Attention** 的 mask 矩阵手画一遍：
   - S_i → S_0..S_i（causal within S）
   - NS_j → 所有 S（full）+ NS_0..NS_j（causal within NS）
   - **你仓库 `OneTransMixedAttn._build_causal_mixed_mask` 里 S-S 实际用的是 BIDIRECTIONAL 不是 causal**，思考为什么（注释提到"S tokens 来自不同 domain，按字母序拼接不具温序列意义"）
4. **Pyramid Stack** 为什么"逐层缩 query 数"能蒸馏到尾部 NS-token？
5. **KV Caching (Cross-Request)** 的工程价值是什么？（本赛题 offline 评估用不上，但报告可以讲）
6. 论文声称 +5.68% GMV，哪些是 Mixed Param 贡献、哪些是 Pyramid Stack 贡献？拆解 ablation

### 对照仓库（重点）

- `baseline/model.py:1044 OneTransMixedAttn` → 对应论文 §Mixed Attention
- `baseline/model.py:1218 OneTransMixedFFN` → 对应论文 §Mixed FFN
- `baseline/model.py:1281 OneTransNSConditioner` → **EXP-013 关键差异**
  - v2 无 NS conditioner: full train 0.848712
  - v3 有 NS conditioner: full train 0.859408（+0.010！）
  - **为什么 NS conditioner 带来 +0.010 提升？飞机上想清楚这个就是这 2 小时最大收获**
- `baseline/model.py:1350 OneTransBlock` → 整体组装

### 验收标准

1. 在纸上画出 Mixed Attention 的 mask 矩阵（L=4, M=3 的小例子，共 7×7 矩阵，标出每个位置 allowed/blocked）
2. 解释清楚 OneTransNSConditioner 的作用机制
3. 能回答："如果我要给 OneTrans 加 Pyramid Stack（每层缩 query 数），应该改仓库哪几处？"

---

## 论文 6: ESMM（Alibaba 2018）

- **URL**: https://arxiv.org/abs/1804.07931
- **核心价值**：广告 pCVR 的**开山经典**，30 分钟读完。你没做过广告，补这一课直接提升报告的业务理论高度

### 带着这 3 个问题读

1. **pCTR / pCVR / pCTCVR** 三者的关系？
   - `pCTCVR = pCTR × pCVR`（乘积关系）
2. **Sample Selection Bias (SSB)** 和 **Data Sparsity (DS)** 问题具体是什么？
   - SSB：传统 CVR 只在"被点击"样本训练，但推理时要对"所有曝光"打分 → 训练/推理分布不一致
   - DS：点击样本只占曝光的 < 1%，CVR 标签极稀疏
3. ESMM 为什么**在整个曝光空间建模**就能同时解决 SSB 和 DS？
   - 用 pCTCVR（全空间可观测）+ pCTR（全空间可观测）间接监督 pCVR（用除法，避免直接训 pCVR 的选择偏差）

### 对这个比赛的价值

读完你会明白 **T15（"本赛题数据全是点击后样本，没有曝光未点击"）为什么这么反常** → 这个比赛其实是在做 **click-conditional CVR**，不是标准工业界 pCVR：
- 标准工业界 pCVR：曝光空间建模（ESMM 路线）
- 本赛题：post-click conditional，数据已经过 click filter

**报告里讲清这个 task framing 是大加分项**，KDD 评审大概率会 challenge 这个点。

### 验收标准

能用 3 分钟给一个没做过广告的人讲清 pCVR 的 SSB 问题，以及为什么 ESMM 能解决。

---

## 飞机上带什么

### 必带
- ✅ 这份清单的打印版
- ✅ 配套 `docs/00-context/model-py-section-map.md` 打印版
- ✅ 5 篇论文 PDF（提前下载到 iPad / Kindle）
  - [ ] Attention Is All You Need（1706.03762）
  - [ ] RoFormer（2104.09864）
  - [ ] HyFormer（2601.12681）
  - [ ] RankMixer（2507.15551 或 huaxiaozhuan 解读）
  - [ ] OneTrans（datawhale 解读 + 尝试搜原文）
  - [ ] ESMM（1804.07931）
- ✅ `baseline/model.py` 打印版（2396 行，建议只打印 §RoPE / §SwiGLU / §RoPEMultiheadAttention / §RankMixerBlock / §MultiSeqHyFormerBlock / §OneTrans* 几个关键段，约 600 行）
- ✅ **A4 白纸至少 10 张**（画 attention 矩阵、reshape、mask 图必备）
- ✅ **两色笔**：黑笔画主图，蓝笔标注维度

### 可选
- 降噪耳机 + 轻量 BGM（有助于长时间专注）
- 保温杯（16 小时注意补水）
- 眼药水

---

## 读完后的飞机落地动作

1. **把 A4 手稿拍照或扫描**，回来后归档到 `docs/50-paper/flight-notes/`（给报告答辩用）
2. **把"不懂清单"落成 `docs/20-tasks/active.md` 的 T-block**（比如"T16 OneTrans Pyramid Stack 可行性分析"）
3. **给我发一份你的"15-16h 复盘答案"**（T5 / MoE / attention variant 分别改哪里），我帮你评估哪个 ROI 最高
