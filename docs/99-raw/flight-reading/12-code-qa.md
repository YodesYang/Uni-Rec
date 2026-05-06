# `baseline/model.py` — 反问式解读（飞行中文讲义）

> **读法**：对照 [`08-model-py-key-sections.md`](08-model-py-key-sections.md) 的代码段读本文件。本文不贴代码（避免和 08 重复），只讲**为什么这么设计**、**不这样会怎样**、**论文对应位置**。
> **语言风格**：中文口语，类比你熟悉的推荐系统。
> **标记**：⭐ 关键理解点，⭐⭐ 不懂就读不下去的节点。

---

## 目录

1. [RotaryEmbedding + apply_rope_to_tensor — RoPE 到底在做什么](#1-ropeembedding--apply_rope_to_tensor)
2. [SwiGLU — 为什么不用普通的 ReLU/GELU](#2-swiglu)
3. [RoPEMultiheadAttention ⭐⭐ — 手撕 attention 的十个 Q](#3-ropemultiheadattention)
4. [CrossAttention — 和 self-attention 唯一的区别](#4-crossattention)
5. [RankMixerBlock ⭐⭐ — 无参 token mixing 为什么能涨点](#5-rankmixerblock)
6. [MultiSeqQueryGenerator — Q tokens 是怎么长出来的](#6-multiseqquerygenerator)
7. [SwiGLUEncoder vs TransformerEncoder vs LongerEncoder — 三选一怎么选](#7-three-encoders)
8. [MultiSeqHyFormerBlock ⭐⭐ — HyFormer 的三步曲](#8-multiseqhyformerblock)
9. [OneTransMixedAttn ⭐⭐ — Mixed Parameterization 的灵魂](#9-onetransmixedattn)
10. [OneTransNSConditioner ⭐⭐⭐ — EXP-013 +0.010 AUC 的秘密](#10-onetransnsconditioner)

---

## 1. RotaryEmbedding + apply_rope_to_tensor

### Q1：为什么要给 Attention 加位置编码？不加会怎样？

Self-attention 的公式是 `softmax(QK^T/√d) V`，**对 token 顺序完全不敏感**——把输入打乱，输出还是同一个（顺序被打乱的）集合。但用户行为序列明显有时序含义（最近点击的商品比一周前的点击更重要），不加位置信息等于把序列当成无序 bag。

**不加的后果**：模型退化成"词袋 + 加权平均"，完全丢失时间演化信号。

### Q2：为什么用旋转位置编码（RoPE）不用原版 Transformer 的 sin/cos 加法编码？

原版 Transformer 是**加法**：`x + PE(pos)`。问题是 token 语义和位置信息被**硬相加**在同一个向量上，深层网络里会互相干扰。

RoPE 是**乘法**（旋转）：`x' = R(θ_pos) · x`。好处：
- 保留向量模长（不改变语义强度）
- `<R(m)·q, R(n)·k>` 只依赖 `m-n`（相对位置），天然满足"最近的更相关"直觉
- 外推性更好（训练用 seq_len=512，推理可以 2048）

**类比**：原版像"身份证号码 + 位置号码拼在一起"，RoPE 像"给每个向量按时间轴旋转一个角度"。后者更优雅。

### Q3：代码里 `inv_freq = 1/base^(2i/d)` 这一步在做什么？

论文 RoFormer §3.4.2 公式：`θ_i = 10000^(-2i/d)`。

不同维度 i 分配不同的旋转频率：
- i=0 时 `θ_0 = 1`（旋转最快，捕捉短程关系）
- i=d/2 时 `θ_{d/2} ≈ 1/10000`（旋转最慢，捕捉长程关系）

**直觉**：每个 channel 负责不同尺度的时序关系，类比 CNN 里多尺度卷积核。

### Q4：`rotate_half` 函数的 `[-x2, x1]` 是什么意思？

这是复数 `z = a + bi` 乘以 `e^{iθ} = cosθ + i·sinθ` 的实部虚部展开：

```
(a + bi)(cosθ + i·sinθ) = (a·cosθ - b·sinθ) + i·(a·sinθ + b·cosθ)
```

把向量分成两半 `[x1, x2]` 当作 `[实部, 虚部]`，那么 `rotate = x·cosθ + rotate_half(x)·sinθ` 就是上面的复数乘法。

**不需要真的算复数**，只要理解 `rotate_half([x1, x2]) = [-x2, x1]` 就是把复数平面上"乘 i"的操作表示成向量形式。

### 📍 论文对应
- RoFormer 论文（arxiv 2104.09864）§3.4 "Rotary position embedding"，公式 (11)-(17)
- 飞机上重点看：Figure 1（旋转的几何图）+ 公式 (15)（最终的 pair-interleave 形式）

---

## 2. SwiGLU

### Q1：SwiGLU 到底是什么？为什么不用 ReLU？

标准 Transformer 的 FFN 是：
```
FFN(x) = W2 · ReLU(W1 · x)
```

SwiGLU 换成：
```
SwiGLU(x) = W2 · (x1 ⊙ SiLU(x2))    # x1, x2 = split(W1·x, 2)
```

**关键差异**：加了一个"门控" `x1 ⊙ ...`。哪些信息能通过，由输入自己决定（类似 LSTM 的 gate）。

### Q2：多出来的门控真的有用吗？

LLaMA、PaLM、DeepSeek 全部用 SwiGLU 不是偶然——同样参数量下 SwiGLU 的 loss 比 ReLU FFN 低 0.1~0.3（在 LLM scale 上这是巨大差异）。

**但代价**：参数量 ~2×（W1 的输出维度从 hidden 变成 2×hidden，SwiGLU 一般把 hidden_mult 从 4 降到 8/3 来弥补）。

### Q3：本仓库为什么 `hidden_mult=4`，不是 8/3？

因为这不是 LLM scale——参数量不是瓶颈，**推理速度**才是（30min 硬限）。
Baseline 测试表明：`hidden_mult=4` + SwiGLU 在这个 task 收敛更快，虽然参数多 1.5×，但在 1.9M 样本上用不完。

### 📍 论文对应
- SwiGLU 原文（Noam Shazeer 2020, arxiv 2002.05202）只有 4 页，非常易读
- LLaMA 论文 §2.2 提到 "We use SwiGLU activation function, introduced by Shazeer (2020)"

---

## 3. RoPEMultiheadAttention

### Q1 ⭐：为什么 PyTorch 有现成的 `nn.MultiheadAttention`，仓库还要手撕？

标准 `nn.MultiheadAttention` 的问题：
1. **不支持 RoPE**（只能用可学的 position embedding 或无位置）
2. W_q/W_k/W_v 被打包在一个 `in_proj_weight` 里，**无法独立冻结或加 LoRA**
3. 接口混乱（query/key/value 和 Q_proj/K_proj/V_proj 两层概念容易搞混）

手撕的好处：每一步都可读、可改、可插入 RoPE。

### Q2：为什么 reshape 到 `(B, H, L, d_head)` 而不是 `(B, L, H, d_head)`？

因为后续要做 `Q @ K^T`：
```
(B, H, L, d_head) @ (B, H, d_head, L) = (B, H, L, L)
```

把 `H` 放前面，每个 head 的 attention 矩阵是 `(L, L)`，可以独立并行计算。如果 `H` 在后面，就变成了 `(B, L, H, L)` 没法直接 matmul。

**一句话**：`H` 放前面 = 多头并行，放后面 = 多头互相纠缠。

### Q3：`scaled_dot_product_attention` 为什么比手写 `softmax(QK^T)` 快？

PyTorch 2.0+ 的 `F.scaled_dot_product_attention` 底层有三种实现：
1. **FlashAttention** (CUDA, 训练/推理最快)
2. **Memory-efficient attention** (Xformers 后端)
3. Math fallback (CPU/老 GPU)

自动选最快的那个。手写 softmax 会产生中间矩阵 `QK^T`（显存 `O(BHL²)`），FlashAttention 用 tile + recompute 把显存降到 `O(BHL)`。

**推理时长影响**：swiglu 82s vs transformer 514s（6×），主要差距就来自 attention 的 `O(L²)` vs SwiGLU 的 `O(L)`。

### Q4：代码里有个 `W_g` 投影是做什么的？

仓库扩展（不在原版 Transformer 里），做 **gated output**：
```
out_gated = W_g(x) * out
```

思路来自 GLU——给 attention 输出加一个由输入驱动的门。实测+0.001 左右 AUC。是一个"小改进但站得住脚"的扩展。

### 📍 论文对应
- Attention Is All You Need §3.2 "Multi-Head Attention"（只读这一节就够，共 2 页）
- **跳读指南**：§1-3 全读，§4 Applications 跳过，§5 Training 跳过

---

## 4. CrossAttention

### Q1：CrossAttention 和 SelfAttention 到底差在哪？

代码结构几乎一样，只有一个差别：
- **SelfAttention**：Q, K, V 全都来自同一个 tensor `x`
- **CrossAttention**：Q 来自 `query_tokens`，K/V 来自 `context_tokens`

```python
# Self
Q = W_q(x); K = W_k(x); V = W_v(x)

# Cross
Q = W_q(query); K = W_k(context); V = W_v(context)
```

### Q2：什么场景用 Cross 不用 Self？

当你有两组语义不同的 token 要做信息交互时：
- **Encoder-Decoder**（机器翻译）：decoder 的 Q 从目标语言来，K/V 从源语言来
- **HyFormer/Query Decoding**：Q 是 "问题 tokens"（num_queries 个学习向量），K/V 是"序列 tokens"
- **DETR**：Q 是 "object query"，K/V 是图像 feature

**核心直觉**：Q 问一个问题，在 K/V 里检索答案。

### Q3：本仓库的 CrossAttention 在哪里被调用？

在 `MultiSeqHyFormerBlock` 的第二步（Query Decode）：
- Q = `MultiSeqQueryGenerator` 生成的 `num_queries` 个 query token
- K/V = 对应 domain 的 encoder 输出（压缩后的序列 token）

每个 domain 独立一套 CrossAttention，4 个 domain（a/b/c/d）就是 4 份参数。

### 📍 论文对应
- HyFormer 论文 §3.4 Multi-Sequence Query Decoding
- 飞机上重点看：Figure 3 的 Cross-Attention 模块

---

## 5. RankMixerBlock

### Q1 ⭐：Token Mixing 到底在做什么？看代码完全没感觉

先理解它的"无参数"本质。假设 T=4（4 个 token），D=12（每个 token 12 维）。

**token_mixing 的三步**：

```
Step 1: reshape (B, T=4, D=12) → (B, T=4, h=4, d_sub=3)
        每个 token 的 12 维 被切成 4 份，每份 3 维。

Step 2: transpose(1, 2) 把 token 轴和 h 轴交换
        (B, 4, 4, 3) → (B, 4, 4, 3)，但语义变了：
        原本第 t 个 token 的第 h 个 subspace
        现在变成了 第 h 个"新 token"的第 t 个 subspace

Step 3: reshape 回 (B, T=4, D=12)
```

**结果**：新的第 0 个 token = `[原 T0 的 h0 块, 原 T1 的 h0 块, 原 T2 的 h0 块, 原 T3 的 h0 块]`。

### Q2：这一步为什么能涨点？直觉是什么？

想象原本 4 个 token 各自代表：用户画像、历史点击、当前广告、上下文。每个 token 在 D 维空间里有自己的"视角"。

Token mixing 做的是：**让 4 个 token 的同一个 subspace 互相换位**。等价于"从同一个切片看 4 个实体"——每个新 token 代表"4 个实体在某个 subspace 上的视图"。

**配合后续 FFN**：每个新 token 过 FFN 后，相当于在 subspace 内做了跨 token 的非线性交互。

**对比 self-attention**：
- Self-attention 用 `softmax(QK^T)` **学** token 间的交互权重（有参数，O(T²)）
- Token mixing **固定**按 subspace reshape（零参数，O(T)）

在 rank/rec 场景下，很多 token 间交互是"全连"的（每个特征都要和其他特征交互），attention 学出的 weight 往往接近 uniform——这时直接用 "固定混合 + FFN" 等价甚至更好。

### Q3 ⭐：`d_model % n_total == 0` 的约束怎么理解？

`n_total = num_queries × num_sequences + num_ns`。例如本仓库默认：
- num_queries = 2（每 domain 2 个 Q）
- num_sequences = 4（a/b/c/d）
- num_ns = 7（user_ns=5 + item_ns=2）

`n_total = 2×4 + 7 = 15`

`d_model = 64`，而 `64 % 15 = 4 != 0` → 不满足约束，会报错。

改 `user_ns_tokens=6, item_ns_tokens=2, num_queries=2`：
`n_total = 2×4 + 8 = 16`，`64 % 16 = 0` ✓

**为什么有这个约束**：token mixing 要求 D 能被 T 整除，才能把每个 token 的 D 维切成 T 份（d_sub = D/T）。

**工程陷阱**：改 num_queries 或 NS token 数时一定要同步检查 d_model 的约束，否则训练一启动就 crash。

### Q4：仓库有 `shared` 和 `per_token` 两种 FFN 模式，差别在哪？

- **shared**（baseline 默认）：所有 T 个 token 共享一套 FFN 参数
- **per_token**（RankMixer 原论文）：每个 token 独立 FFN（T 套参数）

参数量差异：T=15, d_model=64, hidden_mult=4 时，per_token 多 ~495K 参数。

**原论文为什么要 per_token**：防止"主导特征"（点击次数、用户 ID）的梯度冲刷"长尾特征"（冷门 category）——每个 token 独立 FFN 相当于给每个特征子空间分配"独立学习率"。

**本仓库的 ADR-007** 选 shared 为默认，per_token 需要 `--ffn_mode=per_token` 显式开启。飞机上可以思考：这个 task 上到底是 shared 还是 per_token 更好？（EXP 没完全结论，还是开放问题）

### 📍 论文对应
- RankMixer 论文（arxiv 2507.15551）§3 Method
- 飞机上重点看：Figure 2（token mixing 的可视化）+ §3.3 "Per-token FFN"
- 关键 takeaway：**Token Mixing + Per-token FFN = Attention 的零参数替代品**

---

<a id="6-multiseqquerygenerator"></a>
## 6. MultiSeqQueryGenerator

### Q1：Query tokens 是什么？从哪来？

在 HyFormer 的世界观里：
- **Sequence tokens** = 用户真实历史行为（点击、加购、播放...）
- **NS tokens** = 非序列特征（user_id、item_id、age、gender...的 embedding）
- **Query tokens** = 学习出来的"问题" 向量，用来"质询"序列

Query tokens 不是用户真实数据，而是**网络自己学出来的 num_queries 个"问题模板"**。类比 DETR 里的 object query，或 Perceiver 里的 latent query。

### Q2：为什么要显式造 Query tokens？直接拿 NS tokens 当 query 不行吗？

三个原因：
1. **解耦维度**：序列可能很长（L=256），NS 只有 7 个。直接用 NS 做 cross-attn 的 Q，后续 RankMixer 的 n_total 就固定为 7。加 Q tokens 后，可以把 Q 数量当超参调。
2. **不同 domain 不同问题**：a/b/c/d 四个 domain 应该问不同的问题。Q generator 为每个 domain 生成独立 Q tokens。
3. **信息瓶颈**：Q tokens 作为"压缩口袋"，强制序列信息被压进固定 slots，防止过拟合长序列里的噪声。

### Q3：代码里 `GlobalInfo = Concat(NS 均池, Seq 均池)` 这步的直觉？

在生成每个 domain 的 Q tokens 前，先做一个"warm start"：

```
user_summary = mean(user_ns_tokens)    # 用户整体特征
seq_summary  = mean(sequence_tokens)   # 序列整体特征
global = Concat(user_summary, seq_summary)  # 综合信息
Q_d = FFN_d(global)  # 每个 domain 一套 FFN
```

**为什么不直接用可学的 query embedding**：如果 Q 只是学习参数（和输入无关），那所有用户的 Q 是同一组。加上 `global` 条件后，Q 变成 **"基于当前用户的定制化问题"**——不同用户问序列不同的问题。

### Q4：`item_conditioned_query` 是做什么的？

这是仓库扩展（DIN-style），在 Q tokens 生成时额外注入 **当前候选广告 item** 的 embedding：

```
Q_d = FFN_d(global, candidate_item_emb)
```

**类比 DIN**：阿里的 Deep Interest Network，根据候选 item 去"激活"用户历史中的相关兴趣。这里是用 item emb 去"定制"Q tokens，让 Q 问"和当前广告相关的问题"。

EXP 里测试过，开关实际提升不稳定（+0.001~-0.001），目前不是默认。

### 📍 论文对应
- HyFormer 论文 §3.4 Multi-Sequence Query Decoding
- DETR 论文（arxiv 2005.12872）§3 Object Queries（理解 learned query 概念）

---

<a id="7-three-encoders"></a>
## 7. SwiGLUEncoder vs TransformerEncoder vs LongerEncoder

### Q1：为什么要有三个 encoder 选择？

三种不同的计算/表达能力权衡：

| Encoder | 机制 | 复杂度 | 推理时长（本赛题） | 表达力 |
|---|---|---|---|---|
| SwiGLUEncoder | 纯 FFN + 残差 | O(L) | **82s**（最快） | 中（无 token 交互） |
| TransformerEncoder | Self-attention + FFN | O(L²) | **514s**（慢 6×） | 高（全连交互） |
| LongerEncoder | 分段 attn（长序列） | O(L·K) | **203s** | 高（针对长序列） |

### Q2：为什么 SwiGLU 没有 attention 也能打？

这个赛题的序列 **已经被 embedding layer 压缩过**（每个 seq token 代表一次 click/view 的 compact representation），token 之间"语义距离"不大。FFN 足够做 token 内的信息 refine。

此外，**后面还有 CrossAttention（MultiSeqHyFormerBlock Step 2）和 RankMixer（Step 3）** 做跨 token 交互。encoder 不必承担所有交互。

### Q3：LongerEncoder 的"分段"是什么意思？

LongerEncoder 把长序列分成两段：
- **新 top_k 段**（长度 K，比如 K=32 的最近行为）
- **旧全量段**（长度 L，比如 L=256 的所有历史）

执行两种 attention：
- L > top_k 时：**Cross-Attn**，Q = 最近 top_k（`K_latest`），K/V = 全部 L
- L ≤ top_k 时：**Self-Attn**（fallback）

**复杂度**：从 O(L²) 降到 O(L·K)。L=256, K=32 时快 8×。

### Q4：这个赛题该选哪个？

赛题最大序列 L ≈ 200-500（我回忆，具体看 dataset.py），和 LongerEncoder 设计的千-万级不匹配。**默认 SwiGLU 就是最优解**。

LongerEncoder 和 Transformer 都保留为**对照组**：当有人问"SwiGLU 是不是太简单了？"时，我们能 5 秒给出答案：试过了，Transformer 慢 6× 只涨 0 AUC。

### 📍 论文对应
- LONGER 论文（arxiv 2505.04421）§3.2 Cross-attention compression
- 飞机上重点看：Figure 2（top-K selection 的示意图）

---

<a id="8-multiseqhyformerblock"></a>
## 8. MultiSeqHyFormerBlock

### Q1 ⭐：一个 HyFormer block 在做什么？

三步曲：

```
Step 1 (Sequence Evolve): 每个 domain 的序列独立过一个 encoder
  input:  seq_a(B, L_a, D), seq_b(B, L_b, D), seq_c, seq_d
  output: 每个序列在 D 维上被 encoder refine

Step 2 (Query Decode): 每个 domain 的 Q tokens cross-attend 到自己的 seq
  input:  Q_a(B, num_q, D), seq_a_encoded
  output: Q_a_new = CrossAttn(Q_a, seq_a_encoded) → Q tokens 变成"domain a 信息的凝练"

Step 3 (Query Boost): 所有 domain 的 Q + NS tokens 拼起来过 RankMixer
  input:  [Q_a, Q_b, Q_c, Q_d, user_ns, item_ns]
          shape = (B, num_queries * 4 + num_ns, D)
  output: 所有 token 都被 RankMixer 过一遍，做跨 domain + 跨 NS 的交互
```

### Q2：为什么分成三步，不能一步到位？

每一步对应不同类型的信息流：
- **Step 1** 是 **内 domain 的时序演化**（序列内的 item 之间交互）
- **Step 2** 是 **domain 内的抽取**（把长序列压缩成 num_queries 个 slots）
- **Step 3** 是 **跨 domain + 跨 NS 的组合**（所有高层特征一起 mix）

如果直接把所有 token（seq_a + seq_b + ... + NS）拼起来过一个大 attention：
- 序列动辄上百 token，NS 才 7 个，**NS 会被序列淹没**
- 跨 domain 直接交互没意义（domain a 的 token 5 和 domain b 的 token 20 为啥要交互？）
- 计算复杂度爆炸：O((4L + 7)²)

三步结构保证每一步交互发生在**语义对等**的 token 之间。

### Q3：n_total = num_queries × num_sequences + num_ns 的约束从哪来？

Step 3 的 RankMixerBlock 要求 `d_model % n_total == 0`（见 RankMixerBlock Q3）。

**实际计算**（baseline 默认）：
- num_queries=2, num_sequences=4, num_ns=7 → n_total=15 → d_model=64 **不过** (64%15≠0)
- 调成 user_ns=6 或 num_queries=1 才能过

这也解释了为什么 baseline 参数看起来"奇怪"——是被约束强制调出来的。

### Q4：什么时候选 HyFormer，什么时候选 OneTrans？

仓库里这两个都实现了（通过 `use_onetrans` 开关切换）：

| 维度 | HyFormer | OneTrans |
|---|---|---|
| 参数量 | 中等（per domain 独立） | 高（mixed param 每个 NS 独立） |
| 推理速度 | 快（Step 3 只过 Q + NS） | 慢（整个 S 和 NS 都过 attention） |
| 表达力 | 序列内弱交互 + Q 压缩 | 序列和 NS 全连 attention |
| 当前 baseline 选择 | ❌ | ✅（更高 AUC） |

**EXP-013 显示**：OneTrans + NSConditioner (v3) = 0.859408，HyFormer (v2) = 0.848712。**+0.010 AUC**，是当前最大单点提升。所以现在默认 OneTrans。

### 📍 论文对应
- HyFormer 论文 §3.2-3.4
- 飞机上重点看：Figure 2（三步结构图）

---

<a id="9-onetransmixedattn"></a>
## 9. OneTransMixedAttn ⭐⭐

### Q1：OneTrans 的 "Mixed Parameterization" 到底 mixed 了什么？

核心思想：**S tokens 和 NS tokens 用不同的参数矩阵做 attention**。

标准 Transformer：所有 token 共享 W_Q/W_K/W_V/W_O 四个矩阵。

OneTrans：
- **S tokens**（序列）共享一套 `W_Q_s, W_K_s, W_V_s, W_O_s`
- **每个 NS token** 独立一套 `W_Q_ns[j], W_K_ns[j], ...`

参数量对比（M=NS 数量，D=d_model）：
- Standard: `4D²`
- OneTrans S: `4D²`
- OneTrans NS: `4D² × M` (!!)

M=7, D=64 时 NS 部分参数多 7×。

### Q2：为什么 NS 要独立参数？S 可以共享？

**序列 tokens**：都是"用户行为"这一语义类别的实例（不同的点击、加购...），本质同构，共享参数合理。类似 Transformer 处理文本——每个 token 都是词，共享 FFN 合理。

**NS tokens**：`user_id` / `item_id` / `age` / `gender` / `category`... 语义完全不同。强制共享参数等于强迫 `age` 和 `category` 用同一个投影矩阵——不合理。

给每个 NS 独立参数，等于**让每个 NS 特征学自己的 projection**，保留特征异构性。

### Q3 ⭐：代码里 attention mask 是什么样的？

核心是 `_build_causal_mixed_mask` 生成一个 `(L+M, L+M)` 的布尔矩阵：

```
     S-part (L)          NS-part (M)
   ┌──────────────────┬──────────────────┐
S  │  BIDIRECTIONAL   │  masked (NS→S    │
   │  (S-S 全连)       │   in NS block)   │
   ├──────────────────┼──────────────────┤
NS │  allow all       │  CAUSAL          │
   │  (NS 看全部 S)    │  (NS→NS 因果)     │
   └──────────────────┴──────────────────┘
```

- **S→S: BIDIRECTIONAL**（双向）：推荐/广告的序列按字母序拼 domain a/b/c/d，**没有时序含义**，不需要 causal
- **NS→S: 允许**：NS 要看完整的用户历史做决策
- **NS→NS: CAUSAL**（下三角）：NS 按某种逻辑顺序解码（item_ns 依赖 user_ns 等），这里用因果防信息泄漏

这个 mask 设计是 OneTrans 区别于标准 Transformer 的精髓。

### Q4：为什么 S→S 不用 causal？和 LLM 不一样？

LLM：token 是时序的（词的生成顺序），causal 防止看到未来。

广告/推荐 S tokens：**已经是"过去发生过的行为"的 embedding**，按 domain 字母序拼接（a, b, c, d），**字母序不代表时序**。domain a 的第 0 个 token 和 domain d 的第 0 个 token 没有先后关系。

强制 causal 会让**后面 domain 的 token 看不到前面 domain**——丢失跨 domain 信息。所以用 bidirectional。

这是一个容易搞错的点，看代码时如果没注意 mask 语义，会误以为"OneTrans 就是 GPT 换了个壳"。

### 📍 论文对应
- OneTrans 论文（arxiv 2510.26104, WWW 2026）§3 Method, §3.2 Mixed Attention
- 飞机上重点看：Figure 2 + Figure 3（mask 结构图）
- **警告**：论文原版用 causal mask for S，本仓库改成 bidirectional（适配广告场景）。读论文时注意这个 gap。

---

<a id="10-onetransnsconditioner"></a>
## 10. OneTransNSConditioner ⭐⭐⭐（EXP-013 +0.010 AUC 的秘密）

### Q1 ⭐⭐：这个模块到底在做什么？用一句话

在 OneTrans 主 attention 开始前，**给每个 NS token 注入一个"序列摘要"的 warm-start**。

### Q2：为什么需要 warm-start？没有会怎样？

OneTrans 开始时的状态：
- S tokens：包含丰富的序列信息（4 个 domain 的点击历史）
- NS tokens：只是各自特征的独立 embedding（user_id emb, age emb... 完全不知道序列发生了什么）

第 1 层 attention 时，NS tokens 要 query S tokens 找相关信息。但 NS tokens 的初始值是**孤立的特征 embedding**，不知道该 query 什么。

**冷启动问题**：NS→S attention 的 query 能力差，浪费前几层网络。

### Q3：NSConditioner 怎么解决这个？

在 OneTrans 主 block 前插入：

```
for each domain d in [a, b, c, d]:
    s_mean[d] = masked_mean_pool(s_tokens_of_domain_d)
seq_ctx = Concat(s_mean[a], s_mean[b], s_mean[c], s_mean[d])  # (B, 4D)

for each NS token j in [0, ..., M-1]:
    delta[j] = FFN_j(LayerNorm(seq_ctx))  # 每个 NS 独立的 conditioning FFN
    ns_tokens[j] = ns_tokens[j] + delta[j]
```

**效果**：
- 每个 NS token 拿到了"4 个 domain 的平均摘要"
- 每个 NS 用独立 FFN 消化摘要 → 不同 NS 关注摘要的不同方面
- LayerNorm + residual 保证稳定

### Q4：为什么能带 +0.010 AUC？这是巨大提升

回顾 EXP-013：
- v2（HyFormer 纯 baseline）：0.848712
- v3（OneTrans + NSConditioner）：0.859408
- 差值：**+0.010696 AUC**

类比：HyFormer 里有 `MultiSeqQueryGenerator` 给 Q tokens warm-start（Q6 里讲的 `global = Concat(user_summary, seq_summary)`）；**NSConditioner 是把这个机制套用到 OneTrans 的 NS tokens 上**。

**深层原理**：  
网络参数初始化时是随机的，前 N 个 epoch 在学"怎么让 NS 去 query S"。给 NS 一个 warm-start 相当于**跳过这个冷启动阶段**，让网络从一开始就能学真正有用的交互模式。

### Q5 ⭐：这个 insight 有什么通用启发？

> **任何时候你的网络有"异构 token"做 cross-interaction，都应该考虑 warm-start 其中一方**。

类似的例子：
- DETR：object queries 初始化时含随机位置先验
- Perceiver IO：latent queries 是可学习参数 + 输入相关的 conditioning
- Stable Diffusion：cross-attn 的 text query 先过 CLIP encoder

**反问**：这个仓库还有哪里可以应用同样思路？
- a. MultiSeqHyFormerBlock 的 Step 2 CrossAttn：Q tokens 已经有 warm-start（Step 1 的 QueryGenerator）✓
- b. 如果加入 **user × item × candidate** 交互，candidate token 是否需要 warm-start 来自用户/item 的信息？这可能是**下一个创新奖候选**

### Q6：飞机上想清楚这件事

把 Q5 的反问作为 16h 精读的"成果输出"：

> 写一份 `docs/40-experiments/EXP-XXX-warmstart-thought.md` 草稿，列出仓库中 3-5 个"异构 token 交互"的位置，判断哪些可以加类似 NSConditioner 的 warm-start 机制。

如果飞机上想出 1 个站得住脚的位置，这次出差学习就值了。

### 📍 论文对应
- 无对应论文（仓库原创扩展）
- 参考：OneTrans 论文 §3（主结构），HyFormer 论文 §3.4 MultiSeqQueryGenerator（灵感源）
- 飞机上任务：对比 HyFormer 的 `MultiSeqQueryGenerator` 和本仓库的 `OneTransNSConditioner`，理解它们是同一个 design pattern 的两次实现

---

## 飞机上的"小测验"（读完本文的自检）

如果你能不看任何资料回答出以下 5 个问题，说明 16h 飞行学到了核心：

1. 为什么 RoPE 用乘法不用加法？RoPE 和 sinusoidal PE 的关键差异在哪？
2. Token mixing 为什么"无参数但能涨点"？它替代了什么？
3. HyFormer 的三步曲（Evolve → Decode → Boost）每步在解决什么问题？
4. OneTrans 的 attention mask 里 S→S 为什么是 bidirectional？和 LLM 的 causal 有什么本质差异？
5. NSConditioner 为什么能带 +0.010 AUC？这个思路可以泛化到哪里？

能答出 3 个 = 及格，5 个 = 优秀。答不出的话回到对应 § 节再读一遍。

---

**下一步**：读完本文后，建议先读 [`11-paper-roadmap.md`](11-paper-roadmap.md) 建立论文间的关联；再读 [`10-background-primer.md`](10-background-primer.md) 补齐广告/LLM 背景；最后带着具体问题**选择性地**读原论文（不用全读）。
