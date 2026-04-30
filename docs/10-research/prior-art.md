# 论文 / 方向（Prior Art · 按相关度排序）

> 来源：原 `knowledge_base.txt §5`。
> 阅读完一篇论文后，把可落地的想法 → 写成 `docs/10-research/ideas-backlog.md` 的一条；想启动 → 升为 `docs/20-tasks/active.md` 的一个 task。

---

## 5.1 OneTrans — 统一 Tokenizer + Mixed Parameterization

赛题方向直接对应；datawhale 解读已收藏。

- **思路**：S-tokens（序列）共享参数；NS-tokens（非序列）每 token 独立参数
- **Mixed Causal Attention**：
  - S-token 因果地相互 attend（建模时间/意图顺序）
  - NS-token 全局 attend 到所有 S-token + 之前的 NS-token
- **Pyramid Stack**：layer-wise 收缩 query 数量，逐层蒸馏到尾部 NS-token
- **KV Caching（Cross-Request）**：用户行为序列 append-only，跨请求缓存 K/V，把每次请求 O(L_S) → O(ΔL_S)
- **离线指标**：相对 encode-then-interaction 提升 +5.68% GMV（论文）

**与本仓库关联**：`docs/20-tasks/archive.md` T5 已决议暂缓启动（等 T4 编码器消融数字出来）。

---

## 5.2 RankMixer — 硬件感知的同构 backbone

baseline 中已采用其 NS Tokenizer 思想。

- **Multi-head Token Mixing**：参数无关、O(T·D) 替代 self-attention 的 O(T²·D)
- **Per-token FFN**：每个 token 独立 FFN 参数，避免 dominant feature 淹没长尾
- **Sparse-MoE 变体**：每 token 仅激活一部分专家，最少代价扩到 1B+ 参数
- **工业指标**：MFU 4.5% → 45%；抖音 feed 在线 active days +0.3%、in-app duration +1.08%
- **关键约束**：`d_model` 必须是 `T` 的整数倍（baseline 已 enforce）

---

## 5.3 HyFormer — 双向 Query Decoding / Query Boosting

baseline 实现风格的直接来源。

- **Global Tokens** 作为「序列 ↔ 异构特征」的语义接口（不做参数级强耦合）
- 每个 HyFormer 层：
  - Query Decoding：`Q ← CrossAttn(Q, K_l, V_l)` // K/V 来自序列编码器
  - Query Boosting：`Q ← MLP-Mixer + Per-token FFN` // Q 与 NS-token 拼接
- 序列编码可选：Full Transformer / LONGER 风格 / SwiGLU 轻量
- 多序列设计：每条序列独立 Q token 集，靠 Query Boosting 跨序列交互
- **业务部署**：抖音搜索全量部署，AUC vs LONGER+RankMixer：+0.17%（同 FLOPs 下）

---

## 5.4 LONGER — 高效长序列编码

- 用 cross-attention 替代 self-attention：Q 长度 L_H ≪ L_S，Key/Value 取全长
- **复杂度**：O(L_S²) → O(L_H · L_S)
- baseline 中通过 `LongerEncoder` 实现（top_k 采样最近的 K 个有效 token）

---

## 5.5 其他可参考方向

- **HSTU / ULTRA-HSTU**：生成式推荐架构（未在 HyFormer 论文中作为 baseline 对比）
- **Wukong、DCNv2、xDeepFM**：传统特征交互 baseline（性能上界已较低，仅作参考）
- **DIN / SIM / ETA**：长行为序列建模经典论文
- **MultiEpoch（KuaiShou，arXiv:2305.19531）**：高基数 Embedding cold-restart，baseline 已集成
