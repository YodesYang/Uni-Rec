# EXP-020 · T16 Target-Item Seq Injection Proxy + Half-Train（ADR-004 方案 A）

**日期**：2026-05-04 夜  
**作者**：Agent  
**状态**：🔄 Proxy 完成（正向） + Half-Train 进行中

---

## 背景

ADR-004 方案 A：在 `MultiSeqHyFormerBlock` 的 seq encoder 之前，把 `target_item_repr = mean(item_ns_tokens)` 通过 per-sequence `Linear(D, D)` + 可学标量 `α_i`（init=0.1）加到每个 seq step：

```
seq_tokens_i_new = seq_tokens_i + α_i * W_target_i(target_repr).unsqueeze(1)
```

动机：ICQ（EXP-019）只在 query 侧做 target conditioning（half-train E4 Δ+0.000124），seq encoder 内部的 key/value 仍是 target-agnostic；方案 A 从 encoder 输入层就把 target 注入 seq token，放大 target-aware matching 信号。

---

## Proxy 配置

**Job Name**：`t16-additive-a01-t13tf-proxy-swiglu-vr005-rg128x32-s1500`

```text
--target_item_seq_injection additive
--target_inject_alpha_init 0.1
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--limit_train_rgs 128 --limit_valid_rgs 32 --max_train_steps 1500
```

---

## Proxy 结果

| 配置 | best AUC | best LogLoss | 备注 |
|---|---|---|---|
| **T16 additive α=0.1 + hash500k + timefeat** | **0.839564** | **0.241509** | 本次 |
| 对照 1：hash500k + timefeat（EXP-015） | 0.837684 | 0.242104 | 同 hash 配置 |
| 对照 2：timefeat-only（EXP-012） | 0.839285 | 0.241525 | 无 hash |

**关键增量**：

```
Δ(T16 − hash500k+timefeat)  = +0.001880 AUC / −0.000595 LogLoss  ← 真正的对照
Δ(T16 − timefeat-only)      = +0.000279 AUC / −0.000016 LogLoss  ← 跨配置参考
```

---

## 关键发现

### 1. T16 在 proxy 规模直接正向，打破"大参数假负向"规律

与 ICQ（+429k 参数，proxy Δ−0.001417）不同，T16 在 proxy 阶段就是正向信号。
可能原因：T16 的新参数集中（`Linear(D,D) × 4 domains` ≈ 16k）远小于 ICQ。
`α_i=0.1` 初始化接近关闭状态，对基线影响温和，梯度注入稳定。

### 2. T16 补偿了 hash500k 的 proxy 冷启动劣势

EXP-015 的发现：hash500k 在 proxy 上因 table 冷启动显示 Δ−0.001601。
本次 T16 在**同样**的 hash500k 上反而超过 timefeat-only，说明：

> target-item injection 给了每个 seq token 一条稳定的梯度路径（target_repr 来自 item_ns 稠密 embedding），hash embedding 通过"与 target 对齐"获得更清晰的监督信号，**冷启动问题被部分消解**。

这和 EXP-018 中"hash500k × timefeat 正向交互 +0.001752"是同一机制的不同显化——稀疏 hash table 需要稠密信号帮助它稳定更新方向。

### 3. 与 ICQ 的对比

| 维度 | ICQ（EXP-019） | T16（本次） |
|---|---|---|
| 注入位置 | query 侧（global query） | key/value 侧（seq token） |
| 参数量 | +429k | +16k |
| proxy Δ | −0.001417（假负向） | **+0.001880**（真正向） |
| half-train E4 Δ | +0.000124 | 待定 |

T16 信号明显强于 ICQ。ADR-004 §Consequences 预测两者正交，值得 EXP-021 验证叠加。

---

## Half-Train 推进决策

**推进 half-train（4 epoch 全量 RG）**，按 ADR-004 S3 阈值：

- **Job Name**：`t16-additive-a01-t13tf-halftrain-swiglu-vr005-bs128`
- **阈值**：E3 best AUC ≥ 0.868（t13tf E3 ≈ 0.867 + 0.0002 epsilon）
  - 达标 → 推进 S4 full train（10 epoch）
  - 0.866 ≤ E3 < 0.868 → 看斜率，仍单调上升则跑到 E4
  - E3 < 0.866 → 回滚

Half-train 配置（在 proxy 基础上去掉 proxy 限速）：

```text
--target_item_seq_injection additive
--target_inject_alpha_init 0.1
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--num_epochs 4
```

单 epoch ~80min，预期 ~5h20min 回流。

---

## 待回填

- [ ] Half-train E1~E4 AUC / LogLoss 全表
- [ ] α_i（alpha_seq_a/b/c/d）learned 值（从 ckpt 读出看 T16 实际"开"到多大）
- [ ] 与 t13tf 同 E4 对比（同基线增益实测）
- [ ] 若 E3 达标，启动 full train 的 Job Name 与阈值

---

## 相关

- ADR-004（方案 A additive 实现）
- ADR-005（valid→LB gap 重设计，与本实验正交）
- EXP-015（hash500k proxy 冷启动规律起点）
- EXP-019（ICQ 规律，"大参数假负向"记录）
- EXP-018（t13tf full train + 历史 gap 扫描）
