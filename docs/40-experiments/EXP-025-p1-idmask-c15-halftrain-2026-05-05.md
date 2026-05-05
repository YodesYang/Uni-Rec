# EXP-025 · P1 Item-ID Random Mask (seq_c 15%) Half-Train

**日期**：2026-05-05
**作者**：Agent
**状态**：❌ **已证伪**（2026-05-05 · LB 0.842176 vs SOTA 0.84637 · Δ **−0.004194**）

---

## 背景

社区 lexiang 选手 Seed（LB 0.85+）主楼与评论区 2026-05-04 晚抓到两条关键情报：

- **主楼问题-1**：uid 训练集和测试集**几乎全是一次性 user 且无交集**；item id 线下 auc 虚高 → 线上掉；Seed 判断"估计是分布问题"
- **评论区 Pung**：item id 没用可能是因为 **test 中 itemid OOV 率太高** 导致；可以加上**随机 mask**，训练时让模型不要过度依赖 id 特征

这两条直接对口我们 7 次提交稳定观测到的 **dev→LB gap +0.021~+0.030** 系统性现象（EXP-018 / EXP-024）。

EXP-024 方法论复盘后的结论之一："dev 是 train 近未来、LB 是 train 远未来，两个不同任务的 generalization，相关性不稳定"。如果 test 集里 item_id OOV 率高，模型在训练时学到的 id-specific 嵌入在 test 上失效 → dev↑ / LB↓ 的机制成立。

---

## 假设

训练时对 `domain_c_seq_47`（vocab=278M，最长尾 item_id 序列）做 token 粒度随机 mask（概率 p），强制模型在"看不到具体 id"的条件下仍能从上下文（其他 side-info 特征 + 其他 domain）预测 CVR，**相当于 id 通道上的 regularisation**。

**预期效应**：
- Dev AUC 绝对值**略降**（训练"变难"，模型容量受限）
- Dev→LB gap **缩小**，LB AUC 持平或提升
- Mask 作用于 `seq_c`（vocab=278M, hash500k 后仍是最长尾），**比其他 domain 更需要这种正则**

---

## 配置

**Job Name**：`idmaskc15-icq-halftrain-swiglu-vr005-bs128`

与 `icq-t13tf-halftrain-swiglu-vr005-bs128`（EXP-019, LB 0.84637 SOTA）**单变量差异**：

| 参数 | icq-half baseline | 本 job | 变化 |
|---|---|---|---|
| `--id_mask_prob` | 0.0 | **0.15** | +P1 token mask |
| `--id_mask_seq_domains` | 未传 | **c** | 只 mask seq_c |
| `--logloss_rebound_patience` | 未传（0）| 2 | 与 icq-half 同（在 guard 出现后补上）|
| 其他 | 完全一致 | 完全一致 | — |

完整 flag：

```
--item_conditioned_query
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--id_mask_prob 0.15 --id_mask_seq_domains c
--logloss_rebound_patience 2 --logloss_rebound_delta 0.0
--batch_size 128 --num_epochs 4
```

**明确未叠加**（single-variable discipline）：
- `--target_item_seq_injection`（T16，另有独立 full train 在跑）
- `--holdout_ratio`（独立成本未核查前不叠）
- `--valid_gap_ratio`（P3，独立 ablation 已并行 job gap15-icq-halftrain）

---

## 代码落地摘要

- `baseline/dataset.py`：`PCVRParquetDataset.__init__` 新增 `id_mask_prob` / `id_mask_seq_domains`；`_convert_batch` 中 seq 填充后对指定 domain 按 token 粒度 Bernoulli 采样随机置 0（同时 time_bucket 同位置置 0，避免时间通道泄漏被 mask token 的存在）。Mask **双 gate**：`is_training AND shuffle AND prob>0`。domain 名字支持 `'c'` 和 `'seq_c'` 两种写法。
- `baseline/train.py`：新增 CLI `--id_mask_prob` / `--id_mask_seq_domains`，透传 `get_pcvr_data`。
- `baseline/infer.py`：**零改动**，双 gate 保证推理路径 bit-identical。

本地 dataset-only smoke 5/5 通过：
1. 零改动路径 bit-identical ✓
2. 三路切分正确 ✓（P3 同时验证）
3. `prob=0.5` 全 domain mask 比率 **0.497~0.511**（统计预期 50%）✓
4. inference 路径 bit-identical 无论 prob 值 ✓
5. `domains=['c']` 下 seq_c 掉 **0.903**（prob=0.9），seq_a/b/d 完全不变 ✓

---

## 判决阈值

| 场景 | E4 dev AUC 区间 | E4 dev→LB gap 预估 | 行动 |
|---|---|---|---|
| **强成功** | ≥ 0.868（持平 SOTA）| 缩小 | Evaluate；成则 commit 路线，下轮叠 T16 |
| **正常成功** | [0.866, 0.868]，dev 略降 | 持平或缩小 | Evaluate（如 LB ≥ 0.846 即可接受）|
| **边界** | [0.863, 0.866]，dev 明显降 | 不可知 | 先看 gap15 ablation 的 gap 缩减再决定是否消耗 eval quota |
| **失败** | < 0.863 | — | mask=0.15 过强，下轮 0.10 重试；或方向错误 |

**LB 目标**：
- 理想 ≥ 0.847（+0.001 vs SOTA 0.84637）
- 持平 0.846 也算方法成功（证明 id mask 无害可叠加到 T16）
- 若 < 0.844 → P1 方向错误（不只是强度问题）

---

## 回流数据

| Epoch | Step | Val AUC | Val LogLoss | vs icq-half same epoch |
|---|---|---|---|---|
| E1 | 15941 | 0.864440 | 0.22703 | icq-half E1=0.864992 (Δ **−0.000552**) |
| E2 | 31882 | 0.867081 | 0.22503 | icq-half E2=0.867557 (Δ **−0.000476**) |
| **E3** | **47823** | **0.868288** | **0.22382** | icq-half E3=0.867965 (Δ **+0.000323**, **global best**) |
| E4 | 63764 | 0.868272 | 0.22326 | icq-half E4=0.868619 (Δ **−0.000347**, E3 比 E4 更高) |

**best = E3 @ step 47823, local AUC = 0.868288**
stop_reason = max_epochs_completed（未触发 logloss rebound guard）。
总训练时长 ≈ 7h06m。

---

## Evaluation 结果

**Job**：`idmaskc15-icq-halftrain-swiglu-vr005-bs128_eval_*`（2026-05-05 PDT, evaluation quota 1/4 used）
**Ckpt**：`global_step47823.auc=0.868288.best_model`（E3 global best）
**上传**：`baseline/infer.py` 零改动版本（双 gate 保证推理 bit-identical）

| 指标 | 数值 |
|---|---|
| Local Val AUC (E3) | 0.868288 |
| **Leaderboard AUC** | **0.842176** |
| **vs SOTA 0.84637** | **Δ −0.004194** |
| dev→LB gap | **+0.026112** （放大 2× 相对 icq-half 的 +0.0222）|

---

## 小结 · P1 方向证伪

**核心观察**：dev AUC 微升 +0.000323 对应 LB 大跌 −0.004194，**dev 与 LB 方向相反**。这再次证明 EXP-024 复盘的"dev 不是 LB 的单调代理"铁律。

### 机制重新审视

社区 Seed 的原话是 **"id 特征在精排没用真的难绷"** —— 他说的是 **id 特征本身不该用**，不是"用了但 mask 一下能解决"。我们对 Seed 情报的解读方向错了：

- **Pung 评论的建议**："可以加上随机 mask，训练时让模型不要过度依赖 id 特征"
- **我们实际做的**：对 `seq_c` 按 token 以概率 15% 随机置 0 —— 但 **80% token 仍可见**，模型仍依赖 id embedding
- **真实 test 集**：item_id OOV 率高时，**那 80% 没被 mask 的训练路径**完全失效

**本质结论**：input-level random mask 在 "远未来 test" 场景下**无效**。机制上的问题不是正则强度（mask=0.15），而是模型仍然在 80% 时间里学到了"依赖 id"的路径。

### 为什么 dev 看不出问题

Dev valid 集与 train 尾部时间接近，**id 分布基本共享** → 训练路径和 dev 路径都看到 id，模型微调后 dev 反而略升。只有在 test（远未来）时 id 分布漂移才显露问题。

### 不应再尝试的变体

- ❌ `id_mask_prob = 0.10`（更保守）：方向就错了，降强度没意义
- ❌ `id_mask_prob = 0.30` 以上：dev 会崩盘，LB 不会变好
- ❌ `id_mask_seq_domains = c,d`（扩大范围）：同样的机制问题

### 下一步：方向切换到 T23（structural disable）

**正确的路线**：彻底**去掉 item_id 通路**，让模型学习"没有 item_id 的 CVR 预测"。这和 Seed 主楼问题-5 "seqc feat47 跟 itemid 是一样的，做不做 match 特征都掉分" 一致 —— seq_c 的 item_id 序列**本身信号就是噪声**。

T23 代码已就位（`--disable_seq_fids 47`），next 实验 = `disablefid47-icq-halftrain`。

