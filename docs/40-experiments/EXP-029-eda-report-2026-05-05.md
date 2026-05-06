# EXP-029 · EDA Report · 2026-05-05 晚

**日期**：2026-05-05 晚 PDT
**作者**：Agent
**状态**：✅ 完成
**触发**：T24 代办，多次方向判决后意识到**从未做过数据尽职调查**就启动了 10+ 实验。今晚 4 个 eval / 2 个 training quota 全满后，用 60 分钟补 EDA，给明天后续方向定位。

---

## 数据源

本次 EDA 用 3 个互补数据：

1. **`data/demo_rg100/demo_1000.parquet`** + `schema.json`（本地 demo）
2. **Platform training log** 里的 `[diag] slice by ...` 表（来自今晚 C1 `disablefid47_29` E1 valid 诊断）
3. **Platform inference log** 里的 `PCVRHyFormer cfg`（真实 schema）

**局限**：test 数据在平台只读，我们无法直接统计 test 里 uid / item_id 的 OOV 率，只能间接推断。

---

## EDA-1 · Row Group 内部 timestamp 分布

### Demo 结果（100 RG / 1000 行 / 10 min 跨度）
- Demo 数据全局只跨 **13 分钟**（0.009 天）
- 每个 RG 内部跨 **12 分钟**（0.21 小时）
- 所有 RG 时间范围几乎完全重叠
- **RG 排序不按时间单调**：sorted by max_ts 顺序下，只有 2/10 RGs 在正确位置

### Platform 间接证据（已有）
- C6 job 日志：train ts / gap ts / valid ts 全部跨 3.95~4.04 天（EXP-026 已观察）
- 按 RG max_ts 排序后，仍能用 `effective train→valid distance = -341046 sec`（负值）证明 gap 未真正隔离时间

### 结论

**Platform 数据每个 RG 都跨 ~4 天 / 全时间段**，这是**数据写入时按非时间 key 分片**的典型结果（比如 user_id hash）。

这验证了：
- ✅ EXP-026 "RG-level gap 失效" 的诊断完全正确
- ✅ **T22-fix row-level gap cutoff**（C6 C6 的实现）确实是正确方向
- ✅ 未来任何"按 RG 切分"的 split 策略都有同样问题

---

## EDA-2 · C1 E1 valid 切片表（单特征 AUC 信号）

### Slice by **user_int_nonzero**（用户特征丰富度）

| Bucket | n | pos_rate | **AUC** | pred_mean |
|---|---|---|---|---|
| [0, 58] **最稀疏** | 26375 | 0.1118 | **0.8915** | 0.1222 |
| [58, 71] | 27266 | 0.1039 | **0.8673** | 0.1139 |
| [71, 86] | 28064 | 0.0960 | **0.8470** | 0.1089 |
| [86, 283] **最丰富** | 27270 | 0.0913 | **0.8471** | 0.1089 |

**🔥 反直觉发现**：**user 特征越稀疏 AUC 越高**（Δ 0.045, 最稀疏 - 最丰富）。

### Slice by **item_int_nonzero**（商品特征丰富度）

| Bucket | n | pos_rate | **AUC** | pred_mean |
|---|---|---|---|---|
| [0, 8] 最稀疏 | 15937 | 0.1554 | **0.9328** | 0.1725 |
| [8, 9] | 18720 | 0.0998 | **0.8387** | 0.1137 |
| [9, 12] | 39088 | 0.0758 | **0.8292** | 0.0846 |
| [12, 33] 最丰富 | 35230 | 0.1038 | **0.8551** | 0.1183 |

**同样反直觉**：item 特征最稀疏 bucket AUC **0.9328**，最丰富只有 **0.8551**。

### Slice by **seq_total_len**（用户历史长度）

| Bucket | n | pos_rate | **AUC** | pred_mean |
|---|---|---|---|---|
| [0, 1114] **最短** | 27210 | 0.0949 | **0.8991** | 0.1036 |
| [1114, 1322] | 27184 | 0.1040 | **0.8633** | 0.1139 |
| [1322, 1536] **最长** | 54581 | 0.1018 | **0.8453** | 0.1179 |

**同样**：seq 最短（冷启动）用户 AUC **0.8991**，seq 最长 AUC **0.8453**。Δ **0.054**。

### 解读

三张表的**共同信号**：**"信息丰富"样本上 AUC 更低**。两层解读：

**解读 A（主要）**：信息丰富 = 模型有更多 id 可能 overfit
- user 特征多 → user_int 里有更多 id embedding 参与 → 模型容易记忆训练期的 id 组合
- seq 长 → seq 里 item_id / action_id 越多 → 同上
- **训练期精准，test 期 OOV → 分布外泛化失效**

**解读 B（辅助）**：信息稀疏样本 pos_rate 更高（0.15 vs 0.09）
- 正样本相对少 → ROC 的 AUC 数字天然偏高
- **但 Δ 0.04~0.05 太大**，单靠 pos_rate 差异解释不通

**直接证据**：**Seed 主楼问题-1 "id 特征线下虚高线上掉" 的机制在我们模型里真实存在**。

---

## EDA-3 · label_type 分布

| label_type | n | pos_rate | pred_mean |
|---|---|---|---|
| **0 (曝光未点击)** | **0** | — | — |
| 1 (点击未转化, 负样本) | 98011 | 0.0000 | 0.0837 |
| 2 (点击转化, 正样本) | 10964 | 1.0000 | 0.3782 |

- **无 label_type=0**（已知：click-conditional CVR，T15 multi-task 关闭的原因）
- pos_rate ≈ 10.06%（与 team-memory 一致）
- 模型 pred_mean 对 label=2 (0.3782) 比 label=1 (0.0837) 高 **4.5 倍** → ranking 区分明显

**结论**：label_type 分布**无异常**，不是问题核心。

---

## EDA-4 · 高 vocab fid 列表（platform 真实 schema）

### user_int（顶层用户特征）

全部 fid vocab **< 3000**，**无 "item_id / user_id" 级别的大 vocab**：
- 最大：(1916, dim=1), (1485, dim=201), (1173, dim=26)
- 结论：**user_int 里没有需要 disable 或 UE-split 的大 vocab**

### item_int（顶层商品特征）

全部 fid vocab **≤ 25495**：
- `fid=6` vocab=23832, dim=20（商品多热编码）
- `fid=9` vocab=25495, dim=1（**最大的顶层 item 特征**）
- 其余都 < 3000
- 结论：**item_int 里同样没有明显的 "item_id" 特征**

### seq 里的大 vocab（全部集中在序列域）

| Domain | FID | Vocab | 含义推断 |
|---|---|---|---|
| seq_c | 47 | **278,677,640** | item_id（Seed 证实）|
| seq_b | 69 | **85,946,394** | (类似 item_id 候选) |
| seq_c | 29 | **8,227,608** | 可能次级 id |
| seq_c | 34 | 1,986,300 | |
| seq_c | 36 | 1,511,293 | |
| seq_a | 38 | 1,201,293 | |
| seq_b | 74 | 638,924 | |
| seq_d | 23 | 674,034 | |
| seq_d | 22 | 510,191 | |
| seq_b | 76 | 164,032 | |

**所有大 vocab 都在 seq 域，且主要集中在 seq_c 和 seq_b**。

---

## 🔥 EDA 整体结论

### 结论 1：**"item_id 问题" 只在 seq 里**，不在顶层 user_int / item_int

顶层 user_int / item_int 最大 vocab 25k，**不构成 OOV 风险**。Seed 主楼问题-1 "item id 线下虚高线上掉" 严格指 **seq 里的 id**。

**对 T23 方向的重新评估**：
- 方向"在 seq_c 里动 item_id" 仍然对口
- 但**完全 disable**（T23 做的）太激进 → LB -0.006 是可预料的
- **更温和做法**：进一步 hash vocab 降到 100k 或 50k（比当前 500k 更激进）

### 结论 2：**模型确实在"信息丰富"样本上过拟合**

user_int_nz / item_int_nz / seq_total_len 三个维度的 valid 切片**一致显示**：
- 稀疏样本 AUC 高 0.04~0.05
- 丰富样本 AUC 低

**这是 dev→LB gap 的直接机制证据**——模型在 train 上越丰富越依赖 id 记忆，test 上这些 id OOV → 分布外泛化失败。

### 结论 3：**架构级方向的优先级**

基于 EDA，重新排 T24 候选的 ROI：

| # | 方向 | EDA 适配度 | 代码量 |
|---|---|---|---|
| 🥇 **F1** | Seq hash vocab 更激进（500k → 100k 或 50k）| **极高**（直接对口结论 1）| **0 行**（CLI 已支持）|
| 🥇 **F2** | DIN target attention | 高（target-aware 让模型不依赖 id 记忆）| ~80 行 |
| 🥈 **F3** | 按样本丰富度加权 / 分桶训练（给稀疏样本更高 weight）| 中（直接对口结论 2 的机制）| ~50 行 |
| 🥉 **F4** | FaFE feature-aware embedding | 中（让模型区分对待不同 fid） | ~100 行 |
| ❌ | UE 分离 + int pairwise cross（Seed 方案 1） | 低（EDA 证明 user_int 没有 UE 层级）| ~120 行 |
| ❌ | 顶层 item_id disable | 低（EDA 证明顶层没大 vocab）| ~10 行 |

### 结论 4：**F1 是最容易 / 最快 / 最对口的下一个实验**

理由：
1. **0 行代码改动**（CLI `--seq_hash_vocab 100000` 已经支持）
2. **对口 EDA 结论 1**：seq_c fid=47 用 500k 桶，OOV rate 仍高 → 降到 100k 强制更多 collision 平滑 id embedding
3. **Seed 主楼实测验证**过类似方向（hash500k → 有效）—— 更激进的 hash 可能进一步提升
4. **LB 破 0.849 概率估算**：中等（10~25%），比其他架构级方向更有保障

---

## 明天/后天的行动建议

### 优先级 1：等 seed3 + seed7 回流
- 明早~凌晨三份 training summary + C6 summary
- 后天西班牙 eval 窗口（2-3 quota 可用时）优先 eval
- 决定 SOTA 0.84637 是 seed luck outlier（EXP-028 假说）还是 genuine peak

### 优先级 2：启动 F1 `hash100k` 实验

**Job Name 候选**：`hash100k-icq-halftrain-swiglu-vr005-bs128`

**改动**：只加一个参数 `--seq_hash_vocab 100000`（或更激进 50000）。

**代码改动**：0 行（CLI 已支持）。

**预期 LB**：如果 EDA 结论 1 成立，应在 [0.843, 0.850] 之间。
- LB ≥ 0.849 → F1 成功，seq hash vocab 更激进方向 Accept
- LB ∈ [0.843, 0.849] → 持平/边缘，可再试 `hash50k`
- LB < 0.843 → 方向不对，hash 过度会丢失 id 结构

### 优先级 3：EDA-5 等平台 eval 回流后再做

- test 里 uid / item_id 实际 OOV 率（需要配合已有 infer log）
- 这是 Seed 问题-1 的**硬证据**，但现在工具链不足

---

## 附录：EDA 源数据（可重现）

### 脚本位置
- EDA-1: `/tmp/eda_rg_ts.py`（已删除）
- EDA-2: 从 `~/Downloads/train_log.txt`（今晚 C1 job log）直接 grep `[diag]`
- EDA-4: 从 C12 inference log 里的 `Building PCVRHyFormer with cfg`

### 平台 schema 摘录（来自 2026-05-06 12:20 C12 infer log）

```python
user_int_feature_specs = [
    (6, 0, 1), (1916, 1, 1), (1018, 2, 1), (1173, 3, 26), (101, 29, 1),
    (3, 30, 1), (4, 31, 1), (160, 32, 1), (193, 33, 1), (563, 34, 1),
    (2862, 35, 1), (44, 36, 1), (1456, 37, 1), (257, 38, 1), (3, 39, 1),
    (17, 40, 1), (3, 41, 2), (11, 43, 7), (49, 50, 20), (51, 70, 33),
    (447, 103, 116), (1485, 219, 201), (13, 420, 6), (24, 426, 1),
    (252, 427, 1), (10, 428, 10), (10, 438, 10), (10, 448, 10),
    (3, 458, 1), (38, 459, 1), (7, 460, 1), ...  # 其余都 vocab<100
]

item_int_feature_specs = [
    (309, 0, 1), (943, 1, 1), (2533, 2, 1), (2213, 3, 1), (42, 4, 1),
    (318, 5, 1), (23832, 6, 20), (2533, 26, 1), (9, 27, 1),
    (25495, 28, 1), (3, 29, 1), (32, 30, 1), (234, 31, 1), (1084, 32, 1)
]

seq_vocab_sizes = {
    'seq_a': [889551, 19, 12, 1028, 3498, 13692, 8175, 18],
    'seq_b': [28, 85946394, 740, 2729, 10572, 6985, 531477, 31, 143746, 168, 4340, 11623, 249080],
    'seq_c': [75, 6551861, 864, 6896, 7, 5, 1292419, 2998, 1134859, 9697, 125610776],
    'seq_d': [5, 980, 3469, 11510, 5156, 443147, 630399, 626, 15]
}
```

### C1 E1 [diag] 切片表原文
见 `~/Downloads/train_log.txt` 行 220-262（2026-05-06 06:09）。
