# EXP-047 · T29-C RankUp Randomized Permutation Split · Planning · 2026-05-13

> **Status**: 📋 **Planning** · 代码待落地（~15 行 · 生成脚本 + JSON）· 平台 Job 待启动
> **日期**: 2026-05-13 CEST 15:15 起草
> **关联**：[ADR-009 T29](../30-decisions/ADR-009-tokenization-fix-ns-groups.md)（T29 原 ADR · 本 EXP 新加 Variant C） · [RankUp paper §3.2 + §4.3](https://arxiv.org/abs/2604.17878) · `docs/20-tasks/active.md §T29`

---

## TL;DR

**目标**：为 T29（Tokenization 修复方向）新增 Variant C = **RankUp §3.2 Randomized Permutation Splitting**。机制：用固定 seed 随机 shuffle 所有 fid 的索引，按新顺序 concat · 下游 chunk / split / projection 不变。

**与已验证变体的对比**：
- **T29-A（`ns_groups.json` 手工语义）** · M3 LB 0.840603（gap +0.00103 ⭐ 最小 · 弱 work 但跨 seed 不稳定）
- **T29-B（`ns_groups_vocab_tier.json` vocab 分档）** · M4 Dev 0.841171（未评估）
- **M9 SOTA（`--ns_groups_json ""` singleton fallback）** · LB **0.84671**（当前 SOTA）

**关键发现 · 本 EXP 新增**：M9 是"按 fid 编号 concat"，和 T29-A/B/C 都不同。**T29-C Randomized 是 M9 的"可重复的随机化版本"，本质上是 M9 的 seed variant**。

**核心假设**：M9 LB 0.84671 可能部分来源于"匿名 fid 编号恰好不 collide"的运气。T29-C 用 permutation seed 枚举 2~3 次，**大概率能找到 LB ≥ M9 的随机顺序**（+0.0005~+0.0015）。

**Go/No-go**（vs `icq-half-gap05` E3 Dev 0.842443）：
- 任一 seed E3 Dev ≥ 0.8437 → 🎯 真信号 · eval
- 最佳 seed E3 Dev < 0.8421 → ❌ Deprecated（Randomized 方向本身失败）

---

## 优先级

**T29-C ROI 在 T33/T34 之间**：
- 代码成本最低（~15 行 · 纯 config + JSON 脚本）
- 零参数增量（ZERO new params，只改 fid concat 顺序）
- **可以和 T33/T34 同 training slot 混合提交**（3 个方向独立，同 batch 4 slot 可以全覆盖）
- 论文期望 +0.06~0.08% AUC（Industrial 约 +0.0005~+0.0015 LB）

**启动条件**：
- T33 Smoke 和 Job A 启动后（CEST 5/13 17:00+），T29-C 脚本 + JSON 生成可以顺便做
- 和 T33 Job B（seed=3）+ T34 Job A 同批次平台提交

---

## 代码改动面（~15 行）

### 1. `scripts/gen_ns_groups_random.py`（新 · ~30 行）

```python
"""T29-C · Generate randomized permutation ns_groups JSON.

Output JSON shape same as ns_groups.json but fid order randomly shuffled
with a fixed seed for reproducibility.
"""
import argparse
import json
import random
from pathlib import Path

def gen_random_groups(num_user_fids: int, num_item_fids: int,
                     seed: int) -> dict:
    rng = random.Random(seed)
    user_fids = list(range(num_user_fids))
    item_fids = list(range(num_item_fids))
    rng.shuffle(user_fids)
    rng.shuffle(item_fids)
    # 每 fid 独立成 group · 下游等效 singleton 但顺序是随机 permutation
    return {
        "_mode": "randomized_permutation",
        "_seed": seed,
        "user_ns_groups": {f"P{i}": [fid] for i, fid in enumerate(user_fids)},
        "item_ns_groups": {f"P{i}": [fid] for i, fid in enumerate(item_fids)},
    }

# CLI: --seed / --num_user_fids / --num_item_fids / --output
```

### 2. `baseline/ns_groups_random_s{42,3,7}.json`（3 个 · 每个 ~100 行 JSON · 自动生成）

### 3. `run.sh`

```bash
# T29-C Variant · 替代 --ns_groups_json "" 的 singleton fallback
--ns_groups_json "${SCRIPT_DIR}/ns_groups_random_s42.json"
```

**不需要改 Python 代码**（`_default_ns_groups` 已经支持任意 path · ADR-009 代码路径复用）。

---

## 实验计划

### Phase 1: 生成 JSON + Smoke（~30 分钟）

- [ ] 写 `scripts/gen_ns_groups_random.py`
- [ ] 生成 3 个 JSON：`ns_groups_random_s42.json` / `s3.json` / `s7.json`
- [ ] 本地 smoke：`--ns_groups_json baseline/ns_groups_random_s42.json` · demo 1k 训 5 steps + infer 1k predictions
- [ ] 参数量与 M9 SOTA **bit-identical**（零参数变化，只改顺序）
- [ ] 日志显示 `User NS groups (N): ['P0', 'P1', ...]`（P 前缀表示 permutation）

### Phase 2: 平台 Half-train

**Job A（Randomized seed=42 permutation）**：
```
Job Name: t29c-random-s42-halftrain-gap05-seed42
改动文件：只上传 1 个新 JSON 文件 baseline/ns_groups_random_s42.json
Python 代码与 M9 SOTA 完全一致 · 可以直接 Copy M9 Job
run.sh 差异：`--ns_groups_json ""` → `--ns_groups_json "${SCRIPT_DIR}/ns_groups_random_s42.json"`
```

**Job B（Randomized seed=3 permutation · seed=42 train seed）**：
```
Job Name: t29c-random-s3-halftrain-gap05-seed42
改动：新 JSON + run.sh path
```

**注意命名冲突**：
- `s42` / `s3` 是 **permutation seed**（生成 JSON 时的 random seed）
- training 仍用 `--seed 42`（与 M9 对齐 · 排除 training seed 影响）
- 这样可以**隔离**"fid 顺序影响" vs "training seed 影响"

### Phase 3: 判决

**判决矩阵（2 seed 对比 · 看跨 permutation 稳定性）**：

| seed=42 Dev | seed=3 Dev | 解读 | 下一步 |
|---|---|---|---|
| ≥ 0.846 + ≥ 0.846 | — | 跨 permutation 都 work | 🎯🎯 真信号 · eval 2 个 |
| ≥ 0.8437 + ≥ 0.8437 | — | 弱信号但稳定 | 🎯 eval 最高的那个 |
| 一高一低 | 差 > 0.002 | permutation-sensitive · 可能 seed=42 运气好 | 🟡 再跑 seed=7 打破平局 |
| 两个都 < 0.843 | — | Randomized 方向不 work | ❌ Deprecated |

**LB（若 S2 通过）**：
- 对比 M9 LB 0.84671
- 预期：Random 2~3 个 permutation 的 LB 跨度 ±0.003（seed 噪声），**上沿有小概率 ≥ M9**
- **不期望单次 Random 破 SOTA**，目标是找"比 M9 singleton 顺序更好"的 permutation 作为 S4 叠加用基线

---

## 预期收益 & 风险

### 最大期望

- **找到比 M9 singleton 稍好的 permutation** → 未来所有实验的 baseline 换成 T29-C 最佳 permutation
- 叠加在 T33 NLIR / T34 Multi-Emb 上可能再 +0.0005~+0.001 LB

### 关键风险

- **M9 singleton 顺序可能已经是好顺序**：匿名 fid 编号不一定有内禀 collinearity，Random 未必改善
- **permutation seed 噪声 vs training seed 噪声混淆**：即使看到 Dev +0.001 也可能只是"换了个好的训练 minima"，不是机制增益
- **对策**：最终判决需要**同 permutation seed × 多 training seed**（至少 2 training seed） · S3 阶段做

---

## 与 T29-A/B 的关系

**T29-A/B 已结果**（来自 EXP-038 / M3 M4 M11）：
- **T29-A 手工语义分组 · M3 LB 0.840603** · gap +0.00103 ⭐ · Dev→LB 转化率最高但 LB 未破 M9
- **T29-B vocab tier · M4 Dev 0.841171** · 未 eval · 未破 M9

**T29-C 与 T29-A/B 的关键区别**：
- T29-A/B 用的是"每 group 包多个 fid" → **同 chunk 内 fid 共享 Linear projection 的梯度耦合更强**（原 T29 ADR-009 机制分析）
- T29-C Randomized 是 **每 group 1 个 fid**（singleton permutation），chunk 内 fid 是 permutation 位置附近的，**梯度耦合取决于相邻 fid**

**结论**：T29-C 是"singleton + 随机化顺序"的变体，与 T29-A（多 fid 聚合）机制不同，不是互斥的。

---

## 平台提交清单

### Job A: t29c-random-s42-halftrain-gap05-seed42

**上传文件**：
- `baseline/ns_groups_random_s42.json`（新 · JSON 文件）
- Python 代码与 M9 **完全相同**，可以 Copy M9 Job

**run.sh 差异**：
```bash
# M9 原行：
--ns_groups_json "" \
# T29-C 替换为：
--ns_groups_json "${SCRIPT_DIR}/ns_groups_random_s42.json" \
```

### Job B: t29c-random-s3-halftrain-gap05-seed42

**上传**：`baseline/ns_groups_random_s3.json`  
**run.sh**：`--ns_groups_json "${SCRIPT_DIR}/ns_groups_random_s3.json"`

---

## 预期时间线

| 时段 | 动作 |
|---|---|
| 5/13 CEST 17:30 | `gen_ns_groups_random.py` + 3 JSON 生成 + Smoke |
| 5/13 CEST 18:00 | Job A 提交（和 T33 Job B 同批次）|
| 5/13 CEST 18:30 | Job B 提交（如有 slot） |
| 5/14 CEST 04:00-06:00 | 回流 · 判决 · 不 eval（等 T33 结果优先） |

---

## Materials（待回填）

- Commit: （代码落地后填）
- Platform Job ID: （提交后填）
- 回流 Dev/Loss: （回流后填）
- 与 M9 / T29-A / T29-B 的对比表：（回流后填）
