# ADR-009 · Tokenization Fix · Enable ns_groups.json for Fid Ordering（T29）

> **Status**: **Proposed**（2026-05-11 CEST 04:30 起草 · 等 T29-A/B/C half-train 回流判决）
> **Date**: 2026-05-11 Proposed
> **Related Tasks**: T29（见 `docs/20-tasks/active.md`，待 open）
> **Related Experiments**: EXP-038（`t29-tokfix-*-halftrain-gap05`，4 epoch half-train · 待启动）
> **External References**:
> - 小红书 · DECEM（Industrial Track Rank 1, LB 0.852181, 2026-05-08 帖子）：
>   > "Tokenization 设计的小修改（作者原话："简单做了一些修复，让其更加合理化"）"
> - `baseline/ns_groups.json` 原作者（2026-04-26 commit `fb2450b`）手工分组定义（U1~U7 + I1~I4）
> - `docs/10-research/community-intel.md` §工业赛道 Rank 1 DECEM 情报

---

## Context（背景与触发原因）

### 触发证据

#### 1. 机制层：当前 NS tokenizer 的 fid 拼接顺序是"匿名 fid 编号序"

`baseline/model.py::RankMixerNSTokenizer.forward` 的工作流程：

1. 按 `groups` 里 fid 顺序，对每个 fid 查 embedding 得到 `(B, emb_dim)` 向量
2. 全部 concat 成 `(B, total_num_fids × emb_dim)` 扁平向量
3. pad 到可被 `num_ns_tokens` 整除，再 equal-split 成 N 个 chunk
4. 每 chunk 独立 Linear → 得到 N 个 NS token

**关键 insight**：`groups` 只控制 **fid 的 concat 顺序**（不影响 token 数、不影响参数量）。**相邻 fid 进同一 chunk，不相邻 fid 进不同 chunk**。chunk 内的 fid 共享一个 Linear projection，训练时它们的梯度耦合。

#### 2. 历史配置：我们过去 48 天**一直在用"singleton fallback"顺序**

`baseline/run.sh`（2026-04-26 commit `fb2450b` 起）写：
```
--ns_groups_json "" \
```

`train.py::main` 的处理：`os.path.exists("")` 返回 False → 走 fallback：
```python
user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
```

这相当于 **fid 按 schema.entries 的顺序（即 fid 编号序）逐个 concat**。由于 fid 编号已匿名化（4/28 官方答复：all features have been anonymized），**编号序 ≡ 随机序**。

**这是一个作者 4/26 时有意选择的 "快速解耦 group 数量约束"**（当时 num_queries=1 / num_ns=12 的整除性约束让 ns_groups.json 路径不可用）。但**当我们切到 rankmixer tokenizer + user_ns_tokens=5/item_ns_tokens=2 后，num_ns 不再被 groups 长度约束**，`ns_groups.json` 就**可以并且应该**被重新启用 —— 但 run.sh 从未被修正回去。

注释 `# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----` 清楚暗示原作者知道有另一套分组定义，但**没意识到在 rankmixer 路径下它依然生效（控制拼接顺序）**。

#### 3. 外部锚点：DECEM 榜一明说"Tokenization 修复"是 work 方向

2026-05-08 小红书帖子，Rank 1 `DECEM`（LB 0.852181, 方法学峰值 ~0.847）公开 4 个 work 方向：

| # | 方向 | 本项目状态 |
|---|---|---|
| 1 | torch.compile + AMP | ✅ T27 已做（工程增益，不直接上 LB） |
| 2 | 时间特征 | ✅ T12 已做（enable_time_of_day_features） |
| 3 | **Tokenization 修复** | ❌ **0 投入** |
| 4 | DIN + MLP | ✅ T25 bypass 已做（+0.004 LB，方法学真信号） |

**Tokenization 修复是唯一"榜一明说 work 但我们 0 投入"的方向**。不做就错过方法学剩余增量（估计 +0.002~0.005 LB）。

#### 4. 数据层支持：fid vocab 分布是极端长尾

从 `data/demo/schema.json` 读取（demo 1000 样本，真实平台 vocab 绝对值更大但**相对排序稳定**）：

- **user_int（46 fid）**：vocab 跨度 `2 ~ 2,844`（3 个数量级）
  - 大 vocab (>1000)：fid 54, 3, 66, 56, 15, 4 —— 可能是 category / tag / interest ID
  - 中 vocab (100~1000)：11 个 fid
  - 小 vocab (≤100)：34 个 fid —— 可能是 gender/age bucket/platform 等低基数枚举
- **item_int（14 fid）**：vocab 跨度 `3 ~ 35,260`（4 个数量级）
  - 大 vocab (>1000)：fid 16, 11, 7, 12, 8 —— 可能是 item_id / category / brand
  - 小 vocab：fid 9, 83, 13, 81

当前 "fid 编号序 concat" 下的 chunk 内部：
- Chunk 0 = fid 0~8 个 fid × emb_dim
- 比如 user 的 chunk 1 = fid `3, 4, 15, 48, 49, 50, ...`（vocab 1840, 987, 1172, 100, 3, 2, ...）
- **完全"小 vocab 和大 vocab 混在一起"，Linear projection 被迫同时适应快速饱和和慢速稀疏的两种 embedding 特性**

### 问题域

| 维度 | 现状 | 目标 |
|---|---|---|
| Fid 拼接顺序 | 匿名 fid 编号（等于随机） | **语义 / vocab-tier 分组** |
| 同 chunk 内 fid 同质性 | 低（大小 vocab 混杂） | 高（同 tier / 同语义聚集） |
| Dev AUC on gap=0.05 | `icq-half-gap05` baseline 0.842443（E3） | 至少 +0.001（方法学真信号边界） |
| LB 期望（gap=0.05 配置） | baseline LB 0.839223 | +0.002~0.005 |
| 代码风险 | — | **0 行 Python 改动**（纯 config / JSON 文件变更） |
| 平台兼容性 | 平台 default infer.py 支持 ns_groups_json sidecar resolution | ✅ 已验证（`_resolve_ns_groups_path`）|

### Trade-offs（放弃了什么）

- **放弃"按统计相关性聚类分组"**（T29-D 候选）：需要对 1.9M train 计算 fid 两两 mutual info（~46² = 2116 pair × 1.9M 样本 ≈ 重活），且 EDA 能力在 EXP-037 刚证伪（平台 log 通路不透明）。**短期不做**，保留作 5/15 后备选。
- **放弃"按 fid 语义人工分组"超越 baseline/ns_groups.json**：原作者手工分组（U1~U7）已 contain 一些明显语义信号（U2=[48,49,89,90,91] 可能是 time-related / U7=[3,4,55-59,62-66] 可能是 interest dense 对照组），但 anonymized 下没办法做得更好。
- **放弃"修改 RankMixer FFN 架构"**（EXP-037 已暴露 Tokenization 真正的瓶颈在 fid ordering，不在 projection 结构）。

---

## Decision（最终决定做什么）

**T29 分 3 个独立变体单变量测试，全部复用现有 `--ns_groups_json` CLI，零代码改动**：

### T29-A · 启用 `baseline/ns_groups.json`（原作者手工语义分组）

- **改动**：`run.sh` `--ns_groups_json "$SCRIPT_DIR/ns_groups.json"`（去掉空字符串）
- **机制**：fid 按 U1~U7 + I1~I4 的作者手工语义分组顺序 concat
- **为什么首选 A**：
  - 原作者在 `ns_groups.json` 的 `_note_user_dense` / `_note_shared_fids` 注释里明显花过功夫
  - 注释 `_note_shared_fids` 说 "fid 62/63/64/65/66/89/90/91 appear in both user_int and user_dense" —— 这些被放在 U7 或 U2 里聚集（int 版与同 fid 的 dense 版对应），这是**一个真实的先验知识**
  - U3=[80]（单独一个 fid，原作者认为它是特殊信号？）
  - 是 baseline 作者最"低投入"的 Tokenization 假设，**大概率对齐 DECEM 原话"简单做了一些修复"**

### T29-B · 按 vocab tier 分组（`ns_groups_vocab_tier.json`）

- **改动**：`run.sh` `--ns_groups_json "$SCRIPT_DIR/ns_groups_vocab_tier.json"`
- **机制**：fid 按 vocab 大小分 3 tier（small ≤ 100 < mid ≤ 2000 < large），tier 内按 vocab 降序，单个 group
- **为什么做 B**：
  - 机制直接 —— 同 chunk 内 fid 的 embedding 饱和速度相近 → Linear projection 梯度更稳
  - vocab tier 是"可观测 / 可验证"的 —— 即使 anonymized，vocab size 本身是数据里给出的客观量
  - 生成脚本 `scripts/gen_ns_groups_vocab_tier.py` 可复用，复赛阶段重跑生成新 JSON
- **风险**：demo 的 vocab 绝对值和平台真实 vocab 不同（item fid 16 demo 35k / 平台可能几十 M），但**相对 tier 划分稳定**。tier 边界如果偏，可后续调（`--small_upper` / `--mid_upper`）

### T29-C · 根据 7:00 LB 结果决定启动

- **Option C1**: **`baseline/ns_groups.json` × gap=0** —— 若今早 A（`gap=0 × DIN × seed=42` E4 Dev 0.868924）LB 破 SOTA，说明 gap=0 领地还有 headroom，在 gap=0 上叠加 Tokenization
- **Option C2**: **T29-A × seed=3** —— 若 A LB 持平，需要多 seed 看 Tokenization 是否跨 seed 稳定
- **Option C3**: **T29-B 的 tier boundary 扫一档**（如 `--small_upper 50 --mid_upper 500`）

---

## Verification（验证计划 / Go / No-go 阈值）

### S1 本地 smoke（✅ 2026-05-11 CEST 04:30 完成）

- `py_compile` 全过
- T29-A smoke：`Loading NS groups from /.../ns_groups.json`, `User NS groups (7): ['U1', ..., 'U7']`, `num_ns=8, T=16, d_model=64` ✅
- T29-B smoke：`Loading NS groups from /.../ns_groups_vocab_tier.json`, `User NS groups (3): ['T_large', 'T_mid', 'T_small']` ✅
- **训推一致性验证**：T29-A 和 T29-B 各自训练后，infer.py 自动从 `*.best_model/` 目录加载 sidecar `ns_groups*.json`，产出 1000 predictions ✅
- 参数量两者都是 `Total parameters: 353,334,977`（与无 ns_groups 版本相同，证明**仅 fid 顺序变化，参数/容量不变**）

### S2 平台 half-train（待启动）

| 候选 | Job Name | 关键参数 | 判决阈值（vs `icq-half-gap05` baseline E3 Dev 0.842443）|
|---|---|---|---|
| T29-A | `t29-tokfix-halftrain-gap05-seed42` | `--ns_groups_json $SCRIPT_DIR/ns_groups.json` | **E3 Dev ≥ 0.8437**（Δ+0.0013, 3× rerun 噪声）→ 🎯 强信号 |
| T29-B | `t29-vocabtier-halftrain-gap05-seed42` | `--ns_groups_json $SCRIPT_DIR/ns_groups_vocab_tier.json` | 同上 |
| C1~C3 | 7:00 LB 后决定 | — | — |

**判决矩阵（gap=0.05 配置下）**：
- E3 Dev ≥ 0.8437（+0.0013）→ 🎯 方法学真信号 → 立刻 evaluation
- [0.8433, 0.8437) → 🟡 疑似 → seed=3 rerun
- [0.8421, 0.8433) → ⚪ 持平（在 ±0.0003 rerun 噪声 + 采样偏差内）→ Deprecated
- < 0.8421 → ❌ 负向 → 关闭该变体

### S3 Leaderboard 校准（仅 S2 通过后）

- 消耗 1 evaluation quota / 变体
- 阈值：**LB ≥ 0.842**（vs `icq-half-gap05` baseline LB 0.839223 的 Δ+0.003 = 方法学信号）
- 强信号：**LB ≥ 0.845**

### S4 叠加（仅 T29 A 或 B 在 S3 通过后）

- 与 T25 DIN 叠加 → `t29-tok-t25-din-halftrain-gap05-seed42`
- 与 gap=0 叠加 → `t29-tok-gap0-seed-sweep`
- S4 最终 LB 阈值：**≥ 0.847**（真正逼近 DECEM 方法学峰值 0.847）

---

## Consequences（后果）

### 正面

- **零代码风险** —— 纯 JSON + run.sh 参数变更
- **完全独立于其他方向** —— 与 T25 DIN / T27 compile+AMP / T22 gap=0.05 正交，不引入归因歧义
- **可扩展** —— 未来可生成更多变体（vocab tier 边界扫 / 用真实平台 vocab 生成 / 未来 EDA 后做 correlation-based）
- **落地平台成本最低** —— `ns_groups.json` 已在 baseline/ 下，`ns_groups_vocab_tier.json` 生成即可；run.sh 改一行

### 负面

- **半 dev 信号不够可信**：T29 可能在 gap=0.05 dev 上 Δ+0.0005~+0.001，在 ±0.0003 rerun 噪声 + ±0.001 采样偏差下**判决边缘化**。可能需要 2~3 seed 才能分辨出真伪信号。
- **猜测成分**：我们无法验证 DECEM 原话"合理化"具体指哪一种分组方式，T29-A/B 是概率覆盖
- **demo vocab 与平台 vocab 可能偏差**：T29-B 的 tier 边界 `(100, 2000)` 基于 demo 数据，平台真实 vocab 更大时 tier 划分可能不最优。Fallback：**若 T29-B 判决持平或负向，但 T29-A 正向 → 证明是"有意义分组比无意义分组好"，不是"vocab tier 分组最好"**，复赛时换 correlation-based 分组

### 红线检查（必读）

- ❌ 不 ensemble / 不 SWA / 不 multi-seed prediction avg — **T29 每变体独立 half-train，单变量 ablation**
- ❌ 不外部数据 — **ns_groups 信息源 100% 来自 schema.json / run.sh，无外部 data**
- ❌ 不改模型容量 —— **参数量不变（验证过）**
- ✅ 单一架构最终提交 —— **T29 最 work 的那个变体作为"fid 拼接顺序"标准配置，不产生多模型**

---

## Alternatives Considered（被放弃的方案）

### B · 修改 RankMixerNSTokenizer 强制每 fid 一个独立 Linear（like FaFE）

- **放弃理由**：FaFE 已经在 `--enable_fafe` 下实装（T25/H2），但**没在当前 SOTA 路径启用**，独立验证 ROI 前不引入。且与 T29 的"顺序修改"**解决的问题不同**（FaFE 解决 projection 共享问题，T29 解决 projection 内 fid 同质性问题），二者正交可叠加但**不能同时做避免归因歧义**。

### C · 把 user_dense 也参与 ns_groups 分组

- **放弃理由**：当前架构 user_dense 走单独路径（一个 NS token）不进 tokenizer，改动成本大、收益不明确。T29 只改 user_int / item_int 的顺序，**最小变更**。

### D · Correlation-based clustering（统计相关性聚类）

- **放弃理由**：EXP-037 证明平台 EDA 通路不透明（5 次失败 + 3 slot 浪费），本地 demo 1000 样本的 correlation 不稳定。**复赛阶段若有更多 quota + 能拿到 EDA 结果时再做**。

### E · 改 rank_mixer_mode 为 `ns_only` + 深度 Tokenizer

- **放弃理由**：超出本 ADR 范围（属于 block-level 改动，不是 tokenizer-level），且 DECEM 原话暗示修改是"小"的。

---

## Open Questions（悬而未决）

1. **平台真实 vocab 分布**：demo 里 user fid 54=2844 vocab 最大，真实平台可能完全不同（比如 user_id 类 fid 可能真实是 mil 级）。等 T29 平台回流后从日志读真实 `RankMixerNSTokenizer: total_emb_dim=X` 验证边界是否合理。
2. **tier 数量**：当前 3 tier (small/mid/large)，可能 2 tier 或 4 tier 更好，暂不扫以节省 quota。
3. **chunk boundary 对齐**：当前 `chunk_dim = ceil(total_emb_dim / num_ns_tokens)` pad 到整除，tier 边界不会对齐 chunk 边界 → 某个 tier 可能跨 chunk。目前接受这个"软性跨界"，若 T29-B 正向且想继续优化，可研究"强制 tier 边界对齐 chunk 边界"变体（需小改代码）。

---

## References

- EXP-038（T29 实验计划 + 回流数据）
- ADR-003（特征工程总图）
- EXP-035（Kaggle 金牌视角 audit · 列 Tokenization 为头号未消化线索）
- `baseline/model.py:1812-1964`（RankMixerNSTokenizer 实现）
- `baseline/train.py:366-370, 619-632`（ns_groups_json CLI 与 fallback）
- `baseline/infer.py:70-149`（_resolve_ns_groups_path + _load_ns_groups）
- `scripts/gen_ns_groups_vocab_tier.py`（T29-B JSON 生成脚本）
- Community Intel `docs/10-research/community-intel.md` §工业赛道 Rank 1 DECEM
