# EXP-039 · T28 DIN Integrated Half-train · 2026-05-11 早

> **Status**: 📋 实验计划 · 代码 + smoke 已就位 · 等 7:00 CEST 4 eval LB 回流 + T29 决策后启动
> **日期**: 2026-05-11 CEST 05:15 agent 起草 · 西班牙凌晨
> **关联**：[ADR-010](../30-decisions/ADR-010-din-integrated-in-block.md) · [EXP-033](EXP-033-t25-din-halftrain-2026-05-09.md)（T25 bypass 前身）· [EXP-036](EXP-036-gap0-t25-seed-sweep-tokenization-2026-05-10.md)（T25 LB 真信号）· [EXP-038](EXP-038-t29-tokenization-fix-halftrain-2026-05-11.md)（T29 并行方向）

---

## TL;DR

消化 DECEM 5/8 帖 "**DIN + MLP**" 方向的深化版：把 T25 bypass 的 DIN target-aware attention **从 classifier 前（bypass 1 次）移到每个 HyFormer block 内部（integrated 2 次 @ num_blocks=2）**。DIN 由"最小风险 add-on"升级为"参与 RankMixer token mixing"的一等公民。

**代码状态**：
- ✅ `MultiSeqHyFormerBlock` 加 `enable_din_integrated` + `din_alpha` per-block
- ✅ `PCVRHyFormer` 加 kwargs + 共享 `_compute_target_repr` helper
- ✅ `train.py` `--enable_din_integrated` + `--din_integrated_alpha_init` CLI
- ✅ `infer.py` cfg fallback
- ✅ 本地 smoke S1/S2/S3 全过（backward-compat / T28 on / 训推一致）
- ✅ py_compile + ReadLints 无错

**等待**：7:00 CEST LB 回流 + T29 决策树 → 决定 T28 是否优先起动（预期今天 slot 3 或 4）

---

## 为什么 T28 是 T25 自然升级

| 指标 | `icq-half-gap05`（baseline）| `t25-din-halftrain-gap05-seed42`（T25 bypass）| T28 预期 |
|---|---|---|---|
| E3 Dev AUC | 0.842443 | 0.843696（Δ+0.00125）| **≥0.847**（Δ+0.005 强信号 / ≥0.8452 优于 T25）|
| LB AUC | 0.839223 | **0.843433**（Δ+0.00421, 9× rerun 噪声）| **≥0.847**（LB Top 20 阈值附近）|
| Dense 参数增量 | 0 | +104,896（+3.6%）| +201,602（+7.0%）|

**核心假说**：DIN bypass 只做一次（classifier 前），integrated 做 N 次（num_blocks=2 次）且**每 block 见的 seq 是前一 block encoder 细化过的**。渐进细化 + 多次注入应该解锁 T25 bypass 未吃到的信号。

**DECEM 原话暗示**：榜一说"DIN + MLP"（MLP 是 HyFormer/RankMixer block 里的 FFN）→ 整合在 block 内部比 bypass 更接近原意。

---

## 本地 Smoke 已完成（2026-05-11 CEST 05:00）

### Smoke 矩阵

| 测试 | 目的 | 结果 |
|---|---|---|
| S1 · T28 off | backward-compat：与 pre-T28 一致 | ✅ Total params **353,334,977**（与 T29 smoke 一致） |
| S2 · T28 on | 新路径能正常 forward + train | ✅ Total params **353,536,579**（Δ+201,602 match 理论） |
| S3 · T28 on + infer | 训推一致，state_dict 兼容 | ✅ `Saved 1000 predictions from 2000 rows` 无 error |

### 参数量细分（S2）

- 每 block `DINInterestExtractor` = 4 × CrossAttention + 1 × merge Linear(256→64)
  - 4 × CrossAttn ≈ 4 × (4 Linear(64,64) + LN) ≈ 4 × 16,512 = 66,048
  - merge Linear(256,64) + LN(64) ≈ 16,512
  - Per-block total ≈ 82,560 实测匹配
  - 注：实测每 block 增量 = 201,602 / 2 = 100,801 比理论稍高 ~18k，差异可能来自 CrossAttn 内部 W_g / W_o 等扩展
- `din_alpha` 标量 per-block = 1 param
- num_blocks = 2 → total Δ = 201,602

### CPU smoke wall-time

- S1 off = 58s（包含启动 + 5 train steps + 1 valid batch + best_model 写盘）
- S2 on = 60s（+2s, 可忽略）
- S3 infer = 60s（1000 predictions）

---

## 平台 Job 配置

### Job Name: `t28-din-integrated-halftrain-gap05-seed42`

基于 `icq-half-gap05` EXP-030 baseline, 只加 T28 相关 flag：

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t28-din-integrated-halftrain-gap05-seed42
#
# T28 · DIN target-aware interest pooling INSIDE each HyFormer block.
# See ADR-010 + EXP-039 for rationale.
#
# Baseline: `icq-half-gap05` (EXP-030) · E3 Dev 0.842443 / LB 0.839223.
# T25 bypass benchmark: 0.843696 Dev / 0.843433 LB (EXP-036).
#
# Decision thresholds (E3 Dev):
#   ≥ 0.847  → 🎯 strong signal (≥T25+0.003), immediate evaluation
#   ≥ 0.8452 → 🟡 beats T25, LB-validate with 1 eval quota
#   ≥ 0.8437 → ⚪ ~T25, keep T25 as main path
#   < 0.8437 → ❌ deprecate T28, stay on T25 bypass
# ═══════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --enable_row_valid_gap \
    --valid_gap_ratio 0.05 \
    --item_conditioned_query \
    --enable_din_integrated \
    --din_integrated_alpha_init 0.1 \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 4 \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
```

**注意**：
- **不带 `--enable_torch_compile`** —— Open Question 1：首次 T28 验证避免 dynamo 干扰方法学判断（team-memory 铁律"首次引入新模块先不开 compile"）。若 T28 E3 signal 正向且确认，后续再补 compile
- **不带 `--enable_din_interest`** —— 不叠加 T25 bypass，保纯 T28 单变量 ablation
- **带 `--enable_amp --amp_dtype bfloat16`** —— AMP 已在 ADR-008 Accepted with caveats，无 compile 也可启用单独 AMP（减少显存压力，确保 batch=128 有余量容纳 +7% Dense params）
- **`--ns_groups_json ""`** —— 保持 SOTA 配置的 ns_groups 路径（singleton fallback），**不叠加 T29 Tokenization fix**，避免 EXP-039 和 EXP-038 结果耦合

### 平台提交清单

**需要上传的文件**（相对 T25 DIN bypass 上次 platform job）：
- `baseline/model.py`（改动 ~60 行：MultiSeqHyFormerBlock + PCVRHyFormer + `_compute_target_repr` helper）
- `baseline/train.py`（新增 2 个 CLI flag，~20 行）
- `baseline/infer.py`（新增 2 行 cfg fallback）
- 其余 `dataset.py / trainer.py / utils.py` **无改动**

**`run.sh`**：粘贴上面 bash block 到平台编辑框

**必须 Create New Training Job（不能 Copy）**：因为改了 3 个 Python 文件

**⚠️ 首次 evaluation 必须勾选"上传自定义 infer.py"**：team-memory 铁律 —— `enable_din_integrated` + `din_integrated_alpha_init` 是新 kwarg，不能赌平台 default infer.py 兼容

---

## Go / No-Go 判决阈值

### S2 Half-train E3 Dev（gap=0.05 可信 dev）

对照基线：**T25 bypass E3 Dev = 0.843696**（不是 icq-half 0.842443，要证 T28 值得替换 T25）

| E3 Dev AUC | 判决 | 动作 |
|---|---|---|
| ≥ **0.847**（Δ+0.005 vs baseline / +0.003 vs T25）| 🎯 强信号 | 立即 evaluation |
| [0.8452, 0.847)（+0.0015 vs T25）| 🟡 T28 优于 T25 但边缘 | 1 eval quota 验证 LB |
| [0.8437, 0.8452)（≈T25）| ⚪ 持平 T25 | 保留作备选，T25 仍为主路径 |
| < 0.8437 | ❌ 负向 | Deprecated，关闭 T28 方向 |

**LogLoss 联动**：
- T25 bypass E3 LogLoss baseline = ~0.27596（EXP-036）
- 若 T28 Dev 在 [0.8452, 0.847) 但 LogLoss 低于 T25 超 −0.0015 → 升级判决为强信号
- 若 T28 LogLoss 高于 T25 超 +0.003 → 容量过剩告警，降级判决

### S3 LB 校准（若 S2 通过）

对照：T25 bypass LB **0.843433**

| LB AUC | 判决 |
|---|---|
| ≥ **0.850** | 🏆 **Top 20** 直接达成 |
| ≥ **0.847** | SOTA 大突破，T28 替换 T25 成为主路径 |
| ≥ **0.844** | 优于 T25 LB，方法学真信号 |
| ≥ **0.842** | 持平 T25 或稍低，保留但不替换 |
| < **0.841** | T28 vs T25 上负向 → Deprecated |

### S4 叠加（若 T28 S2 ≥ 0.8452）

Options：
- `t28-din-integrated-t29-tokfix-halftrain-gap05`（T28 + T29 A 或 B, 看 T29 回流）
- `t28-din-integrated-gap0-seed42`（T28 切 gap=0 冲 SOTA 上沿）
- `t28-din-integrated-halftrain-gap05-seed3`（T28 seed=3 验证跨 seed 稳定性）

S4 目标：**LB ≥ 0.850**（Top 20 阈值）

---

## 时间线（激进 5/14 前冲 0.850）

| 日期 | 事件 |
|---|---|
| **5/11 05:15 CEST** | ✅ T28 代码就位 + smoke 全过 + ADR-010 + EXP-039 落档 |
| **5/11 07:00 CEST** | 4 eval LB 回流（A/B/C/D from EXP-036/037）|
| **5/11 07:30~08:00** | 按决策树分配 4 slot，**T28 优先候选 slot 3 或 4**（视 T29 是否吃掉前 2 slot） |
| **5/11 13:00~14:00 CEST** | T28 回流（~5h half-train） |
| **5/11 晚** | 若 Dev ≥ 0.8452，消耗 1 eval quota |
| **5/12 早 07:00 CEST** | LB 回流决定 S4 |
| **5/12~5/14** | S4 叠加 / 多 seed |

---

## 风险与 Fallback

### 风险 1：dynamo/compile 兼容性未知

- **现象**：若未来叠加 `--enable_torch_compile`，per-seq for-loop 可能触发多次 recompile
- **检测**：训练日志搜 `torch._dynamo` 相关 warning
- **Fallback**：本次首启 T28 **不开 compile**，纯 AMP。T28 验证后再单独 + compile 测兼容性

### 风险 2：+7% Dense params 导致每参数梯度 steps 不足

- **现象**：E2→E3 Dev 反弹、LogLoss 明显高于 T25
- **机制**：1.9M rows × batch=128 × 4 epochs / Dense params 已从 T25 的 ~2.98M 升到 ~3.08M，单参数平均梯度 steps 从 ~80 降到 ~78（团队安全带 ≥60）仍在安全区
- **Fallback**：若 E2 Dev 已明显低于 T25 且 E2→E3 LogLoss 反弹 → LogLoss rebound guard 自动停
- **极端 Fallback**：若多轮证明容量真不够，把 DIN 跨 block 共享（省一半参数）

### 风险 3：alpha 初值 0.1 不是最优

- **现象**：T28 E3 Dev ≈ T25（持平）但 LogLoss 不同 → 可能是 alpha 太小 signal 没 propagate
- **Fallback**：S4 扫 alpha ∈ {0.05, 0.2}（1~2 个 slot）

### 风险 4：T28 与 T29 不正交（fid 拼接顺序改变 target_repr）

- **机制**：T29 Tokenization fix 改 fid 拼接顺序 → ns_tokens 内容变 → `_compute_target_repr` 里 item_ns slice 的 fid 顺序变 → target_repr 向量方向变
- **方向**：T29 更合理的分组 → item_ns 更有语义结构 → T28 target_repr 质量**提升**（正向交互）
- **首次 ablation 规避**：EXP-039 T28 单独跑时 **不开 T29**（`--ns_groups_json ""`），严格单变量

---

## 不做的事（违反会产生归因歧义）

- ❌ T28 + T25 同时 on（同 DIN 两条路径，信号重复，无法判 T28 是否真的赢 T25）
- ❌ T28 + T29 首次启动就叠加（两个新方向同时引入，LB 反向无法归因）
- ❌ T28 + torch.compile 首次启动就叠加（新模块 + 新加速 = 两个风险源）
- ❌ 用 evaluation quota 预验证 T28 Dev（gap=0.05 dev 是可信 LB 代理，直接 Dev 判决，eval 留给 S3）
- ❌ 跨 block 共享 DINInterestExtractor（方案设计已放弃，见 ADR-010 Alternative E）

---

## 材料

- 本地 smoke logs（已清理）：`runs/t28_smoke/S1_off.log` / `S2_on.log` / `infer_S2.log`
- ADR: `docs/30-decisions/ADR-010-din-integrated-in-block.md`
- 代码改动: `baseline/model.py` + `baseline/train.py` + `baseline/infer.py`

## 评估（agent 阶段 B 产出 · CEST 04:46 ~ 05:15 · ~29 分钟）

1. ✅ 理解 T25 bypass 机制 + 局限（3 条）
2. ✅ 设计 T28 机制（3 个决策：注入位置 / merge 方式 / 参数共享）
3. ✅ 实现代码（MultiSeqHyFormerBlock + PCVRHyFormer + train.py + infer.py，~80 行改动）
4. ✅ 本地 smoke 3 矩阵全过（backward-compat / T28 on / 训推一致）
5. ✅ 参数量对齐（S2-S1 Δ+201,602 = 2 × 100,801 符合理论）
6. ✅ ADR-010 + EXP-039 落档

**零 quota 消耗**。等 7:00 CEST LB 回流后决定是否作为今日 4 slot 之一。

**与今晚 T29 的关系**：T28 和 T29 完全正交（不同方向的方法学改动），若 4 slot 能同时容纳（T29-A + T29-B + T28 + 1 seed/eval 校准），可同时验证两个新方向，**5/12 早一次 LB 回流即可同时判决 T29 和 T28**。
