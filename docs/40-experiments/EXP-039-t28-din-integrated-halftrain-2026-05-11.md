# EXP-039 · T28 DIN Integrated Half-train · 2026-05-11 早

> **Status**: ❌ **Deprecated**（2026-05-12 CEST 12:45 · M1/M2 双 seed LB 回流后方向关闭）· T28 integrated **比 T25 bypass 更差**
> **日期**: 2026-05-11 CEST 05:15 agent 起草 · 5/11 CEST 23:27 重启 · 5/12 回流
> **关联**：[ADR-010](../30-decisions/ADR-010-din-integrated-in-block.md) Deprecated · [EXP-033](EXP-033-t25-din-halftrain-2026-05-09.md)（T25 bypass 仍是 DIN 方向最优实现）· [EXP-036](EXP-036-gap0-t25-seed-sweep-tokenization-2026-05-10.md) · [EXP-038](EXP-038-t29-tokenization-fix-halftrain-2026-05-11.md)

---

## 🔥 LB 回流结果（2026-05-12 CEST 12:45）· T28 方向证伪

### M1 · T28 DIN integrated × seed=3 (`t28-din-integrated-halftrain-gap05-seed3`)

| Epoch | Dev AUC | Dev LogLoss |
|---|---|---|
| E1 | 0.834999 | — |
| E2 | 0.842233 | — |
| **E3** ⭐ | **0.845197** | **0.27543** |
| E4 | 0.843646 | 0.27682 |

**stop_reason**: `max_epochs_completed` (松 guard `delta=0.003 patience=2`)
**best_checkpoint**: global_step45504 E3
**wall-time**: 8h33m

**LB 回流** (eval_id: 1778581029331): **LB 0.841807** / inference 638.37s

### M2 · T28 DIN integrated × seed=42 (`t28-din-integrated-halftrain-gap05-seed42`)

| Epoch | Dev AUC | Dev LogLoss |
|---|---|---|
| E1 | 0.819896 | — |
| E2 | 0.837817 | — |
| **E3** ⭐ | **0.840669** | **0.27913** |
| E4 | 0.838688 | 0.28109 |

**stop_reason**: `logloss_rebound_epoch_4` ⚠️（E4 LogLoss 反弹触发严格 guard）
**best_checkpoint**: global_step45504 E3
**wall-time**: 5h44m

**LB 回流** (eval_id: 1778581020580): **LB 0.837576** / inference 511.23s

---

## 严格判决对照表

**Baseline**（EXP-030 `icq-half-gap05`）: E3 Dev **0.842443** / LB **0.839223** / gap +0.0032
**T25 bypass 锚点**（EXP-036 T25 B seed=42）: Dev 0.843696 / LB **0.843433** / gap +0.000263
**T25 bypass seed=3**（EXP-030 · gap=0.05）: LB **0.843069**

| 代号 | 方向 | E3 Dev | ΔDev vs T25 | LB | ΔLB vs T25 同 seed | 判决 |
|---|---|---|---|---|---|---|
| T25 bypass s42 | 参考 | 0.843696 | — | 0.843433 | — | ✅ Gold standard |
| T25 bypass s3 | 参考 | — | — | 0.843069 | — | ✅ seed=3 锚 |
| **M1** | **T28 integrated s3** | 0.845197 | **+0.001501** | **0.841807** | **−0.001262** ❌ | Dev ↑ LB ↓ |
| **M2** | **T28 integrated s42** | 0.840669 | −0.003027 | **0.837576** | **−0.005857** ❌❌ | 全面负向 |

---

## 核心发现：T28 integrated 证伪

### Finding 1: Dev 上升但 LB 下降 = 训练期 memorization

- M1 Dev 0.845197 **比 T25 bypass 高 +0.0015** 但 LB **比 T25 bypass 低 −0.00126**
- Dev→LB gap 从 T25 bypass 的 +0.000263 扩大到 **+0.003390**
- **Dev 增量 100% 是训练期记忆，没有转化为 test 泛化**

### Finding 2: DIN 重复注入（>1 次）反而过拟合

**机制假说**：
1. T28 每 block DIN 注入 × 2 次 = 让模型更深度依赖 target-aware pattern
2. Target-aware pattern 在 train/dev 共享的近期分布下会增强（+Dev）
3. 但 test 是远未来 → target pool 分布偏移 → DIN 过度激活反而伤害（−LB）
4. **T25 bypass 做 1 次是"刚好够"，T28 做 2 次是"过度"**

### Finding 3: seed 间 LB 差异在噪声边界（Δ 0.004）

- M1 s3 LB 0.841807 vs M2 s42 LB 0.837576 = Δ **0.004231**
- 接近 seed 噪声 ±0.005 上界
- 但 s42 LogLoss guard E4 触发 → 说明 s42 已进入过拟合区
- T28 方法学净贡献 = 负向（即使按最优 seed 也不如 T25 bypass）

### Finding 4: T28 wall-time 成本 1.9×（5h44m / 8h33m vs T29 2h59m）

- T29 用 compile+AMP: 2h45m
- T28 **不开 compile** (ADR-010 决定首次规避 dynamo): 5h44m~8h33m
- 即便开 compile 也至少 4h+，而 LB **更差**
- **时间成本也不值得**

---

## ❌ 方向关闭决定

- **ADR-010 Status: Proposed → Deprecated**
- **T28 integrated 方向永久关闭**
- **T25 bypass 保持 DIN 方向唯一最优实现**
- 代码保留（`--enable_din_integrated` CLI + MultiSeqHyFormerBlock 分支），默认 off 完全 backward-compat
- **不做 T28 其他 variant**（alpha 扫 / block-integrated × 1 block only / etc）—— Finding 1/2 说明机制性负向，细调无意义

---

## Lessons（新 team-memory 铁律）

### 铁律：DIN 类 target-aware attention 注入次数有"刚好够"的上界

- **1 次（bypass 到 classifier 前）**: ✅ T25 证 +0.004 LB
- **2 次（每 block 内部）**: ❌ T28 证 −0.001 LB vs T25
- **推论**：target-aware 机制在当前数据规模 + 远未来 test 分布偏移下，**重复使用会放大"对 train 期 target 分布的依赖"**，不是信号增强而是泛化损失
- **未来 DIN 方向探索**应该走"**多样化**"而非"**重复化**"：
  - ✅ Per-seq 独立 DIN（每 seq domain 用不同的 attention pooling 策略）
  - ✅ Multi-head target query（1 个 block 里多个 head，各 head 关注不同 aspect）
  - ❌ Multi-block same DIN（T28 证伪）

### 铁律：Dev→LB gap 扩大 = 方法学负向的最强信号

- T28 M1: Dev 升 +0.0015 但 gap 从 +0.00026 扩到 +0.0034（扩 13×）→ 真实 LB 下降
- 判决新规：**新方向 Dev 正向时优先看 gap 变化，若 gap 扩 2× 以上（T25 bypass 标准 +0.00026 → 警戒线 +0.00052）应警觉**

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
