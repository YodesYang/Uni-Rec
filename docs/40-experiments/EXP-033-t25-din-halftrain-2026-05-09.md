# EXP-033 · T25 DIN Target Attention 首次 half-train（`t25-din-bypass-halftrain-gap05`）

**日期**：2026-05-09（西班牙时间早 · 待 T27 回流后启动）
**作者**：Agent（与用户共同推演）
**状态**：🚧 Planned · 代码已就位（`enable_din_interest` bypass 版，P2 smoke 2026-05-09 早通过）· 待 T27 Go/No-go 决定 run.sh 版本（A / B）
**关联**：T25（tasks/active.md）· EXP-031（T27 判决） · EXP-029（EDA 识别架构级创新 ROI）· EXP-030（gap=0.05 funnel baseline）

---

## TL;DR

**一次 training quota，回答一个问题**：DIN-style target-aware interest pooling 是否能**突破 SOTA 能力均值 0.8424**（即 ΔDev ≥ +0.0030 vs `icq-half-gap05` baseline 0.842443）？

**判决逻辑**（利用 EXP-030 铁律 ②：gap=0.05 配置下 dev→LB gap = +0.0032 极可信 + EXP-032 铁律 ①：同 config rerun 噪声 ±0.0001 下界）：

| 判决条件（E3 Dev AUC） | 判决 | 动作 |
|---|---|---|
| ≥ 0.847（ΔDev ≥ +0.0046）| 🎯 **强信号** | 立即 evaluation，预期 LB ≈ 0.844（真正突破方法学均值 0.8424） |
| [0.8444, 0.847)（ΔDev ∈ [+0.002, +0.0046)）| 🟡 疑似正向 | 多 seed rerun 确认 |
| [0.8412, 0.8444)（ΔDev ∈ [−0.0012, +0.002)）| ⚪ 持平/噪声内 | 方向 Deprecated，归档，考虑 T28 block-integrated |
| < 0.8412（ΔDev < −0.0012）| ❌ 负向 | 方向关闭 + 本地排查（不 rollback） |

---

## 为什么 T25 是当前最高 ROI 候选

### 外部证据三重交叉

1. **DECEM 榜一（LB 0.852181）明说 "DIN + MLP 是 work 的"**（2026-05-08 小红书情报），已入 team-memory。
2. **UniRec（hojiahao/TAAC2026 41 stars）方案 1** 用 `DIN target attention` 生成 `interest_token`，与本仓库 bypass 版同构。
3. **DIN (Alibaba KDD 2018) 论文**：离线 AUC +0.5~1%（BCE baseline 下）。

### 内部 EDA 对口

EXP-029 结论 ②：模型在"信息丰富样本"上 AUC 低 0.04~0.05（稀疏 user AUC 0.89 vs 丰富 0.85）→ **"id 特征线下虚高线上掉" 机制真实存在**。DIN 的 target-aware attention 恰好对症——**让模型学会在 seq 里选择性加权与 target 相关的位置**，而不是无差别 mean-pool。

### 与已证伪方向的机制差异

| 方向 | 机制 | 结果 |
|---|---|---|
| T16 additive（已证伪）| `seq_token += α · W(target_repr)`（soft, 无差别加） | E3 full-train proxy 负向, Deprecated |
| **T25 DIN bypass** | `interest_i = CrossAttn(target_repr, seq_i)` per domain（hard, 选择性聚焦） | **机制完全不同 — "soft add" vs "selective attention"** |

T16 失败不代表 T25 失败。两者在位置（seq token-side vs bypass-side）+ 机制（additive vs attention pooling）都不同。

---

## Baseline 锚点（EXP-030）

| 指标 | `icq-half-gap05`（EXP-030 baseline） |
|---|---|
| Dev AUC @ E1 | 0.837939 |
| Dev AUC @ E2 | 0.841124 |
| Dev AUC @ E3 best | **0.842443** |
| LB | **0.839223** |
| Dev→LB gap | **+0.0032**（极可信） |
| 参数量 | Dense ~2.88M / Sparse ~350.4M |

### 真实能力均值（EXP-028 / EXP-030 共识）

- 5-seed 均值 LB ≈ **0.8424**
- Seed 噪声 ±0.005（跨度 0.010）
- SOTA 0.84637 = seed 上沿，非方法学峰值
- **"破 SOTA" 新阈值 = 方法学使均值 +0.005，即 ≥0.8474**

---

## 配置 · 两个候选 `run.sh`

### 方案 A（推荐）· T25 + T27 叠加（compile+AMP）

**前提**：EXP-031 T27 Go（E3 Dev ∈ [0.8414, 0.8434] AND wall-time ≤ 0.67×）。

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t25-din-bypass-halftrain-gap05
#
# T25 · DIN-style target-aware interest pooling first half-train.
#
# Config = `icq-half-gap05` (EXP-030 baseline, Dev E3 0.842443) +
# two flags: `--enable_din_interest` (T25, the only methodological
# variable) and compile+AMP (T27 工程加速, transparent if EXP-031 Go).
#
# Go/No-go (E3 Dev AUC):
#   🎯 >= 0.847   -> strong signal, evaluate immediately
#   🟡 [0.8444, 0.847) -> multi-seed rerun
#   ⚪ [0.8412, 0.8444) -> deprecated, consider T28 block-integrated
#   ❌ < 0.8412   -> closed
#
# Seed noise floor: +-0.0001 (EXP-032) — so |Dev change| < 0.0003
# is literally noise. Threshold 0.0030 is 10x noise = real signal.
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
    --enable_din_interest \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 3 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
```

### 方案 B（fallback）· 纯 T25（无 compile+AMP）

**触发条件**：EXP-031 T27 No-go（dev 精度下降 >0.001 或 compile crash）。

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t25-din-bypass-halftrain-gap05-noamp
#
# T25 fallback path — runs DIN in fp32 only because EXP-031 T27
# compile+AMP failed validation. Expect wall-time ~5h instead of
# ~3h; everything else identical to Plan A.
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
    --enable_din_interest \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 3 \
    "$@"
```

---

## 平台提交清单

**必须 Create New Training Job**（baseline code 有改动 vs 已运行的 job，不能 Copy）。

**Job Name**：`t25-din-bypass-halftrain-gap05`（方案 A）/ `t25-din-bypass-halftrain-gap05-noamp`（方案 B）

**上传文件列表**（方案 A / B 都一样）：
- `baseline/trainer.py`（T27 bug 修复版 · commit 38d6ea2）
- `baseline/model.py`（`DINInterestExtractor` + `enable_din_interest` 路径 · 已在主线）
- `baseline/train.py`（`--enable_din_interest` CLI · 已在主线）
- `baseline/dataset.py`（无改动 · 保守起见一起上传）
- `baseline/infer.py`（读取 `enable_din_interest` 的版本 · 已在主线）
- `baseline/utils.py`（无改动）

**run.sh 贴在平台编辑框**：上面的方案 A 或 B（根据 T27 判决）。

**⚠️ 首次 evaluation 必须勾选"上传自定义 infer.py"**：按 team-memory 铁律"新增 kwarg 后平台兼容性"，`enable_din_interest` 是首次进入平台 evaluation 路径的 flag，**不赌**平台默认 infer.py 识别。

---

## 观测指标

### 必看（决定 Go/No-go）

| 指标 | 测量方式 | 判决阈值 |
|---|---|---|
| E3 Dev AUC | 训练日志 `[summary] val_auc_history` | 见 TL;DR 判决表 |
| 每 epoch wall-time | 日志首尾时间戳 | 方案 A 目标 ≤ 67 min/epoch（与 T27 一致）；方案 B ~100 min/epoch |
| 稳定性 | 日志扫 `dynamo` / `Traceback` / `OOM` / `logloss-guard` | guard 触发属正常（不阻塞），crash 类 0 hit |

### 辅助诊断

| 指标 | 期望 | 异常含义 |
|---|---|---|
| Dense params 增量 | **+104,896**（与本地 smoke 一致） | 若差异大 → DIN extractor 实例化异常 |
| E1 Dev AUC | 0.83 ~ 0.84 范围 | 若 <0.80 → DIN 前期对模型干扰严重（可能需 α warmup 设计） |
| E1 → E3 Dev 单调 | 单调升 | 若 E2/E3 反弹 → DIN 让模型提前过拟合 |

---

## 假设与理论预测

### Dev AUC 预期

- **乐观情况**（DECEM + UniRec 直接复现）：E3 Dev **+0.004~+0.008**，即 **0.846~0.850**
- **保守情况**（local AUC → LB 转化率 30~40%）：E3 Dev **+0.002~+0.004**，即 **0.844~0.847**
- **失败情况**（bypass 相对 block-integrated 弱化 50%）：E3 Dev +0.001 或持平

### 与 T16 additive 的对照机制

T16 additive α=0.1 在 full-train E3 0.868061 → E4 触发 guard（E4 Dev 下跌 + LogLoss 反弹），最后 full-train 负向。机制：**uniform target injection 放大了"train 尾部局部模式"的记忆**。

T25 DIN 的差异：**selective attention 让模型只加权与 target 相关的 seq 位置**，对不相关位置 attention weight ≈ 0 → 不增加"无差别记忆压力"。理论上**覆盖同等目标信息的同时不放大过拟合**。

这是 T25 最核心的 risk：如果实测 DIN 也让 E3→E4 出现 LogLoss 反弹，说明"bypass 架构 + CrossAttn" 在本数据量下仍然过拟合 → T28 block-integrated 也会失败（机制同源）。

---

## Go/No-go 决策树

```
T25 half-train 完成后:

├── E3 Dev >= 0.847（ΔDev >= +0.0046, 10× seed noise）
│   └── 🎯 Strong signal
│       └── 动作: 立即 evaluation (eval quota 1/4)
│       └── 预期 LB: 0.842 + 0.005 = 0.847+（真正破方法学均值 0.8424）
│       └── 若 LB ≥ 0.849: DIN Accept, update SOTA
│       └── 若 LB ∈ [0.845, 0.849): seed luck 带内, multi-seed 确认
│       └── 若 LB < 0.845: dev→LB 异常背离, EXP-024 复盘模式
│
├── E3 Dev ∈ [0.8444, 0.847)（ΔDev ∈ [+0.002, +0.0046)）
│   └── 🟡 疑似正向
│       └── 动作: 先跑 seed=3 rerun (+1 training quota, 0 eval quota)
│       └── 若 seed=3 也 +0.002+: Accept, 提 evaluation
│       └── 若 seed=3 持平: 噪声, 归档
│
├── E3 Dev ∈ [0.8412, 0.8444)（ΔDev ∈ [−0.0012, +0.002)）
│   └── ⚪ 持平或噪声内
│       └── 动作: DIN bypass Deprecated
│       └── 考虑 T28 block-integrated（把 DIN 放进 HyFormer block 内）
│       └── 但优先级降低（bypass 都不 work, block-integrated 不一定救）
│
└── E3 Dev < 0.8412（ΔDev < −0.0012）
    └── ❌ 负向
        └── 动作: T25 整体关闭
        └── 本地排查: DIN 初始化是否偏, cross-attn 是否触发 NaN
        └── 不 rollback 代码（`enable_din_interest=False` 默认即关闭）
```

---

## 风险清单

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| DIN CrossAttn 在 bf16 下 softmax 精度不够 | 15%（方案 A 特有） | Dev −0.001~−0.003 | 方案 B fallback；未来给 softmax 加 fp32 cast |
| DIN + compile 触发 dynamo recompile | 20%（方案 A） | wall-time >0.8× baseline | `suppress_errors=True` 已开，降级 AMP-only |
| DIN 早期训练不稳定（Dense params +3.6% 冷启动 warmup 慢）| 10% | E1 Dev 略低 0.835 左右 | 3 epoch 足够收敛，E3 观察即可判决 |
| 平台 default infer.py 不识别 `enable_din_interest` kwarg | **90%**（新 kwarg）| evaluation 提交失败 | **首次 evaluation 必须勾选"上传自定义 infer.py"** |
| 参数量增长让显存超 19GB | <5%（Dense 只加 0.1M） | OOM | 本地 smoke 已验证参数量正常 |

---

## 本地 smoke 验证（已通过 · 2026-05-09 早）

完成 4 项 smoke：

1. ✅ `py_compile` 6 文件 exit 0（含 bug 修复后的 trainer.py）
2. ✅ Train 5 steps：loss 0.62 → 0.32 / AUC 0.722 / best_model 落盘 / `training_summary.json` 正常写出
3. ✅ `train_config.json` 正确记录 `enable_din_interest: True` + 其他 11 项关键 flag
4. ✅ Infer 端到端：1000 predictions from 2000 rows / mean=0.1521 / 无 NaN / cfg 读 `enable_din_interest=True` 并构建模型成功

**参数量对比**（demo 数据）：

| Config | Dense params | Dense tensors | Sparse params | Δ Dense |
|---|---|---|---|---|
| 无 DIN | 2,881,281 | 340 | 350,453,696 | baseline |
| 有 DIN | 2,986,177 | 400 | 350,453,696 | **+104,896（+3.6%）** |

差额完全符合理论估算（4 × CrossAttention + 1 × Linear merge ≈ 100k）。

---

## 后续动作

**若 Accept**（LB ≥ 0.849 或 seed confirmed）：
1. Merge 到主线 run.sh（作为新默认）
2. 考虑 T28 block-integrated 作为"DIN 2.0"升级（更深融合）
3. 下一个架构级 bet：T19 Per-token FFN（单独）

**若 Deprecated / Failed**：
1. `enable_din_interest=False` 仍是默认，代码不 rollback（保留 bypass 作 "possible + block-integrated" 的 fallback 代码）
2. 下一个架构级 bet：T19 Per-token FFN（单独，不叠加 DIN）

---

## 备注 · 为什么不用 full-train / proxy

- **Proxy**（1500 steps）：按 team-memory 铁律"含大型 embedding / 新参数分支的变量 proxy 假负向"，DIN extractor +100k Dense 是新参数分支，proxy 不可靠 → 跳过。
- **Full-train**（10 epoch）：按 EXP-019 教训"half-train E4 LB 可以 > full-train best LB"，当前数据量下 full-train 比 half-train 多的 4~6 小时**没有信息净增量**。half-train 3 epoch + gap=0.05 + E3 dev 判决足够。
