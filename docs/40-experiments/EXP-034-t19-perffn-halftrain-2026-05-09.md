# EXP-034 · T19 RankMixer Per-token FFN 首次 half-train（`t19-perffn-halftrain-gap05-noamp`）

**日期**：2026-05-09（西班牙时间早 · 与 T27 / T25 B 并发提交）
**作者**：Agent（与用户共同推演）
**状态**：🚧 Planned · 代码已在主线（`rank_mixer_ffn_mode=per_token` · ADR-007）· 本地 smoke 2026-05-09 通过
**关联**：T19（tasks/active.md）· ADR-007（RankMixer per-token FFN）· EXP-023（历史 smoke）· 2026-05-04 T19 与 T18 捆绑归档后**从未单独验证**

---

## TL;DR

**一次 training quota，回答一个问题**：把 `RankMixerBlock` 的 shared FFN（仓库现实）换成 per-token FFN（RankMixer 原论文 §3.2），能否**单独**推进 dev AUC？

**判决阈值**（同 EXP-033 · E3 Dev AUC vs `icq-half-gap05` baseline 0.842443）：

| 判决条件（E3 Dev AUC） | 判决 | 动作 |
|---|---|---|
| ≥ 0.847（ΔDev ≥ +0.0046）| 🎯 **强信号** | 立即 evaluation，预期 LB ≈ 0.844（可能破均值 0.8424） |
| [0.8444, 0.847)（ΔDev ∈ [+0.002, +0.0046)）| 🟡 疑似正向 | seed=3 rerun 确认 |
| [0.8412, 0.8444)（ΔDev ∈ [−0.0012, +0.002)）| ⚪ 持平/噪声内 | 方向 Deprecated |
| < 0.8412（ΔDev < −0.0012）| ❌ 负向 | 方向关闭 |

---

## 为什么 T19 必须单独验证（2026-05-04 历史债）

EXP-023（smoke only）后 T19 被并入 **"四合一 half-train"**（T16+T18+T19+holdout），但因 T18 BPR 整体崩盘（LB −0.010，ADR-006 Deprecated），**T19 的独立收益从未被验证**。active.md T19 节点 2026-05-04 归档时明确记："future 若重启 T19，必须**单独**（不叠加 T18）跑 full train 验证"。

**本 EXP 就是兑现这条 TODO**。

## 为什么 T19 ROI 不低

### 外部证据

1. **UniRec `hojiahao/TAAC2026`** 方案 1 深调研结论（community-intel.md §同赛题开源方案）明确列出：
    > "RankMixer Per-token FFN 正确实现 —— 我们的 RankMixerBlock 用 shared FFN，与论文原版差异。**Per-token FFN 专门对抗 dominant feature 淹没长尾，本赛题 item 长尾极端**。"
2. **model-py-section-map.md §RankMixerBlock** 行内注释（2026-05-06 飞机伴读整理）明确标注："**仓库 vs 论文：shared FFN 可能是 +0.002~0.005 AUC 子弹**"
3. **RankMixer 论文 (arXiv 2507.15551 §3.2)**：per-token FFN 的设计动机是 "feature subspace heterogeneity, allocating independent parameters rather than allowing domination by a single shared model"

### 与当前架构的匹配度

| 本赛题特性 | Per-token FFN 动机契合度 |
|---|---|
| 4 个行为域 + 60 列非序列特征 = T=16 tokens 异质性高 | 🟢 极高 — 论文原始动机 |
| `domain_c_seq_47` vocab=2.78 亿（极端长尾）| 🟢 极高 — anti-dominant feature |
| 数据量 1.9M（相对容量紧） | 🟡 中 — 参数量 +34% 需看是否过拟合 |

### 为什么现在做正合适

- **EXP-030 铁律**：架构级创新是唯一破局路径（small ablation / hash 调整全部关闭）
- **EXP-032 铁律 1**：rerun 噪声 ±0.0001 下界建立，T19 能不能 work 只要 ΔDev ≥ 0.0003 就判断得出
- T27（compile+AMP）并发中，若 T27 Go，未来所有 T19 叠加实验享受 1.5× 加速

---

## Baseline 锚点（EXP-030）

| 指标 | `icq-half-gap05`（EXP-030 baseline · shared FFN）|
|---|---|
| Dev AUC @ E1 | 0.837939 |
| Dev AUC @ E2 | 0.841124 |
| Dev AUC @ E3 best | **0.842443** |
| LB | **0.839223** |
| Dev→LB gap | **+0.0032**（极可信） |
| 参数量 | Dense ~2.88M / Sparse ~350.4M |

---

## 配置 · `run.sh`

**相对 `icq-half-gap05` 只加一个 flag**：`--rank_mixer_ffn_mode per_token`。**未叠加 T25 DIN**（严格单变量纪律 · EXP-024 反思铁律 / EXP-032 新战术铁律）。**未叠加 T27 compile+AMP**（因 T27 尚未 Go/No-go）。

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════
# Job: t19-perffn-halftrain-gap05-noamp
#
# T19 · RankMixer per-token FFN (paper-faithful revision).
#
# Rationale: repository currently uses shared FFN across all T=16
# tokens inside RankMixerBlock. The original RankMixer paper
# (arXiv 2507.15551 §3.2) uses per-token FFN to avoid "domination
# by a single shared model". Our data has 4 behavior domains + 60
# heterogeneous NS features + extreme-tail `seq_c/47` vocab=278M
# → per-token FFN's anti-domination motive matches exactly.
#
# Why alone (not stacked with T25 DIN or T27 compile+AMP):
#   - 2026-05-04 T19 was bundled with T18 BPR; T18 crashed LB,
#     T19 got orphaned without a clean signal. This EXP pays off
#     that debt with a strict single-variable ablation.
#   - EXP-024 + EXP-032 new tactical law: combinatorial experiments
#     break attribution. 1 variable per job.
#
# Dense params expected: 2.88M -> 3.87M (+34%, from T=16 × 64 × 256
# × 2 = 524k × 2 blocks = ~1M new dense params for per-token FFN).
# Sparse params unchanged.
#
# Go/No-go (E3 Dev AUC vs baseline 0.842443):
#   🎯 >= 0.847  -> strong signal, evaluate immediately
#   🟡 [0.8444, 0.847) -> seed=3 rerun to confirm
#   ⚪ [0.8412, 0.8444) -> deprecated, close direction
#   ❌ < 0.8412  -> closed, rollback not needed (default=shared)
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
    --rank_mixer_ffn_mode per_token \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 3 \
    "$@"
```

---

## 平台提交清单

**操作**：Copy T27 job（baseline code 相同，`commit 38d6ea2` 或更新 `a8e09ac`，docs-only 改动不影响 training path） → 改 Job Name + 改 run.sh。

**Job Name**：`t19-perffn-halftrain-gap05-noamp`

**无需上传新 code 文件**（Copy 已带最新 baseline）。

**⚠️ 并发 OOM 风险**：Dense +34% 是本批最大显存新增（T25 只 +3.6%、T27 用 bf16 理论降 30%）。三 job 并发时若被平台调度到同卡：
- T27 ~10GB + T25 B ~14GB + T19 ~17GB = **41GB**（单卡 19GB 绝对 OOM）
- 平台大概率会分散到多卡，但要监控

**OOM 应急**：立即改 `--batch_size 96` 重提为 `t19-perffn-halftrain-gap05-noamp-bs96`（bs 96/128 × dense 1.34 ≈ 13GB 估算）。

---

## 观测指标

### 必看（决定 Go/No-go）

| 指标 | 测量方式 | 通过/失败 |
|---|---|---|
| E3 Dev AUC | `[summary] val_auc_history` | 见 TL;DR 判决表 |
| Dense params 实测 | Trainer 启动日志 "Dense params: N tensors, M params" | **M ≈ 3,873,921（±10%）**；偏差大说明 per-token 分支未生效 |
| E1 → E3 单调 | summary | 单调升 = healthy；E2/E3 反弹 = 过拟合（+34% 参数容量太大的信号） |
| 稳定性 | 扫 `Traceback` / `OOM` / `logloss-guard` | guard 触发属正常（不阻塞）；crash 0 hit |

### 辅助诊断

| 指标 | 期望 | 异常含义 |
|---|---|---|
| E1 Dev AUC | ~0.833~0.838 | 若 <0.82：+34% dense 冷启动慢（可能需更多 epoch） |
| Wall-time / epoch | ~100~120 min（dense +34% 带来 ~10~20% 减速） | >150 min → per-token einsum 实现低效 |
| Peak GPU mem | 14~17GB（19GB 卡的 75~90%） | >18GB → batch_size 降 96 |
| LogLoss rebound | E3 或更早触发（如果过拟合） | ✅ guard 会自动停机 E3 best 保留 |

---

## 假设与理论预测

### Dev AUC 预期

- **乐观**（UniRec / 论文数字 30% 折算后）：E3 Dev **+0.002~+0.004**，即 **0.844~0.846**
- **中性**：E3 Dev **+0.0005~+0.002**，即 **0.843~0.844**（shared FFN 在 1.9M 数据下已近饱和）
- **负向**（+34% dense 过拟合 > 架构收益）：E3 Dev **持平或 −0.001**

### T19 和 T25 DIN 的机制差异

| | T25 DIN bypass | T19 Per-token FFN |
|---|---|---|
| 位置 | classifier 前 bypass 拼接 | `RankMixerBlock` 内部 FFN |
| 对 target 的关系 | **Target-aware**（显式 target attention） | **Target-agnostic**（纯表征容量增加） |
| 参数量增量 | Dense +3.6%（很小） | Dense +34%（显著） |
| 过拟合风险 | 低（CrossAttn 有 mask，不加"无差别记忆"） | 中（容量加大 → 过拟合风险线性增长）|
| 与 EXP-029 EDA 的匹配 | 高（解决 "信息丰富样本 AUC 反而低" 机制） | 中（解决异质特征 domination，但我们 EDA 没直接证据） |

**正交性**：两者位置完全不同 → 理论可叠加。**但叠加实验留到各自单独验证过之后**（EXP-032 战术铁律）。

---

## Go/No-go 决策树

```
T19 half-train 完成后:

├── E3 Dev >= 0.847 (ΔDev >= +0.0046, 10× rerun noise)
│   └── 🎯 Strong signal
│       └── Evaluation (1 eval quota)
│       └── If LB ≥ 0.847 → T19 Accept, 叠加 T25（若 T25 也 work）
│       └── If LB ∈ [0.844, 0.847) → seed=3 rerun on T19 only 确认
│       └── If LB < 0.844 → dev→LB 异常背离，不叠加
│
├── E3 Dev ∈ [0.8444, 0.847) (ΔDev ∈ [+0.002, +0.0046))
│   └── 🟡 疑似正向
│       └── seed=3 rerun (+1 training quota)
│       └── If 两个 seed 都 +0.002+ → Accept + evaluate
│       └── If seed=3 持平 → 归档，不追
│
├── E3 Dev ∈ [0.8412, 0.8444) (ΔDev ∈ [−0.0012, +0.002))
│   └── ⚪ 持平 / 噪声内
│       └── T19 Deprecated（paper-faithful 在本数据量下收益饱和）
│       └── 仓库回归 shared FFN 默认
│       └── 下一批 quota 给 T28 DIN block-integrated 或其他架构级 bet
│
└── E3 Dev < 0.8412 (ΔDev < −0.0012)
    └── ❌ 负向（容量过拟合 > 结构收益）
        └── T19 Close，shared FFN 保持默认
        └── 本地排查：is per-token FFN init 正确？（应该每 token Kaiming uniform）
```

---

## 风险清单

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| +34% Dense 参数在 1.9M 数据下过拟合 | 35% | E3 Dev 持平或负 | guard 自动早停 E3 best ckpt 保留 |
| 同卡 OOM（与 T27/T25 B 并发）| 20% | crash ≤5 min | 重提 bs=96 |
| per-token `einsum('btd,tdh->bth')` 在本地 smoke 过但平台 GPU 上不同 | <5% | 崩 / 慢 | 本地 CPU smoke 已测参数量正确 (+993k)；平台 CUDA 分支等同逻辑 |
| 平台 default infer.py 不识别 `rank_mixer_ffn_mode` kwarg | **90%** | evaluation 失败 | 首次 evaluation 必须勾选"上传自定义 infer.py" |
| 首次 LB 后 dev→LB 发散（local AUC 涨但 LB 跌）| 25% | 本方向 ROI 下修 | 按 EXP-024 纪律：单点背离不判死刑,看 seed=3 复现 |

---

## 本地 smoke 验证（已通过 · 2026-05-09 早 06:08）

用 `demo/demo_1000.parquet + CPU + limit_train_rgs=8 + max_train_steps=5`：

- ✅ `py_compile` 6 文件 exit 0（无回归）
- ✅ Train 5 steps：AUC 0.7489，无 crash
- ✅ Dense params **3,873,921** （vs shared 2,881,281 baseline · **+992,640 = +34%**），与理论预测 `T=16 × (64×256 + 256×64) = 1,048,576` 吻合（差 5% 来自 bias term 合并）
- ✅ Sparse params 不变（350,453,696）
- ✅ `training_summary.json` 正常写出

**→ per-token FFN 分支确实在跑（不是 shared 的别名）**，ADR-007 代码落地正确。

---

## 后续动作

**若 T19 Accept（LB ≥ 0.847 seed confirmed）**：
1. T25 也 work 的话 → 起 T25 + T19 叠加 half-train（5/11 之前）
2. T25 不 work 的话 → T19 solo 冲 seed sweep 到 0.85
3. ADR-007 → `Accepted & Deployed`（已经是 Accepted 但 task 关了，重启）
4. team-memory 加一条"per-token FFN 在本赛题 work"

**若 T19 Deprecated / Failed**：
1. 代码不 rollback（`rank_mixer_ffn_mode=shared` 仍是默认）
2. 下个 bet：T28 DIN block-integrated 或 T5 OneTrans v3 gap=0.05 重估
3. team-memory 加一条"per-token FFN 在 1.9M 数据下容量饱和"

---

## 备注 · 为什么不叠加 T25 DIN

虽然两者机制正交（T25 target-aware / T19 capacity expansion），按 **EXP-024 方法论反思 + EXP-032 战术新铁律**：

> 任何组合改动（A+B）full/half-train 提交前，**必须有 A-only / B-only 的 single-variable 数据**

今天并发起 T25 B + T19 是**双轨单变量 ablation**，不是组合实验。如果任一 work，明天再做叠加实验。**今天不抄近路**。

---

## 与 ADR-007 的关系

ADR-007（2026-05-04 Accepted）原计划是 T19 和 T18 叠加跑，但 T18 Deprecated 后 T19 task 也关闭了。**本 EXP 重启 T19，首次单独验证**，ADR-007 保持 Accepted 状态；若本 EXP 判决 T19 Deprecated，ADR-007 在最后追加一节 "§Deprecation Note after EXP-034"。
