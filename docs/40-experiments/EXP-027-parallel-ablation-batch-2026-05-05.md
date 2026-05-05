# EXP-027 · 2026-05-05 四并行 Ablation 批次综合诊断

**日期**：2026-05-05
**作者**：Agent
**状态**：✅ 完成（4/4 job 回流 · 1/4 evaluate · LB SOTA 0.84637 **未破**）

---

## TL;DR

- **4 个 parallel job 全部完成训练**，配合本地 P1/P3 代码落地首次实战
- **1 次 evaluation quota 用于 idmaskc15 E3 ckpt → LB 0.842176** vs SOTA 0.84637 **Δ −0.004194**
- **LB 红线仍为 0.84637（icq-half E4）**，本批次**全部未破**
- 三条方向判决：T16 additive α=0.1 **负向**、T18+T19 **负向** 、P1 id_mask c=15% **证伪**
- 真正有价值的 insight 是 **T22 P3 gap15 的失效诊断**（发现 RG-level 切分无法真正隔离时间窗）

---

## 四个 Job 回流汇总

| Job | 核心变量 | Best Dev AUC @ E | Best Dev LogLoss | Holdout AUC | dev-holdout gap | Early stop | LB | vs SOTA |
|---|---|---|---|---|---|---|---|---|
| 四合一 | T16+T18+T19+holdout, patience=2 | 0.866133 @ **E4** | 0.22503 | 0.867182 @ E4 | −0.00105 | 跑满 E4 | — | （未 eval）|
| T16-only full | T16+holdout, patience=1, 10e | 0.865122 @ **E2** | 0.22524 | 0.866096 @ E2 | −0.00097 | **E3 logloss +0.0001 反弹 → guard 停** | — | （未 eval）|
| **idmaskc15** | icq + id_mask c=15%, patience=2 | **0.868288 @ E3** | **0.22382** | — | — | 跑满 E4 (E4 略降) | **0.842176** | **−0.004194** |
| gap15 | icq + valid_gap=15%, patience=2 | 0.867124 @ E4 | 0.22506 | — | — | 跑满 E4 | — | （未 eval）|
| **Baseline icq-half (EXP-019)** | 原版 | 0.868619 @ E4 | 0.22336 | — | — | — | **0.84637** | SOTA |

---

## 三条方向的独立判决

### ① 四合一（T16+T18+T19+holdout）— Dev 拉胯 0.0025

E4 dev 0.866133 vs icq-half E4 0.868619 = **Δ −0.002486**。holdout 0.867182 vs dev 0.866133 = 0.001 gap（正常）。

**证据**：即使 T18/T19 已在 EXP-022/EXP-023 关闭，与 T16 叠加后**进一步拉胯 dev** 约 0.0025。T16 additive + T18 BPR + T19 per-token FFN 相互间可能有参数干扰，不是简单正交叠加。

**判决**：已有 task 关闭决定（active.md T18/T19 关闭），本轮数据再次验证。

### ② T16-only full （10 epoch）— E3 过早触发 guard

训练曲线：
```
E1 AUC 0.861209 / LogLoss 0.228373
E2 AUC 0.865122 / LogLoss 0.225238  ← best_model（AUC 最高）
E3 AUC 0.865036 / LogLoss 0.225321  ← LogLoss +0.000083 反弹 → guard 停
```

**证据**：
- E2 best 0.865122 **比 icq-half E4 的 0.868619 低 0.003497**
- E3 就触发 logloss rebound（patience=1 严格）
- 不是 icq-full 那种"E5~E6 才过训练"的模式，T16 additive 在 full 尺度上**更快进入过拟合**
- 预估 holdout→LB：0.866096 − 0.022 ≈ **0.844**，**低于 SOTA 0.00237**

**判决**：T16 additive (α=0.1) 在 full train 下**已证伪**。target_item 硬加性注入可能和 sparse embedding 训练节奏冲突；或 α=0.1 的 gate 强度不够"稳定"让模型走 target-aware 路径反而造成 over-regularization。ADR-004 方案 A 单独路线关闭，**除非换成 concatenation 或 gated-sum 模式（方案 B/C）重验**。

### ③ idmaskc15（P1 id_mask c=15%）— dev 微升 / LB 大跌

Dev E3 0.868288 vs icq-half E3 0.867965 = **Δ +0.000323**（唯一正向信号）
LB 0.842176 vs SOTA 0.84637 = **Δ −0.004194**

**这是最具诊断价值的数据**（见 EXP-025 详细机制分析）：

- **Dev 和 LB 方向相反**（dev +0.0003 / LB −0.0042）
- dev→LB gap 从 icq-half 的 +0.0222 **放大到 +0.0261**（放大 2× 幅度！）
- 15% token mask **反而让 dev→LB gap 扩大**

**机制解读**：Input-level random mask 让 **85% token 仍可见**，模型仍依赖 id embedding；dev 近 train → id 分布共享 → dev 还能涨一点点；test 远 train → id OOV 率高 → 85% 的训练路径失效。

**判决**：P1 input-level random mask 在"远未来 test"场景下**机制上无效**。不是强度问题（降到 0.10 也救不了），是方法本身不对口。**T21 关闭**。

### ④ gap15（P3 valid_gap_ratio=15%）— Funnel 未真正起效

见 EXP-026 诊断。关键数据：

```
train ts: [1772238909, 1772588918]   (跨 4.04 天)
gap   ts: [1772247536, 1772588958]   (跨 3.95 天)  ← 与 train 严重重叠
valid ts: [1772247872, 1772588990]   (跨 3.95 天)  ← 与 train 严重重叠
effective train→valid distance = -341046 sec ⚠️
```

**根因**：`valid_split_strategy='time'` 按 RG 的 `max(timestamp)` 排序，但**每个 RG 内部时间跨 4 天**，排序后尾部 RG 的 `min(ts)` 还是 2/28。RG-level gap 无法在当前数据上形成真实时间隔离。

**证据**：dev E4 Δ −0.001495 完全匹配"减 15% train data"的纯预期（无任务变难加成）。

**判决**：P3 RG-level 方案失效，需改为 **row-level gap cutoff**（T22-fix，~50 行新代码）。但优先级排在 T23 之后，等后续再验证。

---

## 整体战略影响

### LB 红线：0.84637（未破）

本轮 4 并行 + 1 evaluation quota 消耗，**LB 无进展**。距离社区 SOTA 0.858 仍差 0.016。距初赛 ddl 约 20 天。

### 剩余已被排除的方向

基于前期 + 本轮：
- ❌ Proxy 上缩时间 feature 工程（P0-A/B/C 饱和）
- ❌ Loss 层 ranking（T18 BPR）
- ❌ 架构容量（T19 per-token FFN）
- ❌ Target-item hard injection α=0.1（T16 additive）
- ❌ Input-level random mask（T21 P1）
- ⚠️ Valid gap RG-level（T22 P3）
- ⚠️ Holdout 独立成本（已知约 −0.002~−0.005，T17 方案 B 暂缓）

### 剩余可能的方向（按 ROI 排序）

| 方向 | 代码量 | 预期 LB 贡献 | 备注 |
|---|---|---|---|
| **T23 seq_c feat47 disable** | ✅ 已就位 (~30 行) | ±0.003 | Seed 主楼问题-5 直接对应 |
| T22-fix row-level gap | ~50 行 | 修 funnel, 让后续实验更可信 | 不直接贡献 LB |
| T16 方案 B/C（concat/gated-sum） | ~80 行 | ±0.002 | α=0.1 additive 已负向，换模式重验 |
| DIN target attention（P2） | ~80 行 | ±0.002 | T16 additive 已负向，DIN 是更激进同方向，风险高 |
| UE 分离 + int pairwise | ~150 行 | ±0.003 | Seed 主楼方案-1，依赖猜哪些 FID 是 UE |

---

## 下一步（LB 冲榜优先级）

1. **T23 `disablefid47-icq-halftrain`** — 直接对口 Seed 问题-5，代码已就位，等明日 training quota 重置启动
2. 如果 T23 LB ≥ 0.846（持平或微升）→ 说明 "disable 路线对" → 下轮扩大 disable 范围（fid=47+29）、或 "disable + T16 方案 B"组合
3. 如果 T23 LB < 0.844 → disable 也无效 → 切换到 **T22-fix（修 funnel）+ 重新跑 icq-half 重建基线**（本周内优先级升）
4. 如果 T23 LB ∈ [0.844, 0.846] → 继续做强度扫描（fid=47,29 / fid=47,36 等）

### Evaluation quota 策略

今日已用 1/4，窗口至 PDT 21:00。剩 3 slot 暂不消耗，等 T23 回流数据有方向信号再 eval。

---

## 对 EXP-024 方法论复盘的验证

EXP-024 定义的"dev AUC 是错误的指挥棒"铁律，本轮**再次验证**：

- idmaskc15 E3 dev +0.000323 → LB −0.004194（**反向 14×**）
- 四合一 dev −0.002486 → (未 eval，holdout 估计 LB ≈ 0.845, 也是负向)

**dev AUC 做决策的可靠性进一步下降**。下一步实验应更依赖 **LB direct evaluation**（Evaluation quota 作为主要校准信号）而非 dev AUC 相对 SOTA 的微小变化。

### Eval quota 积累映射表

| Config | Dev AUC | Holdout AUC | LB AUC | dev→LB gap | holdout→LB gap |
|---|---|---|---|---|---|
| baseline swiglu (filename) | 0.8650 | — | 0.84381 | +0.0212 | — |
| transformer (filename, 错 ckpt) | 0.8641 | — | 0.834015 | +0.0301 | — |
| swiglu + d_model=128 | 0.864881 | — | 0.836944 | +0.0279 | — |
| swiglu + time + vr=0.05 | 0.865884 | — | 0.844059 | +0.0218 | — |
| swiglu + dropout=0.05 | 0.864563 | — | 0.83977 | +0.0248 | — |
| t13tf-full E4 | 0.868495 | — | 0.844956 | +0.0235 | — |
| t13tf-loss-full E9 | 0.868989 | — | 0.838707 | +0.0303 | — |
| icq-full E6 | 0.869138 | — | 0.843599 | +0.0255 | — |
| **icq-half E4** | **0.868619** | — | **0.84637** (SOTA) | **+0.0222** | — |
| 三合一 E4 (T16+T18+ho) | 0.865516 | 0.867196 | 0.836445 | +0.0291 | +0.0308 |
| **idmaskc15 E3** | **0.868288** | — | **0.842176** | **+0.0261** | — |

**gap 稳定区间：[+0.0212, +0.0303]**，均值约 +0.025，**idmaskc15 的 +0.0261 落在 gap 上档**。再次确认 EXP-024 的"gap 系统性，非单点事故"诊断。
