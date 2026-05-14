# EXP-051 · M18/M19/M20 LB 三连阴 + gap=0.05 ceiling 战略反思 · 2026-05-14

> **Status**: 🔬 **战略转折点文档** · CEST 11:48 起草 · 距 ddl 11 天 · LB SOTA 0.84671 (M9 gap=0) · gap to Top 20 = 0.00329
> **日期**: 2026-05-14 CEST 11:48 (用户主动触发"10 年广告算法研究员视角深度反思"后落档)
> **关联**: [EXP-045 T33 NLIR](EXP-045-t33-nlir-halftrain-2026-05-13.md) · [EXP-044 T31 is-missing](EXP-044-t31-is-missing-scheme-d-2026-05-13.md) · [EXP-048 EDA fullscan](EXP-048-eda-fullscan-2026-05-13.md) · [EXP-049 EDA correlation](EXP-049-eda-correlation-2026-05-13.md) · [EXP-035 Kaggle 金牌战略复盘](EXP-035-kaggle-gold-retrospective-2026-05-10.md) · [EXP-040 gap=0 DIN seed sweep](EXP-040-m5-m9-gap0-din-seed-sweep-2026-05-12.md)

---

## TL;DR

5/14 早 evaluation 回流：M18 (T33 NLIR) / M19 (T31 item is-missing) / M20 (seq_d 512→1024) **3 个 gap=0.05 path 方法学方向 LB 全部 ≤ 0.84329**，**无一突破当前 gap=0 path SOTA M9 (0.84671)**。

### 3 个 LB 数据点

| Job | Config (vs T25-bypass-s42 baseline 加什么) | E4 Dev | LB | Dev→LB gap |
|---|---|---|---|---|
| **M20** | + `seq_d 512→1024` (DECEM Trick 2) | 0.844231 (E3) | **0.843287** | +0.000944 |
| **M19** | + T31 item-side is-missing (DECEM Trick 3) | 0.843437 | **0.842033** | +0.001404 |
| **M18** | + T33 NLIR gating | 0.843312 | **0.841364** | +0.001948 |

vs **gap=0.05 path historical LB peak T25-bypass-s42 = 0.843433**:
- M20 Δ−0.000146 (持平)
- M19 Δ−0.0014 (1.75σ rerun 噪声内)
- M18 Δ−0.0021 (2.6σ · 唯一明确负)

### 3 条新铁律

1. **🔥 gap=0.05 path 方法学 ceiling = 0.844 已达 · 不再投资源做 gap=0.05 path 方法学验证** (EXP-035 5/10 警告 4 天后再次实证)
2. **🔥 单 Dev 在 gap=0.05 path 高 +0.0005 LB 不涨 · "Dev/LB 高保真" 假设需要修订**: gap=0.05 path Dev→LB 在 ceiling 附近变成 noise · 0.844 以下高保真 · 0.844 以上失效
3. **🔥 "切换领地" 是 SOTA 冲分阶段唯一路径**: gap=0.05 (方法学验证) → gap=0 (SOTA 上沿搜索) · 已 work 的方法 (T29 vocabtier / T34 DCN / Tokenization) **必须叠加到 gap=0 path 才有 LB 收益**

### 1 个核心战略修正

**原战略 (5/13 EXP-049 落档时)**: "gap=0.05 path 验证方法学 → 最后切到 gap=0 path 冲 SOTA"
**修正战略 (5/14)**: **立即切换** · M22+ 全部走 gap=0 path · M9-Stack 系列 (M9 + T29 / + T34 / + Tokenization / + DIN multi-mode) 直接在 gap=0 path 验证

---

## 1. 详细数据对照

### 1.1 gap=0.05 path 全部 LB 数据点 (含本次 3 数据点)

| Job | Config | E4 Dev | LB | Dev→LB gap | 时间 |
|---|---|---|---|---|---|
| icq-half-gap05 | gap=0.05 × s42 · no-DIN baseline | ~0.840 | 0.83922 | +0.0008 | 早期 |
| icq-half-gap05 (s5) | gap=0.05 × s5 · no-DIN | — | 0.840166 | — | 早期 |
| icq-half-gap05 (s11) | gap=0.05 × s11 · no-DIN | — | 0.841591 | — | 早期 |
| **T25-bypass-s42** ⭐ | gap=0.05 × s42 × T25 DIN bypass | 0.843696 | **0.843433** | +0.000263 | 5/10 |
| T25-bypass-s3 | gap=0.05 × s3 × T25 DIN | ~0.843 | 0.843069 | ~0 | 5/10 |
| T25-bypass-s7 | gap=0.05 × s7 × T25 DIN | — | 0.842373 | — | 5/10 |
| T25-bypass-s11 | gap=0.05 × s11 × T25 DIN | — | 0.841317 | — | 5/10 |
| T25+T27 (compile+AMP) | gap=0.05 × s42 × T25+T27 | — | 0.841801 | — | 5/10 |
| M3 t29-vocabtier-s42 | gap=0.05 × s42 × T29 vocabtier | 0.841629 | 0.840603 | +0.001 | 5/12 |
| **M11** t29-vocabtier-s3 | gap=0.05 × s3 × T29 vocabtier | 0.841455 | 0.84145 | +0.000 | 5/12 |
| M1 t28-din-integrated-s3 | gap=0.05 × s3 × T28 DIN integrated | 0.845197 | 0.841807 | +0.00339 | 5/12 |
| M2 t28-din-integrated-s42 | gap=0.05 × s42 × T28 DIN integrated | 0.840669 | 0.837576 | +0.00309 | 5/12 |
| **M20** | gap=0.05 × s42 × T25 + seq_d=1024 | 0.844231 (E3) | **0.843287** | +0.000944 | **5/14** |
| **M19** | gap=0.05 × s42 × T25 + T31 item is-miss | 0.843437 | **0.842033** | +0.001404 | **5/14** |
| **M18** | gap=0.05 × s42 × T25 + T33 NLIR | 0.843312 | **0.841364** | +0.001948 | **5/14** |

### 1.2 gap=0.05 path LB 分布统计

```
N = 14 data points (gap=0.05 path)
LB min:        0.83758  (M2 T28 integrated s42)
LB max:        0.84343  (T25-bypass-s42 ⭐)
LB mean:       0.84137
LB median:     0.84167
LB std:        0.00163
```

**所有 LB ≤ 0.84343** · 14 个独立 (config × seed) 组合 0/14 突破 0.844 · **gap=0.05 path 真实 LB ceiling ≈ 0.84343 ± seed noise**。

### 1.3 vs gap=0 path SOTA

| Path | Best LB | Mean LB | data points | Dev→LB gap |
|---|---|---|---|---|
| **gap=0 path** | **M9 = 0.84671** | 0.8424 (5 seed icq-half) | ~10 | +0.022 (Dev 0.868 → LB 0.846) |
| **gap=0.05 path** | T25-bypass-s42 = 0.84343 | 0.8414 | 14 | +0.0008 (Dev 0.843 → LB 0.842) |

**两个 path LB 上沿差距 = 0.84671 − 0.84343 = +0.00328** · 几乎等于"5% 训练数据" 的边际贡献。

---

## 2. 机制分析 · 为什么 gap=0.05 path 卡在 0.844

### 2.1 真实机制（10 年研究员视角）

**之前的归因（agent 5/14 11:30 给的）**: "加东西必败 · 加结构 → Dev→LB gap 扩大"
**真实归因（修订）**: gap=0.05 path 数据少 5% (只用 90% 的 train rows · 5% 切到中间 gap · 5% 保留作 valid) → **真实模型能力上限受 train data 量限制 · 不是方法学不 work**

**支撑证据**:
1. M20 加 seq_d=1024 (理论上让模型看到更多信息) · LB 持平 T25-bypass-s42 · 说明**额外信息已经被 ceiling 阻挡**
2. M18/M19 LB 落在 [0.8414, 0.8420] · 与 gap=0.05 path **mean** (0.8414) 几乎完全重合 · 说明落入 seed noise 中心 · **没有"加结构必败" 的方向性 bias**
3. T25 bypass 本身在 gap=0.05 path 跨 4 seed 跨度 0.0021 (0.8413~0.8434) · gap=0.05 path 真实 seed noise std ≈ 0.001 · **M18/M19 LB 与 baseline 差 0.001~0.002 全在 1~2σ 内**

### 2.2 "Dev → LB 高保真" 假设的修订

**之前铁律 (EXP-030)**: gap=0.05 path Dev→LB gap ±0.001~+0.003 · Dev 是 LB 高保真代理
**现在修正**: 这个铁律**只在 LB ceiling 0.844 以下 work**
- LB < 0.842 时: Dev +0.001 → LB +0.001 (高保真)
- LB ≈ 0.843 时: Dev +0.001 → LB +0.000 (饱和)
- LB > 0.844 时: 无样本验证 · 但理论上 dev 进一步涨需要 train data 增加 · gap=0.05 path 卡在 90% data

**操作化**: gap=0.05 path 上 **Dev > 0.844 不再视为"会传到 LB"的有效信号** · 必须切到 gap=0 path 验证。

---

## 3. 战略反思 · EXP-035 警告 4 天后再次实证

### 3.1 EXP-035 (5/10 CEST 09:55) 已说过的话

> **错误 2: 混淆"均值 vs 上沿"战略**
> - SOTA = 单点上沿 (比赛规则是 LB 单点最高), 不是均值
> - 我方 SOTA 0.84637 = gap=0 × seed=42 × **非 DIN** 上沿
> - 要突破 SOTA, **T25 DIN 方法学 gain 必须能"叠加到 gap=0 上沿"**, 不是在 gap=0.05 领地里攀爬
> - **错误逻辑链**: gap=0.05 T25 DIN 均值 0.842 < gap=0 SOTA 0.846 = **在更低的山头上爬更高就能过山顶**？不, 需要**换山头**

### 3.2 5/10~5/14 实际发生的事

| 日期 | gap=0.05 path 投入 | gap=0 path 投入 | 结果 |
|---|---|---|---|
| 5/10 | 4 jobs (T25 seed sweep) | 0 | T25 LB 0.843433 |
| 5/11 | 4 jobs (T28/T29) | 0 | T28 证伪 / T29 弱正 |
| 5/12 | 3 jobs (M6/M7/M8) | M5/M9 启动 | M9 LB **0.84671** SOTA |
| 5/13 | 4 jobs (M14-M17) | M14/M15 (gap=0 seed sweep) | M14/M15 Dev 持平 |
| 5/14 | M18/M19/M20 evaluated | 0 | **3 LB 全 ≤ 0.84343 · LB SOTA 0 进步** |

**4 天投入 gap=0.05 path ~12 jobs · LB SOTA 0 增益**。EXP-035 的警告几乎逐字应验。

### 3.3 为什么我们重复犯错

| 原因 | 解释 |
|---|---|
| **Dev 信号在 gap=0.05 path 比 gap=0 path 更"可读"** | gap=0.05 path Dev→LB gap 小 · agent 倾向于"看 Dev 好就继续" |
| **EDA 数据来自 gap=0.05 path** (M16 EDA piggyback 用 gap=0 但分析方向都是叠加 gap=0.05 的 method) | EXP-048/049 出来的 T31/T34 候选都默认 gap=0.05 path 验证 |
| **agent 没有定期 audit "我们在哪个山头"** | EXP-035 立的"每 48h audit"铁律 5/10 后**未执行过一次** · 直到 5/14 用户反问才触发 |

---

## 4. 立即可执行的修正路线 (5/14 ~ 5/25)

### 4.1 战略原则

| # | 原则 |
|---|---|
| **P1** | **所有新 training job 走 gap=0 path** (`--enable_row_valid_gap` 关闭) · gap=0.05 path 仅用作 historical reference |
| **P2** | **M9-Stack 系列**: M9 (= gap=0 + T25 + timefeat + hash500k) 加单个 stack 候选 · 单变量验证 |
| **P3** | **每 48h audit "DECEM 已说 work 但未投入" 清单** + "gap=0.05 vs gap=0 投入比" + "LB SOTA 增益" 账目 (EXP-035 已立铁律 · 这次必须执行) |
| **P4** | **Top 20 (LB 0.850) 路径 = M9 + 2~3 个正交 stack 全部 work** · 每个 +0.001~0.002 |

### 4.2 M9-Stack 候选清单 (按 ROI 排序)

| 候选 | 配置 | 是否 work 在 gap=0.05 | gap=0 path 预期 LB | 风险 | 代码改动 |
|---|---|---|---|---|---|
| **M22** = M9 + **T29 vocabtier** | `--ns_groups_json src/ns_groups_vocab_tier.json` | ✅ +0.0014 LB (M3) | M9 + 0.001 = **~0.847** | 低 (单 flag) | 0 |
| **M23** = M9 + **T34 DCN cross** | `--enable_dcn_cross` (M21 改 gap=0) | 🟡 M21 dev 在跑 | M9 + 0.001 = **~0.847** | 中 (新模块) | 已 done |
| **M24** = M9 + **DIN target=item-only** | 改 `_compute_target_repr` 只用 item | 🆕 未试 (杭仔 0.85+ 路线) | M9 + 0.001 = **~0.847** | 中 (50 行) | ~50 行 |
| **M25** = M9 + **NS-groups 8-cluster** (T35) | 写 `ns_groups_8cluster.json` | 🆕 未试 (EDA 实证) | M9 + 0.0005 = **~0.847** | 低 (json 配置) | 0 (生成 json) |
| **M26** = M9 + **T16 target-item seq injection** | ADR-004 方案 A (alpha 0.1) | 🆕 未试 | M9 + 0.001 = **~0.847** | 中 (~80 行) | 已 ADR · 未实施 |
| **M27** = M9 + **Recency reweight α=2.5** | `--temporal_weight_alpha 2.5` | 🆕 未试 (EXP-035 ROI 下调到 +0~+0.001) | M9 ± 0.001 | 中 (D-1 占 63% 隐式 recency) | 0 |

**优先级 ROI 排序**:
- M22 (T29 vocabtier 已知 work · 最稳)
- M23 (T34 DCN · M21 已在跑 · 复用 ckpt)
- M24 (杭仔 0.85+ 实证 · 但实现细节不明)
- M25 (EDA 实证强 · ROI 适中)
- M26 (ADR-004 已设计 · 实施成本高)
- M27 (ROI 不确定)

### 4.3 11 天分阶段执行

**Stage 1 · 5/14 ~ 5/16 (3 天 · ~12 GPU slots)**
- 启动 M22 / M23 / M24 / M25 (4 个 stack 验证 · gap=0 path)
- 每天 4 quota evaluate · 收集 4~6 个新 LB 数据点
- 确认 1~2 个 stack work (LB ≥ 0.847)

**Stage 2 · 5/17 ~ 5/20 (4 天 · ~16 GPU slots)**
- 双叠加测试 (M9 + Stack-X + Stack-Y) · 验证可加性
- M9 + Tokenization 进一步实验 (DECEM 真信号未消化)
- 选 top 2 stack work 的方法学进入三叠加阶段

**Stage 3 · 5/21 ~ 5/23 (3 天 · ~12 GPU slots)**
- 三叠加测试 (M9 + Stack-X + Stack-Y + Stack-Z)
- 选 best config (假设触 LB 0.848~0.850)
- gap=0 × 多 seed 上沿搜索

**Stage 4 · 5/24 ~ 5/25 (2 天 · 1 final job)**
- 100% data S4 final training (跳过 valid 切分 · Seed 主楼公开建议)
- 最后 evaluation 提交 · ddl 5/25

---

## 5. 4 个判决问题 (待用户回答)

**Q1**: 同意"切换到 gap=0 path · 不再做 gap=0.05 path 方法学验证"吗？
**Q2**: 今晚 1 quota 用来 eval M21 (gap=0.05 path) 还是留到明早？
**Q3**: M22/M23/M24/M25 启动顺序 · 选 3 个 (推荐 M22 + M23 + M24 / M25)
**Q4**: 5/24 是否做 100% data S4 final?

---

## 6. 后续行动清单

- [x] EXP-051 落档 (本文档)
- [ ] team-memory 加 3 条新铁律 (gap=0.05 ceiling / 切换领地原则 / 每 48h audit 强制执行)
- [ ] active.md 更新最近一次审视
- [ ] commit
- [ ] M21 完成后立即用今日剩余 1 quota eval (CEST 23:44 之后)
- [ ] 凌晨 4 slot 起 M22/M23/M24/M25 (gap=0 path)

---

## 7. Cost Tracking

- 5/14 早 3 evaluation quota 消耗 (全部 LB < 0.844 · 不是 SOTA)
- 5/13 凌晨 3 training slot (M18/M19/M20) · 共 ~15 GPU-hours
- 净 LB SOTA 增益 = 0
- 但 **战略级别**收益 = "gap=0.05 path ceiling 实证 · 切换领地决策依据" · 这是 +∞ 的方法学价值 (终止低 ROI 路径)

---

## 8. 反思

EXP-035 5/10 立的 "金牌视角 48h audit" 铁律 4 天后未执行一次 · 直到用户主动反问 "10 年研究员视角深度反思" 才触发本次 audit。**未来 agent 必须在每个 ≥4 LB 数据点回流的批次后主动做 audit · 不等用户提示**。
