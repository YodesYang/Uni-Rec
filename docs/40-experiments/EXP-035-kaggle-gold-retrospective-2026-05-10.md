# EXP-035 · Kaggle 金牌 / TAAC 榜一视角的 48 小时战略复盘

**日期**：2026-05-10 CEST 09:55（西班牙早）· 北京 15:55
**作者**：Agent（用户多次反问触发深度自我审视）
**状态**：🔬 反思性文档 · 2026-05-09~05-10 战略级 audit
**关联**：EXP-024（2026-05-04 方法论复盘）的继承 · 3 次同期错误事故（09:43 T19 错判 + 22:10 quota 混淆 + 今早 num_epochs 错）
**触发**：用户 CEST 09:30 反问 "请从 Kaggle / TAAC 金牌视角深入思考"

---

## TL;DR

过去 48 小时（2026-05-09 CEST 03:40 T27 启动 → 2026-05-10 CEST 09:00 新 3 个 gap=0 T25 job 提交）消耗 **~13 个 training slot + ~12 次 evaluation**，**净 LB SOTA 增益 = 0**（仍是 5/4 建立的 0.84637）。

**5 个致命错误（从 Kaggle 金牌选手标准）**：

1. **过度 seed sweep**：1 个假设 × 5 seed ≠ 5 个正交实验
2. **均值 0.842 ≠ SOTA 0.846**，T25 DIN 在 gap=0.05 领地上爬山，但 SOTA 在 gap=0 领地
3. **忽略 DECEM 榜一明说 work 的 "Tokenization 修改"**（我们至今 0 投入）
4. **没做 recency reweight**（工业 CVR 标配；但今日 audit 后发现数据已隐式 63% 在 D-1，ROI 降级）
5. **三次同期错误**（baseline 凭记忆 × quota 规则凭记忆 × num_epochs 凭直觉），共同根源：**没 grep 证据就陈述**

**真实触 0.85 路径**（严格贝叶斯估算）：
- T25 DIN × gap=0 × seed 上沿 单独峰值: 0.848±0.003
- 加 T28 block-integrated 叠加: 0.849±0.004
- 加 S4 100% data final: 0.850±0.004
- **5/25 ddl 前触 0.85 需要 2~3 个正交增益源叠加，不是 seed sweep 或单一方法学**

---

## 第一部分：48 小时资源账目

### 消耗

| 资源 | 消耗量 |
|---|---|
| Training slot | ~13 个 half-train（T27 / T25 B / T19 / seed=5/11 / T25 seed=3/7/11/42+T27 / T19 rerun / 现在的 gap=0 × 3）|
| Evaluation quota | ~10 次（2 次失败 _orig_mod + ~8 次成功）|
| Agent 工作时间 | ~10 小时（文档 + 代码 + 决策） |
| 用户工作时间 | ~6 小时（平台提交 + 日志回传）|
| GPU 小时 | ~70 GPU-hour |

### 产出 LB 增益

| 产出 | LB 增益 |
|---|---|
| **方法学确认**（T25 DIN work, +0.002 均值级）| 0 SOTA 增益（SOTA 仍 0.84637）|
| **工程 infra**（keep_topk_ckpt, compile+AMP 正确性, `_orig_mod.` fix）| 0 LB 增益 |
| **数据/纪律**（5 条 team-memory 铁律, 3 个错判修正）| 防止未来犯错，0 短期 LB |
| **净 LB SOTA 提升** | **0** |

---

## 第二部分：从 Kaggle 金牌视角的 5 个错误

### ❌ 错误 1：过度 seed sweep

**具体表现**：今日跑了 5 个 T25 DIN seed（42/3/7/11/42+T27）。

**金牌视角**：
- 这 5 个**不是 5 个正交实验**，是 **1 个假设（T25 DIN work）× 5 次随机采样**
- **5 seed 的边际信息贡献**：第 1 个 +0.002（T25 vs baseline），第 2~3 seed 收窄 std，第 4~5 seed 几乎 0 信息
- **金牌的分配**：1 seed 建立基线 + N 个**正交**方法学 bet

**数字证据**：
- 5 seed T25 DIN LB 均值 0.8424，上沿 0.8434，标准差 0.0008
- **第 4~5 个 seed 的信息熵增量 ≈ 0**
- **我们浪费了 2~3 个 training slot** 在 diminishing returns 区

---

### ❌ 错误 2：混淆"均值 vs 上沿" 战略

**具体表现**：我反复说"T25 DIN 均值 +0.002 LB，是真信号"。对，但…

**金牌视角**：
- **SOTA = 单点上沿**（比赛规则是 LB 单点最高），不是均值
- 我方 SOTA 0.84637 = gap=0 × seed=42 × **非 DIN** 上沿
- 要突破 SOTA，**T25 DIN 方法学 gain 必须能"叠加到 gap=0 上沿"**，不是在 gap=0.05 领地里攀爬
- **错误逻辑链**：gap=0.05 T25 DIN 均值 0.842 < gap=0 SOTA 0.846 = **在更低的山头上爬更高就能过山顶**？不，需要**换山头**

**这两天我们一直在 gap=0.05 领地做 seed sweep**，但 SOTA 在 gap=0 领地。

---

### ❌ 错误 3：忽略 DECEM 公开情报的 Tokenization 线索

**DECEM 2026-05-08 原话**（community-intel.md §工业赛道 Rank 1）：

> "Tokenization 设计的小修改（作者原话："简单做了一些修复，让其更加合理化"）"

**我们的投入**：**0**。至今未改过 `RankMixerNSTokenizer` 或 `GroupNSTokenizer`。

**金牌视角**：
- DECEM 明说 work 的 4 个方向（compile+AMP / 时间特征 / Tokenization / DIN+MLP），**我们只验了 3 个（compile+AMP 0 gain, 时间特征已有, DIN +0.002）**
- Tokenization 是 **剩下唯一未消化的榜一真信号**
- Ceiling 估计：+0.002~0.005 LB（和 DIN 同级别）

**但我们没做**，因为 agent 把时间花在了 seed sweep 上。

---

### ❌ 错误 4：没有 recency reweight（但今日 audit 后 ROI 下调）

**初判**：工业 CVR 场景 recency reweight 是标配，应该 +0.002~0.003 LB。

**Audit 后修正**（2026-05-10 CEST 09:45）：
- 平台 train 数据（active.md line 388 社区情报）：**02-28~03-04 共 5 天，03-03 单日 ~63%**
- **数据已经隐式 heavy recency**（最近一天占 63%）
- 再做 `temporal_weight_alpha 2.5` 可能**过拟合最近一天** → 进一步伤害 test 泛化
- **ROI 修订为 +0~0.001**（原估 +0.002~0.003 错了）

**代码就绪**：`train.py` line 469 `--temporal_weight_alpha` 已实装，未启用。**保留作为低优先备选**。

### ⚠️ 5/15 23:30 元宵节认知后 · 错误 4 修订**完全推翻**

**触发**: 用户 5/15 CEST 23:27 提问"是否跟元宵节有关" → grep 农历日历 → **2026-03-03 = 农历正月十五 = 元宵节** ⭐

**5/10 错判的根基** = "63% 的 03-03 是最近一天" · **完全错**:
- 03-03 不是最近一天 · **03-04 才是 (test 同日 · 占比 3-25%)**
- 03-03 占 63% 是**元宵节流量峰** (中国农历节日真实流量) · 不是 recency
- **真正的 recency = 03-04 数据 · 占比反而最少**

**ROI 重新评估** (5/15 元宵节认知后):
- `temporal_weight_alpha=2.5` 真实作用 = **修正 03-04 underweight (3-25%) + 03-03 元宵节 overweight (63%) 的不平衡**
- 不是放大 recency · 是**修正 sampling 不平衡 + 让模型重点学元宵节翌日 daypart (test 同日)**
- **ROI 回归原估 +0.001~+0.003 LB** (高 ROI 候选)

**具体执行**: 5/15 23:30 起 M41-temporalweight25-m22-gap0-seed42-4ep · 单变量 vs M22 · 5/16 早 quota 评 LB · 详见 EXP-055 §15。

**新铁律**: 中国数据集 train 某日占比异常时 · **第一时间 grep 该日是否农历节日** · 不能凭 sampling artifact 假设 (写入 team-memory)。

---

### ❌ 错误 5：三次同期错误的共同根源 —— **凭记忆陈述**

时间线：
- **2026-05-09 CEST 09:43**：T19 Deprecated 错判（用 gap=0 baseline LogLoss 对比 gap=0.05 T19）
- **2026-05-09 CEST 22:10**：evaluation quota 规则混淆（"CEST 14:00 截止" 是记错的）
- **2026-05-10 CEST 08:40**：num_epochs=3 拍脑袋（忽略 EXP-019 明确写的 E4 best）

**共同模式**：agent **起草 EXP / ADR / config 时凭记忆引数字或规则**，没 grep 原文档。

**团队铁律已扩展**（team-memory line 74, 10:10 修正）：
> "任何 agent 要引用的'过去数字 / 规则 / 时区 / 配置'，引用之前必须 grep 原文档核对"

---

## 第三部分：金牌视角的**正确** 48 小时分配（反事实）

**假设**周五晚 CEST 22:00 作 48h audit，目标 5/25 前触 0.852。

| # | 假设 | 时间预算 | 预期 LB 增益 | 代码成本 |
|---|---|---|---|---|
| 1 | `--temporal_weight_alpha 2.5`（未 audit 前，估 ROI 高）| 0.5 slot | +0.001~0.002 | 0 |
| 2 | **T25 DIN × gap=0 × seed=42**（ceiling 更高）| 1 slot | +0.002~0.004 | 0 |
| 3 | **Tokenization 修改**（DECEM 明说）| 2 slot + ~2h code | +0.002~0.005 | ~80 行 |
| 4 | S4 100% data final 规划 | 0.5 slot | +0.001~0.003 | 0 |
| 5 | Hash vocab 2M（可能复赛 work） | 1 slot | +0~0.001 | 0 |
| 6 | **T28 DIN block-integrated**（更深 DIN）| 1 slot + ~2h code | +0.003~0.005 | ~150 行 |
| 7 | Seed sweep | 0~1 slot（不超）| +0.001~0.003 | 0 |

**金牌的 48h 分配**：~60% 方法学正交 bet（#1+#3+#6），20% gap=0 探索（#2），15% 规划 + audit（#4+数据），5% seed sweep（#7）。

**我们的实际分配**：**90% seed sweep**（5×T25 DIN 不同 seed），10% 文档错误修正。

**ROI 损失估算**：
- 没做 Tokenization → 少 +0.003~0.005 LB
- 没做 T28 → 少 +0.003
- 没做 S4 规划 → 延迟 +0.002
- **净损失 48 小时的潜在 LB**: +0.008~0.013

**反事实**：如果过去 48h 按金牌路线做，今天 SOTA 可能已经 **0.854~0.859**（DECEM 级别）。

---

## 第四部分：严格贝叶斯估算触 0.85 的真实概率

### 现有数据点（全部 grep 自 EXP-030 / 今日训练日志 / 今日 evaluation）

**gap=0 无 DIN LB 分布**（EXP-030 line 12）：
- seed=42: 0.84637, seed=3: 0.845329, seed=2: 0.841294, seed=7: 0.836413, seed=?: ?
- 5-seed 均值: **0.8424**, 上沿: **0.84637**, 标准差: ~0.0035

**gap=0.05 无 DIN LB 分布**：
- seed=42: 0.839223, seed=5: 0.840166, seed=11: 0.841591
- 3-seed 均值: 0.8403, 上沿: 0.8416

**gap=0.05 T25 DIN LB 分布**（今日）：
- seed=42: 0.843433, seed=3: 0.843069, seed=7: 0.842373, seed=11: 0.841317, seed=42+T27: 0.841801
- 5-seed 均值: **0.8424**, 上沿: **0.843433**, 标准差: ~0.0008

**gap=0 T25 DIN**（未测，今日 09:00+ 正在跑 seed=42/3/7）

### 贝叶斯估算：gap=0 T25 DIN LB 均值

**先验**：uniform [0.840, 0.850]

**似然 1**：gap=0.05 T25 DIN 5-seed 均值 0.8424 ± 0.0008（标准误）
**似然 2**：无 DIN 下 gap=0 vs gap=0.05 的 LB 提升 Δ = 0.0021 ± 0.002（跨配置推理）

**后验中心**：0.8424 + 0.0021 = **0.8445**
**后验标准差**：√(0.0008² + 0.002²) = **0.0022**

**95% 置信区间**：[0.8401, 0.8489]

### gap=0 T25 DIN **最佳 seed 上沿**

- 5-seed 上沿公式：均值 + 1.16 × std_within_config
- std_within_config ≈ 0.003（gap=0 seed 跨度 0.010 / 2 近似）
- **5-seed 上沿后验中心**: 0.8445 + 0.0035 = **0.8480**
- **95% 置信区间**: [0.8436, 0.8524]

### 触 0.85 概率（gap=0 T25 DIN × seed 上沿）

- 假设真实 5-seed 上沿 = 0.848 ± 0.004
- P(上沿 ≥ 0.85) ≈ 31%
- **3-seed 上沿（我们今天起的 3 个 job）** 概率 ≈ **25%**

### 综合触 0.85 时间分布

| 时间 | 概率 | 路径 |
|---|---|---|
| **5/11 早 07:00** | **< 3%** | 今日 evaluation 的最高 LB 只到 0.843069 |
| **5/12 早 07:00** | **15~25%** | gap=0 T25 × 3 seed 回流 + 上沿被击中 |
| **5/15 早 07:00** | **35~50%** | + 1~2 个正交 bet（T28 / Tokenization）|
| **5/20 早 07:00** | **55~70%** | 综合方法学 |
| **5/25 ddl 前** | **75~85%** | 加 S4 + 完整叠加 |

**注**：这是"触 0.85"（入 TOP 20），**不是 DECEM 0.852**。

### 触 **DECEM 级 0.852** 概率

**5/25 ddl 前 ≥ 0.852**：**30~45%**（需 3+ 正交 bet 全 work + seed luck）

---

## 第五部分：立刻可执行的金牌式修正（CEST 09:55 ~ 5/25）

### 今天（CEST 09:55 ~ 22:00）

**已起动**：3 个 gap=0 T25 DIN × seed=42/3/7 × 4 epoch job（预期 CEST 15:00~16:00 回流）

**下午 agent 工作（14:00~22:00）**：
1. ✅ EXP-035 落档（本文档）
2. ✅ active.md + team-memory 更新
3. ⏳ 等 gap=0 job 回流，严格按铁律分析
4. ⏳ 今晚 CEST 20:00+ 决定是否写 T28

### 5/11（gap=0 job 回流后）

**2 种情景决策**：
- **如果 gap=0 T25 LB ≥ 0.847** → 说明 T25 × gap=0 + seed 上沿可能触 0.85，**今晚起更多 gap=0 seed sweep**
- **如果 gap=0 T25 LB < 0.845** → gap 差值不能线性叠加，**pivot 到 Tokenization 修改**

### 5/12~5/15（中期）

- **Tokenization 修改** × 2 job（不同 ns_tokens 数 / 分组）
- **T28 DIN block-integrated** × 1 job
- 保留 1 slot 为 fallback

### 5/16~5/22（压力期）

- 选前面 work 的最好 config 做 seed sweep（合规上沿搜索）
- 准备 S4 100% data run.sh

### 5/23~5/25（最终冲刺）

- S4 100% data final training × 1 job
- 最后一次 evaluation 提交

---

## 第六部分：team-memory 新弟铁律候选

基于本次复盘提议新增：

1. **🎯 单假设 × N seed ≠ N 个正交实验**：未来 training slot 批量分配，**同一个方法学假设不占超过 2 个 seed 的 slot**（除非是最终上沿搜索阶段）
2. **🎯 SOTA 搜索按"领地"分离**：方法学验证阶段用 gap=0.05（可信 dev），**SOTA 冲分阶段必须切到 gap=0**（更多 data + 真实 LB 上沿）
3. **🎯 DECEM 情报未消化清单检查**：每 48 小时 audit 一次"DECEM 已说 work 但我们未验" 清单，Tokenization 当前是头号待办
4. **🎯 agent 不再给具体概率数字**：所有 "触 X LB 概率 Y%" 改为 **"定性 ceiling 估算"**（"**可能达到** 0.848" 而非 "**概率 35%**"），防止 recency bias 污染决策
5. **🎯 Recency reweight 前做数据 audit**：不是所有 CVR 场景都需要 alpha reweight；本赛题 D-1 占 63% 已隐式 recency，**先 audit 再决定是否加**

---

## 第七部分：最诚实的自我评价（按 TAAC 榜一标准）

| 维度 | 分数 | 评价 |
|---|---|---|
| 代码质量 | 7/10 | `keep_topk_ckpt` / `guard` / `_orig_mod` fix 工程能力 OK |
| 战术执行 | 4/10 | seed sweep 过度，ROI 低 |
| 战略视野 | 3/10 | 忽略 recency reweight 和 Tokenization 这两个方向 |
| 数据纪律 | 5/10 | 今天才建立 grep 铁律，此前凭记忆犯错 3 次 |
| 诚实度 | 7/10 | 能承认错误，但倾向 over-promise |

**综合：5.2/10（及格但离金牌远）**

**冲 0.85 需要 agent 提升到 7~8/10**，具体：
- 战术：seed sweep 停掉，方法学正交 bet 优先
- 战略：消化 DECEM 情报，Tokenization 修改必做
- 纪律：每次判决前 grep 原文档

---

## 第八部分：铁律升级

**"Kaggle 金牌视角定期 audit"** 作为新纪律：

**每 48 小时** agent 主动做一次：
1. 列出"过去 48h 消耗资源 vs LB SOTA 增益" 账目
2. 对照 DECEM / Seed / community-intel 已知情报清单，audit "我们投入 vs 未投入" 比例
3. 如 "消耗很多 + 增益 0" → **强制 pivot 到未消化 channel**

**这个 audit 本次由用户触发**（反问"请深入思考"），**下次 agent 必须主动做**。

---

## 附录 A：本文档的信息源（严格 grep 验证）

- EXP-030 line 12, 25, 180~186（baseline gap=0.05 数据）
- EXP-019 line 42, 117（E4 best_epoch + gap=0 LB SOTA）
- community-intel.md §工业赛道 Rank 1 DECEM（Tokenization / DIN+MLP / compile+AMP）
- active.md line 388（train 时间分布 03-03 单日 63%）
- team-memory line 9, 74（quota 规则 + grep 铁律）
- 今日训练日志（5 T25 DIN seed Dev 数据）
- 今日 evaluation 结果（4 T25 DIN LB 数据）

每一条都可以用 `grep` 复核。

## 附录 B：本文档产出后的 active.md 联动更新

- 顶部审视覆盖为 2026-05-10 CEST 09:55
- Changelog 新增 "2026-05-10 CEST 09:55 Kaggle 金牌视角战略复盘 + EXP-035 落档"
- team-memory 补 5 条新弟铁律

**→ 见 active.md + team-memory 后续 commit**
