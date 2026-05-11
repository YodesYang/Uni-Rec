# EXP-037 · u4i3 × gap=0 Seed Sweep + EDA Abandoned · 2026-05-10 晚到 2026-05-11 早

## 背景

EXP-036（今日早上）给出 u4i3 × gap=0.05 × E3 Dev 0.843743 的**正向 LogLoss 信号**（vs T25 B 的 0.27841 → 0.27596，Δ−0.00245）。为了验证 "Tokenization u4i3 能否把 gap=0 领地的 SOTA 向上推"，今晚起 2 个 u4i3 × gap=0 × DIN job：

- Slot 1: `t25-din-u4i3-gap0-seed42-4ep`
- Slot 2: `t25-din-u4i3-gap0-seed3-4ep`（松 guard `delta=0.003 patience=2`）

**关联**：EXP-036（u4i3 × gap=0.05 结果）· EXP-019（gap=0 seed=42 × no-DIN × E4 SOTA baseline 0.84637）· EXP-035（Kaggle 金牌战略 audit · "Tokenization 是 TOP 1 未消化线索"）

---

## 训练配置

### Job 1 · `t25-din-u4i3-gap0-seed42-4ep`

- user_ns_tokens **4**, item_ns_tokens **3**（u4i3; 对比 baseline u5i2）
- num_queries 2, d_model 64, T = 2×4 + 8 = **16**（整除检查 ✅）
- seq_hash_vocab 500k, seq_encoder swiglu, ICQ on, DIN bypass on
- valid_split_strategy time, **valid_ratio 0.05（gap=0）**
- seed=42, batch_size 128, num_epochs 4
- `logloss_rebound_patience=1 delta=0.0`（严格）

### Job 2 · `t25-din-u4i3-gap0-seed3-4ep`

- 同上，只改：
  - seed=3
  - `logloss_rebound_patience=2 delta=0.003`（松 guard，让 seed=3 跑到 E4）

---

## 训练结果（2026-05-11 早 CEST 03:00~06:28 回流）

### Job 1 · seed=42 × u4i3 × gap=0 × DIN × 4ep

**Source**：`~/Downloads/train_log.txt` line 1~20（2026-05-11 06:28:21 BJT 回流）

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.865045 | — |
| 2 | 31882 | 0.866301 | — |
| 3 | 47823 | 0.867706 | — |
| **4** | **63764** | **0.868055** ⭐ | **0.22316** |

**stop_reason**: `max_epochs_completed`
**best_checkpoint**: global_step63764.auc=0.868055.best_model
**Top-K**: rank1 E4 0.868055 / rank2 E3 0.867706 / rank3 E2 0.866301

### Job 2 · seed=3 × u4i3 × gap=0 × DIN × 4ep（松 guard）

**Source**：`~/Downloads/train_log.txt` line 22~42（2026-05-11 03:11:40 BJT 回流）

| Epoch | step | Dev AUC | Dev LogLoss |
|---|---|---|---|
| 1 | 15941 | 0.864464 | — |
| 2 | 31882 | 0.866535 | — |
| 3 | 47823 | 0.868105 | — |
| **4** | **63764** | **0.868481** ⭐ | **0.22321** |

**stop_reason**: `max_epochs_completed`（松 guard 生效，未被 E3 停）
**best_checkpoint**: global_step63764.auc=0.868481.best_model
**Top-K**: rank1 E4 0.868481 / rank2 E3 0.868105 / rank3 E2 0.866535

---

## 关键分析

### Finding 1: u4i3 × seed=42 Dev 比 EXP-036 A baseline **明显下降**

**对照基线**（EXP-036 A · 今日早上 gap=0 × seed=42 × 标准 tokenization u5i2 × DIN × E4）：

| Config | E4 Dev | Δvs baseline | 判决 |
|---|---|---|---|
| **baseline u5i2** (EXP-036 A) | **0.868924** | — | — |
| **u4i3** (今晚 seed=42) | **0.868055** | **Δ−0.000869** | **明显负向**（3× rerun 噪声 0.0003 的约 3 倍）|

**机制推测**（假设，未验证）：
- u5i2 = user_ns 5 × chunk_dim(d_model/T=64/16=4)... wait let me re-check
- 实际 T = 2×num_queries×num_sequences + num_ns = 2×4 + num_ns
- u5i2: num_ns = 5+2+1 = 8, T = 16 ✅
- u4i3: num_ns = 4+3+1 = 8, T = 16 ✅（整除不变）
- **真正差异**：user_int 14 个 fid 分到 user_ns=5 组（~2.8 fid/组）vs 4 组（~3.5 fid/组）
- item_int 14 个 fid 分到 item_ns=2 组（~7 fid/组）vs 3 组（~4.7 fid/组）
- u4i3 **item 细化 +1 group** 但 **user 粗化 −1 group**
- gap=0 领地考"近未来记忆" → user 粗化损失 user-level 细节 > item 细化增益
- **在记忆任务上，user 分辨率比 item 分辨率更关键**

### Finding 2: u4i3 × seed=3 Dev 几乎持平（噪声内）

**对照基线**（EXP-036 B · seed=3 × 标准 tokenization × DIN × E3 = 0.867661，guard E3 停）：
- u4i3 × seed=3 × E3 Dev = **0.868105** = Δ+0.000444（E3→E3 对比）
- 松 guard 让 u4i3 跑到 E4 = **0.868481**

**但 baseline seed=3 E3 guard 停，E4 无数据**（EXP-036 B）。无法对称对比。

**预估**：若 EXP-036 B 松 guard 跑 E4，E3 → E4 通常 +0.0003~0.0005 → E4 Dev ~0.8680~0.8682。
- u4i3 E4 0.868481 比 **估算 baseline E4** 略高 +0.0003（1× 噪声）或持平
- **判决：持平，u4i3 在 seed=3 × gap=0 无方法学增益**

### Finding 3: u4i3 机制在 gap=0 vs gap=0.05 表现分叉

**3 个数据点**：
| Config | Dev | 对比基线 Δ |
|---|---|---|
| u4i3 × gap=0.05 × E3（EXP-036 D）| 0.843743 | +0.000047 Dev / **−0.00245 LogLoss** |
| u4i3 × gap=0 × seed=42 × E4 | 0.868055 | **−0.000869 Dev** / −? LogLoss |
| u4i3 × gap=0 × seed=3 × E4 | 0.868481 | 持平 |

**解读**：
- **gap=0.05**（泛化测试）: u4i3 LogLoss 优化有效 → 可能对 LB 正向
- **gap=0**（记忆测试）: u4i3 反而轻微负向 → user_ns 粗化损伤记忆能力

**新铁律候选**（待今早 LB 验证后确认）：**Tokenization 变种应在 gap=0.05 评估，不在 gap=0 评估**（原因：u4i3 在两个领地表现不一致，gap=0.05 是真实泛化能力的代理，gap=0 只是记忆 ceiling 测试）。

---

## LB 待验证项（今日 07:00 前后上榜）

等待 4 evaluation LB 数据：
- A: seed=42 × gap=0 × DIN × E4 (EXP-036)
- B: seed=3 × gap=0 × DIN × E3 (EXP-036)
- C: seed=7 × gap=0 × DIN × E3 (EXP-036)
- D: u4i3 × gap=0.05 × E3 (EXP-036)

**关键看点**：
- 若 A LB > 0.847 → "gap=0 × DIN × seed=42" 叠加 = 新 SOTA，方法学有效
- 若 D LB ≥ 0.840 → Tokenization 在 gap=0.05 有 LB 增益（对应 LogLoss 优化）
- 若 D LB < 0.838 → Tokenization 在 LB 上无效

---

## EDA Platform · 5 次迭代全部失败（放弃）

**动因**：EXP-035 Kaggle 金牌 audit 推荐"seq domain 异构化"（方向 A），但需要平台真数据验证 seq_a/d 是否真是 non-item sequence。

**5 次 EDA 迭代时间线**：
| 版本 | commit | 问题 | 代码 |
|---|---|---|---|
| v1 | ab4251e | OOM exit 137（pa.concat_tables 全量）| ~420 行 |
| v2 | 6934f34 | Success 但 log 截断（没 flush）| +20 行 |
| v3 | 01af1eb | 100k 又 OOM | +10 行 |
| v4 | ee9affa | Per-RG streaming OK 但 log 停在 90/100 | +200 行 accumulator |
| v5 | a12cc69 | base64 gzip blob + 前置 summary 仍截断 | +40 行 |

**消耗**: 3 training slot + ~3 小时 agent 时间 **→ 产出 0 数据**。

**根本原因**：**平台 log UI 采集机制不透明**，每次失败暴露新的隐藏约束：
- OOM threshold（不知道容器限）
- stdout 非 tty 需 flush=True
- Log UI 有某种截断（可能是 top 1000 行，也可能是 container exit 时最后几秒输出丢失）

**最终判决**：**放弃 EDA 路径**（用户 2026-05-11 CEST 00:53 决定）。方向 A 实施路径保持 demo 假设，暂不经真数据校准。

**反面考虑**：
- **未来若真需要 EDA**，应先 **minimal probe**（5 行打 "hello + number" 脚本）探平台 UI 行为，再写大脚本
- 或走 **在线 EDA**（把统计塞到训练 train.py 里，用训练 log 流）
- 或走 **评估侧 EDA**（infer.py 里扫 test 集，用 evaluation log 流）—— 唯一能看 test 集的通路

**本次失误团队纪律总结**：
- 🚨 agent 在 1 个 session 犯 **5 次同类错误**（"平台约束我不知道"），每次盲试
- 🚨 花 3 小时 + 3 slot 在单一失败路径上（EXP-032 铁律 "每批 4 slot ≥3 🟢" 被全面违反）
- 🚨 违反了 EXP-035 iron law "同一假设 ≤2 次失败立刻 pivot"

---

## 下一步决策依据（等今日 LB）

### 决策矩阵

| 若 A LB | 若 D LB | 推荐 |
|---|---|---|
| ≥ 0.847（破 SOTA） | ≥ 0.840 | Tokenization 入场 + gap=0 SOTA 继续冲上沿 |
| ≥ 0.847 | < 0.838 | 只 gap=0 × DIN × seed=42 × full train，放弃 Tokenization |
| 0.843~0.846（持平） | ≥ 0.840 | Tokenization 小优化方向，配合其他方法学 bet |
| < 0.843（跌破 SOTA 均值） | 任何 | 方法学归零，全面重新战略 audit |

### 今晚 0 slot 消耗（2 u4i3 job 已停）

- Training slot 4/4 全空
- Evaluation quota 0 剩（等 CEST 06:00 重置到 4）
- **不盲起 training job**，等 LB → 决定 → 起对的 job

### 明早 CEST 07:00 后的一天节奏（目标）

1. 看 LB（决策输入）
2. 基于 LB 决定 4 evaluation quota 分配
3. 基于 LB + Dev + u4i3 方向结论起 training job（目标 2~3 个 🟢 方法学 bet）

---

## 材料

- 今日训练日志: `~/Downloads/train_log.txt` line 1~42
- EXP-036 基线: `docs/40-experiments/EXP-036-gap0-t25-seed-sweep-tokenization-2026-05-10.md`
- EDA 5 次失败 commits: `ab4251e` / `6934f34` / `01af1eb` / `ee9affa` / `a12cc69`
- EDA 脚本（保留在 repo）: `baseline/eda_platform.py`
- team-memory 新增铁律: line 81（"Dev/LB 两种能力不是精度差"·2026-05-10 CEST 16:40）

## 评估

- **u4i3 × gap=0 方向判决**: 负向 / 持平，不推荐继续
- **u4i3 × gap=0.05 方向判决**: 等 LB（EXP-036 D 数据）
- **EDA 平台路径判决**: 放弃
- **方向 A（seq domain 异构）**: 假设保持，实施路径等未来有更好 EDA 通路再验证
