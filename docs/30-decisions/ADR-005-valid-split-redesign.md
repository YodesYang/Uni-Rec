# ADR-005 · Valid Split Strategy Redesign（降低 valid→LB gap）

> **Status**: Proposed
> **Date**: 2026-05-04
> **Related Tasks**: T17（计划新增到 `docs/20-tasks/active.md`）
> **Related Experiments**: EXP-018（7 次历史提交的 valid→LB gap 扫描）
> **Depends On**: ADR-001（当前 time-split 实现）

---

## Context（背景与触发原因）

### 核心触发：gap 系统性、可重复、且正在扩大

2026-05-04 晚 `t13tf-loss-full-swiglu-vr005-bs128` 评估结果：**Valid AUC 0.868989 → LB AUC 0.838707，gap = 0.0303**，为历史最大。

7 次历史提交 gap 扫描（详见 EXP-018 §历史 Gap 扫描）：

| 提交 | Local Valid AUC | LB AUC | Gap |
|---|---|---|---|
| transformer epoch-1 错 ckpt | 0.8617 | 0.834015 | 0.0277 |
| swiglu baseline (filename) | 0.8650 | 0.84381 | 0.0212 |
| swiglu + time + vr=0.05 | 0.865884 | 0.844059 | 0.0218 |
| swiglu + d_model=128 | 0.864881 | 0.836944 | 0.0279 |
| swiglu + dropout=0.05 | 0.864563 | 0.83977 | 0.0248 |
| t13tf-full E4 | 0.868495 | **0.844956** | 0.0235 |
| **t13tf-loss-full E9** | 0.868989 | **0.838707** | **0.0303** |

**结论**：gap 稳定在 0.021~0.030，**非单点事故**；越往后训越大。

### 机制分析

当前 valid split（ADR-001 `time` 模式）的做法：**按 RG 级 `max(timestamp)` 排序，取尾部 `valid_ratio` 作为 valid**。在本项目数据上这意味着：

- train 覆盖 baseline_train 时段前 95%
- valid 覆盖该时段最后 5%（约 1~2 天的数据）
- test 则是**比 train 更远未来**（官方答复："test timestamps do not overlap with training set and are from a later time period"）

问题：
1. **valid 与 train 在时间上是"紧邻"的**（最后 5% 与前 95% 几乎连续）→ 模型在 train 上学的局部时间模式（hour/weekday/节假日/某个 campaign 正在跑）**对 valid 几乎完全有效** → valid AUC 偏高
2. **test 与 train 是"跳跃"的**（test 是更远未来，有新 campaign 上线 / 旧 campaign 下线 / 节日过去 / 创意改版）→ train 学的局部模式**对 test 失效** → LB AUC 大幅下跌
3. 这是典型的 **valid 作为"train 近未来延续"、test 作为"train 远未来外推"** 的 covariate shift 场景。现有 valid **高估**了模型对 test 的泛化能力

### 与 ADR-006（即将提出的 LogLoss rebound guard 已落地 trainer.py）的关系

LogLoss rebound guard 解决的是 **"在当前 valid 定义下训过头"**，但**它无法修正 valid 本身的定义偏差**。即使完美在 LogLoss 拐点停机（比如 E4），gap 仍有 0.0235。

→ 两条路径是**正交的**，需要同时推进。

---

## Decision（最终决定做什么）

**本 ADR 暂时 Proposed，尚未 Accept**。先列出候选方案、各自取舍、实施成本；等初步探索性实验（低成本 proxy）后再收敛 Accept 哪个。

### 候选方案

#### 方案 A：尾部再切尾部（Triple-split）

把当前 5% valid 进一步切成：
- `valid_holdout_far`（最后 1%）：用于选 best_model、EarlyStopping、LB 代理
- `valid_holdout_near`（倒数第 2~5%）：训练时可见，作为 pseudo-train（gradient 可以流过去）
- 或者：直接把 valid 切成最后 1% + 把前 4% **撤回 train**（更激进）

**优点**：实现最简单，~30 行代码；逻辑上让 valid 距 train 更远 → 更接近 test 的"外推"性质  
**缺点**：valid 太小（1% ≈ 20k 样本）→ AUC 方差变大 → EarlyStopping 不稳；"撤回 train" 等于减少 valid 防过拟合信号

#### 方案 B：双 valid（dev + holdout）

- `dev`（尾部 5% 紧邻 train）：供 EarlyStopping / checkpoint 写入使用
- `holdout`（**预留的最后 N%**，训练期间**完全不访问**，只在训完一次性算指标）：作为 LB 代理

**优点**：holdout 是真正的 LB 代理，不被 EarlyStopping 污染，gap 可量化；保留 dev 规模，防 EarlyStopping 不稳  
**缺点**：需要重新切一次数据 / 多一次 forward pass；holdout 不参与 train 会损失 ~5% 样本

#### 方案 C：模拟 test-time gap（留空窗）

- train: `[T0, T_end - Δ_gap - Δ_valid]`
- valid: `[T_end - Δ_valid, T_end]`
- **gap**: `[T_end - Δ_gap - Δ_valid, T_end - Δ_valid]` **完全丢弃**（不在 train 不在 valid）

这模拟了 test 相对 train 的"跳跃"性质——test 相对 train_end 可能有 几天到一周 的 gap，valid 也应该有相似的 gap。

**优点**：直接模仿 LB 的 covariate shift 特性，最接近真实 LB 信号；如果 gap 和 test 的实际 gap 匹配，valid AUC 会极度接近 LB AUC  
**缺点**：需要先估算 test 相对 baseline_train_end 的实际时间差（官方只说"later time period"，未说具体多远）；Δ_gap 是超参，估错会浪费样本

#### 方案 D：对 valid 做有意加噪 / 特征 mask（不推荐）

- 故意把 valid 里部分稀有 item embedding 做 drop-out，模拟"test 里遇到训练见过但 mature 不够的 item"  
- **不推荐**：加噪和 test 的真实分布偏差没有严格对应关系，属于"瞎猜 test"

### 初步倾向

**先做 B（dev + holdout）**：实现成本低、能立刻拿到 gap 量化指标；而后考虑 C（留空窗）——因为 C 的本质是"让 valid 长得更像 test"，但需要先用 B 的 holdout 信号校准 Δ_gap。

A 作为 **fallback**（如果 B 发现 dev 训出来和 holdout gap 稳定很小，那就直接尾部切小 valid，简化实现）。

---

## Alternatives Considered（其他备选）

- **D（noise injection）** ：不可控，不纳入
- **K-fold / cross-validation**：训练开销 N 倍，且和时序场景冲突（k-fold 打乱时间），放弃
- **直接用 LB 当 valid**：4 次/天额度，且平台 evaluation 需 ~15min 延迟 + ~5min 训练→评估路径，不现实
- **不动 valid，靠正则**（weight decay / dropout / sparse reinit）：ADR-003 已探索 dropout=0.05 反而让 LB 更差（0.83977 < 0.84381），正则无法消除 valid leakage 本身

---

## Consequences（后果与副作用）

### 正向

- 如果 B/C 落地成功，**所有未来实验的决策信号更接近 LB**；不再被 valid AUC 误导着追 local AUC
- 给 T16（ADR-004 target-item 硬交互）及后续所有模型改动提供更可信的 local 信号

### 负向 / 风险

- B 需要多留一份 holdout → 训练数据减少 5% 左右，理论上 best LB 会小幅下降（但 gap 量化价值远超这个损失）
- C 的 Δ_gap 估计错了会变成"valid 太难"，让所有模型的 local AUC 看起来都很差 → 选不出好模型
- 所有方案都需要**重新跑一遍现有 best baseline（t13tf-full）** 来校准新 valid 下的 "baseline 分数"，这是一次额外 full train 成本（~10h + 1 quota）

### 不在本 ADR 覆盖范围

- **不动** `trainer.py::_save_step_checkpoint`（仍按 AUC 选 best_model）——只改 valid **定义**，不改 checkpoint 选择逻辑
- **不动** ADR-004 (T16) 实现——Target-item injection 与 valid split 是两条独立战线

---

## Verification（验证计划）

1. **S0 Static**：`py_compile` + 在 dataset.py 里的 holdout 切分逻辑写 docstring 说明
2. **S1 Local Smoke**：demo 数据跑 1 epoch，确认 holdout 被正确切出、**在训练 forward 中完全不出现**、EarlyStopping 仍作用在 dev 上
3. **S2 Local Sanity**：跑 2 个 shuffled seed，看 dev AUC / holdout AUC 二者之间是否稳定接近（差 <0.005 则说明在 demo 上仍重合，需要进一步验证在 full data 是否有 gap）
4. **S3 Fast Proxy**（first real test）：在 full data 上跑 `limit_train_rgs=64, limit_valid_rgs=16, max_train_steps=1500`，对比 `dev AUC` vs `holdout AUC`，gap 如果 > 0.01 → 证明方案 B 能暴露 valid→LB gap，进入 S4；否则回到 design
5. **S4 Half Train**（2~3 epoch 全数据）：观察每个 epoch 的 `dev AUC / holdout AUC / dev LogLoss / holdout LogLoss` 四个指标，看 holdout AUC 是否更接近 LB 历史分数（~0.844）
6. **S5 Full Train + LB**：若 S4 通过，用 1 次 evaluation quota 提交，验证 holdout AUC ≈ LB AUC（理想情况下差距 < 0.005）

### S0 / S1 实测结果（2026-05-04 晚）

**S0 Static** ✅
- `python3 -m py_compile baseline/{dataset,trainer,train,infer}.py` → exit 0
- `ReadLints` 仅报 basedpyright 环境 import warning（与改动无关）

**S1 Local Smoke** ✅（`./data/demo` · 10 RG · `--valid_ratio 0.2 --holdout_ratio 0.2`）

关键验证点逐项命中：

| 验证点 | 期望 | 实测日志 |
|---|---|---|
| 三路切分正确 | time-sorted 下 `[train][dev][holdout]` 三路 disjoint | `Row Group split (time): 8 train (800 rows), 1 valid (100 rows), 2 holdout (1100 rows)`（10 RG 大小不对称，数学分配正确） |
| holdout_loader 被 attach | 启动日志提示 | `[ADR-005] holdout_loader attached: holdout AUC/LogLoss will be logged every epoch. holdout never drives best_model selection, EarlyStopping, or LogLoss rebound guard.` |
| 训练 forward 不触及 holdout | train_loader 仅用 train RG | train `PCVRParquetDataset: 800 rows ... shuffle=True` vs holdout `PCVRParquetDataset: 1100 rows ... shuffle=False`（只在 eval 流程用） |
| epoch 末 holdout eval 独立跑 | `Start Evaluation - holdout` + 同步指标 | `[ADR-005] Epoch 1 step 5 holdout AUC: 0.631605, LogLoss: 0.390470 (valid AUC: 0.576313, LogLoss: 0.348653; gap_auc=valid-holdout=-0.055293)` |
| best_model 仍按 valid AUC 选 | `best_val_AUC` 是 valid 的值 | `training_summary.json::best_val_auc = 0.576313`（不是 holdout 的 0.631605） |
| backward-compat：不传 `--holdout_ratio` 等价于旧行为 | 无 ADR-005 日志、2 路切分 | `Row Group split (time): 9 train (900 rows), 2 valid (1100 rows)` · 全程无 `[ADR-005]` 日志 |
| Guard：`holdout_ratio>0` 必须配 `valid_split_strategy='time'` | 早期 ValueError | `ValueError: holdout_ratio > 0 requires valid_split_strategy='time' (ADR-005)` |

**demo 数据重叠告警是已知现象**：日志 `[ADR-005] dev/holdout timestamp ranges overlap ... holdout signal may be contaminated` —— demo 只有 10 RG 且 timestamp 高度重叠，属于已知 demo 特性；真实数据 1000+ RG 时 time-sorted 的相邻 RG 时间边界清晰分离，告警不会触发。已在日志里保留这个 warning 以便 full data 上做 sanity check。

**S2 Local Sanity 暂缓**：demo 只有 1~2 个 valid RG，换 seed 也不会带来有意义的 dev/holdout 分布差异。直接跳 S3（full data fast proxy）。

---

## 决策权重（待评估后填 Accept）

待 S3 数据回流后补充：
- 方案 B 的代码成本：约 ~80~120 行，新增 `--holdout_ratio` CLI flag、dataset 新增 `holdout_rg_indices`、trainer 新增 `holdout_loader` + `evaluate_holdout()` 接口
- Δ_gap 估算（方案 C）：如果能从官方答复或 schema 里推出 test 时间区间上限 - train 时间上限 的具体值，C 可以较低成本落地；否则保守方案是 B 优先

---

## 相关文件（预计改动）

- `baseline/dataset.py`: `get_pcvr_data` 增加 `holdout_ratio` 参数，切分逻辑改造
- `baseline/trainer.py`: `__init__` 增加 `holdout_loader` 可选参数，`evaluate` 末尾多算一次 holdout
- `baseline/train.py`: 增加 `--holdout_ratio` CLI flag（默认 0.0 = 兼容老行为）
- `docs/20-tasks/active.md`: 新增 T17 节点
