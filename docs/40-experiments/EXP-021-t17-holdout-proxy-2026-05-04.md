# EXP-021 · T17 Holdout Dual-Valid Proxy（ADR-005 方案 B · S3）

**日期**：2026-05-04 夜
**作者**：Agent
**状态**：🟡 Proxy 完成（**近零 gap，不足以 Accept 也不足以 Reject ADR-005**）→ 等 S4 half-train 数据

---

## 背景

EXP-018 证明本项目 valid→LB gap 系统性稳定在 0.021~0.030（7 次提交全部），典型过拟合现象：E5 起 `AUC 涨 + LogLoss 反弹`。ADR-005 方案 B 提出在尾部切出两路：`[train][dev (5%)][holdout (5%)]`，其中 holdout 训练期间完全不 forward，只做 epoch 末独立评估，作为比 dev 更"未来"的 LB 代理信号。

**S3 的验证问题**：在 proxy 尺度（128 RG + 1500 steps）上，`dev AUC − holdout AUC` 是否已经能观测到 gap？gap 越接近 EXP-018 那 0.023 越好。

---

## Proxy 配置

**Job Name**：`t17-holdout-s3-t13tf-proxy-swiglu-vr005-ho005-rg128x32-s1500`

```text
--valid_split_strategy time
--valid_ratio 0.05
--holdout_ratio 0.05          # ← ADR-005 新增
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--batch_size 128
--limit_train_rgs 128 --limit_valid_rgs 32 --max_train_steps 1500
--logloss_rebound_patience 0  # proxy scale 不可靠，禁用
```

Baseline 对照（不同是 `--holdout_ratio 0.05` 这一项）：EXP-020 同时刻跑的 `t16-additive-a01-proxy`，但这次是纯 t13tf（无 additive injection）以排除 T16 变量，对照是历史 `t13tf proxy` 的 0.8375~0.8385 范围。

---

## Proxy 结果

### 核心读数（step 1500，end-of-proxy）

| 指标 | dev (valid_loader) | holdout (holdout_loader) | gap = dev − holdout |
|---|---|---|---|
| AUC | 0.839843 | **0.839957** | **−0.000114** |
| LogLoss | 0.242461 | 0.241341 | +0.001120 |

**完整日志线索**：

```
2026-05-05 06:47:53 - Max-step Validation | AUC: 0.839843, LogLoss: 0.242461
2026-05-05 06:47:55 - Start Evaluation (PCVRHyFormer) - holdout
2026-05-05 06:49:43 - [ADR-005] Epoch 1 step 1500 holdout AUC: 0.839957,
   LogLoss: 0.241341 (valid AUC: 0.839843, LogLoss: 0.242461;
   gap_auc=valid-holdout=-0.000114)
```

holdout eval 耗时 1:48（880 batch），和 dev eval 耗时相当，预算影响微乎其微。

### 横向对照（proxy 尺度下的 dev-only 历史）

| Proxy Job | dev AUC @ step1500 | Δ |
|---|---|---|
| t13tf proxy（hash500k + timefeat，无 holdout） | ~0.8376 (EXP-015 附近) | baseline |
| **t17-holdout proxy（本次，dev）** | **0.839843** | +0.00224 |
| t16-additive proxy（EXP-020，dev） | 0.839564 | +0.00188 |

dev AUC +0.002 与 t16-additive 接近，不过本次**未开 target injection**，说明这个 +0.002 噪声是同一个 platform run 里的 stochastic variance 而非真正由 holdout split 带来；方案 B 对 dev AUC 本身没有影响（符合设计——holdout 只读取、不参与 train forward，理论上 dev AUC 应该完全等同于无 holdout 的 baseline）。

---

## 判读

按 ADR-005 S3 阶段事先约定的判读表：

| `gap_auc = dev_auc - holdout_auc` | 含义 | 本次 |
|---|---|---|
| > 0.01 | holdout 对 LB 有强预测能力 | — |
| 0.003 ~ 0.01 | 弱信号但方向正确 | — |
| **−0.003 ~ 0.003** | dev 与 holdout 同源，holdout 无独立预测力 | ✅ 本次 **−0.000114** |
| < −0.003 | 异常，切分反了 | — |

**落在"近零 gap"档**：方案 B 在 proxy 尺度上**既没有被证实、也没有被证伪**。

---

## 机制分析：为什么 proxy 尺度下 gap ≈ 0

### 1. 过拟合还没发生，gap 本来就不该大

EXP-018 显示 `t13tf-loss-full` 的 valid→LB gap 0.021~0.030 出现在 **10 epoch full train 的 E4~E9**，典型"E5 后 AUC 涨 LogLoss 反弹"的过拟合拐点之后；在欠拟合区（E1~E2）valid 和 LB 应该接近。

本次 proxy = 1500 steps ≈ 1 epoch 的 1/14，模型远未进入过拟合区。**dev 和 holdout 都还在"学共同通用规律"阶段**，分歧不该大。

**推论**：proxy 尺度不是方案 B 的决定性考场。

### 2. holdout 和 dev 的时间距离在 proxy 尺度下被放大不足

`valid_ratio=0.05 holdout_ratio=0.05` 于 1000 RG：dev ≈ 最后 50 RG，holdout ≈ 紧跟其后的 50 RG。在 time-split 下相邻 50 RG 时间窗大概就几小时到一两天；远小于 train→test 的真实时间 gap（官方未明示但根据 EXP-018 数据推断至少周级）。

若 LB gap 的根源是"时间跳跃",那 5% + 5% 的内部切分只捕捉了**跳跃的一小段**,不足以把那 0.023 全额反映出来。

### 3. dev 和 holdout 的 LogLoss 方向一致

dev LogLoss 0.2425 而 holdout LogLoss 更低 0.2413，holdout "看上去更好"。这说明整个 `[dev + holdout]` 区段在 proxy 尺度下对模型是**同质分布的一块数据**，没有显著分布漂移。

---

## 决策与下一步

### ADR-005 状态维持 Proposed

本次 S3 **不能 Accept**（没证实 holdout 代理能力），**也不足以 Reject**（proxy 尺度本身无法验证过拟合区行为）。

### S4 Half-Train 搭车（强烈推荐，零额外 training quota）

S4 是真正考察 ADR-005 方案 B 的关键实验。策略：**下一个 half-train job 默认加 `--holdout_ratio 0.05`**，零成本附带采集 S4 数据。

**判据**：在 2~3 epoch full-data half-train 里观察每个 epoch 的 `(dev_auc, holdout_auc, dev_logloss, holdout_logloss)` 四指标轨迹：

| 观察信号 | ADR-005 方案 B 状态 | 下一步 |
|---|---|---|
| E1~E3 中后期出现 `dev_auc 继续涨 + holdout_auc 滞涨/下降` | ✅ Accept：方案 B 可作 LB 代理 | 改用 holdout_auc 选 best_model；S5 LB 验证 |
| E1~E3 dev/holdout gap 全程 < 0.003 | ❌ Reject：方案 B 力度不足 | 切到方案 C（train/valid 之间留时间空窗）或 D |
| dev LogLoss 反弹 **早于** holdout LogLoss 反弹 | ⚠️ 观察性 | 说明 dev 比 holdout 更容易过拟合，间接证实 dev 是"近邻"；有指导价值 |

### 其他选项（不推荐）

- **放大 `--holdout_ratio` 到 0.1 再跑 proxy**：占 training quota，且 proxy 尺度的欠拟合问题仍在，ROI 低
- **立刻改方案 C**：过早，需先看 S4 数据

---

## 文件产出

- `docs/30-decisions/ADR-005-valid-split-redesign.md`: Verification 节追加 S3 实测
- `docs/20-tasks/active.md`: T17 节点"下一步"改为 "S4 搭车"
- `.cursor/rules/ai-readme/team-memory.mdc`: 新增一条铁律——proxy 尺度下 dev/holdout gap≈0 不代表方案 B 无效
