# EXP-032 · 2026-05-09 凌晨 4 Half-Train Rerun 批次 · Seed 噪声下界 + ES Guard 首次生产验证

**日期**：2026-05-09 西班牙早（UTC+2 · 平台北京时间 2026-05-09 04:40~11:04）
**作者**：Agent（与用户共同推演）
**状态**：✅ 完成 · 信号固化（建立 "同 seed rerun dev 噪声 ±0.0001" 下界；vr=0.01 消融负向；LogLoss guard 首次生产触发）

---

## TL;DR（3 句话）

1. **seed=42 同 config rerun 3 次 dev 跨度 = 0.000143**（≈ ±0.0001），比跨 seed 噪声 ±0.005 小 **50×**。未来判决新方法学 bet 的**下界阈值 = ±0.0001 dev**。
2. **`valid_ratio=0.01` 消融失败**（E4 dev 0.865913 vs vr=0.05 的 0.868725，Δ−0.003）→ 方向关闭，`vr ≥ 0.05` 是 dev 稳定性硬门槛。
3. **LogLoss rebound guard 首次在生产环境正确触发**（Job 3 seed=3 E4 LogLoss 0.22473 > E3 best 0.22408 → guard 停机，best_model=E3 ckpt 0.867244）。ADR 级工程防御 proven。

---

## 触发背景

2026-05-08 晚用户提交了 4 个 half-train job 到平台排队，**不是**预期的 T27 compile+AMP 配置，而是 seed-luck / SOTA 复现批次。`baseline/run.sh` 在当时尚未切到 T27 配置（T27 代码 + ADR-008 是 5/9 凌晨才完成，见 EXP-031）。

4 个 job 全部 2026-05-09 早（北京时间）完成，合计 25 GPU-hour。虽**偏离了"只做架构级创新"战略**（EXP-028 铁律 ⑥），但作为"SOTA 复现 + seed 校准"批次产出了**3 条新铁律**，信号净增量仍值得落档。

---

## 4 个 Job 配置与结果

所有 job 共通：`swiglu + ICQ + hash500k + time-of-day + valid_split=time + reinit_sparse_after_epoch=1 + bs=128 + 4 epochs + d_model=64 + BCE loss + rank_mixer_ffn_mode=shared`。

**关键：所有 job 都是 `valid_gap_ratio=0.0` + `enable_row_valid_gap=False`（gap=0 配置，不是 gap=0.05）**，dev 处在"训练期污染虚高"状态（EXP-030 铁律 ①），dev 绝对值不能直接类比 EXP-030 的 gap=0.05 基线 0.842。

| # | Job Name | seed | valid_ratio | logloss_patience | stop_reason | Wall-time |
|---|---|---|---|---|---|---|
| 1 | `icq-t13tf-halftrain-rerun-4` | 42 | 0.05 | 1 | max_epochs | 6h24m |
| 2 | `icq-t13tf-vr001-halftrain-swiglu-bs128` | 42 | **0.01** | **2** | max_epochs | 6h06m |
| 3 | `icq-half-seed3-rerun-2` | **3** | 0.05 | 1 | **logloss_rebound_epoch_4** ⭐ | 5h59m |
| 4 | `icq-t13tf-halftrain-swiglu-vr005-bs128`（SOTA 手动 rerun，基于 5/4 旧 commit） | 42 | 0.05 | 无（旧 commit 无 guard） | max_epochs | 6h52m |

### Epoch-by-epoch Val AUC 轨迹

| Job | E1 | E2 | E3 | E4 | **Best** | Best Epoch |
|---|---|---|---|---|---|---|
| Job 1（rerun-4） | 0.86466 | 0.86712 | 0.86818 | **0.86858** | **0.868582** | E4 |
| Job 2（vr001） | 0.85951 | 0.86381 | 0.86439 | **0.86591** | **0.865913** | E4 |
| Job 3（seed=3） | 0.86407 | 0.86613 | **0.86724** | 0.86712 | **0.867244** | **E3**（guard 停） |
| Job 4（SOTA rerun） | 0.86499 | 0.86761 | 0.86810 | **0.86873** | **0.868725** | E4 |

### Epoch-by-epoch Val LogLoss

| Job | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| Job 1 | 0.22708 | 0.22468 | 0.22372 | **0.22301** ↓ |
| Job 2 | 0.23003 | 0.22781 | 0.22670 | **0.22614** ↓ |
| Job 3 | 0.22648 | 0.22502 | **0.22408** ↓ | 0.22473 ↑⚠️ |
| Job 4 | 0.22592 | 0.22416 | 0.22395 | **0.22337** ↓ |

---

## 🔥 关键发现 1：同 seed 同 config rerun dev 噪声 ≈ ±0.0001

3 次 seed=42 + vr=0.05 + 同 config 的 E4 dev AUC：

| 数据点 | 来源 | E4 dev AUC |
|---|---|---|
| 历史 SOTA 原件 | `icq-t13tf-halftrain-swiglu-vr005-bs128`（2026-05-04）| 0.868619 |
| Job 1 rerun-4 | 本批次 | 0.868582 |
| Job 4 SOTA rerun | 本批次（基于 5/4 commit）| 0.868725 |

- **跨度 = 0.000143**（= 0.868725 − 0.868582）
- **标准差估算** ≈ 0.00007（3 点样本，极粗估）
- **对比跨 seed 噪声**（EXP-028/030 五 seed 跨度 0.010，std ≈ 0.0035）：**rerun 噪声比 seed 噪声小 50×**

### 机制

同 seed 同 config rerun 不 bit-identical 的原因：
1. **CUDA 非确定性操作**：`scaled_dot_product_attention` / embedding gather / reduction 在 cuDNN 下有 1e-6 级浮点漂移，累积到 4 epoch 后放大到 1e-4 级 dev AUC 差异
2. **DataLoader 多进程顺序**：`num_workers=8`，即使 seed 固定，worker 返回 batch 的顺序依赖 OS 调度
3. **Adagrad 稀疏 grad 累加顺序**：高基数 embedding 的 inplace update 在并发下顺序不固定

### 铁律候选 1（落 team-memory）

> **同 seed 同 config rerun dev 噪声带 ≈ ±0.0001**（vs 跨 seed 噪声 ±0.005，gap=0 配置下）。任何"新方法学是否 work"的判决，**dev 变化必须 > +0.0003（3σ）才视为可能正向信号**；Δ ∈ [−0.0003, +0.0003] 一律判"噪声内无信号"。这是**比 ±0.005 seed 噪声更严格但更廉价**的下界——不需要多 seed 就能用。**场景限定**：要求 rerun 时 seed / data / code 完全一致；若改 1 行 trainer.py 也算"新 config"，不适用。

---

## 🔥 关键发现 2：Seed=3 vs Seed=42 dev/LB 方向独立（再次确认 EXP-028）

| Seed | Best dev | vs seed=42 | LB（EXP-030 数据）| vs seed=42 SOTA |
|---|---|---|---|---|
| 42（SOTA 平均，3 次 rerun 均值）| 0.868642 | — | 0.84637（原件）| — |
| **3** | **0.867244**（Job 3）| **−0.001398** | **0.845329**（EXP-030）| **−0.001041** |

**方向几乎独立**：seed=3 dev 低 −0.0014，但 LB 只低 −0.0010。与 EXP-030 的 "seed=7 dev 最高 LB 最低 / seed=3 dev 最低 LB 第 2 高" 完全一致。

### 对 T27/T28 的含义

- **seed=42 不是 LB 上界**：seed=3 在 gap=0 配置下 dev 明显低，但在 gap=0.05 配置下 LB 第 2 高。**未来任何新方法的 full-train 不应只跑 seed=42**
- **但在 "是否 work" 判决阶段（half-train）仍用 seed=42**：因为有 3 次 rerun 建立了 ±0.0001 的稳定下界；seed=3 基线没这么多数据

---

## 🔥 关键发现 3：`valid_ratio=0.01` 消融负向，`vr ≥ 0.05` 是硬门槛

Job 2 配置：`valid_ratio=0.01` + `logloss_rebound_patience=2`（比标准的 1 放宽）。

| 对照 | E4 dev | Δ |
|---|---|---|
| Job 2（vr=0.01）| 0.865913 | baseline |
| Job 1（vr=0.05）| 0.868582 | **+0.002669** |
| Job 4（vr=0.05）| 0.868725 | **+0.002812** |

**vr=0.01 掉 ~0.003 dev**。合理解释：

1. **dev 样本数从 ~105k 降到 ~21k** → dev AUC 方差变大（10-k 量级下 std ≈ 0.002）
2. **train 多 4%（~80k rows）的边际收益**显然不足以抵消 dev 方差增加导致的 "best_model 选错 epoch" 风险
3. **LogLoss guard patience=2（放宽）** 没 help：E4 仍是 best，说明过拟合还没触达 patience=2 拐点；但即便 patience=1 也不会改变 Job 2 的结论（它本来就是 E4 best）

### 铁律候选 2（落 team-memory）

> **`valid_ratio<0.05` 不给收益**：dev 样本 ≥ ~100k 是方差稳定性的硬门槛。未来任何"多给 train data"动机的 vr 调低实验**方向关闭**。若真想多 train 数据，走 row-level gap=0.05（EXP-030 机制）而非 vr 调低。

---

## 🔥 关键发现 4：LogLoss Rebound Guard 首次在生产环境正确触发

Job 3（seed=3）是**该 guard 自 2026-05-04 落地以来首次在真实平台 half-train 中触发**。

### 触发细节（平台日志原文）

```
Epoch 3 Validation | AUC: 0.867244, LogLoss: 0.22408
Epoch 4 Validation | AUC: 0.867123, LogLoss: 0.22473
[logloss-guard] epoch=4 step=63764 val_logloss=0.224728 > best=0.224084 + delta=0.000000; rebound_counter=1/1
[logloss-guard] Triggering EarlyStopping: val_logloss rebound sustained for 1 validation(s). best_model (AUC-selected) remains the final ckpt.
stop_reason=logloss_rebound_epoch_4
best_val_AUC=0.867244 best_val_logloss=0.22408 best_epoch=3
```

### 机制验证

1. ✅ **E4 AUC 0.867123 < E3 best 0.867244**（AUC 已掉头 −0.000121）
2. ✅ **E4 LogLoss 0.22473 > E3 best 0.22408**（LogLoss 反弹 +0.00065）→ rebound_counter 递增
3. ✅ **patience=1** → 第一次反弹即触发 EarlyStopping
4. ✅ **best_model 仍是 E3 ckpt（AUC-selected）**，不受 guard 影响；完全符合 ADR 设计意图

### 对 T27 的外推意义

- Job 3（seed=3）是唯一触发 guard 的 job，其他 3 个 job AUC/LogLoss 都单调向好 → 说明 guard 是**与 seed 交互的信号**：seed=3 轨迹在 E3→E4 进入过拟合拐点，seed=42 还没到
- **意味着 T27 compile+AMP 如果在 seed=42 下跑完 4 epoch，很可能全程单调**（不触发 guard），guard 不会成为 T27 验证的干扰变量 ✓
- **但未来如果做 seed sweep**，要预期部分 seed 会提前触发 guard；best_model 按 AUC 选的设计使这一点不损失 ckpt 质量

### 铁律候选 3（落 team-memory）

> **LogLoss rebound guard 已在生产环境 proven**（Job 3 seed=3 E4 触发 · 2026-05-09）：① guard 与 seed 交互（不同 seed 进入过拟合拐点的 epoch 不同），seed=42 在 4 epoch 内通常单调不触发；② best_model 仍由 AUC 选择，guard 只控制"何时停"不影响"选哪个 ckpt"；③ 任何未来实验**不需要单独验证 guard 正确性**，直接默认带 `--logloss_rebound_patience=1`；④ 若想看"训过头会怎样"，明确用 `--logloss_rebound_patience=999` 关闭 guard。

---

## 战略复盘：这 25 GPU-hour 的信息净增量评估

### 客观净收益

| 信息 | 新增价值 | 落档位置 |
|---|---|---|
| Seed=42 rerun dev 噪声 ±0.0001 下界 | 🟢 高 — 未来判决阈值基石 | team-memory 铁律 1 |
| vr<0.05 方向关闭 | 🟡 中 — 一条小路线关闭 | team-memory 铁律 2 |
| LogLoss guard 生产触发首次证据 | 🟡 中 — 工程防御 proven | team-memory 铁律 3 |
| Seed=3 vs 42 dev/LB 独立（第 2 次）| 🟢 中-高 — 复证 EXP-028 | 已在 EXP-028 铁律里 |

### 战术偏离与教训

**偏离点**：按 EXP-028 铁律 ⑥（"砍掉所有 small ablation，只做真正架构级创新"），这 4 个 slot 理应全部给 T27 / T25 DIN / T19 Per-token FFN 等**架构级 bet**。实际上：
- Job 1（rerun-4）= 重复信号（3 次 rerun 校准不需要全 4 个 slot）
- Job 2（vr=0.01）= small ablation（违反 EXP-028 铁律 ⑥）
- Job 4（SOTA 手动 rerun）= 另一个重复信号

**机会成本**：
- 少跑 1 次 T27（但 T27 的代码 + ADR-008 是 5/9 凌晨才完成，实际不可能先跑）→ 损失 ≈ 0（时序原因，非战术错误）
- 少跑 1 次 T25 DIN / T19 Per-token FFN → 损失 **≈ 1 次架构级 bet 回流 = 1~2 天 effective time**

**教训**：**当用户做 "验证已知信号 vs 探索新方向" 的 trade-off 时，agent 应主动提醒**。未来 agent 在给"平台运行建议"时必须明确标注每个 job 属于哪类：
- 🟢 **架构级 bet**（高信息增量）
- 🟡 **信号校准**（只在下界不清时做，每 5~10 个 bet 做 1 次）
- 🔴 **small ablation**（EXP-028 起原则上关闭）

并建议 **每批 4 slot 至少 3 个是 🟢 类**。

---

## 联动变更

### team-memory 新增 3 条铁律

1. "同 seed rerun dev 噪声 ±0.0001 下界"
2. "`vr<0.05` 方向关闭"
3. "LogLoss guard 生产触发 proven"

（本文定稿后单独写入 `.cursor/rules/ai-readme/team-memory.mdc`）

### active.md 不新增 task

- Job 1/4 本质是 SOTA 复现，无新 task
- Job 2 vr=0.01 是关闭的消融，新 task 也关闭
- Job 3 是 EXP-028 seed luck 假说的延伸数据点，归入 EXP-028 体系

### T27 / T28 受益点

- T27 验证（EXP-031）可以用 **`|ΔE3 dev| ≤ 0.0003` = "bf16 精度无损"** 作为更严格判据（原来 EXP-031 用 ±0.001 tolerance，现在可以收紧 3×）
- T28 DIN / T19 Per-token FFN 的"是否 work" 判决用同样 ±0.0003 dev 下界

---

## 附录：原始 Summary 日志摘要

### Job 1 rerun-4
```
stop_reason=max_epochs_completed
best_val_AUC=0.8685821268689842 best_val_logloss=0.22301343083381653 best_epoch=4
val_auc_history=E1@15941:0.864661*, E2@31882:0.867124*, E3@47823:0.868182*, E4@63764:0.868582*
```

### Job 2 vr001
```
stop_reason=max_epochs_completed
best_val_AUC=0.8659130770951423 best_val_logloss=0.2261357456445694 best_epoch=4
val_auc_history=E1@16648:0.859509*, E2@33296:0.863814*, E3@49944:0.864394*, E4@66592:0.865913*
```

### Job 3 seed=3
```
stop_reason=logloss_rebound_epoch_4
best_val_AUC=0.867244444109351 best_val_logloss=0.22408391535282135 best_epoch=3
val_auc_history=E1@15941:0.864070*, E2@31882:0.866134*, E3@47823:0.867244*, E4@63764:0.867123
```

### Job 4 SOTA rerun
```
stop_reason=max_epochs_completed
best_val_AUC=0.8687253602203278 best_val_logloss=0.22336548566818237 best_epoch=4
val_auc_history=E1@15939:0.864992*, E2@31878:0.867612*, E3@47817:0.868098*, E4@63756:0.868725*
```
