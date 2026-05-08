# EXP-031 · T27 torch.compile + AMP bf16 首次 half-train（`t27-compile-amp-halftrain-gap05`）

**日期**：2026-05-09（西班牙时间早 · 待平台提交）
**作者**：Agent（与用户共同推演）
**状态**：🚧 Planned · 实验 run.sh / ADR-008 已 ready，待平台启动
**关联**：T27（tasks/active.md）· ADR-008（torch.compile + AMP 默认化）

---

## TL;DR

**一次 training quota 同时回答三个问题：**

1. **正确性**：bf16 autocast 是否保持 Dev AUC 与 fp32 baseline 一致（±0.001）
2. **加速比**：compile+AMP 能把 epoch wall-time 压到 baseline 的多少（目标 ≤0.67×，即 ≥1.5× 加速）
3. **扩容可行性**：显存占用是否降至 ≤13GB（19GB 卡的 70%），为后续 `d_model=96` / `batch_size=192` 留空间

**Go/No-go 结果将决定是否把 `--enable_torch_compile --enable_amp` 设为后续所有实验的默认配置。**

---

## Baseline 锚点（来自 EXP-030）

| 指标 | `icq-half-gap05`（EXP-030 baseline） |
|---|---|
| Dev AUC @ E1 | 0.837939 |
| Dev AUC @ E2 | 0.841124 |
| Dev AUC @ E3 best | **0.842443** |
| LB | **0.839223** |
| Dev→LB gap | +0.0032（极可信） |
| Epoch wall-time (estimate) | ~100 min/epoch × 3 = ~5h（half-train 用 valid_ratio=0.05 + gap=0.05 少用 ~10% 数据） |
| Peak GPU mem | unknown, ≤ 19GB（历史运行未 OOM） |

> **重要**：由于 `gap=0.05 + valid_ratio=0.05 + bs=128` 配置 seed=42 rerun 的 dev 稳定性 ±0.0001（EXP-028 same-config rerun），**Dev AUC 在这个配置下是确定性的诊断信号**。bf16 autocast 若引入精度损失，E3 Dev 必定 <0.8414。

---

## 配置

完全复刻 baseline，**仅加三个 flag**：

```bash
--enable_torch_compile
--enable_amp
--amp_dtype bfloat16
```

其余配置（完整 `run.sh` 见 `baseline/run.sh`）：

- ICQ: `--item_conditioned_query`
- seq encoder: `--seq_encoder_type swiglu`
- hash: `--seq_hash_vocab 500000`
- 时间特征: `--enable_time_of_day_features`
- Valid split: `--valid_split_strategy time --valid_ratio 0.05`
- **Row-level gap**: `--enable_row_valid_gap --valid_gap_ratio 0.05`
- Seed: `--seed 42`
- Batch / Epoch: `--batch_size 128 --num_epochs 3`

---

## 观测指标

### 必看（决定 Go/No-go）

| 指标 | 测量方式 | 通过条件 |
|---|---|---|
| E3 Dev AUC | 训练日志 | ∈ [0.8414, 0.8434] |
| 每 epoch wall-time | 日志首尾时间戳 | ≤ 67 min（baseline ~100 min/epoch） |
| 训练稳定性 | 日志扫 `dynamo` / `Traceback` / `OOM` | 0 hit |
| 推理兼容性 | best_model publish → evaluation 出分 | 出分正常、LB ≥ 0.838（±0.001 of baseline LB 0.839） |

### 辅助诊断（若异常时参考）

| 指标 | 通过条件（期望） | 异常含义 |
|---|---|---|
| E1 Dev AUC | ~0.8379 ±0.001 | 若显著低：bf16 在 embedding 或 attention 破精度 |
| E1 → E3 Dev 单调 | 单调升 | 若 E2/E3 反弹：autocast 与 sparse embedding 反传交互异常 |
| Peak GPU mem | ≤ 13GB（~68%） | 若 >16GB：AMP 无效或 compile 存额外副本 |
| Epoch 2 vs Epoch 1 wall-time | ≤1.1× | 若 >1.3×：recompile 在 sparse reinit 后成为 bottleneck |

---

## 假设与理论预测

### 数值正确性（bf16）

- **BCE loss 对 bf16 稳定**：σ(logit) 数值范围 [0, 1]，bf16 mantissa 7-bit 足够表示 1e-3 粒度的概率差，BCE 梯度不会 underflow。
- **ICQ 的 cross-attention**：Q·K^T / √d_k 最终过 softmax。bf16 dynamic range 足够（±1e38），**理论上 softmax 无 overflow**。但极端 logit（如初始化期的极大值）可能饱和 → PyTorch 2.7 对 `scaled_dot_product_attention` 内部有 fp32 upcast 兜底。
- **Sparse embedding 存 fp32**：按设计 sparse Adagrad 只对 embedding 走 fp32 不进 autocast，这是 `torch.autocast` 默认行为。

**结论**：**Dev AUC 预期偏差 ≤0.0003**（远小于 ±0.001 tolerance）。若超出，第一反应去查 `MultiSeqHyFormerBlock` 内部的自定义 attention 是否把 softmax 算到了 bf16（需 `scores.float().softmax(-1).to(scores.dtype)`）。

### 加速比（compile + bf16）

- **DECEM 经验**：~2× 加速（他 GPU 未知但大概率 A100 40GB）
- **本项目平台 GPU**：19GB（推测 T4 或 V100 16GB + overcommit，或 A10 24GB）
- **保守预期**：bf16 alone **~1.3×**，compile alone **~1.2×**，叠加 **~1.5~1.8×**
- **墙钟目标**：100min/epoch → **~55~67 min/epoch**；3 epoch 从 5h 降到 **~3h**

### 扩容空间

- **bf16 activation 占用减半**：预估显存从 ~14GB 降到 **~10GB**
- **若落入 10~12GB**：下个实验可以直接上 `d_model=96`（参数量 ~1.5×，显存 ~13~14GB）或 `batch_size=192`（throughput ~1.3×，显存 ~14GB）
- **若落入 >13GB**：AMP 没生效 / 有额外副本，下一步 profile

---

## Go/No-go 决策树

```
T27 half-train 完成后:

├── E3 Dev ∈ [0.8414, 0.8434] AND wall-time ≤ 0.67× baseline AND no crash
│   └── ✅ Accept ADR-008
│       └── 后续所有 half-train / full-train 默认加 compile+AMP
│       └── 下一实验：T28 候选 architecture-level bet
│           ├── 候选 A: DIN block-integrated（target attention 进 HyFormer block 内）
│           └── 候选 B: multi-task click+conversion joint loss
│
├── E3 Dev < 0.8414（精度损失 >0.001）
│   └── ⚠️ 机制排查：
│       ├── 首看 attention softmax dtype
│       ├── 若修复后仍 diff 大 → 降级为 `--enable_amp` only（不 compile）
│       └── 若 AMP 单独就降精度 → 改用 fp16 + GradScaler（复杂）或回退 fp32
│
├── wall-time ≥ 0.8× baseline（加速不足）
│   └── ⚠️ Profile：
│       ├── recompile 是否频繁（epoch 间 time 比大）
│       ├── sparse embedding lookup 是否成了新瓶颈
│       └── 降级到 AMP only（丢 compile）作为稳妥版
│
├── Traceback / OOM / dynamo error
│   └── ❌ Defer ADR-008，修复后重跑；不阻塞其他实验（其他实验继续用 fp32）
│
└── LB < 0.837（推理异常）
    └── ❌ 关键红线：ckpt 格式不兼容 infer.py
        └── 检查 `model.pt` 是否含 `_orig_mod.` 前缀（应该没有）
        └── 若有：立刻修 `_save_step_checkpoint` 走 `_raw_model`（已做，但需二次确认）
```

---

## 风险清单

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| bf16 attention softmax 精度不够 | 30% | Dev -0.001~0.003 | 立即 cast softmax 回 fp32 |
| compile recompile 频繁 | 20% | 加速 <1.3× | 降级到 AMP only |
| platform infer.py 与 raw_model state_dict 不兼容 | 5%（已 local verify） | LB 出分异常 | 回退 T27，重新 evaluate |
| Sparse embedding reinit 后 compile 崩 | 10% | 训练中途 crash | `suppress_errors=True` 已开启，会降级到 eager |
| DECEM post 本身不适用本 GPU | 15% | 加速比差 | 至少 AMP 有效，保留 50% 收益 |

---

## 后续动作

**若 Go**：
1. Merge ADR-008 → Accepted
2. 更新 `CLAUDE.md` 训练示例代码，加 compile+AMP flag
3. 更新 `docs/20-tasks/active.md`，T27 → [x]，开启 T28（选 A 或 B 中更高 ROI 的 architecture bet）
4. 更新 team-memory：铁律新增"compile+AMP 为默认；如遇新 module 不兼容，优先关 compile 保 AMP"

**若 No-go（部分）**：
1. 降级配置（AMP only 或全退）
2. ADR-008 改 Rejected 或 Deferred，记录精确失败原因
3. T27 继续 Open，写 Next Action

---

## 备注 · 为什么不用 proxy smoke

**初版计划是 `limit_train_rgs=50 --max_train_steps=500` 的 proxy smoke**，只回答 "crash 不 crash"。但在 2026-05-09 00:10 的"研究员视角复盘"里否定了这条路径，原因：

1. **平台 training quota 稀缺**（4/天），proxy 烧 1 次只换"不 crash" 一个 bit 的信息。
2. Half-train 同样 1 quota，**换三个信息单元**（正确性 + 加速比 + 扩容），信息密度 3×。
3. `--enable_torch_compile` / `--enable_amp` 的失败模式大多是**累积型**（bf16 精度漂移、recompile 抖动），**proxy 500 steps 看不出来，必须跑到至少 1 完整 epoch**。

这是本项目"从工程师心态切回研究员心态"的第一个实践。
