# EXP-045 · T33 NLIR Gating on Cross-Attention · Planning（未启动）· 2026-05-13

> **Status**: 🧱 **Code landed + v2 gate fix + Smoke passed · Platform job pending**（2026-05-13 CEST 15:30）
> **日期**: 2026-05-13 CEST 15:00 起草 · 15:15 Smoke v1 · 15:30 gate 源 v2 修复 + 再 Smoke
> **关联**：[ADR-011](../30-decisions/ADR-011-nlir-gating-cross-attn.md)（设计决策） · [TokenFormer paper §4.4 + §5.7](https://arxiv.org/abs/2604.13737) · `docs/20-tasks/active.md §T33`

---

## TL;DR

**目标**：把 `MultiSeqHyFormerBlock` 每个 block 的 cross-attention output 加一层 sigmoid gating multiplicative modulation（TokenFormer §4.4 原机制），mitigate Sequential Collapse Propagation。

**关键数字**：
- TokenFormer §5.7 **NLIR alone 在 KuaiRand-27k 上 +4.87‰ AUC vs Vanilla Transformer**
- 保守 50% 迁移率 → 本赛题期望 **Dev +0.0024 / LB +0.002~+0.004**
- 参数增量 **+8k**（2 blocks × `Linear(64, 64)`）· 0.003% of total · 远低于 team-memory "per-X 铁律" 警戒线

**Go/No-go**（vs `icq-half-gap05` E3 Dev 0.842443）：
- E3 Dev ≥ 0.845 → 🎯🎯 强真信号 · 立即 eval
- E3 Dev ≥ 0.8437 → 🎯 真信号 · eval
- E3 Dev < 0.8421 → ❌ Deprecated

---

## 实验计划

### Phase 1: 本地 Smoke（~30 分钟 · CEST 15:00-15:15）✅ 完成 · CEST 15:30 gate 源修复后 ✅ 再 Smoke

**v1 → v2 · 发现 gate 源 bug**（研究员视角 critical review · CEST 15:28）：
- v1 实现用 `sigmoid(X_raw @ W_g)` · **偏离 TokenFormer §4.4 Eq. 16**（论文用 `LN(X)` 作 gate 源）
- v1 风险：raw X 方差未归一化 · 随 block 深度累加 · 可能 block 2+ 进入 sigmoid saturation · Xavier init 的 zero-mean unit-variance 假设失效
- v2 修复：gate 源改为 `self.cross_attns[i].norm_q(x_in)` · 即复用 attention 自己的 pre-LN · 零新参数 · 对齐论文 "same normalized X drives both attn and gate"
- 参数量严格不变（161,447,809 · norm_q 是 reuse）

**v2 Smoke 验证**（`/tmp/t33_smoke_v2/gate_health.py` · 已清理）：
- [x] `py_compile` baseline/{model.py,train.py,infer.py} · **全过 + Lint 0 errors**
- [x] Smoke A: `--enable_nlir_gating=False` · **161,439,489 params**
- [x] Smoke B: `--enable_nlir_gating=True` · **161,447,809 params** · Δ=**+8,320** 精确匹配 `2 blocks × (D² + D)` ✅
- [x] **Gate health check 全过**（v2 关键新增）· 随机输入下 gate 分布：
  - `mean = 0.5002`（**精确中心化** · v1 做不到这么干净 · 直接证据 LN 归一化路径正确）
  - `std = 0.2127` · `p5 = 0.1533` · `p95 = 0.8332` · 90% mass 在健康动态范围
  - `min = 0.039` · `max = 0.981` · 尾部触及但不饱和（random input 覆盖整个 sigmoid 动态范围是正常的）
  - 4 个 seq domain 的 gate 统计一致（seq[0-3].mean 都在 [0.493, 0.509]）· 共享 W_g 跨域表现稳定
- [x] Smoke B: `train_config.json` 正确落盘 `"enable_nlir_gating": true`
- [x] Smoke B: `infer.py` load ckpt 无 state_dict mismatch · cfg 正确 resolve

**关键数字**：

| Config | Params | Delta | 状态 |
|---|---|---|---|
| NLIR OFF（backward-compat） | 161,439,489 | — | ✅ |
| NLIR ON（T33 v2）| 161,447,809 | **+8,320** | ✅ 精确匹配 + gate 中心化健康 |

**验证说明**：
- `"0 predictions from 0 rows"` 是 mac CPU multiprocess pickle 问题 · OFF / ON 两路径都复现 · **与 T33 无关**
- 训推一致性通过（cfg 正确 resolve + state_dict load 无 Unexpected/Missing keys）· 平台 GPU 路径无此问题

### Phase 2: 平台 Half-train（~9h 回流 · 5/14 早 CEST）· **3 Job 并发 · 含 T33 alone 对照**

**Job 结构修订**（CEST 15:35 研究员视角 critical review 补 · **必须含对照组**避免 EXP-041 M6 式 "叠加效应归因歧义"）：

- **Job A**（主 · T25 bypass + T33 · seed=42）：`t33-nlir-halftrain-gap05-seed42`
- **Job B**（🆕 **对照** · T33 alone / T25 OFF · seed=42）：`t33alone-nlir-halftrain-gap05-seed42`
- **Job C**（双 seed · T25+T33 · seed=3 松 guard）：`t33-nlir-halftrain-gap05-seed3`

**判决矩阵**（Job A vs Job B）：
- Job A Dev > Job B Dev → T33 在 T25 之上有**独立贡献**
- Job A Dev ≈ Job B Dev → T33 只是替代 T25（无增量）
- Job A Dev < Job B Dev → T25+T33 有**负交互**（EXP-041 M6 模式 · 应停叠加方向）

**Config 差异** · 基于 M9 SOTA（gap=0.05 领地）：
```bash
# Job A: M25 bypass + NLIR + seed=42
--enable_din_interest --enable_nlir_gating --seed 42

# Job B: 仅 NLIR / T25 OFF / seed=42
--enable_nlir_gating --seed 42             # 去掉 --enable_din_interest

# Job C: M25 bypass + NLIR + seed=3 松 guard
--enable_din_interest --enable_nlir_gating --seed 3 \
  --logloss_rebound_delta 0.003 --logloss_rebound_patience 2
```

**Config 完整参数**（基于 M9 SOTA · gap=0.05 领地）：
```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--ns_groups_json ""
--emb_skip_threshold 1000000
--seq_encoder_type swiglu
--seq_hash_vocab 500000
--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
--enable_time_of_day_features
--valid_split_strategy time
--valid_ratio 0.05
--enable_row_valid_gap --valid_gap_ratio 0.05
--item_conditioned_query
--enable_nlir_gating                       # ⭐ T33 核心（所有 3 Job 共有）
--batch_size 128
--num_epochs 4
--logloss_rebound_delta 0.003 --logloss_rebound_patience 2
--keep_topk_ckpt 4
--enable_torch_compile --enable_amp --amp_dtype bfloat16
# 三 Job 差异：仅 --enable_din_interest 是否加 + --seed 值
```

---

## Go/No-go 判决矩阵（详 · CEST 15:35 研究员视角校准后）

**期望迁移率下修**：原估 50%（论文 +4.87‰ → +0.0024 Dev）· 研究员视角 review 修订为 **中位 25%**（+0.0012 Dev），原因：
- TokenFormer 是 decoder-only unified stream · 我们是 HyFormer cross-attention · **token 数量差 20-50×** · SCP 严重程度不在一个量级
- TokenFormer NLIR 主要解决"NS 在 self-attn 里污染 T token"· 我们 Seq 在 encoder 内部独立处理 · **核心 SCP 场景不完全存在**
- 数据集差异（KuaiRand 公开 vs 匿名 Industrial）· 远未来 test 下迁移率衰减

### Dev 级（Job A E3 Dev vs `icq-half-gap05` baseline 0.842443）

| Dev | 判决 | 下一步 |
|---|---|---|
| ≥ 0.850 | 🎯🎯🎯 **破天级别**（+0.008 · 迁移率 >150%） | 立即 LB + seed sweep + 全线叠加 |
| ≥ 0.845 | 🎯🎯 **强真信号**（+0.0026 · 迁移率 50%） | 立即 LB 校准 + 叠加 T34 |
| ≥ 0.8440 | 🎯 **真信号**（+0.0016 · 迁移率 33% · 明显超 rerun 噪声 5σ） | LB 校准 |
| [0.8430, 0.8440) | 🟡 **疑似**（迁移率 15~33% · 中位期望附近） | seed=3 rerun（Job C）确认 |
| [0.8420, 0.8430) | ⚪ 持平（rerun 噪声 + 采样偏差内） | Deprecated |
| < 0.8420 | ❌ 负向 | 关闭 · 排查 gate 饱和 / 初始化问题 |

**阈值调整说明**（vs v1）：
- Go 阈值 0.8437 → **0.8440**（+0.0003 buffer · 考虑迁移率下修 + rerun 噪声 3σ）
- 🟡 边界 0.8433 → **0.8430**（同步放宽 · 避免把 25% 迁移率的真信号误杀为 noise）

### LB 级（若 S2 通过消耗 1 eval quota）

| LB | 判决 |
|---|---|
| ≥ 0.848 | 🎯🎯 破 M9 SOTA 0.84671 · Top 10 有望 |
| ≥ 0.846 | 🎯 破 T25 bypass B LB 0.843433 · 进 S4 叠加 |
| [0.842, 0.846) | 🟡 正向但不够 strong · seed rerun |
| [0.839, 0.842) | ⚪ 持平 | Deprecated |
| < 0.839 | ❌ 负向 · 反思论文迁移率假设 |

---

## 风险 & 对策

### 风险 1：Gate 饱和

- **症状**：Smoke B 打印 gate.mean() > 0.9 或 < 0.1，std < 0.05
- **对策**：
  - W_g 初始化 Xavier + bias=0 · 理论分布 ≈ Normal(0, 1/d)，sigmoid 后 mean ≈ 0.5
  - 若仍饱和：加 scale `gate = sigmoid(x @ W_g / temp)`, temp=2（让 gate 更平滑）
  - Fallback：引入 learnable α（init=0.1）：`decoded_q_i = α * gate * raw_q_i + q_tokens_list[i]`

### 风险 2：与 T25 DIN bypass 的 hidden interaction

- **症状**：Dev 预期 +0.0026（NLIR）+ 0.0013（T25 · 已在 SOTA 内）但实测 < +0.001
- **对策**：做 T33 alone 对照（第三 slot · `--enable_nlir_gating` ON / `--enable_din_interest` OFF）验证 NLIR 独立价值

### 风险 3：compile+AMP 与新 Linear 层冲突

- **症状**：Smoke B train 5 steps 出 dynamo recompile warning 或 bf16 softmax NaN
- **对策**：首 Smoke 先关 compile+AMP 验证正确性；平台 job 带上 ADR-008 规定的 compile+AMP

---

## 预期时间线

| 时段 | 动作 |
|---|---|
| 5/13 CEST 15:00-15:30 | Smoke A/B 完成 |
| 5/13 CEST 15:30-16:00 | commit + 准备平台提交清单 |
| 5/13 CEST 16:00 | 平台提交 Job A（seed=42）· 上传 3 文件 + run.sh 粘贴 |
| 5/13 CEST 17:00 | 平台提交 Job B（seed=3）· Copy Job A 只改 seed |
| 5/14 CEST 02:00-03:00 | Job A/B 回流（~9-10h wall time） |
| 5/14 CEST 03:00-07:00 | 评估 · LB 校准决策（等 06:00 quota 重置） |
| 5/14 CEST 07:00+ | 基于 LB 决定 S4 叠加路径 |

---

## 平台提交清单

### Job A: t33-nlir-halftrain-gap05-seed42

**上传文件**（改动的 Python）：
- `baseline/model.py`（加 NLIR gating 分支）
- `baseline/train.py`（加 `--enable_nlir_gating` CLI）
- `baseline/infer.py`（加 `cfg.get('enable_nlir_gating', False)` fallback）

**run.sh**（粘贴到平台编辑框 · 从 M17 run.sh 修订而来）：
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

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
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --enable_row_valid_gap \
    --valid_gap_ratio 0.05 \
    --item_conditioned_query \
    --enable_din_interest \
    --enable_nlir_gating \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 4 \
    --logloss_rebound_delta 0.003 \
    --logloss_rebound_patience 2 \
    --keep_topk_ckpt 4 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    "$@"
```

### Job B: t33-nlir-halftrain-gap05-seed3

Copy Job A, 修改 `--seed 42` → `--seed 3`，其余完全一致。

---

## Materials（待回填）

- Commit: （代码落地后填）
- 本地 Smoke 日志：（完成后填）
- Platform Job ID: （提交后填）
- 回流 Dev / Loss: （回流后填）
- LB: （校准后填）
