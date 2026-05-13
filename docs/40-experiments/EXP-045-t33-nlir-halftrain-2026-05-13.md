# EXP-045 · T33 NLIR Gating on Cross-Attention · Planning（未启动）· 2026-05-13

> **Status**: 🧱 **Code landed + Smoke passed · Platform job pending**（2026-05-13 CEST 15:15）
> **日期**: 2026-05-13 CEST 15:00 起草 · 15:15 Smoke 通过
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

### Phase 1: 本地 Smoke（~30 分钟 · CEST 15:00-15:15）✅ 完成

- [x] `py_compile` baseline/{model.py,train.py,infer.py} · **全过 + Lint 0 errors**
- [x] Smoke A: `--enable_nlir_gating=False` · demo 2 step train · **161,439,489 params**（demo 数据 baseline，不同于平台 ~353M）
- [x] Smoke B: `--enable_nlir_gating=True` · demo 2 step train · **161,447,809 params** · Δ = **+8,320** 精确等于 `2 blocks × (D² + D) = 2 × (64² + 64) = 8,320` ✅
- [x] Smoke B: `train_config.json` 正确落盘 `"enable_nlir_gating": true`
- [x] Smoke B: `infer.py` load ckpt 无 state_dict mismatch（log 显示 `'enable_nlir_gating': True` 正确进入 `Building PCVRHyFormer with cfg`）
- [x] **Backward-compat verified**: OFF 路径参数量与 ON 参数量差值严格 = 新增 Linear 理论值，未污染其他模块
- [ ] （未做 · 不阻塞平台启动）Gate 初始化健康检查 · 平台 Job A E1 回流时在日志里看 gate.mean

**关键数字**：

| Config | Params | Delta | 状态 |
|---|---|---|---|
| NLIR OFF（backward-compat） | 161,439,489 | — | ✅ |
| NLIR ON（T33）| 161,447,809 | **+8,320** | ✅ 精确匹配理论值 |

**验证说明**：
- `"0 predictions from 0 rows"` 是 mac CPU multiprocess pickle 问题 · OFF / ON 两路径都复现 · **与 T33 无关**
- 训推一致性通过（cfg 正确 resolve + state_dict load 无 Unexpected/Missing keys）· 平台 GPU 路径无此问题

### Phase 2: 平台 Half-train（~9h 回流 · 5/14 早 CEST）

**Job A（主 · seed=42）**：
```
Job Name: t33-nlir-halftrain-gap05-seed42
改动文件上传：baseline/model.py, baseline/train.py, baseline/infer.py
run.sh 差异：在 M9 config 上加 `--enable_nlir_gating`
```

**Job B（双 seed 验证 · seed=3 · 松 guard）**：
```
Job Name: t33-nlir-halftrain-gap05-seed3
Config: 同 A + --seed 3
```

**Config 参数**（基于当前 run.sh M17 的去 seq_d=768 版本，即 M9 基线）：
```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--ns_groups_json ""
--emb_skip_threshold 1000000
--seq_encoder_type swiglu
--seq_hash_vocab 500000
--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"  # M9 默认 · 不加长
--enable_time_of_day_features
--valid_split_strategy time
--valid_ratio 0.05
--enable_row_valid_gap --valid_gap_ratio 0.05          # gap=0.05 领地
--item_conditioned_query
--enable_din_interest                                   # T25 bypass · 保留
--enable_nlir_gating                                    # ⭐ T33 新增
--seed 42
--batch_size 128
--num_epochs 4
--logloss_rebound_delta 0.003 --logloss_rebound_patience 2
--keep_topk_ckpt 4
--enable_torch_compile --enable_amp --amp_dtype bfloat16
```

**重要**：T33 默认叠加在 T25 bypass 上（`--enable_din_interest`），因为 T25 已证 +0.004 LB。**注意这是"T25+T33 联合"**，不是"T33 alone vs baseline"。Alternative：做 T33 alone（不带 T25）对照版作为第三 slot（可选）。

---

## Go/No-go 判决矩阵（详）

### Dev 级（E3 Dev vs `icq-half-gap05` baseline 0.842443）

| Dev | 判决 | 下一步 |
|---|---|---|
| ≥ 0.850 | 🎯🎯🎯 **破天级别**（+0.008，论文迁移率 100%+） | 立即 LB + seed sweep + 全线叠加 |
| ≥ 0.845 | 🎯🎯 **强真信号**（+0.0026，对标 +4.87‰ 的 50%） | 立即 LB 校准 + 叠加 T34 |
| ≥ 0.8437 | 🎯 **真信号**（+0.0013，3× rerun 噪声） | LB 校准 |
| [0.8433, 0.8437) | 🟡 疑似 | seed=3 rerun 确认 |
| [0.8421, 0.8433) | ⚪ 持平 | Deprecated |
| < 0.8421 | ❌ 负向 | 关闭 · 排查 gate 是否饱和 / 初始化问题 |

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
