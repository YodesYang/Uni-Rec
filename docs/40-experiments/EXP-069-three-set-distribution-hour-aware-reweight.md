# EXP-069 · 三集分布差异理论框架 + Hour-Aware Sample Reweight (方案 D-3) · 2026-05-17

> **Status**: Closed negative · M82 public 0.846197
> **Date**: 2026-05-17 CEST 20:50 起草
> **Trigger**: 用户深入审视"训练集/验证集/测试集差异 + 是否同分布" + 用户决 5/20 ddl 内 0.852 战略
> **Related**: EXP-052 (M22 LB + Test EDA 5 facts) · EXP-056 (Tier-0 EDA · 元宵节认知反转) · EXP-058 (8 大错误 audit) · EXP-059 (M49 LB SOTA 0.848162) · EXP-068 (5/17 晚 batch) · EXP-071 (M75 LB 0.848788 新 SOTA) · ADR-014 (UE split path B) · team-memory line 195 (5/20 ddl 战略约束)

---

## TL;DR

**理论框架** (基于已有 EDA 实证):
- 三集**多维度异质 shift** · 不是简单同/不同分布
- 5 大机制 (按修复优先级): C 多 mode boundary memorization · D user cold start · F hour daypart shift · B feature drift · E hash collision

**5/20 ddl 战略**:
- 方案 A (Multi-stage fine-tune) **平台不兼容** (USER_CACHE_PATH chicken-egg 问题)
- ✅ **方案 D-3 实施**: M41 timestamp reweight (alpha=2.5) + Hour-aware reweight (北京 09~14 × 5) 双叠加
- 期望 LB +0.001~+0.003 (vs M49 SOTA 0.848162)

**实施成果**:
- ✅ `src/dataset.py` PCVRParquetDataset 加 `hour_weight_min/max/multiplier` (3 args + state + filter logic)
- ✅ `src/train.py` 3 新 CLI flags + propagate
- ✅ Smoke test pass · 训练完整完成 · 0 model param 增加 (仅 sample reweight)
- ✅ Commit `9330ce1`

**2026-05-18 Result Addendum**:
- `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep` completed with best dev **0.869709** (E4).
- Compared with current anchor M75-4ep dev **0.870856**, M82 is **-0.001147**.
- Public evaluation returned **0.846197** (684.43s), which is **-0.002591 vs M75 0.848788** and **-0.002657 vs M84 0.848854**.
- D-3 hour-aware reweight is therefore **rejected**. The hoped-for dev-down/LB-up did not materialize.
- Follow-up record: [EXP-072](EXP-072-may18-morning-batch-results.md).

---

## 1. 三集差异理论框架 (基于已有 EDA 实证)

### 1.1 三集时间分布严格汇总

| 集 | 时间窗口 (北京时间) | rows | weekday | 主要 hour |
|----|--------------------|------|---------|----------|
| **Train** | 02-28 ~ 03-04 (5 天) | 2,099,956 | 周六/日/一/二/三 | **06~09 + 22~00 = 51%** |
| **Valid** (gap=0 · 5%) | Train 末尾 5% | ~105k | 主要 03-03 ~ 03-04 | 与 train 类似 |
| **Test** (LB) | 03-04 **08:51 ~ 13:49** (~5h) | 523,244 | **周三 (1 天)** | **09~14** (100%) |

### 1.2 三集 CVR 分布 (EXP-056 实证)

| 日期 | weekday | cnt% | CVR |
|------|---------|------|-----|
| 02-28 (周六) | weekend | 4.1% | **0.024** ⚠️ 极低 |
| 03-01 (周日) | weekend | 3.1% | **0.034** ⚠️ 低 |
| 03-02 (周一) | weekday | 5.7% | 0.119 |
| 03-03 (元宵节) | weekday | 62.5% | 0.101 |
| **03-04 (test 同日)** | weekday | 24.6% | **0.122** ⭐ |

→ Train mean CVR 0.10 vs Test CVR ~0.122 · 有 daypart 内的差异。

### 1.3 三集 user_id 分布 (EXP-052 Fact 4 实证)

- Train user_id distinct: **500,000** (EDA-049)
- Test user_id distinct (capped): **200,000** (EXP-052 Fact 4)
- 每个 test user_id 出现频次 = 1
- **Train ↔ Test 交集 = 0** (100% cold start)

### 1.4 三集 item_int / seq 分布 (EXP-052 Facts 2/3/5)

| fid | Train | Test | Δ |
|------|-------|------|---|
| item_83/84/85 | 0.887 | 0.892 | +0.005 |
| item_81 | 0.707 | 0.665 | −0.042 |
| user_109 | 0.875 | 0.879 | +0.004 |

→ **几乎完全一致** (最大 Δ < 5%) · 几乎 0 covariate shift。

---

## 2. 5 大机制 + 修复优先级

### 2.1 严格分类

| # | 机制 | 影响 | 当前修复 | 修复难度 | 5/20 ddl 适用 |
|---|------|------|---------|---------|--------------|
| **C** | Multi-mode boundary (元宵节 62.5% / 周末 7%) | 🔥 强 | M41 部分修复 + M50 weekend filter (不彻底) | 中 | ✅ |
| **D** | User cold start 100% (train↔test 0% overlap) | 🔥 强 | path B (UE split · M49 +0.000570) | 难 | ✅ (path B in-flight) |
| **F** | Hour daypart shift (test 100% 09~14 · train 13.7%) ⭐ | 🔥 强 | **M41 部分修复 (按 day) · 但未按 hour** | 低 | ✅ **方案 D-3 主修复** |
| **B** | Feature importance drift (dow 在 test 上常数) | ⚠️ 中 | M40 (移 dow) 半证伪 | 中 | 不再投入 |
| **E** | Sequence ID hash collision (seq_c_47 251×) | ⚠️ 中 | hash 500k (EXP-018 已实证 vocab 升级 ROI 不显著) | 难 | 不投入 |
| **A** | Calibration drift (train mean 0.10 / test 0.122) | 微 | (不影响 AUC) | 低优先 | 不投入 |

### 2.2 机制 F (Hour daypart shift) 是当前最大未投入空间

- M41 (`temporal_weight_alpha=2.5`) 升 03-04 整天 sample · LB +0.000254 (微正)
- M50 (`weekend filter`) 删 02-28/03-01 weekend · dev −0.0088 / LB TBD
- **Hour daypart shift 仅按 timestamp** 不按 hour-of-day · M41 是粗粒度修复

→ **方案 D-3 = M41 + hour-aware 二阶修复** · 直接 target 机制 F 精确化。

---

## 3. 方案 D-3 设计 · Hour-Aware Sample Reweight (Combined with Alpha)

### 3.1 核心思想

**M41 升级版**:
- M41: 升 03-04 全天 sample × exp(alpha × norm_ts) · 与 hour 无关
- D-3: M41 基础上 + 升 03-04 北京 09~14 sample × 5 (test daypart 精确 align)

**预期机制**:
- M41 alpha=2.5 让 test daypart 同日 (03-04 全天) 平均 weight ≈ 2.7
- D-3 hour_weight=5 让 test daypart 同小时 (北京 09~14) 额外 weight × 5
- 组合后 test daypart 同小时 sample weight ≈ 2.7 × 5 = **13.5×** (vs train 老段)

### 3.2 实施细节 (54 行 net code)

#### `src/dataset.py` 改动 (+33 行)

**PCVRParquetDataset `__init__` 加 args**:
```python
hour_weight_min: Optional[int] = None,
hour_weight_max: Optional[int] = None,
hour_weight_multiplier: Optional[float] = None,
```

**`_convert_batch` 加 reweight logic**:
```python
if (self.is_training
        and self.hour_weight_multiplier is not None
        and self.hour_weight_min is not None
        and self.hour_weight_max is not None):
    beijing_hour = ((timestamps + 8 * 3600) // 3600) % 24
    in_hour_window = (
        (beijing_hour >= self.hour_weight_min)
        & (beijing_hour <= self.hour_weight_max)
    )
    hw = np.ones(B, dtype=np.float32)
    hw[in_hour_window] = self.hour_weight_multiplier
    hw /= hw.mean().clip(min=1e-6)  # mean weight ≈ 1
    if sample_weight is None:
        sample_weight = hw
    else:
        sample_weight = sample_weight * hw  # 与 M41 alpha 叠加
```

**`get_pcvr_data` 加 args + propagate**:
```python
hour_weight_min: Optional[int] = None,
hour_weight_max: Optional[int] = None,
hour_weight_multiplier: Optional[float] = None,
```

#### `src/train.py` 改动 (+21 行)

**3 新 CLI flags**:
```python
parser.add_argument('--hour_weight_min', type=int, default=-1, ...)
parser.add_argument('--hour_weight_max', type=int, default=-1, ...)
parser.add_argument('--hour_weight_multiplier', type=float, default=-1.0, ...)
```

**Propagate to get_pcvr_data**:
```python
hour_weight_min=(args.hour_weight_min if args.hour_weight_min >= 0 else None),
hour_weight_max=(args.hour_weight_max if args.hour_weight_max >= 0 else None),
hour_weight_multiplier=(args.hour_weight_multiplier
                        if args.hour_weight_multiplier > 0 else None),
```

### 3.3 为什么不是其他方案 (严格 audit)

| 方案 | 实施成本 | 预期 LB | 5/20 ddl 可行 | Status |
|------|---------|---------|---------------|--------|
| **A · Multi-stage fine-tune** | 2-3 天 | +0.003~+0.010 | ⚠️ 平台 ckpt access 不兼容 | ❌ 放弃 |
| **B · DRO** | 5+ 天 | +0.001~+0.003 | ❌ | ❌ |
| **C · Sub-vocab embedding** | 5+ 天 | +0.001~+0.005 | ❌ | ❌ |
| **D-1 · M41 alpha=5.0** | 0 代码 | +0.0005~+0.0015 | ✅ | 备选 |
| **D-2 · Hour-aware only** | 30 行 | +0.001~+0.003 | ✅ | 备选 |
| **D-3 · alpha + hour combined** | 50 行 | +0.001~+0.003 | ✅ | ⭐ 主选 |
| **E · 元宵节 down-weight** | 10 行 | +0.0005~+0.0015 | ✅ | 5/19 备选 |

**方案 A 平台不兼容根因**:
- 平台 training 阶段 env vars: `TRAIN_CKPT_PATH` (output) / `TRAIN_DATA_PATH` (input data)
- 平台**没有** "input ckpt path" env var
- USER_CACHE_PATH 是唯一 cross-job channel · 但需要 chicken-egg 操作 (谁先 fill cache)
- 5/20 ddl 内不可解决

---

## 4. M82 Job 设计

### 4.1 Job 配置

**Job Name**: `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep`

**run.sh** (基于 src/run.sh + 方案 D-3):

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups_vocab_tier.json" \
    --emb_skip_threshold 1000000 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --item_conditioned_query \
    --enable_din_interest \
    --seed 42 \
    --batch_size 128 \
    --num_epochs 4 \
    --logloss_rebound_delta 0.003 \
    --logloss_rebound_patience 2 \
    --keep_topk_ckpt 4 \
    --enable_torch_compile \
    --enable_amp \
    --amp_dtype bfloat16 \
    --num_workers 8 \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512" \
    --enable_ue_split \
    --ue_split_fid 61 \
    --temporal_weight_alpha 2.5 \
    --hour_weight_min 9 --hour_weight_max 14 --hour_weight_multiplier 5.0 \
    "$@"
```

### 4.2 Args verify (启动后 30s 必查)

| Key | 期望值 |
|---|---|
| `seed` | 42 |
| `enable_ue_split` | True |
| `ue_split_fid` | 61 |
| `enable_din_interest` | True |
| `temporal_weight_alpha` | 2.5 |
| `hour_weight_min` | 9 |
| `hour_weight_max` | 14 |
| `hour_weight_multiplier` | 5.0 |
| `enable_q_init_item` | False |
| `enable_per_domain_buckets` | False |

### 4.3 自定义 infer.py 必勾

`hour_weight_*` 是新 args · 写入 train_config.json · 平台默认 infer.py 可能不识别 · **必勾自定义 infer.py**。

实际上 hour_weight 只影响 training (sample reweight) · **不影响 inference forward pass** · 所以 even 平台默认 infer.py 不识别 · 推理仍 work · 但**稳妥起见勾**。

### 4.4 上传文件清单

```
src/train.py
src/trainer.py
src/dataset.py
src/model.py
src/utils.py
src/infer.py
src/infer_schema.py
src/ns_groups_vocab_tier.json
```

---

## 5. 期望 LB + 风险分析

### 5.1 期望 LB

**单变量贡献分解**:
- M22 baseline: LB 0.847592
- + path B (UE split): +0.000570 → 0.848162 (M49 SOTA)
- + temporal_weight (alpha=2.5): +0.000254 (M41 实证微正 · 在 M22 base) · 期望在 path B base 类似
- + hour_weight (alpha + hour 叠加): +0.0005~+0.0015 (估算 · 直接 target 机制 F)

**累积期望 LB**: 0.848162 + 0.000254 + 0.001 = **~0.8494**

**95% 置信区间**: [0.847, 0.851]

### 5.2 风险

| 风险 | 概率 | 影响 |
|------|------|------|
| **Sub-additive (M41 + hour 加性失败)** | 30% | LB ≈ M49 (微正或持平) |
| **Mean weight ≈ 1 normalization 后效应弱化** | 20% | LB +0~+0.0005 (信号 < rerun band 0.0003) |
| **Reweight 让 model 过 specialize 到 09~14 daypart · 其他时间段 underfit · valid AUC 跌但 LB 涨** | 30% | dev↓ LB↑ (与 M41 反例机制一致 · 可接受) |
| **超参 alpha=2.5 / hour_mul=5 不是最优** | 50% | LB +0~+0.001 (调参空间 5/19~5/20) |
| **Reweight 与 path B + DIN 交互不良** | 10% | LB < M49 (灾难 · 但 0 model param 增加 · 风险低) |

### 5.3 5/20 前到 0.852 概率重估

**2026-05-18 修订 (M75 LB 回流后)**:
- 当前 SOTA: **M75 0.848788** (见 EXP-071)
- 距 0.852: **+0.003212**
- 归因 caveat: M75 不是有效 T38/per-domain 机制验证, 更像 path B / UE split 机制族里的更好 public 泛化落点

**多路径累积**:
- M82 (D-3): +0.001 ~ +0.003
- 5/18 早 LB confirm (M51/M53) 任一 work: +0.001 ~ +0.002
- M76 (DCN + UE) work: +0.001 ~ +0.002
- Cross-seed luck (M73/M74 上沿): +0.001 ~ +0.005 (5% 概率单点)
- ckpt selection (E2/E3): +0.001 ~ +0.003 (10% 概率)

**累积期望** (假设独立): M75 anchor 0.848788 + 0.001 (M82) + 0.001 (M51/M53) + 0.001 (M76) + 0.0005 (cross-seed/ckpt) = **LB ~0.8523**

→ **5/20 前到 0.852 概率**: **~25~35%** (M75 抬高 anchor, 但仍缺 +0.0032; 未因 T38 归因上调机制信心)

---

## 6. 启动时序 (5/17 晚 ~ 5/20 晚)

### 5/17 晚 (CEST 20:50 已实施 · 等其他 jobs 完成)

| Time | Event |
|------|-------|
| ~22:00 | M73/M74/M75 完成 dev (现 4 jobs in-flight) |
| ~22:30 | M50 重跑完成 → 提交 M50 LB evaluation (5/17 剩 1 quota) |
| ~22:30 | **启 M82 (path B + alpha + hour) + M76 (path B + DCN)** 并行 (用户决) |
| ~03:30 | M82 + M76 完成 |

### 5/18 早 (CEST 06:00 quota refresh)

**Superseded 2026-05-18**: M82 completed with dev 0.869709, so the priority list below is historical. Current recommendation is in [EXP-072](EXP-072-may18-morning-batch-results.md): M75(E6) for score-max mode, M84 for information-gain mode; M82 is no longer a top quota candidate.

**4 quota 候选优先级** (修订):

1. ⭐⭐⭐ **M82 LB** (D-3 hour-aware reweight)
2. ⭐⭐⭐ **M51 LB** (path B + temporal · ckpt 已 publish · safe)
3. ⭐⭐⭐ **M53 LB** (path B + NLIR · ckpt 已 publish · safe)
4. ⭐⭐ **M75 LB** (path B + T38 v2) **或** **M76 LB** (path B + DCN)

### 5/18 ~ 5/20

- **5/18 中午**: 基于早 LB 决断 5/18 晚战略
- **5/19**: cross-seed 上沿 · 调参 (alpha / hour_mul)
- **5/20**: final cross-seed + ckpt selection 冲刺 0.852

---

## 7. References

- ADR-014 (UE split path B · M49 实证)
- EXP-052 (M22 LB + Test EDA 5 facts)
- EXP-056 (Tier-0 EDA · 元宵节认知)
- EXP-058 (8 大错误 audit)
- EXP-059 (M49 LB SOTA)
- EXP-068 (5/17 晚 batch · M73/M74/M75/M50)
- team-memory line 195 (5/20 ddl 战略)
- src/run.sh (M22 baseline)
- src/dataset.py · _convert_batch (line 1981+ temporal_weight + 新加 hour_weight)
- src/train.py · CLI flags (line 660+ temporal_weight + 新加 3 hour_weight flags)
- platform-state.md (single source of truth)
- commit `9330ce1` (D-3 实施)
