# EXP-068 · 5/17 晚 4 Free Slots Overnight Batch · M73/M74/M75 + M50 重跑 + M76 接力 · 2026-05-17

> **Status**: 🟡 4 jobs in-flight (M73/M74/M75 启 ~CEST 17:00 5/17 + M50 重跑 ~17:30) · M76 等 M50 重跑完接力
> **Date**: 2026-05-17 CEST 18:35 起草
> **Trigger**: 5/17 全天 LB 数据回流 (M55 死 / M71 cross-seed 下沿 / M68 stack 失败) + 用户 5/20 ddl 战略约束 + EDA-67 v2 progress 完成 (blob 暂不分析)
> **Related**: EXP-067 (EDA-67 v2 design) · EXP-065 (5/16 晚 batch · 部分混淆纠正) · EXP-061 (5/16 晚 stack overnight) · EXP-059 (M49 LB SOTA) · ADR-014 (UE split path B) · team-memory line 195 (5/20 ddl 新铁律)

---

## TL;DR

**今晚 4 slots 设计 (5/17 18:35 起)**:
- 🟢 **M73** (path B × seed=2) · cross-seed 第 3 数据点
- 🟢 **M74** (path B × seed=5) · cross-seed 第 4 数据点 · path B 路径 触饱和阈值 (≥4)
- 🟢 **M75** (path B + per-domain T38 v2) · T38 v2 fix + UE stack 验证
- 🟡 **M50 重跑** (path B + weekend filter) · M50 ckpt 5/17 15:53 已过期 · 重训挽救 LB 评估
- ⏸️ **M76** (path B + DCN cross · T34 ready) · M50 重跑完接力 ~22:30 起 · ~03:30 5/18 完成

**5/20 deadline 战略约束** (team-memory line 195 新铁律 5/17 16:14 立):
- cross-seed 主线开放 (撤销 5/22 lock)
- path B cross-seed 上沿搜索 (s42 0.848 / s11 0.846 跨度 0.0015 偏小 · 但仍可期)
- stack accumulation 探索 (T38 v2 / DCN 路径未死)
- ckpt selection 战略激活 (M68 E3 ckpt 5/18 早评)

---

## 1. 背景 · 5/17 全天 audit

### 1.1 5/17 早 4 jobs 完成 + LB 回流

**5/17 早起 4 jobs** (用户 14:15~15:38 启动 · 19:00~21:30 完成):

| Job | Config | seed | dev (E4) | E3 dev | LB |
|---|---|---|---|---|---|
| **M68** | path B + Q_init + DIN bypass (3 stack) | 42 | 0.870376 | 0.869659 | **0.846468** ❌ |
| **M65** | path A + per-domain T38 v2 + DIN bypass | 42 | 0.868973 | 0.868623 | 待评 |
| **M68 v2** | M68 same config rerun | 42 | 0.870355 | 0.869836 | 待评 |
| **M71** | path B + DIN × seed=11 (cross-seed) | 11 | 0.870344 | 0.869952 | **0.846641** ⚠️ |

**今晚 LB 回流闭环**:
- M55 LB 0.844343 (Bilinear gating · user-side gating 路径关闭)
- M71 LB 0.846641 (path B × s11 cross-seed · 第 2 数据点 · 下沿)
- M68 LB 0.846468 (path B + Q_init stack · **net −0.001694** · Q_init in path B 失败)

### 1.2 关键洞察

**Path B cross-seed 跨度** (基于 2 数据点):
- s42 (M49): 0.848162 ⭐ SOTA
- s11 (M71): 0.846641
- **跨度 0.001521** (M22 5 seed 跨度 0.011 的 14%)

→ Path B 路径 cross-seed 上沿可能 < 0.852 (基于跨度 0.001521 外推):
- 假设上沿 = s42 + 0.001 ≈ 0.849 (远低于 0.852)
- 必须 stack accumulation 才能达 0.852

**Stack 加性失败规律加强**:
- M68 (path B + Q_init): **dev −0.000365 / LB −0.001694** · stack 反向放大 4.6×
- M50 (path B + weekend filter): dev −0.0088 双 stack 净损害
- M51 (path B + temporal_weight): dev TBD · 待 5/18 早 LB

→ 5/14~5/17 stack 加性失败率 ~80% · 但 T38 v2 + UE / DCN + UE 是新组合 · 必须真测。

### 1.3 用户 5/20 ddl 战略约束 (5/17 16:14 立)

**用户硬 deadline**: 0.852 by 5/20 (5/22 推前 2 天)

**team-memory line 195 新铁律** (5/17 commit `541ef12`):
- ① 5/17~5/20 cross-seed 主线开放 (LB > 0.8484 阈值放宽)
- ② path B cross-seed 优先于 M22 路径
- ③ 重复 cross-seed ≥ 4 数据点饱和阈值
- ④ stack accumulation 同步进行
- ⑤ ckpt selection 战略激活

**5/20 前到 0.852 真实概率** (修订估): **~20~30%** (M68 stack 失败后下调)

---

## 2. 4 Jobs 设计逻辑

### 2.1 多样性 (EXP-035 5/10 铁律)

| Slot | Job | 类型 | 路径 |
|---|---|---|---|
| 1 | M73 | cross-seed | path B × s2 |
| 2 | M74 | cross-seed | path B × s5 |
| 3 | M75 | stack | path B + T38 v2 |
| 4 | M50 重跑 | rescue | path B + weekend (LB 评估保障) |

**接力**: M76 (path B + DCN cross) 等 M50 重跑完成。

### 2.2 EDA-064 三滤镜框架 audit

| Job | 滤镜 1 (架构 vs 数据) | 滤镜 2 (OOV-friendly) | 滤镜 3 (信息熵 vs 噪声) | 滤镜 4 (实证锚) | 总过 |
|---|---|---|---|---|---|
| M73 | ✅ 架构 | ✅ | ✅ (cross-seed 高熵) | ✅ EXP-028 | **4/4** |
| M74 | ✅ 架构 | ✅ | ✅ | ✅ | **4/4** |
| M75 | ✅ 架构 + 数据 | ✅ | ⚠️ stack 失败率 80% | ⚠️ T38 v1 silent skip · v2 fix M65 弱信号 | **3/4** |
| M76 | ✅ 架构 + 数据 | ✅ | ⚠️ stack | ✅ DECEM +0.001 锚 | **3.5/4** |

### 2.3 排除 jobs (今晚不上)

- ❌ **M55 重训** (Bilinear) · LB 已闭环死
- ❌ **DCN alone** (M23 LB −0.011 死) · 路径关闭
- ❌ **per-X 独立参数** (T19 / T28 / FaFE deprecated)
- ❌ **8-cluster ns_groups** (json 未 ready · 不上)
- ❌ **multi_emb_k=2 / Global Token** (M38v2 / M39v2 dev 弱负 · 不 stack)
- ❌ **M50 单训 evaluation** · 用户已决重训保 LB

---

## 3. M73 · path B × seed=2 (cross-seed 第 3 数据点)

**Job Name**: `M73-m22-uesplit-cs2-gap0-4ep`

**机制**: T37 path B (UE split 已 verified) × seed=2 · 测 path B 路径 cross-seed 上沿。

**期望**:
- dev: 0.870 ± (LB rerun band ±0.0001 · cross-seed 跨度 ~±0.005)
- LB: 0.845~0.848 (M22 路径 s2 LB 0.841 数据点参考 · 但 path B + DIN base 期望偏高)

**Args verify (启动后 30s 必查 · 5 keys)**:
- ✅ `seed: 2`
- ✅ `enable_ue_split: True` / `ue_split_fid: 61`
- ✅ `enable_q_init_item: False`
- ✅ `enable_per_domain_buckets: False`
- ✅ `enable_dcn_cross: False`
- ✅ `enable_din_interest: True` (baseline 默认)

**run.sh** (用户已贴到平台 ~CEST 17:00):

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
    --seed 2 \
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
    "$@"
```

---

## 4. M74 · path B × seed=5 (cross-seed 第 4 数据点)

**Job Name**: `M74-m22-uesplit-cs5-gap0-4ep`

**机制**: 同 M73 · 仅 `seed=5`。

**期望**: 同 M73。

**Args verify**: 同 M73 · 仅 `seed: 5`。

**run.sh**: 同 M73 · 仅 `--seed 5`。

---

## 5. M75 · path B + per-domain T38 v2 (M52' 真修复)

**Job Name**: `M75-m22-uesplit-perdomain-gap0-seed42-4ep`

**机制**: T37 path B (UE split) + T38 v2 fix (per-domain time bucket boundaries) · 验证 T38 v2 + UE stack。

**与历史区别**:
- M28 (5/14): T38 v1 silent skip bug · per-domain 实际未生效
- M52' (5/16 晚): T38 v1 silent skip bug v2 (column name mismatch) · 仍未生效
- M65 (5/17 早): T38 v2 fix 后 + path A · dev 0.868973 (vs M22 +0.000397 弱信号)
- **M75**: T38 v2 + path B 首测 · 期望 dev > M65 (因 path B 已 +0.002122)

**期望**:
- dev: 0.870~0.871 (M49 base 0.870741 + T38 v2 弱信号 +0.0001~+0.0005)
- LB: 0.847~0.849 (期望 stack 加性 +0.0005~+0.001 · 但 sub-additive 风险)

**Args verify**:
- ✅ `seed: 42`
- ✅ `enable_ue_split: True`
- ✅ **`enable_per_domain_buckets: True`** ⭐
- ✅ `enable_q_init_item: False`
- ✅ `enable_dcn_cross: False`

**T38 v2 fit log 验证 (启动后 5 min 必看)**:
启动后 5 min 内应看到 4 行类似:
```
[T38] per-domain bucket fit · seq_a · n_samples=NNN · boundaries=[...]
[T38] per-domain bucket fit · seq_b · n_samples=NNN · boundaries=[...]
[T38] per-domain bucket fit · seq_c · n_samples=NNN · boundaries=[...]
[T38] per-domain bucket fit · seq_d · n_samples=NNN · boundaries=[...]
```
**任一 `n_samples=0` · silent skip 仍存在 · 立即 cancel**。

**run.sh**: 同 M73 · 多加 `--enable_per_domain_buckets` + seed=42 (不是 2)。

---

## 6. M76 · path B + DCN cross (T34 · M50 重跑后接力)

**Job Name**: `M76-m22-uesplit-dcn-gap0-seed42-4ep`

**Status**: ⏸️ pending · M50 重跑完成 ~CEST 22:30 后启 · 完成 ~03:30 5/18 · ckpt 5/19 03:30 过期 OK

**机制**: T37 path B + T34 DCN-V2 cross layer (社区 DECEM 实证 +0.001 LB)。

**DCN config (default)**:
- `--dcn_cross_user_fids "1,49"` (EDA-049 实证 high MI user-side fids)
- `--dcn_cross_item_fids "5,6,7,10,12"` (EDA-049 实证 high MI item-side fids)
- `--dcn_cross_layers 2` (DCN-V2 default)

**期望**:
- dev: 0.870~0.871 (M49 base + DCN 弱信号)
- LB: 0.847~0.849 (DECEM 实证 +0.001 锚)

**Args verify**:
- ✅ `seed: 42`
- ✅ `enable_ue_split: True`
- ✅ **`enable_dcn_cross: True`** ⭐
- ✅ `dcn_cross_user_fids: "1,49"`
- ✅ `dcn_cross_item_fids: "5,6,7,10,12"`
- ✅ `dcn_cross_layers: 2`

**run.sh**: 同 M73 · 多加 `--enable_dcn_cross` + seed=42。

**自定义 infer.py 必勾**: T34 平台首测。

---

## 7. M50 重跑 · path B + weekend filter (rescue)

**Status**: 🟡 in-flight (~CEST 17:30 起 · 完成 ~22:30)

**重跑原因**: 5/16 训完的 M50 ckpt **5/17 15:53 24h 过期** · 5/17 18:35 现在已过期 · 重训挽救今晚 1 quota LB 评估机会。

**Config**: 与原 M50 一致 (path B + `--enable_weekend_only_filter`):

```bash
--enable_ue_split --ue_split_fid 61 \
--enable_weekend_only_filter \
```

**期望** (与 5/16 实测 M50 一致):
- dev: 0.861974 ± 0.0001 (rerun band)
- LB: 0.840~0.848 (反例验证 · 但 ROI 极低)

**今晚 1 quota 用法**: M50 重跑完成 ~22:30 后立即提交 evaluation。

---

## 8. 启动顺序 + 时序 (CEST)

| 时刻 | 事件 |
|---|---|
| 17:00 | 用户启 M73 (slot 1) |
| 17:00 | 用户启 M74 (slot 2) |
| 17:00 | 用户启 M75 (slot 3) |
| 17:30 | 用户启 M50 重跑 (slot 4 · 替代原 M76 计划) |
| ~22:00 | M73 / M74 / M75 完成 |
| ~22:30 | M50 重跑完成 + 用户提交 M50 LB evaluation (5/17 剩 1 quota) |
| ~22:30 | 用户启 M76 (slot 接力 M50 重跑) |
| ~03:30 5/18 | M76 完成 |
| 04:22 5/18 ⚠️ | 5/16 晚 M51/M52/M53/M54 ckpt 24h 过期红线 |
| 06:00 5/18 | quota refresh · 4 quota 评 5/16 晚 ckpt + M75 + others |

**关键时序约束**:
- M50 重跑完 → 立即 evaluate LB (5/17 剩 1 quota 唯一窗口)
- 5/18 04:22 前 publish M51/M53 (用户 5/17 已决今晚 publish)
- 5/18 早 06:00 quota refresh 后立即评 4 个 (M51 / M53 / M75 / +1 弹性)

---

## 9. 5/18 早 quota 决策预案

**5/18 早 4 quota 候选优先级** (M73/M74/M75 dev 出后修订):

### 优先级 ⭐⭐⭐ (必评)

1. **M51 LB** (5/16 晚 path B + temporal_weight) · ckpt 5/18 04:22 过期 · 必须明早评
2. **M53 LB** (5/16 晚 path B + NLIR) · ckpt 同样过期 · 必须明早评
3. **M75 LB** (今晚 path B + T38 v2) · 验 stack 加性是否 work

### 优先级 ⭐⭐

4. **M68 E3 ckpt LB** (ckpt selection 战略首测) · E3 dev 0.869659 · 验 best_dev ≠ best_LB
5. **M76 LB** (今晚 path B + DCN) · DECEM +0.001 锚 · 但 03:30 才完成
6. **M65 LB** (path A + T38 v2) · 验 T38 v2 单变量 LB
7. **M68 v2 LB** (LB rerun band confirm)

### 优先级 ⭐

8. **M73 / M74 LB** (path B cross-seed s2/s5)

### 5/18 早默认推荐

**M51 + M53 + M75 + M68 E3 ckpt** (3 必评 ckpt 过期 + 1 ckpt selection 首测)

**或** **M51 + M53 + M75 + M76** (3 必评 + 1 stack 真信号)

→ 取决 M76 完成时间 (03:30 实际还是 03:00) + 5/18 早 06:00 quota 时段。

---

## 10. Risks (已识别)

### Risk 1 · M50 重跑后 LB ROI 极低 (probability ~95%)

- M50 dev 0.861974 → LB 触 0.852 概率 < 0.5%
- 用户已决 keep · 信息熵小但 quota 已 commit
- **Mitigation**: 5/17 剩 1 quota · 用作 M50 LB 已 commit · 不浪费 (评失败也不消耗 quota)

### Risk 2 · M75 stack sub-additive (probability ~50%)

- 5/14~5/17 stack 加性失败率 80% · 但 T38 v2 + UE 是新组合
- **Mitigation**: 即使 sub-additive · 信息熵仍高 (验 stack 路径是否全死)

### Risk 3 · path B cross-seed s2/s5 LB 全 < 0.847 (probability ~40%)

- 已 2 数据点 (s42 / s11) 跨度 0.0015 偏小
- 如 s2/s5 也低于 s42 · path B cross-seed 上沿确认 < 0.852
- **Mitigation**: 数据点本身有价值 (确认上沿) · 帮助 5/19+ 决 M76 / 其他方向

### Risk 4 · 5/16 晚 ckpt 5/18 04:22 过期前 publish 失败 (probability ~5%)

- 用户 18:35 已决今晚 publish M51/M53 · 但用户操作可能延迟
- **Mitigation**: 5/18 04:22 前 verify · 若延迟 · 5/18 早立即 publish + 评

---

## 11. References

- ADR-014 (UE split path B · M49 LB 0.848162 实证)
- EXP-059 (M49 LB 实证 SOTA)
- EXP-061 (5/16 晚 M51-M54 stack overnight)
- EXP-064 (三滤镜框架)
- EXP-065 (5/16 晚 M60/M55/M54'/M52' batch)
- EXP-067 (EDA-67 v2 design)
- team-memory line 195 (5/20 ddl 新铁律 · 5/17 commit `541ef12`)
- team-memory line 109 (失败 evaluation 不消耗 quota · 5/17)
- src/run.sh (M22 baseline · canonical fork)
- src/train.py (T38/T34/T37 flags)
- platform-state.md (single source of truth)
