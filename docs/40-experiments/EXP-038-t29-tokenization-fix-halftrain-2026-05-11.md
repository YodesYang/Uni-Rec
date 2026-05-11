# EXP-038 · T29 Tokenization Fix Half-train · 2026-05-11 早

> **Status**: 📋 实验计划 · 2 job 待启动（等 7:00 CEST LB 回流后决策最终提交顺序）
> **日期**: 2026-05-11 CEST 04:30 agent 起草 · 西班牙凌晨
> **关联**：[ADR-009](../30-decisions/ADR-009-tokenization-fix-ns-groups.md) · [EXP-035](EXP-035-kaggle-gold-retrospective-2026-05-10.md) · [EXP-037](EXP-037-u4i3-gap0-seed-sweep-eda-abandoned-2026-05-11.md)

---

## TL;DR

消化 DECEM 5/8 明说的 "Tokenization 修复" 方向（团队过去 48 天 0 投入），本批次 2 个 half-train job：

- **T29-A**：启用 `baseline/ns_groups.json`（原作者手工语义分组 U1~U7 + I1~I4）
- **T29-B**：启用 `baseline/ns_groups_vocab_tier.json`（按 vocab tier 分 3 组）

共识：**零代码改动**，只改 `run.sh` 一行 `--ns_groups_json` 参数，100% 平台兼容。

---

## 背景（上下文见 ADR-009）

简短版：
- `RankMixerNSTokenizer` 里 `groups` 控制 fid 拼接顺序 → 相邻 fid 进同一 chunk → 同 chunk 内 fid 共享一个 Linear projection
- 当前 `--ns_groups_json ""` 走 singleton fallback = fid 编号序（≡ 随机序，因为 fid 匿名化）
- DECEM 榜一明说 "Tokenization 修复是 work 的小改动"
- 原作者 4/26 commit 的 `baseline/ns_groups.json` 已有手工分组（U1~U7），但当时因 group tokenizer 路径的整除性约束被关掉，切到 rankmixer 后从未重新启用

---

## 本地 Smoke 已完成（2026-05-11 CEST 04:30）

### T29-A Smoke

```
$ python baseline/train.py \
    --data_dir ./data/demo \
    --ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json ./baseline/ns_groups.json \
    ... --max_train_steps 5 --device cpu
```

关键日志：
```
Loading NS groups from .../baseline/ns_groups.json
User NS groups (7): ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']
Item NS groups (4): ['I1', 'I2', 'I3', 'I4']
RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
PCVRHyFormer model created: num_ns=8, T=16, d_model=64
Total parameters: 353,334,977
```

用户 NS groups（fid schema index 序）：
```
[[0, 3], [4, 5, 25, 26, 27], [22], [7, 8, 9, 10, 24],
 [23, 28, 29], [6, 16, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
 [1, 2, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]]
```

**Infer 验证**：
```
$ MODEL_OUTPUT_PATH=.../best_model EVAL_DATA_PATH=.../data/demo ... python baseline/infer.py
Saved 1000 predictions from 2000 rows
```

- sidecar `ns_groups.json` 自动写入 best_model/
- `train_config.json` 里 `"ns_groups_json": "ns_groups.json"`（basename-rewritten）
- infer.py `_resolve_ns_groups_path` + `_load_ns_groups` 自动 recover 正确 fid 顺序

### T29-B Smoke

```
$ python baseline/train.py \
    ... --ns_groups_json ./baseline/ns_groups_vocab_tier.json \
    ... --max_train_steps 5 --device cpu
```

关键日志：
```
Loading NS groups from .../baseline/ns_groups_vocab_tier.json
User NS groups (3): ['T_large', 'T_mid', 'T_small']
Item NS groups (3): ['T_large', 'T_mid', 'T_small']
RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
Total parameters: 353,334,977
```

User NS groups（可见 fid 按 vocab 降序 reordering）：
```
[[10], [1, 21, 12, 3, 2, 9, 20, 13, 24, 8, 7],
 [4, 19, 18, 11, 29, 23, 15, 22, 17, 25, 26, 27, 44, 45, 30, 0, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 5, 14, 16, 28, 36, 43, 6]]
```

**参数量一致性**：T29-A / T29-B / 原 singleton baseline 三者 `Total parameters` 完全相同 = `353,334,977` → **仅 fid 拼接顺序变化，容量不变**，纯单变量 ablation。

### py_compile / lint

```
$ python -m py_compile baseline/*.py scripts/gen_ns_groups_vocab_tier.py
OK
```

---

## 平台 Job 配置

### 共通 Baseline（复制自 `icq-half-gap05` EXP-030 · 同配置）

- `d_model=64`, `emb_dim=64`, `num_queries=2`, `hidden_mult=4`
- `seq_encoder_type=swiglu`, `num_hyformer_blocks=2`, `num_heads=4`
- `seq_hash_vocab=500000`, `emb_skip_threshold=1000000`
- `enable_time_of_day_features`, `item_conditioned_query`
- `ns_tokenizer_type=rankmixer`, `user_ns_tokens=5`, `item_ns_tokens=2`
- **valid_split_strategy=time, valid_ratio=0.05, enable_row_valid_gap, valid_gap_ratio=0.05** ← **gap=0.05 领地**
- `seed=42`, `batch_size=128`, `num_epochs=4`
- `enable_torch_compile`, `enable_amp`, `amp_dtype=bfloat16` ← T27 加速默认
- `logloss_rebound_patience=1`（严格 guard）

### T29-A Job

```bash
# Job Name: t29-tokfix-halftrain-gap05-seed42
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time --valid_ratio 0.05 \
    --enable_row_valid_gap --valid_gap_ratio 0.05 \
    --item_conditioned_query \
    --seed 42 --batch_size 128 --num_epochs 4 \
    --enable_torch_compile --enable_amp --amp_dtype bfloat16 \
    "$@"
```

### T29-B Job

```bash
# Job Name: t29-vocabtier-halftrain-gap05-seed42
# 只改 --ns_groups_json 一行：
#   --ns_groups_json "${SCRIPT_DIR}/ns_groups_vocab_tier.json" \
# 其余与 T29-A 完全一致
```

### 平台提交清单

**需要上传的改动文件**：**2 个 JSON + 1 个 script**，**无 Python 代码变更**
- `baseline/ns_groups.json`（复用 4/26 原文件，无改动；但确保上传到 training code package）
- `baseline/ns_groups_vocab_tier.json`（新增）
- `scripts/gen_ns_groups_vocab_tier.py`（新增，可选 —— 若平台只跑 baseline/ 目录可不上传）

**run.sh**：平台编辑框里直接粘贴上面对应版本。

**不需要新上传 Python 代码** —— `train.py` / `model.py` / `infer.py` / `dataset.py` / `trainer.py` 相对上次 T25 DIN 提交**零改动**。平台上可以 Copy 上次 T25 DIN 的 job，只改 run.sh（换 ns_groups JSON 文件名）。

**⚠️ 训练代码包里需要包含 `baseline/ns_groups.json` 和 `baseline/ns_groups_vocab_tier.json`** 两个文件。否则 `${SCRIPT_DIR}/ns_groups.json` 会走 `os.path.exists=False` fallback 到 singleton，结果 = 旧 SOTA 配置，浪费 quota。

---

## Go / No-Go 判决阈值

### S2 Half-train（4 epoch）判决

对照基线：`icq-half-gap05-icq-halftrain` E3 Dev 0.842443（EXP-030 §C · gap=0.05 配置, same seed=42）

| E3 Dev AUC | 判决 | 下一步 |
|---|---|---|
| ≥ **0.8437** | 🎯 **强真信号**（Δ+0.0013 = 3× rerun 噪声 0.0003 + 采样偏差） | 立刻 evaluation |
| [0.8433, 0.8437) | 🟡 疑似 | seed=3 rerun |
| [0.8421, 0.8433) | ⚪ 持平（噪声内）| Deprecated |
| < 0.8421 | ❌ 负向 | 关闭变体 |

**LogLoss 联动**：若 Dev AUC 在 [0.8433, 0.8437) 但 LogLoss 低于 baseline E3（0.27917）超 −0.0015 → 升级判决为真信号

### S3 LB 校准（若 S2 通过）

对照基线：`icq-half-gap05` LB 0.839223

| LB AUC | 判决 |
|---|---|
| ≥ **0.847** | 🏆 **SOTA 破 + 方法学巨大增益** → 优先深化该变体 + 叠加 T25 DIN |
| ≥ **0.842** | 方法学真信号 → 保留并叠加其他方向 |
| ≥ **0.840** | 微弱正向，观察 seed rerun |
| < **0.839** | 持平或负向 → Deprecated |

**决策联动**：若 A **和** B 都 LB ≥ 0.842，说明"有意义分组 > 无意义分组"（两种任何分组都 work），优先用 A（更贴近 DECEM 原话"简单修复"语义）。若**仅** B work，说明"vocab tier 分组"是关键（不只是"非默认分组"），未来做更精细的 vocab tier 扫描。

### S4 叠加（若 A 或 B 在 S3 通过）

- `t29-tok-t25-din-halftrain-gap05-seed42`（+ DIN bypass）
- `t29-tok-halftrain-gap0-seed42`（切到 gap=0 领地冲 SOTA 上沿）

S4 目标：**LB ≥ 0.850**（Top 20 阈值）

---

## 时间线（激进 5/14 前冲 0.850）

| 日期 | 事件 | 备注 |
|---|---|---|
| **5/11 07:00 CEST** | 今日 4 eval LB 回流（A/B/C/D from EXP-036/037）| 决定 T29 提交优先级 |
| **5/11 07:00~08:00** | 提交 T29-A + T29-B 平台 training job（占 2/4 slot）| 留 2 slot 给其他方向 |
| **5/11 07:30~08:00** | 按决策树起另外 2 个 slot：T25 DIN seed=11 / 其他 | 根据 A/B/C/D LB |
| **5/11 12:30~13:30 CEST** | T29-A / T29-B 回流（~5h half-train + compile warmup）| 按 S2 表判决 |
| **5/11 晚** | 消耗 1~2 eval quota 做 S3 校准 | 看 Dev 判决结果 |
| **5/12 早 07:00 CEST** | S3 LB 回流 | S4 方向决定 |
| **5/12~5/14** | S4 叠加实验（若正向）| 目标 LB 0.850 |

---

## 风险与 Fallback

### 风险 1：Dev 信号太弱判决边缘

- **现象**：E3 Dev 在 [0.8433, 0.8437) 之间
- **原因**：Tokenization 的方法学增益可能只有 +0.0005~0.0010 Dev（小于 rerun + sampling 噪声合计 ~0.0013）
- **Fallback**：seed=3 rerun（1 个 training slot）；若 seed 间方向一致（A seed=42 和 seed=3 都 +0.0005~+0.0008 Dev），升级为真信号

### 风险 2：平台真实 vocab 与 demo 偏差大

- **现象**：T29-B 的 `(small_upper, mid_upper) = (100, 2000)` 在平台真 vocab 下 tier 划分不合理（比如所有 user fid 在真实 vocab 下都 > 1000 落进 T_mid/T_large，小 tier 空）
- **检测**：训练启动日志 `RankMixerNSTokenizer: 46 fids, total_emb_dim=2944` 是否变化（如果 emb_skip_threshold 过滤的 fid 数变了，total_num_fids 会变）
- **Fallback**：**无**（需要等平台回流看 tier 分布合理性；若分布失衡，下一轮用真实平台 vocab 重生成 JSON）

### 风险 3：作者 ns_groups.json 设计是"拍脑袋"无信号

- **现象**：T29-A LB 持平或负向
- **原因**：U1~U7 是 4/26 原作者猜的，可能本身不比 fid 编号序好
- **Fallback**：只信任 T29-B（vocab tier 有可观测数据支持），关闭 A

### 风险 4：Tokenization 修复方向在本赛题数据规模下根本 marginal

- **现象**：T29-A 和 T29-B 都持平
- **原因**：1.9M rows 下 Linear projection 有足够 capacity 学出 fid 异质性，chunk 顺序无所谓
- **Fallback**：关闭整个 T29 方向，转 T28 DIN block-integrated（方法学 ceiling 最高剩余方向）

---

## 不做的事

- ❌ T29-A + T29-B 同一个 job 里叠加（不是正交变量，是竞争假设）
- ❌ 直接同时起 A gap=0 + B gap=0.05（不该把 Tokenization 变量和 gap 领地变量混着测）
- ❌ 用 evaluation quota 做 "T29 Dev 校准"（EXP-030 已经证明 gap=0.05 Dev 是可信 LB 代理，不用额外 LB 校准就能判 Go/No-go）
- ❌ seed sweep at this stage（每 variant 只 seed=42 一枪；EXP-035 铁律"单假设 ≤ 2 seed 除非最终上沿搜索"）

---

## 材料

- 本地 smoke logs: `runs/t29_smoke/variant_A.log` / `variant_B.log` / `infer_A.log` / `infer_B.log`
- 生成的 JSON: `baseline/ns_groups_vocab_tier.json`
- 生成器: `scripts/gen_ns_groups_vocab_tier.py`
- 原作者 JSON: `baseline/ns_groups.json`（未修改）
- ADR: `docs/30-decisions/ADR-009-tokenization-fix-ns-groups.md`

## 评估

**今晚 agent 阶段 A 产出**（CEST 04:12 ~ 04:55 · 约 45 分钟）：
1. ✅ 理解 Tokenization 机制（RankMixerNSTokenizer / groups 作用）
2. ✅ 定位根因（run.sh 2026-04-26 起误关 ns_groups，过去 48 天从未启用过作者的手工分组）
3. ✅ 设计两个变体（A = 手工分组 / B = vocab tier）
4. ✅ 生成 `scripts/gen_ns_groups_vocab_tier.py` + `baseline/ns_groups_vocab_tier.json`
5. ✅ 本地 smoke 全通过（train + infer + 参数量一致性）
6. ✅ ADR-009 + EXP-038 落档

**零代码改动**（只 1 个 JSON 新文件 + 1 个生成脚本），**零平台风险**（infer.py 侧 sidecar resolution 机制已验证），**等 7:00 CEST LB 回流后起 platform job**。

**Top 20 目标进展**：Tokenization 是 DECEM 公开 4 个 work 方向里唯一未消化的；若 A 或 B 任一 LB ≥ 0.842 = 真信号，叠加 T25 DIN + seed 上沿有望**5/14 前触 0.850**（aggressive commitment 时间线）。
