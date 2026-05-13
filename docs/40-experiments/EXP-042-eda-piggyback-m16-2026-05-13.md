# EXP-042 · EDA 搭车 · M16 M9 Rerun + EDA Collection · 2026-05-13

> **Status**: 🔬 **Running** · M16 已启动 CEST 11:56 · 晚上回流后分析 EDA 数据决定下一步方向
> **日期**: 2026-05-13 CEST 11:00-11:56 代码实现 + 启动
> **关联**：[EXP-037](EXP-037-u4i3-gap0-seed-sweep-eda-abandoned-2026-05-11.md)（前一次 EDA 5 次失败教训）· [EXP-040](EXP-040-m5-m9-gap0-din-seed-sweep-2026-05-12.md)（M9 破 SOTA）· [docs/10-research/community-intel.md §DECEM-0512](../10-research/community-intel.md)（DECEM 6 大 Tricks）

---

## TL;DR

**目标**：在真实平台 1.9M 数据上收集 4 类关键 EDA 数据（Q1 user/item_id 频次分布 / Q2 per-fid null_rate × CVR gap / Q4 seq 真实长度分位数 / Q5 time_diff 分位数），**为 T30/T31/T32 方向提供数据支持**，同时 M9 config rerun 得到第二 seed=42 × gap=0 数据点验证 M9 LB 0.84671 稳定性。

**核心设计**：**"搭车策略"绕过 EXP-037 独立 EDA 5 次失败陷阱** —— 不独立起 EDA job · 在 M16 training 流程里顺带收集 · 通过 `training_summary.json['eda']` 字段回传（已验证 work 的通路）。

---

## 背景：为什么需要 EDA

### 战略缺陷认知（2026-05-13 CEST 11:00 用户反思触发）

截至今早 M9/M10/M11/M14/M15 结果回流：
- ✅ gap=0 × T25 DIN × seed=42 (M9) LB **0.84671** · 刷新 SOTA +0.00034
- ✅ T29 vocabtier 跨 seed 弱正向
- ❌ T25+T29 叠加双 seed 都失败（M6/M10 s42 / M7 s3）
- ❌ T28 / FaFE 全部证伪（独立化参数三连败铁律）

**剩余可行方向（DECEM Tricks 未消化）**：
| 方向 | 问题 |
|---|---|
| T30 时间桶分位数 | **没真实 time_diff 分布** · 手工调边界 = 盲调 |
| T31 is-missing 信号 | **没各 fid 缺失率 × CVR 相关性** · 不知哪些 fid 有信号 |
| T32 Q_init | 机制与 T25 bypass 部分冲突 · 风险未知 |

**所有未消化方向都需要 EDA 数据支撑**。

### EXP-037 教训（5 次失败根因分析）

2026-05-11 独立 EDA 脚本：
- v1 OOM (`pa.concat_tables` 全量)
- v2 Success 但 log 截断（没 flush）
- v3 100k 又 OOM
- v4 Per-RG streaming OK 但 log 停在 90/100
- v5 base64 gzip blob 仍截断

**根因**: 所有尝试走 "platform log UI 回传大数据" · 但 log UI 有未知上限（1000 行 or container exit 丢最后几秒）。

**EXP-042 规避策略**: 不通过 stdout 回传 · **走 `training_summary.json` 通路**（ckpt_dir 落盘 · 已知稳定 work，从 EXP-019 到 M15 每次都成功）。

---

## 代码架构

### 改动范围（最小侵入）

| 文件 | 改动 | 行数 |
|---|---|---|
| `baseline/dataset.py` | `PCVRParquetDataset`: 加 `enable_eda_dump` / `eda_reservoir_size` kwargs · `_init_eda_state` / `_update_eda_stats` / `_reservoir_extend` / `finalize_eda` 四个新方法 · `_convert_batch` 末尾加一处调用 | ~220 行 |
| `baseline/train.py` | CLI `--enable_eda_dump` / `--eda_reservoir_size` · 传给 `get_pcvr_data` | ~20 行 |
| `baseline/trainer.py` | `_write_training_summary` 里读 `train_loader.dataset.finalize_eda()` · 写 `summary['eda']` | ~20 行 |
| `baseline/dataset.py::get_pcvr_data` | 透传 kwargs · 当 `enable_eda_dump=True` 时强制 `num_workers=0`（主进程收集 state · 避免 worker copy 问题）| ~15 行 |

### 核心机制

#### 1. Reservoir Sampling（Algorithm R）

Q4/Q5 需要分位数但不想全存数据（OOM 风险）。用**水库采样** 保留 10k~20k 样本：

```python
def _reservoir_extend(self, entry, new_values, max_size):
    # Phase 1: 填满 reservoir
    # Phase 2: 按概率 k/seen 替换现有 slot
```

Reservoir size 20k → 分位数精度 ~0.1% · 内存 ~160KB · 足够。

#### 2. Per-fid Missing + CVR Accumulator

每 fid 维护 `[n_null, n_total, pos_null, pos_notnull]` 4 累加器：
- `null_rate = n_null / n_total`
- `cvr_null = pos_null / n_null`
- `cvr_notnull = pos_notnull / (n_total - n_null)`
- `cvr_gap = cvr_null - cvr_notnull` ← **按 |cvr_gap| 排序输出 top-20**

**"null" 定义**：`user_int[:, offset:offset+dim].all(axis=1) == 0`（dataset 已把 null/<=0 统一映射为 0）

#### 3. user_id / item_id 频次（Q1）

- 用 Python dict counter 累加 · cap 200k 防 OOM
- 输出 `freq_1` / `freq_2_10` / `freq_gt10` 占比 + `max_freq` / `mean_freq`
- **这个比 Seed 帖子"user_id 几乎全是一次性"这种定性描述要 quantitative**

#### 4. num_workers=0 约束

`enable_eda_dump=True` 时强制 `num_workers=0`。理由：
- `DataLoader num_workers>0` 时每 worker 持 dataset 的**副本** · 主进程 `train_loader.dataset._eda_state` 不会被更新
- 修复成本（跨 worker aggregate）较大 · 但 EDA 是一次性诊断 · 忍受 ~20% throughput 降级即可

**预计 M16 wall-time**: ~6-8h（vs M9 5h · 增加 20% EDA 开销 + 失去 workers 并行）

---

## 本地 Smoke 验证（CEST 11:45-11:55）

### Demo data · 1 epoch · 10 steps

```
$ python baseline/train.py --data_dir ./data/demo ... \
    --enable_eda_dump --eda_reservoir_size 5000 \
    --max_train_steps 10 --device cpu --num_workers 0
```

**关键输出**:
```
[summary] EDA collected: 1000 samples observed, user_id_distinct=1000,
          item_id_distinct=837
```

### training_summary.json['eda'] 结构验证

```json
{
  "version": 1,
  "global": {
    "total_samples_observed": 1000,
    "total_pos": 124,
    "global_cvr": 0.124
  },
  "q1_user_id": {
    "distinct": 1000,
    "freq_1": 1.0,       // ← 100% user_id 只出现 1 次（demo 特性）
    "freq_2_10": 0.0,
    "freq_gt10": 0.0,
    "max_freq": 1,
    "mean_freq": 1.0
  },
  "q2_user_int_missing_top": [
    {"col": "user_int_feats_49", "null_rate": 0.007, "cvr_null": 0.286,
     "cvr_notnull": 0.123, "cvr_gap": 0.163, ...},
    ...
  ],
  "q4_seq_length": {
    "seq_a": {"p50": 256, "p99": 256, ...},  // demo 序列都打满
  },
  "q5_time_diff": {
    "seq_a": {"p50": 2536531, "p90": 8061638, "p99": 12927464, ...}
  }
}
```

### Smoke 的 3 大观察（demo 规模 · 指示性）

1. **demo 里所有 user_id 都是 distinct**（freq_1=1.0），**印证 Seed 帖子"user 几乎全一次性"** · 平台 1.9M 会给精确数字

2. **seq_a 的 p50 = 256**（截断到 seq_max_lens 上限），说明 **序列真实长度远超 256** · **DECEM Trick 2 "按 p99 调整 seq_max_lens" 方向有效** · 平台数据会给真 p99

3. **time_diff seq_a: p50 ≈ 29 天 / p99 ≈ 149 天** · 当前 BUCKET_BOUNDARIES 在 5s~60s 有 12 桶，在 1s~7d 密集占大半（64 个桶），但**数据的 p1 就是 7.3 小时** · **大量桶被浪费**（几乎没有 sample 映射到 5s~1h 范围）→ **DECEM Trick 1 "按分位数重设桶" 方向极有潜力** · 平台数据会给 p1/p5/p25/p50/p75/p90/p95/p99/p99.9 全谱

**这 3 个 smoke 发现直接给出了 T30 和 Trick 2 的 ROI 排序**。

---

## M16 平台 Job 配置

### Job Name: `M16-m9rerun-eda-gap0-seed42-4ep`

**目标双获**:
1. **EDA 数据采集** · 写 `training_summary.json['eda']` · 晚上回流后分析
2. **M9 config rerun 验证** · 第二个 seed=42 × gap=0 × T25 DIN 数据点 · 检验 LB 0.84671 是否稳定（rerun 噪声 ±0.003 内）

**与 M9 差异**:
- `+ --keep_topk_ckpt 4` · 新 default（本次起生效）· top-4 ckpt 给 eval 多候选
- `+ --enable_eda_dump` · EDA 搭车
- `+ --logloss_rebound_delta 0.003 --logloss_rebound_patience 2` · 松 guard（M9 严格 guard · 本次宽松让训练跑完 E4）
- `--num_workers 8 → 被 EDA 自动降 0` · 主进程 state 收集约束

**预计 wall-time**: 6-8h（EDA +20% + workers=0 开销）· 晚上 CEST 18:00-20:00 回流

---

## 分析计划（等 M16 回流后）

### Phase 1: 从 training_summary.json 提取 EDA data

```bash
# 训练完成后手动下载 training_summary.json
cat training_summary.json | python -c "import sys, json; \
    d = json.load(sys.stdin); print(json.dumps(d['eda'], indent=2))" \
    > eda_m16.json
```

### Phase 2: 基于 EDA 做方向判决

| EDA 问题 | 结果解读 | 下一步决策 |
|---|---|---|
| **Q1 user_id freq_1** | > 0.95 | 确认 id 路线无用 · 主力靠行为序列 · T31 / T30 优先 |
| **Q1 item_id freq_1** | > 0.80 | 同上 |
| **Q2 per-fid cvr_gap** | 若 ≥ 3 个 fid \|gap\| > 2% | **T31 有强信号** · 只对这些 fid 做 missing indicator 值得 |
| **Q2 per-fid cvr_gap** | 所有 fid \|gap\| < 0.5% | T31 方向**关闭** · 节省代码时间 |
| **Q4 seq p99** | p99 > 当前 seq_max_len × 1.2 | **DECEM Trick 2**值得做（提升 seq_max_lens）· 预期 +0.001~0.002 LB |
| **Q4 seq p99** | p99 ≤ seq_max_len | Trick 2 不值得做 |
| **Q5 time_diff 分布** | 主要分布在 > 24h 区间 | **T30 方向重设分位数桶边界** · 精准版 |
| **M16 rerun LB (若 Dev ≥ 0.868)** | 0.846~0.848 | M9 稳定 · "SOTA 破"成立 |

### Phase 3: 下一批方向排序（等 EDA 数据后 finalize）

目前假设（smoke 观察支持）:
- **T30 时间桶分位数** · 高概率 work（smoke 数据显示当前桶 50%+ 浪费）· 实施 ~30 行代码
- T31 is-missing · 取决于 Q2 结果 · smoke 里 demo fid_49 cvr_gap +0.163 但样本仅 7 个，不可信 · 等平台真数据
- **Trick 2 seq_max_lens** · 高概率 work（smoke 显示截断）· 实施 ~10 行
- T32 Q_init · 与 T25 机制可能冲突 · 等 M16 EDA 给真 item_id 分布后再判决

---

## 与 EXP-037 的对比

| 维度 | EXP-037 (失败) | EXP-042 (本次) |
|---|---|---|
| 数据通路 | stdout log UI | **training_summary.json** (已验证 stable) |
| 实施方式 | 独立 EDA 脚本 | **搭车 training job** (M16) |
| 资源消耗 | 3 slot + 3h · 0 产出 | 1 slot · M9 rerun 有独立价值 |
| 失败风险 | 不明平台约束多次踩坑 | 本地 smoke 已验证 · 机制可靠 |
| 数据规模 | 100k~1M 样本 (OOM 反复) | **完整 1.9M train 样本自然消费** |
| 采集范围 | 限定 5 个假设 | 4 大问题 (Q1/Q2/Q4/Q5) · ~400 数字 |

---

## 不做的事（严格约束避免 EXP-037 重蹈覆辙）

1. ❌ 不独立 EDA job · 必须搭车 training
2. ❌ 不通过 stdout 回传大数据 · 只用 summary.json
3. ❌ 不为 EDA 专门起 4 slot · 最多 1 slot 搭车
4. ❌ 不做 "好奇心" 级 EDA · 只做 Q1~Q5 五个**直接指向 LB 方向**的问题
5. ❌ 单 EDA job 失败 → **不重试** · 退回 "无 EDA 数据决策" 模式

---

## 时间线

- ✅ **CEST 11:00-11:55** 代码 + 本地 smoke 验证
- ✅ **CEST 11:55** M16 平台 job 提交
- ⏳ **CEST 18:00-20:00** 预计 M16 回流 · Dev + LB (optional) + EDA 数据
- ⏳ **今晚** 基于 EDA 数据设计下一批方向（T30 / Trick 2 / T31 or not）

---

## Materials

- 代码改动: `baseline/dataset.py` (+220 行) · `baseline/train.py` (+20 行) · `baseline/trainer.py` (+20 行)
- Smoke log: 已清理（见 commit `ba8c...` diff）
- M16 run.sh: 同 active.md 贴出版本 · commit 到 `baseline/run.sh`
