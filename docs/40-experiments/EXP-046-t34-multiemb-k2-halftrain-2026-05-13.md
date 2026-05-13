# EXP-046 · T34 RankUp Multi-Embedding K=2 · Planning（未启动）· 2026-05-13

> **Status**: 📋 **Planning** · 代码待落地 · Smoke 未做 · 平台 Job 待启动
> **日期**: 2026-05-13 CEST 15:10 起草
> **关联**：[ADR-012](../30-decisions/ADR-012-multi-embedding-k2.md)（设计决策） · [RankUp paper §3.3 + §4.2](https://arxiv.org/abs/2604.17878) · `docs/20-tasks/active.md §T34`
> **依赖**：本 EXP **优先级低于 EXP-045**（T33 NLIR）· 代码改动面更大 · VRAM 压力高 · T33 Smoke 通过后再启动 T34 代码开发

---

## TL;DR

**目标**：给 `RankMixerNSTokenizer` 每个 non-skipped fid 分配 **K=2 个独立 `nn.Embedding` table**，concat + project 回 emb_dim。RankUp §4.2 Table 1 显示 "Global Token + Multi-Emb" 组合是 5 组件里最大单项 **+0.13%~+0.21% 线上 AUC**。

**关键数字**：
- RankUp 同团队证据 · 保守 50% 迁移率 → 本赛题期望 **Dev +0.0009~+0.0018 / LB +0.001~+0.003**
- 参数增量 **+~300M embedding params**（+100% of current 300M sparse embedding footprint）
- VRAM 增量估算 **+3-5 GB**（optimizer state included），**可能需要 batch size 128 → 96**

**Go/No-go**（vs `icq-half-gap05` E3 Dev 0.842443）：
- E3 Dev ≥ 0.846 → 🎯🎯 强真信号（对标 RankUp +0.18% AUC）
- E3 Dev ≥ 0.8437 → 🎯 真信号
- E3 Dev < 0.8421 → ❌ Deprecated

---

## 优先级与启动条件

**为什么 T34 优先级 < T33**：
1. 代码改动面 2× T33（~60 行 vs ~40 行）
2. VRAM 压力 **显著更大**（可能要降 batch size）
3. 论文独立收益估算 +0.18% 比 T33 NLIR 的 +4.87‰ **低 1.6×**
4. 与 T33 正交 S4 可叠加，先验证 T33 再叠加 T34 ROI 更高

**启动条件**（满足任一即启动 T34 代码落地）：
- T33 Smoke 通过 + T33 平台 Job A 已提交（CEST 5/13 17:00+）
- T33 平台 Dev 回流 ≥ 0.8437 真信号 → 必做 T34 作为 S4 叠加候选
- T30 / T31 EDA 驱动版平台回流后仍有 training slot 空闲

---

## 实验计划

### Phase 1: 本地 Smoke（~1h · T33 完成后启动）

- [ ] `py_compile` 全过
- [ ] Smoke A: `--multi_emb_k 1` · 参数量与 M9 SOTA **bit-identical**（353,334,977）
- [ ] Smoke B: `--multi_emb_k 2` · 参数量增量验证：
  - 期望 ~2× embedding + (K-1) × Linear(128, 64) × num_fids
  - demo 下精确数字待测
- [ ] Smoke B: 两 embedding table 初始化**不同**（抽查 `embs[0][0].weight` vs `embs[0][1].weight`）
- [ ] Smoke B: Forward shape 正确（B, emb_dim）下游 concat / split 不变
- [ ] Smoke B: train 5 steps 无崩溃
- [ ] Smoke B: infer.py 加载 `cfg['multi_emb_k']=2` 产出 1000 predictions
- [ ] Smoke B: 本地 RAM 峰值（demo）**估算平台 VRAM**：
  - demo 1k vocab 小 · 参数增量 ~2MB · 推算平台 **≈ +300M params × 4 bytes = +1.2 GB**
  - Optimizer state: Adagrad 对 sparse 每 param +1 float, AdamW 对 dense 每 param +2 float
  - **总 VRAM 增量估算：+3-5 GB**（embedding + adagrad state + 可能 checkpoint buffer）

### Phase 2: 平台 Half-train（~10-11h · 估算 +20% 时间 vs M9）

**Job A（主 · seed=42 · batch=96 保守）**：
```
Job Name: t34-multiemb-k2-halftrain-gap05-seed42-bs96
改动文件：baseline/model.py (RankMixerNSTokenizer), baseline/train.py, baseline/infer.py
batch size 降到 96（从 128）以规避 OOM
```

**Job B（双 seed · seed=3 · batch=96 · 松 guard）**：
```
Job Name: t34-multiemb-k2-halftrain-gap05-seed3-bs96
```

**Config 参数**（M9 基础 + T34 ⭐ 新增）：
```bash
--multi_emb_k 2                 # ⭐ T34 新增
--batch_size 96                 # ⭐ 从 128 降到 96 规避 OOM
# 其他与 T33 config 相同（T25 bypass 保留）
```

**关键决策点**：T34 是否叠加 T33 NLIR？
- 如果 T33 S2 通过，**T34 也带上 `--enable_nlir_gating`** 作为"T25+T33+T34 三叠加"
- 如果 T33 S2 失败，**T34 只带 T25 bypass**（不带 T33），单变量归因

### Phase 3: 对照（batch=128 K=1 · 分离 batch 影响）

**Job C（对照组 · 验证 batch 降到 96 本身是否掉 Dev）**：
```
Job Name: batch96-k1-halftrain-gap05-seed42
配置：M9 完全相同但 batch=96（其他不变）
```

**判决**：
- 若 Job C E3 Dev < M9 baseline E3 Dev 超过 0.0005（seed 噪声），说明**batch=96 本身有 penalty**，T34 相对 Job C 才是净增量
- 若 Job C E3 Dev 持平 baseline，batch 降档无 penalty，T34 直接和 baseline 比

**这个对照 Job 可以和 T34 Job A 并行启动**（2 slots），用同一 training slot batch 降档实验。

---

## Go/No-go 判决矩阵（详）

### Dev（E3 Dev · 可能要同时看 batch=128 vs batch=96）

| T34 Dev (bs=96) | 对照 Dev (bs=96, k=1) | 净增量 ΔDev | 判决 |
|---|---|---|---|
| 0.846+ | 0.842± | +0.004 | 🎯🎯 强真信号 |
| 0.844+ | 0.842± | +0.002 | 🎯 真信号 |
| 0.843+ | 0.842- | +0.001 | 🟡 边缘 · seed rerun |
| 0.842± | 0.842± | ~0 | ⚪ 持平 · Deprecated |
| <0.840 | 0.842± | -0.002 | ❌ 负向 |

### LB

| LB (T34 bs=96) | vs T25 bypass B 0.843433 | 判决 |
|---|---|---|
| ≥ 0.848 | +0.0046 | 🎯🎯 破 M9 SOTA |
| ≥ 0.846 | +0.0026 | 🎯 真信号 · S4 叠加 |
| ≥ 0.842 | -0.001 | 🟡 可能 batch 降档 penalty |
| < 0.839 | -0.004+ | ❌ 负向 |

---

## 风险 & 对策

### 风险 1（高）：VRAM OOM

- **症状**：平台启动 ~5 min 后 CUDA OOM
- **对策**：
  - 首选：batch=96（当前 config）
  - 备选：batch=64 + `--num_workers 4`（省 data loader 内存）
  - 最后：降 emb_dim（但会影响 d_model 整除性，不优先）

### 风险 2：训练时间 +20% 超 10h

- 预估 4 epoch ~10-11h · ADR-008 明确 "full-train ≥10h 推荐 compile+AMP"，**保持开 compile+AMP**
- 若回流时间 > 12h 要告警

### 风险 3：K=2 embedding warmup 不足

- **症状**：E1 Dev 比 K=1 baseline 明显低 -0.005+（未收敛）
- **对策**：
  - Adagrad 对 sparse embedding 有自适应 learning rate，理论上应该收敛
  - 若严重，下一版可试 `--num_epochs 6` 看 E5/E6 是否补上

### 风险 4：batch=96 本身 penalty

- 由 Job C 对照组隔离
- 若 penalty > 0.0005，必须基于 "T34 vs batch=96 baseline" 判决，不能和 batch=128 baseline 比

---

## 预期时间线

| 时段 | 动作 |
|---|---|
| 5/13 CEST 16:00-18:00 | T33 代码 + Smoke 完成 |
| 5/14 CEST 03:00-07:00 | T33 平台回流 · 决定是否启动 T34 |
| 5/14 CEST 10:00-12:00 | T34 代码落地 + Smoke |
| 5/14 CEST 13:00 | T34 Job A + Job C 平台提交 |
| 5/15 CEST 02:00-04:00 | T34 Job A/C 回流 |
| 5/15 CEST 04:00-06:00 | 基于 Dev + 对照做 LB 决策 |

---

## 平台提交清单（待 T33 回流后补细节）

### Job A: t34-multiemb-k2-halftrain-gap05-seed42-bs96

**上传文件**：
- `baseline/model.py`（RankMixerNSTokenizer 加 K 支持）
- `baseline/train.py`（加 `--multi_emb_k` CLI）
- `baseline/infer.py`（加 fallback）
- **可能还需要**：如果 T33 通过，保留 T33 代码改动（NLIR）

**run.sh 关键差异 vs M9**：
```bash
--multi_emb_k 2
--batch_size 96
```

### Job C: batch96-k1-halftrain-gap05-seed42（对照）

**改动**：只 `--batch_size 96`，其他与 M9 完全一致（不加 `--multi_emb_k`）

---

## Materials（待回填）

- Commit: （代码落地后填）
- Smoke 日志：（完成后填）
- Platform Job ID: （提交后填）
- 回流 Dev/Loss: （回流后填）
- LB: （校准后填）
- VRAM 实测：（从平台 log 里读 peak mem）
