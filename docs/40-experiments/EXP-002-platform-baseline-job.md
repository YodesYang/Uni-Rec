# EXP-002 · 平台首个 baseline job（GPU / batch=256 / 1.9M 行 / 198.7M params）

> **Date**: 2026-04-30（晨）
> **Status**: Completed（训练成功；评估额度等 T7 解除阻塞）
> **Related Tasks**: T3（已 archived，本 EXP 即结案证据）/ T4 / T7
> **Platform**: Tencent Angel
> **Commit**: T8/T10 之前的 baseline tip

---

## Setup

- **数据**：平台真实初赛数据 / 1,900,000 行 / 1,000 个 parquet 分片
- **配置**：默认 `run.sh`（即 `--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 --emb_skip_threshold 1000000 --num_workers 8 --batch_size 256`）
- **资源**：GPU（19GB） / 9 CPU / 55GB RAM（平台标配）

---

## Result

### 关键指标对比（demo vs 平台真实）

| 指标 | demo (1k) | 平台 (1.9M) | 一致？ |
|---|---|---|---|
| user_int fid 数 | 46 | 46 | ✓ |
| item_int fid 数 | 14 | 14 | ✓ |
| domain seq 数 | 4（a/b/c/d） | 4（a/b/c/d） | ✓ |
| 总参数量 | 160.9 M | **198.7 M** | sparse vocab 真值更大（demo 偏小），但**字段集**一致 |
| 训练 AUC | 0.6761~0.6799（CPU 2 epoch） | 待回填（用户报告公榜 0.843371） | — |
| 推理时长 | — | 待 T9 profile 回填 | — |

---

## Findings（关键发现）

1. **fid 集合稳定**：平台 job 启动日志确认 46 user_int + 14 item_int + 4 domain seq 与 [EXP-001](./EXP-001-baseline-cpu-demo.md) demo 完全一致 → `docs/00-context/data-schema.md §3` 字段集**不需要扩展**

2. **`emb_skip_threshold=1e6` 在真实数据上未触发**：demo 里 `seq_a` 跳了 1/8 个 fid（demo 在该 fid 上 vocab 估值偏大），平台真实数据里**未跳**（实际 vocab < 1e6）。说明真实 sparse 参数比 demo 多 1 个 tensor / +37.7M 参数。**不影响正确性**，只是 sparse table 数量不同

3. **真实数据 1.9M 行 / 1000 parquet 分片** → 本地为 demo 写的 `data/demo_rg100/` 重写步骤仅 demo 需要，平台无此问题

4. **平台 Evaluation 页已预置 `infer.py` (15.6K)** → 原 T7「新建 infer.py」计划作废，改为「Publish 后下载默认 infer.py 审查兼容性」（详见 active.md T7 节）

---

## 后续阻塞与决策（影响后续 EXP）

- **T7 必须先解除阻塞**：我们已偏离 baseline 默认（`ns_tokenizer=rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2`），不修 infer.py 必然 Eval Failed，公榜永远没分
- **T8 + T10 工具已就绪**：下一次平台 job 起来时，可一并验证 `overlap_sec` + 4 张切片表，**单次评估同时回流两批信号**（详见 [ADR-001](../30-decisions/ADR-001-time-aware-valid-split.md) + [ADR-002](../30-decisions/ADR-002-slice-diagnostics.md)）

---

## Reproduce Recipe

平台上直接 `bash baseline/run.sh`（环境变量平台自动注入）。

如需在本地用 demo 重现 198.7M 参数量级（不可能，受限于 demo 数据 vocab） → 不可重现，本 EXP 是平台真实数据**专属**结论。

---

## 待回填字段

- [ ] **训练 AUC（首次 platform job 收敛）** — 用户报告公榜 0.843371，离线训练 AUC 待补
- [ ] **`overlap_sec`（filename 模式）** — 等下次 job 看 ADR-001 sanity log
- [ ] **4 张切片表（user_int_nonzero / item_int_nonzero / seq_total_len / label_type）** — 等下次 job 看 ADR-002
- [ ] **Inference Time** — T9 profile 数字（已知公榜其它队伍报过 625.28s ≈ 10min）
