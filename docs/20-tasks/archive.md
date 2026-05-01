# Archived Tasks（已完成 · 只读）

> 已完成 `[x]` task 的精简归档。每条只留**一句话 + 链接**，详细数字在对应 EXP，详细决策在对应 ADR。
> 如需复盘 → 优先看 EXP/ADR；只在历史回溯时回看 task 节点本身。

---

| ID | 状态 | 完成日期 | 一句话总结 | 详见 |
|----|------|----------|-----------|------|
| T1 | `[x]` | 2026-04-28 | 跑通 baseline 在 1k demo 上，AUC 0.6761 / 0.6799（2 epoch CPU） | [`EXP-001`](../40-experiments/EXP-001-baseline-cpu-demo.md) |
| T2 | `[x]` | 2026-04-26 | 写 `baseline/infer_schema.py` 自动从 parquet 列名正则识别四类特征生成 schema.json | `baseline/infer_schema.py` |
| T3 | `[x]` | 2026-04-30（晨） | 平台首个 training job 启动成功；46 user_int + 14 item_int + 4 domain seq 与 demo 完全一致 | [`EXP-002`](../40-experiments/EXP-002-platform-baseline-job.md) |
| T7 | `[x]` | 2026-04-30（夜） | 平台默认 `infer.py` 完全动态读取 `train_config.json` + `schema.json` 构建模型；transformer / swiglu / longer 三种 encoder + rankmixer 自定义配置全部 Eval Success（0.834 / 0.844 / 0.842） → 无需自定义上传 infer.py | [`EXP-002 §Findings #5`](../40-experiments/EXP-002-platform-baseline-job.md) |
| T9 | `[x]` | 2026-04-30（夜） | 三组编码器实测推理时长 81.93s ~ 513.9s，最长不到 30min 硬上限的 30%；swiglu vs transformer 推理快 6.3×，编码器选型对 infer time 影响显著 | [`EXP-002 §Findings #6`](../40-experiments/EXP-002-platform-baseline-job.md) |
| T4 | `[x]` | 2026-04-30（夜） | 三种 encoder 在真实数据上 Leaderboard AUC：swiglu 0.84381 > longer 0.8422 > transformer 0.834；**swiglu 既最快（35min/epoch）又最高分**，与 HyFormer Table 1 "Transformer ≥ SwiGLU" 期望反向；后续以 swiglu 为基线 backbone | [`EXP-002 §编码器消融评估`](../40-experiments/EXP-002-platform-baseline-job.md) |

---

## T1 / T2 / T3 完整原文档备份

为便于后续回溯，原 `document/TODO.md` 中 T1 / T2 / T3 节点的完整内容（含完整 recipe / 踩坑表 / 复现步骤）已分别落到：

- T1 → `docs/40-experiments/EXP-001-baseline-cpu-demo.md`
- T2 → `baseline/infer_schema.py` 自带 docstring，无需额外文档
- T3 → `docs/40-experiments/EXP-002-platform-baseline-job.md`
