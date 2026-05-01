# 实验日志（EXP · Experiments）

> **结果只读**，一次 job / 一次实验一个文件，按时间顺序追加。
> 实验跑完后回填本目录；任何后续对照 → **直接引用 EXP-NNN**，不在 task 节点里抄数字。

---

## 何时该写 EXP

只要满足以下任一条件 → 升 EXP：

1. 跑出**可复现的数字**（AUC / Loss / 资源开销 / 推理时长）
2. 平台 job 完成（不论结果好坏）
3. 重要的本地基准实验（如 baseline 复现、消融对照）

写完 EXP 之后，`docs/20-tasks/active.md` 对应位置改成「→ 详见 [EXP-NNN](../40-experiments/EXP-NNN-*.md)」一句话即可。

---

## EXP 模板

```markdown
# EXP-NNN · <一句话标题>

> **Date**: YYYY-MM-DD
> **Status**: Completed | Failed | Aborted
> **Related Task**: T<N> (link)
> **Related ADR**: ADR-NNN (link)（如果实验是验证某个 ADR 的）
> **Platform**: 本地 macOS / Tencent Angel / ...
> **Commit**: <git short hash>

## Setup
- 数据：<路径 / 行数 / 文件数>
- 配置：<关键超参 + 完整命令>
- 资源：<硬件 / 依赖版本>

## Result
- 关键指标表（AUC / LogLoss / Params / Train time / Infer time / GPU peak mem）
- 关键 log 截取（保留路径 + 时间戳，便于回查）

## Findings（关键发现）
- 与预期一致 / 不一致的点
- 触发的下一步决策（升 ADR / 改 task / 改 config）

## Reproduce Recipe
- 复现步骤（自洽，复制即可执行）
- 已知踩坑表（避免后人重蹈覆辙）
```

---

## 当前 EXP 索引

| ID | 标题 | 日期 | 平台 | 关键结论 |
|----|------|------|------|----------|
| [EXP-001](./EXP-001-baseline-cpu-demo.md) | 本地 CPU baseline 跑通 1k demo（AUC 0.6761/0.6799） | 2026-04-28 | macOS / CPU | baseline 文件 ≡ 官方训练模板，2 epoch 33s 跑通 |
| [EXP-002](./EXP-002-platform-baseline-job.md) | 平台首个 baseline job（GPU / batch=256 / 1.9M 行 / 198.7M params） | 2026-04-30（晨） | Tencent Angel | fid 集合与 demo 完全一致；总参数 198.7M（demo 160.9M）|
| [EXP-003](./EXP-003-platform-batch-2026-05-01.md) | 平台批次实验（swiglu-time / d_model=128 / dropout=0.05） | 2026-05-01 | Tencent Angel | d_model=128 明显退步；time split 需新建 job 才能有效验证；dropout=0.05 待评估 |
