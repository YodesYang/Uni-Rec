# 架构决策记录（ADR · Architecture Decision Records）

> **永不删，决策后只读**。一个决策一个文件，按时间顺序追加，不重排。
> 后续如果某个决策被推翻，**写新 ADR 标记 "Supersedes ADR-NNN"**，原 ADR 保留以保留历史血缘。

---

## 何时该写 ADR

只要 `docs/20-tasks/active.md` 上一条 task 同时满足以下任一条件 → 升 ADR：

1. 涉及"为什么这么选 / 与备选方案的对比"
2. 涉及反向风险评估（"什么情况下要回滚"）
3. 改动会影响默认行为或数据流（user rule "影响数据/采样/权重/特征" 红线）
4. 决策 1 个月后回看时，自己也得花 10+ 分钟才能想起当初为什么这么选

写完 ADR 之后，`active.md` 对应位置改成「→ 详见 [ADR-NNN](../30-decisions/ADR-NNN-*.md)」一句话即可。

---

## ADR 模板

参考 [Michael Nygard 的 ADR 格式](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)，本仓库简化为 5 节：

```markdown
# ADR-NNN · <一句话标题>

> Status: Accepted | Superseded by ADR-XXX | Deprecated
> Date: YYYY-MM-DD
> Related Tasks: T<N> (link)
> Related Experiments: EXP-NNN (link)

## Context（背景与触发原因）
- 触发证据 / 引发讨论的事实是什么？
- 问题域 / 限制条件 / 时间压力？

## Decision（最终决定做什么）
- 一句话总结
- 关键技术细节（参数、文件位置、行数）

## Alternatives Considered（备选方案）
- A: ...（pros / cons / 为什么没选）
- B: ...

## Consequences（影响与外溢）
- 正向：解锁了什么 / 提升了什么
- 负向 / 待观察：留下什么风险 / 后续要监控什么

## Verification（如何验证决策正确）
- 已验证：本地 / demo 数据上的实测结果
- 待验证：平台 job / 公榜回流后看哪些指标
```

---

## 当前 ADR 索引

| ID | 标题 | 状态 | 日期 | 相关 Task |
|----|------|------|------|-----------|
| [ADR-001](./ADR-001-time-aware-valid-split.md) | 验证集按 Row Group 时间序切分（vs 行级切分） | Accepted | 2026-04-30 | T8 |
| [ADR-002](./ADR-002-slice-diagnostics.md) | valid 集 4 维切片诊断（默认开，无 CLI 开关） | Accepted | 2026-04-30 | T10 |
| [ADR-003](./ADR-003-feature-engineering-plan.md) | 特征工程作为 P0 路径（4 类候选 + 防泄露 + 实施时序） | Accepted | 2026-05-01 | 待 T11~T14 |
| [ADR-004](./ADR-004-target-item-seq-conditioning.md) | Target-Item 硬交互注入序列侧（seq-token 级 target-item conditioning） | Accepted | 2026-05-04 | T16 |
| [ADR-005](./ADR-005-valid-split-redesign.md) | Valid Split Strategy Redesign（dev + holdout 双 valid，降低 valid→LB gap） | Proposed | 2026-05-04 | T17 |
| [ADR-006](./ADR-006-combined-auc-loss.md) | CombinedAUCLoss = WeightedBCE + PairwiseBPR（对齐 AUC 评估指标的 ranking loss） | ❌ **Deprecated**（2026-05-04 晚 10 · LB −0.00993 证伪） | 2026-05-04 | T18（已关闭） |
| [ADR-007](./ADR-007-rank-mixer-per-token-ffn.md) | RankMixerBlock Per-token FFN（修正仓库 shared FFN 与论文原版偏差） | Accepted | 2026-05-04 | T19 |
