# Next Year Playbook · 明年比赛开局手册

> 目标：明年新赛题开始时，用 30-60 分钟恢复本赛季最重要的经验，避免从零摸索。

## 0. 开局先读

按顺序读：

1. [`README.md`](./README.md)
2. [`final-recap-2026.md`](./final-recap-2026.md)
3. [`../00-context/competition.md`](../00-context/competition.md)
4. [`../00-context/codebase-map.md`](../00-context/codebase-map.md)
5. [`../00-context/platform-state.md`](../00-context/platform-state.md) 顶部 + 搜关键模型名

如果只有 10 分钟，只记住：

```text
不要先堆模型。先确认数据时间轴、public/test 分布假设、validation 是否可信。
```

## 1. 第一批必须确认的问题

| 问题 | 为什么重要 | 怎么查 |
|---|---|---|
| train/test 是否时间不重叠 | 决定 validation 设计 | EDA 时间戳范围、官方答疑、eval log |
| label 是否延迟成熟 | 决定负样本权重/窗口 | label by time、conversion delay proxy |
| public/private 是否同分布 | 决定是否 public overfit 风险 | 多窗口 EDA + leaderboard drift |
| 数据是否按 row group/文件含时间结构 | 决定切分方式 | RG min/max timestamp |
| sequence timestamps 是否同一语义 | 决定 time bucket | per-domain timestamp column EDA |
| 推理是否有硬时限 | 决定模型复杂度 | infer time profile |

## 2. 开局实验模板

### 2.1 Baseline 复刻

目标：先跑通平台，不追求分数。

```text
baseline smoke
  -> full train default
  -> eval default checkpoint
  -> verify train_config/schema/infer compatibility
```

验收：

- 有可评估 checkpoint。
- inference 不超时。
- train/eval 使用同一版 `model.py` / `dataset.py` / `infer.py`。
- 记录 baseline public AUC、eval time、train time。

### 2.2 时间验证最小闭环

目标：赛季前 48 小时内完成，不要拖到最后。

```text
normal time split
fixed clean window A
fixed clean window B
same-day sanity window
```

对每个候选记录：

```text
normal AUC / logloss
clean AUC / logloss
aux window AUC / logloss
public AUC if evaluated
```

## 3. 模型主线优先级

从本赛季经验看，优先级应该是：

1. **数据/validation alignment**
2. **time bucket / recency representation**
3. **target-aware sequence modeling**
4. **multi-task / delayed feedback objective**
5. **feature interactions**
6. **seed / checkpoint / soup**

不要反过来。seed/soup 是收尾工具，不是主线突破工具。

## 4. 实验决策规则

### 4.1 什么时候可以 public eval

候选满足以下至少两条，再考虑 eval：

- 属于已 public-positive 的家族。
- normal validation 不差。
- clean-window 不红。
- aux window 有一致增益。
- 机制假设清楚，且实现经过小样本/日志验证。
- 不是单纯 high logloss、high aux、seed lucky。

### 4.2 什么时候不要 eval

默认不评：

- 只提升 normal dev，clean-window 红。
- 只提升 aux，但 primary/logloss 明显伤。
- 只是 seed/epoch/soup 小扰动，且没有公共正向家族支撑。
- 属于已被 public 拒绝的机制近邻。
- eval quota 紧张时的“反正试一下”。

## 5. 本赛季可复用起点

如果明年赛题仍然是类似广告推荐 / pCVR / sequence modeling，可以从这些代码概念起步：

```text
PCVRHyFormer
RankMixerNSTokenizer
SwiGLU sequence encoder
per-domain time buckets
item-conditioned query
UE dense split
DIN interest
multi-task click/conversion loss
aux validation windows
```

但不要机械复用以下具体数字：

```text
valid_ratio=0.02
seq_d:512
seed=7
conversion_loss_weight=2.0
specific timestamp windows
```

这些是 2026 数据分布下的结果，不是普适常数。

## 6. 明年更早做的事情

本赛季后期才做、但明年应该提前：

- validation oracle audit。
- train/public time-window backtest。
- eval log EDA 解码。
- public eval quota ledger。
- negative-results table。
- exact recipe registry。
- code upload/version checklist。

## 7. 明年不要重复的管理错误

- 不要让 `run_*.sh` 模板散落在 `src/`，继续放 `platform_run_templates/` 或更清晰的暂存目录。
- 每个平台任务启动后，第一时间核对 Args dict。
- 每个 public eval 后，立刻更新 `platform-state.md`。
- 不要让日志只存在聊天里，重要补充要落文档。
- 不要在最后一天才系统思考验证指标和 public 分布假设。

## 8. 最小可执行 checklist

```text
[ ] 官方规则/数据/时间线读完
[ ] baseline train/eval 跑通
[ ] schema + timestamp columns 自动识别
[ ] normal split + clean window A/B 建好
[ ] eval quota ledger 建好
[ ] first 8 jobs 分成 4 个机制 + 4 个诊断
[ ] 每个 job 的 Args 和结果落 platform-state.md
[ ] 每天结束写 10 行复盘：今天哪个信号变可信 / 变不可信
```

## 9. 对未来自己的提醒

如果明年又遇到一个“听起来很对，但一直不涨”的阶段，先问：

```text
我们是在解决数据集真实问题，
还是在优化一个被污染/错位的验证集？
```

这个问题比再开 10 个模型任务更值钱。
