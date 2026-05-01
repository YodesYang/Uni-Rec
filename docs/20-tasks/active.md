# Active Tasks · TAAC 2026

> **单一事实源**。任何 task 的状态变更只在本文件落定。
> meta-spec（审视清单 / 状态约定 / 编号约定）见 `docs/20-tasks/README.md`。
> 已完成 `[x]` task 见 `docs/20-tasks/archive.md`。

---

**最近一次审视**：2026-05-01（PDT 19:15 · time split 修复后实验纪律调整）· 结论：冲前五目标下，`swiglu` 仍是主线骨干，`d_model=128` 路线确认低 ROI（leaderboard 0.836944，较基线 0.84381 退步 -0.0069）；`swiglu+time` 必须以 2026-05-01 修复 `PCVRParquetDataset(row_groups=...)` 后重新上传的 job 为准，修复前 time job 不再作为路线结论依据；`swiglu+dropout=0.05` 训练完成待评估；transformer re-train 仍在跑。当前 Top5 路径：① 回收 transformer-retrain + dropout + 修复后 swiglu-time 线上分；② 建立 fast proxy 训练开关，避免每个想法默认等待 10h full train；③ 优先推进 T8.5 row-level timestamp cutoff，再按 ADR-003 做 P0 特征工程。

---

## 任务列表

### T11 `[!]` 建立快速实验漏斗，降低 10h full train 探索成本

**决策（2026-05-01）**：full train 只作为高置信候选复核，不再作为默认探索路径。所有非平凡代码改动 / 超参想法先走 fast proxy，再决定是否消耗完整训练与评估额度。

**实施顺序**：
1. 在训练链路补快速实验开关：`max_train_steps`、`limit_train_rgs`、`limit_valid_rgs` 或等价参数
2. 建立固定 proxy 流程：静态检查 → 小样本 smoke → 1 epoch / limited RG → 2~3 epoch 半程确认 → full train
3. 每个平台 job 记录 commit、job name、关键 args、valid AUC / LogLoss、public score、结论，建立 valid→leaderboard 映射
4. 后续 T8.5 / P0 特征工程先过 proxy，再投 10h full train

**当前优先级**：高。T8.5 与 P0 特征都可能需要多轮验证，若不先做 gating，等待成本会继续主导迭代速度。

### T5 `[~]` 复现 OneTrans Mixed-Param 版本（暂缓 · 等 T4 数字）

**决策（2026-04-26）**：暂缓启动。理由：
- HyFormer 论文 Table 1 中 OneTrans 仅领先 LONGER+RankMixer 0.05~0.08% AUC，在拿到本地 T4 数字之前贸然投 600~800 行新代码 ROI 低
- 复赛二期再启动可结合 T4 的最佳编码器选项做对比
- 维持「最小变更」红线，避免污染 `baseline/model.py`

**代价评估**（启动时参考）：约新增 600~800 行代码，集中改动 `model.py`。

**关键差异 vs 当前 baseline 的 HyFormer 风格**：
- 把 S-tokens / NS-tokens 拼到同一注意力矩阵，不再用 Cross-Attention 桥接
- Attention Q/K/V 与 FFN 的权重做 Mixed Parameterization：S-token 共享一组 W；每个 NS-token 独立一组 W
- Causal Mask 摆放：NS-tokens 在 S-tokens 之后，吃完整 S 上下文
- 可选 Pyramid Stack（逐层缩 query 数）

**启动信号**（满足任一即可重启）：
- T4 的 LONGER 配置在 demo / 初赛上 AUC 已收敛，且距离公开 SOTA 差距 ≤ 0.1%
- 距离复赛截止 ≥ 3 周，有足够的离线 + 联调预算
- 你明确要冲创新奖（Unified Module Innovation Award）

---

### T6 `[~]` 草拟最终方案技术报告（KDD 答辩用）

(2026-04-30 夜：**`[!]` → `[~]`** — T4 三组编码器实测数据已落 EXP-002，原阻塞条件「T1 + T4 至少 2 组数据」满足，启动草稿。)

**当前状态**：
- ✅ 草稿骨架就位：[`docs/50-paper/draft-v0.md`](../50-paper/draft-v0.md)（Motivation / Method / Experiments / Discussion 四节，已填充本仓库已有事实）
- ⏳ 待补内容：① T8 time-aware split 实测数据（等下一轮 platform job）；② re-publish transformer step=38050 后的 leaderboard 增益；③ T10 切片表导出的薄弱片段定性分析

**模板规划**：参考 HyFormer / OneTrans / RankMixer 三篇论文的 Section 结构。本届赛事**无 Online A/B**，把那一节替换为 "Reproducibility & Engineering Notes"（infer.py 兼容性 + 推理时长 profile + 评估配额管理策略）。

**风险评估**：
- 报告草稿是**纯文档动作**，无代码改动 / 无评估开销，可与 T8 / T10 并行推进
- 5/25 ddl 前必须定稿；建议 T8 / T10 数据回流 → 5/15 前完成 v1，留 10 天 polish

---

### T8 `[~]` 把验证集切分从 random 改为按时间切

**代码已落地**（2026-04-30）— 详见决策推理 [`docs/30-decisions/ADR-001-time-aware-valid-split.md`](../30-decisions/ADR-001-time-aware-valid-split.md)。

(2026-04-30 夜：**重要性升级** — EXP-002 训练日志显示 Local Val AUC vs Leaderboard AUC 系统性 gap ≈ 0.021~0.028，强烈指向 filename 模式下 valid 集与 train 集时间高度重叠 → 与 test 集的真实 future 分布存在系统偏差。把 gap 缩到 ≤ 0.015 是 T6 报告的关键数据点。)

**当前状态**：
- ✅ `--valid_split_strategy {filename,time}` CLI flag 落地，默认 `filename`（行为不变）
- ✅ sanity log 自动打印 train/valid ts 范围 + `overlap_sec`
- ✅ 本地 demo 验证（filename overlap=765s / time overlap=770s，是 demo 数据特性）
- ⚠️ **本轮 EXP-002 三组训练 jobs 启动于 T8 探针落地之前 → log 里没有 `[T8]` 行** → 必须新一轮 platform job（baseline tip 含探针）才能拿到真实 overlap_sec
- ⚠️ **EXP-003 第一轮 swiglu+time 结果暂不采信**：用户确认该轮是在旧 swiglu 任务链路上改参数提交，日志缺有效探针证据；已新建 job 重跑
- ⏳ **待新一轮平台 job 验证 overlap_sec 真实值** → 决定是否切换到 `time` 策略

**残留待平台验证**（下一次 platform job 时）：
1. 默认参数（`filename`）跑一次 → 看 `[T8] train ts: ... valid ts: ... overlap_sec: N` 这一行
2. 若 `overlap_sec > 0` → 加 `--valid_split_strategy time` 再跑一次，看 overlap_sec 是否 ≤ 0
3. 若 `time` 模式仍 > 0 → 升级 T8.5（行级切，改在 `PCVRParquetDataset._convert_batch` 里按 row 的 `timestamp` 与 cutoff 比较、丢早样本）
4. **新增**：把 time 模式的 Local Val AUC 也提交一次 leaderboard 评估（消耗 1 配额），验证「缩小 train/eval gap 后是否同时缩小 train/test gap」 → 若 gap 缩到 ≤ 0.015 是 T6 报告核心数据点

**待自验证的两条社区线索**（仅留 trace，不动当前实现）：
1. **训练集 timestamp 严重不均衡**：社区一截图示 train 跨 02-28~03-04 共 5 天，**03-03 单日 1.31M / 总 2.10M ≈ 63%**，其它天 3~25%。若属实 → "取最后 10%" 实际几乎全是 03-04 末段；按 timestamp 排序后用更小切片（如 5%）或许更接近真实 future 边缘。
2. **疑似时间穿越**：评估日志中出现 `2026-02-27 11:00:13`（早于 train min `2026-02-28 00:35:09`），但字段标签不明（可能是 `timestamp` / `label_time` / 序列内 ts_fid 之一）；下次 job 在 train + eval 各打印 `timestamp.{p1,p50,p99}` + 序列 `ts_fid.{p1,p50,p99}` 自验证。

---

### T10 `[~]` valid 集切片诊断 — 工具已落地，待平台 job 数据决定 P0 特征工程方向

**代码已落地**（2026-04-30）— 详见决策推理 [`docs/30-decisions/ADR-002-slice-diagnostics.md`](../30-decisions/ADR-002-slice-diagnostics.md)。

**当前状态**：
- ✅ `dataset.py` 加 4 个 `_diag_*` 字段 + `trainer.py::_slice_diagnose` 方法
- ✅ 默认每次 evaluate 都跑，无 CLI 开关，开销 < 1s
- ✅ 推理路径完全无影响（`is_training=False` 不写入 `_diag_*`）
- ⚠️ **本轮 EXP-002 三组训练 jobs 启动于 T10 探针落地之前 → log 里没有 `[diag]` 行** → 必须新一轮 platform job 才能拿到 4 张切片表
- ⚠️ **EXP-003 第一轮 swiglu+time 同样缺 `[diag]` 有效证据**（旧任务链路改参数提交），已新建 job 重跑
- ⏳ **待新一轮平台 job 4 张切片表回流** → 决定 P0 特征工程方向

**待平台 job 验证后再分类决策**（每个分支对应不同的 P0 特征工程方向，**实施方案见 [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md)**）：

| 平台 job 切片表观察 | 解读 | P0 优先级动作 → 详见 ADR-003 |
|---|---|---|
| `label_type=0` 占比 > 50% | 任务是 CTCVR（曝光为主） | 维持 baseline 损失，重点做 user/item 缺失计数 dense 特征（→ P0-A） |
| `label_type=0` 占比 < 5% | 任务是 click-conditional CVR | 重新评估损失（可能需要 IPS 或 ESMM 风格辅助任务）（→ P1） |
| `user_int_nonzero` 桶间 AUC 差 > 0.02 | 用户冷启动是薄弱点 | P0-A：缺失计数 dense feature |
| `item_int_nonzero` 桶间 AUC 差 > 0.02 | 商品冷启动是薄弱点 | P0-A + P0-C：item_int_nonzero + 历史 ctr/cvr 统计（防泄露 cutoff） |
| `seq_total_len` 桶间 AUC 差 > 0.02 | 用户活跃度是薄弱点 | P0-B：每个 domain 的 seq_len + ts_range 单独成 dense 特征 |
| `label_type=1` 与 `=2` 的 pred_mean 差距小 | 模型未充分区分点击 vs 转化 | P1：转辅助任务（CTR + CVR multi-task） |

**最小动作**（下次平台 job 触发后 30 分钟内）：
1. 看 evaluate log 里 `[diag]` 开头的 4 张表
2. 对照上表，把对应 P0 改动写成 30 行 diff 提交
3. 表里数字 + 决策结论 → 落 `docs/40-experiments/EXP-NNN-platform-XXX.md`，本节只留链接

**与 T8 联动**：T8 sanity log 与 T10 4 张表都在 evaluate 输出里，**一次 platform job 同时回流两批信号**。

---

## Changelog

- **2026-05-01 PDT 19:15（实验纪律入库）**：把“不要用 10h full train 做默认探索”的规则写入 team-memory，并新增 T11。后续路线改为分层漏斗：静态 / smoke → fast proxy → 半程确认 → full train → leaderboard eval；当前最优工程动作是先补快速训练开关，再推进 T8.5 row-level timestamp cutoff 与 ADR-003 P0 特征工程。

- **2026-05-01 PDT 09:35（EXP-003 第一批回收）**：新增 [`EXP-003`](../40-experiments/EXP-003-platform-batch-2026-05-01.md) 记录三组 swiglu 批次实验。关键结论：① `d_model=128` 线上显著退步（0.836944），判定低 ROI 路线，暂时关闭；② `swiglu+time` 本轮来自旧任务链路修改提交，探针证据缺失，结论暂不采信并已新建 job 重跑；③ `dropout=0.05` 训练完成待评估；④ transformer re-train 仍在跑。T8/T10 节点同步加“结果有效性”提示，避免后续误读。

- **2026-05-01 PDT 00:55（ADR-003 特征工程规划 + Ensemble 红线入库）**：用户两连问触发本次更新 — ① "ensemble 不允许吧？" → 我回看 docs/00-context/competition.md line 77 确认禁令 + 4/3 官方 Q&A 例外（MoE OK），承认我之前 Stage 3 的 multi-seed/multi-encoder ensemble 推荐违规；② "我们目前有规划特征工程吗？" → 回看确认仅有 T10 探针决策表雏形，无正式规划。**修补动作**：① 升 [ADR-003](../30-decisions/ADR-003-feature-engineering-plan.md) — 4 类候选（P0-A 缺失计数 +0.003~0.008 / P0-B 序列活跃度 +0.003~0.010 / P0-C 历史 CVR 防泄露 +0.005~0.012 / P1 multi-task +0.005~0.015）+ 5/3~5/15 EXP-003~007 实施时序 + Ensemble 合规核对清单；② team-memory.mdc 补「Ensemble 红线 + MoE 例外 + SWA 灰色」铁律；③ active.md T10 切片决策表追加 → ADR-003 链接；④ 30-decisions/README.md 索引补 ADR-003。距初赛 ddl 24 天。

- **2026-04-30 PDT 21:50（训练日志回流 + T4 收口 + T6 启动）**：用户上传完整 train_log.txt（1092 行，覆盖三组训练直至 Early Stop）。① **T4 `[~]` → `[x]` 归档**：swiglu 既最快（35min/epoch，3.5×）又最高 Local Val AUC（0.8650）+ 最高 leaderboard（0.84381），与 HyFormer Table 1 反向；② **T6 `[!]` → `[~]`**：原阻塞「T1 + T4 至少 2 组数据」满足，启动草稿 `docs/50-paper/draft-v0.md`；③ **新发现：transformer 发布错 ckpt** — 训练已到 epoch 6 (Best step=38050 Val 0.8641)，发布的是 step=7610 (Val 0.8617)，重新发布是 30 秒动作零代码改动预计 +0.002 leaderboard；④ **新发现：Local Val vs Leaderboard 系统性 gap ≈ 0.021~0.028** — 强烈指向 filename split 的 train/val 时间高度重叠，T8 重要性升级；⑤ T8/T10 inline 增加「本轮训练早于探针落地」说明，仍需新一轮 platform job 验证；⑥ docs/README.md 顶部赛事状态行同步刷新。距初赛 ddl 25 天。

- **2026-04-30 PDT 21:30（榜单重置后立刻三连击）**：① 官方 16:00 AOE（= 21:00 PDT）重置 Industrial Track 榜单，9~22 分钟内提交三组编码器评估（baseline 44897 / swiglu 44891 / longer 44851），**全部 Eval Success**；② **T7 `[!]` → `[x]` 归档**：inference log 铁证显示平台默认 `infer.py` 完全动态读取 `train_config.json` + `schema.json` 构建模型，对偏离 baseline 默认的所有自定义配置（rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2 / 三种 encoder）原生兼容；③ **T9 `[ ]` → `[x]` 归档**：实测推理时长 81.93s ~ 513.9s，最长不到 30min 上限的 30%，安全余量充裕；④ **T4 实质性完成三组对比**（swiglu 既最快又最高 AUC，与论文期望相反），收口决策留给用户拍板；⑤ **配额提升入库**：4 资源 / 4 次每日提交（补偿规则）；⑥ team-memory.mdc 补两条铁律（infer.py 动态加载 + leaderboard Delayed Feedback 机制）。距初赛 ddl 25 天。

- **2026-04-30 深夜（第三轮 · 知识库重组）**：把原 `document/TODO.md` 387 行 + `document/knowledge_base.txt` 340 行重组为 6 个职责单一的子目录（详见 `docs/README.md`）。本文件 `active.md` 现在只装任务本体；T8 / T10 决策推理外移到 ADR-001 / ADR-002；T1 / EXP-002 数字外移到 `40-experiments/`。原顶部累积的 3 条「最近一次审视」压缩为 1 条 + Changelog 入口。Cursor rule `.cursor/rules/todo-review.mdc` 路径绑定同步更新（`document/TODO.md` → `docs/20-tasks/active.md`）。

- **2026-04-30 深夜（第二轮）**：T10 valid 集切片诊断落地。改动 2 个文件 / ~120 行 / 无 lint 错误。详见 `docs/30-decisions/ADR-002-slice-diagnostics.md`。

- **2026-04-30 深夜**：T8 代码落地。改动 3 个文件 / +85 行，无 lint 错误。详见 `docs/30-decisions/ADR-001-time-aware-valid-split.md`。

- **2026-04-30 夜**：抓取乐享比赛社区 141 题（`docs/99-raw/community-snapshots/lexiang_questions_2026-04-30-06-46-51.{json,md}`），按 organization 字段过滤出 26 条 Administrators / TAAC 官方答复，沉淀到 `docs/10-research/community-intel.md`。本次审视带来 5 项硬变更：① **T7 `[~]` → `[!]` 升级** — 官方明确「infer.py 必须与训练一致」；② **新增 T8** 验证集按时间切；③ **新增 T9** 推理 30min profile；④ T4 撤回"并行启动消融"判断；⑤ T2/T3 inline 加情报佐证。距初赛 ddl 25 天。

- **2026-04-30 晨**：平台首个 baseline job 启动成功（GPU / batch=256 / 1.9M 行 / 1000 分片）。详见 `docs/40-experiments/EXP-002-platform-baseline-job.md`。沉淀三件事：① **T3 `[x]` 结案**（搬到 archive）；② **T7 修正** — 平台 Evaluation 页**已预置 infer.py (15.6K)**，原"新建文件"计划作废，改为「Publish 后下载默认 infer.py 审查兼容性」；③ T4 启动 swiglu 第一组消融。

- **2026-04-29 晚**：第一轮 baseline 提交平台训练成功（默认配置 = `transformer` 编码器，等同 T4-B 首跑）。新增 **T7「补 `infer.py`」**（评估硬约束）；T3 仍 `[!]`。

- **2026-04-29**：T1 环境切换 venv → conda（`tencent-unirec`，python 3.11.15 / torch 2.10.0 CPU）。详见 EXP-001。

- **2026-04-28**：T1 完成。详见 EXP-001。

- **2026-04-26**：从 `knowledge_base.txt §8` 迁出 6 条 TODO；T5 决议暂缓；新增 7 步审视清单。
