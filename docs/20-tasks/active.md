# Active Tasks · TAAC 2026

> **单一事实源**。任何 task 的状态变更只在本文件落定。
> meta-spec（审视清单 / 状态约定 / 编号约定）见 `docs/20-tasks/README.md`。
> 已完成 `[x]` task 见 `docs/20-tasks/archive.md`。

---

**最近一次审视**：2026-05-04（PDT 深夜 · P0-C smoke 通过 + Multi-task P1 判定不可行）· 结论：P0-C 历史 CVR 代码链路已完整实现（dataset/train/trainer/infer 均已覆盖），本机 smoke 验证 sidecar 构建成功（800 rows, 13 scalar item fids, global_prior=0.125）；OOM 是本机沙盒内存限制非代码 bug，平台可正常运行。P1 Multi-task CTR+CVR 确认不可行：数据集全为点击样本（仅 label_type=1/2，无曝光样本 label_type=0），无法构建独立 CTR 监督信号。下一步：P0-C 平台 proxy，Job Name `p0c-timefeat-proxy-swiglu-vr005-rg128x32-s1500`。

---

## 任务列表

### T11 `[~]` 建立快速实验漏斗，降低 10h full train 探索成本

**决策（2026-05-01）**：full train 只作为高置信候选复核，不再作为默认探索路径。所有非平凡代码改动 / 超参想法先走 fast proxy，再决定是否消耗完整训练与评估额度。

**实施顺序**：
1. ✅ 在训练链路补快速实验开关：`max_train_steps`、`limit_train_rgs`、`limit_valid_rgs`
2. 建立固定 proxy 流程：静态检查 → 小样本 smoke → 1 epoch / limited RG → 2~3 epoch 半程确认 → full train
3. 每个平台 job 记录 commit、job name、关键 args、valid AUC / LogLoss、public score、结论，建立 valid→leaderboard 映射
4. 后续 T8.5 / P0 特征工程先过 proxy，再投 10h full train
5. 每次给用户一个平台运行建议时，必须同时给可复制的 Job Name，格式优先用 `<theme>-<scope>-<model>-vr<ratio>-rg<train>x<valid>-s<steps>`。

**当前状态**：
- ✅ 代码开关已落地：`--limit_train_rgs`、`--limit_valid_rgs`、`--max_train_steps`，默认 0，不影响 full train
- ✅ 平台 fast proxy smoke job 已验证：`64/16 RG + max_train_steps=500` 在 4 分半内完成，final AUC 0.824108 / LogLoss 0.252679，详见 [`EXP-004`](../40-experiments/EXP-004-fast-proxy-smoke-2026-05-02.md)
- ✅ 训练结束摘要已落地：`trainer.py` 在 Early Stop / max steps / 正常结束时打印 `[summary]`，并写 `training_summary.json` 到 ckpt 与 log 目录，避免平台上翻长日志 / 曲线找 best AUC
- ✅ best checkpoint 目录名已追加 local valid AUC：形如 `global_step48060.layer=2.head=4.hidden=64.auc=0.865884.best_model`，保留 `global_step` 前缀与 `.best_model` 后缀，符合平台命名约束
- ✅ T8.5 + P0-A/B 本地 smoke 通过：2 step demo train + 自定义 `infer.py` 产出 1000 条 predictions，详见 [`EXP-005`](../40-experiments/EXP-005-mainline-t8p0-local-smoke-2026-05-02.md)
- ✅ T8.5 + P0-A/B 平台 fast proxy 已回流：`t8p0-proxy-swiglu-vr005-rg64x16-s500` 跑通，但 local AUC 0.819652 / LogLoss 0.291509，需拆分归因，详见 [`EXP-006`](../40-experiments/EXP-006-mainline-t8p0-platform-proxy-2026-05-02.md)
- ✅ 归因 proxy 已回流：P0-only 正向，T8.5-only 是组合下降主因；下一步优先 P0-only larger proxy / full train，详见 [`EXP-007`](../40-experiments/EXP-007-t8p0-attribution-proxies-2026-05-02.md)
- ✅ P0 larger proxy 全部回流：P0-B seq-stats-only 当前最强，P0-A count-only 也强，P0-only 组合未叠加；短期 full train 优先 P0-B，其次 P0-A，详见 [`EXP-008`](../40-experiments/EXP-008-p0-larger-proxy-first-returns-2026-05-02.md)
- ⏳ P0 full train 首轮回流：`p0seq` completed，best local AUC 0.865128；`p0only` completed，best local AUC 0.865100；`p0count` partial log 到 epoch 8，best local AUC 0.865044；三者均未超过 no-P0 time reference 0.865884，详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)
- ✅ 运行建议纪律补充：后续所有平台任务建议必须附带 Job Name，避免日志 / ckpt 回收时难以对应实验。
- ⚠️ 平台并行能力修正：当前可以提交 3-4 个 training job，但 2026-05-02 晚出现同卡多进程 / 单次 20.49GiB allocation OOM；后续只把轻量 64/16 proxy 打包并行。128/32 larger proxy 与 full train 默认 `--batch_size 128`，且最多 1-2 个重 job 并行。
- ⏳ 待 P0-A full train 最终 `[summary]` 回流后，决定 P0 分支是否整体收口；阈值暂定 `best local AUC >= 0.8665` 才考虑 evaluation，`>= 0.8685` 才强推。已完成/当前回流的 `p0seq` / `p0only` / `p0count` 都不建议 evaluation。
- ✅ sparse reinit 对照已回流：no-reinit proxy 快速崩盘（best 0.837562 / last 0.619751），default reinit 更稳（best 0.845306）；关闭 no-reinit full train 路线，详见 [`EXP-010`](../40-experiments/EXP-010-sparse-reinit-proxy-control-2026-05-02.md)。
- ✅ **dense log1p 修复对照已回流**：`user_dense_feats_62~66` 为原始计数统计量（10^3~10^5），与 fid=61/87 的预归一化 embedding（10^-1）尺度差 4~5 个量级；log1p 修复后 proxy best AUC +0.000145 / LogLoss -0.000323，方向正确；`dense_log1p_fids=62,63,64,65,66` 已设为默认，详见 [`EXP-011`](../40-experiments/EXP-011-dense-log1p-proxy-2026-05-03.md)。

**当前优先级**：高。T8.5 与 P0 特征都可能需要多轮验证，后续新想法默认先过本任务定义的 gating。

### T12 `[~]` Impression 绝对时间特征（P0-D）

**来源**：社区情报 ray（Academic Track 第 27 名）报告"加时间特征 0.8135 → 0.824"（+0.0105），是其所有优化中最大单点。我们的 P0-B（`enable_seq_stats_features`：序列长度 + ts_range）在 full train 上没有复现类似收益，因此 ray 的"时间特征"指的是**不同的东西**。

**假设**：impression 请求时刻的绝对时间编码——小时（0~23）、星期几（0~6）、是否周末——是广告 CVR 的强信号（不同时段用户购买意愿差异大），而我们当前完全没有这类特征。

**实施方案**：
- 从 `timestamp` 列提取：`hour_of_day`（0~23，周期编码 sin/cos）、`day_of_week`（0~6，周期编码）、`is_weekend`（0/1）
- 作为新的 synthetic dense feature，接入 `user_dense` 通道（或单独一个 token）
- CLI flag：`--enable_time_of_day_features`，默认 False，opt-in

**分层漏斗**：
1. `py_compile` + demo smoke ✅
2. `timefeat-proxy-swiglu-vr005-rg128x32-s1500` 对比 log1p baseline ✅ → AUC +0.000808 / LogLoss −0.002033，详见 [`EXP-012`](../40-experiments/EXP-012-timefeat-proxy-2026-05-04.md)
3. ⏳ **full train**：`timefeat-full-swiglu-vr005-bs128`，阈值 `>=0.8665` 考虑 evaluation

**当前优先级**：高，proxy 信号明显，升为 full train 下一轮首选。

---

### T14 `[x]` P0-C 历史 CVR 特征 — 已收口（full train 验证负向）

**结论（追溯 2026-05-03，2026-05-04 补档）**：`p0c-full-swiglu-vr005-bs128` 已在无 timefeat baseline 上跑过 full train，best local AUC **0.865097**（E4, Early Stop E9），vs reference 0.865884，Δ = **−0.000787**。与 P0-A/B/MNW 一起系统性低于参考线 0.0007~0.0015，HyFormer 对这类 dense 统计特征已饱和。

与 hash embedding 不同（proxy 失败 → full train 正向），P0-C 是 **full train 上就已失败**，说明是特征兼容性问题而非冷启动问题，无需再尝试 P0-C+timefeat 组合。路线关闭。详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)。

---

### T15 `[x]` P1 Multi-task CTR+CVR — 已判定不可行

**结论（2026-05-04）**：数据集全为**点击后**样本（label_type=1：点击无转化 / label_type=2：点击有转化），**不含任何曝光样本（label_type=0）**。没有独立的曝光→点击监督信号，无法构建 CTR 辅助任务。Multi-task CTR+CVR 路线正式关闭。

---

### T13 `[ ]` domain_c_seq_47 vocab 截断（高基数 item ID 处理）

**来源**：社区情报 Stochastic 陷阱三："domain_c_seq_47 的 item ID 量级与顶层 item_id 相同（0~278M）"；我们的 schema 确认 vocab=278677640。

**当前状态（bug）**：`run.sh` 里 `--emb_skip_threshold 1000000`，fid=47 的 vocab=278M 超过阈值 → **整个特征输出全零向量**，等于完全丢弃了 domain_c 序列里的主 item ID 信号。

**修复方案**：两条路：
- **方案 A（截断）**：在 `dataset.py` 的 `_convert_batch` 里，对 vocab 超大的 item ID 特征，将值限制在 `[0, clip_threshold]`，超出归 UNK（0），这样可以保留高频 ID，同时 embedding table 大小可控
- **方案 B（hash embedding）**：在模型侧对超大 vocab 的特征做 hash 取模，映射到固定大小的 embedding table（例如取模到 1M），损失少量 collision，但保留了所有 ID 的可学习表示

方案 A 代码改动小，先实验；方案 B **已验证失效**（见下）。

**分层漏斗**：
1. 本地 smoke 验证 embedding table 大小和显存
2. ✅（方案 B）`t13-hash500k-timefeat-proxy-swiglu-vr005-rg128x32-s1500` proxy 回流：AUC 0.837684，**低于** timefeat-only 对照 0.839285，Δ−0.001601 → 方案 B **负向，放弃**。详见 [`EXP-015`](../40-experiments/EXP-015-t13-hash-proxy-2026-05-04.md)
3. 方案 A（vocab 截断）尚未实验，作为备选

**当前优先级**：低。方案 B 已失败；方案 A 需代价估算后与其他候选竞争额度。不占用当前 full train 资源。

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
- ⚠️ **EXP-004 fast proxy 已显示 RG-level split 仍重叠**：`time overlap_sec=302765`，`filename overlap_sec=304064`，说明单靠 Row Group 排序可能无法保证 clean split；需等 full job 复核后启动 T8.5 row-level timestamp cutoff
- ⚠️ **修复后 full time job 复核同向**：`valid_ratio=0.10 overlap_sec=341068` 已 Early Stop；`valid_ratio=0.05 overlap_sec=341086` 且 leaderboard AUC 0.844059。RG-level time split 无法消除 row-level overlap，线上仅微升，T8.5 row-level timestamp cutoff 应升为下一轮高优先级工程动作。详见 [`EXP-003`](../40-experiments/EXP-003-platform-batch-2026-05-01.md)
- ✅ **T8.5 代码本地 smoke 通过**：行级 cutoff 后 demo `candidate_rows=600 / train_rows=480 / valid_rows=120`，详见 [`EXP-005`](../40-experiments/EXP-005-mainline-t8p0-local-smoke-2026-05-02.md)
- ✅ **T8.5 + P0-A/B 平台 fast proxy 已跑通但不达升级阈值**：row-level cutoff 后 valid rows=8,414，AUC 0.819652 / LogLoss 0.291509，详见 [`EXP-006`](../40-experiments/EXP-006-mainline-t8p0-platform-proxy-2026-05-02.md)
- ✅ **拆分归因 proxy 已完成**：T8.5-only AUC 0.821174 / LogLoss 0.289528，是组合下降主因；T8.5 当前保留为诊断，不作为 leaderboard full train 主线，详见 [`EXP-007`](../40-experiments/EXP-007-t8p0-attribution-proxies-2026-05-02.md)

**当前 platform 验证项**：
1. 等 `p0count-full-swiglu-vr005-bs128` 最终 `[summary]` 回流。
2. 不评估当前 P0 full 分支：`p0seq` / `p0only` / `p0count` best local AUC 0.865128 / 0.865100 / 0.865044，均未超过 no-P0 time reference 0.865884。
3. 后续若 P0-B / P0-A 进入 evaluation，必须上传同版 `infer.py` / `dataset.py` / `model.py`，不要只依赖平台默认 infer。

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
- ✅ **EXP-004 fast proxy 已回流 4 张切片表**：`user_int_nonzero` / `item_int_nonzero` / `seq_total_len` 桶间 AUC 差均远超 0.02，强触发 ADR-003 的 P0-A / P0-B 候选；因只训练 500 step，仍待 full job 复核
- ✅ **full time job 切片表复核同向**：`label_type=0` 持续为 0，valid 仅含 `label_type=1/2`；`user_int_nonzero`、`item_int_nonzero`、`seq_total_len` 分桶 AUC 差继续明显，P0-A/P0-B 方向成立，P1 multi-task 暂不优先
- ✅ **P0-A/P0-B 代码本地 smoke 通过**：缺失计数 + 序列长度/时间跨度作为 opt-in synthetic dense features，使用 `log1p` 缩放；自定义 `infer.py` 确认训练/推理 shape 一致，详见 [`EXP-005`](../40-experiments/EXP-005-mainline-t8p0-local-smoke-2026-05-02.md)
- ✅ **P0-A/B 与 T8.5 组合平台 proxy 跑通但暂不升级**：AUC 0.819652 / LogLoss 0.291509，切片差异仍明显，需 P0-only proxy 判断特征本身收益，详见 [`EXP-006`](../40-experiments/EXP-006-mainline-t8p0-platform-proxy-2026-05-02.md)
- ✅ **P0-only proxy 正向**：AUC 0.826286 / LogLoss 0.251651，较 EXP-004 time proxy AUC +0.002178 且 LogLoss -0.001028；P0-A/B 升为短期 leaderboard 主线，详见 [`EXP-007`](../40-experiments/EXP-007-t8p0-attribution-proxies-2026-05-02.md)
- ✅ **P0-B seq-stats-only larger proxy 当前最强**：AUC 0.836734 / LogLoss 0.242303，略优于 P0-A count-only 0.836497 / 0.243201，二者均明显优于 P0-only 组合；短期 full train 优先 P0-B，其次 P0-A，详见 [`EXP-008`](../40-experiments/EXP-008-p0-larger-proxy-first-returns-2026-05-02.md)
- ⏳ **P0 full train 首轮回流中**：`p0seq` completed，best local AUC 0.865128 / LogLoss 0.225103；`p0only` completed，best local AUC 0.865100 / LogLoss 0.224833；`p0count` partial log 到 epoch 8，best local AUC 0.865044 / LogLoss 0.225021；三者均未超过 no-P0 time reference，详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)

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

- **2026-05-04 PDT（3-blocks proxy 负向 → 所有 proxy 候选耗尽）**：`t14-3blocks-timefeat-proxy-swiglu-vr005-rg128x32-s1500` AUC 0.838003 / LogLoss 0.242790，低于 2-blocks timefeat-only 基线 0.839285，Δ−0.001282。1500 步 proxy 可能不足以热身更深模型，但按漏斗规则不推进；维持 `num_hyformer_blocks=2` 默认。至此本轮所有 proxy 候选（log1p/T12/focal/ICQ/T13/3-blocks）全部回流，唯一正向信号 T12 已进入 full train，进入"等待 full train 结果"阶段。详见 [`EXP-016`](../40-experiments/EXP-016-3blocks-proxy-2026-05-04.md)。

- **2026-05-04 PDT（T13 hash-embedding proxy 负向回流）**：`t13-hash500k-timefeat-proxy-swiglu-vr005-rg128x32-s1500` AUC 0.837684 / LogLoss 0.242104，低于 timefeat-only 对照 0.839285，Δ−0.001601 AUC。hash-mod 500k 引入 collision 噪声大于其恢复 domain_c_seq_47 信号的收益；方案 B（hash embedding）正式放弃，方案 A（vocab 截断）降为低优先级备选。当前无待进行的 proxy，等 `timefeat-full-swiglu-vr005-bs128` 回流。详见 [`EXP-015`](../40-experiments/EXP-015-t13-hash-proxy-2026-05-04.md)。

- **2026-05-03 PDT（P0 路线收口 + OneTrans 全量回流）**：P0-C full best AUC 0.865097（E4，Early Stop E9），MNW full best AUC 0.864357（E11，仍在跑），均低于 no-P0 reference 0.865884。P0-A/B/C/组合 + MNW 共 6 轮全量训练全部失败，差值 -0.0007~-0.0015，P0 特征工程路线正式关闭。OneTrans v2 full（无 NS conditioner）14 epoch best 0.848712，停止；OneTrans v3 full（有 NS conditioner）E10 best 0.859408，仍在运行，NS conditioner 使 full train 提升 +0.010。详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)、[`EXP-013`](../40-experiments/EXP-013-onetrans-full-train-2026-05-03.md)。

- **2026-05-04 PDT（T12 time-of-day proxy 对照回流）**：`timefeat-proxy` best AUC 0.839283 / LogLoss 0.241525；`notimefeat-proxy` best AUC 0.838475 / LogLoss 0.243558。Δ+0.000808 AUC / −0.002033 LogLoss，是 log1p 修复收益的 5~6 倍。hour_sin/cos + dow_sin/cos + is_weekend 5 维圆形编码，FID=900004。T12 状态升为 `[~]`，升为 full train 候选，阈值 >=0.8665 考虑 evaluation。详见 [`EXP-012`](../40-experiments/EXP-012-timefeat-proxy-2026-05-04.md)。

- **2026-05-03 PDT（新增 T12 + T13）**：社区情报深度分析后落地两条新 task。T12：impression 绝对时间特征（小时/星期/是否周末），对应 ray 报告的 +0.0105 主因候选，优先级高；T13：domain_c_seq_47 vocab=278M 因 emb_skip_threshold=1M 被整个跳过，修复方案为 vocab 截断或 hash embedding，优先级中（先做 T12）。

- **2026-05-03 PDT（dense log1p 修复对照回流）**：`log1p-proxy-swiglu-vr005-rg128x32-s1500`（log1p ON）best AUC 0.838475 / LogLoss 0.243558；`nolog1p-proxy-swiglu-vr005-rg128x32-s1500`（log1p OFF）best AUC 0.838330 / LogLoss 0.243881。EDA 确认 user_dense_feats_62~66 是原始计数统计量（万级），与 fid=61/87 的预归一化 embedding（小浮点）尺度差 4~5 个量级；log1p 修复方向正确，`dense_log1p_fids="62,63,64,65,66"` 已设为 CLI 默认值，后续所有训练自动生效。详见 [`EXP-011`](../40-experiments/EXP-011-dense-log1p-proxy-2026-05-03.md)。

- **2026-05-02 PDT（sparse reinit proxy 对照回流）**：`noreinit-proxy-swiglu-vr005-rg128x32-e4` 与 `reinit-proxy-swiglu-vr005-rg128x32-e4` 完成。no-reinit (`reinit_sparse_after_epoch=999`) best AUC 0.837562 @ epoch 1 后一路跌到 0.619751；default reinit best AUC 0.845306 @ epoch 3。结论：关闭 no-reinit full train 路线，不 evaluation 这两个 proxy。详见 [`EXP-010`](../40-experiments/EXP-010-sparse-reinit-proxy-control-2026-05-02.md)。

- **2026-05-02 PDT（P0-count full train partial 回流）**：`p0count-full-swiglu-vr005-bs128` 日志回收到 epoch 8，真实 `batch_size=128`，`enable_count_features=True`、`enable_seq_stats_features=False`，best local AUC 0.865044 / LogLoss 0.225021 @ epoch 6 step 95634，epoch 7/8 回落。结论：低于 no-P0 time reference 0.865884，除非后续意外跳到 `>=0.8665`，不建议 evaluation。详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)。

- **2026-05-02 PDT（P0-only full train 完整回流）**：`p0only-full-swiglu-vr005` 已完成，日志真实 `batch_size=256`，`enable_count_features=True`、`enable_seq_stats_features=True`，best local AUC 0.865100 / LogLoss 0.224833 @ epoch 7 step 56014，Early Stop 于 epoch 12。结论：低于 no-P0 time reference 0.865884，不建议 evaluation；P0 组合 full train 未复现 larger proxy 正向外推。详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)。

- **2026-05-02 PDT（P0 full train 首轮回流）**：`p0seq-full-swiglu-vr005-bs128` 已完成，但日志真实 `batch_size=256`，best local AUC 0.865128 / LogLoss 0.225103 @ epoch 6，低于 no-P0 `swiglu + time + valid_ratio=0.05` reference 0.865884；`p0count` 和 `p0only` 仍在训练，用户报告当前 best 分别约 0.865044 / 0.865100。结论：暂不 evaluation，等剩余两组完整回流后再决策。详见 [`EXP-009`](../40-experiments/EXP-009-p0-full-train-first-return-2026-05-02.md)。

- **2026-05-02 PDT（P0 larger proxy 全部回流）**：`p0only-proxy-swiglu-vr005-rg128x32-s1500`、`p0seq-proxy-swiglu-vr005-rg128x32-s1500`、`p0count-proxy-swiglu-vr005-rg128x32-s1500` 均完成。P0-B seq-stats-only AUC 0.836734 / LogLoss 0.242303 最强；P0-A count-only AUC 0.836497 / LogLoss 0.243201 也很强；P0-only 组合 best AUC 0.831841 / LogLoss 0.244238，没有叠加收益。结论：短期 full train 优先 P0-B，其次 P0-A，组合暂缓。详见 [`EXP-008`](../40-experiments/EXP-008-p0-larger-proxy-first-returns-2026-05-02.md)。

- **2026-05-02 PDT（归因 proxy 回流）**：`p0only-proxy-swiglu-vr005-rg64x16-s500` 与 `t85only-proxy-swiglu-vr005-rg64x16-s500` 完成。P0-only 在相同 RG-level time valid 上 AUC 0.826286 / LogLoss 0.251651，优于 EXP-004 time proxy；T8.5-only AUC 0.821174 / LogLoss 0.289528，接近 EXP-006 组合下降。结论：P0-A/B 是当前正向主线，T8.5 当前保留为诊断，不直接 full train。详见 [`EXP-007`](../40-experiments/EXP-007-t8p0-attribution-proxies-2026-05-02.md)。

- **2026-05-02 PDT（平台并行能力补充）**：用户确认当前一次可并行 3-4 个 training job。后续平台建议不再默认单 job 串行，而是优先打包为 3-4 个互补 proxy：主线 larger proxy + P0-A/P0-B ablation + 必要的 full/half confirmation。

- **2026-05-02 PDT（并行 OOM 反馈）**：用户反馈两个任务出现 CUDA OOM：单次尝试分配 20.49GiB，而 GPU total 19.12GiB，日志同时列出多个 GPU 进程。修正并行策略：轻量 proxy 可并行，larger proxy / full train 默认降 `batch_size=128`，并减少重 job 并发，避免平台同卡复用或峰值显存超限。

- **2026-05-02 PDT（主线平台 proxy 回流）**：`t8p0-proxy-swiglu-vr005-rg64x16-s500` 确认平台跑到最新版代码：`max_train_steps=500`、`limit_train_rgs=64`、`limit_valid_rgs=16`、`enable_row_time_cutoff=True`、`enable_count_features=True`、`enable_seq_stats_features=True` 全部生效；T8.5 cutoff 后 `candidate_rows=168253 / train_rows=159839 / valid_rows=8414`，best local AUC 0.819652 / LogLoss 0.291509。结论：链路通过，但组合 proxy 低于 EXP-004 原 time proxy，暂不 full train，下一步做 T8.5-only 与 P0-only 拆分归因。详见 [`EXP-006`](../40-experiments/EXP-006-mainline-t8p0-platform-proxy-2026-05-02.md)。

- **2026-05-02 PDT（主线本地 smoke 完成）**：T8.5 row-level timestamp cutoff + ADR-003 P0-A/P0-B synthetic dense features 已落地并通过 demo smoke。验证项：`py_compile` PASS；2-step train smoke PASS；T8.5 cutoff 过滤出 `train_rows=480 / valid_rows=120`；best ckpt 写出 AUC 命名与 `training_summary.json`；新增 `baseline/infer.py` 并成功输出 1000 条 demo predictions。重要约束：开启 P0-A/B 后 evaluation 需上传同版 `infer.py` / `dataset.py` / `model.py`，不能假设平台默认 infer 会生成 synthetic dense features。详见 [`EXP-005`](../40-experiments/EXP-005-mainline-t8p0-local-smoke-2026-05-02.md)。

- **2026-05-02 PDT（修复后 time/0.05 线上回收）**：`swiglu-split_strategy_time-valid_ratio_0.05` leaderboard AUC 0.844059，较旧 swiglu 基线 0.84381 微升 +0.000249；当日 leaderboard 最优 0.85035，仍差 0.006291，当前排名约 190+。结论：time split/valid_ratio 调参不是破局点；RG-level split 的 3.5~4 天 overlap 已在 proxy/full jobs 中复现，下一步主线切到 T8.5 row-level timestamp cutoff + ADR-003 P0-A/P0-B 特征工程。

- **2026-05-02 PDT 14:25（best ckpt 名称追加 AUC）**：确认平台 ckpt 子目录约束为 `global_step` 前缀、≤300 字符、合法字符 `[a-zA-Z0-9_.=-]` 后，将 best checkpoint 命名扩展为 `global_stepN.layer=...head=...hidden=...auc=0.xxxxxx.best_model`。该改动只写入 local valid AUC 元信息，保留 `.best_model` 后缀与旧目录清理逻辑，不涉及 evaluation score / ensemble / 外部数据。

- **2026-05-02 PDT 14:10（训练结束摘要落地）**：`trainer.py` 新增 end-of-run `[summary]` 与 `training_summary.json`，记录 `best_val_AUC`、`best_val_logloss`、`best_epoch`、`best_global_step`、`best_checkpoint_dir`、last val metrics 与 compact AUC history。后续平台训练完成后直接搜 `[summary]` 或看 JSON，避免为了决定 evaluation 额度而翻 TensorBoard 曲线 / 长日志。

- **2026-05-02 PDT 13:50（修复后 full time jobs 回流）**：`swiglu-valid_split_strategy_time valid_ratio=0.10` 已 Early Stop，best local AUC 0.863248 @ step 53004，`overlap_sec=341068`；`swiglu-split_strategy_time-valid_ratio_0.05` 截至 epoch 9 仍在训练，当前 best local AUC 0.865884 @ step 48060，`overlap_sec=341086`。两者与 EXP-004 proxy 一致证明 RG-level time split 仍不 clean，T8.5 row-level timestamp cutoff 升为下一轮高优先级；evaluation 额度应优先等 0.05 任务完成后再投，0.10 暂不急评。

- **2026-05-02 PDT 10:40 / 12:10（EXP-004 fast proxy smoke 回流）**：`swiglu + valid_ratio=0.05 + 64/16 RG + max_train_steps=500` 平台 smoke 对照完成，验证 T11 链路有效；T8 日志显示 RG-level `time overlap_sec=302765`、`filename overlap_sec=304064`，强提示 T8.5 row-level timestamp cutoff；T10 切片表显示 `user_int_nonzero`、`item_int_nonzero`、`seq_total_len` 桶间 AUC 差均远超 0.02，强触发 ADR-003 P0-A/P0-B，但因 proxy 只训练 500 step，需等待 full job 复核后决定是否投 full train / leaderboard。

- **2026-05-01 PDT 19:35（T11 fast proxy 开关落地）**：训练链路新增 `--limit_train_rgs`、`--limit_valid_rgs`、`--max_train_steps`，默认 0 行为不变；fast proxy job 可以在常规 split 后限制 RG 数，并在指定 optimizer step 后立即 final validation + 保存 best_model，用于把 shape / 数据切分 / 初步收敛问题从 10h full train 前置暴露。

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
