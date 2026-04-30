# Active Tasks · TAAC 2026

> **单一事实源**。任何 task 的状态变更只在本文件落定。
> meta-spec（审视清单 / 状态约定 / 编号约定）见 `docs/20-tasks/README.md`。
> 已完成 `[x]` task 见 `docs/20-tasks/archive.md`。

---

**最近一次审视**：2026-04-30（深夜 · 知识库重组完成）· 结论：把原 `document/TODO.md` 387 行拆成 `20-tasks/{README, active, archive}` 三件套；T8 / T10 的决策推理外移到 ADR-001 / ADR-002；T1 的 demo 数字 + 平台首次 job 数字外移到 EXP-001 / EXP-002；ideas-backlog（旧 §6）外移到 `docs/10-research/ideas-backlog.md`。本文件现在只装**任务本身**，决策与实验数据全部链接外部，行数从 387 降到 ~150。**关键待办（按 ROI 排序）**：① T7 `[!]` infer.py 对齐（**初赛硬阻塞**，没它公榜永远没分） → 等首次 platform job 出 best_ckpt 后审查默认 infer.py；② T8 + T10 工具就绪 → 下一次 platform job 一并验证 overlap_sec + 4 张切片表，**单次评估同时回流两批信号**；③ T9 推理 30min profile（首次 ckpt publish 后跟 T7 一起做）；④ T4 编码器消融待 T7 解除阻塞后再消耗评估额度。距初赛 ddl 25 天（5/25）。

---

## 任务列表

### T4 `[~]` 复现 HyFormer 三种序列编码器消融

baseline 中三种编码器都已实现：`baseline/model.py` 的 `SwiGLUEncoder` / `TransformerEncoder` / `LongerEncoder`，由 `--seq_encoder_type {swiglu, transformer, longer}` 切换。

(2026-04-29 晚: 首个平台 job 默认 `transformer`，已等同 T4-B 首跑；下一步在平台 "Copy" 该 job 切 `swiglu` / `longer`。)

(2026-04-30 夜: **撤回"并行启动消融"判断** — 官方 4/27 答复明确「Each team allocated a single resource quota, shared across train+inference, **strictly serial**」；后已扩到 2 quota 但仍串行。**评估每天 3 次，北京时间 12:00 刷新**，失败/取消的不计数。**新约束**：三组消融 + 调参的总评估开销受限 9 次/3 天，T4-A swiglu 跑完出 AUC 才能起 T4-C longer；`--seed 42` 复现性 + best_ckpt 选择规则不变。)

**消融矩阵**（其它超参对齐）：

```text
共享基线参数：lr=1e-4, batch_size=512, num_hyformer_blocks=2,
            num_heads=4, d_model=64, num_queries=2,
            ns_tokenizer_type=rankmixer, user_ns_tokens=5, item_ns_tokens=2
```

```bash
# A. SwiGLU（最低延迟，无 attention）
bash baseline/run.sh --seq_encoder_type swiglu

# B. Transformer（最高表达，O(L^2)）
bash baseline/run.sh --seq_encoder_type transformer

# C. LONGER（折中，cross-attn top_k 压缩）
bash baseline/run.sh --seq_encoder_type longer --seq_top_k 50

# D. （可选）LONGER + causal mask
bash baseline/run.sh --seq_encoder_type longer --seq_top_k 50 --seq_causal
```

每组用相同 `--seed 42` 跑 ≥ 3 epoch，记录：`Val AUC / LogLoss / Params / FLOPs / Step latency / GPU peak mem / **Infer time**`（infer time 列与 T9 联动）。

**期望**（参考 HyFormer 论文 Table 1）：

```text
Transformer ≥ LONGER ≥ SwiGLU （AUC）
SwiGLU      ≪  LONGER ≪ Transformer （FLOPs）
LONGER 在初赛 100+ 字段、复赛 10× 数据上一般 ROI 最高。
```

**阻塞**：T7（infer.py 对齐）未解除前，跑了也上不了榜，纯烧评估额度。

---

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

### T6 `[!]` 草拟最终方案技术报告（KDD 答辩用）

**阻塞点**：缺至少 T1（baseline 数）+ T4（编码器消融数），否则报告会空洞。

**解除条件**：T1 + T4 中至少 2 组配置跑出可信 AUC 数字。

**模板规划**：参考 HyFormer / OneTrans / RankMixer 三篇论文的 Section 结构（Motivation / Method / Experiments / Online A/B / Discussion）。

---

### T7 `[!]` 修改平台默认 `infer.py` 与自定义训练配置对齐

(2026-04-30 夜: **`[~]` → `[!]` 升级**。社区情报锁定结论 — 官方 4/28 答复「`infer.py` / `model.py` used during inference must be exactly identical to the one used during training」，详见 `docs/10-research/community-intel.md` "推理模型load错误" 一题。配合多条「评估出 AUC 但不上榜」的报告（4 条独立投诉，跨 4/26→4/29）→ 印证不少队伍正撞这个坑。我们已偏离 baseline 默认（`ns_tokenizer=rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2`，T4 还会切 `seq_encoder`），不修 infer.py 必然 Eval Failed。)

**阻塞条件**：
- 必须等首个平台 baseline job 跑出 best_ckpt 并 Publish 后，从 Evaluation 页下载默认 `infer.py (15.6K)` + `model.py (63.6K)` + `dataset.py (30K)`，先 diff 看其加载策略

**解除条件（任一即可）**：
1. 默认 infer.py 走 `train_config.json` 完全动态构建 model → 直接用 → 改回 `[~]` 然后结案
2. 默认 infer.py 含硬编码超参 → 修改 infer.py 三件套（model.py / infer.py）使之与训练完全一致（**修改现有文件，不新建，符合最小变更红线**） → 提交评估 → 公榜出现自定义模型分数 → 关闭

**审查 checklist**（下载默认文件后逐条核对）：
1. infer.py 是否 `json.load("train_config.json")` 动态读取 `ns_tokenizer_type / seq_encoder_type / num_queries / user_ns_tokens / item_ns_tokens` 来构建 model
2. model.py 类签名是否与我们训练时 import 的一致（可能我们在自定义 model.py 加了新 nn.Module → 必须同步到 infer 包）
3. dataset.py infer 时是否走 `is_training=False` 路径（label 列全 0，不依赖 label_type）
4. 推理输出格式（predictions 字段名 / user_id 覆盖率）是否能产出 leaderboard 期望的字段；社区有题报「Missing 310000 answer user_ids in predictions（need full coverage）」→ 推理必须**全量覆盖测试集 user_id**

**风险评估**：
- 评估管线断 → 训练分数永远上不了榜（本届硬阻塞，无 workaround）
- 不影响数据 / 采样 / 权重 / 特征训练侧，**无 metric 影响，纯工程对齐**
- 与 T9（推理 30min 硬约束）联动 — 改 infer.py 时顺便测推理时长

---

### T8 `[~]` 把验证集切分从 random 改为按时间切

**代码已落地**（2026-04-30）— 详见决策推理 [`docs/30-decisions/ADR-001-time-aware-valid-split.md`](../30-decisions/ADR-001-time-aware-valid-split.md)。

**当前状态**：
- ✅ `--valid_split_strategy {filename,time}` CLI flag 落地，默认 `filename`（行为不变）
- ✅ sanity log 自动打印 train/valid ts 范围 + `overlap_sec`
- ✅ 本地 demo 验证（filename overlap=765s / time overlap=770s，是 demo 数据特性）
- ⏳ **待平台 job 验证 overlap_sec 真实值** → 决定是否切换到 `time` 策略

**残留待平台验证**（下一次 platform job 时）：
1. 默认参数（`filename`）跑一次 → 看 `[T8] train ts: ... valid ts: ... overlap_sec: N` 这一行
2. 若 `overlap_sec > 0` → 加 `--valid_split_strategy time` 再跑一次，看 overlap_sec 是否 ≤ 0
3. 若 `time` 模式仍 > 0 → 升级 T8.5（行级切，改在 `PCVRParquetDataset._convert_batch` 里按 row 的 `timestamp` 与 cutoff 比较、丢早样本）

**待自验证的两条社区线索**（仅留 trace，不动当前实现）：
1. **训练集 timestamp 严重不均衡**：社区一截图示 train 跨 02-28~03-04 共 5 天，**03-03 单日 1.31M / 总 2.10M ≈ 63%**，其它天 3~25%。若属实 → "取最后 10%" 实际几乎全是 03-04 末段；按 timestamp 排序后用更小切片（如 5%）或许更接近真实 future 边缘。
2. **疑似时间穿越**：评估日志中出现 `2026-02-27 11:00:13`（早于 train min `2026-02-28 00:35:09`），但字段标签不明（可能是 `timestamp` / `label_time` / 序列内 ts_fid 之一）；下次 job 在 train + eval 各打印 `timestamp.{p1,p50,p99}` + 序列 `ts_fid.{p1,p50,p99}` 自验证。

---

### T9 `[ ]` 测推理时长 + 控在 30min 硬上限以内

(2026-04-30 夜新增 · 触发证据: 官方 4/27 答复「A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks」，超时即被系统 kill。社区已有 4 条「推理被 stop」投诉。)

**最小动作**（首次 ckpt publish 后 30 分钟）：
1. publish 第一个 ckpt 后看 evaluation log 里的 `Inference Time` 字段（社区案例报过 625.28s ≈ 10min，距 30min 还有余量）
2. 若 > 1500s（留 5min 安全边际） → 立即降 batch_size 推理 / 切 `longer` 编码器（top_k 压缩）
3. 数字回填 → `docs/40-experiments/EXP-002-platform-baseline-job.md` 作为 baseline 推理性能基线

**联动**：
- **T7**：改 infer.py 时顺手把推理 batch 调成显存允许的最大值，加 `torch.no_grad()` + bf16 推理（若 baseline 未启）
- **T4**：消融时记录每组的「Train AUC / Eval AUC / Train time / **Infer time** / Params」

**风险评估**：
- 不影响训练 / 数据 / 权重，只在推理阶段调 batch + dtype
- 若 bf16 vs fp32 推理结果 AUC 差异 > 0.001 → 退回 fp32，改用 batch 调优

---

### T10 `[~]` valid 集切片诊断 — 工具已落地，待平台 job 数据决定 P0 特征工程方向

**代码已落地**（2026-04-30）— 详见决策推理 [`docs/30-decisions/ADR-002-slice-diagnostics.md`](../30-decisions/ADR-002-slice-diagnostics.md)。

**当前状态**：
- ✅ `dataset.py` 加 4 个 `_diag_*` 字段 + `trainer.py::_slice_diagnose` 方法
- ✅ 默认每次 evaluate 都跑，无 CLI 开关，开销 < 1s
- ✅ 推理路径完全无影响（`is_training=False` 不写入 `_diag_*`）
- ⏳ **待平台 job 4 张切片表回流** → 决定 P0 特征工程方向

**待平台 job 验证后再分类决策**（每个分支对应不同的 P0 特征工程方向）：

| 平台 job 切片表观察 | 解读 | P0 优先级动作 |
|---|---|---|
| `label_type=0` 占比 > 50% | 任务是 CTCVR（曝光为主） | 维持 baseline 损失，重点做 user/item 缺失计数 dense 特征 |
| `label_type=0` 占比 < 5% | 任务是 click-conditional CVR | 重新评估损失（可能需要 IPS 或 ESMM 风格辅助任务） |
| `user_int_nonzero` 桶间 AUC 差 > 0.02 | 用户冷启动是薄弱点 | P0：把 `user_int_nonzero` 自身做 dense 特征（带分桶 embedding） |
| `item_int_nonzero` 桶间 AUC 差 > 0.02 | 商品冷启动是薄弱点 | P0：item_int_nonzero 同上 + item_id 历史 ctr/cvr 统计 |
| `seq_total_len` 桶间 AUC 差 > 0.02 | 用户活跃度是薄弱点 | P0：每个 domain 的 seq_len 单独成 dense 特征 |
| `label_type=1` 与 `=2` 的 pred_mean 差距小 | 模型未充分区分点击 vs 转化 | P1：转辅助任务（CTR + CVR multi-task） |

**最小动作**（下次平台 job 触发后 30 分钟内）：
1. 看 evaluate log 里 `[diag]` 开头的 4 张表
2. 对照上表，把对应 P0 改动写成 30 行 diff 提交
3. 表里数字 + 决策结论 → 落 `docs/40-experiments/EXP-NNN-platform-XXX.md`，本节只留链接

**与 T8 联动**：T8 sanity log 与 T10 4 张表都在 evaluate 输出里，**一次 platform job 同时回流两批信号**。

---

## Changelog

- **2026-04-30 深夜（第三轮 · 知识库重组）**：把原 `document/TODO.md` 387 行 + `document/knowledge_base.txt` 340 行重组为 6 个职责单一的子目录（详见 `docs/README.md`）。本文件 `active.md` 现在只装任务本体；T8 / T10 决策推理外移到 ADR-001 / ADR-002；T1 / EXP-002 数字外移到 `40-experiments/`。原顶部累积的 3 条「最近一次审视」压缩为 1 条 + Changelog 入口。Cursor rule `.cursor/rules/todo-review.mdc` 路径绑定同步更新（`document/TODO.md` → `docs/20-tasks/active.md`）。

- **2026-04-30 深夜（第二轮）**：T10 valid 集切片诊断落地。改动 2 个文件 / ~120 行 / 无 lint 错误。详见 `docs/30-decisions/ADR-002-slice-diagnostics.md`。

- **2026-04-30 深夜**：T8 代码落地。改动 3 个文件 / +85 行，无 lint 错误。详见 `docs/30-decisions/ADR-001-time-aware-valid-split.md`。

- **2026-04-30 夜**：抓取乐享比赛社区 141 题（`docs/99-raw/community-snapshots/lexiang_questions_2026-04-30-06-46-51.{json,md}`），按 organization 字段过滤出 26 条 Administrators / TAAC 官方答复，沉淀到 `docs/10-research/community-intel.md`。本次审视带来 5 项硬变更：① **T7 `[~]` → `[!]` 升级** — 官方明确「infer.py 必须与训练一致」；② **新增 T8** 验证集按时间切；③ **新增 T9** 推理 30min profile；④ T4 撤回"并行启动消融"判断；⑤ T2/T3 inline 加情报佐证。距初赛 ddl 25 天。

- **2026-04-30 晨**：平台首个 baseline job 启动成功（GPU / batch=256 / 1.9M 行 / 1000 分片）。详见 `docs/40-experiments/EXP-002-platform-baseline-job.md`。沉淀三件事：① **T3 `[x]` 结案**（搬到 archive）；② **T7 修正** — 平台 Evaluation 页**已预置 infer.py (15.6K)**，原"新建文件"计划作废，改为「Publish 后下载默认 infer.py 审查兼容性」；③ T4 启动 swiglu 第一组消融。

- **2026-04-29 晚**：第一轮 baseline 提交平台训练成功（默认配置 = `transformer` 编码器，等同 T4-B 首跑）。新增 **T7「补 `infer.py`」**（评估硬约束）；T3 仍 `[!]`。

- **2026-04-29**：T1 环境切换 venv → conda（`tencent-unirec`，python 3.11.15 / torch 2.10.0 CPU）。详见 EXP-001。

- **2026-04-28**：T1 完成。详见 EXP-001。

- **2026-04-26**：从 `knowledge_base.txt §8` 迁出 6 条 TODO；T5 决议暂缓；新增 7 步审视清单。
