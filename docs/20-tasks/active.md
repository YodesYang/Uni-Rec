# Active Tasks · TAAC 2026

> **单一事实源**。任何 task 的状态变更只在本文件落定。
> meta-spec（审视清单 / 状态约定 / 编号约定）见 `docs/20-tasks/README.md`。
> 已完成 `[x]` task 见 `docs/20-tasks/archive.md`。

---

**最近一次审视**：2026-05-04 晚 5（PDT · T17 S3 proxy 回流 · T16 half-train 进行中）· 结论：① **T17 S3 proxy 完成**（job `t17-holdout-s3-t13tf-proxy...s1500`）：dev AUC 0.839843 / holdout AUC 0.839957 / **gap = −0.000114**（近零），落在"无法证实也无法证伪"档；② **gap≈0 机制明确**：proxy 仅 1500 steps ≈ E1 的 1/14，未进入过拟合区（EXP-018 那 0.023 gap 是 E5 后产物），5%+5% 切分时间跨度小于 train→test 真实跳跃；③ **ADR-005 维持 Proposed**，S4 是真正判决点；④ **S4 搭车策略**（零 training quota 成本）：下一个 half-train 默认加 `--holdout_ratio 0.05`，观察 E1~E3 每 epoch 四指标 `(dev/holdout × AUC/LogLoss)` 轨迹；⑤ **T16 half-train 进行中**（另一会话推荐并启动）；⑥ **今日 evaluation quota**：已用 1（t13tf-loss-full），剩 3；⑦ **training quota**：已用 2（T16 proxy + T17 S3 proxy），1 个被 T16 half-train 占用。

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
- ✅ **ICQ half-train 回流（2026-05-04）**：`icq-t13tf-halftrain-swiglu-vr005-bs128` E4 best AUC 0.868619 / LogLoss 0.223356，vs t13tf EXP-018 E4 0.868495，Δ+0.000124；proxy 时负向翻转为正，决策推进 full train。详见 [`EXP-019`](../40-experiments/EXP-019-icq-halftrain-2026-05-04.md)。
- ⏳ **ICQ full train 待启动**：Job Name `icq-t13tf-full-swiglu-vr005-bs128`（10 epoch），evaluation 阈值 ≥ 0.870。
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

### T13 `[~]` domain_c_seq_47 vocab 截断（高基数 item ID 处理）

**来源**：社区情报 Stochastic 陷阱三："domain_c_seq_47 的 item ID 量级与顶层 item_id 相同（0~278M）"；我们的 schema 确认 vocab=278677640。

**当前状态（bug）**：`run.sh` 里 `--emb_skip_threshold 1000000`，fid=47 的 vocab=278M 超过阈值 → **整个特征输出全零向量**，等于完全丢弃了 domain_c 序列里的主 item ID 信号。

**修复方案**：两条路：
- **方案 A（截断）**：在 `dataset.py` 的 `_convert_batch` 里，对 vocab 超大的 item ID 特征，将值限制在 `[0, clip_threshold]`，超出归 UNK（0），这样可以保留高频 ID，同时 embedding table 大小可控
- **方案 B（hash embedding）**：在模型侧对超大 vocab 的特征做 hash 取模，映射到固定大小的 embedding table（例如取模到 1M），损失少量 collision，但保留了所有 ID 的可学习表示

**分层漏斗**：
1. 本地 smoke 验证 embedding table 大小和显存
2. ⚠️ （方案 B）proxy 阶段：`t13-hash500k-timefeat-proxy-swiglu-vr005-rg128x32-s1500` AUC 0.837684 vs timefeat-only 0.839285，Δ−0.001601 → proxy 负向。详见 [`EXP-015`](../40-experiments/EXP-015-t13-hash-proxy-2026-05-04.md)
3. ✅ **（方案 B · hash500k）full train 翻正**：`t13tf-full-swiglu-vr005-bs128` best local AUC 0.868495（E4），leaderboard 0.844956，vs no-P0 time baseline +0.000897 LB。`t13only-full` E7 best AUC 0.866571（hash500k 单独 Δ+0.000687 vs baseline 0.865884）。hash500k 与 timefeat **几乎是独立可加的**（初版 E4 快照误判"+0.001752 强交互"）。详见 [`EXP-018`](../40-experiments/EXP-018-t13tf-full-train-2026-05-04.md)
4. ⚠️ **（方案 B · hash2m）full train 无显著优势**：`t13-hash2m-timefeat-full-swiglu-vr005-bs128` E9 best 0.868483，几乎持平 hash500k E4 的 0.868495（Δ−0.000012），proxy 显示 2M 低碰撞优势**未能在 full train 兑现**。假说：本项目数据规模下「可控碰撞 = 隐式正则化」，已入 team-memory 铁律
5. 方案 A（vocab 截断）尚未实验，作为备选

**当前优先级**：低-中。方案 B hash500k 已作为 `t13tf-full` 一部分上线并贡献约 +0.0007 local AUC；不再继续加桶数（2M 已验证无增益）。后续可考虑组合：hash500k + **AUC-select** EarlyStopping + 更长训练（看 t13only 能否在 E10+ 继续上涨）。方案 A 暂不优先——要和 target-item 硬交互等更高潜力方向竞争 quota。

### T16 `[~]` Target-Item 硬交互注入序列侧（ADR-004 方案 A · Additive）

**决策推理外移至 [ADR-004](../30-decisions/ADR-004-target-item-seq-conditioning.md)**。本节只记任务状态与 next action。

**背景一句话**：MultiSeqHyFormer 的 seq encoder 当前是 **target-agnostic** 的（target 只通过 query 端进入 via ICQ）。在 key/value 侧也显式注入 target-item 表征 → 从根源上让 encoder 变 target-aware, 是当前理论上**破局 SOTA 可能性最高的单点改造**。

**已完成**：
- ✅ **S1 本地 smoke 通过**（2026-05-04 晚）: `TargetItemSeqInjector` 类 + CLI 参数 + train_config 落盘 + infer 兼容, ~120 行代码, demo 10 step train + 1000 predictions + 可复现性验证全部 OK
- ✅ ADR-004 Draft → Accepted

**待验证**（ADR-004 §Verification 表 S2~S5）:

| 阶段 | 环境 | 期望信号 | 决策阈值 |
|---|---|---|---|
| S2 fast proxy | 128/32 RG + 1500 steps | 相对 t13tf-only proxy 基线 AUC 不退超过 0.002 | 退 <0.002 继续 S3; 退 >0.002 **也进** S3(ICQ "大参数假负向"规律) |
| S3 half-train | 2~3 epoch 全 RG | E3 best AUC ≥ t13tf E3 best + 0.0002 | 达标进 S4 |
| S4 full train | 10 epoch + batch 128 | best AUC ≥ 0.869 | 达标进 S5 |
| S5 leaderboard | 1 evaluation slot | LB ≥ 0.845 | 达标 commit 方案 A |

**下一步**：
1. 明日启动平台 fast proxy job: `t16-target-add-a01-proxy-swiglu-vr005-rg128x32-s1500`（参数 `--target_item_seq_injection additive --target_inject_alpha_init 0.1`）
2. 保留 `icq-t13tf-full` 回流数据作为 "ICQ only" 基线对照
3. 若 S2 proxy 通过, S3 half-train 作 `t16-target-add-a01-halftrain-...`
4. **S4 full train 默认带双指标 ES 保护**（2026-05-04 落地）：`--logloss_rebound_patience 1`（默认值），一旦 val_logloss 反弹即停；best_model 仍按 AUC 选择。这大幅降低 "训过头 → LB 反而掉" 的风险（参考 t13tf-loss-full E4→E9 教训）。

**当前优先级**：**最高**。距 SOTA 还差约 0.006 LB, ADR-003 的 P0-A/B/C 全部失效后, T16 是破局可能性最大的剩余单点方向。evaluation 额度策略: 先不占用任何 evaluation slot, 等 full train 回流后再决定。**注意**：即便 T16 full train local AUC 显著领先, 若 local-LB gap 仍保持在 0.023~0.030（ADR-005 未落地前的现状）, 也只有当 **local AUC ≥ 0.870** 时才可能超越 t13tf-full 的 LB 0.844956 ——**局 = LB 而非 local AUC**。

---

### T17 `[~]` Valid Split Strategy Redesign（ADR-005 Proposed）

**决策推理外移至 [ADR-005](../30-decisions/ADR-005-valid-split-redesign.md)**（Status: Proposed）。本节只记任务状态。

**触发**：2026-05-04 晚 `t13tf-loss-full` LB = 0.838707 事件 + 7 次历史提交 gap 扫描（0.021~0.030 稳定），gap 是系统性的 valid 定义偏差。

**核心假说**：当前 time-split 尾部 valid 与训练尾部紧邻 → 模型在 valid 上"近未来延续"很成功但在 test"远未来外推"失败。需要 holdout split 提供 LB 代理信号。

**候选方案**（详见 ADR-005 §Decision）：
- **方案 B（优先）**：dev + holdout 双 valid，holdout 训练期间完全不访问
- **方案 C（次选）**：train/valid 之间留时间空窗，模拟 test-时间-gap
- 方案 A/D 作为 fallback / 不推荐

**已完成**：
- ✅ ADR-005 Draft → Proposed
- ✅ trainer.py LogLoss rebound guard 落地（与 T17 并行的短期止损措施）
- ✅ **代码落地完成**（dataset.py `holdout_ratio` 切分 + trainer.py `holdout_loader` + `evaluate_holdout()` + train.py `--holdout_ratio` CLI · backward-compat 保持）
- ✅ **S0/S1 本地验证通过**（`py_compile` + demo smoke 3 条路径：enabled 三路切分正确、backward-compat 零开销、guard 报错路径工作）
- ✅ **S3 platform proxy 完成**（EXP-021）：dev=0.839843 / holdout=0.839957 / gap=−0.000114（近零），落在"无法证实也无法证伪"档；机制明确（proxy 尺度未进入过拟合区 + 切分时间跨度太小），ADR-005 维持 Proposed

**下一步（S4 搭车策略，零 training quota 成本）**：
1. **下一个 half-train（2~3 epoch 全数据）默认加 `--holdout_ratio 0.05`**，无论是 T16 seed-2 复现还是新实验，都零成本附带采集 S4 数据
2. 观察 E1~E3 每 epoch 的 `(dev_auc, holdout_auc, dev_logloss, holdout_logloss)` 四指标轨迹：
   - **如果**中后期出现 `dev_auc 继续涨 + holdout_auc 滞涨/下降` 或 dev LogLoss 提前 holdout 反弹 → Accept，方案 B 可作 LB 代理
   - **如果** E1~E3 全程 gap < 0.003 → Reject 方案 B，切到方案 C（留时间空窗）
3. S5：若 S4 Accept，用 1 次 evaluation quota 做 LB 校准

**当前优先级**：中（S4 搭车实现，等 T16 half-train 回流）。ADR-005 的判决权已从独立 job 转移到"下一次 half-train 附带"，节省 1 个 training quota。

**何时 Accept ADR-005**：S4 half-train 数据显示 dev/holdout 在过拟合区分叉（gap > 0.005）且方向与 EXP-018 LB gap 一致后。

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

- **2026-05-04 PDT 晚 5（T17 S3 proxy 回流 · ADR-005 暂缓判决 · S4 搭车策略）**：`t17-holdout-s3-t13tf-proxy` 跑完，dev AUC 0.839843 / holdout AUC 0.839957 / **gap = −0.000114**（近零）。按 S3 事先判据落在"无法证实也无法证伪"档。三点机制解释写入 EXP-021：① proxy 仅 1500 steps ≈ E1 的 1/14 未进过拟合区（EXP-018 的 0.023 gap 是 E5 后产物），dev/holdout 本就不该分叉；② 5%+5% 内部切分时间跨度远小于 train→test 真实跳跃（至少周级），信号稀释；③ proxy 阶段 dev/holdout LogLoss 方向一致说明同质分布区。**ADR-005 维持 Proposed**，判决权移到 S4。**S4 搭车策略**（节省 1 training quota）：下一个 half-train 默认加 `--holdout_ratio 0.05`，观察 E1~E3 四指标轨迹；若中后期出现 `dev AUC 继续涨 + holdout 滞涨` 或 dev LogLoss 提前反弹 → Accept；若 E1~E3 全程 gap<0.003 → Reject 方案 B 切方案 C。新增 EXP-021 完整记录 proxy 配置/结果/机制/S4 判据。

- **2026-05-04 PDT 晚 4（T17 代码落地 · ADR-005 S0/S1 验证通过）**：按 ADR-005 方案 B 落地 dev+holdout 双 valid 框架。① `baseline/dataset.py` `get_pcvr_data` 新增 `holdout_ratio` 参数，time-split 下切分顺序改为 `[train][dev (valid_ratio)][holdout (holdout_ratio)]` 三路 disjoint；返回值从 3 元组扩展到 4 元组 `(train_loader, valid_loader, holdout_loader, train_dataset)`。② `baseline/trainer.py` `__init__` 加 `holdout_loader: Optional[DataLoader]` 参数；`evaluate()` 重构为 `_evaluate_on_loader(loader, split_tag, run_slice_diag)` 的薄壳，新增 `evaluate_holdout()` 方法在每次 validation 后被 `_record_validation_result` 调用；holdout 指标**只记日志 + 写进 `validation_history[i]['holdout_auc/logloss']`，不参与 best_model / EarlyStopping / LogLoss guard**。③ `baseline/train.py` 加 `--holdout_ratio` CLI（默认 0.0 保 backward-compat），unpack 新 4 元组并传 `holdout_loader` 给 trainer。④ Guard 设计：`holdout_ratio>0` 必须配 `valid_split_strategy='time'`（早期 ValueError），且与 `enable_row_time_cutoff` 互斥。⑤ **S0/S1 本地验证全过**：`py_compile` exit 0；demo smoke 实测 10 RG → 8 train/1 valid/2 holdout 正确分配，日志流出 `[ADR-005] Epoch 1 step 5 holdout AUC: 0.631605, LogLoss: 0.390470 (valid AUC: 0.576313, ... gap_auc=valid-holdout=-0.055293)`；backward-compat 路径（不传 flag）零 `[ADR-005]` 日志，与旧 job 100% 等价。ADR-005 保持 Proposed，等 S3 平台 proxy 回流后再 Accept。

- **2026-05-04 PDT 晚 3（t13tf-loss-full LB 回流 + 双指标 ES 落地 + ADR-005 起草）**：`t13tf-loss-full-swiglu-vr005-bs128_eval_1777927874214` LB AUC = **0.838707**，vs `t13tf-full E4` LB = 0.844956，Δ−0.006249，valid AUC 反而 +0.000494。epoch 表诊断（E1~E8 完整）显示 E4→E5 是过拟合清晰拐点：AUC +0.000583 但 LogLoss +0.000608 反弹，之后多训 4 epoch **local AUC 仅再涨 0.000640 但 LB 掉 0.006249（10× 负向杠杆）**。三项响应动作：① **trainer.py 落地 LogLoss rebound guard**（`--logloss_rebound_patience` 默认 1 / `--logloss_rebound_delta` 默认 0.0）：AUC 仍作 best_model 选择（比赛指标对齐），val_logloss 反弹即停；本地 smoke 双路径（enabled 触发 + disabled 保留老行为）验证通过，guard 日志 `[logloss-guard]` + stop_reason `logloss_rebound_epoch_N` 准确。② **EXP-018 回填** 全 epoch 表 + 7 次历史 gap 扫描（0.021~0.030），LB 实测 0.838707 并重写"教训"段。③ **team-memory** 两条新铁律：EarlyStopping 多指标联合（旧的"按 AUC 选"单指标规则被合并推翻）+ valid→LB gap 系统性（决策红线改以 "能否 > LB 0.844956" 为准,不以 local AUC 为准）。④ **ADR-005 Proposed**（Valid Split Strategy Redesign）：候选方案 B (dev+holdout) 优先，需 S3 proxy 验证 holdout AUC 对 LB 的预测能力。⑤ **T17 新增**（ADR-005 落地任务），与 T16 并行推进。⑥ T16 S4 full train 默认带双指标 ES 保护，过训练风险降低。今日 evaluation quota 剩 3。

- **2026-05-04 PDT 晚 2（T16 代码落地 + 平台 job 收口 + 评估额度修正）**：① T16（ADR-004 方案 A Additive）代码落地: `baseline/model.py` 新增 `TargetItemSeqInjector` 类（`off` / `additive` 两模式,α init=0 为 identity safe 起点）,`PCVRHyFormer.forward` / `predict` 前注入, train.py 加 `--target_item_seq_injection` + `--target_inject_alpha_init`, infer.py 同步读, 共 ~120 行代码; S1 本地 smoke 全部通过（demo 10 step train + 1000 predictions + 可复现性 + 注入生效）。② ADR-004 Draft → Accepted, 索引更新。③ 平台 `t13tf-loss-full` E9 升到 0.868989（目前所有 full-train 最高）,决定消耗 1 个 evaluation slot（Publish 前走三步检查,确认 ckpt 是 E9 AUC 0.868989 那次）;`t13-hash2m-timefeat-full` + `t13only-full` 停止释放 training quota; `icq-t13tf-full` 保留继续跑。④ **评估配额事实修正**: team-memory 原写"4 次/天", 实际为 **24h 滚动窗口 4 次,PDT 21:00 重置,PDT 24:00 榜单刷新** —— 今日 4 slot 到 PDT 21:00 前都未消耗, 仍可用。

- **2026-05-04 PDT 晚（四路 full-train epoch 回流 + EXP-018 归因修正）**：用户同步 `t13-hash2m-timefeat-full` E9 0.868483 / `t13only-full` E7 **0.866571** / `t13tf-loss-full` E5 实际 best 0.868932 被 E8 0.868571 覆盖 / `icq-t13tf-full` 刚起第一个 epoch。触发三项修正：① EXP-018 初版 E4 快照判 hash500k 单独 Δ−0.001057 有害 → E7 修正为 Δ+0.000687 正向,hash500k 与 timefeat **几乎独立可加**（原 "+0.001752 强交互" 是 E4 快照假象）;② 2M 桶 full train 相对 500k 几乎持平,"可控碰撞 = 隐式正则化" 假说成立;③ LogLoss-select EarlyStopping 静默丢弃 AUC 最优 ckpt（E5 0.868932 被 overwrite）。team-memory 补 2 条铁律（hash vocab size / AUC-select）,T13 升 `[~]` 并标记方案 B hash500k 已为 t13tf-full 正式组件。当前仅 `icq-t13tf-full` 值得等结果；若 ≥ 0.870 消耗一个 evaluation 配额。evaluation 额度修正：**严格 4 次/天，不可累积不可弹性**。

- **2026-05-04 PDT（ICQ half-train 回流 → 推进 full train）**：`icq-t13tf-halftrain-swiglu-vr005-bs128` E4 best AUC 0.868619 / LogLoss 0.223356，vs t13tf（EXP-018 E4）0.868495，Δ+0.000124；训练曲线全程单调上升，无早停迹象。ICQ proxy（Δ−0.001417）→ half-train 正向，再次验证"大参数分支 proxy 假负向"规律。决策：推进 `icq-t13tf-full-swiglu-vr005-bs128`（10 epoch），evaluation 阈值 ≥ 0.870，不评估 half-train E4。详见 [`EXP-019`](../40-experiments/EXP-019-icq-halftrain-2026-05-04.md)。

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
