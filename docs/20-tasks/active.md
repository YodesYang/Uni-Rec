# Active Tasks · TAAC 2026

> **单一事实源**。任何 task 的状态变更只在本文件落定。
> meta-spec（审视清单 / 状态约定 / 编号约定）见 `docs/20-tasks/README.md`。
> 已完成 `[x]` task 见 `docs/20-tasks/archive.md`。

---

**最近一次审视**：2026-05-13 西班牙下午 14:40（UTC+2 · 🧱 **T30 + T31 代码并行落地 · smoke 通过 · 等 M16 EDA 回流决定参数**）· 结论：① **CEST 12:20-13:30 T30 + T31 并行开发完成**（用户请求"同时推进"）· 利用 M16 EDA 训练 6-8h 窗口做代码开发 · 无训练 slot 阻塞。② **T30（时间桶可配置）** · `baseline/dataset.py` `DEFAULT_BUCKET_BOUNDARIES` + `parse_bucket_boundaries()` helper · 严格长度校验（63 values）+ 单调 + 正数 · CLI `--time_bucket_boundaries` · infer.py fallback · **默认值未变**（backward-compat 严格）· [`EXP-043`](../40-experiments/EXP-043-t30-time-bucket-config-2026-05-13.md) 落档。③ **T31（is-missing 信号 Scheme D 双路冗余）** · 用户 CEST 12:50 选 Scheme D（Path A vocab+1 top slot + Path B user_dense 尾部 bitmap）· `_resolve_is_missing_plan` + `_link_is_missing_plan_col_idx` + `_convert_batch` 新 path · **model 零改动**（纯 dataset 层 · 不触发 "+X% 参数 必死" 铁律）· CLI 3 个 flags · [`EXP-044`](../40-experiments/EXP-044-t31-is-missing-scheme-d-2026-05-13.md) 落档。④ **本地 Smoke 验证全部通过**：T30（默认值 bit-identical · 非法值 raise · 自定义切换生效）· T31（关闭时 user_dense_dim=755 不变 · 开启时精确 +N · bitmap ∈ {0,1} · vs-1 路由正确 · fid 86/94 missing_rate 75%/62.5% 方向对）。⑤ **前置条件**：两 task 都等 M16 CEST 18:00-20:00 EDA 回流 · T30 用 `q5_time_diff` 生成分位数边界 · T31 用 `q2_user_int_missing_top` 筛 |cvr_gap|>2% 的 fid。⑥ **下一步**：M16 回流后 → 写 `scripts/gen_time_bucket_quantile.py`（~20 行）+ 基于 EDA 选 T31 fid 清单 → 半程验证 job（阈值 vs icq-half-gap05 E3 Dev 0.842443 · ≥0.8437 真信号 / 0.8433-0.8437 rerun / <0.8430 deprecated）。⑦ **本地 41 commits 未 push** · T30+T31 代码待 commit（2 EXP + dataset/train/infer）。⑧ **M14/M15 training 仍在跑** · 今晚回流。

---

**上一条审视（保留参考）**：2026-05-13 西班牙中午 12:00（UTC+2 · 🔬 **M9 破 SOTA + T25+T29 叠加双 seed 证伪 + EDA 搭车 M16 启动**）· 结论：① **今早 4 LB 回流**：🏆 **M9 (gap=0×T25×s42) LB 0.84671** 刷新 SOTA (+0.00034 vs 旧 0.84637) · 但 Δ < 1× seed 噪声 · 战略价值 > 数字意义；❌ M7 (T25+T29×s3 叠加) **LB 0.839593** Dev 高但 LB 暴跌 Δ−0.00348 vs T25 bypass s3 · Dev→LB gap 扩 15× · "LogLoss 低 ≠ LB 好" 反例；❌ M10 (T25+T29×s42 叠加) LB **0.84145** Δ−0.00198 vs T25 B · **M7/M10 双 seed 一致证伪 T25+T29 叠加方向**（昨天"seed 敏感"铁律修正为"叠加本身不 work"）；🟡 M11 (T29 only s3) LB **0.841455** Δ+0.00085 vs M3 s42 · T29 跨 seed 弱稳定但不如 T25 bypass 单独。（完整 changelog 见下方）

---

## 任务列表

### T11 `[~]` 建立快速实验漏斗，降低 10h full train 探索成本

**决策（2026-05-01）**：full train 只作为高置信候选复核，不再作为默认探索路径。所有非平凡代码改动 / 超参想法先走 fast proxy，再决定是否消耗完整训练与评估额度。

**2026-05-12 CEST 21:20 补充**：
- ✅ **所有 new training job 默认 `--keep_topk_ckpt 4`**（保存 top 4 ckpt，规避 "best_model 自动选错"风险，给 eval 多选择空间 · EXP-019 + 杭仔情报教训同向）
- ✅ **Half-train 默认 4 epoch + 松 LogLoss guard**（`delta=0.003 patience=2`）· 本赛题 E3-E4 是"最佳甜蜜点"不是"欠拟合"（EXP-019 铁律：full E6 Dev 更高但 LB 降 0.003）

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

### T18 `[x]` CombinedAUCLoss — ❌ 已证伪关闭（LB −0.00993 vs SOTA）

**2026-05-04 晚 10 终结**：三合一 E4 LB **0.836445** vs SOTA 0.84637 **Δ−0.00993**,ADR-006 Deprecated。BPR 破坏 sigmoid calibration,在"远未来 test"场景下 gap 放大 3~4×。详见 [`ADR-006 §Deprecation Notice`](../30-decisions/ADR-006-combined-auc-loss.md) 与 [`EXP-022 §终结状态`](../40-experiments/EXP-022-t18-auc-loss-smoke-2026-05-04.md)。代码保留(`combined_auc_loss` 纯函数),默认 `loss_type=bce`。

---

### T19 `[!]` RankMixer Per-token FFN — ⏸️ 2026-05-09 CEST 10:10 状态修正：Inconclusive-needs-rerun（原 09:43 Deprecated 判决基于错误 baseline 对比已撤回）

**2026-05-09 CEST 10:10 修正**：agent 在 09:43 下 Deprecated 判决时犯了严重错误——**用 gap=0 配置的 baseline 数字对比 gap=0.05 配置的 T19**，得出"ΔDev E2 −0.00098 + LogLoss Δ+0.057 异常"错判。**真实 gap=0.05 baseline 数据（EXP-030 §C）**：E2 AUC 0.837453（反弹）/ E2 LogLoss 0.28516（反弹）。T19 实测 E2 AUC **0.840142**（**+0.0027 真信号级**）/ E2 LogLoss **0.28208**（**−0.003 优于 baseline**）。

**T19 被 LogLoss guard 早停不是方向失败**：
- T19 E1→E2 LogLoss 反弹 +0.00035（很小，baseline 反弹 +0.00236 的 1/7）, 但 `logloss_rebound_delta=0.0` 过敏触发了 guard
- baseline 自己 E2 反弹幅度更大却没停（当时 EXP-030 job 可能用了不同 guard config 或代码版本早于 guard）
- **gap=0.05 配置下 E2 LogLoss 小反弹是噪声级抖动**，不应当做过拟合信号

**真实状态**：T19 E2 表现疑似正向（+0.0027 AUC vs baseline E2），但因 guard 早停未能跑到 E3（baseline best），**无法对齐判决阈值**。需要 rerun。

**下一步（rerun）**：`t19-perffn-halftrain-gap05-rerun-loose-guard`
- 完全同 T19 config + `--logloss_rebound_delta 0.003 --logloss_rebound_patience 2`
- 跑完 E3 后按原 EXP-034 Go/No-go 表判决
- 如果 E3 ≥ 0.847 → 真信号 → evaluation
- 优先级：**中**（T25 B 回流后视情况）—— 若 T25 B work 优先投 T25 evaluation; 若 T25 B 也失败, T19 rerun 是下一个候选 bet

**代码保留**：`--rank_mixer_ffn_mode per_token` 分支仍在，默认 `shared`。ADR-007 Status 从 Deprecated 升为 Paused-Needs-Rerun。

详见 [`EXP-034 §Correction`](../40-experiments/EXP-034-t19-perffn-halftrain-2026-05-09.md) + [`ADR-007 §Correction after EXP-030 baseline review`](../30-decisions/ADR-007-rank-mixer-per-token-ffn.md)。

---

### T20 `[!]` Funnel 重设计 — 根治 dev→LB 系统性背离

**触发**：2026-05-04 一日 3 次 evaluation,2 次 LB 反向（合计 −0.016 LB），暴露现有 funnel (proxy → half-train → full-train → evaluation) **按 dev AUC 决策**在本赛题 distribution shift 场景下**系统性失效**。

**完整复盘见 [`EXP-024`](../40-experiments/EXP-024-methodology-retrospective-2026-05-04.md)**，本节只记 next action。

**核心诊断**：
- Dev AUC 测 "train 前 95% → train 尾部 5%" 的 generalization；LB AUC 测 "train 100% → 远未来 test" 的 generalization；**两个不同任务**。
- Proxy / half-train 都无法区分"慢性泛化失败",只能过滤"代码 bug / 极端负向"。
- 正交子弹假设有漏洞,多个 target-conditioning / ranking 改动可能**共同加大 overfitting**。
- 组合实验（T16+T18+holdout）违反单变量 ablation 纪律 → LB 失败无法归因。

**改革方向（按优先级）**：

| # | 方向 | 预期工作量 | 预期价值 |
|---|---|---|---|
| A | Valid split 重设计: `train[0, 80%] \| 空窗[80, 95%] \| valid[95, 100%]`(ADR-005 方案 C) | ~60 行 `dataset.py` | 🔴 治本,提升 dev→LB 相关性 |
| B | 纪律: 组合实验前必须先有单变量 ablation | 纯纪律,写入 `.cursor/rules` | 🔴 避免归因歧义 |
| C | Trainer 加 `keep_topk_ckpt=3`,保留多 epoch ckpt | ~50 行 `trainer.py` | ✅ **已实装（2026-05-09）** · `--keep_topk_ckpt N` CLI + `*.topk_{rank}/` 目录 + `topk_checkpoints` summary 字段 + 4× evaluation 放大 |
| D | Proxy 重新定位: 只过滤代码 bug,不做方法学判决 | 纯纪律 | 🟡 释放 proxy quota 给 ablation |
| E | Evaluation 前 5~10 次明确用于建立 dev→LB 映射 | 纯纪律 | 🟡 有了映射后再追 SOTA |
| F | Ranking-family loss（BPR / ListMLE / Contrastive）降优先级 | 规则 | 🟢 避免重犯 T18 错误 |

**下一步（等 T16-only full train 回流后）**：
1. **用 T16-only LB 核查 holdout 独立成本**：若 T16-only 带 holdout LB < 0.846 → holdout 有 −0.003 以上成本,立刻去掉 holdout 并重设计
2. **启动方向 A（valid split 重设计）的代码实现**,作为下一轮实验的默认 valid 定义
3. **写 `.cursor/rules/experiment-discipline.mdc`**,落方向 B/D/E/F 的纪律铁律

**当前优先级**：**最高**。在 funnel 修复之前,任何方法学级创新（例如 DIN interest token, Feature Cross Layer, Muon optimizer）的 LB 信号都**不可信**,**不应该投入 quota**。

---

### T21 `[x]` P1 Item-ID Random Mask — ❌ 已证伪关闭（LB 0.842176 vs SOTA 0.84637 Δ−0.004194）

**2026-05-05 终结**：`idmaskc15-icq-halftrain-swiglu-vr005-bs128` E3 dev **0.868288**（vs icq-half E3 0.867965 +0.000323）→ evaluation **LB 0.842176**（vs SOTA 0.84637 **Δ−0.004194**）。Dev 和 LB 方向相反；dev→LB gap 从 icq-half 的 +0.0222 放大到 **+0.0261**。

**机制**：input-level random mask 让 85% token 仍可见，模型仍依赖 id embedding。Dev 近 train（id 分布共享）略涨；test 远 train（id OOV 率高）85% 训练路径失效。**不是强度问题**（降到 0.10 也救不了），方法本身在"远未来 test"场景无效。

详见 [`EXP-025`](../40-experiments/EXP-025-p1-idmask-c15-halftrain-2026-05-05.md) 与 [`EXP-027`](../40-experiments/EXP-027-parallel-ablation-batch-2026-05-05.md)。

代码保留（`--id_mask_prob` / `--id_mask_seq_domains` 默认 0），不回滚；未来若有"structural mask + input mask"组合路径可复用。

---

### T22 `[✅机制]` P3 Valid Split Gap — 机制成功 / 参数需调优

**2026-05-05 晚 RG-level 诊断**：`gap15-icq-halftrain` (RG-level) 失效, `effective train→valid distance = -341046 sec`。
**2026-05-06 凌晨 row-level 补充**：**`rowgap15-icq-halftrain` (C6 row-level) LB 0.833987 + Dev 0.837987 = Dev→LB gap +0.004 (历史最小)！**

**修订判决（2026-05-06）**: Funnel **机制成功**（dev 确实趋近 test 分布），但 `valid_gap_ratio=0.15` 扔 15% train 数据代价过重。**最优 gap_ratio 预测 ≈ 0.05**（用较少 train loss 换可信 dev）。

**行动**: 后续所有实验默认带 **`--valid_gap_ratio 0.05 --enable_row_valid_gap`**：
- 代价: 训练数据减少 5% (可控)
- 收益: dev 变得可信 (gap 从 0.022 降到 ~0.010)
- 对所有决策质量是质的提升

详见 [`EXP-026`](../40-experiments/EXP-026-p3-gap15-halftrain-2026-05-05.md) §补遗 + [`EXP-028`](../40-experiments/EXP-028-seed-luck-hypothesis-2026-05-05.md) §补遗。

---

### T23 `[x]` seq_c feat47 Structural Disable — ⚠️ 部分证伪关闭（LB 0.840344 vs SOTA 0.84637）

**2026-05-05 终结**：`disablefid47-icq-halftrain-swiglu-vr005-bs128` E4 dev **0.868192** (vs icq-half E4 0.868619 Δ−0.000427) → evaluation **LB 0.840344** (vs SOTA 0.84637 **Δ−0.006026**, dev→LB gap **+0.0279**, 历史最大档)。

**C1 `disablefid47_29` evaluation 同晚回流**: LB **0.836736** (Δ−0.009634) → 证实 disable 范围扩大 → LB 进一步下降。

**⚠️ 判决复杂化**：同晚 C12 icq_seed2 (完全同 SOTA config 只换 seed) LB = **0.841294**（Δ−0.005076 vs SOTA）,说明 **SOTA 0.84637 本身可能包含 seed luck**。按新铁律解读: T23 单独 −0.006 可能是 **seed 噪声 (±0.003) + 真实方法学负向 (−0.003)** 的叠加,而非纯方法学损害 0.006。C1 的 −0.010 超出 seed 噪声上界,方法学真实负向证据更强。

**解读**：Seed 主楼问题-5 "seqc feat47 跟 itemid 是一样的,做不做 match 特征都掉分" 不等于"删掉 feat47 安全"。Seed 没说过"删除后持平",我们误推导。结构化 disable 在 test 上伤害约 0.003~0.006,证明 `seq_c feat47` 在 test 上**有真实 signal**(即使它对 dev 贡献 ≈ 0)。

详见 [`EXP-028`](../40-experiments/EXP-028-seed-luck-hypothesis-2026-05-05.md)。

代码保留（`--disable_seq_fids` 默认空），不回滚。未来若回到 "disable 路线" 需要搭配 structural replacement (例如同时加 seq_c 的 action-type embedding 重组)。

---

### T24 `[~]` EDA 与架构级创新（next batch）

**触发**：T21/T22/T23 三个方向全部证伪或 Blocked, EXP-028 "SOTA seed luck" 发现后, 战略切换到**架构级创新 + EDA 驱动**。

**✅ EDA 已完成（2026-05-05 晚 · EXP-029）**：
- **结论 1**：platform 顶层 user_int / item_int 最大 vocab 只有 25k，**"item_id OOV 问题" 只在 seq 里**，不在顶层 → UE 分离 / 顶层 disable 方向**作废**
- **结论 2**：模型在"信息丰富"样本上 AUC 低 0.04~0.05（**稀疏 user AUC 0.89 vs 丰富 0.85**）→ Seed 主楼问题-1 "id 特征线下虚高线上掉" 的**机制在我们模型里真实存在**
- **结论 3**：大 vocab 全部在 seq 域（seq_c/47 = 2.8 亿, seq_b/69 = 8600万, seq_c/29 = 820万 等）→ **hash vocab 更激进是最对口方向**

**重新排 ROI（基于 EDA）**：

| # | 方向 | EDA 适配度 | 代码量 | 优先级 |
|---|---|---|---|---|
| 🥇 **F1 Seq hash vocab 更激进**（500k → 100k 或 50k）| **极高** | **0 行**（CLI 已有）| **最高** |
| 🥇 F2 DIN target attention | 高 | ~80 行 | 高 |
| 🥈 F3 样本丰富度加权 | 中 | ~50 行 | 中 |
| 🥉 F4 FaFE feature-aware embedding | 中 | ~100 行 | 中 |
| ❌ UE 分离 | 低 | ~120 行 | **抛弃**（EDA 证 user_int 无 UE 层级）|
| ❌ 顶层 item_id disable | 低 | ~10 行 | **抛弃**（EDA 证顶层无大 vocab）|

### T24 `[x]` F1 hash vocab 激进方向 — 持平关闭（LB 0.838417 vs 新基线 0.839223, ΔLB −0.0008）

**2026-05-07 终结**：`hash100k-gap05-icq-halftrain` LB **0.838417** vs `icq-half-gap05-icq-halftrain` LB **0.839223** = **ΔLB −0.000806**（seed 噪声 ±0.005 内）。

**结论**：F1 方向无 LB 信号。hash 从 500k 降到 100k 的效果: Dev 降 0.002 (容量变紧), LB 几乎不变。**不再试 hash50k/25k**。

### T25 `[~]` G1 DIN Target Attention（代码已就位 · 待 T27 判决后启动）

**触发**：EXP-030 显示 small ablation / hash 调整全部持平，**架构级创新是唯一破局路径**。DIN 是 Seed 主楼方案 2 "fafe+din" 中对口最好的方向。

**机制 vs 已证伪 T16 additive**:
- T16 additive: target_embed **加性注入** seq 每个 token（soft, 低表达力）—— 已在 full train 证伪
- G1 DIN: target_embed **作 query 对 seq cross-attention**（hard attention, 高表达力）—— **完全不同机制**

**实施**（已完成 · 2026-05-09 早）:
- ✅ `DINInterestExtractor` 模块 in `baseline/model.py:563-666`（4 × CrossAttention per domain + merge Linear, ~100 行）
- ✅ `PCVRHyFormer._apply_din_interest_bypass` in `baseline/model.py:2677-2703`（bypass 接到 classifier 前）
- ✅ CLI `--enable_din_interest` in `baseline/train.py:413`
- ✅ `baseline/infer.py` 读取 `cfg['enable_din_interest']`，自定义 infer.py 兼容
- ✅ 本地 smoke 4/4 通过（`py_compile`, train 5 steps, train_config 落盘, infer 1000 predictions）—— 详见 `EXP-033` §本地 smoke
- ✅ 参数量增量实测 **Dense +104,896（+3.6%）**，与理论估算一致
- ✅ EXP-033 实验计划 + 2 个 run.sh 版本（A=叠加 T27, B=fallback 纯 DIN）已落档

**训练 config**：`--valid_gap_ratio 0.05 --enable_row_valid_gap`（新默认 · EXP-030）

**判决阈值**（E3 Dev AUC vs `icq-half-gap05` E3 baseline 0.842443 · 在 gap=0.05 可信 dev 下）:
- ≥ 0.847（ΔDev ≥ +0.0046 = **10× seed rerun 噪声 ±0.0001**）→ 🎯 强信号，立即 evaluation
- [0.8444, 0.847) → 🟡 疑似，先 seed=3 rerun
- [0.8412, 0.8444) → ⚪ 持平 / 噪声内，Deprecated
- < 0.8412 → ❌ 负向，关闭

**下一步**（等 T27 EXP-031 回流）:
- 若 T27 Go（bf16 dev 无损 + ≥1.5× 加速）→ 用 EXP-033 方案 A（T25 + compile+AMP 叠加）
- 若 T27 No-go → 用 EXP-033 方案 B（纯 DIN, 无 compile/AMP）
- 首次 evaluation 必须勾选"上传自定义 infer.py"（`enable_din_interest` 是新 kwarg，team-memory 铁律）

详见 [`EXP-029`](../40-experiments/EXP-029-eda-report-2026-05-05.md) + [`EXP-033`](../40-experiments/EXP-033-t25-din-halftrain-2026-05-09.md)。

---

### T28 `[x]` DIN Integrated Inside HyFormer Block — ❌ 已证伪关闭（2026-05-12 · LB M1 0.841807 vs T25 bypass 0.843433 Δ−0.00126 / M2 0.837576 全面负向）

**2026-05-12 终结**：双 seed LB 回流确认 T28 integrated（DIN 每 block 内部注入 2 次）**比 T25 bypass（1 次）更差**：
- M1 seed=3: Dev 0.845197（+0.0015 vs T25）但 LB 0.841807（**−0.00126 vs T25 bypass seed=3 0.843069**），Dev→LB gap 扩 13× 到 +0.00339
- M2 seed=42: Dev 0.840669 / LB 0.837576（全面负向）+ E4 LogLoss guard 触发
- **机制**：DIN 注入 2 次放大训练期 target 分布依赖，test 远未来分布偏移 → 过度激活反而伤害
- **铁律**：target-aware 机制"做 1 次刚好够，做 2 次过度" · 未来 DIN 探索走"多样化"（per-seq / multi-head）不走"重复化"

**ADR-010 Status: Proposed → Deprecated**。代码保留（`--enable_din_integrated` CLI + `MultiSeqHyFormerBlock` per-block DIN 分支默认 off，backward-compat bit-identical）。

**T25 bypass 保持 DIN 方向唯一最优实现**（gap=0.05 seed=42 LB 0.843433）。

详见 [`EXP-039`](../40-experiments/EXP-039-t28-din-integrated-halftrain-2026-05-11.md) + [`ADR-010 §Deprecation`](../30-decisions/ADR-010-din-integrated-in-block.md)。

---

### T29 `[~]` Tokenization Fix · 启用 ns_groups.json 修改 fid 拼接顺序（DECEM 榜一方向）

**触发**（2026-05-11 CEST 04:30）：EXP-035 Kaggle 视角 audit 把 "Tokenization 修复" 列为 DECEM 4 个公开 work 方向中**唯一未消化的方向**。agent 深度审视后确认：`baseline/run.sh` 过去 48 天一直写 `--ns_groups_json ""`（空字符串）走 singleton fallback，等于**fid 按匿名编号序随机拼接进 RankMixerNSTokenizer 的 chunk**。DECEM 原话"简单做了一些修复，让其更加合理化"大概率指这件事。

**机制**：`RankMixerNSTokenizer` 里 `groups` 决定 fid 的 concat 顺序，相邻 fid 进同一 chunk → 共享一个 Linear projection → fid 的梯度耦合。当前"随机顺序"让 chunk 内部大/中/小 vocab 的 fid 混杂，Linear 难以同时优化。

**已完成**（2026-05-11 CEST 04:30 · agent 阶段 A · 西班牙凌晨）:
- ✅ [ADR-009](../30-decisions/ADR-009-tokenization-fix-ns-groups.md) 起草（Proposed）
- ✅ [EXP-038](../40-experiments/EXP-038-t29-tokenization-fix-halftrain-2026-05-11.md) 实验计划落档
- ✅ 生成 `scripts/gen_ns_groups_vocab_tier.py`（按 vocab tier 生成 ns_groups JSON）
- ✅ 生成 `baseline/ns_groups_vocab_tier.json`（T29-B 配置）
- ✅ 本地 smoke 两个变体全通过：train + infer 都 OK，参数量不变（353,334,977）验证"只改顺序不改容量"
- ✅ py_compile 全过

**2 个变体**：
- **T29-A**：启用 `baseline/ns_groups.json`（4/26 原作者手工分组 U1~U7 + I1~I4）—— 最贴近 DECEM "简单修复" 语义
- **T29-B**：启用 `baseline/ns_groups_vocab_tier.json`（按 vocab tier 分 3 组：small ≤ 100 < mid ≤ 2000 < large）—— 机制驱动，同 chunk 内 fid embedding 饱和速度相似

**判决阈值**（vs `icq-half-gap05` E3 Dev 0.842443 · gap=0.05 可信 dev）:
- E3 Dev ≥ 0.8437（Δ+0.0013 = 3× rerun 噪声 0.0003 + 采样偏差）→ 🎯 真信号立即 evaluation
- [0.8433, 0.8437) → 🟡 seed=3 rerun
- [0.8421, 0.8433) → ⚪ 持平 Deprecated
- < 0.8421 → ❌ 负向关闭

**LB 阈值**（若 S2 通过消耗 1 eval quota）：vs `icq-half-gap05` LB 0.839223
- LB ≥ 0.847 → SOTA 破 + 优先叠加 T25 DIN + gap=0
- LB ≥ 0.842 → 真信号保留并叠加
- < 0.839 → 持平 Deprecated

**下一步**（等 2026-05-11 07:00 CEST EXP-036/037 LB 回流后）:
1. 按 EXP-036/037 decision tree 决定 4 slot 分配（T29-A + T29-B 占 2 slot）
2. **注意**：平台上需上传 `baseline/ns_groups.json` + `baseline/ns_groups_vocab_tier.json` 两个 JSON（无 Python 代码改动，可 Copy 上次 T25 DIN job）
3. 若 S2 通过，S3 LB 校准，S4 与 T25 DIN 叠加

**当前优先级**：**最高**（Top 20 冲刺主路径，5/14 前 LB 0.850 目标的关键依赖）。

---

### T30 `[~]` 时间桶边界按分位数重设计（DECEM Tricks #1）

**触发**（2026-05-12 CEST 19:40 · XHS 情报 0512 收集后）：DECEM 付费笔记 Tricks 总结 2/2 页第 1 项明说 "BUCKET_BOUNDARIES 是手工设计 64 个对数边界 5 秒到 1 年，手工边界可能有空挡或过度集中桶浪费 Embedding 容量，建议按分位数重设"。DECEM 实测 +0.001 LB，冰淇淋确认同样提升。我们 T12 已做 `hour_of_day/day_of_week` 时间特征（✅），但 **time_diff 桶边界** 完全没调过。

**实施路径**（~30 行代码）:
1. 在 `baseline/dataset.py` 找到 `BUCKET_BOUNDARIES` 定义（或 time_diff 分桶逻辑）
2. 写 1 个 `scripts/gen_time_bucket_quantile.py`：对 demo 数据 + 平台训练数据（如有）的 time_diff 列做统计，输出按 1/64, 2/64, ..., 63/64 分位数的 64 个边界值
3. 替换默认 BUCKET_BOUNDARIES 为分位数版本（保留 CLI flag 可切换）

**判决阈值**（vs `icq-half-gap05` E3 Dev 0.842443）:
- E3 Dev ≥ 0.8437（+0.0013 = 3× rerun 噪声）→ 🎯 真信号
- [0.8433, 0.8437) → 🟡 seed=3 rerun
- < 0.8430 → ❌ Deprecated

**与 T29 / T25 bypass 的关系**：完全正交可叠加。T30 是时间桶分桶边界改动（dataset 层），T29 是 fid 拼接顺序（tokenizer 层），T25 是 DIN bypass（model 层）。

**当前优先级**：**高**（DECEM 明确 work 且代码成本最低）

**当前状态**：🧱 **代码落地完成**（CEST 12:20-12:30 · EXP-043 落档）· `parse_bucket_boundaries` + `PCVRParquetDataset` 构造参数 + CLI `--time_bucket_boundaries` + infer.py fallback · 本地 smoke 验证通过（默认值 bit-identical · 非法值正确 raise · 自定义值切换生效）· **默认值未变**（等 M16 EDA `q5_time_diff` 回流后生成分位数版）· 平台 job 待启动

---

### T31 `[~]` is-missing 信号（DECEM Tricks #3）

**触发**（2026-05-12 · 同上情报源）：DECEM Tricks #3 明说 "大量 null 字段 (86,94,99-104,108,109)，baseline 统一映射为 0 (padding)，如果缺失率与转化率高度相关就丢了关键信号"，建议 (a) 加 "missing" 专用 slot vocab_size 配置区分 padding=0 独立 (b) 加 **缺失指示位** 二值特征 concat 进模型。**广告业务把"缺失本身作为信号"是常识级做法**。

**实施路径**（~50 行代码）:
1. 在 `baseline/dataset.py` 特征提取时，对每个 `user_int_feats_{fid}` / `item_int_feats_{fid}` 额外提取一个 `is_missing_{fid}` 布尔向量（true 表示该样本该字段原始值是 null / <= 0）
2. 在 `baseline/model.py` 加一个独立 `missing_indicator` embedding table（vocab_size=2 × num_fids）或把 is_missing 作为 concat 到 dense bypass 里
3. 方案权衡（需设计）：
   - **方案 A**：每 fid 一个 missing embedding（参数量 +60 × d_model）
   - **方案 B**：共享 missing embedding（参数量 +2 × d_model）
   - **方案 C**：直接 concat 0/1 二值到 dense bypass（零参数）

**判决阈值**：同 T30

**当前优先级**：**高**（DECEM 未给幅度但广告业务常识 +0.001~0.003）

**当前状态**：🧱 **代码落地完成 · Scheme D 双路冗余**（CEST 12:30-13:30 · EXP-044 落档）· 用户 CEST 12:50 选 **Scheme D**（Path A vocab_size+1 top slot + Path B dense bitmap · 两路冗余）· dataset 层纯改动（model 零代码）· CLI `--enable_is_missing --is_missing_user_int_fids <CSV> --is_missing_item_int_fids <CSV>` · 本地 smoke 验证通过（关闭时 `user_dense_dim=755` 不变 · 开启时精确 +N · bitmap ∈ {0,1} · top slot 路由正确 · missing_rate 与 DECEM 描述方向一致）· **等 M16 EDA q2 回流确定选中 fid 清单**（|cvr_gap| > 2% 纳入 / < 0.5% 排除）

---

### T32 `[~]` DIN Q_init = MLP(item_emb)（DECEM Tricks #6 · 思路 A）

**触发**（2026-05-12 · 同上情报源）：DECEM Tricks #6 给了 DIN 集成的两种思路：
- **思路 A**：在 MultiSeqHyFormerBlock 的 cross-attention 里，让 Q tokens 的初始值由 item embedding 引导：`Q_init = MLP(item_emb)` 而非 `MLP(GlobalInfo)`
- **思路 B**：加 DIN 分支 `q_i = softmax(Q_target · K_seq)` 结果 fusion

我们 T25 bypass **走的是思路 B 的简化版**（classifier 前旁路）+0.004 LB（比 DECEM 的 +0.001 好 4×，可能因为我们基线更低）。T28 integrated（每 block DIN）已证过度 −0.001 LB。**T32 = 思路 A 的具体实现**，和 T25 bypass 机制完全不同：
- T25 bypass：独立 DIN 模块在 classifier 前 concat
- T32 Q_init：改 MultiSeqQueryGenerator 内部，让 Q tokens 初始化基于 item_emb 而非 GlobalInfo
- **理论可叠加**（不同模块不同阶段）

**实施路径**（~50 行代码）:
1. 在 `baseline/model.py` `MultiSeqQueryGenerator` 类加 `--din_q_init_from_item` CLI flag
2. flag 开启时，Q tokens 的 FFN 输入从 `Concat(NS_mean, Seq_i_mean)` 改为 `Concat(NS_mean, Seq_i_mean, item_mean)`（或把 item_emb MLP 直接作为 Q 初始化，取代 GlobalInfo）
3. 保持 ICQ 不变（与 T32 正交）
4. 首次 evaluation 必须上传自定义 infer.py（新 kwarg）

**判决阈值**（vs T25 bypass E3 Dev 0.843696）:
- E3 Dev ≥ 0.845 → 🎯 真信号，立即 eval
- [0.844, 0.845) → 🟡 seed rerun
- < 0.8437 → ❌ Deprecated

**与 T25 bypass 的关系**：机制不同，**首次 ablation 需单变量**（T25 off + T32 on），验证 work 后再考虑 T25+T32 叠加。

**当前优先级**：**中-高**（机制深度 > T30/T31，代码成本类似 ~50 行）

**当前状态**：📋 planned

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

- **2026-05-13 西班牙中午 12:00（UTC+2 · 🔬 M9 破 SOTA + 叠加证伪 + EDA 搭车 M16 启动）**：① **今早 4 LB**：M9 **0.84671** (刷 SOTA +0.00034 vs 0.84637, 但 <1σ seed 噪声) / M10 0.84145 (T25+T29 s42 叠加失败 Δ−0.002) / M11 0.841455 (T29 only s3 弱正向) / M7 0.839593 (T25+T29 s3 **Dev→LB gap 扩 15× 证伪**)。② **T25+T29 叠加方向双 seed 一致证伪关闭**（M7 Δ−0.00348 / M10 Δ−0.00198 · 昨天"叠加 seed 敏感"铁律修正为"叠加本身不 work"）。③ "LogLoss 低 ≠ LB 好" 反例：M7 LogLoss 项目最低但 LB 倒数 → 未来判决只信 LB，Dev/LogLoss 只排除负向。④ **M14/M15 seed sweep 已启动**（s11/s7 · 建立 gap=0 × DIN 4-seed 能力带 · 今晚回流）。⑤ **用户反思"E3-E4 是否欠拟合"**：EXP-019 铁律确认 E3-E4 甜蜜点 · **`--keep_topk_ckpt 4` 设为新 default**（baseline/run.sh + active.md + team-memory）。⑥ **用户提议 EDA**：深思后采取"搭车策略"（EXP-037 5 次独立 EDA 失败教训 · 不再盲试）· 通路走 `training_summary.json['eda']` 稳定字段。⑦ **CEST 11:00-11:55 EDA 代码 + smoke 完成**：`baseline/dataset.py +220 行`（4 类 EDA 累加器 · reservoir sampling Q4/Q5）· `trainer.py/train.py` 各 +20 行 · **num_workers=0 自动降级**避免 worker copy 问题 · demo smoke 验证 summary['eda'] 字段正确输出。⑧ **Smoke 3 大指示性发现**：user_id freq_1=100%（印证 Seed 一次性）/ seq_a p50=256 截断（Trick 2 方向有效）/ time_diff p1=7.3h 说明当前 BUCKET_BOUNDARIES 5s~1h 桶被浪费（T30 方向极有潜力）。⑨ **M16 `M16-m9rerun-eda-gap0-seed42-4ep` CEST 11:55 启动**：M9 config rerun + EDA 搭车 · 双获（验证 M9 稳定性 + 拿 1.9M 真 EDA 数据）· 预计 CEST 18-20:00 回流。⑩ **EXP-042 落档**（含 smoke 验证 + 分析计划 + 方向判决表）。⑪ **今晚 EDA 数据 + M14/M15 + M16 回流后**：基于 Q5 真实 time_diff 设计 T30 · 基于 Q2 per-fid cvr_gap 决定 T31 是否值得。

- **2026-05-12 西班牙晚 21:00（UTC+2 · 🎯 深思后启动 M10/M11 · 3 slot 空余变 1 空余）**：① 用户 CEST 20:51 问"M9 在跑 + 3 slot 空余起什么"。② **深思 6 个候选方向** (T30 时间桶 / T31 is-missing / T32 Q_init / M10 M6 disambiguate / M11 T29 双 seed / hash vocab)。③ 选 **M10 + M11 优先** 2 slot（0 代码成本 / 高信息价值）：**M10** `M10-t25bp-t29vt-gap05-seed42-loose-4ep`（M7 同 config 但 seed=3→42 · disambiguate M6 guard vs seed 问题）+ **M11** `M11-t29vt-gap05-seed3-loose-4ep`（M3 同 config 但 seed=42→3 + 松 guard · 补 T29 双 seed 数据）。④ **留 1 slot 空闲** · 明早基于 M7/M9 LB 信号驱动起 M12。⑤ **CEST 20:55 两个 job 已启动**。⑥ **预期回流 CEST ~5/13 02:00-02:10**。⑦ **明早 CEST 06:00 quota 重置** 4 个候选 ckpt（M7/M9/M10/M11）对应 4 quota。

- **2026-05-12 西班牙晚 20:30（UTC+2 · 🎯 M6/M7 T25+T29 叠加回流 · 跨 seed 分化 · M7 LogLoss 项目最低 · 明早必评 M7）**：① **M6 叠加失败**（seed=42）：E3 Dev 0.841291 / LogLoss 0.27927 / E2 非单调 / E4 guard 触发 · vs T25 B 0.843696 **Δ−0.00241 明显负向** · **不评**。② **M7 叠加成功**（seed=3 松 guard）：E3 Dev **0.843524** / LogLoss **0.27642 项目史上最低 gap=0.05 LogLoss**（比 T25 B 低 0.0018 = 显著 calibration 改善）/ E4 持续下降未触发 guard · **必评**（过期风险最大 · 5/12 23:15 完成 → 5/13 23:15 过期）。③ **新铁律 "叠加效应 seed 敏感性"**：M6 vs M7 Δ Dev 0.00223 超 seed 噪声 2-4× · 机制 = T29 vocabtier 重排 fid 顺序改变 T25 bypass 学到的 target_repr 对应关系 · **未来方向叠加实验必须双 seed 验证**（s42 + s3 各一）。④ **M7 LB 预期** [0.843, 0.845] · 3 角度推理（Dev→LB gap / seed=3 基线叠加 / LogLoss 改善）· 有小概率破 SOTA 0.84637。⑤ **明早 CEST 06:00 quota 重置后立刻评 M7**（最高优先）· 其次 M9 视 Dev 再定。⑥ **M1-M8 全部回流聚类** · 今日 9 个 Dev 数据点覆盖 T28/T29/T25+T29/FaFE/gap=0 × DIN 各路径，**T25 bypass s42 alone (0.843433) 和 M7 T25+T29 s3 (LogLoss 0.27642) 是 gap=0.05 双上沿候选**。⑦ **EXP-041 完整补档** · 含 M6/M7 完整数据 + 叠加 seed 敏感性机制图 + ckpt 过期时间表。⑧ **本地 35 commits 未 push**。

- **2026-05-12 西班牙晚 20:00（UTC+2 · 🔥 M8 FaFE 证伪 + "独立化方向三连败"新铁律）**：① **M8 回流**：`M8-fafe-t25bp-gap05-seed42-4ep` E3 Dev 0.842646 / E4 Dev 0.834809 断崖 −0.00784 / guard E4 触发 / wall-time 6h41m（+130k FaFE params 拖慢 2×）。vs T25 bypass s42 E3 Dev 0.843696 = **Δ−0.00105（3.5× rerun 噪声明显负向）**。② **FaFE 方向关闭**（不 eval M8 · Dev 机制性负向）。③ **"独立化" 方向三连败新铁律**：T19 per-token FFN (+34% Dense 必死) + T28 per-block DIN (+7% LB 负向) + M8 FaFE per-fid transform (+4.5% Dev 负向) = 同类"每参数梯度 signal 不足 → 被迫记忆训练期分布 → 远未来 test 失败"机制。**未来禁止所有 per-X 独立参数类探索**。④ **M6/M7 仍 waiting** 今晚回流 · T25+T29 叠加验证。**M9 明早 CEST 04:00 回流** · 唯一填 EXP-036 A 缺口 ckpt。⑤ **EXP-041 落档** · M6/M7/M8 专项 + 独立化三连败机制图表。⑥ **战略确认**: 今晚已回流 M5 + M8 均证伪，独立"深挖结构创新"方向连吃三败（T19/T28/FaFE），**Top 20 唯一可行路径 = 多个外部信号注入 +0.001~0.002 叠加**（T25 bypass +0.004 已吃 · T29 vocabtier +0.0014 已证 · T30 时间桶 / T31 is-missing / T32 Q_init 待验）。⑦ **本地 34 commits 未 push**。

- **2026-05-12 西班牙晚 19:40（UTC+2 · 🔥 XHS 情报 0512 消化 · DECEM Tricks 2/2 + 杭仔 0.850012 LB · T30/T31/T32 新方向）**：① **用户 19:09 收集 37 张截图**（小红书 DECEM 付费笔记 Tricks 总结 2/2 页 + 杭仔公开 LB 0.850012 截图 + 广小鹅冒险岛/广告算法大赛交流两大群聊讨论）。② **DECEM Tricks 2/2 页 6 大方向完整 OCR 并落档到 `docs/10-research/community-intel.md` §DECEM-0512**：Trick 1 时间桶边界分位数 / Trick 2 seq_max_lens 域特定 p99 / Trick 3 is-missing 信号 / Trick 4 ns_groups 按相关性+SENet / Trick 5 FM/DCN 交叉 ⚠️ / Trick 6 Q_init=MLP(item_emb)。DECEM 原话"**我这里的提升是 +0.001**"，"**不要尝试过大的改动**"，冰淇淋确认相同提升。③ **杭仔技术路线**（LB 0.850012）：用 DIN 完全替代 Transformer+cross attention 序列模块（不是我们 bypass 方式）/ DIN target 只用 item / query generator 分开 / 手动保存中间 epoch 防过拟合（best vs last E 差 5‰）/ 训练速度 "2h 就训好了" "6h 15 epoch"。但被群内用户质疑 "真有点烟雾弹" 且 DECEM 附和 "肯定实现细节的问题"，**大方向可参考具体不可照搬**。④ **战略判决修正**：我们 T25 bypass +0.004 LB 已吃 DECEM 最大 trick，**剩 5 tricks 最多再吃 +0.005** → 综合 ceiling **0.849**，5/14 Top 20 0.850 **勉强可达难度大**。⑤ **新增 3 个高 ROI task**：T30 时间桶分位数（~30 行 / +0.001 LB）/ T31 is-missing 信号（~50 行 / +0.001~0.003 LB）/ T32 DIN Q_init（~50 行 / +0.001~0.003 LB），全部 gap=0.05 领地可叠加 T25 bypass。⑥ **战略 pivot**：停止 gap=0 DIN 探索（M5 证伪 · M9 回流前不扩方向），主力转 T30/T31/T32 gap=0.05 叠加路线。⑦ **情报文档完整落档**：`docs/99-raw/community-snapshots/xhs-decem-0512/` 37 张截图 + 详细 README.md OCR 摘录 + `docs/10-research/community-intel.md` 方法论章节。⑧ **下一步待决策**：等 M6/M7/M8/M9 今晚/明早陆续回流，再决定 T30/T31/T32 的 training 起动时序。⑨ **本地 33 commits 未 push**。

- **2026-05-12 西班牙晚 18:45（UTC+2 · 🔥 M5 LB 0.841812 负向 · T25 bypass 黄金三元组揭示 · M9 补回）**：① **M5 回流**：`M5-t25bypass-gap0-seed3-4ep` E4 Dev 0.868450 / LB **0.841812** / Dev→LB gap **+0.0266 项目史上最高**。vs EXP-030 icq_seed3 (同 seed=3 + gap=0 + no-DIN) LB 0.845329 = **Δ−0.003517 方法学明确负向**。② **"T25 bypass 黄金三元组" 铁律发现**：`bypass + seed=42 + gap=0.05` 是 DIN 方向唯一稳定 work 配置（T25 B LB 0.843433）· 偏离任一维度显著变差：seed 42→3 gap=0.05 仍 work (LB 0.843069) / bypass→integrated 任何 seed 负向 (LB 0.837-0.842) / gap=0.05→gap=0 × seed=3 负向 (M5 LB 0.8418)。三元组外组合集中在 "0.8418 吸引子" 附近（M1 T28 integrated s3 gap=0.05 LB 0.841807 与 M5 LB 0.841812 差 0.000005 惊人巧合）。③ **EXP-040 落档**（专门记录 M5/M9 gap=0 × DIN seed sweep 实验 + 3 大 Finding + 新铁律候选 + M9 情景决策树）。④ **M9 `M9-t25bp-gap0-seed42-4ep` CEST 18:45 启动**：补回 EXP-036 A 丢失缺口（seed=42 + gap=0 + DIN × E4 Dev 0.868924 过期），是当前**唯一未被证伪的 "DIN × gap=0" 候选**，预计明早 CEST ~04:00 回流。⑤ **M6/M7/M8 今晚陆续回流预期**（M6/M7 gap=0.05 叠加 / M8 FaFE 首测）。⑥ **M9 情景决策树**：LB ≥ 0.847 破 SOTA → 起 M10 三重叠加冲 0.849~0.851 / LB ∈ [0.844, 0.847] 持平 → gap=0.05 叠加路线 / LB < 0.844 → DIN × gap=0 方向**彻底关闭**，Top 20 只能靠 gap=0.05 多方向叠加。⑦ **Top 20 (LB 0.850) 路径更新**：原"M5 破 SOTA 直通 Top 20"不成立 → 新路径 = M9 破 SOTA (成功率 ~50%) + M6/M7 叠加 + 可能的 M10 三重叠加。⑧ **新命名规则生效**：未来所有 job Job Name 以 `M<N>-...` 开头便于回溯。⑨ **今日 eval quota 剩 0**，明早 CEST 06:00 重置到 4。

- **2026-05-12 西班牙中午 13:00（UTC+2 · 🎯 M1/M2/M3 LB 回流 · T28 integrated 证伪 · T29 vocabtier 弱正向 · M5-M8 启动）**：① **4 个 LB 数据点**：M1 t28-din-integrated-s3 LB **0.841807**（Dev 0.845197 / gap +0.00339）/ M2 t28-din-integrated-s42 LB **0.837576**（Dev 0.840669 / gap +0.00309）/ M3 t29-vocabtier-s42 LB **0.840603**（Dev 0.841629 / gap **+0.00103 最小 = 项目 gap=0.05 历史最小**）/ M4 t29-tokfix-s42 Dev 0.841171 E4（未评 · 和 M3 差 0.0005 信息增量低）。② **T28 integrated 证伪 → T28 `[~]`→`[x]` 关闭**：双 seed 一致负向（M1 Dev +0.0015 vs T25 bypass 但 LB −0.00126；M2 全面负向 + E4 guard 触发），机制 = DIN 注入 2 次放大训练期 target 分布依赖，test 远未来分布偏移下过度激活反害泛化。**ADR-010 Proposed → Deprecated**。③ **T29 vocabtier 弱正向真信号**：Dev→LB gap 扩只 +0.00103（vs 基线 +0.0032 / T25 bypass +0.00026 / T28 integrated +0.0034）= Dev 信号 100% 转化 LB 信号，非 memorization；ΔLB +0.00138 贴近 seed 噪声下界但 gap 极小让信号可信。vocabtier 胜 tokfix +0.0005 Dev（机制：vocab tier 让 chunk 内 fid 基数均匀梯度稳定 > 匿名特征下手工语义分组）。④ **新铁律 3 条**：(a) DIN 类 target-aware 机制"做 1 次刚好够，做 2 次过度"，未来走"多样化"（per-seq 独立 / multi-head）不走"重复化"（multi-block same）；(b) Dev→LB gap 扩 2× 以上（T25 bypass 标准 +0.00026 → 警戒线 +0.00052）是方法学负向最强信号；(c) evaluation quota 诈尸 grep 铁律（agent 落档 "X quota 已提交" 必须附平台实际 eval ID 否则无效）。⑤ **CEST 12:53 启动 4 新 training（M5-M8 新命名规则 `M<N>-...`）**：M5 `M5-t25bypass-gap0-seed3-4ep`（🏆 最高优先 · 最可能破 SOTA · T25 bypass + gap=0 + seed=3 首次组合）/ M6 `M6-t25bp-t29vt-gap05-seed42`（T25+T29 叠加 s42）/ M7 `M7-t25bp-t29vt-gap05-seed3`（T25+T29 叠加 s3 · 松 guard）/ M8 `M8-fafe-t25bp-gap05-seed42`（FaFE 首次平台测试）。⑥ **M5 预期 LB 0.846~0.850**（T25 bypass +0.004 LB 迁移到 gap=0 SOTA 领地 × seed=3 历史好运），**最有希望首次破 SOTA 0.84637**。⑦ **文档更新**：EXP-038 LB 补档 + EXP-039 LB 补档 + ADR-010 Deprecation Notice + 30-decisions/README.md 索引 + T28 task `[x]`。⑧ **今日 eval quota 剩 1**（M4 跳过），晚上看 M5 Dev 决定是否用。⑨ **本地 30 commits 未 push**。

- **2026-05-11 西班牙晚 23:27（UTC+2 · 🚀 4 方法学 training job 重启 · 过期 ckpt 全部无法恢复 · 今日 3 eval quota 放弃）**：① **事实对齐**：EXP-036 A/B/D 和 EXP-037 的 u4i3-seed42/seed3 ckpt **全部过期**（5/10 CEST 16:00 训练完成 → 5/11 CEST 10:00 平台 24h 自动清理），其中 **EXP-036 A Dev 0.868924（项目史上最高）永久丢失**，战略损失 +0.001~0.003 LB。原因：agent 5/10 文档记"4 eval quota 全部提交"但实际只 evaluate 成功 ≤1 个，其余 ckpt 未及时 publish → 24h 后被清。② **agent 凭记忆文档错误铁律补**：team-memory 需新增 "agent 在落档 'X quota 已提交' 时必须 grep 平台实际 evaluation ID 并附在文档，不能凭印象写"。③ **今晚 22:45 agent 推的 4 job 用户主动停**：理由是 GPU 串行 → 想先喂给 evaluation，但发现无可 evaluate ckpt → 直接放弃今日 3 eval quota（明早 CEST 06:00 重置清零，不用就浪费但**已没有选择**）。④ **CEST 23:27 重启 4 training job**（3 🟢 方法学 + 1 🥈 seed 校准）：Slot 1 `t29-tokfix-halftrain-gap05-seed42`（启用 `baseline/ns_groups.json`）/ Slot 2 `t29-vocabtier-halftrain-gap05-seed42`（启用 `baseline/ns_groups_vocab_tier.json`）/ Slot 3 `t28-din-integrated-halftrain-gap05-seed42`（--enable_din_integrated --din_integrated_alpha_init 0.1, **不开 compile**）/ Slot 4 `t28-din-integrated-halftrain-gap05-seed3`（同 Slot 3 + 松 guard `delta=0.003 patience=2`, 让 seed=3 跑到 E4）。⑤ **4 job 全部 Create New Training Job**（不 Copy · 规避文件上传不全风险）· 上传 `baseline/` 全部 6 Python + ns_groups JSON（Slot 1/2 需要）· run.sh 粘贴到平台编辑框。⑥ **判决阈值明确**（明早 CEST 07:00 后应用）：T28-s42 Dev ≥ 0.847 → 🏆 破 SOTA candidate 必 eval / T29-A-B Dev ≥ 0.844 → Tokenization 真信号 eval / 全部 Dev 持平在 [0.843, 0.844] → 战略复盘不 eval 留 quota。⑦ **不重跑过期 ckpt**（反思：昨天刚 pivot 战略到方法学 bet，今晚追 EXP-036 A seed 上沿 = 倒退 + T28 integrated 本身就是 A 的升级 + seed 噪声 ±0.005 下 Dev 0.868924 可能是 luck 非真信号）。⑧ **Top 20 目标（5/14 LB 0.850）节奏**：今晚这批是 48h 内 3 个方法学 bet 密度峰值；明早判决后 5/12 全天 slot 再排 1~2 个 + S4 叠加；5/13~5/14 冲 0.850 最后窗口。⑨ **本地 27 commits 未 push**（昨晨 EXP-037 + 今晨 T29 5 文件 + 今晨 T28 4 文件 + 本次审视）。

- **2026-05-11 西班牙凌晨 05:20（UTC+2 · 🎯 T28 DIN Integrated 代码就位 · agent 阶段 B · 零 quota 消耗 ~34 分钟）**：① **实现**：把 T25 bypass 的 `DINInterestExtractor` 从 classifier 前（1 次）移到每个 `MultiSeqHyFormerBlock` 内部（num_blocks=2 次）。注入点 = Query Decoding 后、Token Fusion 前：`decoded_qs = [q_i + self.din_alpha * interest.unsqueeze(1) for q_i in decoded_qs]`。每 block 独立 DIN + 独立 learnable `din_alpha`（init 0.1）+ 渐进细化（每 block 见 encoder 细化过的 seq）。② **代码改动**: `baseline/model.py` ~60 行（MultiSeqHyFormerBlock + PCVRHyFormer + 新 `_compute_target_repr` shared helper）/ `baseline/train.py` ~15 行（2 新 CLI flag）/ `baseline/infer.py` 2 行（cfg fallback）。`dataset.py / trainer.py / utils.py` 零改动。③ **本地 smoke 3 矩阵全过**: S1 T28 off → 353,334,977 params（与 pre-T28 bit-identical backward-compat 完美）/ S2 T28 on → 353,536,579 params（Δ+201,602 = 2 blocks × 100,801 符合理论估算）/ S3 T28 on + infer.py → 1000 predictions 无 state_dict 错误训推一致。py_compile + ReadLints 全过。④ **判决阈值**（vs T25 bypass E3 Dev 0.843696 而非 baseline 0.842443 —— 要证值得替换 T25）: ≥0.847 强信号 / [0.8452, 0.847) 疑似优于 T25 / [0.8437, 0.8452) 持平 T25 / <0.8437 Deprecated。⑤ **首次 T28 不开 torch.compile**（避免 dynamo 干扰方法学判断），只开 AMP bf16。**不与 T29 Tokenization fix 叠加**（保 `--ns_groups_json ""` SOTA 路径，严格单变量）。**不与 T25 bypass 叠加**（`--enable_din_interest` off）。⑥ **首次 evaluation 必须上传自定义 infer.py**（新 kwarg · team-memory 铁律）。⑦ **ADR-010 + EXP-039 落档**，T28 task `[~]` 加入 active.md，30-decisions/README.md 索引更新。⑧ **与 T29 的关系**：完全正交（tokenizer 层 vs block 层），S4 可叠加。⑨ **今日 4 slot 最优分配**（等 7:00 CEST LB 回流最终定）：三情景决策树（见顶部审视）。

- **2026-05-11 西班牙凌晨 04:55（UTC+2 · 🎯 Tokenization 方向代码就绪 · agent 阶段 A 产出 · 零 quota 消耗 ~45 min）**：用户深度战略反思请求 + aggressive 5/14 冲 0.850 承诺触发。① **定位 Tokenization 根因**：`run.sh` 2026-04-26 commit `fb2450b` 起写 `--ns_groups_json ""`（空字符串）→ `os.path.exists=False` → fallback 到 singleton 分组 ≡ fid 编号序 concat（匿名化等价随机）。4/26 原作者设计的 `baseline/ns_groups.json`（U1~U7 + I1~I4 手工语义分组）**过去 48 天从未被启用过**。② **机制分析**: `RankMixerNSTokenizer` groups 控制 fid concat 顺序 → 相邻 fid 进同一 chunk → 共享一个 Linear projection → fid 梯度耦合。当前"随机顺序"让同 chunk 内大/中/小 vocab 混杂。③ **DECEM 榜一情报对齐**: DECEM 5/8 帖明说 "Tokenization 做了小修改让其合理化"，本项目 **0 投入**。EXP-035 audit 中列为头号未消化线索。④ **2 个变体就绪**: T29-A = 启用 `baseline/ns_groups.json`（贴近 DECEM "简单修复" 语义），T29-B = `baseline/ns_groups_vocab_tier.json`（按 vocab tier 分 3 组 · 新生成）。⑤ **本地 smoke 全过**: `py_compile` OK / T29-A train 5 steps + infer 1000 predictions OK / T29-B 同样 OK / 参数量两者都是 `353,334,977`（与无 ns_groups 版本相同，证明**只改顺序不改容量**）/ sidecar `ns_groups*.json` 自动写入 best_model/ 且 infer.py 正确 resolve。⑥ **文档落档**: `ADR-009-tokenization-fix-ns-groups.md`（Proposed）+ `EXP-038-t29-tokenization-fix-halftrain-2026-05-11.md` + `scripts/gen_ns_groups_vocab_tier.py` + `baseline/ns_groups_vocab_tier.json`。⑦ **平台提交最低成本**: **零 Python 代码改动**，可 Copy 上次 T25 DIN job，只改 run.sh 一行 `--ns_groups_json` 参数 + 上传 2 个 JSON 文件。⑧ **判决阈值**: E3 Dev ≥ 0.8437（vs baseline 0.842443 Δ+0.0013 = 3× rerun 噪声）→ 真信号；LB ≥ 0.842（vs baseline LB 0.839223）→ evaluation 后保留方向。⑨ **T29 task `[~]` 新增**, 优先级最高。⑩ **当前 slot**: training 4/4 全空 / evaluation 0/4（等 CEST 06:00 重置到 4），**决策策略**: 等 CEST 07:00 4 eval LB 回流后按决策树分配 4 slot（T29-A + T29-B 各占 1, 剩 2 按 LB 定）。⑪ **战略承诺**: 5/14 前 LB 0.850（Top 20），每 48h 需一个正向方法学信号。

- **2026-05-11 西班牙凌晨 04:10（UTC+2 · ⚠️ u4i3 × gap=0 Dev 实测负向 + EDA 5 次失败放弃）**：2 u4i3 × gap=0 × DIN job 回流（`~/Downloads/train_log.txt`）：① seed=42 × E4 **Dev 0.868055**（vs EXP-036 A 0.868924 **Δ−0.000869**, 明显负向）；② seed=3 × E4（松 guard）**Dev 0.868481**（持平估算）。**u4i3 在 gap=0 领地失败**，机制假设：user_ns 5→4 粗化损伤记忆能力 > item_ns 2→3 细化增益。**gap=0 vs gap=0.05 分叉**：EXP-036 D（u4i3 × gap=0.05）LogLoss 优化 −0.00245，今晚 gap=0 反而 Dev 负向 → 新铁律候选 "Tokenization 变种应在 gap=0.05 评估，不在 gap=0"。③ **EDA platform 5 次迭代全部失败**（v1 OOM / v2 log truncate / v3 OOM / v4 stopped at 90/100 / v5 base64 blob 仍截断）消耗 3 slot + 3h agent 时间 → 产出 0 真数据。根因：平台 log UI 采集机制不透明。用户 CEST 00:53 决定放弃 EDA 路径，方向 A（seq domain 异构）保持 demo 假设不改。④ **agent 纪律违反**：在单一失败路径上盲试 5 次违反 EXP-035 "同一假设 ≤2 次失败立刻 pivot" 铁律；EXP-032 "每批 4 slot ≥3 🟢" 被全面违反（4 slot 里 2 个 u4i3 + 3 个 EDA，后者全部 0 产出）。⑤ **EXP-037 落档**（u4i3 gap=0 结果 + EDA 5 次失败详细复盘 + 下一步决策矩阵）。⑥ **本地 25 commits 未 push**，今日 4 evaluation LB 等 CEST 07:00 上榜。⑦ **下一步**：不盲起新 training，等 4 LB 数据后决策。

- **2026-05-10 西班牙下午 16:25（UTC+2 · 🚀 今晚 2 training slot 起飞 · Tokenization u4i3 × gap=0 × DIN 双 seed 验证）**：用户 16:21 关键纠正："我随时有 4 training slot，训练中任务完成会释放 slot" → agent 之前把 slot 误当每日配额。修正后决策：① **今晚起 2 🟢 方法学 job**：`t25-din-u4i3-gap0-seed42-4ep`（Slot 1）+ `t25-din-u4i3-gap0-seed3-4ep`（Slot 2, 松 guard `delta=0.003 patience=2`），都零代码改动 Copy 即可。② **留 2 slot 给明早 07:00 CEST LB 回流后信号驱动**，不盲起。③ **假设**：u4i3 × gap=0 × DIN 是否超过今日 A（seed=42 gap=0 DIN 标准 Tokenization E4 Dev 0.868924）。判决 Dev ≥0.8695 真信号方法学 / 0.8688~0.8695 持平 / <0.8688 u4i3 在 gap=0 失效。④ **每假设 ≤2 seed 铁律遵守**（u4i3 只 seed=42 + seed=3，不扩到 seed=7/11）。⑤ **avoid 的假候选**: vr=0.01 → EXP-032 方向关闭 (team-memory line 65, agent 今晚一度险复犯后在 pre-flight grep 自查纠正); S4 100% data → best config 未锁定, 过早做; FaFE → CLI 首次启用, 无平台 smoke, 风险大于 0.5h 的候选回流价值。⑥ **工作流自审**: agent 今晚一度推"Slot 1 vr=0.01" 被自身 grep team-memory 拦住 → "grep before quoting" 铁律实战生效; 推 FaFE 被自身识别 "CLI 首次启用" 风险降级。**本地 15 commits 未 push**（含今早 EXP-035 + 今下午 EXP-036 + 15:50 active.md 审视）。

- **2026-05-10 西班牙下午 16:10（UTC+2 · 🎉 seed=42 × gap=0 × DIN × E4 Dev 0.868924 史上最高 + 4 evaluation 已提交）**：① **A · seed=42 × gap=0 × DIN × E4 Dev 0.868924** 超 EXP-019 seed=42 no-DIN E4 baseline 0.868619 **+0.000305**（3× rerun 噪声），**DIN 在 gap=0 首次有正向 Dev 信号**。② **B seed=3 / C seed=7** × gap=0 × DIN 被 LogLoss guard E3 停（E3 后反弹）, best Dev 0.867661 / 0.866622。③ **D Tokenization u4i3** (user_ns 5→4, item_ns 2→3, T=16 维持) × gap=0.05 × E3 Dev **0.843743**（vs T25 B 0.843696 +0.00005 噪声内），但 **LogLoss 更低 −0.00245**（校准信号）。④ **关键机制发现**：gap=0 下 DIN gain +0.000305 ≪ gap=0.05 下 +0.00125 → **data completeness 消耗大部分 DIN 贡献**（边际递减）。⑤ **4 evaluation 全部提交**（quota 全用）：A/B/C/D 4 ckpt，5/11 早 07:00 CEST 看 LB。⑥ **预测**（定性 ceiling, 不给概率）：A 最可能突破 SOTA 0.84637（0.846~0.850 区间），D 是 Tokenization 首次 LB 验证（持平~微升）。⑦ **num_epochs=4 修正价值验证**：A E3→E4 Δ+0.000782 Dev，若停 E3 会丢失。⑧ **seed=3/7 guard 早停提醒**：未来 gap=0 × DIN 实验默认 `--logloss_rebound_delta 0.003 --logloss_rebound_patience 2` 松 guard。⑨ **EXP-036 落档**（~280 行 + LB 情景决策树）。⑩ **本地 15 commits 未 push**（含今早 EXP-035 + 今下午 EXP-036）。

- **2026-05-10 西班牙早 09:55（UTC+2 · 🔬 **Kaggle 金牌视角战略复盘 + EXP-035 落档**）**：用户反问 "请从专业广告算法工程师角度、Kaggle/TAAC 竞赛金牌选手角度再深入看看" 触发 agent 48h audit。**残酷发现**: 48h 消耗 ~13 training slot + ~12 evaluation → **净 LB SOTA 增益 = 0**（仍是 5/4 的 0.84637）。**5 个致命错误**：(a) 过度 seed sweep（1 假设 × 5 seed）; (b) 混淆"均值 vs 上沿"（T25 DIN 均值 0.842 ≠ SOTA 0.846）; (c) 忽略 DECEM 明说的 Tokenization 修改（0 投入）; (d) recency reweight 未做; (e) 3 次同期错误同根"凭记忆陈述"。**今日 T25 DIN seed sweep 实测 LB**: 5-seed 均值 **0.8424**, 上沿 **0.8434**（seed=42, 昨夜）, 距 0.85 差 0.007。**战略 pivot**: 切换山头到 gap=0 领地，3 个 gap=0 × T25 DIN × seed=42/3/7 × 4 epoch job 已启动（CEST 09:36~10:06 错峰）。**贝叶斯估算 gap=0 T25 DIN × 5-seed 上沿**: 后验中心 0.848 ± 0.004 → P(≥0.85) ≈ 31%。**冲 0.85 真实时间线**（金牌式）: 5/12 早 15~25% / 5/15 早 35~50% / 5/25 ddl 前 **75~85%**（不是之前的 >90% 乐观估计）。**EXP-035 落档**: ~280 行 Kaggle 视角 audit + 5 错误拆解 + 正确 48h 分配反事实 + 5 条新铁律建议。**agent 纪律升级**: (i) 不再给具体概率数字（recency bias）; (ii) 未来 48h 强制 audit + DECEM 情报清单检查; (iii) Tokenization 是 TOP 1 未消化线索。**下午决策**: 等 gap=0 3 job 回流（CEST ~15:00~16:00），根据数据决定晚上 pivot 到 T28 还是 Tokenization。**本地 14 commits 未 push**。

- **2026-05-10 西班牙早 08:35（UTC+2 · 🎉 T25 B LB 0.843433 突破 gap=0.05 × DIN bypass 新记录 + 修复 _orig_mod bug）**：昨夜 4 evaluation 结果全出（修复 infer.py 后成功 publish）: T25 B **0.843433**（Dev 0.843696 / Dev→LB gap +0.000263 ⭐极小）/ seed=11 0.841591 / seed=5 0.840166 / T19 rerun loose guard 0.838142。**T25 DIN bypass 方法学真信号**（gap=0.05 基线 LB 0.839223 → T25 B LB 0.843433 Δ+0.00421 = 9× rerun 噪声）。**今日发现 2 个 infra bug**: (a) T27 compile+AMP 的 EarlyStopping 传 `self.model` (compile wrapper) → state_dict 带 `_orig_mod.` 前缀 → T25+T27 evaluation 失败 → fix in `trainer.py:820,835` + `infer.py:_load_state_dict` strip prefix; (b) `run.sh` `num_epochs=3` 是 agent 凭记忆拍的错误值（EXP-019 明确记录 E4 是 best，应是 4 epoch）→ fix 默认为 4。**commits**: `1a7b312` (fix adr-008 _orig_mod + EarlyStopping) + `6c4e117` (fix config num_epochs 3→4)。**下一步**: 基于 T25 B 真信号起 4 个 T25 seed sweep（seed=3/7/11 × 4 epoch + T25+T27 重提）。

- **2026-05-09 西班牙早 10:15（UTC+2 · 🚨 **重大错误修正 · T19 Deprecated 撤回 + T27 LogLoss 解读修正**）**：agent 09:43 凭记忆用**错误配置的 baseline**（gap=0 的 icq-half vs gap=0.05 的 T19）对比，导致两个错判：① **T19 判 Deprecated 错误**：用 gap=0 baseline E2 AUC 0.841124 / LogLoss 0.22468 对比 gap=0.05 T19 E2 AUC 0.840142 / LogLoss 0.28208，得出"Δ−0.00098 + Δ+0.057 异常"误判。**真实 gap=0.05 baseline (EXP-030 §C)**: E2 AUC **0.837453**（反弹） / E2 LogLoss **0.28516**（反弹）→ T19 **E2 AUC 领先 baseline +0.0027 / E2 LogLoss 更低 −0.003** → **疑似正向不是 Deprecated**。② **T27 "LogLoss 抬高 +0.054" 错误判**：同样 baseline 错位。真实对比 T27 LogLoss 0.28397/0.27901/0.27673 vs baseline 0.28280/0.28516/0.27917 → **T27 LogLoss 轨迹优于 baseline**（E2 不反弹 / E3 绝对值更低）。③ **机制问题**: `logloss_rebound_delta=0.0` 在 gap=0.05 配置下过敏（T19 E2 反弹 +0.00035 是 baseline 反弹 +0.00236 的 1/7 就被停 guard）,未来 run.sh 默认建议 `delta=0.003`。**修正动作**: (a) EXP-034 §Correction + 状态 Deprecated→Inconclusive-needs-rerun; (b) ADR-007 Status Deprecated→Paused-Needs-Rerun + §Correction after EXP-030 baseline review; (c) EXP-031 §分析 1 修正 + §Verdict LogLoss caveat 撤回; (d) ADR-008 §Post-EXP-031 caveat 修正; (e) team-memory **撤回 "容量扩张 bet 必死" 铁律**（证据不足）+ 修正 "T27 LogLoss 抬高 bf16 嫌疑" 铁律 + **新增铁律"对比 baseline 必须核对配置"**（grep 原始 EXP 而非凭记忆 / 判决依赖的数据点 + 来源文档路径必须列出给用户 review）; (f) active.md T19 `[x]` → `[!]`。**修订 0.85 概率**: 明早 5/10 07:00 从 25~35% 上调回 **30~40%**（T19 不是失败）; 5/11 早 **60~70%**。**教训**: 在做"当前 job vs baseline" 对比之前, agent **必须 grep 原始 EXP 文档核对 baseline 配置一致性**, 不能凭印象引数字。**耗时**: 这个错误从 09:43 发生 到 10:15 发现修正共 32 分钟, 如果没发现, 会指导后续 5/10~5/25 所有决策错误方向。**T25 B 仍在跑** —— 今日最后方法学弹药, 回流时**必须用正确 gap=0.05 baseline 对比**（E3 AUC 0.842443 / LogLoss 0.27917）。

- **2026-05-09 西班牙早 09:55（UTC+2 · ⚠️ T27 compile+AMP 回流 Accepted with caveats · LogLoss 抬高待 T25 B 确认）**：① **T27 `t27-compile-amp-halftrain-gap05` 完成**：E1/E2/E3 Dev = 0.836638 / 0.840812 / **0.842773**; baseline E3 0.842443; **Δ+0.00033 在 EXP-032 ±0.0003 rerun 噪声带内** → **bf16 正确性通过** ✅; wall-time **4h01m**(vs baseline ~5h) **加速 1.25×** ⚠️（未达 ADR-008 目标 1.5× 也未达 DECEM 报告 2×，推测平台 GPU 非 A100 / sparse reinit 触发 dynamo recompile）; 0 crash / 0 OOM ✅。② **ADR-008 Status: Proposed → Accepted with caveats**，补 §Post-EXP-031 Verdict: full-train ≥10h 强烈推荐（省 2h 值得）, half-train 默认开（省 1h）, smoke/proxy 不开, infer.py 永远不开。③ ⚠️ ~~**LogLoss 抬高 +0.054**（T27 0.276 vs baseline 0.222, bf16 softmax 饱和症状, AUC 无损但校准损失）~~ **10:15 修正撤回**（baseline 对比错误, T27 LogLoss 实际优于 baseline）。④ **T27 ckpt 不 publish/evaluate**: E3 Dev 仅+0.00033 方法学无增益, 占 evaluation quota 不划算。⑤ **间接数据点**: T27 E3 Dev 0.842773 是今天第 2 个"seed=42 gap=0.05" rerun 数据点（baseline 0.842443 第 1 个）, 跨度 0.00033 在 ±0.0003 带内, **稍微扩充 EXP-032 rerun 噪声带到 3 数据点**。⑥ **T25 B 仍在跑** —— 今日唯一剩下的方法学弹药, 预期西班牙时间 11:00~12:00 回流; 如失败明早触 0.85 概率 <15%。⑦ EXP-031 §Result+§Verdict+§Lessons 补档; team-memory 新增 1 条铁律"T27 可选不强制"。⑧ **0.85 概率修订**: 明早 5/10 保持 25~35%（T27 工程加速不改方法学路径）, 5/11 早 55~65%。

- **2026-05-09 西班牙早 09:43（UTC+2 · ⚠️ T19 首发回流 ❌ Deprecated · keep_topk_ckpt 实装 · 冲 0.85 概率下调）**：① **T19 `t19-perffn-halftrain-gap05-noamp` 证伪关闭**: E2 Dev 0.840142（baseline 0.841124 Δ−0.00098 = 10× rerun 噪声）+ E2 Val LogLoss **0.28208**（baseline 0.22468 Δ+0.057 绝对值异常）→ LogLoss guard E2 触发早停（`logloss_rebound_epoch_2`），没跑到 baseline 的 E3 best 就崩。**机制**：+34% Dense params（+992,640）在 1.9M × 30k steps/epoch 摊到每参数 ~30 steps 梯度,warmup 严重不足 → E2 刚收敛 → E3 立刻 overfit。**与 Seed 主楼 "pure model 改造都掉分, 数据量不够" 完全印证**。**T19 task 从 `[~]` 改回 `[x]` Deprecated**; ADR-007 Status 降为 Deprecated 并补 §Deprecation Note; EXP-034 补 §Result+§Lessons; team-memory 新增铁律 "容量扩张 bet 必死 · Dense 增幅≥20% 直接跳过 · 每参数平均梯度 steps 硬约束 ≥60 安全/30~60 高风险/<30 必崩"。② **keep_topk_ckpt 实装完成**（commit 34823f4）: `--keep_topk_ckpt N` CLI + `*.topk_{rank}/` 目录（与 `.best_model/` 并存不冲突）+ 2 阶段 rename 避免冲突 + summary topk_checkpoints 字段; K=3/K=1 双路径 smoke 通过。**含义**：后续所有 half-train 默认 `--keep_topk_ckpt 4`, 单 job 产出 4 个候选 ckpt, **evaluation quota 4× 放大**, 大幅缓解"best AUC != best LB"（icq-full E6 vs half E4 教训）的 ckpt 选错风险。③ **T27 compile+AMP 慢于预期**: 启动 CEST 03:40 预期 3h 回流, 09:43 已跑 ~6h 仍未结束（远超 baseline ~5h）→ compile+AMP **加速未达标**, ADR-008 判决倾向"部分 Go"或 No-go, 等日志。④ **T25 B 仍在跑**, 预期 CEST 10:00~10:30 回流, **今日唯一剩下的方法学弹药**。⑤ **修订冲 0.85 概率**（相对 06:05）: 明早 5/10 07:00 从 35~45% 降到 **25~35%**（T19 归零）; 5/11 早从 60~70% 降到 **55~65%**（T25 + seed sweep 仍可行）。⑥ **战术启示**: 今日 4 slot 里 3 个是 🟢 方法学 bet（T27 / T25 B / T19）, T19 失败是 50%×50%×... 的概率事件, 符合 EXP-032 战术铁律"每批 ≥3 🟢" 方向正确, 不悔。⑦ **本地 7 commits 未 push**（待 T25 B 回流后统一 push）。

- **2026-05-09 西班牙早 06:12（UTC+2 · 🎯 三路并发架构级 bet + 冲 0.85 战略 · 路径 C）**：平台 LB 约 20 人达 0.85+，我方 SOTA 0.84637（均值 0.8424）差 Δ 0.008 能力均值。采用**路径 C（方法学叠加 + seed sweep 混合）**：① 三路并发 🟢 架构级 bet 启动（严格遵守 EXP-032 战术铁律"每批 4 slot ≥3 🟢"）: T27 compile+AMP（工程加速 running, 03:40 启动）+ T25 DIN bypass 方案 B（方法学 bet）+ **T19 Per-token FFN**（方法学 bet · 本地 smoke Dense +34% = +992,640 已验证, EXP-034 落档）。② **T19 task 从 `[x]` 重启为 `[~]`**：偿还 2026-05-04 "四合一被污染未单独验证" TODO, 严格单变量 ablation, 不叠加 T25/T27。③ **统一判决阈值**（EXP-033/034）：E3 Dev ≥ 0.847（ΔDev ≥ +0.0046 = 10× EXP-032 rerun 噪声 ±0.0001）真信号; [0.8444, 0.847) 疑似; [0.8412, 0.8444) 持平 Deprecated; <0.8412 负向。④ **时间线预估**：今日 11:00~14:00 三方向 E3 Dev 回流 → 晚上择机 evaluation 或起 seed sweep → **5/11 早**（4 seed sweep + 1 次 LB 校准）**触 0.85 概率 60%**，悲观 5/13 早。⑤ **并发 OOM 风险 20%**：T27 ~10GB（bf16）+ T25 B ~14GB + T19 ~17GB 同卡绝对 OOM, 分卡则 OK; T19 应急 `--batch_size 96` 降到 ~13GB。⑥ **quota 使用**: training 3/4 满（T27 + T25 B + T19）/ 1 空闲; evaluation 4/4 全剩。⑦ **EXP-034 落档**（T19 实验计划, 对齐 EXP-031/033 结构，~260 行）。⑧ **本地 5 commits 未 push**，等今日三方向全部回流后统一 push。

- **2026-05-09 西班牙早 05:55（UTC+2 · T27 平台启动成功 + T25 代码就位 + EXP-032 落档 + 5 条新铁律）**：① **T27 修复**: 平台首次启动报 `UnboundLocalError: local variable 'torch' referenced before assignment`，根因 `baseline/trainer.py::__init__` 内 `try: import torch._dynamo` 让 Python 把整个函数作用域的 `torch` 标为 local。改用 `getattr(torch, '_dynamo', None)` 访问子模块（commit `38d6ea2`）。T27 `t27-compile-amp-halftrain-gap05` 已在平台运行中，预期 ~3h 回流。② **EXP-032 落档**（昨晚 4 个 half-train rerun batch · 25 GPU-hour）：3 次 seed=42 同 config rerun dev 跨度 **0.000143** → 同 rerun 噪声 ±0.0001（跨 seed 噪声 ±0.005 的 50× 小）; Job 2 `valid_ratio=0.01` 消融 E4 dev 0.865913 Δ−0.003 → 方向关闭; Job 3 seed=3 E4 LogLoss 0.22473 反弹 → LogLoss guard **首次生产触发**, best_model=E3 AUC-selected。③ **team-memory 新增 5 条铁律**: (a) Python 函数内 `import X.submod` 陷阱（工程）; (b) rerun 噪声 ±0.0001 下界（判决阈值）; (c) vr<0.05 方向关闭; (d) LogLoss guard proven; (e) platform job 建议必须标注 🟢 架构级 / 🟡 信号校准 / 🔴 small ablation, 每批 4 slot ≥3 🟢。④ **T25 G1 DIN 代码就位**: `DINInterestExtractor` (bypass 版, 4×CrossAttn + merge, `model.py:563-666`) + `enable_din_interest` CLI + infer.py 兼容; 本地 smoke 4/4 通过（train 5 steps loss 0.62→0.32 / train_config 正确落盘 / infer 1000 predictions / Dense 参数 +104,896 符合理论）; **EXP-033 实验计划落档**, 2 个 run.sh (A=叠加 T27 推荐, B=fallback) 准备好。⑤ **T25 判决阈值**（利用 EXP-032 铁律 ①）：E3 Dev ≥ 0.847（ΔDev ≥ +0.0046 = 10× rerun 噪声）真信号立即 eval; [0.8444, 0.847) 疑似; [0.8412, 0.8444) 持平 Deprecated; <0.8412 负向。⑥ **战术偏离复盘**（EXP-032 战略章节）: 昨晚 4 slot 全是 🟡/🔴 类（2 次 SOTA rerun + seed=3 + vr 消融），0 个 🟢 类 → 损失 1 次架构级 bet 回流时间; 未来必须严格按 ≥3 🟢 配比。⑦ **quota**: training 1 用（T27）/ 3 空闲（T25 待启动占 1 后剩 2）; evaluation 0 用 / 4 剩。

- **2026-05-07 西班牙早 09:40 (UTC+2) / PDT 00:40（🔥 4 evaluation 回流 · SOTA 确认 seed luck 上沿 · Gap=0.05 funnel 实测成功 · F1 关闭 · T25 DIN 启动）**：**LB 数据**: `icq-half-gap05` **0.839223** / `hash100k-gap05` **0.838417** / `icq_seed3` **0.845329** (第 2 高！) / `icq_seed7` **0.836413** (最低)。**3 大发现**: (1) **Seed 噪声 ±0.005**（5 seed LB 跨度 0.010, 均值 0.8424, SOTA 是上沿）, **Dev 与 LB 在 seed 维度几乎独立** (seed=7 dev 最高 LB 最低, seed=3 反向), 实锤 "SOTA = seed luck" 假说; (2) **Gap=0.05 dev→LB gap 实测 +0.002~+0.003**(vs gap=0 的 +0.022~+0.032, 小 **10×**) → **dev 变极可信 LB 代理**, 精度 ±0.003, 未来方向级 ablation 不花 eval quota 就能筛方向; (3) **F1 hash100k 持平 (ΔLB −0.0008 在噪声内), 不再追 hash 激进方向**。**战略级转折**: (a) **真实能力带 [0.836, 0.846]**, SOTA 不是天花板而是 seed 上沿运气; (b) **破 SOTA 新阈值**: 不是 "单点 LB ≥ 0.849", 而是 "方法学使能力均值 0.8424 提升 0.005+" (即 **LB ≥ 0.8474 等价能力提升**); (c) **实验迭代 10× 加速**: 5h half-train + gap=0.05 看 dev 直接判决, 不用 eval quota; (d) 下一批唯一方向 = **G1 DIN target attention** (Seed 主楼方案 2 对口, hard attention 与 T16 additive 机制完全不同, ~80 行 model.py)。**T24 F1 归档** (持平关闭), **T25 G1 新建** (DIN)。**EXP-030 落档** 4 eval 完整分析 + 5 seed LB 分布图 + 判决阈值矩阵。**当日 quota**: evaluation 4 全用 (剩 0); training 4 slot 空闲, G1 DIN 代码完成后立刻启动。**注**: 用户今日下午/晚西班牙当地时间有空, Agent 文档 + 代码 + smoke 今日全部完成。

- **2026-05-06 凌晨 (PDT 晚 22:40)（🔥 C6 rowgap15 LB = 0.833987 揭示 "row-level gap 让 dev 可信" 重大 insight · T22 从 Blocked 改为 "机制成功"）**：C6 E2 ckpt evaluation 完成, LB **0.833987** / Dev 0.837987 / **Dev→LB gap +0.004** (其他所有实验 gap 都在 0.022~0.030, C6 小 5~7 倍)。**关键发现**: Row-level valid gap 确实改变了 dev 的性质——之前 dev ≈ 0.868 是"训练期近未来" AUC 含污染, C6 dev ≈ 0.838 是"真正模拟 test 分布" 的 AUC。**T22 判决修订**: 从 "Blocked RG-level 失效" 改为 "机制成功 / 参数需调优"——funnel 机制确实工作, 但 `valid_gap_ratio=0.15` 扔 15% train 数据代价过重, 最佳 gap_ratio 预测 ≈ 0.05 (线性插值: dev 0.848 / gap 0.010 / LB 0.838~0.840)。**真实能力带估算**: [0.835, 0.842], SOTA 0.84637 包含 ~0.004 "训练期污染 + seed luck" 虚高。**下一批实验默认策略修订**: 所有 T24 job 带 **`--valid_gap_ratio 0.05 --enable_row_valid_gap`** —— 代价 5% train loss, 收益 dev 变可信 (gap ~0.010 vs 当前 0.022~0.030), 为所有后续决策质量升级。**新基线需建**: `icq-half-gap05-icq-halftrain` (0 代码) 作为带 gap 配置的 LB 锚点, 预测 LB ~0.838~0.840, 与之对比做 F1 hash100k 等新方法判决 (破 0.845 = 真信号)。**文档**: EXP-028 加 C6 LB + "row-level gap 机制成功" 补遗; EXP-026 补 row-level 版本 LB 表 + 线性插值预测; active.md T22 状态 + T24 "默认带 gap=0.05"。**今天战略收获**: 真正的破局不是 "seed luck 假说"，而是 **"row-level gap 解锁可信 dev"**——从明天开始我们不再是 "瞎猜 dev"。

- **2026-05-05 PDT 晚 22:30（EDA 完成 · EXP-029 · F1 hash100k 定位下轮最高优先级 · seed3/seed7 training 启动）**：① 启动 2 training job: M1 `icq_seed3-icq-halftrain` + M2 `icq_seed7-icq-halftrain` (seed=3/7, 其他与 icq-half SOTA 完全相同, 0 代码改动), 为 **seed luck 假说**建立 multi-seed LB 数据点; ② **EDA 60 分钟完成** (EXP-029), **3 大发现**: (a) platform 顶层 user_int/item_int 最大 vocab 25k (fid=9 item_int), "item_id OOV 问题" **只在 seq 域** → UE 分离 / 顶层 disable 方向**作废**; (b) C1 E1 valid 切片表 (`[diag] slice by`): user_int_nonzero **稀疏 AUC 0.8915 vs 丰富 0.8471** (Δ0.045), seq_total_len **短 0.8991 vs 长 0.8453** (Δ0.054), item_int_nonzero 类似 → **模型在信息丰富样本上真实过拟合 id**, 验证 Seed 主楼问题-1 机制; (c) 大 vocab 全在 seq 域 (seq_c/47=2.8亿, seq_b/69=8600万, seq_c/29=820万, seq_c/34=200万, seq_c/36=150万); ③ **T24 重新排 ROI**: 🥇 F1 `seq_hash_vocab 更激进` (500k→100k 或 50k, **0 代码**) / 🥇 F2 DIN target attention (~80 行) / 🥈 F3 样本丰富度加权 / 🥉 F4 FaFE / ❌ UE 分离 (EDA 作废) / ❌ 顶层 disable (EDA 作废); ④ **下个实验定位**: **`hash100k-icq-halftrain`** 是最对口 + 最简单的架构级方向, 后天落地西班牙启动; ⑤ **今晚 E+EDA 策略**: 不再提交新 training (quota 平衡与验证节奏), 让 M1/M2/C6 跑完, 文档收尾; ⑥ **quota 时间线**: 明天飞机上 seed3/seed7 回流 + C6 E4 出, 后天西班牙当地 02-06 (PDT 17-21) 将有 2-3 eval quota 可用, 优先 eval seed3/seed7 建立噪声带, 再 eval F1 hash100k。

- **2026-05-05 PDT 晚 21:40（🔥 "SOTA seed luck" 假说提出 · T23 归档 · 3 次 evaluation · T24 架构级创新开启）**：今晚 eval 窗口重置后用 3 次 quota (剩 1): ① T23 `disablefid47-icq-halftrain` LB **0.840344** (Δ−0.006 vs SOTA); ② C1 `disablefid47_29` LB **0.836736** (Δ−0.010); ③ **C12 `icq_seed2-icq-halftrain` LB 0.841294** (**完全同 icq-half config 只换 `--seed 2`**, Δ−0.005, **Dev 几乎相同 Δ=−0.00004**)。**🔥 关键发现**: **Evaluation system deterministic** (用户确认同 ckpt 两次 eval LB 完全一致) → C12 的 LB 差异 **100% 来自训练轨迹** (seed → sparse reinit + optimizer state 分岔)。**推论**: SOTA 0.84637 很可能包含 seed luck, 真实 LB 能力带 **[0.840, 0.846]** 中位 0.843, SOTA 是 outlier。**铁律**: ① **Dev 0.868 附近完全不能作为 LB 代理**(LB/Dev 信息量比 10~126×); ② **"LB 负向证伪" 阈值调整**: Δ−0.010+ = 真方法学负向 / −0.004~−0.007 = 疑似(也可能 seed 噪声) / −0.003 内 = 纯 seed 噪声; ③ **不再追 0.846 单点 SOTA 作为基线**, 真实基线 ≈ multi-seed 均值 0.843~0.844; ④ **破 SOTA 判决阈值从 "LB ≥ 0.847" 升为 "LB ≥ 0.849"** (明显高于 seed 噪声上界); ⑤ **砍掉所有 small ablation**, 只做真正架构级创新 (DIN / UE 分离 / FaFE)。**文档**: EXP-028 落档详细推演 + 反面考虑 (反面 1: icq-half seed=42 可能真正触达罕见 LB minima, 反面 2: LB 差异可能主要来自 ckpt 时机而非 seed); EXP-025/026/027 全部加 "interpretation-revised" 脚注; team-memory 新铁律 (第 8 条 seed luck)。**T23 归档** (方法学负向 + seed 噪声叠加), **T24 新增** (EDA + 架构级创新 next batch)。**状态**: C6 rowgap15 仍在跑 E4, quota 剩 1, 明日用户飞行期间不方便操作, 计划今晚不再 evaluate (除非 C6 E4 异常高)。

- **2026-05-05 PDT 中午 11:50（4 并行 ablation 批次回流 + T21 证伪 + T22 Blocked + T23 落地）**：4 个 parallel job 全部回流, 1 次 evaluation 消耗后 LB 红线 0.84637 未破。① **四合一 E4 dev 0.866133** (vs icq-half 0.868619 Δ−0.0025), holdout 0.867182 预估 LB ≈ 0.845, 不 eval; ② **T16-only full E2 best 0.865122** (vs icq-half E4 Δ−0.0035), E3 触发 logloss rebound guard 停, T16 additive α=0.1 full train 下更早过拟合, **ADR-004 方案 A 单独路线关闭**; ③ **idmaskc15 E3 dev 0.868288** (vs icq-half E3 +0.000323, 唯一正向 dev 信号) → **evaluation LB 0.842176** (vs SOTA 0.84637 **Δ−0.004194**, dev→LB gap +0.0261 放大 2×), **T21 P1 证伪关闭**; 机制: input-level random mask 85% token 仍可见,模型仍依赖 id embedding,dev 近 train 略涨, test 远 train 失效; ④ **gap15 E4 dev 0.867124** (Δ−0.0015 完全匹配"减 15% train data"纯预期), 日志 `effective train→valid distance = -341046 sec` 证明 **RG-level 切分无法隔离时间** (RG 内部时间跨 4 天), **T22 P3 Blocked**, 修方向 = row-level gap cutoff (T22-fix 暂不启动); ⑤ **T23 seq_c feat47 structural disable 代码落地**: `baseline/dataset.py` PCVRParquetDataset `disable_seq_fids` 字段 + `_convert_batch` slot 置零 + `get_pcvr_data` 传三路 / `train.py` CLI `--disable_seq_fids` / `baseline/infer.py` `_parse_disable_seq_fids` 透传 → **训推一致**; 本地 smoke 5/5 通过 (None/[] bit-identical / `[47]` 只归零 seq_c slot=10 14138 ids / shuffle=True/False 同效果 / lengths 未变 / 非法输入抛 ValueError); 对口 Seed 主楼问题-5 "seqc feat47 跟 itemid 是一样的 做不做 match 特征都掉分"; ⑥ **EXP-024 "dev 不是 LB 单调代理" 铁律再次验证**: idmaskc15 dev +0.000323 → LB −0.004194, 反向放大 14×; ⑦ **EXP-025 / EXP-026 / EXP-027 落档**, EXP-027 汇总表 + 机制小结 + 剩余方向 ROI; ⑧ **quota**: evaluation 1/4 用剩 3; training 4 slot 全空闲等 PDT 21:00 前启动下一轮 `disablefid47-icq-halftrain` (T23 首个实测)。**LB 红线 0.84637 未破, 距 ddl ~20 天, 剩余方向 ROI 排序: T23 seq_c disable (最高, 代码已就位) > T22-fix row-level gap (修 funnel 不直接贡献 LB) > T16 方案 B/C > DIN / UE 分离**。

- **2026-05-05 PDT 凌晨 0:30（🔥 社区 Seed 主楼深度解析 · P1 + P3 代码落地 · 2 job 启动）**：用户推送 Seed (LB 0.85+) 主楼 "腾讯广告算法大赛 Day x" 截图,抓到 5 条未消化情报。3 条直接对口本项目: ① 问题-1 "uid 训练集测试集几乎全是一次性 user 且无交集 / item id 线下 auc 虚高 → 线上掉 / 估计是分布问题" —— 证实我们 EXP-018 观测到的 dev→LB gap +0.021~+0.030 机制是 id 特征 OOV 导致; ② 问题-4 "pure model 主流 rankmixer/onetrans/hyformer/mixformer 都改 pure attention + FFN 为了 MFU,Seed 实测都掉分,可能数据量不够" —— 排除 kernel-fused 重写路线,本项目 HyFormer + RankMixerNS 方向对口; ③ 问题-5 "seqc feat47 跟 itemid 是一样的,做不做 match 特征都掉分" —— 证实 T13 hash500k 谨慎处理正确,且反向说明 seq_c 作为 item_id 序列信号稀薄。评论区 Pung 建议 "item id OOV 率太高 → 训练时随机 mask" —— 方向与上述一致。**代码落地 2 条正交子弹**: ① **T21 P1 Item-ID Random Mask** (baseline/dataset.py PCVRParquetDataset + get_pcvr_data, baseline/train.py,共 ~115 行): 训练时对指定 domain 按 token 粒度 Bernoulli 采样 mask,双 gate (`is_training AND shuffle AND prob>0`) 保证推理 bit-identical。domain 名字支持 `'c'` 和 `'seq_c'` 两种。5/5 smoke 通过: prob=0.5 全 domain 实际丢弃率 0.497~0.511 / domain=['c'] prob=0.9 只掉 seq_c 0.903。② **T22 P3 Valid Split Gap** (baseline/dataset.py::get_pcvr_data 新增 `valid_gap_ratio` 参数): time-sorted RG 切分顺序变为 `[train | gap (dropped) | valid | holdout]`,gap 区间完全丢弃不参与训练也不算指标。参数校验齐全 (需 `valid_split_strategy='time'` / 与 `enable_row_time_cutoff` 互斥 / `valid + holdout + gap < 1.0`)。日志 `[P3] gap ts (dropped) + effective train→valid distance`。**2 job 启动**: `idmaskc15-icq-halftrain-swiglu-vr005-bs128` (~00:00 PDT, +`--id_mask_prob 0.15 --id_mask_seq_domains c`) 与 `gap15-icq-halftrain-swiglu-vr005-bs128` (00:28 PDT, +`--valid_gap_ratio 0.15`),都是 icq-half SOTA (LB 0.84637) 的**单变量 ablation**,严格遵守 EXP-024 的归因纪律。**training quota 4/4 全满** (T16-only full + 四合一 + P1 + P3),evaluation 3/4 剩 1 slot。EXP-025 / EXP-026 落档, active.md T21/T22 新增, ADR-005 方案 C 进入首个实测阶段。**P3 判决逻辑关键: 主指标不是 dev 绝对值,而是 `dev→LB gap`**,dev 绝对值下降是预期且正常的。infer.py 零改动,与平台默认 infer.py 完全兼容。

- **2026-05-04 PDT 晚 11（🔬 方法论深度复盘 · EXP-024 落档 · T20 新增 · ADR-006 机制修订）**：用户反思 "为什么离线屡屡提升,评估却反向?funnel 是不是有问题?" + "有没有可能测试集有问题?"。系统性复盘今日 3 次 evaluation (其中 2 次 LB 反向 −0.006 / −0.010,合计 LB 下降 −0.016)。**核心诊断**: funnel 设计本身有系统性缺陷 —— dev AUC 测 "train 前 95% → 尾部 5%" 的 generalization,LB AUC 测 "train 100% → 远未来 test" 的 generalization,是**两个不同任务**;按 dev 决策在本赛题 distribution shift 场景下相关性不稳定(3×~11× 放大,方向不固定);Proxy 和 half-train 无法区分"慢性泛化失败";"正交子弹可叠加" 假设站不住(target-conditioning + ranking loss 可能共同加大 overfitting);组合实验违反单变量 ablation 纪律(三合一 LB −0.010 无法干净归因)。**test 集 bug 假设**: 大概率否定(gap 系统性 + SOTA 0.858 可达 + ICQ 正向响应 3 证据),但 "版本核查" 5 分钟可做。① **EXP-024 落档**(独立方法论复盘文档): 完整包含所有 LB 数据点、funnel 缺陷诊断、6 个改革方向(valid split 重设计 / 归因实验纪律 / keep_topk_ckpt / proxy 重新定位 / evaluation 作数据源 / ranking loss 降优先级)、P0/P1/P2/P3 可执行行动项、7 条最诚实的自我批评(过早给 T18 判死刑 / holdout 成本低估 / dev 涨阈值机械化 / 忽视归因实验的合理消耗方式);② **T20 `[!]` 新增**: Funnel 重设计 task,**优先级最高**,在 funnel 修复之前任何方法学创新 LB 信号**不可信**,不应投 quota;③ **ADR-006 Deprecation Notice 机制修订**: 原 "BPR 破坏 calibration" 论证逻辑站不住(AUC 是排序指标,单调变换不改 AUC),修订为更严谨的 **"BPR 学到的排序依赖 train 绝对分数分布,在 test 分布偏移下不稳定" 机制**,加入**归因歧义段**(三合一是 3 变量组合实验,LB −0.010 分解: T18 -0.003~-0.007 / holdout -0.002~-0.005 / 架构差 ±0.002 / patience ±0.001),明确 T18 Deprecated 判决仍成立(机制+辅证)但贡献幅度不可独立归因;④ **team-memory 2 条铁律更新**: 原"BPR 破坏 calibration"改为排序稳定性机制 + 归因歧义 + 组合实验禁令(第 5 条); 新增第 6 条 "Holdout 切分可能有隐藏 train data loss 成本,T16-only full train 可检验,若 LB < 0.846 则反证立刻去掉 holdout";⑤ **最诚实小结**: 在积累信号的关键节点,**反思 > 继续推进**;今天早 6 小时进入复盘模式可能用 2 次 evaluation 就能获得今天 3 次的信息量,并避免三合一组合实验错误。

- **2026-05-04 PDT 晚 10（🔥 T18 LB 实锐证伪 −0.00993 · T18/T19 task 关闭 · T16-only full train 接替）**：三合一 E4 evaluation 结果回流:LB **0.836445**(dev 0.865516 / holdout 0.867196)。vs SOTA 0.84637 **Δ−0.00993**,dev→LB gap **+0.0291** / holdout→LB gap **+0.0308**,**双双历史最大**,Holdout 未能成为 LB 代理。诊断：CombinedAUCLoss(BPR α=0.3)破坏 sigmoid 绝对值 calibration,在"远未来 test + 时间分布偏差"场景下 gap 放大 3~4×,外部 anchor(Tencent 同分布 test +0.7~1.26% GMV)不可直接照搬到本赛题。① **ADR-006 Accepted → Deprecated**,首部添加 Deprecation Notice 完整记录机制与影响;② **T18 task 关闭**,active.md T18 节简化为关闭声明,代码保留(`combined_auc_loss` 纯函数 + trainer `elif loss_type=='auc'` 分支不回滚,便于未来 calibration-aware ranking loss 验证),默认 `--loss_type bce`;③ **T19 task 同步关闭**:四合一 E1/E2 dev 持续落后三合一 0.0008~0.0016,四合一不可 evaluation(与 T18 叠加必 LB 崩盘),T19 独立收益未单独验证,ADR-007 保留 Accepted 但 task 关闭,future 若重启必须不叠加 T18 单独 full train;④ **EXP-022 落终结状态**,附三合一全 E1~E4 epoch 表 + evaluation 对照;⑤ **team-memory 新铁律**(第 5 条):BPR/Ranking loss 在"远未来 test"破坏 calibration + holdout→LB 经验公式仅在纯 BCE 路径下有效 + 不同 loss 跨类比 gap 不可预测 + 外部 anchor 同分布 test ablation 不可直接照搬;⑥ **T16-only full train 接替**:run.sh 已切换到 `target_item_seq_injection=additive + alpha=0.1 + holdout_ratio=0.05 + logloss_rebound_patience=1`(严格,vs halftrain 的 patience=2),Job Name `t16-add-a01-ho005-lgrp1-full-swiglu-vr005-bs128`,10 epoch + LogLoss guard 保护,预期 best ckpt dev ≈ 0.869, LB ≈ 0.847~0.848(突破 SOTA 0.84637 +0.001);⑦ **本地 smoke 通过**:Args 正确(T16 on / T18 off / T19 off / T17 on / guard p=1),LogLoss guard banner 打出,best_model 落盘正常,py_compile + ReadLints 全过;⑧ **当日 quota**:evaluation 3/4 已用(t13tf-loss-full + icq-half E4 + 三合一 E4),剩 1;training quota 四合一停止释放 1 slot,T16-only full train 占 1,剩 2。

- **2026-05-04 PDT 晚 9（⭐ LB SOTA 0.84637 刷新 · icq-half E4 evaluation 成功 · 3 条新铁律）**：用户 evaluation `icq-t13tf-halftrain-swiglu-vr005-bs128_eval_1777952701341` → LB AUC **0.84637**,vs 前 SOTA t13tf-full E4 LB 0.844956 提升 **+0.001414**,**建立新 LB 天花板**。配合早前 `icq-t13tf-full` E6 LB 0.843599（local 0.869138 global best）构成黄金对照：**local 涨 +0.000519 但 LB 反跌 −0.00277（负向杠杆 ~5.3×）**。机制：full E5 起 LogLoss 首次反弹（+0.000156）标志过拟合开始,E5~E6 的 AUC 增益来自 train/dev 共享的局部时间模式记忆,test 远未来泛化失败;half-train E4 未跨越拐点,泛化更优。**3 条新铁律写入 `team-memory.mdc`**：① Half-train E4 ckpt 的 LB 可以优于 full-train best ckpt —— 新实验首次 evaluation **优先评 half-train E4**,而非等 full train 完成；② valid→LB gap 随训练长度扩大而非常数（icq-half E4 +0.0223 → icq-full E6 +0.0255,E5+ 每 epoch ≈ +0.0005~0.0015）,更准确公式 `ΔLB ≈ Δlocal − Δgap`；③ Holdout AUC 在 half-train E1/E2 比 dev AUC 更稳定/更优（三合一/四合一 E1~E2 实测 holdout 持续高于 dev ~0.001~0.002）,决策"要不要 evaluation" 用 holdout 估 LB（经验公式 LB ≈ holdout − 0.022,待后续校准）而非 dev。EXP-019 扩充补档全 10 epoch full train 表 + evaluation 对照 + 3 条新铁律。**LB 红线从 0.844956 升为 0.84637**,后续所有策略评估阈值以此为准。**当日 evaluation quota**：2/4（t13tf-loss-full + icq-half E4）,剩 2 slots 至 PDT 21:00 窗口重置。三合一/四合一 half-train 仍在跑,等 E3 判定是否突破 0.84637。

- **2026-05-04 PDT 晚 8（T16 E3=0.868061 达标 → 四合一 half-train run.sh 就位）**：T16 half-train E3 回流 0.868061,E1 0.8645 → E2 0.8664 → E3 0.868061（ΔE2→E3 +0.00166 健康斜率）。判决：**超 ADR-004 S3 阈值 +0.00086**,信号强度是 ICQ（+0.000096）的 **10×**,确认 target-item seq 侧注入（additive α=0.1）在 half-train 规模上是真实的方法学级收益。决策走"方案 A 四合一 half-train",把 T16 / T18 / T19 三条正交子弹 + T17 holdout 搭车打包一次 4 epoch half-train。① **保守参数调整**：T18 `auc_loss_weight` 从草稿 0.3 → **0.2**,`auc_margin` 从 0.5 → **0.0**(vanilla BPR),理由: T19 per-token FFN 已加 +993k dense 参数（Dense 2.47M → 3.46M, 容量 ~15×）+ BPR 梯度叠加可能放大噪声,首次组合保守探路,E3 回流后再根据信号决定是否 α=0.3 / margin=0.5 乐观升级。② **run.sh 覆写为四合一版本**（`baseline/run.sh`）,保留 T16-only + baseline-only 两个 fallback 注释块便于回滚。③ **本地四合一 smoke 全过**：Args 完整带 `target_item_seq_injection='additive' + alpha_init=0.1` + `loss_type=auc, auc_loss_weight=0.2, auc_margin=0.0, auc_max_pairs=32` + `rank_mixer_ffn_mode='per_token'` + `holdout_ratio=0.15`；`Row Group split (time): 9 train, 1 valid, 1 holdout` 四路切分正确;`[ADR-005] holdout_loader attached`;Dense params **3.46M**（基线 2.47M,+993k 符合 T19 预期）;trainer banner 正确读入 `loss_type=auc, auc_loss_weight=0.2, auc_margin=0.0`;best_model + summary.json 落盘;epoch 末 holdout eval 正常算 gap_auc。④ **平台提交格式**：Job Name `t16-add-a01-t18-auc-w02-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`;上传 **5 个改动文件**（model/train/trainer/utils/infer.py）**必须** Create New Training Job（不能 Copy）。⑤ **E3 判决阈值**：≥0.872 强信号进 full train / 0.870~0.872 中性进 / 0.869~0.870 ≈ T16-only,T18/T19 零贡献 / <0.868 拆路径归因。⑥ **EXP-022 / EXP-023 更新**记录实际提交参数。

- **2026-05-04 PDT 晚 7（T19 Per-token FFN 落地 + 四合一 half-train 策略成型）**：接 T18 之后,用户拍板走"选项 2":15 min 准备四合一 half-train run.sh,剩余时间写 T19。外部 anchor：`docs/00-context/model-py-section-map.md` §RankMixer 已明确标注 1 个月"仓库 shared FFN vs 论文 per-token FFN 偏差是 +0.002~0.005 AUC 子弹",RankMixer 原论文（arXiv 2507.15551 §3.2）动机是"feature subspace heterogeneity, allocating independent parameters ... rather than allowing domination by a single shared model",对口本赛题 `domain_c_seq_47` vocab=278M 极端长尾。① **代码落地**：`baseline/model.py` `RankMixerBlock.__init__` 加 `ffn_mode: str = 'shared'` 参数,per_token 路径用 `nn.Parameter` 创建 `fc1_weight/fc2_weight (T, D, H)/(T, H, D)` + 对应 bias,forward 用 `torch.einsum('btd,tdh->bth', x, fc1_weight)` 向量化实现（避免 Python for-loop,保持 GPU 并行 + torch.compile 兼容）；初始化按 T 个独立 Kaiming uniform,数值上等价 T 个独立 `nn.Linear`。`MultiSeqHyFormerBlock` 和 `PCVRHyFormer.__init__` 加 `rank_mixer_ffn_mode` 透传;`train.py` CLI 注册 `--rank_mixer_ffn_mode {shared, per_token}` 默认 shared;`infer.py` fallback `cfg.get(..., 'shared')`;`trainer.py [summary]` 白名单加字段。共 ~100 行。② **S0/S1 验证全过**：`py_compile` 6 文件 exit 0；10 项单元级数值检查（参数量 D=64/T=16/hm=4 下 **shared 33,344 vs per_token 529,664 = 15.88×** / forward shape / **per-token 独立性**：扰动 token 3 fc1_weight *1000 只 token 3 变 188 其他严格 0 / **数值等价独立 nn.Linear max|Δ| = 0.00e+00** / 非法 ffn_mode 抛 ValueError / 三种 mode × 两种 ffn_mode 矩阵兼容 / `MultiSeqHyFormerBlock` 透传 +496k dense/block）；2 路径 demo smoke `shared` vs `per_token`,**shared 默认 AUC 0.8737376434 与 T18 backward-compat case bit-identical**（向后兼容零破坏）,`per_token` train loss 显著降低 0.31 vs 0.37 证明容量生效。③ **ADR-007 Accepted**,30-decisions README 索引更新。④ **T19 `[~]` 新增** active.md 节点,S2~S5 漏斗明确。⑤ **EXP-023 建立**。⑥ **四合一 half-train 策略成型**：`t16-add-a01-t18-auc-w03-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`,把 T16/T18/T19 三条正交子弹 + T17 holdout 搭车打包一次 4 epoch half-train,一次 training quota 采集三条独立子弹信号 + 搭车 holdout 判决数据,保守预期 E3 +0.002~0.005 AUC。⑦ **下一步**：等 T16 E3（当前 E1 0.8645 / E2 0.8664）决定走四合一 half-train 还是拆路径。⑧ **.gitignore 补 docs/99-raw/external-refs/** 屏蔽外部 repo clone。

- **2026-05-04 PDT 晚 6（T18 CombinedAUCLoss 落地 + 外部情报深读 + 同赛题 UniRec 参考吸收）**：用户发起"尽快超 SOTA"战略讨论 → Agent 深度搜集外部学界/工业界情报（GitHub `hojiahao/TAAC2026` UniRec 41 stars 同赛题方案完整深读 + arXiv 2403.14144 Tencent 生产数据 + CIKM 2025 InterFormer + Meta HSTU 2.0 + MuonRec + CAN 2022 + PWiseR 2024）。关键发现：**CombinedAUCLoss = (1-w)·BCE + w·PairwiseBPR** 是唯一同时满足"同赛题开源默认 loss + 腾讯自己 production +0.70~1.26% GMV 数据背书 + 本 repo 完全未消化 + 与 T16/T17 正交"四条的**工业级杠杆**，ROI 最高。① **代码落地**：`baseline/utils.py` 新增 `pairwise_bpr_loss` + `combined_auc_loss` ~130 行纯函数（含 `max_pairs` P×N 爆炸保护 + 空正负 batch 零梯度保护 + `auc_weight=0` 短路等价 BCE 的向后兼容分支）；`baseline/trainer.py::_binary_loss` 加 `elif loss_type == 'auc'` 分支直接 return scalar；`baseline/train.py` CLI 加 `--auc_loss_weight / --auc_margin / --auc_max_pairs` 3 参数，`--loss_type` choices 扩为 `{bce, focal, auc}`，默认 `bce` 不变。`baseline/infer.py` 零改动。共 ~180 行代码改动。② **S0/S1 验证全过**：`py_compile` 6 文件 exit 0；8 项单元级数值检查（BPR 正样本全高 loss→0 / 正样本全低 loss→10 / 空正负保护 / `combined=(1-w)·BCE+w·BPR` 加权恒等 / `auc_weight=0` 等价 BCE / 同 seed 可复现 / 非法参数抛错 / `sample_weight` 仅影响 BCE）；3 路径 demo 集成 smoke `bce` vs `auc w=0` vs `auc w=0.3`，前两者 **bit-identical**（auc=0.8737376434, logloss=0.3066050708），向后兼容红线通过；`train_config.json` 正确落盘 4 新字段。③ **ADR-006 Accepted**，30-decisions README 索引更新。④ **T18 `[~]` 新增** active.md 节点，S2~S5 漏斗与阈值明确（S2 proxy ≥ −0.002 / S3 halftrain ≥ +0.0003 / S4 full ≥ 0.870 / S5 LB ≥ 0.847）。⑤ **EXP-022 建立**，完整记录 S1 数据与 S2 配置。⑥ 下一步：等 quota 释放启 `t18-auc-w03-t13tf-proxy` 或与 T16 full train 打包成三合一 half-train（T16 + T18 + holdout）。

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
