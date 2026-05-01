# ADR-003 · 特征工程作为 P0 路径（候选清单 + 防泄露设计 + 实施时序）

> **Status**: Accepted
> **Date**: 2026-05-01
> **Related Tasks**: 待开 T11/T12/T13/T14（本 ADR 落地后由 active.md 创建）
> **Related Experiments**: 待 EXP-003（T8/T10 数据回流）后启动

---

## Context（背景与触发原因）

### 触发证据

- **leaderboard 差距**：当前最佳 0.84381（swiglu）vs 社区 SOTA 0.85809，**差 +0.0143**
- **ensemble 路径被禁**：`docs/00-context/competition.md` line 77「严禁模型集成 / 模型融合（ensemble）」→ 原计划 Stage 3 的 multi-seed/multi-encoder ensemble **不可用**，必须用单模型方案补回这部分增益（见 §Alternatives B）
- **架构换路 ROI 极低**：往届 TAAC 冠军 60-70% 时间花在特征工程（详见 chat 历史 2026-05-01 02:00）；OneTrans / HSTU 等新架构在 25 天预算 + 评估额度受限下**期望 ROI 为负**
- **特征语义未公开**：官方 4/28 答复「all features have been anonymized」→ 全员只能从**统计性质**做特征工程；ADR-002 切片诊断已给出 4 维统计探针就是为此准备
- **T10 数据即将回流**：当前 4 个 platform jobs（baseline-retrain / swiglu+time / swiglu+d_model=128 / swiglu+dropout=0.05）跑完后，可一次性拿到 4 张 `[diag]` 切片表 → 必须有**预先准备好的可执行方案**，否则数据回流时仍要临时拍脑袋

### 问题域

- **不能为每类候选特征各跑一次实验**（评估额度每天 4 次串行，4 类候选 × 多次调参 = 配额秒空）
- **不能边写代码边等数据**（5/3 ~ 5/15 的 12 天窗口太紧，如果数据回流时还在写代码会浪费 2~3 天）
- **必须有防泄露设计**（PCVR 任务里历史正例率类特征极易引入未来标签泄漏 → 离线高分但在线崩盘）
- **必须保留 ensemble 兼容性核对**（任何方案不能滑入"事实上的 ensemble"灰色区，比如 SWA 当前算 ⚠️ 灰色）

### 时间压力

- 距初赛 ddl **24 天**（5/25）
- 评估配额：4 次/天 × 24 天 = 96 次理论值 → 实际可用 ≤ 60 次（去掉失败 / 取消 / 留守报告）
- 最终模型必须在 5/22 前定稿，留 3 天兜底 + 报告 v1 → v2 finalize

---

## Decision（最终决定做什么）

**承诺把特征工程定为 5/3 ~ 5/15 的 P0 路径**，按 4 类候选 × 4 个 EXP 切片实施，每类候选预先备好实施方案 + 防泄露设计 + ensemble 合规核对，等 T10 切片表回流后**当晚**触发对应 EXP（不重复决策、不临时拍脑袋）。

### 4 类候选特征（按 ROI × 实施摩擦力综合排序）

| 优先级 | 候选名称 | 触发条件（T10 切片表观察） | 预期 AUC 增益 | 代码改动 |
|---|---|---|---|---|
| **P0-A** | **缺失计数 dense feature** | `user_int_nonzero` 桶间 AUC 差 > 0.02（用户冷启动是薄弱点）| +0.003 ~ +0.008 | dataset.py +30 行 |
| **P0-B** | **序列活跃度 dense feature** | `seq_total_len` 桶间 AUC 差 > 0.02（活跃度异常是薄弱点）| +0.003 ~ +0.010 | dataset.py +20 行 |
| **P0-C** | **历史正例率（防泄露版）** | `item_int_nonzero` 桶间 AUC 差 > 0.02 OR `label_type=2` pred_mean 系统性低估 | **+0.005 ~ +0.012** | dataset.py +60 行 + 一次性预处理脚本 |
| **P1** | **Multi-task CTR + CVR head** | `label_type=1` 与 `=2` pred_mean 差距小（模型不区分点击 vs 转化） | +0.005 ~ +0.015 | model.py +20 行 + trainer.py +30 行 + loss 改 |

> **保守期望叠加**（4 类全做且全部成功）：+0.015 ~ +0.045 AUC → leaderboard 0.844 → **0.86~0.88**，**完全可能反超 SOTA 0.858**。
> **现实期望**（按往届成功率 ~60% 估算）：+0.010 ~ +0.025 → leaderboard 0.85~0.87，**保底也能接近 SOTA**。

### 实施时序（5/3 ~ 5/15，12 天）

| 时段 | 触发条件 | EXP 编号 | 改动文件 | 验证产出 |
|---|---|---|---|---|
| 5/3 ~ 5/4 | T10 数据回流 → 选择 P0 方向 | EXP-003 | 0 行（仅决策） | T10 切片表正式归档 + 锁定 P0 方向 |
| 5/5 ~ 5/7 | 触发 P0-A or P0-B | EXP-004 | dataset.py +30~50 行 | 离线 AUC + 1 次 leaderboard 评估 |
| 5/8 ~ 5/10 | 触发 P0-C（最高增益但最复杂） | EXP-005 | dataset.py +60 行 + 预处理脚本 | 离线 AUC + 1 次 leaderboard |
| 5/11 ~ 5/13 | 触发 P1（multi-task） | EXP-006 | model.py + trainer.py + loss | 离线 AUC + 1 次 leaderboard |
| 5/14 ~ 5/15 | 全量集成验收 | EXP-007 | （无新代码，只组合）| 最终 leaderboard 数 + 决定是否 5/16 启动 §5.3 后备方案 |

### 关键技术细节（4 类候选的实施模板）

**P0-A · 缺失计数 dense feature**：
- 文件：`baseline/dataset.py::_convert_batch`（在 user/item 字段聚合后）
- 逻辑：`user_int_nonzero_count = sum(user_int_feats[i] > 0 for i in range(46))` → 作为 user dense feature 第 5 列追加
- CLI：`--enable_count_features`（默认关，opt-in；满足 user rule"默认行为不变"红线）
- 防泄露：天然无泄漏（基于当前样本本身的 sparse 字段统计）

**P0-B · 序列活跃度 dense feature**：
- 文件：`baseline/dataset.py::_convert_batch`（seq 处理段后）
- 逻辑：4 个 domain 的 `seq_*_len` + `seq_*_max_ts - seq_*_min_ts` 共 8 维 dense feature
- CLI：`--enable_seq_stats_features`（默认关）
- 防泄露：基于序列数据本身长度/时间范围，无标签泄漏风险

**P0-C · 历史正例率（防泄露版本）** ⚠️ 最难，最高增益：
- 文件：`baseline/dataset.py` + 新增 `baseline/build_history_stats.py` 一次性预处理脚本
- 逻辑：
  1. 离线脚本扫训练集，**按 item_id 分组**，统计**早于该 row 的 timestamp** 的 `label==1` 占比 → 写入 `USER_CACHE_PATH/item_history_cvr.parquet`
  2. dataset.py 训练时按 item_id + 当前 timestamp lookup，作为 dense feature
  3. **关键防泄露**：lookup 时严格 `cutoff = current_ts - 1day`，确保用 ≥ 1 天前的统计；item_id 第一次出现时填全局 mean
- CLI：`--enable_history_cvr_features` + `--history_cvr_cutoff_sec 86400`（默认关）
- 风险：**最容易出 bug 的部分**，需 demo 上端到端验证 cutoff 实施正确性

**P1 · Multi-task CTR + CVR head**：
- 文件：`baseline/model.py`（PCVRHyFormer 末尾分类器） + `baseline/trainer.py`（loss）
- 逻辑：
  1. classifier 输出 2 维（click + conversion）→ 用 `--action_num 2` 启用（baseline 已有 flag）
  2. loss = `BCE(click_logit, label_type≥1) + BCE(conv_logit, label_type==2) * 2.0`（CVR 权重更高）
  3. 推理时用 `conv_logit` 输出
- CLI：`--multi_task_loss`（默认关；启用时强制 `--action_num 2`）
- 防泄露：无（multi-task 是同一 forward pass 的两个 head）

### Ensemble 合规核对清单（每个 EXP 实施前必跑）

每个 EXP 改动前，必须确认：

1. ❌ **不能**多个独立训练的 ckpt 做 prediction averaging
2. ❌ **不能**多个 seed 跑出来的 model 做 rank averaging
3. ❌ **不能**TTA（推理时多次前向取平均）
4. ✅ **可以**单一 model + multi-task loss（同 forward pass 的多 head）
5. ✅ **可以**MoE / Sparse-MoE（同模型内的 expert，端到端联合训练）
6. ⚠️ **SWA 灰色区**：若 EXP-007 想用，**先到赛事群发问澄清**，不要假设
7. ✅ **可以**hyperparameter search → 最后只发布 1 个最强 ckpt（本质是 model selection，不算 ensemble）

---

## Alternatives Considered（备选方案）

### A. 等 T10 数据回流再临时决策（不预先准备 ADR）

- **Pros**：不浪费今晚时间；候选方案保持灵活
- **Cons**：数据回流后还要现写方案 + 防泄露设计 + 调试 → 至少损失 2~3 天 budget；评估配额会被"探索性"实验吞噬；防泄露设计若仓促做容易 bug
- **结论**：**拒绝**。25 天预算下任何"边等边写"都是 ROI 黑洞

### B. ~~Multi-seed × Multi-encoder ensemble（原 Stage 3 计划）~~

- **Pros**：往届 RecSys 比赛常用；预期 +0.003~0.008 AUC
- **Cons**：**官方明确禁止**（competition.md line 77）→ **直接违规**
- **结论**：**拒绝（违规）**；增益由 P1 multi-task + P0-C 历史 CVR 替代覆盖

### C. 换架构（OneTrans / HSTU / 生成式推荐）

- **Pros**：理论天花板更高（+0.005~0.020）；冲创新奖必须做
- **Cons**：600~1500 行新代码；调通 5~10 天；25 天 budget 下期望 ROI 为负（详见 chat 历史 2026-05-01 02:00 战略评估）
- **结论**：**拒绝（本届）**；启动条件已写在 active.md T5（distance-to-SOTA ≤ 0.1% 才考虑），目前 1.43% 不满足

### D. 重做 hyperparameter sweep（继续在 swiglu 上调参）

- **Pros**：4 个 jobs 已在跑，验证当前架构上限
- **Cons**：**正在做**，但单纯 hp sweep 期望增益已捉襟见肘（+0.001~0.005）；不能替代特征工程
- **结论**：**并行**，不冲突；但不作为 P0 路径（hp 只是 P2）

### E. 推理后处理（Calibration / Isotonic Regression）

- **Pros**：易实施 +0.001~0.003 AUC；纯推理后处理，无训练改动
- **Cons**：增益太小，不足以 close gap 到 SOTA；放在 5/22~5/24 兜底阶段做即可
- **结论**：**采纳为 P3**，本 ADR 不展开

---

## Consequences（影响与外溢）

### 正向

- **解决 ensemble 禁令的真空**：Stage 3 原本依赖 ensemble 拿增益的部分，由 P1 multi-task + P0-C 历史 CVR 替代覆盖，**预期增益反而比原 ensemble 计划更大**
- **预先准备 → 数据回流即可触发**：T10 切片表一回流，对应 EXP-NNN 当晚就能开干，**不浪费 1 天 budget**
- **防泄露设计前置**：P0-C 的 cutoff_sec 设计在写代码前就讨论清楚，避免离线高分上线崩盘
- **完全合规**：每个 EXP 实施前跑「Ensemble 合规核对清单」，0 风险滑入灰色区
- **paper 友好**：T6 报告 §3.4 / §3.5 的 TODO 段，每个 EXP 完成都能直接填一节

### 负向 / 待观察

- **预测的"防泄露"可能仍有 bug**：P0-C 是高风险项；mitigation 是 **EXP-005 启动前在 demo 数据上端到端验证 cutoff 逻辑正确性**（同 ADR-002 的 demo 验证流程）
- **multi-task loss 权重难调**：CVR 权重默认 2.0 是拍脑袋，可能需要 0.5~5.0 区间扫描；mitigation 是 EXP-006 先跑默认权重，若 AUC 退步再扫
- **dataset.py 改动会扩散**：P0-A/B/C 都改 `_convert_batch`，可能引入数值类型 bug；mitigation 是每个 EXP 都先在 demo 1k 上跑 1 epoch 验证 AUC ≈ 0.6761 不变（与 EXP-001 对齐）
- **EXP-007 集成验收无明确成功标准**：兜底是「至少有 1 个 EXP 改动后的 leaderboard ≥ 0.85」，否则启动 §5.3 后备方案

### 与 ensemble 禁令的兼容性

| 改动 | 是否独立训练？ | 是否 prediction averaging？ | 合规判定 |
|---|---|---|---|
| P0-A/B/C dense feature | ❌ 单一训练 | ❌ 单一前向 | ✅ 合规 |
| P1 Multi-task | ❌ 单一训练 | ❌ 单一前向（取 conv head） | ✅ 合规 |
| P3 Calibration | ❌ 后处理 | ❌ 函数变换 | ✅ 合规（不改 model） |
| ~~Multi-seed ensemble~~ | ✅ 多次独立 | ✅ 平均 | ❌ **违规** |
| ~~SWA~~ | ❌ 同一训练轨迹 | ❌ 权重平均出 1 model | ⚠️ **灰色 → 必须问官方** |

---

## Verification（如何验证决策正确）

### 已验证（写本 ADR 时）

- ✅ 与 user rules 全部红线对齐：最小变更（每个 EXP 限定文件）/ 不新建 test 文件 / 默认行为不变（每类 CLI flag 默认关）/ 输出克制（无 helper / debug 文件）
- ✅ 与 ensemble 禁令兼容（已逐条核对）
- ✅ 与 ADR-001 / ADR-002 流水线兼容（dense feature 不影响 T8 的 ts_split，不影响 T10 的 _diag_*）

### 待验证（5/3 ~ 5/15 各 EXP 落地后）

| EXP | 成功标准 | 失败回滚条件 |
|---|---|---|
| EXP-003（T10 数据归档 + 决策） | 4 张切片表完整 + P0 方向锁定 1 个 | 数据缺失 → 重启一次 platform job 拿数据 |
| EXP-004（P0-A 或 P0-B） | demo AUC 不变 + leaderboard +0.002 以上 | demo AUC 变化 → bug，回滚；leaderboard 退步 → 先关 flag 排查 |
| EXP-005（P0-C 历史 CVR） | demo AUC 不变 + 离线 AUC 提升 + leaderboard +0.005 以上 | **离线高 leaderboard 退步 → 必然是泄漏**，立即回滚 + 重审 cutoff |
| EXP-006（P1 multi-task） | demo AUC 不变 + leaderboard +0.003 以上 | 不收敛 / loss NaN → 调 CVR 权重 |
| EXP-007（全量集成） | leaderboard ≥ 0.85（保底）/ ≥ 0.86（理想） | 触发 §5.3 后备：5/16 启动 SWA 询问 / OneTrans 紧急启动 |

### 验证方案（user rule "影响数据 / 采样 / 权重 / 特征"）

- **影响：**Yes（P0-A/B/C 加 dense feature；P1 改 loss）
- **预期 metric 影响**：每个 EXP 都给出预期 AUC 增益区间（见 §Decision 表）
- **如何验证**：每个 EXP 都跑「demo 1k 上 1 epoch 验证 AUC=0.6761 不变 → 平台真实数据 1 epoch 验证基线 AUC ~0.86 不变 → 才能关 flag → 再开 flag 跑 leaderboard 评估」三步走
- **回滚方案**：每个 EXP 的 CLI flag 默认关 → 关 flag 即可秒级回滚到 baseline 行为
