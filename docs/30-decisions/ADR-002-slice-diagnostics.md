# ADR-002 · valid 集 4 维切片诊断（默认开，无 CLI 开关）

> **Status**: Accepted
> **Date**: 2026-04-30
> **Related Tasks**: [T10](../20-tasks/active.md#t10--valid-集切片诊断--工具已落地待平台-job-数据决定-p0-特征工程方向)
> **Related Experiments**: 待平台 job 后回填

---

## Context（背景与触发原因）

### 触发证据

- **用户报告**：社区有人通过特征工程优化 AUC 到 0.85809，我们 baseline 0.843371，差距 +0.0148
- **用户决策**：选「先做误差分析，找薄弱点；改动限定在特征工程 + 训练策略」
- **官方答复（4/28）**：「all features have been anonymized, and the organizing committee cannot confirm specific feature type classifications」 → **没人能靠语义猜特征**，全员只能从**统计性质**出发做特征工程

### 问题域

- 不知道 baseline 在 valid 集上**哪一类样本预测得最差**
- 不知道**哪个统计维度的薄弱点最值得做特征工程**
- 评估额度紧张（每天 3 次串行）→ **不能为了 4 个候选方向各跑一次实验**

### 时间压力

- 距初赛 ddl 25 天，需要尽快锁定 P0 特征工程方向

---

## Decision（最终决定做什么）

在 valid 集每次 evaluate 时**默认运行** 4 维切片诊断，**不加 CLI 开关**：

| 切片维度 | 分桶方式 | 输出 |
|----------|----------|------|
| `user_int_nonzero` | 4-quantile | 每桶 (n / pos_rate / AUC / pred_mean) |
| `item_int_nonzero` | 4-quantile | 同上 |
| `seq_total_len`（4 个 domain 累加） | 4-quantile | 同上 |
| `label_type`（原始 0/1/2） | categorical hardcode | 每类 (n / pos_rate / AUC / pred_mean) |

### 关键技术细节

- 文件：`baseline/dataset.py::_convert_batch` 末尾在 `is_training=True` 分支加 4 个 `_diag_*` 字段
- 文件：`baseline/trainer.py::evaluate()` 收集这些字段 + 新增 `_slice_diagnose` 方法
- diff：~120 行，2 个文件，0 lint 错误
- **下划线前缀** `_diag_*` → 训练 forward 的 `_make_model_input` 不读这些字段 → 训练逻辑零副作用
- **`is_training=False` 不写入** → 推理路径（infer.py）完全不受影响 → T7（infer.py 对齐）零阻塞
- 开销：4 次 AUC 计算 + 分桶，< 1s（vs 整次 evaluate 数十秒）

---

## Alternatives Considered（备选方案）

### A. 加 CLI 开关 `--enable_slice_diag`

- **Pros**：完全可控，默认关
- **Cons**：开销 < 1s，根本不需要为了这点开销加开关复杂度；忘记开就拿不到诊断数据，浪费一次评估额度
- **结论**：拒绝（功能开关泛滥是反模式）

### B. 把诊断字段直接加到模型输入（让模型自己学）

- **Pros**：理论上模型可以利用这些 meta 信号
- **Cons**：会改变训练逻辑（违反"先诊断再改"的工程纪律）；与 T8 实施时序冲突
- **结论**：拒绝（这是 P0 特征工程候选，应该等切片表数据再决策，而不是同时做）

### C. 用 hook 包装 evaluate（不改 dataset / trainer 文件）

- **Pros**：完全无侵入
- **Cons**：需要新增 hook 模块文件，违反 user rule"输出克制"；逻辑复杂度反而上升
- **结论**：拒绝

### D. 单独写诊断脚本（离线对 ckpt + valid 数据跑）

- **Pros**：与训练流程完全解耦
- **Cons**：需要把 valid 数据下载到本地；与 evaluate 流程双源；**会浪费一次 ckpt publish + 评估额度** 来跑诊断
- **结论**：拒绝（关键反对理由是"浪费评估额度"）

---

## Consequences（影响与外溢）

### 正向

- **零评估额度消耗**：每次 evaluate 自动跑诊断，4 张表与 ADR-001 sanity log 一起回流
- **决策路径清晰**：T10 节点已给出 6 行决策表（见 active.md），每个观察分支对应一个 P0 改动
- **零侵入**：训练 / 推理路径完全无副作用（demo 验证 AUC 0.6761 与 EXP-001 完全一致）

### 负向 / 待观察

- **demo 数据信息量有限**：100 行 valid 让 `seq_total_len` 退化为 1 桶 [160,256]；正式数据应有更广分布
- **重大新发现**：demo 完全没有 `label_type=0` 样本（n=0）→ demo 是 click-conditional CVR 数据，不是 CTCVR。**若平台真实数据也是这个分布，会改变损失函数与辅助任务设计的整个方向** → 已在 `docs/00-context/data-schema.md §3.6` 标记
- 4 个 numpy → tensor 转换的额外内存开销（~B*16 bytes）：每 batch 256 行 = ~4KB，可忽略

---

## Verification（如何验证决策正确）

### 已验证（本地 demo · 1 epoch CPU · 33s）

- ✅ 代码无 lint 错误
- ✅ 完整 train.py 端到端跑通
- ✅ Epoch 1 Validation AUC = **0.6761** 与 [EXP-001](../40-experiments/EXP-001-baseline-cpu-demo.md) 完全一致 → 训练逻辑零副作用
- ✅ 4 张切片表全部按预期输出（详见 active.md T10 节）

### 待验证（下一次 platform job）

| 切片表观察 | 决策路径 |
|---|---|
| `label_type=0` 占比 | 决定任务本质（CTCVR vs click-CVR）→ 决定损失函数方向 |
| 4 个连续维度桶间 AUC 最大差距 | 锁定 P0 特征工程方向（缺失计数 / 序列活跃度 / 商品冷启动）|
| 桶 AUC 与整体 AUC 的相对偏离 | 验证模型在哪个用户/商品 segment 上失灵 |

### 验证方案（user rule "影响数据/采样的改动须给出验证方案"）

- **训练 AUC 不应变化**：本 ADR **不动训练数据/采样/权重**，只多收集了 4 个 diag 字段；任何 AUC 变化 = bug，立即回滚
- **推理路径未改动**：T7 改 infer.py 时无需考虑 `_diag_*` 字段（is_training=False 不写入）
- **4 张表必须出现**：每次 evaluate 完都应看到 `[diag] slice by ...` 4 行 logging；缺失 = bug
