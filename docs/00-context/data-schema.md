# 数据 Schema（Data Schema · demo_1000.parquet）

> **稳定文档**。来源：原 `knowledge_base.txt §3`。
> **2026-04-30 平台真值校核**：46 user_int + 14 item_int + 4 domain seq 与 demo 完全一致（详见 `docs/40-experiments/EXP-002-platform-baseline-job.md`），字段集稳定，不需要扩 fid 列表。
> **2026-04-10 起为「扁平列布局」**：所有特征都是顶层列，不再嵌套 struct/array。

---

## 文件总览

| 项 | 值 |
|---|---|
| 行数 | 1,000（demo） / 1,900,000（平台真实） |
| 列数 | 120 |
| 文件大小 | ~39MB（demo，仅作 schema 参考；正式数据量级远超） |

### 120 列分 6 类

| 类别 | 列数 | dtype | 说明 |
|---|---|---|---|
| ID & Label | 5 | int64 / int32 | 核心标识与标签 |
| User Int Features | 46 | int64 / list<int64> | 用户整型特征 |
| User Dense Features | 10 | list<float> | 用户稠密特征 |
| Item Int Features | 14 | int64 / list<int64> | 物品整型特征 |
| Domain Sequence Features | 45 | list<int64> | 4 个域的行为序列 |
| **合计** | **120** | | |

---

## 3.1 ID & Label 列（5 列，非空）

```
user_id        int64
item_id        int64
label_type     int32   ← 行为类型；正样本判定的核心字段（见 §3.6）
label_time     int64   ← 行为发生时间戳
timestamp      int64   ← 事件（曝光/请求）时间戳
```

**约定**：当 `user_int_feats_{fid}` 与 `user_dense_feats_{fid}` 共享同一 `{fid}`，它们对齐描述同一实体/信号（int 描述离散面，dense 描述连续面）。

---

## 3.2 User Int Features（46 列）

- **标量 int64（35 列）**：`user_int_feats_{1, 3, 4, 48-59, 82, 86, 92-109}`
- **list<int64>（11 列，多值/multi-hot）**：`user_int_feats_{15, 60, 62-66, 80, 89-91}`

## 3.3 User Dense Features（10 列）

- **list<float>（10 列）**：`user_dense_feats_{61-66, 87, 89-91}`

## 3.4 Item Int Features（14 列）

- **标量 int64（13 列）**：`item_int_feats_{5-10, 12-13, 16, 81, 83-85}`
- **list<int64>（1 列）**：`item_int_feats_{11}`

## 3.5 Domain Sequence Features（45 列，4 个行为域）

| Domain | fid 范围 | 列数 |
|---|---|---|
| `domain_a_seq_*` | 38-46 | 9 |
| `domain_b_seq_*` | 67-79, 88 | 14 |
| `domain_c_seq_*` | 27-37, 47 | 12 |
| `domain_d_seq_*` | 17-26 | 10 |

每个 domain 内部包含若干 `list<int64>` 列：
- 一列是「时间戳序列」（与主样本 timestamp 求差，可做 time-bucketing）
- 其余列是该序列的 side-info（item_id、category_id、author_id 等）
- 具体「哪个 fid 是 ts_fid」由 `schema.json` 中的 `cfg['ts_fid']` 指定

> **平台情报**（4/28 官方答复）：「demo 数据多了一个序列域特征，但 all features have been anonymized」。下次平台 job 日志若打印 5 个 seq 域，需补一行确认；若仍是 4 个，说明新序列域命名变了，需到列名集合里 grep 排查。

---

## 3.6 label_type 的语义（重要 · 与 baseline 强相关）

- `label_type` 是一个 int32，取值多于 0/1，编码不同动作类型
- **baseline 中正样本定义**（`baseline/dataset.py`）：
  ```python
  label = 1   iff   label_type == 2     # 即「转化」
  label = 0   其他所有取值（含 null）
  ```
  代码片段：
  ```python
  labels = (batch.column[label_type].fill_null(0)
              .to_numpy().astype(int64) == 2).astype(int64)
  ```

**官方答复**（4/27 / 4/28）：`label_type=2` 即「24h 内转化的正样本」。baseline 现行处理与官方一致。

**推论与注意事项**：
- 0 / 1 / 2 / ... 这些 label_type 可能分别对应 曝光/点击/转化/其它互动
- 训练目标是「点击后转化」（PCVR），因此正负样本定义务必以 `label_type==2` 为准
- 多目标实验时（如 CTR/CVR 联合），可以把 label_type 进一步拆解为多个 0/1 头

**Demo 重要发现**（来自切片诊断 ADR-002）：
- demo 数据完全没有 `label_type=0` 样本（n=0），只有 `label_type=1` (n=88, pos_rate=0) 和 `label_type=2` (n=12, pos_rate=1)
- 若平台真实数据也是这个分布 → 任务本质是 **click-conditional CVR** 而非 **CTCVR**
- 这会改变后续特征工程方向（点击偏置 / 辅助任务设计）。**待平台 job 切片表回流后再决策**

**转化回流**（CVR 任务通病，需要警惕）：
- 训练数据通常按时间截断，最后几天的 `label=0` 可能是「假阴」（转化尚未回流）
- 经验做法：评估时去掉 tail-K 天，或对 tail-K 天样本做权重 / 延迟反馈建模（Delayed Feedback Modeling）
- TAAC 2026 评测在线下进行（基于划分好的验证/测试集），但仍需关注训练数据的回流截断
