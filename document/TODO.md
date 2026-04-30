# TencentUniRec · TAAC 2026 内部 TODO

> 单一事实源（single source of truth）。任何 TODO 状态改动只在本文件落定。
> `document/knowledge_base.txt` §8 仅保留 stub，避免双源失同步。

**最近一次审视**：2026-04-30（夜）· 结论：抓到乐享社区 141 题（含 26 条 Administrators / TAAC 官方答复，详见 `document/community/_industry_intel.md`），共 5 项硬情报落地——① 任务确认为 **CVR**（label_type=2 = 24h 内转化），与 baseline 现行处理一致；② **infer.py / model.py 在推理时必须与训练时完全一致** → **T7 升 `[!]` 阻塞**（我们已改 ns_tokenizer / seq_encoder / num_queries，平台默认 infer.py 100% 加载失败）；③ **测试集时间不重叠且在训练集之后**，但 baseline `valid_ratio=0.1` 是随机切 → **新增 T8** 改时间切；④ **推理 30 分钟硬上限** → 新增 T9 profile；⑤ 资源配额已迭代为「全队 2 quota，串行执行」+ 评估次数「每天 3 次，北京时间 12:00 刷新」。距初赛 ddl 25 天（5/25）。

---

## 审视清单（每次阅读本文件必跑 · 7 步）

按顺序执行。任何一步发现偏差 → 立刻动手修补，不留到「下次再说」。

1. **状态校验**：每条 `[~]` / `[!]` 状态是否仍准确？是否有暗中完成未打勾的工作？
2. **时间线压力**：对照 `knowledge_base.txt §2`（赛事时间表），距离最近 ddl 还有多少天？哪些 TODO 必须在此前推进？
3. **优先级重排**：高 ROI（杠杆大、阻塞少）的 TODO 是否排在前面？是否有「等结果型」TODO 可以并行启动？
4. **Recipe 时效**：每条 `[~]` 的 recipe 与当前 `baseline/` 代码、`schema.json`、依赖版本是否仍一致？
5. **红线复核**：执行路径是否仍满足 user rules 的四条红线？
   - 最小变更（不大改 baseline 文件）
   - 不新建 test 文件 / 文件夹
   - 生产安全（默认行为不变 / 配置开关 / 影响数据·采样·权重·特征的改动须给出验证方案）
   - 输出克制（无废弃 helper / debug 文件）
6. **下一步具象化**：对每条 `[~]` / `[ ]`，写一句「未来 30 分钟可做的最小动作」。写不出 → 拆分、合并或降级为 `[ ]`。
7. **沉淀回写**：在顶部「最近一次审视」打上 ISO 日期 + ≤ 1 行结论。任何状态变更须在对应 TODO 节点附 inline 注释 `(YYYY-MM-DD: ...)`。

输出规范：审视结论用结构化表格 / 列表呈现，不要埋在散文段落里。

---

## 状态约定

| 标记 | 含义 |
|------|------|
| `[x]` | 完成 |
| `[~]` | 进行中（含可复用 recipe） |
| `[ ]` | 待启动 |
| `[!]` | 阻塞（须注明阻塞条件 + 解除条件） |

---

## TODO 列表

### T1 `[x]` 跑通 baseline 在 1k demo 上，记录 AUC / FLOPs / Params

**完成时间**：2026-04-28 · 平台：本地 macOS / CPU only / py3.11.7 / torch 2.11 CPU

**结果**（默认 run.sh 参数 + `--batch_size 32 --num_epochs 2 --device cpu --seq_max_lens seq_X:64`）：

| 指标 | 值 |
|------|---|
| Total Params | 160.94 M（Sparse 158.45 M + Dense 2.49 M） |
| Epoch 1 Val AUC / LogLoss | 0.6761 / 0.3380 |
| Epoch 2 Val AUC / LogLoss | 0.6799 / 0.3327 |
| 训练总耗时 | ~33 s（包含 2 次 reinit + 2 次 eval + 2 次 ckpt） |
| 单 batch CPU forward | 658 ms（B=32, seq_max_len=256） |
| ckpt 路径 | `runs/demo/ckpt/global_step58.layer=2.head=4.hidden=64.best_model/model.pt` |

**FLOPs**：未在 CPU 跑通时测量（fvcore 对自定义 nn.Module 不准）。GPU 上跑 T4 时再用 `torch.profiler` 测准。

**关键发现**（结合 PDF）：
1. **baseline 文件 ≡ 官方训练模板**：PDF p4 列出的 `dataset.py / model.py / run.sh / train.py / trainer.py / utils.py` 与 `baseline/` 一字不差。
2. **平台环境变量**（PDF p51 / p54）：
   - 训练：`USER_CACHE_PATH` / `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH`
   - 评估：`USER_CACHE_PATH` / `MODEL_OUTPUT_PATH` / `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `EVAL_INFER_PATH`
   - ⚠ baseline `train.py` 用了 `TRAIN_LOG_PATH`，但 PDF 未列；本地 OK，平台上需自行兜底默认。
3. **平台规格**（PDF p28 / p30）：Ubuntu 22.04、Python 3.10.20、CUDA 12.6、19GB GPU / 9 CPU / 55GB RAM。本地 py3.11 与平台 py3.10 接近，无明显兼容问题。
4. **ckpt 命名约束**（PDF p51-53）：必须以 `global_step` 前缀，目录名 ≤ 300 字符。baseline `_build_step_dir_name` 已遵守。
5. **公榜参考**（PDF p16）：HyFormer 配置 0.836~0.838 AUC；run-test 系列 0.851~0.861。

**踩坑 & 解法**：
| 现象 | 原因 | 解法 |
|------|------|------|
| `iter(train_loader)` 0 batch | demo 文件只有 1 row group，被 `valid_ratio=0.1` 取整后吃光 train | 用 `pq.write_table(t, ..., row_group_size=100)` 重写为 10 RG，训练 → 9 RG / 验证 → 1 RG。正式初赛数据多分片，不会有此问题。 |
| sysctlbyname IOError 警告 | macOS 沙箱 + pyarrow CPU detection | 警告无害，可忽略；平台 Linux 无此问题 |
| 大 vocab Embedding 占显存 | demo 里 ts 列被识别为 vocab≈1.7e9 | 已在 schema 中标 `ts_fid`（baseline `_load_schema` 自动剥离）；ID 列建议 `--emb_skip_threshold 1000000`（run.sh 默认） |

**复现 Recipe（自洽，复制即可执行）**：

```bash
# ① 创建 conda 环境（一键复现，详见仓库根 environment.yml）
conda env create -f environment.yml
conda activate tencent-unirec

# ② 下载 demo 数据
mkdir -p data/demo data/demo_rg100 runs/demo/{ckpt,log,tb}
curl -sSL -o data/demo/demo_1000.parquet \
    https://huggingface.co/datasets/TAAC2026/data_sample_1000/resolve/main/demo_1000.parquet

# ③ 重写 demo 为多 row group（避免 valid_ratio 把 train 吃光，仅 demo 需要）
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/demo/demo_1000.parquet')
pq.write_table(t, 'data/demo_rg100/demo_1000.parquet', row_group_size=100)
"

# ④ 自动生成 schema.json（详见 T2）
python baseline/infer_schema.py \
    --parquet data/demo_rg100/demo_1000.parquet \
    --output  data/demo_rg100/schema.json

# ⑤ 启动训练（CPU；GPU 上去掉 --device cpu 即可）
TRAIN_DATA_PATH=data/demo_rg100 \
TRAIN_CKPT_PATH=runs/demo/ckpt \
TRAIN_LOG_PATH=runs/demo/log \
TRAIN_TF_EVENTS_PATH=runs/demo/tb \
bash baseline/run.sh \
    --batch_size 32 --num_epochs 2 --num_workers 0 \
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:64,seq_d:64" \
    --device cpu
```

> 平台注入：`TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH` 由 Tencent Angel 平台自动设置；上平台时 ⑤ 这块只保留 `bash baseline/run.sh` 即可。
>
> `TRAIN_LOG_PATH` 不在平台 env 列表中（已在 baseline 加 `/tmp` fallback，见 `bd1af1b`）。

---

### T2 `[x]` 写一份 schema.json 自动生成脚本

**实现**：`baseline/infer_schema.py`（2026-04-26 完成）

(2026-04-30 夜: 社区情报佐证 — 官方 4/27 / 4/28 答复 `label_type=2` 即「24h 内转化的正样本」，dataset.py:512 早已正确 remap 为 1/0；`infer_schema.py` 的 4 条正则不匹配 `label_type` → 被静默跳过 → schema.json 不含 label_type → 结果正确。**残留风险**：脚本对所有未匹配列都静默跳过，若平台未来加非 `*_feats_*` / `domain_*_seq_*` 命名的特征列会被漏掉；现状不修（最小变更红线），日常 review 重跑后人工 diff 列名集合即可。)

**关键点**：
- 从列名正则识别四类特征（`user_int` / `user_dense` / `item_int` / `domain_X_seq`）
- 整型列：`vocab_size = max(positive value) + 1`（可改 percentile）
- list 列：`dim = 实际最大长度`（同样可改 percentile）
- `ts_fid` 启发式：序列内某 fid 的正值中位数 `> 1e9`（典型 Unix 秒级时间戳）
- 输出 `domain_key` 用 baseline 兼容的 `seq_a / seq_b / seq_c / seq_d`，prefix 字段保留原始 `domain_a_seq` 等列名前缀

**CLI**：

```bash
python baseline/infer_schema.py \
    --parquet <file_or_dir> \
    --output  schema.json \
    [--vocab-percentile 1.0] \
    [--dim-percentile  1.0] \
    [--ts-min 1000000000] \
    [--sample-rows 100000]
```

**风险与回归点（拉到正式数据后必查）**：
- demo 1k 行可能 `vocab_size` 严重低估（真实数据上 fid 可能跳到 1e7+）；正式数据建议全量扫描或 `--vocab-percentile 0.999`
- `ts_fid` 启发式失败时（例如某 side-info 也是大数值），手工把对应 domain 的 `ts_fid` 字段改成正确 fid 即可
- 默认 `dim=max` 在长尾极端长序列下会撑出大显存；正式数据先用 `--dim-percentile 0.99`

---

### T3 `[x]` 拉取真实初赛数据后，更新 knowledge_base.txt §3 的 fid 列表

**结案时间**：2026-04-30（晨）· 通过平台首个 training job 的启动日志间接验证，无需把数据下载到本地。

(2026-04-30 夜: 社区情报新增 — 4/28 官方提到 demo 数据多了一个序列域特征，但只答 "all features have been anonymized";只要新列名仍符合 `domain_X_seq_N` 模式 `infer_schema.py` 会自动 detect 把 seq 域数从 4 → 5。**最小动作**：下次平台 job 日志若打印 5 个 seq 域,本节直接补一行确认;若仍是 4 个,说明新序列域命名变了,需到列名集合里 grep 排查。)

**结论**：fid 集合与 demo §3 完全一致 → §3 维持现状。证据见下表：

| 指标 | demo (1k) | 平台 (1.9M) | 一致？ |
|---|---|---|---|
| user_int fid 数 | 46 | 46 | ✓ |
| item_int fid 数 | 14 | 14 | ✓ |
| domain seq 数 | 4（a/b/c/d） | 4（a/b/c/d） | ✓ |
| 总参数量 | 160.9 M | 198.7 M | sparse vocab 真值更大（demo 偏小），但**字段集**一致 |

**残留观察（不阻塞，沉淀给 T4 / 调参用）**：
- `emb_skip_threshold=1e6`：demo 里 seq_a 跳了 1/8 个 fid，平台真实数据里**未跳**（实际 vocab < 1e6）。说明 demo 在该 fid 上 vocab 估值偏大；不影响正确性，只是真实 sparse 参数比 demo 多 1 个 tensor / +37.7 M 参数。
- 真实数据 1.9M 行 / 1000 个 parquet 分片 → 本地为 demo 写的 `data/demo_rg100/` 重写步骤仅 demo 需要，平台无此问题。

---

### T4 `[~]` 复现 HyFormer 三种序列编码器消融

(2026-04-29 晚: 首个平台 job 默认 `transformer`，已等同 T4-B 首跑；下一步在平台 "Copy" 该 job 切 `swiglu` / `longer` 即可并行启动 T4-A / T4-C，无需本地再跑。)

(2026-04-30 夜: **撤回上面的"并行启动"判断** — 官方 4/27 答复明确「Each team allocated a single resource quota, shared across train+inference, **strictly serial**」;后已扩到 2 quota 但仍串行。**评估每天 3 次,北京时间 12:00 刷新**(早期官方答 AOE 15:59:59,实际已迭代),失败/取消的不计数。**新约束**:三组消融 + 调参的总评估开销受限 9 次/3 天,T4-A swiglu 跑完出 AUC 才能起 T4-C longer;`--seed 42` 复现性 + best_ckpt 选择规则不变。)

baseline 中三种编码器都已实现：`baseline/model.py` 的 `SwiGLUEncoder` / `TransformerEncoder` / `LongerEncoder`，由 `--seq_encoder_type {swiglu, transformer, longer}` 切换。

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

每组用相同 `--seed 42` 跑 ≥ 3 epoch，记录：`Val AUC / LogLoss / Params / FLOPs / Step latency / GPU peak mem`

**期望**（参考 HyFormer 论文 Table 1）：

```text
Transformer ≥ LONGER ≥ SwiGLU （AUC）
SwiGLU      ≪  LONGER ≪ Transformer （FLOPs）
LONGER 在初赛 100+ 字段、复赛 10× 数据上一般 ROI 最高。
```

---

### T5 `[~]` 复现 OneTrans Mixed-Param 版本（暂缓 · 等 T4 数字）

**决策（2026-04-26）**：选 (a)，暂缓启动。理由：
- HyFormer 论文 Table 1 中 OneTrans 仅领先 LONGER+RankMixer 0.05~0.08% AUC，在拿到本地 T4 数字之前贸然投 600~800 行新代码 ROI 低
- 复赛二期再启动可结合 T4 的最佳编码器选项做对比
- 维持「最小变更」红线，避免污染 `baseline/model.py`

**代价评估（启动时参考）**：约新增 600~800 行代码，集中改动 `model.py`。

**关键差异（vs 当前 baseline 的 HyFormer 风格）**：
- 把 S-tokens / NS-tokens 拼到同一注意力矩阵，不再用 Cross-Attention 桥接
- Attention Q/K/V 与 FFN 的权重做 Mixed Parameterization：S-token 共享一组 W；每个 NS-token 独立一组 W
- Causal Mask 摆放：NS-tokens 在 S-tokens 之后，吃完整 S 上下文
- 可选 Pyramid Stack（逐层缩 query 数）

**启动信号（满足任一即可重启）**：
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

(2026-04-30 夜: **`[~]` → `[!]` 升级**。社区情报锁定结论 — 官方 4/28 答复「**`infer.py` / `model.py` used during inference must be exactly identical to the one used during training**」(出处见 `_industry_intel.md` "推理模型load错误" 一题)。配合多条「评估出 AUC 但不上榜」的报告(4 条独立投诉,跨 4/26→4/29) → 印证不少队伍正撞这个坑。我们已偏离 baseline 默认 (`ns_tokenizer=rankmixer / user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2`,T4 还会切 `seq_encoder`),不修 infer.py 必然 Eval Failed。)

**阻塞条件**：
- 必须等首个平台 baseline job 跑出 best_ckpt 并 Publish 后,从 Evaluation 页下载默认 `infer.py (15.6K)` + `model.py (63.6K)` + `dataset.py (30K)`,先 diff 看其加载策略

**解除条件（任一即可）**：
1. 默认 infer.py 走 `train_config.json` 完全动态构建 model → 直接用 → 改回 `[~]` 然后结案
2. 默认 infer.py 含硬编码超参 → 修改 infer.py 三件套(model.py / infer.py)使之与训练完全一致(**修改现有文件,不新建,符合最小变更红线**) → 提交评估 → 公榜出现自定义模型分数 → 关闭

**审查 checklist（下载默认文件后逐条核对）**：
1. infer.py 是否 `json.load("train_config.json")` 动态读取 `ns_tokenizer_type / seq_encoder_type / num_queries / user_ns_tokens / item_ns_tokens` 来构建 model
2. model.py 类签名是否与我们训练时 import 的一致(可能我们在自定义 model.py 加了新 nn.Module → 必须同步到 infer 包)
3. dataset.py infer 时是否走 `is_training=False` 路径(label 列全 0,不依赖 label_type)
4. 推理输出格式(predictions 字段名 / user_id 覆盖率)是否能产出 leaderboard 期望的字段;社区有题报"Missing 310000 answer user_ids in predictions(need full coverage)" → 推理必须**全量覆盖测试集 user_id**

**风险评估**：
- 评估管线断 → 训练分数永远上不了榜(本届硬阻塞,无 workaround)
- 不影响数据 / 采样 / 权重 / 特征训练侧,**无 metric 影响,纯工程对齐**
- 与 T9 (推理 30min 硬约束)联动 — 改 infer.py 时顺便测推理时长

---

### T8 `[ ]` 把验证集切分从 random 改为按时间切

(2026-04-30 夜新增 · 触发证据: 官方 4/26 在 "basic questions on dataset" 答复「**test set timestamps do not overlap with training set and are from a later time period**」, 但 baseline `dataset.py` 当前用 `valid_ratio=0.1` 随机切 row group → **验证集与训练集时间分布相同, 而真实测试集是未来** → 离线 AUC 与公榜 AUC 必然有 gap, 调参方向会被带偏。这是**当前公榜大多数队伍尚未做**的离线-在线一致性修复, ROI 极高。)

(2026-04-30 夜补 · **待自验证的两条社区线索**, 未纳入下面切法设计, 仅留 trace:
1. **训练集 timestamp 严重不均衡**: 社区一截图示 train 跨 02-28~03-04 共 5 天, **03-03 单日 1.31M / 总 2.10M ≈ 63%**, 其它天 3~25%。若属实 → "取最后 10%" 实际几乎全是 03-04 末段;按 timestamp 排序后用更小切片(如 5%) 或许更接近真实 future 边缘 — 等下次平台 job 自打 `np.bincount(timestamps_in_days)` 验证后再定切法。
2. **疑似时间穿越**: 同帖示评估日志中出现 `2026-02-27 11:00:13`(早于 train min `2026-02-28 00:35:09`)。但用户截图的字段标签不明(可能是 `timestamp` / `label_time` / 序列内 ts_fid 之一,后两者"早"是设计如此非穿越) — 下次 job 在 train + eval 各打印 `timestamp.p1/p50/p99` + 序列 `ts_fid.p1/p50/p99` 自验证;若 `eval timestamp.p1 < train timestamp.max` 才视为真穿越,届时本节切法假设要重审。)

**最小动作**（30 分钟）：
1. grep `valid_ratio` / `is_training` / row_group_range 在 baseline 调用链, 确认切法位置
2. 改 `train.py` 里切分逻辑: 按 `_rg_list` 的 row group 顺序取**最后 10%** 作 valid, 而非随机切(原始 parquet 在平台上**已按 timestamp 写入**, row group 顺序 ≈ 时间序; 本地 demo 因 `row_group_size=100` 重写过也保留时间序)

**验证方案**（user rule "影响数据/采样的改动须给出验证方案"）：
- **离线 vs 在线一致性**: 改前/改后用同一 ckpt 跑两次 eval, 看离线 AUC 数字与公榜 AUC 的 gap 缩小多少(预期: 改前 gap 0.01~0.03, 改后 gap < 0.005; 若反之 → 说明 row group 不是时间序, 回滚)
- **AUC 绝对值预期**: 改后离线 AUC **可能下降** 0.005~0.015(因为切走了"近期"训练样本), 但公榜 AUC 期望**上升** 0.003~0.01 — 这是健康信号
- **Sanity**: 切完打印 train / valid 时间区间, 确认 valid 的 min_ts >= train 的 max_ts

**配置开关**：加 `--valid_split_strategy {random,time}`, **默认 `random`(行为不变)**, 显式 `--valid_split_strategy time` 启用; 符合 user rule "默认行为不变 / 配置开关"

---

### T9 `[ ]` 测推理时长 + 控在 30min 硬上限以内

(2026-04-30 夜新增 · 触发证据: 官方 4/27 答复「**A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks**」, 超时即被系统 kill。社区已有 4 条"推理被 stop"投诉。我们 baseline 用 `transformer` 编码器 + 完整 sparse vocab(198.7M params) 在 0.2 GPU(~16GB 可用显存) 上推理速度未知, 必须 profile。)

**最小动作**（首次 ckpt publish 后 30 分钟）：
1. publish 第一个 ckpt 后看 evaluation log 里的 `Inference Time` 字段(社区案例报过 625.28s ≈ 10min, 距 30min 还有余量)
2. 若 > 1500s(留 5min 安全边际) → 立即降 batch_size 推理 / 切 longer 编码器(top_k 压缩)
3. 数字回填到本节作为 baseline 推理性能基线

**联动 T7**：改 infer.py 时顺手把推理 batch 调成显存允许的最大值, 加 `torch.no_grad()` + bf16 推理(若 baseline 未启)

**联动 T4**：消融时记录每组的「Train AUC / Eval AUC / Train time / **Infer time** / Params」, infer time 列原本不在 T4 表里, 本节补上

**风险评估**：
- 不影响训练 / 数据 / 权重, 只在推理阶段调 batch + dtype
- 若 bf16 vs fp32 推理结果 AUC 差异 > 0.001 → 退回 fp32, 改用 batch 调优

---

## 编号迁移记录

`knowledge_base.txt §8.x` → 本文件 `Tx`，一一对应：

| 旧编号 | 新编号 | 状态 |
|--------|--------|------|
| §8.1 | T1 | `[x]` |
| §8.2 | T2 | `[x]` |
| §8.3 | T3 | `[x]`（2026-04-30 结案 · 平台日志验证 fid 集合一致） |
| §8.4 | T4 | `[~]` |
| §8.5 | T5 | `[~]` |
| §8.6 | T6 | `[!]` |
| —    | T7 | `[!]`（2026-04-29 新增 / 2026-04-30 晨修正:平台预置 infer.py 转为审查 / 2026-04-30 夜升级 `[!]`:官方明确 infer.py 必须与训练一致,我们已偏离 baseline → 必修） |
| —    | T8 | `[ ]`（2026-04-30 夜新增 · 验证集按时间切,fix baseline random valid 与"测试集是未来"的不一致） |
| —    | T9 | `[ ]`（2026-04-30 夜新增 · 推理 30min 硬上限 profile） |

---

## Changelog

- **2026-04-30 夜**：抓取乐享比赛社区 141 题(`document/community/lexiang_questions_2026-04-30-06-46-51.{json,md}`,工具 `tools/lexiang_questions_crawl`),按 organization 字段过滤出 26 条 Administrators / TAAC 官方答复,沉淀到 `document/community/_industry_intel.md`。本次审视带来 5 项硬变更:① **T7 `[~]` → `[!]` 升级** — 官方明确"infer.py 必须与训练一致",我们偏离了 baseline 默认超参,首次 publish 必修 infer.py 三件套;② **新增 T8** 验证集按时间切(官方"测试集时间不重叠且在训练集之后" ⊕ baseline 当前随机切 = 离线-在线一致性 bug,**ROI 最高的 baseline 改进**);③ **新增 T9** 推理 30min 硬上限 profile;④ T4 撤回"并行启动消融"判断 — 实际配额"全队 2 quota,严格串行"+评估每天 3 次北京时间 12:00 刷新(早期官方答 AOE 15:59:59 已迭代);⑤ T2/T3 inline 加情报佐证(label_type=2 处理已正确;新序列域若符合命名模式 infer_schema.py 自动覆盖)。距初赛 ddl 25 天。
- **2026-04-30 晨**：平台首个 baseline job 启动成功（GPU / batch=256 / 1.9M 行 / 1000 分片）。沉淀三件事：① **T3 升 `[x]` 结案**——日志确认 46 user_int + 14 item_int + 4 domain seq 与 demo §3 完全一致；总参数量 198.7M（vs demo 160.9M），sparse vocab 真值更大但字段集未变。② **T7 修正**——平台 Evaluation 页**已预置 infer.py (15.6K)**，原"新建文件"计划作废，改为「Publish 后下载默认 infer.py 审查兼容性」，回归 user rule 最小变更红线。③ T4 启动 swiglu 第一组消融（剩余 1 个资源额度可立刻使用）。距初赛 ddl 25 天。
- **2026-04-29 晚**：第一轮 baseline 提交平台训练成功（默认配置 = `transformer` 编码器，等同 T4-B 首跑）。沉淀本次审视：T4 阻塞实质解除，下一步走平台 "Copy" 流程并行启动 T4-A (swiglu) / T4-C (longer)，避免本地重跑；新增 **T7「补 `infer.py`」**（评估硬约束，未补即无法上榜，待用户显式授权新建文件后启动）；T3 仍 `[!]`，等首个平台 job 日志回流 vocab/dim 真实值后判定是否升 `[~]`。距初赛 ddl 26 天。
- **2026-04-29**：T1 环境切换 venv → conda（`tencent-unirec`，python 3.11.15 / torch 2.10.0 CPU）。recipe 第 ① 步从 `python -m venv` 改为 `conda create`；后续步骤删除 `.venv/bin/` 前缀。conda env 跑通 1 epoch AUC 0.6761，与 venv 完全一致（seed=42 复现）。`.venv/` 已删除（释放 922MB）。
- **2026-04-28**：T1 完成。结合 PDF 全 58 页验证 baseline 即官方训练模板；本地 CPU 33s 跑通 demo，AUC 0.6799。沉淀「踩坑 & 解法」表 + Recipe（含 row_group_size 重写步骤）。回写 PDF 关键事实（环境变量清单、平台规格、ckpt 命名约束、公榜参考分数）。
- **2026-04-26**：创建文件，从 `knowledge_base.txt §8` 迁出 6 条 TODO；T5 决议暂缓；新增 7 步审视清单。
